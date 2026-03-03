# -*- coding: utf-8 -*-
"""
Riley Holdeman
SmartHome Gesture Part 2

Utilized template provided by:
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

from handshape_feature_extractor import HandShapeFeatureExtractor


# Loading and building feature model
handshape_extractor = HandShapeFeatureExtractor.get_instance()

model = tf.keras.models.load_model("cnn_model.h5")
print("Model loaded:", model.input_shape, "->", model.output_shape)

# Penultimate layer output
feature_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
print("Feature model output shape:", feature_model.output_shape)


# Helper functions
def extract_middle_frame(video_path: str):
    """Extract the middle frame from a video (returns a BGR frame)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # pick middle frame if needed
    if frame_count <= 0:
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            raise RuntimeError(f"No frames in video: {video_path}")
        return frames[len(frames) // 2]

    mid_idx = frame_count // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
    ret, frame = cap.read()
    cap.release()

    # Fallback on an invalid frame
    if not ret or frame is None:
        cap = cv2.VideoCapture(video_path)
        for offset in range(1, 15):
            for idx in (mid_idx - offset, mid_idx + offset):
                if idx < 0 or idx >= frame_count:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret2, frame2 = cap.read()
                if ret2 and frame2 is not None:
                    cap.release()
                    return frame2
        cap.release()
        raise RuntimeError(f"Could not read a valid frame from: {video_path}")

    return frame


def preprocess_frame(frame_bgr):

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (200, 200), interpolation=cv2.INTER_AREA)

    x = resized.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=-1)  # (200,200,1)
    x = np.expand_dims(x, axis=0)   # (1,200,200,1)
    return x


def get_embedding_for_video(video_path: str):
    frame = extract_middle_frame(video_path)
    x = preprocess_frame(frame)
    embedding = feature_model.predict(x, verbose=0)[0]  # shape: (64,)
    return embedding


def cosine_similarity(a, b, eps=1e-9):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def build_train_library(train_dir="traindata"):
    
    train_dir = Path(train_dir)
    video_exts = (".mp4", ".mov", ".avi", ".m4v", ".mkv")

    train_embeddings = {}

    for label_folder in sorted([p for p in train_dir.iterdir() if p.is_dir()]):
        label = label_folder.name

        # collect videos inside label folder
        vids = [p for p in label_folder.iterdir() if p.suffix.lower() in video_exts]
        if not vids:
            continue

        train_embeddings[label] = []
        for vp in sorted(vids):
            emb = get_embedding_for_video(str(vp))
            train_embeddings[label].append(emb)

        print(f"Loaded train label '{label}': {len(train_embeddings[label])} videos")

    if not train_embeddings:
        raise RuntimeError(f"No training videos found under {train_dir.resolve()}")

    return train_embeddings


def predict_label_for_video(test_video_path: str, train_embeddings: dict):
   
    test_emb = get_embedding_for_video(test_video_path)

    best_label = None
    best_score = -1.0

    for label, emb_list in train_embeddings.items():
        # best match within this label
        label_best = max(cosine_similarity(test_emb, emb) for emb in emb_list)
        if label_best > best_score:
            best_score = label_best
            best_label = label

    return best_label, best_score


# Main program
if __name__ == "__main__":
    #Build training library
    train_lib = build_train_library("traindata")

    #Label -> integer mapping for Results.csv
    label_map = {
        "0": 0, "1": 1, "2": 2, "3": 3, "4": 4,
        "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
        "DecreaseFanSpeed": 10,
        "FanOff": 11,
        "FanOn": 12,
        "IncreaseFanSpeed": 13,
        "LightOff": 14,
        "LightOn": 15,
        "SetThermo": 16
    }

    #Collecting test videos
    test_dir = Path("test")
    patterns = ("*.mp4", "*.mov", "*.avi", "*.m4v", "*.mkv")

    test_videos = []
    for pat in patterns:
        test_videos.extend(test_dir.rglob(pat))

    if not test_videos:
        raise RuntimeError(f"No test videos found in {test_dir.resolve()}")

    #Predict for each test video/store results
    results = []
    print("\n--- Predictions ---")

    for tv in sorted(test_videos):
        pred_label, score = predict_label_for_video(str(tv), train_lib)
        print(f"{tv.name} -> {pred_label} (cosine={score:.4f})")

        if pred_label not in label_map:
            raise RuntimeError(f"Unknown predicted label '{pred_label}' for file {tv.name}")

        results.append(label_map[pred_label])

    #Write Results.csv
    np.savetxt("Results.csv", results, fmt="%d")
    print(f"\nWrote Results.csv with {len(results)} rows.")
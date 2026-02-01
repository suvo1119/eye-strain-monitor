import os
import urllib.request
import cv2
import numpy as np

# MediaPipe Tasks imports
from mediapipe.tasks.python.core import base_options as base_options_lib
from mediapipe.tasks.python.vision import face_landmarker
from mediapipe.tasks.python.vision.core import image as mp_image
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_lib

# Model file (will be downloaded automatically if missing)
MODEL_FILENAME = "face_landmarker_v2.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-assets/face_landmarker_v2.task"

if not os.path.exists(MODEL_FILENAME):
    print(f"Model not found. Downloading {MODEL_FILENAME} from official MediaPipe assets...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_FILENAME)
    print("Download complete.")

# Create Face Landmarker with VIDEO running mode
base_options = base_options_lib.BaseOptions(model_asset_path=MODEL_FILENAME)
options = face_landmarker.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=running_mode_lib.VisionTaskRunningMode.VIDEO,
    num_faces=1,
)
landmarker = face_landmarker.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

from collections import deque
import time

# --- Eye strain monitoring settings ---
WINDOW_SECONDS = 60  # rolling window for metrics
BLINK_EAR_THRESH = 0.25  # threshold for detecting blink/closed eye
PERCLOS_EAR_THRESH = 0.25  # threshold to consider frame as "eyes closed" for PERCLOS

RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_EYE = [33, 160, 158, 133, 153, 144]

# Counters and recommendation settings
blink_count = 0
last_recommendation_time = 0
RECOMMENDATION_COOLDOWN = 300  # seconds between recommendation prompts
RECOMMENDATION_LINES = [
    "High eye strain detected!",
    "Recommendation: Follow 20-20-20 (look 20 ft away for 20 s every 20 min)",
    "Or take a 5-minute break from screens",
]

# Buffers for metrics
frames_buffer = deque()  # stores tuples (timestamp, ear, is_closed)
blink_events = deque()  # stores tuples (start_time, end_time, duration_ms)
current_blink_start = None

# Session-level tracking
session_start_time = time.time()
session_frames = []  # stores tuples (timestamp, ear, is_closed)
session_blinks = []  # stores tuples (start_time, end_time, duration_ms)

def EAR(landmarks, eye):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye]

    vertical = np.linalg.norm(
        np.array([p2.x, p2.y]) - np.array([p6.x, p6.y])
    )
    horizontal = np.linalg.norm(
        np.array([p1.x, p1.y]) - np.array([p4.x, p4.y])
    )

    return vertical / horizontal if horizontal != 0 else 0.0

def purge_old_entries(now):
    # purge frames older than WINDOW_SECONDS
    while frames_buffer and frames_buffer[0][0] < now - WINDOW_SECONDS:
        frames_buffer.popleft()
    # purge blink events older than WINDOW_SECONDS
    while blink_events and blink_events[0][0] < now - WINDOW_SECONDS:
        blink_events.popleft()


def compute_metrics():
    now = time.time()
    purge_old_entries(now)
    frames_len = len(frames_buffer)
    if frames_len == 0:
        return {
            "blinks_per_min": 0.0,
            "perclos": 0.0,
            "avg_ear": 0.0,
            "avg_blink_duration_ms": 0.0,
            "blinks_in_window": 0,
        }

    perclos = 100.0 * sum(1 for (_, _, closed) in frames_buffer if closed) / frames_len
    avg_ear = sum(v for (_, v, _) in frames_buffer) / frames_len
    blinks_in_window = len(blink_events)
    avg_blink_duration_ms = (
        sum(b[2] for b in blink_events) / blinks_in_window if blinks_in_window > 0 else 0.0
    )
    blinks_per_min = blinks_in_window * (60.0 / WINDOW_SECONDS)

    return {
        "blinks_per_min": blinks_per_min,
        "perclos": perclos,
        "avg_ear": avg_ear,
        "avg_blink_duration_ms": avg_blink_duration_ms,
        "blinks_in_window": blinks_in_window,
    }


def strain_level(metrics):
    # Simple heuristic combining blink rate and PERCLOS
    bpm = metrics["blinks_per_min"]
    per = metrics["perclos"]
    dur = metrics["avg_blink_duration_ms"]

    # Evaluate
    if bpm >= 12 and per < 10 and dur < 300:
        return ("Low", (0, 255, 0))  # green
    if 8 <= bpm < 12 or 10 <= per < 20 or 300 <= dur < 400:
        return ("Mild", (0, 255, 255))  # yellow
    if bpm < 8 or per >= 20 or dur >= 400:
        return ("High", (0, 0, 255))  # red
    return ("Moderate", (0, 165, 255))


frame_idx = 0
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 30.0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(frame_idx * (1000.0 / fps))
        detection_result = landmarker.detect_for_video(mp_img, timestamp_ms)
        now = time.time()

        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]
            ear_left = EAR(landmarks, LEFT_EYE)
            ear_right = EAR(landmarks, RIGHT_EYE)
            ear = (ear_left + ear_right) / 2.0

            is_closed = ear < PERCLOS_EAR_THRESH

            # record frame
            frames_buffer.append((now, ear, is_closed))
            session_frames.append((now, ear, is_closed))

            # blink event detection (start/end)
            if ear < BLINK_EAR_THRESH:
                if current_blink_start is None:
                    current_blink_start = now
            else:
                if current_blink_start is not None:
                    duration_ms = (now - current_blink_start) * 1000.0
                    blink_events.append((current_blink_start, now, duration_ms))
                    session_blinks.append((current_blink_start, now, duration_ms))
                    blink_count += 1
                    current_blink_start = None

            # compute metrics
            metrics = compute_metrics()
            level_text, level_color = strain_level(metrics)

            # display metrics
            cv2.putText(frame, f"Blinks(total): {blink_count}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Blinks/min: {metrics['blinks_per_min']:.1f}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(frame, f"Avg blink(ms): {metrics['avg_blink_duration_ms']:.0f}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(frame, f"PERCLOS(%): {metrics['perclos']:.1f}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(frame, f"Strain: {level_text}", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, level_color, 2)

            # Show recommendation overlay when High strain
            if level_text == "High":
                nowt = time.time()
                # update timestamp for recommendation (throttling/recording)
                if nowt - last_recommendation_time >= RECOMMENDATION_COOLDOWN:
                    last_recommendation_time = nowt
                # draw semi-transparent rectangle as background for recommendation
                w_offset = 30
                h_offset = 180
                rect_w = frame.shape[1] - 2 * w_offset
                rect_h = 20 * len(RECOMMENDATION_LINES) + 20
                overlay = frame.copy()
                cv2.rectangle(overlay, (w_offset, h_offset), (w_offset + rect_w, h_offset + rect_h), (0, 0, 255), -1)
                alpha = 0.6
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                for i, line in enumerate(RECOMMENDATION_LINES):
                    y = h_offset + 20 + i * 22
                    cv2.putText(frame, line, (w_offset + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        else:
            # no face detected: show waiting text
            cv2.putText(frame, "No face detected", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Eye Monitor", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

        frame_idx += 1
finally:
    # Generate session report
    import csv
    import math
    import datetime

    session_end_time = time.time()
    duration_s = session_end_time - session_start_time
    duration_min = duration_s / 60.0 if duration_s > 0 else 0.0

    total_frames = len(session_frames)
    total_blinks = len(session_blinks)
    avg_blink_duration_ms = (
        sum(b[2] for b in session_blinks) / total_blinks if total_blinks > 0 else 0.0
    )
    overall_perclos = (
        100.0 * sum(1 for (_, _, closed) in session_frames) / total_frames if total_frames > 0 else 0.0
    )
    avg_ear = sum(ear for (_, ear, _) in session_frames) / total_frames if total_frames > 0 else 0.0
    blinks_per_min = total_blinks / duration_min if duration_min > 0 else 0.0

    final_metrics = {
        "blinks_per_min": blinks_per_min,
        "perclos": overall_perclos,
        "avg_ear": avg_ear,
        "avg_blink_duration_ms": avg_blink_duration_ms,
        "blinks_in_window": total_blinks,
    }
    final_level, final_color = strain_level(final_metrics)

    os.makedirs("reports", exist_ok=True)
    ts = datetime.datetime.fromtimestamp(session_start_time).strftime("%Y%m%d_%H%M%S")
    txt_path = f"reports/session_{ts}.txt"
    csv_path = f"reports/session_{ts}.csv"

    with open(txt_path, "w") as f:
        f.write("Eye Strain Session Report\n")
        f.write(f"Start: {datetime.datetime.fromtimestamp(session_start_time).isoformat()}\n")
        f.write(f"End: {datetime.datetime.fromtimestamp(session_end_time).isoformat()}\n")
        f.write(f"Duration (s): {duration_s:.1f}\n\n")
        f.write(f"Total frames: {total_frames}\n")
        f.write(f"Total blinks: {total_blinks}\n")
        f.write(f"Blinks/min: {blinks_per_min:.2f}\n")
        f.write(f"Avg blink duration (ms): {avg_blink_duration_ms:.1f}\n")
        f.write(f"PERCLOS (%): {overall_perclos:.2f}\n")
        f.write(f"Avg EAR: {avg_ear:.4f}\n")
        f.write(f"Final strain level: {final_level}\n")

    # write minute-by-minute CSV log
    num_minutes = max(1, math.ceil(duration_s / 60.0))
    with open(csv_path, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["minute_index", "start_iso", "frames", "blinks_in_min", "blinks_per_min", "perclos", "avg_blink_ms"])
        for i in range(num_minutes):
            start_i = session_start_time + i * 60
            end_i = start_i + 60
            frames_i = [fr for fr in session_frames if start_i <= fr[0] < end_i]
            frames_count = len(frames_i)
            blinks_i = [b for b in session_blinks if start_i <= b[0] < end_i]
            blinks_in_min = len(blinks_i)
            perclos_i = (
                100.0 * sum(1 for (_, _, c) in frames_i if c) / frames_count if frames_count > 0 else 0.0
            )
            avg_blink_ms_i = (
                sum(b[2] for b in blinks_i) / blinks_in_min if blinks_in_min > 0 else 0.0
            )
            bpm_i = blinks_in_min  # per-minute
            writer.writerow([i, datetime.datetime.fromtimestamp(start_i).isoformat(), frames_count, blinks_in_min, f"{bpm_i:.1f}", f"{perclos_i:.1f}", f"{avg_blink_ms_i:.1f}"])

    print(f"Session report saved: {txt_path}")
    print(f"Minute log saved: {csv_path}")

    # Close resources
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()

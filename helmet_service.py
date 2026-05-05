"""
helmet_service.py - Final Version: Rely on Compliance Rate
"""

import cv2
import time
import os
from typing import List
from ultralytics import YOLO

CLASS_NAMES = {0: "person", 1: "helmet", 2: "no_helmet", 3: "head"}
HELMET_IDS  = {1}
PERSON_IDS  = {0}
NO_HELM_IDS = {2, 3}

GREEN  = (0, 200, 80)
RED    = (0, 60, 220)
BLUE   = (220, 150, 0)
WHITE  = (255, 255, 255)
DARK   = (20, 20, 20)
YELLOW = (0, 215, 255)


def _draw_box(frame, box, cls_id, conf, is_violation=False):
    x1, y1, x2, y2 = map(int, box)
    color = RED if is_violation else (GREEN if cls_id in HELMET_IDS else BLUE)
    label = f"{CLASS_NAMES.get(cls_id, str(cls_id))} {conf:.0%}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
    if is_violation:
        cv2.putText(frame, "NO HELMET!", (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED, 2, cv2.LINE_AA)


def _draw_stats(frame, persons, helmets, violations, fps):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), DARK, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, f"Persons: {persons}",      (10, 32),      cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE,  1)
    cv2.putText(frame, f"Helmets: {helmets}",      (170, 32),     cv2.FONT_HERSHEY_SIMPLEX, 0.65, GREEN,  2)
    cv2.putText(frame, f"No Helmet: {violations}", (320, 32),     cv2.FONT_HERSHEY_SIMPLEX, 0.65, RED,    2)
    cv2.putText(frame, f"FPS: {fps:.1f}",          (w - 110, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.60, YELLOW, 1)
    bar_color = RED if violations > 0 else GREEN
    status    = "  VIOLATION DETECTED" if violations > 0 else "  ALL SAFE"
    cv2.rectangle(frame, (0, h - 35), (w, h), bar_color, -1)
    cv2.putText(frame, status, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)


def run_helmet_detection(
    video_path: str,
    output_dir: str = "outputs",
    model_path: str = "best.pt",
    conf_threshold: float = 0.45,
    scale: float = 0.5,
    sample_every_n_frames: int = 6,
) -> dict:
    

    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fw      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    out_w   = int(fw * scale)
    out_h   = int(fh * scale)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_path   = os.path.join(output_dir, f"{video_name}_annotated.mp4")

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (out_w, out_h))

    frame_results    = []
    frame_idx        = 0
    violation_frames = 0
    safe_frames      = 0
    peak_persons     = 0
    peak_helmets     = 0
    prev_time        = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        if frame_idx % sample_every_n_frames != 0:
            resized = cv2.resize(frame, (out_w, out_h))
            writer.write(resized)
            continue

        results = model(frame, conf=conf_threshold, verbose=False)[0]

        persons_boxes  = []
        helmets_boxes  = []
        nohelm_boxes   = []
        all_detections = []
        conf_vals      = []

        for det in results.boxes:
            cls_id   = int(det.cls[0])
            conf_val = float(det.conf[0])
            box      = det.xyxy[0].tolist()
            conf_vals.append(conf_val)

            is_viol = cls_id in NO_HELM_IDS

            det_dict = {
                "class_id":    cls_id,
                "class_name":  CLASS_NAMES.get(cls_id, str(cls_id)),
                "confidence":  round(conf_val, 4),
                "bbox":        [round(v, 1) for v in box],
                "is_violation": is_viol,
            }
            all_detections.append(det_dict)

            if cls_id in PERSON_IDS:
                persons_boxes.append((box, conf_val))
            elif cls_id in HELMET_IDS:
                helmets_boxes.append((box, conf_val))
            elif cls_id in NO_HELM_IDS:
                nohelm_boxes.append((box, conf_val))

        persons_count = len(persons_boxes)
        helmets_count = len(helmets_boxes)
        no_helmet_count = len(nohelm_boxes)

        # ====================== Violation ======================
        if helmets_count >= 1:
            frame_violations = 0
        else:
            unmatched = max(0, persons_count - helmets_count)
            frame_violations = unmatched + no_helmet_count

        if no_helmet_count >= 3:
            frame_violations = 1

        peak_persons = max(peak_persons, persons_count)
        peak_helmets = max(peak_helmets, helmets_count)

        if frame_violations > 0:
            violation_frames += 1
        else:
            safe_frames += 1

        # رسم
        for box, cv_ in helmets_boxes:
            _draw_box(frame, box, 1, cv_, is_violation=False)
        for i, (box, cv_) in enumerate(persons_boxes):
            _draw_box(frame, box, 0, cv_, is_violation=(i >= len(helmets_boxes)))
        for box, cv_ in nohelm_boxes:
            _draw_box(frame, box, 2, cv_, is_violation=True)

        now = time.time()
        fps = 1 / (now - prev_time + 1e-6)
        prev_time = now
        _draw_stats(frame, persons_count, helmets_count, frame_violations, fps)

        resized = cv2.resize(frame, (out_w, out_h))
        writer.write(resized)

        frame_results.append({
            "frame_number":     frame_idx,
            "persons_count":    persons_count,
            "helmets_count":    helmets_count,
            "no_helmet_count":  no_helmet_count,
            "violations_count": frame_violations,
            "status":           "VIOLATION" if frame_violations > 0 else "SAFE",
            "confidence_avg":   round(sum(conf_vals) / len(conf_vals), 4) if conf_vals else None,
            "detections":       all_detections,
        })

    cap.release()
    writer.release()

    total_analyzed = violation_frames + safe_frames

    # ── النسبة هي اللي تحدد الوضع النهائي ─────────────────────────────
    compliance_rate = round(safe_frames / total_analyzed * 100, 1) if total_analyzed > 0 else 100.0

    # القرار النهائي يعتمد على النسبة فقط
    overall_status = "SAFE" if compliance_rate >= 85 else "VIOLATION"

    return {
        "video_filename":    os.path.basename(video_path),
        "total_frames":      total_analyzed,
        "frame_results":     frame_results,
        "total_persons":     peak_persons,
        "total_helmets":     peak_helmets,
        "total_violations":  violation_frames,
        "safe_frames":       safe_frames,
        "compliance_rate":   compliance_rate,
        "overall_status":    overall_status,
        "output_video_path": out_path,
    }
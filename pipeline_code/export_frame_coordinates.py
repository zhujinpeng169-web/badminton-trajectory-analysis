"""Export per-frame shuttle and two-player pixel coordinates."""
import argparse
import csv
import json
import os

import cv2
import numpy as np
from ultralytics import YOLO


def load_ball_csv(path):
    data = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                frame = int(float(row["Frame"]))
                data[frame] = (int(float(row.get("Visibility", 0))),
                               float(row.get("X", -1)), float(row.get("Y", -1)))
            except (KeyError, TypeError, ValueError):
                continue
    return data


def point_in_quad(point, quad):
    return cv2.pointPolygonTest(quad.astype(np.int32), (float(point[0]), float(point[1])), False) >= 0


def flow_point(prev_gray, gray, point, max_move):
    if prev_gray is None or point is None:
        return None
    p0 = np.float32([[point]])
    p1, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None,
                                                  winSize=(21, 21), maxLevel=3,
                                                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    if p1 is None or status is None or int(status[0, 0]) != 1:
        return None
    out = tuple(float(v) for v in p1[0, 0])
    if error is not None and float(error[0, 0]) > 35.0:
        return None
    if np.linalg.norm(np.subtract(out, point)) > max_move:
        return None
    return out


def choose_players(candidates, state, net_y, max_jump):
    """Keep logical near/far identities stable across detector ID changes."""
    remaining = list(candidates)
    chosen = {"near": None, "far": None}
    for role in ("near", "far"):
        lock = state[role + "_id"]
        last = state[role + "_point"]
        side_candidates = [c for c in remaining if (c[1] > net_y) == (role == "near")]
        pool = side_candidates or remaining
        if not pool:
            continue
        exact = [c for c in pool if lock is not None and c[2] == lock]
        if exact:
            item = exact[0]
        elif last is not None:
            item = min(pool, key=lambda c: np.linalg.norm(np.subtract(c[:2], last)))
            if np.linalg.norm(np.subtract(item[:2], last)) > max_jump:
                continue
        else:
            item = max(pool, key=lambda c: c[1]) if role == "near" else min(pool, key=lambda c: c[1])
        chosen[role] = item
        remaining.remove(item)
    return chosen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="41.MP4")
    p.add_argument("--ball-csv", default="output_tracknet/41_ball.csv")
    p.add_argument("--output", default="output_coordinates/41_coordinates.csv")
    p.add_argument("--model", default="yolov8s-pose.pt")
    p.add_argument("--conf", type=float, default=0.18)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--device", default="")
    p.add_argument("--court-points", default="291,314,690,314,951,535,48,535")
    p.add_argument("--net-y", type=float, default=405.0)
    p.add_argument("--max-track-jump-px", type=float, default=140.0)
    p.add_argument("--flow-max-move-px", type=float, default=45.0)
    p.add_argument("--flow-hold-frames", type=int, default=10)
    p.add_argument("--ball-speed-cap-mps", type=float, default=100.0)
    p.add_argument("--max-frames", type=int, default=0)
    args = p.parse_args()

    ball = load_ball_csv(args.ball_csv)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    model = YOLO(args.model)
    court = [float(v) for v in args.court_points.split(",")]
    court_quad = np.float32([[court[i], court[i + 1]] for i in range(0, 8, 2)])
    destination = np.float32([[0, 0], [6.1, 0], [6.1, 13.4], [0, 13.4]])
    homography = cv2.getPerspectiveTransform(court_quad, destination)

    fields = ["frame", "time_sec", "ball_x", "ball_y", "ball_visible",
              "ball_court_x_m", "ball_court_y_m", "ball_speed_mps", "ball_direction_change_deg", "ball_jump_flag",
              "player_near_x", "player_near_y", "player_far_x", "player_far_y",
              "player_near_track_id", "player_far_track_id", "player_near_source", "player_far_source"]
    state = {"near_id": None, "far_id": None, "near_point": None, "far_point": None,
             "near_missing": 0, "far_missing": 0}
    prev_gray = None
    prev_ball_m = None
    prev_ball_frame = None
    prev_ball_angle = None
    ball_distances = []
    ball_speeds = []
    ball_missing = 0
    ball_valid = 0
    ball_jump_frames = 0
    ball_direction_changes_gt30 = 0
    with open(args.output, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = model.track(source=frame, persist=True, tracker="bytetrack.yaml",
                                  classes=[0], conf=args.conf, imgsz=args.imgsz,
                                  device=args.device or None, verbose=False)[0]
            candidates = []
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else [0] * len(boxes)
                for box, tid, conf in zip(boxes, ids, confs):
                    x1, y1, x2, y2 = map(float, box)
                    if x2 - x1 < 10 or y2 - y1 < 20:
                        continue
                    candidates.append(((x1 + x2) / 2, y2, int(tid), float(conf)))
            candidates = [c for c in candidates if cv2.pointPolygonTest(
                court_quad.astype("int32"), (float(c[0]), float(c[1])), False) >= 0]
            chosen = choose_players(candidates, state, args.net_y, args.max_track_jump_px)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            player_items = {}
            for role in ("near", "far"):
                item = chosen[role]
                source = "detect"
                if item is not None:
                    state[role + "_id"] = item[2]
                    state[role + "_point"] = item[:2]
                    state[role + "_missing"] = 0
                elif state[role + "_point"] is not None and state[role + "_missing"] < args.flow_hold_frames:
                    fp = flow_point(prev_gray, gray, state[role + "_point"], args.flow_max_move_px)
                    if fp is not None and point_in_quad(fp, court_quad):
                        state[role + "_point"] = fp
                        state[role + "_missing"] += 1
                        item = (fp[0], fp[1], state[role + "_id"], 0.0)
                        source = "flow"
                    else:
                        state[role + "_missing"] += 1
                        item = None
                        source = "missing"
                else:
                    state[role + "_missing"] += 1
                    item = None
                    source = "missing"
                player_items[role] = (item, source)
            vis, bx, by = ball.get(frame_idx, (0, -1, -1))
            ball_m = None
            ball_speed = -1.0
            direction_change = -1.0
            jump = 0
            if vis:
                ball_m = cv2.perspectiveTransform(np.float32([[[bx, by]]]), homography)[0, 0]
                ball_valid += 1
                if prev_ball_m is not None and prev_ball_frame is not None:
                    delta = ball_m - prev_ball_m
                    step = float(np.linalg.norm(delta))
                    dt = max((frame_idx - prev_ball_frame) / fps, 1e-9)
                    ball_speed = step / dt
                    if ball_speed <= args.ball_speed_cap_mps:
                        ball_distances.append(step)
                        ball_speeds.append(ball_speed)
                    else:
                        jump = 1
                        ball_jump_frames += 1
                    angle = float(np.degrees(np.arctan2(delta[1], delta[0])))
                    if prev_ball_angle is not None:
                        direction_change = abs((angle - prev_ball_angle + 180.0) % 360.0 - 180.0)
                        if direction_change > 30.0:
                            ball_direction_changes_gt30 += 1
                    prev_ball_angle = angle
                prev_ball_m = ball_m
                prev_ball_frame = frame_idx
            else:
                ball_missing += 1
            row = {"frame": frame_idx, "time_sec": f"{frame_idx / fps:.6f}",
                   "ball_x": bx if vis else -1, "ball_y": by if vis else -1,
                   "ball_visible": int(bool(vis)),
                   "ball_court_x_m": round(float(ball_m[0]), 4) if ball_m is not None else -1,
                   "ball_court_y_m": round(float(ball_m[1]), 4) if ball_m is not None else -1,
                   "ball_speed_mps": round(ball_speed, 4) if ball_speed >= 0 else -1,
                   "ball_direction_change_deg": round(direction_change, 4) if direction_change >= 0 else -1,
                   "ball_jump_flag": jump}
            for prefix, role in (("player_near", "near"), ("player_far", "far")):
                item, source = player_items[role]
                row[prefix + "_x"] = round(item[0], 2) if item else -1
                row[prefix + "_y"] = round(item[1], 2) if item else -1
                row[prefix + "_track_id"] = item[2] if item else -1
                row[prefix + "_source"] = source
            writer.writerow(row)
            prev_gray = gray
            frame_idx += 1
            if frame_idx % 500 == 0:
                print(f"processed {frame_idx}/{total}")
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break
    cap.release()
    print(f"wrote {frame_idx} frames to {args.output}")
    summary = {
        "frames": frame_idx, "fps": fps, "duration_sec": frame_idx / fps,
        "ball_valid_frames": ball_valid, "ball_coverage_pct": 100.0 * ball_valid / max(1, frame_idx),
        "ball_distance_m_filtered": float(sum(ball_distances)),
        "ball_mean_speed_mps_filtered": float(np.mean(ball_speeds)) if ball_speeds else None,
        "ball_max_speed_mps_filtered": float(max(ball_speeds)) if ball_speeds else None,
        "ball_jump_frames": ball_jump_frames,
        "ball_direction_changes_gt30deg": ball_direction_changes_gt30,
        "ball_missing_frames": ball_missing,
        "notes": ["Ball speed is a 2D homography projection.", "Jump frames are retained and flagged in the CSV."]
    }
    summary_path = os.path.splitext(args.output)[0] + "_ball_analysis.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"wrote ball analysis to {summary_path}")


if __name__ == "__main__":
    main()

"""
Evaluation metrics for leaf identity tracking.

Implements:
- Standard tracking: HOTA, AssA, IDF1, MOTA, ID switches
- Custom recovery metrics: recovery success rate, error rate, delay
- Single-leaf area curve accuracy
"""

import json
import os
from collections import defaultdict

import numpy as np


# ── Standard tracking metrics ─────────────────────────────────────────

def compute_iou_matrix(gt_instances, pred_instances):
    """Compute IoU matrix between GT and predicted instances."""
    n_gt = len(gt_instances)
    n_pred = len(pred_instances)
    iou_matrix = np.zeros((n_gt, n_pred))

    for i, gt in enumerate(gt_instances):
        for j, pred in enumerate(pred_instances):
            gt_bbox = gt['bbox']
            pred_bbox = pred['bbox']

            y1 = max(gt_bbox[0], pred_bbox[0])
            x1 = max(gt_bbox[1], pred_bbox[1])
            y2 = min(gt_bbox[2], pred_bbox[2])
            x2 = min(gt_bbox[3], pred_bbox[3])

            inter = max(0, y2 - y1) * max(0, x2 - x1)
            area_gt = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
            area_pred = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
            union = area_gt + area_pred - inter

            iou_matrix[i, j] = inter / (union + 1e-6)

    return iou_matrix


def hungarian_match_gt_pred(gt_instances, pred_instances, iou_threshold=0.5):
    """
    Match GT instances to predictions using IoU + Hungarian algorithm.
    Returns: list of (gt_idx, pred_idx) matches, unmatched GTs, unmatched preds.
    """
    from scipy.optimize import linear_sum_assignment

    if not gt_instances or not pred_instances:
        return [], list(range(len(gt_instances))), list(range(len(pred_instances)))

    iou_matrix = compute_iou_matrix(gt_instances, pred_instances)
    cost_matrix = 1.0 - iou_matrix

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    matched_gt = set()
    matched_pred = set()

    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_threshold:
            matches.append((r, c))
            matched_gt.add(r)
            matched_pred.add(c)

    unmatched_gt = [i for i in range(len(gt_instances)) if i not in matched_gt]
    unmatched_pred = [i for i in range(len(pred_instances)) if i not in matched_pred]

    return matches, unmatched_gt, unmatched_pred


def compute_tracking_metrics(ground_truth_seq, predictions_seq, iou_threshold=0.5):
    """
    Compute standard multi-object tracking metrics.

    ground_truth_seq: list of frames, each with 'instances' containing 'color_idx'
    predictions_seq: list of frames, each with 'assignments' (det_idx -> track_id)
                     and 'instances'

    Returns dict with MOTA, IDF1, HOTA, AssA, ID switches, fragments.
    """
    n_frames = len(ground_truth_seq)
    assert len(predictions_seq) == n_frames

    # Accumulators
    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_id_switches = 0
    total_fragments = 0

    # For IDF1: accumulate match scores over all frames
    # gt_id -> list of (frame, pred_track_id) for each matched GT
    gt_to_pred_over_time = defaultdict(list)

    # Per-frame GT-pred mapping
    frame_gt_pred_map = []

    for fi in range(n_frames):
        gt_frame = ground_truth_seq[fi]
        pred_frame = predictions_seq[fi]

        gt_instances = gt_frame['instances']
        pred_instances = pred_frame['instances']

        # Get predictions with track IDs
        pred_with_ids = []
        for di, inst in enumerate(pred_instances):
            tid = pred_frame['assignments'].get(di, -1)
            pred_with_ids.append({**inst, 'track_id': tid})

        # Match GT to predictions
        matches, unmatched_gt, unmatched_pred = hungarian_match_gt_pred(
            gt_instances, pred_with_ids, iou_threshold
        )

        total_gt += len(gt_instances)
        total_tp += len(matches)
        total_fp += len(unmatched_pred)
        total_fn += len(unmatched_gt)

        # Map GT color_idx to pred track_id
        frame_map = {}
        for gt_idx, pred_idx in matches:
            gt_color = gt_instances[gt_idx]['color_idx']
            pred_tid = pred_with_ids[pred_idx]['track_id']
            frame_map[gt_color] = pred_tid
            gt_to_pred_over_time[gt_color].append((fi, pred_tid))

        frame_gt_pred_map.append(frame_map)

    # ID switches: when a GT identity changes its predicted track ID
    prev_gt_pred = {}
    for fi, frame_map in enumerate(frame_gt_pred_map):
        for gt_color, pred_tid in frame_map.items():
            if gt_color in prev_gt_pred:
                if prev_gt_pred[gt_color] != pred_tid:
                    total_id_switches += 1
            prev_gt_pred[gt_color] = pred_tid

    # Compute MOTA
    mota = 1.0 - (total_fn + total_fp + total_id_switches) / (total_gt + 1e-6) if total_gt > 0 else 0.0

    # Compute IDF1
    # For each GT identity, find the best-matching predicted track
    gt_ids = set()
    for frame_map in frame_gt_pred_map:
        gt_ids.update(frame_map.keys())

    if not gt_ids:
        idf1 = 0.0
    else:
        # Build global GT-identity to pred-track mapping
        # For each GT id, count how many frames it was correctly matched
        idtp = 0  # identity true positives
        idfp = 0  # identity false positives
        idfn = 0  # identity false negatives

        # Simple approach: for each GT, find the most frequent pred track
        gt_to_pred_global = {}
        for gt_color in gt_ids:
            pred_counts = defaultdict(int)
            for fi, frame_map in enumerate(frame_gt_pred_map):
                if gt_color in frame_map:
                    pred_counts[frame_map[gt_color]] += 1
            if pred_counts:
                best_pred = max(pred_counts, key=pred_counts.get)
                gt_to_pred_global[gt_color] = best_pred

        # Count IDTP, IDFP, IDFN
        for fi, frame_map in enumerate(frame_gt_pred_map):
            matched_preds = set()
            for gt_color, pred_tid in frame_map.items():
                if gt_color in gt_to_pred_global and gt_to_pred_global[gt_color] == pred_tid:
                    idtp += 1
                    matched_preds.add(pred_tid)
                else:
                    idfn += 1

            # Count pred tracks not matched to their GT
            all_preds_in_frame = set(frame_map.values())
            for pred_tid in all_preds_in_frame:
                if pred_tid not in matched_preds:
                    # Check if this pred track is the "correct" one for some GT
                    is_correct_for_some = any(
                        gt_to_pred_global.get(gc) == pred_tid
                        for gc in gt_ids
                    )
                    if not is_correct_for_some:
                        idfp += 1

        id_precision = idtp / (idtp + idfp + 1e-6)
        id_recall = idtp / (idtp + idfn + 1e-6)
        idf1 = 2 * id_precision * id_recall / (id_precision + id_recall + 1e-6)

    # Compute HOTA (simplified)
    # HOTA = sqrt(AssA * DetA)
    # DetA = TP / (TP + FP + FN)
    # AssA = average association score
    det_a = total_tp / (total_tp + total_fp + total_fn + 1e-6) if (total_tp + total_fp + total_fn) > 0 else 0.0

    # Association accuracy: for each matched TP, check if the association is correct
    ass_scores = []
    for gt_color in gt_ids:
        if gt_color not in gt_to_pred_global:
            continue
        correct_pred = gt_to_pred_global[gt_color]
        n_correct = 0
        n_total = 0
        for fi, frame_map in enumerate(frame_gt_pred_map):
            if gt_color in frame_map:
                n_total += 1
                if frame_map[gt_color] == correct_pred:
                    n_correct += 1
        if n_total > 0:
            ass_scores.append(n_correct / n_total)

    ass_a = np.mean(ass_scores) if ass_scores else 0.0
    hota = np.sqrt(ass_a * det_a) if ass_a > 0 and det_a > 0 else 0.0

    return {
        'MOTA': mota,
        'IDF1': idf1,
        'HOTA': hota,
        'AssA': ass_a,
        'DetA': det_a,
        'ID_switches': total_id_switches,
        'total_gt': total_gt,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'n_frames': n_frames,
    }


# ── Custom recovery metrics ───────────────────────────────────────────

def compute_recovery_metrics(ground_truth_seq, predictions_seq, missing_events):
    """
    Compute identity recovery metrics for natural occlusion events.

    missing_events: list of {
        'plant_id', 'color_idx', 'disappear_frame', 'reappear_frame',
        'gap_hours', ...
    }

    Each event has exactly one of four outcomes:
      recovered     - the leaf is re-assigned to its pre-occlusion track
      wrong_recovery- it is assigned to a different track that existed
                      before the occlusion (identity swap)
      new_track     - it starts a fresh track ID (identity not recovered)
      not_found     - it is not matched to any predicted instance within
                      the post-occlusion window

    Recovery/error rates are therefore computed over *all* evaluated
    events, not only over the events in which a match was made.

    Returns:
        recovery_stats: {
            'recovery_success_rate': float,
            'error_recovery_rate': float,
            'new_track_rate': float,
            'mean_recovery_delay_hours': float,
            'n_events_evaluated': int,
            'per_event_results': list,
        }
    """
    # Build GT color -> pred track mapping per frame
    frame_gt_pred_map = []
    track_first_frame = {}
    for fi in range(len(ground_truth_seq)):
        gt_frame = ground_truth_seq[fi]
        pred_frame = predictions_seq[fi]

        gt_instances = gt_frame['instances']
        pred_instances = pred_frame['instances']

        matches, _, _ = hungarian_match_gt_pred(gt_instances, pred_instances, 0.5)

        frame_map = {}
        for gt_idx, pred_idx in matches:
            gt_color = gt_instances[gt_idx]['color_idx']
            pred_tid = pred_frame['assignments'].get(pred_idx, -1)
            frame_map[gt_color] = pred_tid
            if pred_tid not in track_first_frame:
                track_first_frame[pred_tid] = fi

        frame_gt_pred_map.append(frame_map)

    # Evaluate each missing event
    per_event = []
    n_success = 0
    n_error = 0
    n_new = 0
    n_missing = 0
    recovery_delays = []

    for event in missing_events:
        cidx = event['color_idx']
        disc_f = event['disappear_frame']
        reap_f = event['reappear_frame']
        gap_h = event['gap_hours']

        # Get pred track ID before disappearance
        pre_track_id = None
        for fi in range(max(0, disc_f - 3), disc_f):
            if fi < len(frame_gt_pred_map) and cidx in frame_gt_pred_map[fi]:
                pre_track_id = frame_gt_pred_map[fi][cidx]
                break

        if pre_track_id is None:
            # No reliable pre-disappearance identity
            per_event.append({
                'event': event,
                'status': 'no_pre_identity',
                'recovery_success': None,
            })
            continue

        # Check what happens at and after reappearance
        post_track_id = None
        recovery_frame = None
        for fi in range(reap_f, min(reap_f + 10, len(frame_gt_pred_map))):
            if cidx in frame_gt_pred_map[fi]:
                post_track_id = frame_gt_pred_map[fi][cidx]
                recovery_frame = fi
                break

        if post_track_id is None:
            per_event.append({
                'event': event,
                'status': 'not_found',
                'recovery_success': False,
                'pre_track_id': pre_track_id,
            })
            n_missing += 1
            continue

        post_is_old = track_first_frame.get(post_track_id, reap_f) < disc_f

        if post_track_id == pre_track_id:
            status = 'recovered'
            n_success += 1
            # Compute delay
            if recovery_frame is not None and reap_f < len(ground_truth_seq):
                delay_hours = (
                    ground_truth_seq[recovery_frame]['hours_since_start'] -
                    ground_truth_seq[reap_f]['hours_since_start']
                )
                recovery_delays.append(delay_hours)
        elif post_is_old:
            status = 'wrong_recovery'
            n_error += 1
        else:
            status = 'new_track'
            n_new += 1

        per_event.append({
            'event': event,
            'status': status,
            'recovery_success': status == 'recovered',
            'pre_track_id': pre_track_id,
            'post_track_id': post_track_id,
            'recovery_frame': recovery_frame,
            'gap_hours': gap_h,
        })

    n_evaluated = n_success + n_error + n_new + n_missing
    denom = n_evaluated + 1e-6
    mean_delay = np.mean(recovery_delays) if recovery_delays else None

    return {
        'recovery_success_rate': n_success / denom,
        'error_recovery_rate': n_error / denom,
        'new_track_rate': n_new / denom,
        'not_found_rate': n_missing / denom,
        'mean_recovery_delay_hours': mean_delay,
        'n_events_evaluated': n_evaluated,
        'n_success': n_success,
        'n_error': n_error,
        'n_new_track': n_new,
        'n_not_found': n_missing,
        'per_event_results': per_event,
    }


# ── Single-leaf area curve metrics ────────────────────────────────────

def compute_area_curve_metrics(ground_truth_seq, predictions_seq, gt_pred_mapping=None):
    """
    Compute single-leaf area curve accuracy.

    For each GT leaf (identified by color_idx), compare the predicted area
    trajectory to the ground truth area trajectory.

    gt_pred_mapping: dict[gt_color] -> pred_track_id.
                     If None, computed from the sequences.

    Returns:
        area_metrics: {
            'mae_pixels': float,
            'mae_relative': float,
            'coverage': float,
            'n_leaves_evaluated': int,
            'per_leaf_metrics': dict,
        }
    """
    # Build GT color -> pred track mapping if not provided
    if gt_pred_mapping is None:
        gt_pred_mapping = {}
        frame_maps = []
        for fi in range(len(ground_truth_seq)):
            gt_frame = ground_truth_seq[fi]
            pred_frame = predictions_seq[fi]
            matches, _, _ = hungarian_match_gt_pred(
                gt_frame['instances'], pred_frame['instances'], 0.5
            )
            fmap = {}
            for gt_idx, pred_idx in matches:
                gc = gt_frame['instances'][gt_idx]['color_idx']
                pt = pred_frame['assignments'].get(pred_idx, -1)
                fmap[gc] = pt
            frame_maps.append(fmap)

        # Use most frequent mapping
        gt_counts = defaultdict(lambda: defaultdict(int))
        for fmap in frame_maps:
            for gc, pt in fmap.items():
                gt_counts[gc][pt] += 1
        for gc, pred_counts in gt_counts.items():
            if pred_counts:
                gt_pred_mapping[gc] = max(pred_counts, key=pred_counts.get)

    # Compute area curves
    per_leaf = {}
    all_errors = []
    all_relative_errors = []
    total_observable = 0
    total_covered = 0

    for gt_color, pred_tid in gt_pred_mapping.items():
        gt_areas = []
        pred_areas = []
        covered_frames = 0
        total_frames = 0

        for fi in range(len(ground_truth_seq)):
            gt_frame = ground_truth_seq[fi]
            pred_frame = predictions_seq[fi]

            # Find GT instance
            gt_area = None
            for inst in gt_frame['instances']:
                if inst['color_idx'] == gt_color:
                    gt_area = inst['pixel_count']
                    break

            # Find pred instance
            pred_area = None
            if gt_area is not None:
                total_frames += 1
                for di, inst in enumerate(pred_frame['instances']):
                    if pred_frame['assignments'].get(di) == pred_tid:
                        pred_area = inst['pixel_count']
                        break

            if gt_area is not None:
                gt_areas.append(gt_area)
                pred_areas.append(pred_area if pred_area is not None else 0)
                if pred_area is not None:
                    covered_frames += 1
                    total_covered += 1
                total_observable += 1

        if gt_areas:
            gt_arr = np.array(gt_areas)
            pred_arr = np.array(pred_areas)
            errors = np.abs(gt_arr - pred_arr)
            relative_errors = errors / (gt_arr + 1e-6)

            per_leaf[gt_color] = {
                'mae_pixels': float(errors.mean()),
                'mae_relative': float(relative_errors.mean()),
                'coverage': covered_frames / (total_frames + 1e-6),
                'n_frames': total_frames,
            }
            all_errors.extend(errors.tolist())
            all_relative_errors.extend(relative_errors.tolist())

    coverage = total_covered / (total_observable + 1e-6) if total_observable > 0 else 0.0

    return {
        'mae_pixels': float(np.mean(all_errors)) if all_errors else 0.0,
        'mae_relative': float(np.mean(all_relative_errors)) if all_relative_errors else 0.0,
        'coverage': coverage,
        'n_leaves_evaluated': len(per_leaf),
        'per_leaf_metrics': per_leaf,
    }


# ── Comprehensive evaluation ──────────────────────────────────────────

def evaluate_tracker(ground_truth_seq, predictions_seq, missing_events=None, tracker_name="unknown"):
    """
    Run all evaluation metrics and return comprehensive results.
    """
    # Standard tracking metrics
    tracking = compute_tracking_metrics(ground_truth_seq, predictions_seq)

    # Recovery metrics
    recovery = None
    if missing_events:
        recovery = compute_recovery_metrics(ground_truth_seq, predictions_seq, missing_events)

    # Area curve metrics
    area = compute_area_curve_metrics(ground_truth_seq, predictions_seq)

    results = {
        'tracker_name': tracker_name,
        'tracking': tracking,
        'recovery': recovery,
        'area_curve': area,
    }

    return results


def format_results_table(all_results):
    """Format results from multiple trackers into a comparison table."""
    lines = []
    lines.append("=" * 100)
    lines.append(f"{'Tracker':<25} {'HOTA':>8} {'AssA':>8} {'IDF1':>8} {'MOTA':>8} {'ID_sw':>8} {'Recov%':>8} {'AreaMAE':>10}")
    lines.append("-" * 100)

    for name, res in all_results.items():
        t = res['tracking']
        r = res.get('recovery') or {}
        a = res.get('area_curve') or {}

        recov_str = f"{r.get('recovery_success_rate', 0)*100:.1f}" if r else "N/A"
        area_str = f"{a.get('mae_pixels', 0):.1f}" if a else "N/A"

        lines.append(
            f"{name:<25} {t['HOTA']:>8.4f} {t['AssA']:>8.4f} {t['IDF1']:>8.4f} "
            f"{t['MOTA']:>8.4f} {t['ID_switches']:>8d} {recov_str:>8} {area_str:>10}"
        )

    lines.append("=" * 100)
    return "\n".join(lines)

"""
T0: Identity ground-truth audit.

Validate that colour-based leaf identity is consistent across frames,
identify natural disappearance/reappearance events, and build the
missing-event catalogue for tracking evaluation.
"""

import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np


def audit_plant_identity(plant_seq, max_gap_hours=48.0):
    """
    Audit a single plant's temporal sequence for identity consistency.

    For each colour index, track:
    - First appearance frame/time
    - Last continuous appearance frame/time
    - Disappearance events (gaps > threshold)
    - Reappearance events
    - Potential identity errors (same colour reappears after very long gap)

    Returns:
        audit: {
            'plant_id': str,
            'color_trajectories': {color_idx: trajectory_info},
            'missing_events': list of events,
            'identity_warnings': list of warnings,
            'summary': summary_stats,
        }
    """
    plant_id = plant_seq[0]['plant_id'] if plant_seq else "unknown"

    # Build per-colour presence timeline
    color_presence = defaultdict(list)  # color_idx -> [(frame_idx, timestamp, hours)]
    for fi, frame in enumerate(plant_seq):
        for inst in frame['instances']:
            cidx = inst['color_idx']
            color_presence[cidx].append((fi, frame['timestamp'], frame['hours_since_start']))

    color_trajectories = {}
    missing_events = []
    identity_warnings = []

    for cidx, presence_list in sorted(color_presence.items()):
        presence_list.sort(key=lambda x: x[0])

        if len(presence_list) == 0:
            continue

        # Find gaps in presence.  A gap means the colour was absent in at
        # least one intermediate frame (curr_fi - prev_fi >= 2); a normal
        # sampling interval between two consecutive observations is NOT a
        # disappearance.
        gaps = []
        for i in range(1, len(presence_list)):
            prev_fi, prev_ts, prev_h = presence_list[i - 1]
            curr_fi, curr_ts, curr_h = presence_list[i]
            gap_frames = curr_fi - prev_fi
            if gap_frames < 2:
                continue
            gap_hours = curr_h - prev_h
            gaps.append({
                'gap_start_frame': prev_fi,
                'gap_end_frame': curr_fi,
                'gap_start_time': prev_ts.isoformat() if prev_ts else None,
                'gap_end_time': curr_ts.isoformat() if curr_ts else None,
                'gap_hours': gap_hours,
                'gap_frames': gap_frames,
            })

        # Classify gaps
        disappearances = []
        reappearances = []
        for gap in gaps:
            if gap['gap_hours'] >= max_gap_hours:
                identity_warnings.append({
                    'color_idx': cidx,
                    'type': 'long_gap_identity_uncertain',
                    'gap_hours': gap['gap_hours'],
                    'message': f"Colour {cidx} has {gap['gap_hours']:.1f}h gap; identity may not be recoverable",
                })

            disappearances.append({
                'color_idx': cidx,
                'disappear_frame': gap['gap_start_frame'],
                'reappear_frame': gap['gap_end_frame'],
                'gap_hours': gap['gap_hours'],
                'gap_start_time': gap['gap_start_time'],
                'gap_end_time': gap['gap_end_time'],
            })

        # Build trajectory segments (continuous presence periods)
        segments = []
        by_frame = {p[0]: p for p in presence_list}
        seg_start = presence_list[0]
        for gap in gaps:
            seg_end = by_frame[gap['gap_start_frame']]
            segments.append({
                'start_frame': seg_start[0],
                'end_frame': seg_end[0],
                'start_time': seg_start[1].isoformat() if seg_start[1] else None,
                'end_time': seg_end[1].isoformat() if seg_end[1] else None,
                'n_frames': seg_end[0] - seg_start[0] + 1,
            })
            seg_start = by_frame[gap['gap_end_frame']]

        # Last segment
        if presence_list:
            seg_end = presence_list[-1]
            segments.append({
                'start_frame': seg_start[0],
                'end_frame': seg_end[0],
                'start_time': seg_start[1].isoformat() if seg_start[1] else None,
                'end_time': seg_end[1].isoformat() if seg_end[1] else None,
                'n_frames': seg_end[0] - seg_start[0] + 1,
            })

        color_trajectories[cidx] = {
            'total_appearances': len(presence_list),
            'n_segments': len(segments),
            'n_gaps': len(gaps),
            'segments': segments,
            'first_appearance': presence_list[0][1].isoformat() if presence_list[0][1] else None,
            'last_appearance': presence_list[-1][1].isoformat() if presence_list[-1][1] else None,
            'total_span_hours': presence_list[-1][2] - presence_list[0][2] if len(presence_list) > 1 else 0,
        }

        # Build missing events for this colour
        for disc in disappearances:
            missing_events.append({
                'plant_id': plant_id,
                'color_idx': cidx,
                'disappear_frame': disc['disappear_frame'],
                'reappear_frame': disc['reappear_frame'],
                'gap_hours': disc['gap_hours'],
                'gap_start_time': disc['gap_start_time'],
                'gap_end_time': disc['gap_end_time'],
                'event_type': 'occlusion' if disc['gap_hours'] < max_gap_hours else 'long_absence',
            })

    # Summary statistics
    all_colors = sorted(color_presence.keys())
    summary = {
        'plant_id': plant_id,
        'n_frames': len(plant_seq),
        'n_unique_colors': len(all_colors),
        'n_colors_with_gaps': sum(1 for c in all_colors if color_trajectories[c]['n_gaps'] > 0),
        'n_missing_events': len(missing_events),
        'n_identity_warnings': len(identity_warnings),
        'mean_instances_per_frame': np.mean([f['n_instances'] for f in plant_seq]),
        'time_span_hours': plant_seq[-1]['hours_since_start'] - plant_seq[0]['hours_since_start'] if len(plant_seq) > 1 else 0,
    }

    return {
        'plant_id': plant_id,
        'color_trajectories': color_trajectories,
        'missing_events': missing_events,
        'identity_warnings': identity_warnings,
        'summary': summary,
    }


def audit_all_plants(sequences, output_dir):
    """
    Run identity audit on all plant sequences.
    Saves per-plant audit results and a global summary.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_audits = {}
    global_events = []
    global_warnings = []
    global_summary = []

    for plant_id, seq in sequences.items():
        if not seq:
            continue
        audit = audit_plant_identity(seq)
        all_audits[plant_id] = audit
        global_events.extend(audit['missing_events'])
        global_warnings.extend(audit['identity_warnings'])
        global_summary.append(audit['summary'])

        # Save per-plant audit
        plant_out = os.path.join(output_dir, f"audit_{plant_id}.json")
        with open(plant_out, 'w') as f:
            json.dump(audit, f, indent=2, default=str)

    # Global summary
    total_events = len(global_events)
    events_by_gap_duration = {
        '<6h': sum(1 for e in global_events if e['gap_hours'] < 6),
        '6-24h': sum(1 for e in global_events if 6 <= e['gap_hours'] < 24),
        '24-48h': sum(1 for e in global_events if 24 <= e['gap_hours'] < 48),
        '>48h': sum(1 for e in global_events if e['gap_hours'] >= 48),
    }

    global_report = {
        'n_plants_audited': len(all_audits),
        'total_missing_events': total_events,
        'events_by_gap_duration': events_by_gap_duration,
        'total_identity_warnings': len(global_warnings),
        'per_plant_summary': global_summary,
    }

    report_path = os.path.join(output_dir, "audit_report_global.json")
    with open(report_path, 'w') as f:
        json.dump(global_report, f, indent=2, default=str)

    # Save all events as a flat list for easy loading
    events_path = os.path.join(output_dir, "missing_events.json")
    with open(events_path, 'w') as f:
        json.dump(global_events, f, indent=2, default=str)

    print(f"Audit complete:")
    print(f"  Plants: {len(all_audits)}")
    print(f"  Missing events: {total_events}")
    print(f"  Events by gap: {events_by_gap_duration}")
    print(f"  Identity warnings: {len(global_warnings)}")
    print(f"  Report: {report_path}")

    return all_audits, global_report


def validate_color_consistency(sequences, sample_plants=None):
    """
    Deep validation: check that the same colour index always maps to
    a spatially consistent leaf (not reused for different leaves).

    For each colour in each plant:
    - Check spatial overlap between first and last appearance
    - Check for unreasonable jumps in position
    - Check if colour is shared across different spatial locations simultaneously

    Returns validation report.
    """
    if sample_plants is None:
        sample_plants = list(sequences.keys())[:30]

    report = []
    for plant_id in sample_plants:
        if plant_id not in sequences:
            continue
        seq = sequences[plant_id]

        # Build per-colour, per-frame spatial data
        color_frames = defaultdict(list)
        for fi, frame in enumerate(seq):
            for inst in frame['instances']:
                color_frames[inst['color_idx']].append({
                    'frame_idx': fi,
                    'center': inst['center'],
                    'bbox': inst['bbox'],
                    'area': inst['pixel_count'],
                    'hours': frame['hours_since_start'],
                })

        plant_report = {
            'plant_id': plant_id,
            'n_colors': len(color_frames),
            'issues': [],
        }

        for cidx, observations in sorted(color_frames.items()):
            if len(observations) < 2:
                continue

            # Check for simultaneous appearances (same frame, same colour, different location)
            frame_groups = defaultdict(list)
            for obs in observations:
                frame_groups[obs['frame_idx']].append(obs)

            for fi, group in frame_groups.items():
                if len(group) > 1:
                    # Same colour appears multiple times in one frame
                    centers = [g['center'] for g in group]
                    max_dist = max(
                        np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
                        for i, c1 in enumerate(centers)
                        for j, c2 in enumerate(centers) if j > i
                    ) if len(centers) > 1 else 0
                    if max_dist > 50:  # pixels
                        plant_report['issues'].append({
                            'type': 'same_color_multiple_instances',
                            'color_idx': cidx,
                            'frame_idx': fi,
                            'n_instances': len(group),
                            'max_distance': max_dist,
                        })

            # Check spatial consistency across time
            centers = [obs['center'] for obs in observations]
            center_std = np.std(centers, axis=0)
            max_jump = 0
            for i in range(1, len(centers)):
                dist = np.sqrt((centers[i][0]-centers[i-1][0])**2 + (centers[i][1]-centers[i][1])**2)
                max_jump = max(max_jump, dist)

            if max_jump > 200:  # very large jump
                plant_report['issues'].append({
                    'type': 'large_spatial_jump',
                    'color_idx': cidx,
                    'max_jump_pixels': max_jump,
                    'center_std': center_std.tolist(),
                    'n_observations': len(observations),
                })

        report.append(plant_report)

    # Summary
    total_issues = sum(len(r['issues']) for r in report)
    print(f"Color consistency validation ({len(report)} plants):")
    print(f"  Total issues found: {total_issues}")
    for r in report:
        if r['issues']:
            print(f"  {r['plant_id']}: {len(r['issues'])} issues")
            for issue in r['issues'][:3]:
                print(f"    - {issue['type']}: color {issue.get('color_idx', '?')}")

    return report


def build_natural_missing_catalogue(sequences, audit_results, min_gap_hours=6.0, max_gap_hours=72.0):
    """
    Build a catalogue of natural missing events suitable for evaluation.

    Filters:
    - Gap must be between min_gap_hours and max_gap_hours
    - Leaf must have reliable identity before disappearance (>= 2 consecutive frames)
    - Leaf must reappear and be observable for >= 2 frames after
    - Record visible neighbours at disappearance and reappearance

    Returns list of evaluation-ready events.
    """
    catalogue = []

    for plant_id, audit in audit_results.items():
        if plant_id not in sequences:
            continue
        seq = sequences[plant_id]

        for event in audit['missing_events']:
            cidx = event['color_idx']
            gap_h = event['gap_hours']

            if gap_h < min_gap_hours or gap_h > max_gap_hours:
                continue

            disc_f = event['disappear_frame']
            reap_f = event['reappear_frame']

            # Check pre-disappearance reliability
            pre_count = 0
            for fi in range(disc_f - 1, -1, -1):
                if fi < len(seq) and any(inst['color_idx'] == cidx for inst in seq[fi]['instances']):
                    pre_count += 1
                else:
                    break

            # Check post-reappearance reliability
            post_count = 0
            for fi in range(reap_f, min(reap_f + 10, len(seq))):
                if any(inst['color_idx'] == cidx for inst in seq[fi]['instances']):
                    post_count += 1
                else:
                    break

            if pre_count < 2 or post_count < 2:
                continue

            # Count visible neighbours at disappearance
            pre_neighbors = 0
            if disc_f < len(seq):
                pre_neighbors = len(seq[disc_f]['instances']) - 1  # exclude self

            # Count visible neighbours at reappearance
            post_neighbors = 0
            if reap_f < len(seq):
                post_neighbors = len(seq[reap_f]['instances']) - 1

            catalogue.append({
                'plant_id': plant_id,
                'color_idx': cidx,
                'disappear_frame': disc_f,
                'reappear_frame': reap_f,
                'gap_hours': gap_h,
                'gap_start_time': event['gap_start_time'],
                'gap_end_time': event['gap_end_time'],
                'pre_reliable_frames': pre_count,
                'post_reliable_frames': post_count,
                'pre_neighbors': pre_neighbors,
                'post_neighbors': post_neighbors,
                'event_type': event['event_type'],
            })

    # Sort by gap duration
    catalogue.sort(key=lambda e: e['gap_hours'])

    print(f"Natural missing catalogue: {len(catalogue)} events")
    if catalogue:
        gaps = [e['gap_hours'] for e in catalogue]
        print(f"  Gap range: {min(gaps):.1f}h - {max(gaps):.1f}h")
        print(f"  Median gap: {np.median(gaps):.1f}h")

    return catalogue


if __name__ == "__main__":
    import sys
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.data_parser import build_plant_sequences_fast

    manifest = os.environ.get(
        'MANIFEST_PATH',
        str(PROJECT_ROOT.parent / 'phenoleaf_lowlabel' / 'data_audit' / 'manifest.json')
    )
    print("Loading plant sequences...")
    sequences = build_plant_sequences_fast(manifest)

    output_dir = os.environ.get('OUTPUT_DIR', str(PROJECT_ROOT / 'data' / 'identity_audit'))
    os.makedirs(output_dir, exist_ok=True)
    print("\n=== Running identity audit ===")
    audits, report = audit_all_plants(sequences, output_dir)

    print("\n=== Validating color consistency ===")
    validation = validate_color_consistency(sequences)

    print("\n=== Building natural missing catalogue ===")
    catalogue = build_natural_missing_catalogue(sequences, audits)

    cat_path = os.path.join(output_dir, "natural_missing_catalogue.json")
    with open(cat_path, 'w') as f:
        json.dump(catalogue, f, indent=2, default=str)
    print(f"Catalogue saved: {cat_path}")

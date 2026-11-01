"""
Main experiment runner for the leaf identity tracking paper.

Orchestrates:
1. Data loading and identity audit (T0)
2. Baseline and proposed tracker execution (T1-T3)
3. Evaluation and result collection
4. Result reporting and saving
"""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# Resolve project root relative to this script
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_parser import build_plant_sequences_fast, parse_mask_instances
from src.identity_audit import (
    audit_all_plants,
    validate_color_consistency,
    build_natural_missing_catalogue,
)
from src.trackers import (
    TRACKERS,
    run_tracker_on_plant,
    GeometricHungarianTracker,
    GeometricShapeTracker,
    AppearanceMemoryTracker,
    RelationTracker,
)
from evaluation.metrics import (
    evaluate_tracker,
    format_results_table,
    compute_tracking_metrics,
    compute_recovery_metrics,
    compute_area_curve_metrics,
)


# ── Configuration ─────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Data
    'manifest_path': r"D:\qodertu\phenoleaf_lowlabel\data_audit\manifest.json",
    'plant_split_path': r"D:\qodertu\phenoleaf_lowlabel\splits\plant_split.json",

    # Output
    'output_dir': str(PROJECT_ROOT / "results"),
    'audit_dir': str(PROJECT_ROOT / "data" / "identity_audit"),

    # Tracker configs
    'tracker_configs': {
        'geometric_hungarian': {
            'max_distance': 150.0,
            'min_iou': 0.1,
            'max_missing_hours': 48.0,
        },
        'geometric_shape': {
            'max_distance': 150.0,
            'min_iou': 0.1,
            'max_missing_hours': 48.0,
            'shape_weight': 0.5,
        },
        'appearance_memory': {
            'max_distance': 150.0,
            'min_iou': 0.1,
            'max_missing_hours': 48.0,
            'memory_length': 5,
            'appearance_weight': 0.3,
        },
        'relation_tracker': {
            'max_distance': 150.0,
            'min_iou': 0.1,
            'max_missing_hours': 72.0,
            'memory_length': 8,
            'relation_weight': 0.3,
            'min_anchors': 2,
            'rejection_threshold': 0.6,
            'margin_threshold': 0.15,
            'new_leaf_penalty': 0.2,
        },
    },

    # Experiment
    'max_plants': None,  # None = all plants
    'random_seed': 42,
    'iou_threshold': 0.5,
}


def load_config(config_path=None):
    """Load configuration from file or use defaults."""
    config = DEFAULT_CONFIG.copy()
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            user_config = json.load(f)
        config.update(user_config)
    return config


# ── Phase T0: Data validation ─────────────────────────────────────────

def run_phase_t0(config):
    """
    T0: Validate ground truth identity and build missing event catalogue.
    """
    print("=" * 60)
    print("PHASE T0: Ground-truth identity validation")
    print("=" * 60)

    manifest_path = config['manifest_path']
    audit_dir = config['audit_dir']
    os.makedirs(audit_dir, exist_ok=True)

    # Load sequences
    print("\n[1/4] Loading plant sequences...")
    t0 = time.time()
    sequences = build_plant_sequences_fast(manifest_path)
    print(f"  Loaded {len(sequences)} plants in {time.time()-t0:.1f}s")

    # Save sequence summary
    summary_path = os.path.join(audit_dir, "plant_sequences_summary.json")
    seq_summary = {}
    for pid, seq in sequences.items():
        seq_summary[pid] = {
            'n_frames': len(seq),
            'time_span_hours': seq[-1]['hours_since_start'] if seq else 0,
            'mean_instances': np.mean([f['n_instances'] for f in seq]) if seq else 0,
        }
    with open(summary_path, 'w') as f:
        json.dump(seq_summary, f, indent=2)
    print(f"  Summary saved: {summary_path}")

    # Run identity audit
    print("\n[2/4] Running identity audit...")
    t0 = time.time()
    audits, report = audit_all_plants(sequences, audit_dir)
    print(f"  Audit complete in {time.time()-t0:.1f}s")

    # Validate color consistency
    print("\n[3/4] Validating color consistency...")
    validation = validate_color_consistency(sequences)

    # Build missing event catalogue
    print("\n[4/4] Building natural missing catalogue...")
    catalogue = build_natural_missing_catalogue(sequences, audits)

    cat_path = os.path.join(audit_dir, "natural_missing_catalogue.json")
    with open(cat_path, 'w') as f:
        json.dump(catalogue, f, indent=2, default=str)

    print(f"\nT0 Summary:")
    print(f"  Total plants: {len(sequences)}")
    print(f"  Total missing events: {len(catalogue)}")
    print(f"  Catalogue saved: {cat_path}")

    return sequences, audits, catalogue


# ── Phase T1: Baseline tracking ──────────────────────────────────────

def run_phase_t1(config, sequences, catalogue):
    """
    T1: Run all baseline trackers and proposed method.
    """
    print("\n" + "=" * 60)
    print("PHASE T1: Tracker execution")
    print("=" * 60)

    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Load plant split
    with open(config['plant_split_path']) as f:
        split = json.load(f)

    train_plants = [tuple(p) for p in split['train_plants']]
    val_plants = [tuple(p) for p in split['val_plants']]
    test_plants = [tuple(p) for p in split['test_plants']]

    print(f"\nPlant split: train={len(train_plants)}, val={len(val_plants)}, test={len(test_plants)}")

    # Group sequences by plant_id for split lookup
    plant_to_seq = {}
    for pid, seq in sequences.items():
        plant_to_seq[pid] = seq

    # Select evaluation plants (use val + test for now)
    eval_plant_ids = list(set(val_plants + test_plants))
    eval_plant_ids = [pid for pid in eval_plant_ids if pid[0] in sequences]

    if config['max_plants']:
        eval_plant_ids = eval_plant_ids[:config['max_plants']]

    print(f"Evaluating on {len(eval_plant_ids)} plants")

    # Build per-plant missing events
    plant_events = defaultdict(list)
    for event in catalogue:
        plant_events[event['plant_id']].append(event)

    # Run each tracker
    tracker_configs = config['tracker_configs']
    all_results = {}

    for tracker_name, tracker_config in tracker_configs.items():
        print(f"\n--- Running {tracker_name} ---")
        t0 = time.time()

        tracker_results = []
        per_plant_results = {}

        for plant_id in eval_plant_ids:
            if plant_id not in sequences:
                continue

            seq = sequences[plant_id]
            if not seq:
                continue

            # Run tracker
            tracker_cls = TRACKERS[tracker_name]
            tracker = tracker_cls(tracker_config)
            predictions = tracker.process_sequence(seq)

            # Build GT sequence in same format
            gt_seq = []
            for frame in seq:
                gt_seq.append({
                    'instances': frame['instances'],
                    'hours_since_start': frame['hours_since_start'],
                })

            # Evaluate
            events = plant_events.get(plant_id, [])
            result = evaluate_tracker(gt_seq, predictions, events, tracker_name)
            tracker_results.append(result)
            per_plant_results[plant_id] = result

        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")

        # Aggregate results
        agg = aggregate_results(tracker_results, tracker_name)
        all_results[tracker_name] = {
            'aggregated': agg,
            'per_plant': per_plant_results,
            'elapsed_seconds': elapsed,
        }

    # Print comparison table
    print("\n" + format_results_table({k: v['aggregated'] for k, v in all_results.items()}))

    # Save results
    results_path = os.path.join(output_dir, "tracker_results.json")
    saveable_results = {}
    for name, res in all_results.items():
        saveable_results[name] = {
            'aggregated': res['aggregated'],
            'elapsed_seconds': res['elapsed_seconds'],
        }
    with open(results_path, 'w') as f:
        json.dump(saveable_results, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")

    return all_results


def aggregate_results(per_plant_results, tracker_name):
    """Aggregate per-plant results into overall metrics."""
    if not per_plant_results:
        return {'tracker_name': tracker_name, 'n_plants': 0}

    # Average tracking metrics
    tracking_keys = ['HOTA', 'AssA', 'IDF1', 'MOTA', 'ID_switches', 'total_gt', 'total_tp', 'total_fp', 'total_fn']
    agg_tracking = {}
    for key in tracking_keys:
        values = [r['tracking'][key] for r in per_plant_results if key in r['tracking']]
        if key in ('ID_switches', 'total_gt', 'total_tp', 'total_fp', 'total_fn'):
            agg_tracking[key] = sum(values)
        else:
            agg_tracking[key] = np.mean(values) if values else 0.0

    # Recompute rates from totals
    total_gt = agg_tracking.get('total_gt', 0)
    total_tp = agg_tracking.get('total_tp', 0)
    total_fp = agg_tracking.get('total_fp', 0)
    total_fn = agg_tracking.get('total_fn', 0)
    agg_tracking['MOTA'] = 1.0 - (total_fn + total_fp + agg_tracking['ID_switches']) / (total_gt + 1e-6) if total_gt > 0 else 0.0

    # Average recovery metrics
    recovery_vals = [r['recovery'] for r in per_plant_results if r.get('recovery')]
    agg_recovery = None
    if recovery_vals:
        agg_recovery = {
            'recovery_success_rate': np.mean([v['recovery_success_rate'] for v in recovery_vals]),
            'error_recovery_rate': np.mean([v['error_recovery_rate'] for v in recovery_vals]),
            'n_events_evaluated': sum(v['n_events_evaluated'] for v in recovery_vals),
            'n_success': sum(v['n_success'] for v in recovery_vals),
            'n_error': sum(v['n_error'] for v in recovery_vals),
        }

    # Average area metrics
    area_vals = [r['area_curve'] for r in per_plant_results if r.get('area_curve')]
    agg_area = None
    if area_vals:
        agg_area = {
            'mae_pixels': np.mean([v['mae_pixels'] for v in area_vals]),
            'mae_relative': np.mean([v['mae_relative'] for v in area_vals]),
            'coverage': np.mean([v['coverage'] for v in area_vals]),
            'n_leaves_evaluated': sum(v['n_leaves_evaluated'] for v in area_vals),
        }

    return {
        'tracker_name': tracker_name,
        'n_plants': len(per_plant_results),
        'tracking': agg_tracking,
        'recovery': agg_recovery,
        'area_curve': agg_area,
    }


# ── Phase T2: Ablation ───────────────────────────────────────────────

def run_ablation(config, sequences, catalogue):
    """
    T2: Run ablation experiments.
    """
    print("\n" + "=" * 60)
    print("PHASE T2: Ablation study")
    print("=" * 60)

    # Ablation variants of the relation tracker
    ablation_configs = {
        'proposed_full': {
            'max_distance': 150.0,
            'min_iou': 0.1,
            'max_missing_hours': 72.0,
            'memory_length': 8,
            'relation_weight': 0.3,
            'min_anchors': 2,
            'rejection_threshold': 0.6,
            'margin_threshold': 0.15,
            'new_leaf_penalty': 0.2,
        },
        'no_relation': {  # Remove relation term
            'max_distance': 150.0,
            'min_iou': 0.1,
            'max_missing_hours': 72.0,
            'memory_length': 8,
            'relation_weight': 0.0,  # no relation
            'min_anchors': 2,
            'rejection_threshold': 0.6,
            'margin_threshold': 0.15,
            'new_leaf_penalty': 0.2,
        },
        'no_anchor_weighting': {  # Relation without anchor reliability
            'max_distance': 150.0,
            'min_iou': 0.1,
            'max_missing_hours': 72.0,
            'memory_length': 8,
            'relation_weight': 0.3,
            'min_anchors': 1,  # lower anchor requirement
            'rejection_threshold': 0.6,
            'margin_threshold': 0.15,
            'new_leaf_penalty': 0.2,
        },
        'no_memory': {  # No history, only last observation
            'max_distance': 150.0,
            'min_iou': 0.1,
            'max_missing_hours': 72.0,
            'memory_length': 1,  # only last
            'relation_weight': 0.3,
            'min_anchors': 2,
            'rejection_threshold': 0.6,
            'margin_threshold': 0.15,
            'new_leaf_penalty': 0.2,
        },
        'no_rejection': {  # No rejection mechanism
            'max_distance': 150.0,
            'min_iou': 0.1,
            'max_missing_hours': 72.0,
            'memory_length': 8,
            'relation_weight': 0.3,
            'min_anchors': 2,
            'rejection_threshold': 1.0,  # very high = never reject
            'margin_threshold': 0.0,
            'new_leaf_penalty': 0.2,
        },
    }

    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    with open(config['plant_split_path']) as f:
        split = json.load(f)
    val_plants = [tuple(p) for p in split['val_plants']]
    eval_plant_ids = [pid for pid in val_plants if pid in sequences]

    plant_events = defaultdict(list)
    for event in catalogue:
        plant_events[event['plant_id']].append(event)

    ablation_results = {}

    for variant_name, variant_config in ablation_configs.items():
        print(f"\n--- Ablation: {variant_name} ---")

        tracker = RelationTracker(variant_config)
        variant_results = []

        for plant_id in eval_plant_ids:
            seq = sequences[plant_id]
            if not seq:
                continue

            tracker_variant = RelationTracker(variant_config)
            predictions = tracker_variant.process_sequence(seq)

            gt_seq = [{'instances': f['instances'], 'hours_since_start': f['hours_since_start']} for f in seq]
            events = plant_events.get(plant_id, [])
            result = evaluate_tracker(gt_seq, predictions, events, variant_name)
            variant_results.append(result)

        agg = aggregate_results(variant_results, variant_name)
        ablation_results[variant_name] = agg

        t = agg['tracking']
        print(f"  HOTA={t['HOTA']:.4f}  AssA={t['AssA']:.4f}  IDF1={t['IDF1']:.4f}")
        if agg.get('recovery'):
            print(f"  Recovery={agg['recovery']['recovery_success_rate']:.3f}")

    # Save ablation results
    abl_path = os.path.join(output_dir, "ablation_results.json")
    with open(abl_path, 'w') as f:
        json.dump(ablation_results, f, indent=2, default=str)
    print(f"\nAblation results saved: {abl_path}")

    return ablation_results


# ── Main entry point ──────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Leaf Identity Tracking Experiments")
    parser.add_argument('--config', type=str, default=None, help='Config JSON path')
    parser.add_argument('--phase', type=str, default='all', choices=['t0', 't1', 't2', 'all'])
    parser.add_argument('--max-plants', type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.max_plants:
        config['max_plants'] = args.max_plants

    print("Leaf Identity Tracking Experiment")
    print(f"Config: {json.dumps(config, indent=2, default=str)[:500]}")

    if args.phase in ('t0', 'all'):
        sequences, audits, catalogue = run_phase_t0(config)
    else:
        # Load cached data
        sequences = build_plant_sequences_fast(config['manifest_path'])
        audit_dir = config['audit_dir']
        cat_path = os.path.join(audit_dir, "natural_missing_catalogue.json")
        with open(cat_path) as f:
            catalogue = json.load(f)
        audits = {}

    if args.phase in ('t1', 'all'):
        results = run_phase_t1(config, sequences, catalogue)

    if args.phase in ('t2', 'all'):
        ablation = run_ablation(config, sequences, catalogue)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

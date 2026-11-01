"""
Full experiment: load from cache, run all trackers, evaluate, save results.
"""
import sys, os, json, time
from pathlib import Path
from collections import defaultdict
import numpy as np

# Resolve project root relative to this script
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR
sys.path.insert(0, str(PROJECT_ROOT))

from src.cache_manager import load_sequences_cache
from src.identity_audit import audit_plant_identity, build_natural_missing_catalogue
from src.trackers import (
    GeometricHungarianTracker, GeometricShapeTracker,
    AppearanceMemoryTracker, RelationTracker,
)
from evaluation.metrics import evaluate_tracker, format_results_table
from scipy.stats import wilcoxon

# Data paths - update these to point to your local data
MANIFEST = os.environ.get(
    'MANIFEST_PATH',
    str(PROJECT_ROOT.parent / 'phenoleaf_lowlabel' / 'data_audit' / 'manifest.json')
)
SPLIT_PATH = os.environ.get(
    'SPLIT_PATH',
    str(PROJECT_ROOT.parent / 'phenoleaf_lowlabel' / 'splits' / 'plant_split.json')
)
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', str(PROJECT_ROOT / 'results'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Step 1: Load cached data ──────────────────────────────────────────
print("[1/5] Loading cached sequences...")
t0 = time.time()
sequences = load_sequences_cache(MANIFEST)
print(f"  Loaded {len(sequences)} sequences in {time.time()-t0:.1f}s")

# ── Step 2: Missing-event catalogue (genuine occlusion events) ────────
print("\n[2/5] Loading missing-event catalogue...")
catalogue_path = os.path.join(OUTPUT_DIR, "event_catalogue.json")
if os.path.exists(catalogue_path):
    with open(catalogue_path) as f:
        catalogue = json.load(f)
    print(f"  Loaded {len(catalogue)} genuine events from cache")
else:
    audits = {}
    for seq_key, seq in sequences.items():
        if not seq:
            continue
        audits[seq_key] = audit_plant_identity(seq, max_gap_hours=48.0)
    catalogue = build_natural_missing_catalogue(
        sequences, audits, min_gap_hours=12.0, max_gap_hours=72.0)
    for ev in catalogue:
        ev['seq_key'] = ev['plant_id']
    with open(catalogue_path, 'w') as f:
        json.dump(catalogue, f, indent=2, default=str)
    print(f"  Built {len(catalogue)} genuine events (cached)")

# ── Step 3: Load plant split ──────────────────────────────────────────
print("\n[3/5] Loading plant split...")
with open(SPLIT_PATH) as f:
    split = json.load(f)

# Convert split plants to seq_keys
train_keys = set()
val_keys = set()
test_keys = set()
for pid, rid in split['train_plants']:
    train_keys.add(f"{pid}__{rid}")
for pid, rid in split['val_plants']:
    val_keys.add(f"{pid}__{rid}")
for pid, rid in split['test_plants']:
    test_keys.add(f"{pid}__{rid}")

# Use val + test for evaluation
eval_keys = sorted(val_keys | test_keys)
eval_keys = [k for k in eval_keys if k in sequences]
print(f"  Eval plants: {len(eval_keys)} (val={len(val_keys & set(sequences.keys()))}, test={len(test_keys & set(sequences.keys()))})")

# Build per-plant event lookup
plant_events = defaultdict(list)
for ev in catalogue:
    sk = ev.get('seq_key', ev['plant_id'])
    plant_events[sk].append(ev)

# ── Step 4: Run all trackers ──────────────────────────────────────────
print("\n[4/5] Running trackers...")

tracker_classes = {
    'geometric_hungarian': GeometricHungarianTracker,
    'geometric_shape': GeometricShapeTracker,
    'appearance_memory': AppearanceMemoryTracker,
    'relation_tracker': RelationTracker,
}

tracker_configs = {
    # All methods share the same 72 h memory window, so any difference in
    # recovery is attributable to the association evidence rather than to a
    # longer tolerated occlusion.
    'geometric_hungarian': {'max_distance': 150.0, 'min_iou': 0.1, 'max_missing_hours': 72.0},
    'geometric_shape': {'max_distance': 150.0, 'min_iou': 0.1, 'max_missing_hours': 72.0, 'shape_weight': 0.5},
    'appearance_memory': {'max_distance': 150.0, 'min_iou': 0.1, 'max_missing_hours': 72.0, 'memory_length': 5, 'appearance_weight': 0.3},
    'relation_tracker': {
        'max_distance': 150.0, 'min_iou': 0.1, 'max_missing_hours': 72.0,
        'memory_length': 16, 'geo_weight': 0.6, 'relation_weight': 0.4,
        'hist_weight': 0.1, 'min_anchors': 2, 'anchor_dist_threshold': 250,
        'max_anchors': 6, 'rejection_threshold': 6.0, 'no_anchor_gate': 6.0,
        'anchor_support': 0.45, 'extended_distance': 300.0,
        'veto_threshold': 0.45, 'verify_active': True,
    },
}

all_results = {}
per_plant_store = {}
event_store = {}


def bootstrap_ci(success_flags, n_boot=2000, seed=0):
    """Percentile bootstrap CI for a recovery success rate."""
    if not success_flags:
        return None
    rng = np.random.default_rng(seed)
    arr = np.asarray(success_flags, dtype=float)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


for tname, tcls in tracker_classes.items():
    print(f"\n  --- {tname} ---")
    t0 = time.time()
    config = tracker_configs[tname]
    
    per_plant = []
    per_plant_metrics = []
    plant_event_results = []
    for seq_key in eval_keys:
        seq = sequences[seq_key]
        if not seq:
            continue
        
        tracker = tcls(config)
        predictions = tracker.process_sequence(seq)
        
        gt_seq = [{'instances': f['instances'], 'hours_since_start': f['hours_since_start']} for f in seq]
        events = plant_events.get(seq_key, [])
        
        result = evaluate_tracker(gt_seq, predictions, events if events else None, tname)
        per_plant.append(result)

        rec = result.get('recovery')
        per_plant_metrics.append({
            'seq_key': seq_key,
            'HOTA': result['tracking']['HOTA'],
            'AssA': result['tracking']['AssA'],
            'IDF1': result['tracking']['IDF1'],
            'MOTA': result['tracking']['MOTA'],
            'ID_switches': result['tracking']['ID_switches'],
            'recovery_success_rate': rec['recovery_success_rate'] if rec else None,
            'error_recovery_rate': rec['error_recovery_rate'] if rec else None,
            'n_events': rec['n_events_evaluated'] if rec else 0,
        })
        if rec:
            for pe in rec['per_event_results']:
                if pe.get('recovery_success') is None:
                    continue
                ev = pe['event']
                plant_event_results.append({
                    'plant': seq_key,
                    'color_idx': ev.get('color_idx'),
                    'disappear_frame': ev.get('disappear_frame'),
                    'reappear_frame': ev.get('reappear_frame'),
                    'gap_hours': ev.get('gap_hours'),
                    'post_neighbors': ev.get('post_neighbors'),
                    'status': pe['status'],
                    'success': pe['recovery_success'],
                })
    
    elapsed = time.time() - t0
    per_plant_store[tname] = per_plant_metrics
    event_store[tname] = plant_event_results
    
    # Aggregate
    if per_plant:
        with_rec = [r for r in per_plant if r.get('recovery')]
        agg_tracking = {
            'HOTA': np.mean([r['tracking']['HOTA'] for r in per_plant]),
            'AssA': np.mean([r['tracking']['AssA'] for r in per_plant]),
            'IDF1': np.mean([r['tracking']['IDF1'] for r in per_plant]),
            'MOTA': np.mean([r['tracking']['MOTA'] for r in per_plant]),
            'ID_switches': sum(r['tracking']['ID_switches'] for r in per_plant),
            'total_tp': sum(r['tracking']['total_tp'] for r in per_plant),
            'total_fp': sum(r['tracking']['total_fp'] for r in per_plant),
            'total_fn': sum(r['tracking']['total_fn'] for r in per_plant),
        }
        
        # Aggregate recovery (pooled over events, unconditional)
        recov_rates = [r['recovery']['recovery_success_rate'] for r in with_rec]
        error_rates = [r['recovery']['error_recovery_rate'] for r in with_rec]
        n_events = sum(r['recovery']['n_events_evaluated'] for r in with_rec)
        succ = [1.0 if pe['success'] else 0.0 for pe in plant_event_results]
        err = [1.0 if pe['status'] == 'wrong_recovery' else 0.0
               for pe in plant_event_results]
        new = [1.0 if pe['status'] == 'new_track' else 0.0
               for pe in plant_event_results]
        nf = [1.0 if pe['status'] == 'not_found' else 0.0
              for pe in plant_event_results]
        ci = bootstrap_ci(succ)
        
        agg_recovery = {
            'recovery_success_rate': float(np.mean(succ)) if succ else 0,
            'recovery_success_rate_plantmean': float(np.mean(recov_rates)) if recov_rates else 0,
            'recovery_success_rate_pooled': float(np.mean(succ)) if succ else 0,
            'recovery_ci95': ci,
            'error_recovery_rate': float(np.mean(err)) if err else 0,
            'new_track_rate': float(np.mean(new)) if new else 0,
            'not_found_rate': float(np.mean(nf)) if nf else 0,
            'error_recovery_rate_plantmean': float(np.mean(error_rates)) if error_rates else 0,
            'n_events': n_events,
            'n_plants_with_events': len(with_rec),
        }
        
        # Aggregate area
        area_maes = [r['area_curve']['mae_pixels'] for r in per_plant if r.get('area_curve')]
        area_covs = [r['area_curve']['coverage'] for r in per_plant if r.get('area_curve')]
        
        agg_area = {
            'mae_pixels': np.mean(area_maes) if area_maes else 0,
            'coverage': np.mean(area_covs) if area_covs else 0,
        }
        
        all_results[tname] = {
            'tracking': agg_tracking,
            'recovery': agg_recovery,
            'area_curve': agg_area,
            'elapsed': elapsed,
        }
        
        t = agg_tracking
        r = agg_recovery
        a = agg_area
        print(f"    HOTA={t['HOTA']:.4f}  AssA={t['AssA']:.4f}  IDF1={t['IDF1']:.4f}  MOTA={t['MOTA']:.4f}  ID_sw={t['ID_switches']}")
        print(f"    Recovery={r['recovery_success_rate']:.3f}  CI={r['recovery_ci95']}  "
              f"Error={r['error_recovery_rate']:.3f}  NewTrack={r['new_track_rate']:.3f}  Events={r['n_events']}")
        print(f"    AreaMAE={a['mae_pixels']:.1f}px  Coverage={a['coverage']:.3f}")
        print(f"    Time: {elapsed:.1f}s")

# ── Paired significance test (per-plant recovery rates) ───────────────
stats = {}
if 'relation_tracker' in per_plant_store and 'geometric_hungarian' in per_plant_store:
    base = {m['seq_key']: m for m in per_plant_store['geometric_hungarian']}
    ours = {m['seq_key']: m for m in per_plant_store['relation_tracker']}
    common = [k for k in ours if k in base
              and ours[k]['recovery_success_rate'] is not None
              and base[k]['recovery_success_rate'] is not None]
    diffs = np.array([ours[k]['recovery_success_rate'] -
                      base[k]['recovery_success_rate'] for k in common])
    try:
        stat, p = wilcoxon(diffs)
    except ValueError:
        stat, p = None, None
    stats['relation_vs_geometric_recovery'] = {
        'n_plants': len(common),
        'median_diff': float(np.median(diffs)),
        'mean_diff': float(np.mean(diffs)),
        'wilcoxon_stat': float(stat) if stat is not None else None,
        'p_value': float(p) if p is not None else None,
    }
    print("\nPaired test (per-plant recovery rate, ours vs geometric):")
    if p is not None:
        print(f"  n={len(common)}  median diff={np.median(diffs)*100:+.2f} pts  "
              f"mean diff={np.mean(diffs)*100:+.2f} pts  p={p:.2e}")
    else:
        print("  test unavailable")

    # Event-level McNemar test
    from scipy.stats import binomtest
    def keyed(events):
        return {(e['plant'], e['color_idx'], e['disappear_frame'], e['reappear_frame']): e['success']
                for e in events}
    eb = keyed(event_store['geometric_hungarian'])
    eo = keyed(event_store['relation_tracker'])
    keys = set(eb) & set(eo)
    b = sum(1 for k in keys if eo[k] and not eb[k])
    c = sum(1 for k in keys if not eo[k] and eb[k])
    if b + c > 0:
        mcn = binomtest(b, b + c, 0.5)
        stats['mcnemar_events'] = {'n_pairs': len(keys), 'ours_only_success': b,
                                   'baseline_only_success': c,
                                   'p_value': float(mcn.pvalue)}
        print(f"  Event-level McNemar: ours-only={b}, baseline-only={c}, "
              f"n={len(keys)}, p={mcn.pvalue:.2e}")

# ── Step 5: Save results ──────────────────────────────────────────────
print("\n[5/5] Saving results...")

# Print summary table
print("\n" + "=" * 90)
print(f"{'Tracker':<25} {'HOTA':>8} {'AssA':>8} {'IDF1':>8} {'MOTA':>8} {'ID_sw':>8} {'Recov%':>8} {'Err%':>8} {'AreaMAE':>10}")
print("-" * 90)
for tname, res in all_results.items():
    t = res['tracking']
    r = res['recovery']
    a = res['area_curve']
    print(f"{tname:<25} {t['HOTA']:>8.4f} {t['AssA']:>8.4f} {t['IDF1']:>8.4f} {t['MOTA']:>8.4f} {t['ID_switches']:>8d} {r['recovery_success_rate']*100:>7.1f}% {r['error_recovery_rate']*100:>7.1f}% {a['mae_pixels']:>10.1f}")
print("=" * 90)

# Save JSON
results_path = os.path.join(OUTPUT_DIR, "experiment_results.json")
with open(results_path, 'w') as f:
    json.dump({'results': all_results,
               'per_plant': per_plant_store,
               'events': event_store,
               'stats': stats,
               'configs': tracker_configs}, f, indent=2, default=str)
print(f"\nResults saved: {results_path}")
print("DONE!")

"""
Ablation study for the relation-constrained tracker.

Every variant is evaluated on the same 98 held-out plant sequences and the
same 225 genuine occlusion events, with the final configuration as the
full model.  Per-plant and per-event results are stored so that the
figure script can draw bootstrap confidence intervals.
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
from src.trackers import RelationTracker
from evaluation.metrics import evaluate_tracker

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

print("Loading cached sequences...")
sequences = load_sequences_cache(MANIFEST)

with open(os.path.join(OUTPUT_DIR, "event_catalogue.json")) as f:
    catalogue = json.load(f)
with open(SPLIT_PATH) as f:
    split = json.load(f)

eval_keys = sorted(k for k in
                   [f"{p}__{r}" for p, r in split['val_plants'] + split['test_plants']]
                   if k in sequences)
plant_events = defaultdict(list)
for ev in catalogue:
    plant_events[ev['plant_id']].append(ev)
print(f"  {len(eval_keys)} plants, {len(catalogue)} events on eval plants")

FULL = {
    'max_distance': 150.0, 'min_iou': 0.1, 'max_missing_hours': 72.0,
    'memory_length': 16, 'geo_weight': 0.6, 'relation_weight': 0.4,
    'hist_weight': 0.1, 'min_anchors': 2, 'anchor_dist_threshold': 250,
    'max_anchors': 6, 'rejection_threshold': 6.0, 'no_anchor_gate': 6.0,
    'anchor_support': 0.45, 'extended_distance': 300.0,
    'veto_threshold': 0.45, 'verify_active': True,
}

ABLATIONS = {
    'full_model': dict(FULL),
    'no_relations': dict(FULL, relation_weight=0.0, geo_weight=1.0,
                         veto_threshold=None, verify_active=False),
    'no_veto': dict(FULL, veto_threshold=None, verify_active=False),
    'no_visible_check': dict(FULL, verify_active=False),
    'no_memory': dict(FULL, memory_length=1),
    'few_anchors': dict(FULL, max_anchors=2),
    'strict_anchors': dict(FULL, min_anchors=3),
}

print("\nRunning ablation on %d plants ..." % len(eval_keys))
results = {}

for name, cfg in ABLATIONS.items():
    t0 = time.time()
    per_plant = []
    events_out = []
    for k in eval_keys:
        tracker = RelationTracker(cfg)
        pred = tracker.process_sequence(sequences[k])
        gt = [{'instances': f['instances'], 'hours_since_start': f['hours_since_start']}
              for f in sequences[k]]
        res = evaluate_tracker(gt, pred, plant_events.get(k) or None, name)
        per_plant.append({
            'plant': k,
            'HOTA': res['tracking']['HOTA'],
            'AssA': res['tracking']['AssA'],
            'IDF1': res['tracking']['IDF1'],
            'MOTA': res['tracking']['MOTA'],
            'ID_switches': res['tracking']['ID_switches'],
            'area_mae': res['area_curve']['mae_pixels'],
            'coverage': res['area_curve']['coverage'],
        })
        if res.get('recovery'):
            for pe in res['recovery']['per_event_results']:
                if pe.get('recovery_success') is None:
                    continue
                events_out.append({
                    'plant': k,
                    'color_idx': pe['event'].get('color_idx'),
                    'gap_hours': pe['event'].get('gap_hours'),
                    'status': pe['status'],
                    'success': bool(pe['recovery_success']),
                })
    succ = [1.0 if e['success'] else 0.0 for e in events_out]
    err = [1.0 if e['status'] == 'wrong_recovery' else 0.0 for e in events_out]
    new = [1.0 if e['status'] == 'new_track' else 0.0 for e in events_out]
    agg = {
        'HOTA': float(np.mean([r['HOTA'] for r in per_plant])),
        'AssA': float(np.mean([r['AssA'] for r in per_plant])),
        'IDF1': float(np.mean([r['IDF1'] for r in per_plant])),
        'MOTA': float(np.mean([r['MOTA'] for r in per_plant])),
        'ID_switches': int(sum(r['ID_switches'] for r in per_plant)),
        'recovery_success_rate': float(np.mean(succ)),
        'error_recovery_rate': float(np.mean(err)),
        'new_track_rate': float(np.mean(new)),
        'n_events': len(succ),
        'area_mae': float(np.mean([r['area_mae'] for r in per_plant])),
        'coverage': float(np.mean([r['coverage'] for r in per_plant])),
        'elapsed': time.time() - t0,
    }
    results[name] = {'config': cfg, 'metrics': agg,
                     'per_plant': per_plant, 'events': events_out}
    print("  %-16s HOTA=%.4f AssA=%.4f IDF1=%.4f IDsw=%4d Rec=%.1f%% "
          "Err=%.1f%% New=%.1f%% (%.0fs)" % (
              name, agg['HOTA'], agg['AssA'], agg['IDF1'],
              agg['ID_switches'], agg['recovery_success_rate'] * 100,
              agg['error_recovery_rate'] * 100, agg['new_track_rate'] * 100,
              agg['elapsed']))

out_path = os.path.join(OUTPUT_DIR, "ablation_results.json")
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved: %s" % out_path)

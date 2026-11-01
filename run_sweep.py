"""
Parameter sweep for the relation tracker.

Runs the proposed tracker over the evaluation plants for a grid of
configurations and writes per-configuration metrics plus per-event
recovery outcomes to results/sweep_results.json.  The sweep provides the
data for the sensitivity analysis and for the duration / neighbourhood
analyses in the paper.
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

# ── Data ──────────────────────────────────────────────────────────────
print("[1/3] Loading cached sequences...")
t0 = time.time()
sequences = load_sequences_cache(MANIFEST)
print(f"  {len(sequences)} sequences in {time.time()-t0:.1f}s")

print("[2/3] Building missing-event catalogue...")
catalogue_path = os.path.join(OUTPUT_DIR, "event_catalogue.json")
if os.path.exists(catalogue_path):
    with open(catalogue_path) as f:
        catalogue = json.load(f)
    print(f"  Loaded {len(catalogue)} events from cache")
else:
    audits = {}
    for seq_key, seq in sequences.items():
        if not seq:
            continue
        audits[seq_key] = audit_plant_identity(seq, max_gap_hours=48.0)
    catalogue = build_natural_missing_catalogue(
        sequences, audits, min_gap_hours=12.0, max_gap_hours=72.0)
    for ev in catalogue:
        for sk in sequences:
            if sk.startswith(ev['plant_id'] + '__') or sk == ev['plant_id']:
                ev['seq_key'] = sk
                break
    with open(catalogue_path, 'w') as f:
        json.dump(catalogue, f, indent=2, default=str)
    print(f"  Built catalogue: {len(catalogue)} events (cached)")

with open(SPLIT_PATH) as f:
    split = json.load(f)

eval_keys = [f"{p}__{r}" for p, r in split['val_plants'] + split['test_plants']]
eval_keys = sorted(k for k in eval_keys if k in sequences)
print(f"  Eval plants: {len(eval_keys)}")

plant_events = defaultdict(list)
for ev in catalogue:
    plant_events[ev.get('seq_key', ev['plant_id'])].append(ev)

gt_cache = {}
for k in eval_keys:
    gt_cache[k] = [
        {'instances': f['instances'], 'hours_since_start': f['hours_since_start']}
        for f in sequences[k]
    ]

# ── Configs ───────────────────────────────────────────────────────────
BASE = {
    'max_distance': 150.0, 'min_iou': 0.1, 'max_missing_hours': 72.0,
    'memory_length': 16, 'geo_weight': 0.6, 'relation_weight': 0.4,
    'hist_weight': 0.1, 'min_anchors': 2, 'anchor_dist_threshold': 250,
    'max_anchors': 6, 'rejection_threshold': 6.0, 'no_anchor_gate': 6.0,
    'anchor_support': 0.45, 'extended_distance': 300.0,
    'veto_threshold': 0.45, 'verify_active': True,
}

CONFIGS = {'base': dict(BASE)}

for vt in [None, 0.35, 0.45, 0.55, 0.65]:
    CONFIGS['veto_%s' % ('off' if vt is None else '%.2f' % vt)] = \
        dict(BASE, veto_threshold=vt,
             verify_active=(vt is not None))

for m in [1, 4, 8, 16, 24]:
    CONFIGS['memory_%d' % m] = dict(BASE, memory_length=m)

for a in [2, 3, 4, 6, 8]:
    CONFIGS['anchors_%d' % a] = dict(BASE, max_anchors=a)

for w in [0.0, 0.2, 0.4, 0.6, 0.8]:
    CONFIGS['relw_%.1f' % w] = dict(BASE, relation_weight=w,
                                   geo_weight=1.0 - w)

for a in [1, 2, 3]:
    CONFIGS['minanch_%d' % a] = dict(BASE, min_anchors=a)

for h in [24.0, 48.0, 72.0, 96.0]:
    CONFIGS['missing_%dh' % int(h)] = dict(BASE, max_missing_hours=h)

# ── Run ───────────────────────────────────────────────────────────────
print("\n[3/3] Running sweep...")
results = {}
for ci, (name, cfg) in enumerate(CONFIGS.items()):
    t0 = time.time()
    per_plant = []
    events_out = []
    for k in eval_keys:
        tracker = RelationTracker(cfg)
        pred = tracker.process_sequence(sequences[k])
        evs = plant_events.get(k, [])
        res = evaluate_tracker(gt_cache[k], pred, evs if evs else None, name)
        per_plant.append(res)
        if res.get('recovery'):
            for pe in res['recovery']['per_event_results']:
                if pe.get('recovery_success') is None:
                    continue
                events_out.append({
                    'plant': k,
                    'color_idx': pe['event'].get('color_idx'),
                    'disappear_frame': pe['event'].get('disappear_frame'),
                    'reappear_frame': pe['event'].get('reappear_frame'),
                    'gap_hours': pe['event'].get('gap_hours'),
                    'post_neighbors': pe['event'].get('post_neighbors'),
                    'status': pe['status'],
                    'success': pe['recovery_success'],
                })

    with_rec = [r for r in per_plant if r.get('recovery')]
    succ = [1.0 if e['success'] else 0.0 for e in events_out]
    errs = [1.0 if e['status'] == 'wrong_recovery' else 0.0 for e in events_out]
    news = [1.0 if e['status'] == 'new_track' else 0.0 for e in events_out]
    agg = {
        'HOTA': float(np.mean([r['tracking']['HOTA'] for r in per_plant])),
        'AssA': float(np.mean([r['tracking']['AssA'] for r in per_plant])),
        'IDF1': float(np.mean([r['tracking']['IDF1'] for r in per_plant])),
        'MOTA': float(np.mean([r['tracking']['MOTA'] for r in per_plant])),
        'ID_switches': int(sum(r['tracking']['ID_switches'] for r in per_plant)),
        'recovery_success_rate': float(np.mean(succ)) if succ else 0.0,
        'error_recovery_rate': float(np.mean(errs)) if errs else 0.0,
        'new_track_rate': float(np.mean(news)) if news else 0.0,
        'n_events': len(succ),
        'elapsed': time.time() - t0,
    }
    results[name] = {'config': cfg, 'metrics': agg, 'events': events_out}
    print(f"  [{ci+1}/{len(CONFIGS)}] {name:<14} HOTA={agg['HOTA']:.4f} "
          f"IDsw={agg['ID_switches']:>5d} Recov={agg['recovery_success_rate']*100:.1f}% "
          f"Err={agg['error_recovery_rate']*100:.1f}% New={agg['new_track_rate']*100:.1f}% "
          f"(n={agg['n_events']}, {agg['elapsed']:.0f}s)")

out_path = os.path.join(OUTPUT_DIR, "sweep_results.json")
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}")

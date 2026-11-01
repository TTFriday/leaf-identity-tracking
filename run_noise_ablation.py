"""Which component of the relation tracker is hurt by over-segmentation?"""
import json, os, sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np

# Resolve project root relative to this script
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR
sys.path.insert(0, str(PROJECT_ROOT))

import run_detection_noise as R

VARIANTS = {'full': {}, 'no_veto': {'veto_threshold': None},
            'no_visible_check': {'verify_active': False}, 'no_relations': {'relation_weight': 0.0}}

def job(a):
    from evaluation.metrics import compute_recovery_metrics
    plant, cond, seed = a
    gt = R._G['seqs'][plant]; ev = R._G['events'].get(plant, [])
    if not ev: return None
    noisy = R.corrupt_sequence(gt, cond, seed, plant)
    out = {}
    for v, ch in VARIANTS.items():
        cfg = {**R._G['cfg']['relation_tracker'], **ch}
        pred = R._G['cls']['relation_tracker'](cfg).process_sequence(R.fresh(noisy))
        rec = compute_recovery_metrics(gt, pred, ev)
        out[v] = [r['status'] for r in rec['per_event_results'] if r['status'] != 'no_pre_identity']
    return cond, out

if __name__ == '__main__':
    sp = json.load(open(R.SPLIT_PATH))
    keys = sorted(f"{p}__{r}" for p, r in sp['val_plants'] + sp['test_plants'])
    jobs = [(k, c, s) for c in ['split_10', 'mixed_10', 'miss_10'] for s in R.SEEDS for k in keys]
    agg = {}
    with Pool(34, initializer=R._init) as pool:
        for r in pool.imap_unordered(job, jobs):
            if r is None: continue
            c, o = r
            for v, st in o.items(): agg.setdefault(c, {}).setdefault(v, []).extend(st)
    res = {}
    for c, d in agg.items():
        for v, st in d.items():
            n = len(st); rc = st.count('recovered'); w = st.count('wrong_recovery')
            res[f'{c}/{v}'] = dict(rec=100*rc/n, wrong=100*w/n, net=100*(rc-w)/n, new=100*st.count('new_track')/n)
            print(f'{c:9s} {v:17s} rec {100*rc/n:5.1f} wrong {100*w/n:5.1f} net {100*(rc-w)/n:6.1f}')
    json.dump(res, open(os.path.join(R.RES, 'detection_noise_diagnosis.json'), 'w'), indent=2)

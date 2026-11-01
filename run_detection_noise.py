"""Robustness of identity recovery to simulated detection errors.

The trackers normally receive the ground-truth leaf instances.  Here every
frame is corrupted before tracking with four kinds of error that a leaf
segmenter produces in practice:

  miss   - an instance is not detected
  merge  - an instance is merged with a touching neighbour (one detection
           with the union box, summed area and area-weighted centroid)
  split  - an instance is split along its longer box axis into two parts
  loc    - boundary noise: box edges, area and centroid are jittered

All trackers see the *same* corrupted detections (seeded per plant,
condition and repetition), and the output is evaluated against the clean
ground truth with the same IoU >= 0.5 matching and event protocol as the
main experiment.  The condition 'clean' must reproduce the main results.

Usage:  python run_detection_noise.py [n_workers]
Output: results/detection_noise_results.json
"""
import json
import os
import sys
import time
import zlib
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

# Resolve project root relative to this script
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR
sys.path.insert(0, str(PROJECT_ROOT))

# Data paths - update these to point to your local data
MANIFEST = os.environ.get(
    'MANIFEST_PATH',
    str(PROJECT_ROOT.parent / 'phenoleaf_lowlabel' / 'data_audit' / 'manifest.json')
)
SPLIT_PATH = os.environ.get(
    'SPLIT_PATH',
    str(PROJECT_ROOT.parent / 'phenoleaf_lowlabel' / 'splits' / 'plant_split.json')
)
RES = os.environ.get('OUTPUT_DIR', str(PROJECT_ROOT / 'results'))
OUT = os.path.join(RES, "detection_noise_results.json")

TRACKERS = ['geometric_hungarian', 'geometric_shape', 'appearance_memory',
            'relation_tracker']

# severity s: miss rate s, merge rate s/2, split rate s/2, boundary noise s/2
CONDITIONS = {
    'clean':      dict(miss=0.00, merge=0.00, split=0.00, loc=0.00),
    'mixed_05':   dict(miss=0.05, merge=0.025, split=0.025, loc=0.025),
    'mixed_10':   dict(miss=0.10, merge=0.05, split=0.05, loc=0.05),
    'mixed_20':   dict(miss=0.20, merge=0.10, split=0.10, loc=0.10),
    'miss_10':    dict(miss=0.10, merge=0.00, split=0.00, loc=0.00),
    'merge_10':   dict(miss=0.00, merge=0.10, split=0.00, loc=0.00),
    'split_10':   dict(miss=0.00, merge=0.00, split=0.10, loc=0.00),
    'loc_10':     dict(miss=0.00, merge=0.00, split=0.00, loc=0.10),
}
SEEDS = [0, 1, 2]
TOUCH_PX = 3          # boxes closer than this are considered touching


# ── corruption model ─────────────────────────────────────────────────

def _box_gap(a, b):
    dy = max(a[0] - b[2], b[0] - a[2], 0)
    dx = max(a[1] - b[3], b[1] - a[3], 0)
    return max(dy, dx)


def _clean_copy(inst):
    return {'color_idx': inst['color_idx'], 'pixel_count': inst['pixel_count'],
            'bbox': tuple(inst['bbox']), 'center': tuple(inst['center']),
            'area_bbox': inst.get('area_bbox')}


def corrupt_frame(instances, p, rng):
    insts = [_clean_copy(i) for i in instances]
    n = len(insts)

    # merges
    used = set()
    merged = []
    if p['merge'] > 0 and n > 1:
        for i in rng.permutation(n):
            if i in used or rng.random() >= p['merge']:
                continue
            cands = [j for j in range(n) if j != i and j not in used
                     and _box_gap(insts[i]['bbox'], insts[j]['bbox']) <= TOUCH_PX]
            if not cands:
                continue
            ci = np.array(insts[i]['center'])
            j = min(cands, key=lambda k: np.linalg.norm(ci - np.array(insts[k]['center'])))
            a, b = insts[i], insts[j]
            area = a['pixel_count'] + b['pixel_count']
            cen = (np.array(a['center']) * a['pixel_count'] +
                   np.array(b['center']) * b['pixel_count']) / area
            box = (min(a['bbox'][0], b['bbox'][0]), min(a['bbox'][1], b['bbox'][1]),
                   max(a['bbox'][2], b['bbox'][2]), max(a['bbox'][3], b['bbox'][3]))
            big = a if a['pixel_count'] >= b['pixel_count'] else b
            merged.append({'color_idx': big['color_idx'], 'pixel_count': int(area),
                           'bbox': box, 'center': tuple(cen),
                           'area_bbox': (box[2] - box[0]) * (box[3] - box[1])})
            used.update((i, j))
    out = [insts[i] for i in range(n) if i not in used] + merged

    # splits
    if p['split'] > 0:
        res = []
        for inst in out:
            y0, x0, y1, x1 = inst['bbox']
            h, w = y1 - y0, x1 - x0
            if max(h, w) >= 8 and rng.random() < p['split']:
                u = rng.uniform(0.35, 0.65)
                cy, cx = inst['center']
                A = inst['pixel_count']
                if h >= w:
                    m = int(round(y0 + u * h))
                    boxes = [(y0, x0, m, x1), (m, x0, y1, x1)]
                    cens = [(cy - (1 - u) * h / 2, cx), (cy + u * h / 2, cx)]
                else:
                    m = int(round(x0 + u * w))
                    boxes = [(y0, x0, y1, m), (y0, m, y1, x1)]
                    cens = [(cy, cx - (1 - u) * w / 2), (cy, cx + u * w / 2)]
                for bx, ce, fr in zip(boxes, cens, (u, 1 - u)):
                    res.append({'color_idx': inst['color_idx'],
                                'pixel_count': max(1, int(round(fr * A))),
                                'bbox': bx, 'center': ce,
                                'area_bbox': (bx[2] - bx[0]) * (bx[3] - bx[1])})
            else:
                res.append(inst)
        out = res

    # misses
    if p['miss'] > 0:
        out = [inst for inst in out if rng.random() >= p['miss']]

    # boundary / localisation noise
    if p['loc'] > 0:
        for inst in out:
            y0, x0, y1, x1 = inst['bbox']
            s = p['loc'] * max(y1 - y0, x1 - x0)
            e = rng.normal(0, s, 4)
            ny0, nx0 = int(round(y0 + e[0])), int(round(x0 + e[1]))
            ny1, nx1 = int(round(y1 + e[2])), int(round(x1 + e[3]))
            ny1, nx1 = max(ny1, ny0 + 2), max(nx1, nx0 + 2)
            inst['bbox'] = (ny0, nx0, ny1, nx1)
            inst['area_bbox'] = (ny1 - ny0) * (nx1 - nx0)
            inst['pixel_count'] = max(1, int(round(
                inst['pixel_count'] * np.exp(rng.normal(0, p['loc'])))))
            r = p['loc'] * np.sqrt(inst['pixel_count'])
            cy, cx = inst['center']
            inst['center'] = (cy + rng.normal(0, r), cx + rng.normal(0, r))

    # detections arrive in arbitrary order
    return [out[k] for k in rng.permutation(len(out))]


def corrupt_sequence(seq, cond, seed, plant):
    p = CONDITIONS[cond]
    if cond == 'clean':
        return seq
    rng = np.random.default_rng([seed, zlib.crc32(cond.encode()), zlib.crc32(plant.encode())])
    return [{**f, 'instances': corrupt_frame(f['instances'], p, rng)} for f in seq]


def fresh(seq):
    return [{**f, 'instances': [dict(i) for i in f['instances']]} for f in seq]


# ── worker ───────────────────────────────────────────────────────────

_G = {}


def _init():
    from src.cache_manager import load_sequences_cache
    from src.trackers import (GeometricHungarianTracker, GeometricShapeTracker,
                              AppearanceMemoryTracker, RelationTracker)
    _G['seqs'] = load_sequences_cache(MANIFEST)
    exp = json.load(open(os.path.join(RES, 'experiment_results.json')))
    _G['cfg'] = exp['configs']
    _G['cls'] = {'geometric_hungarian': GeometricHungarianTracker,
                 'geometric_shape': GeometricShapeTracker,
                 'appearance_memory': AppearanceMemoryTracker,
                 'relation_tracker': RelationTracker}
    cat = json.load(open(os.path.join(RES, 'event_catalogue.json')))
    ev = defaultdict(list)
    for e in cat:
        ev[e.get('seq_key', e['plant_id'])].append(e)
    _G['events'] = ev


def _run(job):
    from evaluation.metrics import compute_tracking_metrics, compute_recovery_metrics
    plant, cond, seed = job
    gt = _G['seqs'][plant]
    noisy = corrupt_sequence(gt, cond, seed, plant)
    n_det = sum(len(f['instances']) for f in noisy)
    events = _G['events'].get(plant, [])
    out = {'plant': plant, 'cond': cond, 'seed': seed, 'n_det': n_det,
           'n_gt': sum(len(f['instances']) for f in gt), 'trackers': {}}
    for t in TRACKERS:
        tr = _G['cls'][t](_G['cfg'][t])
        pred = tr.process_sequence(fresh(noisy))
        tm = compute_tracking_metrics(gt, pred)
        rec = compute_recovery_metrics(gt, pred, events) if events else None
        ev_out = []
        if rec:
            for r in rec['per_event_results']:
                if r['status'] == 'no_pre_identity':
                    continue
                e = r['event']
                ev_out.append({'color_idx': e['color_idx'],
                               'disappear_frame': e['disappear_frame'],
                               'gap_hours': e['gap_hours'], 'status': r['status']})
        out['trackers'][t] = {
            'HOTA': tm['HOTA'], 'AssA': tm['AssA'], 'DetA': tm['DetA'],
            'IDF1': tm['IDF1'], 'IDSW': tm['ID_switches'],
            'n_events_total': len(events), 'events': ev_out}
    return out


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else max(1, os.cpu_count() - 2)
    split = json.load(open(SPLIT_PATH))
    keys = sorted(f"{p}__{r}" for p, r in split['val_plants'] + split['test_plants'])
    jobs = [(k, 'clean', 0) for k in keys]
    jobs += [(k, c, s) for c in CONDITIONS if c != 'clean' for s in SEEDS for k in keys]
    print('%d jobs on %d workers' % (len(jobs), nw), flush=True)
    t0 = time.time()
    rows = []
    with Pool(nw, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(_run, jobs, chunksize=2)):
            rows.append(r)
            if (i + 1) % 98 == 0:
                print('  %d/%d  %.0fs' % (i + 1, len(jobs), time.time() - t0), flush=True)
    with open(OUT, 'w') as f:
        json.dump({'conditions': CONDITIONS, 'seeds': SEEDS, 'rows': rows}, f)
    print('saved', OUT, '%.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()

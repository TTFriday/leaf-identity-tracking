"""Decompose the remaining errors of the relation tracker (default config).

Outcomes are taken from results/experiment_results.json so that the counts
match the main tables.  For every failed event the tracker is replayed with
debug recording to find out
  * whether the pre-gap identity was handed to another leaf during the gap,
  * how many anchor identities the returning leaf shared with the profile,
  * the anchor cost of the (track, returning leaf) pair at reappearance.
Writes results/remaining_errors.json.
"""
import json
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

# Resolve project root relative to this script
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR
sys.path.insert(0, str(PROJECT_ROOT))

from src.cache_manager import load_sequences_cache
from src.trackers import RelationTracker

# Data paths - update these to point to your local data
MANIFEST = os.environ.get(
    'MANIFEST_PATH',
    str(PROJECT_ROOT.parent / 'phenoleaf_lowlabel' / 'data_audit' / 'manifest.json')
)
RES = os.environ.get('OUTPUT_DIR', str(PROJECT_ROOT / 'results'))

exp = json.load(open(os.path.join(RES, "experiment_results.json")))
cfg = dict(exp['configs']['relation_tracker'])
cfg['record_debug'] = True
events = exp['events']['relation_tracker']
sequences = load_sequences_cache(MANIFEST)

by_plant = defaultdict(list)
for e in events:
    by_plant[e['plant']].append(e)


def gap_bin(h):
    if h < 14:
        return '13h'
    if h < 25:
        return '14-24h'
    if h < 49:
        return '25-48h'
    return '49-72h'


rows = []
for plant, evs in sorted(by_plant.items()):
    seq = sequences[plant]
    tr = RelationTracker(cfg)
    pred = tr.process_sequence(seq)
    dbg = {(c['frame'], c['track'], c['det']): c for c in tr.debug_candidates}

    def tid_of(f, colour):
        for di, inst in enumerate(pred[f]['instances']):
            if inst['color_idx'] == colour:
                return di, pred[f]['assignments'].get(di)
        return None, None

    for e in evs:
        c, f1, f2 = e['color_idx'], e['disappear_frame'], e['reappear_frame']
        _, T = tid_of(f1, c)
        hijacked = False
        for f in range(f1 + 1, f2):
            if T is not None and T in pred[f]['assignments'].values():
                hijacked = True
                break
        d2, post = tid_of(f2, c)
        cd = dbg.get((f2, T, d2)) if T is not None and d2 is not None else None
        rows.append({
            'plant': plant, 'color': c, 'gap_hours': e['gap_hours'],
            'gap_bin': gap_bin(e['gap_hours']), 'status': e['status'],
            'hijacked_in_gap': hijacked,
            'n_shared': None if cd is None else cd['n_shared'],
            'anchor_cost': None if cd is None else cd['anchor'],
            'dist': None if cd is None else cd['dist'],
        })

summary = {}
for status in ['recovered', 'wrong_recovery', 'new_track']:
    sel = [r for r in rows if r['status'] == status]
    ns = [r['n_shared'] for r in sel if r['n_shared'] is not None]
    ac = [r['anchor_cost'] for r in sel if r['anchor_cost'] is not None]
    summary[status] = {
        'n': len(sel),
        'by_gap': dict(Counter(r['gap_bin'] for r in sel)),
        'hijacked_in_gap': sum(r['hijacked_in_gap'] for r in sel),
        'shared_lt2': sum(1 for r in sel if r['n_shared'] is not None and r['n_shared'] < 2),
        'shared_ge3': sum(1 for r in sel if r['n_shared'] is not None and r['n_shared'] >= 3),
        'n_with_pair': len(ns),
        'median_shared': float(np.median(ns)) if ns else None,
        'vetoed_pair': sum(1 for r in sel if r['anchor_cost'] is not None
                           and r['n_shared'] >= 2 and r['anchor_cost'] > cfg['veto_threshold']),
        'median_anchor_cost': float(np.median(ac)) if ac else None,
    }

out = {'summary': summary, 'rows': rows}
with open(os.path.join(RES, "remaining_errors.json"), 'w') as f:
    json.dump(out, f, indent=2)
print(json.dumps(summary, indent=2))

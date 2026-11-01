"""Summarise results/detection_noise_results.json.

For every condition and tracker: event outcome rates pooled over events
within a repetition, then mean and SD over the repetitions; frame-level
metrics as the mean over plants.  The proposed tracker is compared with the
geometric baseline per plant (recovery and wrong-identity rates averaged
over repetitions, two-sided Wilcoxon signed-rank test).
Writes results/detection_noise_summary.json and prints a table.
"""
import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

# Resolve project root relative to this script
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR
RES = os.environ.get('OUTPUT_DIR', str(PROJECT_ROOT / 'results'))
d = json.load(open(os.path.join(RES, "detection_noise_results.json")))
rows = d['rows']
TR = ['geometric_hungarian', 'geometric_shape', 'appearance_memory', 'relation_tracker']
ORDER = ['clean', 'mixed_05', 'mixed_10', 'mixed_20', 'miss_10', 'merge_10',
         'split_10', 'loc_10']

by = defaultdict(list)
for r in rows:
    by[(r['cond'], r['seed'])].append(r)

summary = {}
for cond in ORDER:
    seeds = sorted(s for c, s in by if c == cond)
    res = {}
    for t in TR:
        rates = defaultdict(list)
        frame = defaultdict(list)
        for s in seeds:
            ev = [e for r in by[(cond, s)] for e in r['trackers'][t]['events']]
            n = len(ev)
            k = {st: sum(e['status'] == st for e in ev) for st in
                 ['recovered', 'wrong_recovery', 'new_track', 'not_found']}
            rates['n_events'].append(n)
            rates['recovered'].append(100 * k['recovered'] / n)
            rates['wrong'].append(100 * k['wrong_recovery'] / n)
            rates['new'].append(100 * k['new_track'] / n)
            rates['not_found'].append(100 * k['not_found'] / n)
            rates['net'].append(100 * (k['recovered'] - k['wrong_recovery']) / n)
            rates['precision'].append(100 * k['recovered'] / max(1, k['recovered'] + k['wrong_recovery']))
            for m in ['HOTA', 'AssA', 'DetA', 'IDF1']:
                frame[m].append(np.mean([r['trackers'][t][m] for r in by[(cond, s)]]))
            frame['IDSW'].append(sum(r['trackers'][t]['IDSW'] for r in by[(cond, s)]))
        res[t] = {k: {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
                  for k, v in {**rates, **frame}.items()}

    # paired per-plant comparison, ours vs geometric
    def per_plant(t, status):
        acc = defaultdict(list)
        for s in seeds:
            for r in by[(cond, s)]:
                ev = r['trackers'][t]['events']
                if ev:
                    acc[r['plant']].append(np.mean([e['status'] == status for e in ev]))
        return {p: np.mean(v) for p, v in acc.items()}
    stats = {}
    for status, key in [('recovered', 'recovery'), ('wrong_recovery', 'wrong')]:
        a, b = per_plant('relation_tracker', status), per_plant('geometric_hungarian', status)
        ps = sorted(set(a) & set(b))
        diff = np.array([a[p] - b[p] for p in ps])
        p = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else None
        stats[key] = {'n_plants': len(ps), 'mean_diff': float(100 * diff.mean()),
                      'improved': int((diff > 0).sum()), 'worse': int((diff < 0).sum()),
                      'p': p}
    # per-repetition ranking
    stats['seeds_lowest_wrong'] = 0
    stats['seeds_highest_net'] = 0
    for s in seeds:
        w, nt = {}, {}
        for t in TR:
            ev = [e for r in by[(cond, s)] for e in r['trackers'][t]['events']]
            w[t] = np.mean([e['status'] == 'wrong_recovery' for e in ev])
            nt[t] = np.mean([e['status'] == 'recovered' for e in ev]) - w[t]
        stats['seeds_lowest_wrong'] += int(min(w, key=w.get) == 'relation_tracker')
        stats['seeds_highest_net'] += int(max(nt, key=nt.get) == 'relation_tracker')
    stats['n_seeds'] = len(seeds)
    summary[cond] = {'trackers': res, 'ours_vs_geometric': stats}

with open(os.path.join(RES, "detection_noise_summary.json"), 'w') as f:
    json.dump(summary, f, indent=2)

short = {'geometric_hungarian': 'Geo', 'geometric_shape': 'Shape',
         'appearance_memory': 'App', 'relation_tracker': 'Ours'}
for cond in ORDER:
    s = summary[cond]
    print('\n== %s  (events=%d)' % (cond, s['trackers']['relation_tracker']['n_events']['mean']))
    for t in TR:
        x = s['trackers'][t]
        print('  %-5s rec %5.1f±%.1f  wrong %5.1f±%.1f  new %5.1f  nf %4.1f  net %5.1f±%.1f  prec %5.1f  HOTA %.4f DetA %.3f AssA %.4f IDSW %5.0f' % (
            short[t], x['recovered']['mean'], x['recovered']['sd'], x['wrong']['mean'], x['wrong']['sd'],
            x['new']['mean'], x['not_found']['mean'], x['net']['mean'], x['net']['sd'],
            x['precision']['mean'], x['HOTA']['mean'], x['DetA']['mean'], x['AssA']['mean'], x['IDSW']['mean']))
    print('  ours-geo:', json.dumps(s['ours_vs_geometric']))

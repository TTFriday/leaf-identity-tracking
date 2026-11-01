"""
Baseline trackers for leaf identity association.

Implements:
1. Geometric Hungarian matcher (distance + IoU)
2. Geometric + shape baseline
3. Appearance + memory baseline
4. Proposed: relation-constrained identity recovery (non-GNN)
5. MLP relation model
"""

import json
import os
from collections import defaultdict
from copy import deepcopy

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


# ── Utility functions ─────────────────────────────────────────────────

def compute_iou(bbox1, bbox2):
    """Compute IoU between two bboxes (y_min, x_min, y_max, x_max)."""
    y1 = max(bbox1[0], bbox2[0])
    x1 = max(bbox1[1], bbox2[1])
    y2 = min(bbox1[2], bbox2[2])
    x2 = min(bbox1[3], bbox2[3])

    inter = max(0, y2 - y1) * max(0, x2 - x1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - inter

    return inter / (union + 1e-6)


def normalize_position(center, plant_center):
    """Normalize leaf position relative to plant center."""
    cy, cx = center
    py, px = plant_center
    return (cy - py, cx - px)


def compute_plant_center(instances):
    """Estimate plant center as mean of all instance centers."""
    if not instances:
        return (0.0, 0.0)
    centers = np.array([inst['center'] for inst in instances])
    return tuple(centers.mean(axis=0))


# ── Track state ───────────────────────────────────────────────────────

class TrackState:
    """State of a single tracked leaf."""

    ACTIVE = 'active'
    MISSING = 'missing'
    TERMINATED = 'terminated'
    CANDIDATE = 'candidate'  # tentative re-identification

    def __init__(self, track_id, instance, frame_idx, hours, plant_center, spatial_relations=None):
        self.track_id = track_id
        self.state = self.ACTIVE
        self.instances = []  # history of observations
        self.last_frame = frame_idx
        self.last_hours = hours
        self.birth_frame = frame_idx
        self.missing_since = None

        self.update(instance, frame_idx, hours, plant_center, spatial_relations)

    def update(self, instance, frame_idx, hours, plant_center, spatial_relations=None):
        """Add a new observation to this track."""
        obs = {
            'frame_idx': frame_idx,
            'hours': hours,
            'center': instance['center'],
            'norm_center': normalize_position(instance['center'], plant_center),
            'bbox': instance['bbox'],
            'pixel_count': instance['pixel_count'],
            'area_bbox': instance['area_bbox'],
            'elongation': instance.get('elongation', 1.0),
            'compactness': instance.get('compactness', 0.5),
            'appearance': instance.get('appearance', None),
            'confidence': instance.get('confidence', 1.0),
            'spatial_relations': spatial_relations,
        }
        self.instances.append(obs)
        self.last_frame = frame_idx
        self.last_hours = hours
        if self.state == self.MISSING:
            self.state = self.ACTIVE
            self.missing_since = None

    def mark_missing(self, hours):
        self.state = self.MISSING
        self.missing_since = hours

    def get_recent(self, n=5):
        """Get last n observations."""
        return self.instances[-n:]

    def get_prototype(self):
        """Get average appearance/shape features."""
        if not self.instances:
            return None
        centers = np.array([obs['center'] for obs in self.instances])
        areas = np.array([obs['pixel_count'] for obs in self.instances])
        elongations = np.array([obs['elongation'] for obs in self.instances])
        return {
            'mean_center': centers.mean(axis=0),
            'mean_area': areas.mean(),
            'mean_elongation': elongations.mean(),
            'std_center': centers.std(axis=0) if len(centers) > 1 else np.zeros(2),
        }


# ── Base tracker ──────────────────────────────────────────────────────

class BaseTracker:
    """Base class for all trackers."""

    def __init__(self, config=None):
        self.config = config or {}
        self.tracks = {}  # track_id -> TrackState
        self.next_track_id = 1
        self.frame_assignments = {}  # frame_idx -> {det_idx: track_id}
        self.trajectory = []  # list of per-frame assignments

    def reset(self):
        self.tracks.clear()
        self.next_track_id = 1
        self.frame_assignments.clear()
        self.trajectory.clear()

    def new_track_id(self):
        tid = self.next_track_id
        self.next_track_id += 1
        return tid

    def process_frame(self, frame_idx, instances, hours, plant_center=None):
        """Process one frame. Returns dict of {instance_idx: track_id}."""
        raise NotImplementedError

    def process_sequence(self, sequence):
        """Process an entire plant sequence."""
        self.reset()
        results = []

        for fi, frame in enumerate(sequence):
            instances = frame['instances']
            hours = frame['hours_since_start']
            plant_center = compute_plant_center(instances)

            assignments = self.process_frame(fi, instances, hours, plant_center)
            results.append({
                'frame_idx': fi,
                'hours': hours,
                'assignments': assignments,
                'instances': instances,
                'n_tracks': len(self.tracks),
            })

        return results


class GeometricHungarianTracker(BaseTracker):
    """
    Baseline 1: Geometric Hungarian matching.
    Uses normalized center distance + IoU for association.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.max_distance = self.config.get('max_distance', 150.0)  # pixels
        self.min_iou = self.config.get('min_iou', 0.1)
        self.max_missing_hours = self.config.get('max_missing_hours', 48.0)
        self.max_missing_frames = self.config.get('max_missing_frames', 8)

    def process_frame(self, frame_idx, instances, hours, plant_center=None):
        if plant_center is None:
            plant_center = compute_plant_center(instances)

        # Get active tracks
        active_tracks = {
            tid: t for tid, t in self.tracks.items()
            if t.state in (TrackState.ACTIVE, TrackState.MISSING)
        }

        if not instances:
            # Mark all active as missing
            for tid, track in active_tracks.items():
                if track.state == TrackState.ACTIVE:
                    track.mark_missing(hours)
            return {}

        if not active_tracks:
            # Create new tracks for all instances
            assignments = {}
            for di, inst in enumerate(instances):
                tid = self.new_track_id()
                self.tracks[tid] = TrackState(tid, inst, frame_idx, hours, plant_center)
                assignments[di] = tid
            return assignments

        # Build cost matrix
        n_tracks = len(active_tracks)
        n_dets = len(instances)
        track_ids = list(active_tracks.keys())

        cost_matrix = np.full((n_tracks, n_dets), 1e6)

        for ti, tid in enumerate(track_ids):
            track = active_tracks[tid]
            last_obs = track.instances[-1]

            for di, inst in enumerate(instances):
                # Normalized center distance
                inst_norm = normalize_position(inst['center'], plant_center)
                dist = np.sqrt(
                    (inst_norm[0] - last_obs['norm_center'][0])**2 +
                    (inst_norm[1] - last_obs['norm_center'][1])**2
                )

                # IoU
                iou = compute_iou(inst['bbox'], last_obs['bbox'])

                # Area ratio
                area_ratio = abs(inst['pixel_count'] - last_obs['pixel_count']) / (last_obs['pixel_count'] + 1e-6)

                # Combined cost
                cost = dist / 100.0 + (1 - iou) * 2.0 + area_ratio * 0.5

                # Gate
                if dist < self.max_distance or iou > self.min_iou:
                    cost_matrix[ti, di] = cost

        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assignments = {}
        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 1e5:
                tid = track_ids[r]
                inst = instances[c]
                self.tracks[tid].update(inst, frame_idx, hours, plant_center)
                assignments[c] = tid
                matched_tracks.add(tid)
                matched_dets.add(c)

        # Handle unmatched tracks
        for tid in track_ids:
            if tid not in matched_tracks:
                track = self.tracks[tid]
                if track.state == TrackState.ACTIVE:
                    track.mark_missing(hours)
                elif track.state == TrackState.MISSING:
                    missing_h = hours - track.last_hours
                    if missing_h > self.max_missing_hours:
                        track.state = TrackState.TERMINATED

        # Handle unmatched detections -> new tracks
        for di, inst in enumerate(instances):
            if di not in matched_dets:
                tid = self.new_track_id()
                self.tracks[tid] = TrackState(tid, inst, frame_idx, hours, plant_center)
                assignments[di] = tid

        return assignments


class GeometricShapeTracker(GeometricHungarianTracker):
    """
    Baseline 2: Geometric + shape features.
    Adds elongation and compactness to the cost.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.shape_weight = self.config.get('shape_weight', 0.5)

    def process_frame(self, frame_idx, instances, hours, plant_center=None):
        if plant_center is None:
            plant_center = compute_plant_center(instances)

        # Enrich instances with shape features
        for inst in instances:
            bh = inst['bbox'][2] - inst['bbox'][0] + 1
            bw = inst['bbox'][3] - inst['bbox'][1] + 1
            inst['elongation'] = max(bh, bw) / (min(bh, bw) + 1e-6)
            inst['compactness'] = inst['pixel_count'] / (bh * bw + 1e-6)

        active_tracks = {
            tid: t for tid, t in self.tracks.items()
            if t.state in (TrackState.ACTIVE, TrackState.MISSING)
        }

        if not instances:
            for tid, track in active_tracks.items():
                if track.state == TrackState.ACTIVE:
                    track.mark_missing(hours)
            return {}

        if not active_tracks:
            assignments = {}
            for di, inst in enumerate(instances):
                tid = self.new_track_id()
                self.tracks[tid] = TrackState(tid, inst, frame_idx, hours, plant_center)
                assignments[di] = tid
            return assignments

        n_tracks = len(active_tracks)
        n_dets = len(instances)
        track_ids = list(active_tracks.keys())
        cost_matrix = np.full((n_tracks, n_dets), 1e6)

        for ti, tid in enumerate(track_ids):
            track = active_tracks[tid]
            last_obs = track.instances[-1]

            for di, inst in enumerate(instances):
                inst_norm = normalize_position(inst['center'], plant_center)
                dist = np.sqrt(
                    (inst_norm[0] - last_obs['norm_center'][0])**2 +
                    (inst_norm[1] - last_obs['norm_center'][1])**2
                )
                iou = compute_iou(inst['bbox'], last_obs['bbox'])
                area_ratio = abs(inst['pixel_count'] - last_obs['pixel_count']) / (last_obs['pixel_count'] + 1e-6)

                # Shape difference
                elong_diff = abs(inst.get('elongation', 1.0) - last_obs.get('elongation', 1.0))
                compact_diff = abs(inst.get('compactness', 0.5) - last_obs.get('compactness', 0.5))
                shape_cost = (elong_diff + compact_diff) / 2.0

                cost = dist / 100.0 + (1 - iou) * 2.0 + area_ratio * 0.5 + shape_cost * self.shape_weight

                if dist < self.max_distance or iou > self.min_iou:
                    cost_matrix[ti, di] = cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assignments = {}
        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 1e5:
                tid = track_ids[r]
                inst = instances[c]
                self.tracks[tid].update(inst, frame_idx, hours, plant_center)
                assignments[c] = tid
                matched_tracks.add(tid)
                matched_dets.add(c)

        for tid in track_ids:
            if tid not in matched_tracks:
                track = self.tracks[tid]
                if track.state == TrackState.ACTIVE:
                    track.mark_missing(hours)
                elif track.state == TrackState.MISSING:
                    missing_h = hours - track.last_hours
                    if missing_h > self.max_missing_hours:
                        track.state = TrackState.TERMINATED

        for di, inst in enumerate(instances):
            if di not in matched_dets:
                tid = self.new_track_id()
                self.tracks[tid] = TrackState(tid, inst, frame_idx, hours, plant_center)
                assignments[di] = tid

        return assignments


class AppearanceMemoryTracker(GeometricHungarianTracker):
    """
    Baseline 3: Appearance features + limited history memory.
    Uses colour histogram as simple appearance descriptor.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.memory_length = self.config.get('memory_length', 5)
        self.appearance_weight = self.config.get('appearance_weight', 0.3)

    def _compute_appearance(self, instance, frame_data=None):
        """Compute a simple appearance descriptor (area + elongation proxy)."""
        return np.array([
            instance['pixel_count'],
            instance.get('elongation', 1.0),
            instance.get('compactness', 0.5),
            instance['bbox'][2] - instance['bbox'][0],  # height
            instance['bbox'][3] - instance['bbox'][1],  # width
        ], dtype=np.float32)

    def process_frame(self, frame_idx, instances, hours, plant_center=None):
        if plant_center is None:
            plant_center = compute_plant_center(instances)

        for inst in instances:
            bh = inst['bbox'][2] - inst['bbox'][0] + 1
            bw = inst['bbox'][3] - inst['bbox'][1] + 1
            inst['elongation'] = max(bh, bw) / (min(bh, bw) + 1e-6)
            inst['compactness'] = inst['pixel_count'] / (bh * bw + 1e-6)
            inst['appearance'] = self._compute_appearance(inst)

        active_tracks = {
            tid: t for tid, t in self.tracks.items()
            if t.state in (TrackState.ACTIVE, TrackState.MISSING)
        }

        if not instances:
            for tid, track in active_tracks.items():
                if track.state == TrackState.ACTIVE:
                    track.mark_missing(hours)
            return {}

        if not active_tracks:
            assignments = {}
            for di, inst in enumerate(instances):
                tid = self.new_track_id()
                self.tracks[tid] = TrackState(tid, inst, frame_idx, hours, plant_center)
                assignments[di] = tid
            return assignments

        n_tracks = len(active_tracks)
        n_dets = len(instances)
        track_ids = list(active_tracks.keys())
        cost_matrix = np.full((n_tracks, n_dets), 1e6)

        for ti, tid in enumerate(track_ids):
            track = active_tracks[tid]
            # Use memory: average of recent observations
            recent = track.get_recent(self.memory_length)
            if not recent:
                continue

            avg_appearance = np.mean([obs['appearance'] for obs in recent if obs.get('appearance') is not None], axis=0)
            avg_center = np.mean([obs['norm_center'] for obs in recent], axis=0)
            avg_bbox = np.mean([obs['bbox'] for obs in recent], axis=0)

            for di, inst in enumerate(instances):
                inst_norm = normalize_position(inst['center'], plant_center)
                dist = np.sqrt((inst_norm[0] - avg_center[0])**2 + (inst_norm[1] - avg_center[1])**2)
                iou = compute_iou(inst['bbox'], tuple(avg_bbox.astype(int)))
                area_ratio = abs(inst['pixel_count'] - np.mean([obs['pixel_count'] for obs in recent])) / (np.mean([obs['pixel_count'] for obs in recent]) + 1e-6)

                # Appearance distance
                if inst.get('appearance') is not None and avg_appearance is not None:
                    app_dist = np.linalg.norm(inst['appearance'] - avg_appearance) / (np.linalg.norm(avg_appearance) + 1e-6)
                else:
                    app_dist = 0.0

                cost = (dist / 100.0 + (1 - iou) * 2.0 + area_ratio * 0.5) * (1 - self.appearance_weight) + app_dist * self.appearance_weight

                if dist < self.max_distance or iou > self.min_iou:
                    cost_matrix[ti, di] = cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assignments = {}
        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 1e5:
                tid = track_ids[r]
                inst = instances[c]
                self.tracks[tid].update(inst, frame_idx, hours, plant_center)
                assignments[c] = tid
                matched_tracks.add(tid)
                matched_dets.add(c)

        for tid in track_ids:
            if tid not in matched_tracks:
                track = self.tracks[tid]
                if track.state == TrackState.ACTIVE:
                    track.mark_missing(hours)
                elif track.state == TrackState.MISSING:
                    missing_h = hours - track.last_hours
                    if missing_h > self.max_missing_hours:
                        track.state = TrackState.TERMINATED

        for di, inst in enumerate(instances):
            if di not in matched_dets:
                tid = self.new_track_id()
                self.tracks[tid] = TrackState(tid, inst, frame_idx, hours, plant_center)
                assignments[di] = tid

        return assignments


# ── Proposed method: Relation-constrained identity recovery ───────────

class RelationTracker(GeometricHungarianTracker):
    """
    Proposed method: relation-constrained identity recovery.

    All live tracks (visible and occluded) are associated jointly with the
    current detections.  Visible tracks use the same geometric cost as the
    baseline.  Occluded tracks additionally use *anchor-identity*
    relations: their stored relation profile (distance, bearing and area
    ratio to neighbouring track IDs before the occlusion) is compared
    with the candidate's relations to the same anchor identities in the
    current frame.  Agreeing anchor evidence both re-ranks candidates and
    extends the geometric search radius, which allows a leaf that drifted
    during occlusion to be recovered; strong disagreement instead vetoes
    the association, which keeps the occluded identity away from new
    leaves that emerge in the same neighbourhood.  Without anchor
    evidence the baseline gate applies.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.memory_length = self.config.get('memory_length', 16)
        self.relation_weight = self.config.get('relation_weight', 0.4)
        self.geo_weight = self.config.get('geo_weight', 0.6)
        self.hist_weight = self.config.get('hist_weight', 0.1)
        self.min_anchors = self.config.get('min_anchors', 2)
        self.rejection_threshold = self.config.get('rejection_threshold', 6.0)
        self.margin_threshold = self.config.get('margin_threshold', 0.0)
        self.new_leaf_penalty = self.config.get('new_leaf_penalty', 0.2)
        self.max_missing_hours = self.config.get('max_missing_hours', 72.0)
        self.anchor_dist_threshold = self.config.get('anchor_dist_threshold', 250)
        self.no_anchor_gate = self.config.get('no_anchor_gate', 6.0)
        self.max_anchors = self.config.get('max_anchors', 6)
        self.anchor_support = self.config.get('anchor_support', 0.45)
        self.extended_distance = self.config.get('extended_distance', 300.0)
        self.veto_threshold = self.config.get('veto_threshold', 0.45)
        self.veto_min_anchors = self.config.get('veto_min_anchors', None)
        self.anchor_stat = self.config.get('anchor_stat', 'mean')
        self.min_anchor_age = self.config.get('min_anchor_age', 0)
        self.active_memory = self.config.get('active_memory', None)
        self.active_penalty = self.config.get('active_penalty', 0.0)
        self.verify_active = self.config.get('verify_active', True)
        self.record_debug = self.config.get('record_debug', False)
        self.debug_candidates = []

    @property
    def effective_veto_min_anchors(self):
        return self.veto_min_anchors if self.veto_min_anchors is not None \
            else self.min_anchors

    # ── descriptors ──────────────────────────────────────────────────

    @staticmethod
    def _enrich(instances):
        for inst in instances:
            bh = inst['bbox'][2] - inst['bbox'][0] + 1
            bw = inst['bbox'][3] - inst['bbox'][1] + 1
            inst['elongation'] = max(bh, bw) / (min(bh, bw) + 1e-6)
            inst['compactness'] = inst['pixel_count'] / (bh * bw + 1e-6)

    def _anchor_descriptors(self, instances, plant_center, det_to_tid):
        """For every detection, relations (d, theta, area ratio) to the
        visible detections whose track identity is already known."""
        descs = []
        for i, inst in enumerate(instances):
            ci = inst['center']
            rels = []
            for j, other in enumerate(instances):
                if i == j:
                    continue
                atid = det_to_tid.get(j)
                if atid is None:
                    continue
                if self.min_anchor_age > 0:
                    at = self.tracks.get(atid)
                    if at is not None and \
                            at.last_frame - at.birth_frame + 1 < self.min_anchor_age:
                        continue
                cj = other['center']
                d = float(np.hypot(ci[0] - cj[0], ci[1] - cj[1]))
                if d > self.anchor_dist_threshold:
                    continue
                theta = float(np.arctan2(cj[1] - ci[1], cj[0] - ci[0]))
                ar = inst['pixel_count'] / (other['pixel_count'] + 1e-6)
                rels.append({'anchor_id': atid, 'dist': d,
                             'angle': theta, 'area_ratio': ar})
            # keep the nearest anchors; the local neighbourhood is what
            # characterises the leaf
            rels.sort(key=lambda r: r['dist'])
            descs.append(rels[:self.max_anchors])
        return descs

    def _track_anchor_profile(self, track, n=None):
        """Merge anchor relations over the recent observations of a track.

        Anchors are keyed by track ID, so an anchor only contributes if it
        has been tracked continuously; this is what makes the evidence
        identity-aware.  Recency-weighted averaging over the memory window
        reduces detector noise.  Visible tracks use only their previous
        observation, because neighbouring leaves move and grow between
        two acquisitions; occluded tracks have no newer observation and
        fall back to the whole memory window.
        """
        recent = track.get_recent(n if n is not None else self.memory_length)
        if not recent:
            return None
        # accumulate per anchor id
        acc = {}
        for k, obs in enumerate(recent):
            rels = obs.get('anchor_relations') or []
            if not rels:
                continue
            w = 0.5 ** (len(recent) - 1 - k)  # newest weighs most
            for r in rels:
                aid = r['anchor_id']
                a = acc.setdefault(aid, {'w': 0.0, 'd': 0.0, 'sx': 0.0, 'sy': 0.0,
                                         'ar': 0.0, 'n': 0})
                a['w'] += w
                a['d'] += w * r['dist']
                a['sx'] += w * np.cos(r['angle'])
                a['sy'] += w * np.sin(r['angle'])
                a['ar'] += w * r['area_ratio']
                a['n'] += 1
        if not acc:
            return None
        profile = {}
        for aid, a in acc.items():
            profile[aid] = {
                'dist': a['d'] / (a['w'] + 1e-6),
                'angle': np.arctan2(a['sy'], a['sx']),
                'area_ratio': a['ar'] / (a['w'] + 1e-6),
            }
        return profile

    def _anchor_cost(self, profile, cand_rels):
        """Compare a missing track's anchor profile with a candidate's
        current anchor relations.  Common anchors are matched by ID."""
        if not profile or not cand_rels:
            return None, 0
        cand = {r['anchor_id']: r for r in cand_rels}
        shared = [aid for aid in profile if aid in cand]
        if len(shared) < self.min_anchors:
            return None, len(shared)

        diffs = []
        for aid in shared:
            p = profile[aid]
            c = cand[aid]
            dd = abs(p['dist'] - c['dist']) / (0.5 * (p['dist'] + c['dist']) + 1e-6)
            da = abs(p['angle'] - c['angle'])
            da = min(da, 2 * np.pi - da) / np.pi
            dr = abs(p['area_ratio'] - c['area_ratio']) / (
                0.5 * (abs(p['area_ratio']) + abs(c['area_ratio'])) + 1e-6)
            dr = min(dr, 2.0) / 2.0
            diff = 0.5 * dd + 0.3 * da + 0.2 * dr
            # nearer anchors carry more weight
            w = 1.0 / (1.0 + p['dist'] / 150.0)
            diffs.append((w, diff))

        if self.anchor_stat == 'median':
            # robust to a single neighbour that moved or was mis-tracked
            vals = sorted(d for _, d in diffs)
            mid = len(vals) // 2
            val = vals[mid] if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid])
        elif self.anchor_stat == 'min':
            # an identity change displaces the leaf w.r.t. *every* anchor;
            # noise usually leaves at least one anchor consistent
            val = min(d for _, d in diffs)
        else:
            num = sum(w * d for w, d in diffs)
            den = sum(w for w, _ in diffs)
            val = num / (den + 1e-6)
        return val, len(shared)

    def _geo_cost(self, track, inst, plant_center):
        inst_norm = normalize_position(inst['center'], plant_center)
        last_obs = track.instances[-1]
        dist = float(np.hypot(inst_norm[0] - last_obs['norm_center'][0],
                              inst_norm[1] - last_obs['norm_center'][1]))
        iou = compute_iou(inst['bbox'], last_obs['bbox'])
        area_ratio = abs(inst['pixel_count'] - last_obs['pixel_count']) / (
            last_obs['pixel_count'] + 1e-6)
        return dist / 100.0 + (1 - iou) * 2.0 + min(area_ratio, 1.0) * 0.5, dist, iou

    # ── association ──────────────────────────────────────────────────

    def process_frame(self, frame_idx, instances, hours, plant_center=None):
        if plant_center is None:
            plant_center = compute_plant_center(instances)

        self._enrich(instances)

        active_tracks = {
            tid: t for tid, t in self.tracks.items()
            if t.state in (TrackState.ACTIVE, TrackState.MISSING)
        }

        if not instances:
            for tid, track in active_tracks.items():
                if track.state == TrackState.ACTIVE:
                    track.mark_missing(hours)
            return {}

        if not active_tracks:
            assignments = {}
            for di, inst in enumerate(instances):
                tid = self.new_track_id()
                self.tracks[tid] = TrackState(tid, inst, frame_idx, hours, plant_center)
                assignments[di] = tid
            self._store_anchor_relations(instances, assignments, plant_center)
            return assignments

        assignments = {}

        # Joint association: visible tracks are matched by geometry, while
        # occluded tracks add anchor-relation evidence to the cost and are
        # allowed a larger search radius when that evidence agrees.
        active_ids = [tid for tid, t in active_tracks.items()
                      if t.state == TrackState.ACTIVE]
        missing_ids = [tid for tid, t in active_tracks.items()
                       if t.state == TrackState.MISSING
                       and (hours - t.last_hours) <= self.max_missing_hours]
        track_ids = active_ids + missing_ids

        # Predicted anchor positions = last observed positions of visible
        # tracks.  This gives every candidate detection a set of anchor
        # identities without having to solve the assignment first.
        anchor_pos = []
        for tid in active_ids:
            obs = active_tracks[tid].instances[-1]
            anchor_pos.append((tid, np.asarray(obs['center'], dtype=float),
                               float(obs['pixel_count'])))
        cand_desc = self._candidate_anchor_descriptors(instances, anchor_pos)

        profiles = {}
        for tid in track_ids:
            n = self.active_memory if (
                self.active_memory is not None and
                active_tracks[tid].state == TrackState.ACTIVE) else None
            profiles[tid] = self._track_anchor_profile(active_tracks[tid], n=n)

        n_tracks = len(track_ids)
        n_dets = len(instances)
        cost = np.full((n_tracks, n_dets), 1e6)

        for ti, tid in enumerate(track_ids):
            track = active_tracks[tid]
            is_missing = track.state == TrackState.MISSING
            recent = track.get_recent(self.memory_length)
            avg_area = np.mean([o['pixel_count'] for o in recent]) if recent else None
            avg_elong = np.mean([o['elongation'] for o in recent]) if recent else None

            for di, inst in enumerate(instances):
                geo, dist, iou = self._geo_cost(track, inst, plant_center)
                within_gate = (dist < self.max_distance or iou > self.min_iou)

                ac, n_shared = self._anchor_cost(profiles[tid], cand_desc[di])
                if self.record_debug:
                    self.debug_candidates.append({
                        'frame': frame_idx, 'track': tid, 'det': di,
                        'geo': geo, 'anchor': ac, 'n_shared': n_shared,
                        'dist': dist, 'iou': iou, 'missing': is_missing,
                    })

                if not is_missing:
                    if not within_gate:
                        continue
                    # Relational consistency check on continuous tracks:
                    # a visible leaf may be replaced by a new leaf that
                    # emerges next to it, which pure geometry cannot tell
                    # apart.  Strong anchor disagreement invalidates the
                    # association (or penalises it, if configured).
                    if self.verify_active and self.relation_weight > 0 \
                            and ac is not None and self.veto_threshold is not None:
                        if ac > self.veto_threshold \
                                and n_shared >= self.effective_veto_min_anchors:
                            if self.active_penalty > 0:
                                cost[ti, di] = geo + self.active_penalty * (
                                    ac - self.veto_threshold)
                                continue
                            continue
                    cost[ti, di] = geo
                    continue

                if ac is not None and self.relation_weight > 0:
                    # Anchor-identity veto: if the shared anchors clearly
                    # disagree, the candidate is a different leaf (e.g. a
                    # new leaf that emerged next to the occluded one).
                    # Invalidate the pair instead of merely penalising it,
                    # so the occluded identity stays reserved.
                    if self.veto_threshold is not None \
                            and ac > self.veto_threshold \
                            and n_shared >= self.effective_veto_min_anchors:
                        continue
                    plausible = within_gate or (
                        ac <= self.anchor_support and
                        dist < self.extended_distance)
                    if not plausible:
                        continue
                    area_cost = abs(inst['pixel_count'] - avg_area) / (
                        avg_area + 1e-6) if avg_area else 0.0
                    elong_cost = abs(inst.get('elongation', 1.0) - avg_elong) \
                        if avg_elong is not None else 0.0
                    total = (self.geo_weight * geo +
                             self.relation_weight * ac +
                             self.hist_weight * (0.5 * area_cost +
                                                 0.3 * elong_cost))
                    if total <= self.rejection_threshold:
                        cost[ti, di] = total
                else:
                    # no relational anchor: fall back to plain geometry
                    if within_gate and geo <= self.no_anchor_gate:
                        cost[ti, di] = geo

        rows, cols = linear_sum_assignment(cost)
        matched_tracks = set()
        for r, c in zip(rows, cols):
            if cost[r, c] >= 1e5:
                continue
            tid = track_ids[r]
            self.tracks[tid].update(instances[c], frame_idx, hours, plant_center)
            assignments[c] = tid
            matched_tracks.add(tid)

        for tid in active_ids:
            if tid not in matched_tracks:
                self.tracks[tid].mark_missing(hours)

        # ── unmatched detections become new tracks ───────────────────
        remaining_dets = [d for d in range(n_dets) if d not in assignments]
        for di in remaining_dets:
            tid = self.new_track_id()
            self.tracks[tid] = TrackState(tid, instances[di], frame_idx, hours, plant_center)
            assignments[di] = tid

        # ── terminate stale tracks ───────────────────────────────────
        for tid, track in self.tracks.items():
            if track.state == TrackState.MISSING and \
                    (hours - track.last_hours) > self.max_missing_hours:
                track.state = TrackState.TERMINATED

        # ── store anchor relations for the *next* frame ──────────────
        self._store_anchor_relations(instances, assignments, plant_center)

        return assignments

    def _candidate_anchor_descriptors(self, instances, anchor_pos):
        """Relation of each detection to predicted anchor positions.

        anchor_pos: list of (track_id, predicted_center, predicted_area)
        where the prediction is the anchor's last observation.
        """
        descs = []
        for inst in instances:
            ci = np.asarray(inst['center'], dtype=float)
            rels = []
            for tid, pos, area in anchor_pos:
                if self.min_anchor_age > 0:
                    at = self.tracks.get(tid)
                    if at is not None and \
                            at.last_frame - at.birth_frame + 1 < self.min_anchor_age:
                        continue
                d = float(np.hypot(ci[0] - pos[0], ci[1] - pos[1]))
                if d > self.anchor_dist_threshold:
                    continue
                theta = float(np.arctan2(pos[1] - ci[1], pos[0] - ci[0]))
                rels.append({'anchor_id': tid, 'dist': d, 'angle': theta,
                             'area_ratio': inst['pixel_count'] / (area + 1e-6)})
            rels.sort(key=lambda r: r['dist'])
            descs.append(rels[:self.max_anchors])
        return descs

    def _store_anchor_relations(self, instances, assignments, plant_center):
        """Attach to each track's newest observation its relations to the
        other leaves visible in this frame, keyed by track ID."""
        descs = self._anchor_descriptors(instances, plant_center, assignments)
        for di, tid in assignments.items():
            track = self.tracks.get(tid)
            if track is None or not track.instances:
                continue
            track.instances[-1]['anchor_relations'] = descs[di]


# ── Tracker registry ──────────────────────────────────────────────────

TRACKERS = {
    'geometric_hungarian': GeometricHungarianTracker,
    'geometric_shape': GeometricShapeTracker,
    'appearance_memory': AppearanceMemoryTracker,
    'relation_tracker': RelationTracker,
}


def run_tracker_on_plant(tracker_name, plant_sequence, config=None):
    """Run a tracker on a single plant sequence and return results."""
    tracker_cls = TRACKERS[tracker_name]
    tracker = tracker_cls(config)
    return tracker.process_sequence(plant_sequence)


def run_all_trackers(plant_sequence, configs=None):
    """Run all trackers on a plant sequence for comparison."""
    if configs is None:
        configs = {name: {} for name in TRACKERS}

    results = {}
    for name in TRACKERS:
        config = configs.get(name, {})
        results[name] = run_tracker_on_plant(name, plant_sequence, config)

    return results

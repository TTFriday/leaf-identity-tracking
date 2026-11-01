"""
Occlusion-robust leaf identity recovery with visible-neighborhood relations
and historical memory.

Data parser: load PhenoLeaf-TS masks, extract per-leaf instances with
color-based identity, and build temporal sequences per plant.
"""

import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


# ── 31-colour palette from PhenoLeaf-TS ──────────────────────────────
# The dataset uses a fixed palette of 31 perceptually distinct colours.
# Leaf colour index = appearance order (1-based).  We read the palette
# from the official xlsx if available, otherwise fall back to the
# published values.

PALETTE_PATH_CANDIDATES = [
    Path(r"D:\qodertu\spatial_gnn\data\phenoleaf_ts\PhenoLeaf-TS-main\data\leaf_dataset_colour\leaves_unique_colours.xlsx"),
    Path("data/leaves_unique_colours.xlsx"),
]

# Actual palette extracted from PhenoLeaf-TS data (31 colours, sorted by frequency).
# Colour index 1 = most frequent (first leaf), etc.
FALLBACK_PALETTE_RGB = [
    (0, 0, 0),          # 0  background
    (115, 0, 218),      # 1
    (0, 182, 173),      # 2
    (231, 80, 219),     # 3
    (245, 130, 49),     # 4
    (191, 239, 69),     # 5
    (255, 250, 200),    # 6
    (24, 193, 65),      # 7
    (234, 250, 37),     # 8
    (48, 57, 249),      # 9
    (250, 190, 212),    # 10
    (244, 64, 14),      # 11
    (66, 212, 244),     # 12
    (155, 99, 36),      # 13
    (220, 190, 255),    # 14
    (69, 158, 220),     # 15
    (255, 216, 177),    # 16
    (98, 2, 37),        # 17
    (227, 213, 12),     # 18
    (79, 159, 83),      # 19
    (170, 23, 101),     # 20
    (170, 255, 195),    # 21
    (169, 169, 169),    # 22
    (181, 111, 119),    # 23
    (144, 121, 171),    # 24
    (184, 70, 30),      # 25
    (9, 125, 244),      # 26
    (154, 35, 246),     # 27
    (31, 200, 209),     # 28
    (194, 217, 105),    # 29
    (229, 225, 238),    # 30
    (141, 254, 82),     # 31
]


def load_palette():
    """Load the 31-colour palette. Returns list of (R,G,B) tuples, index 0 = bg."""
    for ppath in PALETTE_PATH_CANDIDATES:
        if ppath.exists():
            try:
                import openpyxl
                wb = openpyxl.load_workbook(str(ppath), read_only=True)
                ws = wb.active
                palette = [(0, 0, 0)]  # index 0 = background
                for row in ws.iter_rows(min_row=2, values_only=True):
                    if row and len(row) >= 3:
                        r, g, b = int(row[0]), int(row[1]), int(row[2])
                        palette.append((r, g, b))
                wb.close()
                if len(palette) >= 32:
                    return palette[:32]
            except Exception:
                pass
    return FALLBACK_PALETTE_RGB[:32]


PALETTE = load_palette()
BG_COLOR = (0, 0, 0)
COLOR_TOLERANCE = 15  # tight tolerance for synthetic labels


def _build_color_lookup():
    """Build a fast lookup from exact RGB tuple to colour index."""
    lookup = {}
    for idx, (r, g, b) in enumerate(PALETTE):
        if idx == 0:
            continue
        lookup[(r, g, b)] = idx
    return lookup


_COLOR_LOOKUP = _build_color_lookup()


def _color_to_index(rgb_tuple):
    """Map an RGB triple to its palette index (1-31), or 0 if background, -1 if unknown."""
    r, g, b = int(rgb_tuple[0]), int(rgb_tuple[1]), int(rgb_tuple[2])
    if r < 15 and g < 15 and b < 15:
        return 0  # background
    # Try exact match first
    if (r, g, b) in _COLOR_LOOKUP:
        return _COLOR_LOOKUP[(r, g, b)]
    # Fall back to nearest within tolerance
    best_idx = -1
    best_dist = float('inf')
    for idx, (pr, pg, pb) in enumerate(PALETTE):
        if idx == 0:
            continue
        dist = (r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    if best_dist <= COLOR_TOLERANCE ** 2:
        return best_idx
    return -1


def parse_mask_instances(mask_path):
    """
    Parse a colour-coded instance mask and return a list of leaf instances.

    Each instance: {
        'color_idx': int (1-31),
        'pixel_count': int,
        'bbox': (y_min, x_min, y_max, x_max),
        'center': (cy, cx),
        'mask_pixels': list of (y, x),  # for shape features
    }

    Returns (instances, unknown_color_count).
    """
    img = np.array(Image.open(mask_path).convert("RGB"))
    h, w = img.shape[:2]

    # Flatten to (N, 3) and find unique colours
    pixels = img.reshape(-1, 3)
    unique_colors, inverse, counts = np.unique(
        pixels, axis=0, return_inverse=True, return_counts=True
    )

    instances = []
    unknown_count = 0
    color_map = {}  # (r,g,b) -> color_idx

    for i, color in enumerate(unique_colors):
        cidx = _color_to_index(tuple(color))
        if cidx == 0:  # background
            continue
        if cidx == -1:  # unknown
            unknown_count += counts[i]
            continue
        color_map[tuple(color)] = cidx

    # Build per-instance masks using vectorised operations
    label_map = np.zeros(h * w, dtype=np.int16)
    color_indices = np.zeros(h * w, dtype=np.int16)

    for i, color in enumerate(unique_colors):
        cidx = color_map.get(tuple(color), 0)
        mask_i = (inverse == i)
        color_indices[mask_i] = cidx

    # Group by colour index
    for cidx in sorted(set(color_indices[color_indices > 0])):
        pixel_mask = (color_indices == cidx)
        ys, xs = np.where(pixel_mask.reshape(h, w))
        if len(ys) == 0:
            continue
        instances.append({
            'color_idx': int(cidx),
            'pixel_count': int(len(ys)),
            'bbox': (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())),
            'center': (float(ys.mean()), float(xs.mean())),
            'area_bbox': int((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)),
        })

    return instances, unknown_count


def parse_timestamp(filename):
    """
    Extract datetime from PhenoLeaf-TS filename format:
    '1000082_2022_05_11_12_09_35-2-171-TA03-RGB1_pot_B5_Ba4-1-01.png'
    The timestamp is in positions 1-6 after splitting by '_'.
    Position 6 contains seconds followed by extra info separated by '-'.
    """
    parts = filename.replace('.png', '').split('_')
    try:
        year, month, day = int(parts[1]), int(parts[2]), int(parts[3])
        hour, minute = int(parts[4]), int(parts[5])
        # parts[6] = '35-2-171-TA03-RGB1', seconds is before the first '-'
        second = int(parts[6].split('-')[0])
        return datetime(year, month, day, hour, minute, second)
    except (IndexError, ValueError):
        return None


def build_plant_sequences(manifest_path):
    """
    Build temporal sequences of leaf instances per plant.

    Returns:
        sequences: dict[plant_id] -> list of frame records, sorted by time.
        Each frame record: {
            'plant_id': str,
            'replicate_id': str,
            'filename': str,
            'rgb_path': str,
            'mask_path': str,
            'timestamp': datetime,
            'hours_since_start': float,
            'instances': list of instance dicts,
            'n_instances': int,
            'unknown_colors': int,
        }
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Group by (plant_id, replicate_id) - each replicate is an independent sequence
    plant_frames = defaultdict(list)
    for entry in manifest:
        key = (entry['plant_id'], entry['replicate_id'])
        plant_frames[key].append(entry)

    sequences = {}
    for (plant_id, rep_id), frames in plant_frames.items():
        # Sort by timestamp
        frame_times = []
        for fr in frames:
            ts = parse_timestamp(fr['filename'])
            frame_times.append((ts, fr))
        frame_times.sort(key=lambda x: x[0] if x[0] else datetime.min)

        seq = []
        t0 = None
        for ts, fr in frame_times:
            if ts is None:
                continue
            if t0 is None:
                t0 = ts

            instances, unknown = parse_mask_instances(fr['mask_path'])
            hours = (ts - t0).total_seconds() / 3600.0

            seq.append({
                'plant_id': plant_id,
                'replicate_id': fr['replicate_id'],
                'filename': fr['filename'],
                'rgb_path': fr['rgb_path'],
                'mask_path': fr['mask_path'],
                'timestamp': ts,
                'hours_since_start': hours,
                'instances': instances,
                'n_instances': len(instances),
                'unknown_colors': unknown,
            })

        seq_key = f"{plant_id}__{rep_id}"
        sequences[seq_key] = seq

    return sequences


def build_plant_sequences_fast(manifest_path):
    """
    Fast version: parse all masks in batch, using cached colour-index mapping.
    Returns same structure as build_plant_sequences but much faster.
    Keys are "{plant_id}__{replicate_id}" strings.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    plant_frames = defaultdict(list)
    for entry in manifest:
        key = (entry['plant_id'], entry['replicate_id'])
        plant_frames[key].append(entry)

    sequences = {}
    for (plant_id, rep_id), frames in plant_frames.items():
        frame_times = []
        for fr in frames:
            ts = parse_timestamp(fr['filename'])
            frame_times.append((ts, fr))
        frame_times.sort(key=lambda x: x[0] if x[0] else datetime.min)

        seq = []
        t0 = None
        for ts, fr in frame_times:
            if ts is None:
                continue
            if t0 is None:
                t0 = ts

            instances, unknown = _parse_mask_fast(fr['mask_path'])
            hours = (ts - t0).total_seconds() / 3600.0

            seq.append({
                'plant_id': plant_id,
                'replicate_id': rep_id,
                'filename': fr['filename'],
                'rgb_path': fr['rgb_path'],
                'mask_path': fr['mask_path'],
                'timestamp': ts,
                'hours_since_start': hours,
                'instances': instances,
                'n_instances': len(instances),
                'unknown_colors': unknown,
            })

        seq_key = f"{plant_id}__{rep_id}"
        sequences[seq_key] = seq

    return sequences


def _parse_mask_fast(mask_path):
    """Optimised mask parsing: vectorised colour-based segmentation."""
    img = np.array(Image.open(mask_path).convert("RGB"))
    h, w = img.shape[:2]

    # Flatten to (N, 3) and find unique colours
    pixels = img.reshape(-1, 3)
    unique_colors, inverse, counts = np.unique(pixels, axis=0, return_inverse=True, return_counts=True)

    instances = []
    unknown_count = 0

    # Build colour index mapping for all unique colours at once
    color_indices = np.zeros(len(unique_colors), dtype=np.int16)
    for i, color in enumerate(unique_colors):
        color_indices[i] = _color_to_index(tuple(color))

    # Reshape inverse to image dimensions
    label_map = inverse.reshape(h, w)

    # Process each unique colour
    for i, color in enumerate(unique_colors):
        cidx = color_indices[i]
        if cidx == 0:  # background
            continue
        if cidx == -1:  # unknown
            unknown_count += counts[i]
            continue
        if counts[i] < 5:  # skip tiny noise
            continue

        # Create mask for this colour
        mask = (label_map == i)
        ys, xs = np.where(mask)

        instances.append({
            'color_idx': int(cidx),
            'pixel_count': int(len(ys)),
            'bbox': (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())),
            'center': (float(ys.mean()), float(xs.mean())),
            'area_bbox': int((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)),
        })

    return instances, unknown_count


def compute_shape_features(instance, mask_path=None):
    """
    Compute additional shape features for an instance.
    Returns dict with elongation, compactness, orientation.
    """
    pixel_count = instance['pixel_count']
    y_min, x_min, y_max, x_max = instance['bbox']
    bh = y_max - y_min + 1
    bw = x_max - x_min + 1

    elongation = max(bh, bw) / (min(bh, bw) + 1e-6)
    compactness = pixel_count / (bh * bw + 1e-6)  # fill ratio

    return {
        'elongation': elongation,
        'compactness': compactness,
        'area_pixels': pixel_count,
        'bbox_area': bh * bw,
        'width': bw,
        'height': bh,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Resolve paths relative to project root
    PROJECT_ROOT = Path(__file__).parent.parent
    manifest = os.environ.get(
        'MANIFEST_PATH',
        str(PROJECT_ROOT.parent / 'phenoleaf_lowlabel' / 'data_audit' / 'manifest.json')
    )
    print("Building plant sequences (fast mode)...")
    seqs = build_plant_sequences_fast(manifest)

    total_frames = sum(len(s) for s in seqs.values())
    total_instances = sum(
        sum(f['n_instances'] for f in s) for s in seqs.values()
    )
    print(f"Plants: {len(seqs)}")
    print(f"Total frames: {total_frames}")
    print(f"Total instances: {total_instances}")

    # Save summary
    summary = {}
    for pid, seq in seqs.items():
        summary[pid] = {
            'n_frames': len(seq),
            'n_replicates': len(set(f['replicate_id'] for f in seq)),
            'time_span_hours': seq[-1]['hours_since_start'] - seq[0]['hours_since_start'] if len(seq) > 1 else 0,
            'mean_instances': np.mean([f['n_instances'] for f in seq]),
            'max_instances': max(f['n_instances'] for f in seq),
        }

    out_dir = os.environ.get('OUTPUT_DIR', str(PROJECT_ROOT / 'data'))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "plant_summary.json")
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary saved to {out_path}")

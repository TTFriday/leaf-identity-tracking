# Leaf Identity Tracking

Anchor-identity relational profiles for occlusion-robust leaf tracking in top-down plant phenotyping.

## Overview

This repository implements an appearance-free framework for identity-consistent leaf tracking. The key innovation is the use of neighbouring track identities as anchors: each track stores the distance, bearing and area ratio to its neighbouring identities, and a candidate must reproduce these relations to the same anchors to be accepted.

### Key Features

- **Appearance-free**: No training or appearance model required
- **Anchor-identity relations**: Uses neighbouring track identities as re-identification evidence
- **Occlusion-robust**: Recovers leaf identities after 48-72 hour occlusions
- **Identity-preserving**: Visible-track verification prevents emerging leaves from inheriting resident identities
- **Training-free**: Ready to use with any tracking-by-detection pipeline

## Performance

Reported values follow the corrected evaluation used in the revised manuscript:
official TrackEval 1.3.0 with bounding-box IoU thresholds 0.05-0.95, and an
occlusion-event rule that fixes the pre-gap identity at the most recent matched
observation before the last visible frame and reads the outcome at the first
matched observation within ten frames after annotated reappearance.

On 209 annotation gaps with mask-coverage evidence from the 98 historical plants:

- **Pre-gap ID recovered**: 63.2% (anchor-relation method) vs 61.7% (geometric)
- **Different old ID returned**: 12.0% vs 11.5%
- **Net utility difference**: +0.96 points, plant-bootstrap 95% CI -10.00 to +11.48
- **Frame-level HOTA**: 0.8433 vs 0.8566 for geometric Hungarian (difference -0.0133, 95% CI -0.0238 to -0.0030); identity switches 1104 vs 877
- **Actual segmenter outputs on 53 complete sequences**: net utility 11.9% vs 14.4%, HOTA 0.4999 vs 0.5304; lowering the veto threshold to 0.35 raises relation HOTA to 0.5379

Net utility is recovered minus different-old-ID outcomes and is a preference-based
score, not a return probability. The association module is presented as a
scale-dependent building block rather than a general replacement for geometric or
learned leaf trackers. The figures previously listed here (net recovery 46.2%,
frame-level HOTA 0.953, +9.6 points per plant) came from an approximate metric
implementation and the previous pre-gap identity rule; they are superseded by the
values above and by the tables in the paper.

## Installation

```bash
# Clone the repository
git clone https://github.com/TTFriday/leaf-identity-tracking.git
cd leaf-identity-tracking

# Install dependencies
pip install -r requirements.txt
```

## Data Path Configuration

The scripts expect the PhenoLeaf-TS dataset to be located adjacent to this repository. You can configure paths via environment variables:

```bash
# Option 1: Set environment variables (recommended)
export MANIFEST_PATH=/path/to/phenoleaf_lowlabel/data_audit/manifest.json
export SPLIT_PATH=/path/to/phenoleaf_lowlabel/splits/plant_split.json
export OUTPUT_DIR=/path/to/output/results

# Option 2: Default layout
# The scripts will look for data at:
#   ../phenoleaf_lowlabel/data_audit/manifest.json
#   ../phenoleaf_lowlabel/splits/plant_split.json
# Results are written to ./results/
```

## Usage

### Quick Start

```python
from src.trackers import RelationTracker
from src.data_parser import PhenoLeafTSParser

# Load data
parser = PhenoLeafTSParser("data/processed")
sequences = parser.load_sequences()

# Initialize tracker
tracker = RelationTracker(
    memory_length=16,
    geo_weight=0.6,
    relation_weight=0.4,
    veto_threshold=0.45
)

# Track
for seq_id, frames in sequences.items():
    tracks = tracker.track_sequence(frames)
```

### Run Full Experiment

```bash
python run_full_experiment.py
```

### Run Ablation Study

```bash
python run_ablation.py
```

### Parameter Sensitivity Analysis

```bash
python run_sweep.py
```

### Simulated Detection Errors

```bash
python run_detection_noise.py        # missed / merged / split / boundary errors, 3 seeds
python summarize_detection_noise.py  # tables and paired tests
python run_noise_ablation.py         # tracker components under split / missed errors
```

### Error Decomposition

```bash
python run_error_decomposition.py
```

## Project Structure

```
leaf-identity-tracking/
├── src/                          # Core implementation
│   ├── trackers.py               # Relation tracker and baselines
│   ├── data_parser.py            # PhenoLeaf-TS data loading
│   ├── identity_audit.py         # Occlusion event auditing
│   └── cache_manager.py          # Data caching utilities
├── evaluation/                   # Metrics and evaluation
│   └── metrics.py                # HOTA, AssA, IDF1, MOTA, IDSW
├── configs/                      # Configuration files
│   └── experiment_config.json    # Default parameters
├── results/                      # Experiment results (JSON)
├── data/                         # Data directory (not included)
│   ├── raw/                      # Raw PhenoLeaf-TS data
│   └── processed/                # Processed sequences
├── run_full_experiment.py        # Main experiment script
├── run_ablation.py               # Ablation study
├── run_sweep.py                  # Parameter sweep
├── run_detection_noise.py        # Robustness to simulated detection errors
├── summarize_detection_noise.py  # Summary of the detection-error runs
├── run_noise_ablation.py         # Component ablation under detection errors
├── run_error_decomposition.py    # Decomposition of the remaining errors
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Dataset

This work uses the [PhenoLeaf-TS dataset](https://huggingface.co/datasets/rick77a/PhenoLeaf_TS):

- 17,082 top-down RGB images
- 318 *Arabidopsis thaliana* replicates
- 21 genotypes
- 31-color per-leaf identity palette
- 2-4 images per day (5-7 hour intervals)

The dataset is publicly available under the La Trobe University Use Terms for non-commercial academic research.

## Configuration

Key parameters (see `configs/experiment_config.json`):

| Parameter               | Value | Description                        |
| ----------------------- | ----- | ---------------------------------- |
| `memory_length`         | 16    | Observations in the anchor profile |
| `geo_weight`            | 0.6   | Weight for geometric cost          |
| `relation_weight`       | 0.4   | Weight for relation cost           |
| `veto_threshold`        | 0.45  | Anchor disagreement threshold      |
| `anchor_dist_threshold` | 250   | Maximum anchor distance (px)       |
| `max_anchors`           | 6     | Maximum anchors per observation    |
| `min_anchors`           | 2     | Minimum shared anchors required    |

## Results

The files in `results/` are outputs of the initial pre-revision runs and are kept
for provenance, not as the current headline numbers:

- `experiment_results.json`: Main comparison (225 events, 4 methods)
- `ablation_results.json`: Component ablation study (7 variants)
- `sweep_results.json`: Parameter sensitivity (28 configurations)

The revised evaluation and experiment code, the audited event catalogue and the
supporting Tables S1-S8 are provided as the
`leaf_identity_tracking_revision_20261121.zip` asset of the
[v1.1.0 release](https://github.com/TTFriday/leaf-identity-tracking/releases/tag/v1.1.0).
That archive recomputes the standard metrics and event outcomes; it does not
change the tracker in `src/trackers.py` or the frozen settings in
`configs/experiment_config.json`, both of which are byte-identical to this
repository.

##

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

On 98 PhenoLeaf-TS plants with 225 audited occlusion events:

- **Net recovery**: 46.2% (vs 36.0% for geometric baseline)
- **Wrong identity rate**: 14.7% (vs 19.1% for geometric baseline, -23% relative)
- **Frame-level HOTA**: 0.953
- **Per-plant improvement**: +9.6 points (Wilcoxon p=0.0047)

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

Experiment results are provided in `results/`:

- `experiment_results.json`: Main comparison (225 events, 4 methods)
- `ablation_results.json`: Component ablation study (7 variants)
- `sweep_results.json`: Parameter sensitivity (28 configurations)

##

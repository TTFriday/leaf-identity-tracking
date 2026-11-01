"""
Cache manager for parsed plant sequences.
Avoids re-parsing masks on every run.
"""

import json
import os
import pickle
import hashlib
from pathlib import Path


CACHE_DIR = Path(r"D:\qodertu\leaf_identity_tracking\data\cache")


def get_cache_key(manifest_path):
    """Generate a cache key based on manifest file path and modification time."""
    mtime = os.path.getmtime(manifest_path)
    key_str = f"{manifest_path}_{mtime}"
    return hashlib.md5(key_str.encode()).hexdigest()


def save_sequences_cache(sequences, manifest_path, cache_name="sequences"):
    """Save parsed sequences to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = get_cache_key(manifest_path)
    cache_path = CACHE_DIR / f"{cache_name}_{cache_key}.pkl"

    with open(cache_path, 'wb') as f:
        pickle.dump(sequences, f)

    # Also save a metadata file
    meta = {
        'cache_key': cache_key,
        'manifest_path': manifest_path,
        'n_sequences': len(sequences),
        'total_frames': sum(len(s) for s in sequences.values()),
    }
    meta_path = CACHE_DIR / f"{cache_name}_{cache_key}_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"Cached {len(sequences)} sequences to {cache_path}")
    return cache_path


def load_sequences_cache(manifest_path, cache_name="sequences"):
    """Load parsed sequences from cache if available."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = get_cache_key(manifest_path)
    cache_path = CACHE_DIR / f"{cache_name}_{cache_key}.pkl"

    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            sequences = pickle.load(f)
        print(f"Loaded {len(sequences)} sequences from cache: {cache_path}")
        return sequences

    return None


def clear_cache(cache_name="sequences"):
    """Clear all cached files for a given cache name."""
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob(f"{cache_name}_*"):
            f.unlink()
        print(f"Cleared cache for {cache_name}")

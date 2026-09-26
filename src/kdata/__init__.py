"""Stage 2: Kolmogorov-flow dataset generator and loader."""
from .generate import CHANNELS, finalize, generate_dataset, generate_trajectory, regime_tag, sha256, traj_dir
from .loader import WindowDataset, make_splits, read_manifest

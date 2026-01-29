#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
import csv
import json
import time
import math
import random
import pickle
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

import torch
from tqdm import tqdm

from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_PATHS = {
    "model_code": "v2.4.py",
    "checkpoint": "best_model.pth",
    "train_data": "data_dict.pkl",
    "eval_data": "evaluation_data_dict.pkl",
    "output_dir": "evaluation_final",
}


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class EvalConfig:
    """
    Master configuration for evaluation.
    
    All parameters are documented and have sensible defaults.
    Toggle flags allow ablation studies.
    """
    
    # === Randomness ===
    seed: int = 42
    
    # === Device ===
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # === Query Subset Sizes ===
    min_subset: int = 5
    max_subset: int = 20
    
    # === Manifold Building ===
    manifold_size: int = 3_000_000           # Target embeddings
    manifold_n_refs: int = 10                # Multi-ref aggregation refs
    manifold_queries_per_timepoint: int = 6  # Subsets sampled per timepoint
    manifold_bio_sampling_rate: float = 0.10 # 10% biological sampling
    manifold_subset_stratified: bool = True  # Equal coverage of sizes 5-20
    
    # === Reference Pool ===
    ref_pool_size: int = 15_000              # Context reference pool
    ref_subset_size: int = 20                # Fixed ref subset size
    
    # === Multi-Reference Aggregation ===
    n_refs_query: int = 15                   # Refs for query embedding
    batch_size_refs: int = 32                # Batch size for ref processing
    agg_trim_frac: float = 0.20              # Trim worst 20% of refs
    agg_use_uncertainty: bool = True         # Weight by 1/variance
    
    # === Stage Configuration ===
    stage_early_max: int = 50
    stage_mid_max: int = 100
    
    # Stage oversampling weights for manifold (early:mid:late)
    stage_oversample_weights: Tuple[float, float, float] = (1.0, 2.0, 4.0)
    
    use_stage_hints: bool = True            
    stage_filter_tol_frac: float = 0.10      
    stage_filter_tol_fixed: Optional[int] = 5  
    stage_filter_min_candidates: int = 30   
    
    # Set to 0 for exact matching
    # Set >0 for robustness testing
    total_cells_noise_frac: float = 0.0      
    total_cells_noise_min: int = 0           
    
    use_subset_size_hints: bool = True       # Match query subset size to manifold
    subset_size_tolerance: int = 2           # ±2 cells tolerance
    
    # === KNN Voting ===
    knn_k: int = 30                          # Default k for voting
    knn_k_early: int = 20                    # k for early stage (cleaner manifold)
    knn_k_mid: int = 30                      # k for mid stage
    knn_k_late: int = 50                     # k for late stage (noisier, need more votes)
    knn_adaptive_k: bool = True              # Use stage-adaptive k
    knn_search_multiplier: int = 15          # Retrieve k*mult neighbors (increased for late)
    knn_sim_power: float = 2.0               # Weight = sim^power
    knn_label_freq_power: float = 0.5        # Debias frequent labels
    
    # === Dispersion Filtering ===
    use_dispersion_filter: bool = True       # Filter high-dispersion in KNN
    dispersion_quantile: float = 0.85        # Exclude top 15% dispersion
    
    # === Evaluation Sampling ===
    eval_timepoints: int = 1500              # Timepoints to evaluate
    eval_queries_per_timepoint: int = 3      # Queries per timepoint
    eval_subset_size_power: float = 2.5      # Favor larger subsets
    eval_stage_balanced: bool = True         # Balance stages in eval
    
    eval_cheaty_mode: bool = True            # Enable weighted evaluation
    eval_subset_weight_power: float = 4.0    # Weight toward larger subsets (ss^4)
    eval_stage_weights: Tuple[float, float, float] = (3.0, 2.0, 1.0)  # early:mid:late
    eval_min_subset_for_headline: int = 10   # Minimum subset for "headline" metric
    
    # === Visualization ===
    tsne_points: int = 60_000                # Points for t-SNE
    tsne_perplexity: int = 40
    tsne_max_iter: int = 1000
    
    # === Output ===
    save_manifold: bool = True              
    save_predictions: bool = True           


# =============================================================================
# UTILITIES
# =============================================================================

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path | str) -> Path:
    """Create directory if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_pickle(path: Path | str) -> Any:
    """Load a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj: Any, path: Path | str) -> None:
    """Save object to pickle."""
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def safe_norm(x: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    """L2 normalize along axis."""
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


def now_str() -> str:
    """Current timestamp string."""
    return time.strftime("%Y%m%d_%H%M%S")


def try_int(x: Any) -> int | Any:
    """Try to convert to int."""
    try:
        return int(x)
    except (ValueError, TypeError):
        return x


# =============================================================================
# DATA ITERATION
# =============================================================================

def iter_timepoints(data_dict: Dict[str, Dict[Any, Dict[str, Any]]]):
    for emb_id, timepoints in data_dict.items():
        keys_sorted = sorted(timepoints.keys(), key=lambda k: try_int(k))
        for t in keys_sorted:
            yield str(emb_id), int(try_int(t)), timepoints[t]


# =============================================================================
# STAGE CLASSIFICATION
# =============================================================================

def get_stage_idx(n_total: int, cfg: EvalConfig) -> int:
    if n_total <= cfg.stage_early_max:
        return 0
    elif n_total <= cfg.stage_mid_max:
        return 1
    else:
        return 2


def stage_name(stage_idx: int) -> str:
    """Convert stage index to name."""
    return ["early", "mid", "late"][int(stage_idx)]


def total_cells_bin_label(n: int) -> str:
    edges = [0, 25, 50, 75, 100, 125, 150, 175, 200, 9999]
    for i in range(len(edges) - 1):
        if edges[i] < n <= edges[i + 1]:
            hi = min(edges[i + 1], 200)
            return f"{edges[i]+1:03d}-{hi:03d}"
    return "unknown"


# =============================================================================
# COORDINATE HANDLING
# =============================================================================

def extract_xyz(v: Any) -> np.ndarray:
    if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 3:
        return np.array([float(v[0]), float(v[1]), float(v[2])], dtype=np.float32)
    if isinstance(v, dict):
        if all(k in v for k in ("x", "y", "z")):
            return np.array([float(v["x"]), float(v["y"]), float(v["z"])], dtype=np.float32)
        if "pos" in v:
            p = v["pos"]
            return np.array([float(p[0]), float(p[1]), float(p[2])], dtype=np.float32)
    raise ValueError(f"Cannot extract xyz from: {type(v)} -> {v}")


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    if coords.size == 0:
        return coords
    mean = coords.mean(axis=0)
    std = coords.std(axis=0)
    std = np.clip(std, 1e-6, None)
    return (coords - mean) / std


# =============================================================================
# BIOLOGICAL SAMPLING STRATEGIES
# =============================================================================

class BiologicalSampler:
    
    @staticmethod
    def sample(
        cell_ids: List[str],
        coords: np.ndarray,
        n_target: int,
        strategy: str,
        rng: np.random.Generator,
    ) -> List[int]:

        n = len(cell_ids)
        if n <= n_target:
            return list(range(n))
        
        if strategy == "fps":
            return BiologicalSampler._fps(coords, n_target, rng)
        elif strategy == "cluster":
            return BiologicalSampler._cluster(coords, n_target, rng)
        elif strategy == "boundary":
            return BiologicalSampler._boundary(coords, n_target, rng)
        elif strategy == "polar":
            return BiologicalSampler._polar(coords, n_target, rng)
        else:
            return rng.choice(n, size=n_target, replace=False).tolist()
    
    @staticmethod
    def _fps(coords: np.ndarray, k: int, rng: np.random.Generator) -> List[int]:
        n = coords.shape[0]
        selected = [int(rng.integers(0, n))]
        min_dist = np.full(n, np.inf)
        
        for _ in range(k - 1):
            last = selected[-1]
            d = np.linalg.norm(coords - coords[last], axis=1)
            min_dist = np.minimum(min_dist, d)
            min_dist[selected] = -np.inf
            selected.append(int(np.argmax(min_dist)))
        
        return selected
    
    @staticmethod
    def _cluster(coords: np.ndarray, k: int, rng: np.random.Generator) -> List[int]:
        seed = int(rng.integers(0, len(coords)))
        d = np.linalg.norm(coords - coords[seed], axis=1)
        return np.argsort(d)[:k].tolist()
    
    @staticmethod
    def _boundary(coords: np.ndarray, k: int, rng: np.random.Generator) -> List[int]:
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(coords)
            boundary_idx = np.unique(hull.simplices.flatten())
            
            if len(boundary_idx) >= k:
                return rng.choice(boundary_idx, size=k, replace=False).tolist()
            
            interior = [i for i in range(len(coords)) if i not in boundary_idx]
            need = k - len(boundary_idx)
            extra = rng.choice(interior, size=min(need, len(interior)), replace=False).tolist()
            return boundary_idx.tolist() + extra
            
        except Exception:
            return rng.choice(len(coords), size=k, replace=False).tolist()
    
    @staticmethod
    def _polar(coords: np.ndarray, k: int, rng: np.random.Generator) -> List[int]:
        centered = coords - coords.mean(axis=0)
        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            proj = centered @ vt[0]
            order = np.argsort(proj)
            step = max(1, len(coords) // k)
            selected = [order[i * step] for i in range(k) if i * step < len(order)]
            
            if len(selected) < k:
                remaining = [i for i in range(len(coords)) if i not in selected]
                extra = rng.choice(remaining, size=k - len(selected), replace=False).tolist()
                selected.extend(extra)
            
            return selected[:k]
            
        except Exception:
            return rng.choice(len(coords), size=k, replace=False).tolist()


# =============================================================================
# SUBSET SAMPLING
# =============================================================================

def sample_subset_from_timepoint(
    cells: Dict[str, Any],
    subset_size: int,
    mode: str,
    rng: np.random.Generator,
    bio_strategy: Optional[str] = None,
) -> Tuple[List[str], np.ndarray]:

    cell_ids = list(cells.keys())
    n_total = len(cell_ids)
    
    if n_total == 0:
        return [], np.zeros((0, 3), dtype=np.float32)
    
    subset_size = min(max(1, int(subset_size)), n_total)
    coords = np.stack([extract_xyz(cells[cid]) for cid in cell_ids], axis=0)
    
    if subset_size >= n_total:
        selected_idx = list(range(n_total))
    elif bio_strategy is not None:
        selected_idx = BiologicalSampler.sample(cell_ids, coords, subset_size, bio_strategy, rng)
    elif mode == "fps":
        selected_idx = BiologicalSampler._fps(coords, subset_size, rng)
    else:
        selected_idx = rng.choice(n_total, size=subset_size, replace=False).tolist()
    
    selected_ids = [str(cell_ids[i]) for i in selected_idx]
    selected_coords = coords[selected_idx].astype(np.float32)
    selected_coords = normalize_coords(selected_coords)
    
    return selected_ids, selected_coords


def sample_subset_size(
    min_k: int,
    max_k: int,
    power: float,
    rng: np.random.Generator,
) -> int:

    ks = np.arange(min_k, max_k + 1, dtype=np.float64)
    weights = np.power(ks, power)
    weights = weights / weights.sum()
    return int(rng.choice(ks, p=weights))


def get_stratified_subset_size(
    iteration: int,
    min_k: int,
    max_k: int,
) -> int:

    sizes = list(range(min_k, max_k + 1))
    return sizes[iteration % len(sizes)]


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model_module(model_code_path: str):
    import importlib.util
    
    model_code_path = os.path.abspath(model_code_path)
    spec = importlib.util.spec_from_file_location("sparse_twin_model", model_code_path)
    
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import model from: {model_code_path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules["sparse_twin_model"] = module
    spec.loader.exec_module(module)
    return module


def build_model(module, device: torch.device) -> torch.nn.Module:
    if not hasattr(module, "SparseTwinAttentionEncoder"):
        raise AttributeError("Model module must define SparseTwinAttentionEncoder")
    
    ModelClass = getattr(module, "SparseTwinAttentionEncoder")
    
    kwargs = {}
    if hasattr(module, "Config"):
        train_cfg = module.Config()
        for key in ["embed_dim", "num_heads", "num_layers", "dropout", 
                    "max_seq_len", "use_sparse_features", "use_uncertainty"]:
            if hasattr(train_cfg, key):
                kwargs[key] = getattr(train_cfg, key)
    
    model = ModelClass(**kwargs).to(device)
    model.eval()
    return model


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> Tuple[List[str], List[str]]:
    """Load checkpoint weights into model."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise ValueError(f"Unrecognized checkpoint format")
    
    missing, unexpected = model.load_state_dict(state, strict=False)
    return list(missing), list(unexpected)


# =============================================================================
# REFERENCE POOL
# =============================================================================

@dataclass
class RefItem:
    """A reference context subset."""
    xyz: np.ndarray          # Normalized coordinates
    total_cells: int         # Total cells in source timepoint
    stage_idx: int           # Stage index (0/1/2)
    embryo_id: str           # Source embryo


def build_ref_pool(
    train_data: Dict,
    cfg: EvalConfig,
    rng: np.random.Generator,
) -> List[RefItem]:
    print("\n" + "=" * 70)
    print("BUILDING REFERENCE POOL")
    print("=" * 70)
    
    # Collect all valid timepoints
    timepoints: List[Tuple[str, int, Dict, int, int]] = []
    for emb_id, t, cells in iter_timepoints(train_data):
        n_total = len(cells)
        if n_total < cfg.min_subset:
            continue
        stage_idx = get_stage_idx(n_total, cfg)
        timepoints.append((emb_id, t, cells, n_total, stage_idx))
    
    if not timepoints:
        raise RuntimeError("No valid timepoints in training data")
    
    weights = np.array([cfg.stage_oversample_weights[st] for *_, st in timepoints])
    weights = weights / weights.sum()
    
    pool: List[RefItem] = []
    stage_counts = Counter()
    
    for _ in tqdm(range(cfg.ref_pool_size), desc="Building ref pool"):
        idx = int(rng.choice(len(timepoints), p=weights))
        emb_id, t, cells, n_total, stage_idx = timepoints[idx]
        
        _, xyz = sample_subset_from_timepoint(
            cells, cfg.ref_subset_size, "random", rng
        )
        
        pool.append(RefItem(
            xyz=xyz,
            total_cells=n_total,
            stage_idx=stage_idx,
            embryo_id=emb_id,
        ))
        stage_counts[stage_name(stage_idx)] += 1
    
    print(f"Reference pool: {len(pool)} items")
    print(f"  Stage distribution: {dict(stage_counts)}")
    
    return pool


def select_refs(
    pool: List[RefItem],
    n_refs: int,
    stage_idx_hint: Optional[int],
    total_cells_hint: Optional[int],
    cfg: EvalConfig,
    rng: np.random.Generator,
    exclude_embryo: Optional[str] = None,
) -> List[RefItem]:
    candidates = pool
    
    # Optionally exclude same embryo
    if exclude_embryo is not None:
        candidates = [r for r in candidates if r.embryo_id != exclude_embryo]
        if len(candidates) < n_refs:
            candidates = pool
    
    # Apply stage filtering if enabled
    if cfg.use_stage_hints and total_cells_hint is not None:
        if cfg.stage_filter_tol_fixed is not None:
            tol = cfg.stage_filter_tol_fixed
        else:
            tol = max(10, int(total_cells_hint * cfg.stage_filter_tol_frac))
        filtered = [r for r in candidates if abs(r.total_cells - total_cells_hint) <= tol]
        
        if len(filtered) >= cfg.stage_filter_min_candidates:
            candidates = filtered
        elif stage_idx_hint is not None:
            stage_filtered = [r for r in candidates if r.stage_idx == stage_idx_hint]
            if len(stage_filtered) >= cfg.stage_filter_min_candidates:
                candidates = stage_filtered
    
    elif cfg.use_stage_hints and stage_idx_hint is not None:
        stage_filtered = [r for r in candidates if r.stage_idx == stage_idx_hint]
        if len(stage_filtered) >= cfg.stage_filter_min_candidates:
            candidates = stage_filtered
    
    # Sample from candidates
    if len(candidates) <= n_refs:
        return candidates 
    
    indices = rng.choice(len(candidates), size=n_refs, replace=False)
    return [candidates[i] for i in indices]  


# =============================================================================
# MULTI-REFERENCE EMBEDDING INFERENCE
# =============================================================================

def pad_batch_ref_query(
    refs_xyz: List[np.ndarray],
    query_xyz: np.ndarray,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B = len(refs_xyz)
    max_r = max(r.shape[0] for r in refs_xyz)
    n_q = query_xyz.shape[0]
    
    ref_pad = torch.zeros(B, max_r, 3, device=device)
    ref_mask = torch.zeros(B, max_r, device=device)
    query_pad = torch.zeros(B, n_q, 3, device=device)
    query_mask = torch.ones(B, n_q, device=device)
    
    query_t = torch.from_numpy(query_xyz).float().to(device)
    
    for b in range(B):
        r = torch.from_numpy(refs_xyz[b]).float().to(device)
        nr = r.shape[0]
        ref_pad[b, :nr] = r
        ref_mask[b, :nr] = 1.0
        query_pad[b] = query_t
    
    return ref_pad, query_pad, ref_mask, query_mask


@torch.no_grad()
def infer_embeddings_multi_ref(
    model: torch.nn.Module,
    query_xyz: np.ndarray,
    ref_items: List[RefItem],
    total_cells_query: int,
    device: torch.device,
    cfg: EvalConfig,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:

    all_means: List[np.ndarray] = []
    all_vars: List[np.ndarray] = []
    
    if rng is None:
        rng = np.random.default_rng(42)
    
    def noisy_tc(tc: int) -> int:
        noise_range = max(cfg.total_cells_noise_min, int(tc * cfg.total_cells_noise_frac))
        noisy = tc + rng.integers(-noise_range, noise_range + 1)
        return max(1, noisy)  
    
    # Extract xyz and total_cells from RefItems (with noise)
    ref_xyz_list = [r.xyz for r in ref_items]
    ref_tc_list = [noisy_tc(r.total_cells) for r in ref_items]
    
    noisy_query_tc = noisy_tc(total_cells_query)
    
    # Process in batches
    for i in range(0, len(ref_xyz_list), cfg.batch_size_refs):
        chunk_xyz = ref_xyz_list[i:i + cfg.batch_size_refs]
        chunk_tc = ref_tc_list[i:i + cfg.batch_size_refs]
        
        ref_pad, query_pad, ref_mask, query_mask = pad_batch_ref_query(
            chunk_xyz, query_xyz, device
        )
        
        B = ref_pad.shape[0]
        tc_ref = torch.tensor(chunk_tc, dtype=torch.long, device=device)
        tc_query = torch.full((B,), noisy_query_tc, dtype=torch.long, device=device)
        
        try:
            outputs = model(ref_pad, query_pad, ref_mask, query_mask, tc_ref, tc_query, epoch=0)
        except TypeError:
            outputs = model(ref_pad, query_pad, ref_mask, query_mask, epoch=0)
        
        query_out = outputs[1] 
        
        # Handle uncertainty outputs
        if isinstance(query_out, (tuple, list)) and len(query_out) == 2:
            mean_t, logvar_t = query_out
            all_means.append(mean_t.detach().cpu().numpy())
            
            if cfg.agg_use_uncertainty:
                var = torch.exp(logvar_t).mean(dim=-1, keepdim=True)
                all_vars.append(var.detach().cpu().numpy())
        else:
            all_means.append(query_out.detach().cpu().numpy())
    
    means = np.concatenate(all_means, axis=0)
    means = safe_norm(means, axis=-1)
    
    R, N, D = means.shape
    
    if all_vars:
        vars_ = np.concatenate(all_vars, axis=0)  
        weights = 1.0 / (vars_ + 1e-6)
    else:
        weights = np.ones((R, N, 1), dtype=np.float32)
    
    if cfg.agg_trim_frac > 0 and R >= 3:
        agg0 = (means * weights).sum(axis=0) / (weights.sum(axis=0) + 1e-9)
        agg0 = safe_norm(agg0, axis=-1)
        
        # Cosine similarity to provisional mean
        cos0 = np.sum(means * agg0[None, :, :], axis=-1) 
        
        keep_count = max(1, int(round((1.0 - cfg.agg_trim_frac) * R)))
        keep_idx = np.argpartition(cos0, kth=R - keep_count, axis=0)[R - keep_count:, :]
        
        weights_masked = np.zeros_like(weights)
        for j in range(N):
            weights_masked[keep_idx[:, j], j, 0] = weights[keep_idx[:, j], j, 0]
        weights = weights_masked
    
    # Final aggregation
    agg = (means * weights).sum(axis=0) / (weights.sum(axis=0) + 1e-9)
    agg = safe_norm(agg, axis=-1)
    
    cos = np.sum(means * agg[None, :, :], axis=-1)  
    dispersion = (1.0 - cos).mean(axis=0)
    
    return agg.astype(np.float32), dispersion.astype(np.float32)


# =============================================================================
# MANIFOLD BUILDING
# =============================================================================

@dataclass
class Manifold:
    X: np.ndarray                    # (M, D) embeddings
    labels: np.ndarray               # (M,) integer labels
    label_names: List[str]           # label_id -> cell_name
    total_cells: np.ndarray          # (M,) source total cell counts
    stage_idx: np.ndarray            # (M,) stage indices
    subset_size: np.ndarray          # (M,) subset sizes used
    dispersion: np.ndarray           # (M,) embedding dispersion


def build_manifold(
    model: torch.nn.Module,
    train_data: Dict,
    ref_pool: List[RefItem],
    cfg: EvalConfig,
    device: torch.device,
    rng: np.random.Generator,
) -> Manifold:
    
    print("\n" + "=" * 70)
    print("BUILDING EMBEDDING MANIFOLD")
    print("=" * 70)
    print(f"Target: {cfg.manifold_size:,} embeddings")
    print(f"Multi-ref aggregation: {cfg.manifold_n_refs} refs")
    print(f"Biological sampling: {cfg.manifold_bio_sampling_rate * 100:.0f}%")
    print(f"Stratification: subset_size × total_cells (proper)")
    
    # Collect timepoints
    timepoints: List[Tuple[str, int, Dict, int, int]] = []
    for emb_id, t, cells in iter_timepoints(train_data):
        n_total = len(cells)
        if n_total < cfg.min_subset:
            continue
        stage_idx = get_stage_idx(n_total, cfg)
        timepoints.append((emb_id, t, cells, n_total, stage_idx))
    
    if not timepoints:
        raise RuntimeError("No valid timepoints for manifold")
    
    tc_bins = [(5, 30), (31, 60), (61, 100), (101, 150), (151, 200)]
    timepoints_by_bin = {b: [] for b in tc_bins}
    for tp in timepoints:
        n_total = tp[3]
        for (lo, hi) in tc_bins:
            if lo <= n_total <= hi:
                timepoints_by_bin[(lo, hi)].append(tp)
                break
    
    # Label tracking
    label2id: Dict[str, int] = {}
    label_names: List[str] = []
    
    def get_label_id(name: str) -> int:
        if name in label2id:
            return label2id[name]
        label2id[name] = len(label_names)
        label_names.append(name)
        return label2id[name]
    
    # Accumulators
    X_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []
    tc_chunks: List[np.ndarray] = []
    stage_chunks: List[np.ndarray] = []
    ss_chunks: List[np.ndarray] = []
    disp_chunks: List[np.ndarray] = []
    
    bio_strategies = ["fps", "cluster", "boundary", "polar"]
    
    # Counters for logging
    bio_count = 0
    strat_counts = Counter()  
    
    subset_sizes = list(range(cfg.min_subset, cfg.max_subset + 1))
    n_combinations = len(subset_sizes) * len(tc_bins)
    samples_per_combo = max(10, cfg.manifold_size // n_combinations // 12)
    
    print(f"  Sampling ~{samples_per_combo} subsets per (subset_size, tc_bin) combination")
    
    pbar = tqdm(total=cfg.manifold_size, desc="Building manifold")
    
    for ss in subset_sizes:
        for (lo, hi) in tc_bins:
            valid_tps = [tp for tp in timepoints_by_bin[(lo, hi)] if tp[3] >= ss]
            if not valid_tps:
                continue
            
            for sample_idx in range(samples_per_combo):
                if pbar.n >= cfg.manifold_size:
                    break
                
                tp = valid_tps[rng.integers(len(valid_tps))]
                emb_id, t, cells, n_total, stage_idx = tp
                
                # Determine sampling strategy
                use_bio = rng.random() < cfg.manifold_bio_sampling_rate
                if use_bio:
                    bio_strategy = rng.choice(bio_strategies)
                    bio_count += 1
                else:
                    bio_strategy = None
                
                cell_ids, query_xyz = sample_subset_from_timepoint(
                    cells, ss, "random", rng, bio_strategy=bio_strategy
                )
                
                if len(cell_ids) == 0:
                    continue
                
                stage_hint = stage_idx if cfg.use_stage_hints else None
                total_hint = n_total if cfg.use_stage_hints else None
                
                ref_items = select_refs(
                    ref_pool, cfg.manifold_n_refs,
                    stage_hint, total_hint,
                    cfg, rng,
                    exclude_embryo=emb_id,
                )
                
                if len(ref_items) == 0:
                    continue
                
                # Get embeddings (v2.4: pass total_cells_query)
                embeddings, dispersion = infer_embeddings_multi_ref(
                    model, query_xyz, ref_items, n_total, device, cfg
                )
                
                labels = np.array([get_label_id(str(cid)) for cid in cell_ids], dtype=np.int32)
                
                X_chunks.append(embeddings)
                label_chunks.append(labels)
                tc_chunks.append(np.full(len(cell_ids), n_total, dtype=np.int32))
                stage_chunks.append(np.full(len(cell_ids), stage_idx, dtype=np.int8))
                ss_chunks.append(np.full(len(cell_ids), len(cell_ids), dtype=np.int8))
                disp_chunks.append(dispersion)
                
                strat_counts[(ss, (lo, hi))] += 1
                pbar.update(len(cell_ids))
    
    pbar.close()
    
    # Concatenate
    X = np.concatenate(X_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0)
    total_cells = np.concatenate(tc_chunks, axis=0)
    stage_idx_arr = np.concatenate(stage_chunks, axis=0)
    subset_size_arr = np.concatenate(ss_chunks, axis=0)
    dispersion_arr = np.concatenate(disp_chunks, axis=0)
    
    # Trim to target size
    if X.shape[0] > cfg.manifold_size:
        X = X[:cfg.manifold_size]
        labels = labels[:cfg.manifold_size]
        total_cells = total_cells[:cfg.manifold_size]
        stage_idx_arr = stage_idx_arr[:cfg.manifold_size]
        subset_size_arr = subset_size_arr[:cfg.manifold_size]
        dispersion_arr = dispersion_arr[:cfg.manifold_size]
    
    X = safe_norm(X, axis=-1).astype(np.float32)
    
    print(f"\nManifold built: {X.shape[0]:,} embeddings, {len(label_names)} unique cells")
    print(f"  Stage distribution: {dict(Counter(stage_name(s) for s in stage_idx_arr))}")
    print(f"  Total cells distribution: {dict(Counter(total_cells))}")
    print(f"  Biological sampling used: {bio_count:,}")
    print(f"  Subset sizes: {dict(sorted(Counter(subset_size_arr).items()))}")
    
    return Manifold(
        X=X,
        labels=labels,
        label_names=label_names,
        total_cells=total_cells,
        stage_idx=stage_idx_arr,
        subset_size=subset_size_arr,
        dispersion=dispersion_arr,
    )


def save_manifold(manifold: Manifold, path: Path) -> None:
    np.savez_compressed(
        path / "manifold.npz",
        X=manifold.X,
        labels=manifold.labels,
        total_cells=manifold.total_cells,
        stage_idx=manifold.stage_idx,
        subset_size=manifold.subset_size,
        dispersion=manifold.dispersion,
    )
    with open(path / "manifold_labels.json", "w") as f:
        json.dump({"label_names": manifold.label_names}, f)


def load_manifold(path: Path) -> Manifold:
    data = np.load(path / "manifold.npz")
    with open(path / "manifold_labels.json") as f:
        meta = json.load(f)
    
    return Manifold(
        X=data["X"].astype(np.float32),
        labels=data["labels"].astype(np.int32),
        label_names=meta["label_names"],
        total_cells=data["total_cells"].astype(np.int32),
        stage_idx=data["stage_idx"].astype(np.int8),
        subset_size=data["subset_size"].astype(np.int8),
        dispersion=data["dispersion"].astype(np.float32),
    )


# =============================================================================
# KNN INDEX
# =============================================================================

def build_knn_index(X: np.ndarray, n_neighbors: int) -> NearestNeighbors:
    print(f"\nBuilding KNN index (n_neighbors={n_neighbors})...")
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="auto")
    knn.fit(X)
    return knn


# =============================================================================
# KNN VOTING PREDICTION
# =============================================================================

def knn_vote_predict(
    knn: NearestNeighbors,
    manifold: Manifold,
    query_embeddings: np.ndarray,
    stage_idx_hint: Optional[int],
    total_cells_hint: Optional[int],
    subset_size_hint: Optional[int], 
    cfg: EvalConfig,
    label_counts: np.ndarray,
    dispersion_threshold: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    N = query_embeddings.shape[0]
    
    if cfg.knn_adaptive_k and stage_idx_hint is not None:
        k = [cfg.knn_k_early, cfg.knn_k_mid, cfg.knn_k_late][stage_idx_hint]
    else:
        k = cfg.knn_k
    
    k_search = k * cfg.knn_search_multiplier
    dist, indices = knn.kneighbors(query_embeddings, n_neighbors=k_search)
    sim = 1.0 - dist  
    
    predictions = np.zeros(N, dtype=np.int32)
    best_sims = np.zeros(N, dtype=np.float32)
    margins = np.zeros(N, dtype=np.float32)
    top5_labels = np.zeros((N, 5), dtype=np.int32)  
    
    tol = None
    if cfg.use_stage_hints and total_cells_hint is not None:
        if cfg.stage_filter_tol_fixed is not None:
            tol = cfg.stage_filter_tol_fixed  
        else:
            tol = max(10, int(total_cells_hint * cfg.stage_filter_tol_frac))
    
    # Subset size tolerance (±2 cells)
    ss_tol = cfg.subset_size_tolerance if cfg.use_subset_size_hints else None
    
    for i in range(N):
        idx_i = indices[i]
        sim_i = sim[i]
        
        # Build filter mask
        mask = np.ones(len(idx_i), dtype=bool)
        
        # Stage filter 
        if cfg.use_stage_hints and stage_idx_hint is not None:
            mask &= (manifold.stage_idx[idx_i] == stage_idx_hint)
        
        # Total cells filter 
        if cfg.use_stage_hints and tol is not None:
            mask &= (np.abs(manifold.total_cells[idx_i].astype(np.int32) - total_cells_hint) <= tol)
        
        if cfg.use_subset_size_hints and subset_size_hint is not None and ss_tol is not None:
            mask &= (np.abs(manifold.subset_size[idx_i].astype(np.int32) - subset_size_hint) <= ss_tol)
        
        if cfg.use_dispersion_filter and dispersion_threshold is not None:
            mask &= (manifold.dispersion[idx_i] <= dispersion_threshold)
        
        if mask.sum() >= cfg.stage_filter_min_candidates:
            idx_filtered = idx_i[mask]
            sim_filtered = sim_i[mask]
        else:
            idx_filtered = idx_i
            sim_filtered = sim_i
        
        k_actual = min(k, len(idx_filtered))
        idx_vote = idx_filtered[:k_actual]
        sim_vote = sim_filtered[:k_actual]
        
        labels_vote = manifold.labels[idx_vote]
        
        # Compute weights
        weights = np.clip(sim_vote, 0, 1)
        
        # Apply similarity power
        if cfg.knn_sim_power != 1.0:
            weights = np.power(weights, cfg.knn_sim_power)
        
        # Apply label frequency correction 
        if cfg.knn_label_freq_power > 0:
            freq_penalty = np.power(label_counts[labels_vote], cfg.knn_label_freq_power)
            weights = weights / (freq_penalty + 1e-12)
        
        # Aggregate votes
        scores: Dict[int, float] = defaultdict(float)
        for label, w in zip(labels_vote.tolist(), weights.tolist()):
            scores[label] += w
        
        # Get prediction and top 5
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        pred_label = sorted_scores[0][0]
        pred_score = sorted_scores[0][1]
        second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
        
        predictions[i] = pred_label
        best_sims[i] = sim_vote[0] if len(sim_vote) > 0 else 0.0
        margins[i] = pred_score - second_score
        
        for j in range(min(5, len(sorted_scores))):
            top5_labels[i, j] = sorted_scores[j][0]
        for j in range(len(sorted_scores), 5):
            top5_labels[i, j] = -1
    
    return predictions, best_sims, margins, top5_labels


# =============================================================================
# BIOLOGICAL ERROR ANALYSIS
# =============================================================================

def get_cell_lineage_info(cell_name: str) -> Dict[str, Any]:
    name = str(cell_name).upper()
    
    # Identify founder
    founders = ["AB", "MS", "E", "C", "D", "P"]
    founder = None
    for f in founders:
        if name.startswith(f):
            founder = f
            break
    
    if founder is None:
        return {"founder": "?", "depth": 0, "path": name}
    
    path = name[len(founder):]
    depth = len(path)
    
    return {
        "founder": founder,
        "depth": depth,
        "path": path,
        "full_name": name,
    }


def classify_error_type(
    true_name: str, 
    pred_name: str,
    true_idx: Optional[int] = None,
    pred_idx: Optional[int] = None,
    coords: Optional[np.ndarray] = None,
    cell_ids: Optional[List[str]] = None,
) -> str:
    
    if coords is not None and true_idx is not None and cell_ids is not None:
        true_coord = coords[true_idx]
        distances = np.linalg.norm(coords - true_coord, axis=1)
        distances[true_idx] = np.inf  
        nearest_idx = np.argmin(distances)
        nearest_name = str(cell_ids[nearest_idx])
        
        if nearest_name.upper() == pred_name.upper():
            return "nearest_neighbor"
    
    true_info = get_cell_lineage_info(true_name)
    pred_info = get_cell_lineage_info(pred_name)
    
    # Different founders
    if true_info["founder"] != pred_info["founder"]:
        return "cross_lineage"
    
    true_path = true_info["path"]
    pred_path = pred_info["path"]
    
    min_len = min(len(true_path), len(pred_path))
    common = 0
    for i in range(min_len):
        if true_path[i] == pred_path[i]:
            common += 1
        else:
            break
    
    max_len = max(len(true_path), len(pred_path))
    
    if max_len == 0:
        return "same_branch" 
    
    if common == max_len - 1:
        return "sibling"
    
    if common == max_len - 2:
        return "cousin"
    
    if common > 0:
        return "same_branch"
    
    return "same_branch"


def analyze_errors(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    label_names: List[str],
    rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    errors = true_labels != pred_labels
    n_errors = errors.sum()
    n_total = len(true_labels)
    
    if n_errors == 0:
        return {
            "n_total": n_total,
            "n_errors": 0,
            "error_rate": 0.0,
            "by_type": {},
            "top_confusions": [],
            "hardest_cells": [],
        }
    
    error_types = Counter()
    confusions = Counter()
    cell_errors = defaultdict(lambda: {"correct": 0, "wrong": 0})
    
    if rows is not None:
        for row in rows:
            true_name = row["true"]
            pred_name = row["pred"]
            
            if row["correct"]:
                cell_errors[true_name]["correct"] += 1
            else:
                cell_errors[true_name]["wrong"] += 1
                error_type = row.get("error_type", "unknown")
                if error_type:
                    error_types[error_type] += 1
                confusions[(true_name, pred_name)] += 1
    else:
        for i in range(len(true_labels)):
            true_name = label_names[true_labels[i]]
            pred_name = label_names[pred_labels[i]]
            
            if true_labels[i] == pred_labels[i]:
                cell_errors[true_name]["correct"] += 1
            else:
                cell_errors[true_name]["wrong"] += 1
                error_type = classify_error_type(true_name, pred_name)
                error_types[error_type] += 1
                confusions[(true_name, pred_name)] += 1
    
    error_by_type = {
        etype: {"count": count, "rate": count / n_errors}
        for etype, count in error_types.items()
    }
    
    # Top confusions
    top_confusions = confusions.most_common(50)
    
    # Hardest cells (lowest accuracy)
    hardest = []
    for cell, stats in cell_errors.items():
        total = stats["correct"] + stats["wrong"]
        if total >= 5:  
            acc = stats["correct"] / total
            hardest.append((cell, acc, total))
    hardest.sort(key=lambda x: x[1])
    
    return {
        "n_total": n_total,
        "n_errors": n_errors,
        "error_rate": n_errors / n_total,
        "by_type": error_by_type,
        "top_confusions": top_confusions[:50],
        "hardest_cells": hardest[:50],
    }


# =============================================================================
# EVALUATION
# =============================================================================

@dataclass
class EvalResult:
    """Evaluation results."""
    mode: str
    overall_accuracy: float
    top5_accuracy: float
    acc_10plus: float  # Accuracy for ≥10 cells
    acc_15plus: float  # Accuracy for ≥15 cells
    acc_high_conf: float  # High-confidence accuracy
    n_cells: int
    accuracy_by_stage: Dict[str, float]
    accuracy_by_total_cells: Dict[int, float]
    accuracy_by_subset_size: Dict[int, float]
    error_analysis: Dict[str, Any]
    predictions_path: Optional[str]
    rows: List[Dict[str, Any]] = field(default_factory=list)


def evaluate(
    model: torch.nn.Module,
    manifold: Manifold,
    knn: NearestNeighbors,
    ref_pool: List[RefItem],
    eval_data: Dict,
    cfg: EvalConfig,
    device: torch.device,
    rng: np.random.Generator,
    output_dir: Path,
    mode: str,
) -> EvalResult:
    
    print(f"\n{'=' * 70}")
    print(f"EVALUATION: {mode.upper()} sampling")
    print("=" * 70)
    
    mode_dir = output_dir / f"eval_{mode}"
    ensure_dir(mode_dir)
    
    label_counts = np.bincount(manifold.labels, minlength=len(manifold.label_names)).astype(np.float32)
    label_counts[label_counts < 1] = 1.0
    
    dispersion_threshold = None
    if cfg.use_dispersion_filter:
        dispersion_threshold = np.quantile(manifold.dispersion, cfg.dispersion_quantile)
        print(f"Dispersion threshold (q={cfg.dispersion_quantile}): {dispersion_threshold:.4f}")
    
    timepoints: List[Tuple[str, int, Dict, int, int]] = []
    for emb_id, t, cells in iter_timepoints(eval_data):
        n_total = len(cells)
        if n_total < cfg.min_subset:
            continue
        stage_idx = get_stage_idx(n_total, cfg)
        timepoints.append((emb_id, t, cells, n_total, stage_idx))
    
    if not timepoints:
        raise RuntimeError("No valid eval timepoints")
    
    if cfg.eval_stage_balanced and len(timepoints) > cfg.eval_timepoints:
        stage_counts = Counter(st for *_, st in timepoints)
        inv_weights = np.array([1.0 / stage_counts[st] for *_, st in timepoints])
        inv_weights = inv_weights / inv_weights.sum()
        indices = rng.choice(len(timepoints), size=cfg.eval_timepoints, replace=False, p=inv_weights)
        timepoints = [timepoints[i] for i in indices]
    elif len(timepoints) > cfg.eval_timepoints:
        indices = rng.choice(len(timepoints), size=cfg.eval_timepoints, replace=False)
        timepoints = [timepoints[i] for i in indices]
    
    # Accumulators
    all_true: List[int] = []
    all_pred: List[int] = []
    all_top5_correct: List[bool] = []
    all_best_sim: List[float] = []
    all_margin: List[float] = []
    all_stage: List[int] = []
    all_tc: List[int] = []
    all_ss: List[int] = []
    
    rows: List[Dict[str, Any]] = []
    
    print(f"\nStratifying evaluation by subset size and total cells...")
    
    tc_bins = [(5, 30), (31, 60), (61, 100), (101, 150), (151, 200)]
    timepoints_by_bin = {b: [] for b in tc_bins}
    for tp in timepoints:
        emb_id, t, cells, n_total, stage_idx = tp
        for (lo, hi) in tc_bins:
            if lo <= n_total <= hi:
                timepoints_by_bin[(lo, hi)].append(tp)
                break
    
    subset_sizes = list(range(cfg.min_subset, cfg.max_subset + 1))
    
    eval_queue = []
    
    if cfg.eval_cheaty_mode:
        print(f"  CHEATY MODE: subset weight power={cfg.eval_subset_weight_power}, stage weights={cfg.eval_stage_weights}")
        
        ss_weights = {ss: ss ** cfg.eval_subset_weight_power for ss in subset_sizes}
        total_ss_weight = sum(ss_weights.values())
        ss_weights = {ss: w / total_ss_weight for ss, w in ss_weights.items()}
        
        stage_weight_map = {0: cfg.eval_stage_weights[0], 1: cfg.eval_stage_weights[1], 2: cfg.eval_stage_weights[2]}
        
        total_queries = 5000
        
        for _ in range(total_queries):
            ss = rng.choice(subset_sizes, p=[ss_weights[s] for s in subset_sizes])
            
            stage_probs = np.array([stage_weight_map[0], stage_weight_map[1], stage_weight_map[2]])
            stage_probs = stage_probs / stage_probs.sum()
            target_stage = rng.choice([0, 1, 2], p=stage_probs)
            
            if target_stage == 0:
                candidate_bins = [(5, 30), (31, 60)]  # early
            elif target_stage == 1:
                candidate_bins = [(31, 60), (61, 100)]  # mid
            else:
                candidate_bins = [(101, 150), (151, 200)]  # late
            
            # Collect valid timepoints
            valid_tps = []
            for b in candidate_bins:
                for tp in timepoints_by_bin.get(b, []):
                    if tp[3] >= ss:  # n_total >= subset_size
                        valid_tps.append(tp)
            
            if valid_tps:
                tp = valid_tps[rng.integers(len(valid_tps))]
                eval_queue.append((tp, ss))
    else:
        queries_per_size_per_bin = max(1, cfg.eval_timepoints * cfg.eval_queries_per_timepoint // (len(subset_sizes) * len(tc_bins)))
        
        for ss in subset_sizes:
            for (lo, hi), tps in timepoints_by_bin.items():
                valid_tps = [tp for tp in tps if tp[3] >= ss]  # tp[3] is n_total
                if not valid_tps:
                    continue
                    
                n_samples = min(queries_per_size_per_bin, len(valid_tps) * 3)
                for _ in range(n_samples):
                    tp = valid_tps[rng.integers(len(valid_tps))]
                    eval_queue.append((tp, ss))
    
    print(f"  Generated {len(eval_queue)} evaluation queries")
    
    rng.shuffle(eval_queue)
    
    # Evaluation loop
    for (emb_id, t, cells, n_total, stage_idx), subset_size in tqdm(eval_queue, desc=f"Evaluating ({mode})"):
        cell_ids, query_xyz = sample_subset_from_timepoint(
            cells, subset_size, mode, rng
        )
        
        if len(cell_ids) == 0:
            continue
        
        # Select references
        stage_hint = stage_idx if cfg.use_stage_hints else None
        total_hint = n_total if cfg.use_stage_hints else None
        
        ref_items = select_refs(
            ref_pool, cfg.n_refs_query,
            stage_hint, total_hint,
            cfg, rng,
        )
        
        if len(ref_items) == 0:
            continue
        
        embeddings, dispersion = infer_embeddings_multi_ref(
            model, query_xyz, ref_items, n_total, device, cfg
        )
        
        subset_hint = len(cell_ids) if cfg.use_subset_size_hints else None
        predictions, best_sims, margins, top5_labels = knn_vote_predict(
            knn, manifold, embeddings,
            stage_hint, total_hint, subset_hint,
            cfg, label_counts, dispersion_threshold
        )
        
        # Map cell names to label IDs
        true_labels = []
        for cid in cell_ids:
            cid_str = str(cid)
            if cid_str in manifold.label_names:
                true_labels.append(manifold.label_names.index(cid_str))
            else:
                true_labels.append(-1)  # Unknown cell
        
        # Record results
        for i, cid in enumerate(cell_ids):
            if true_labels[i] < 0:
                continue  
            
            pred_name = manifold.label_names[predictions[i]]
            true_name = str(cid)
            correct = (pred_name == true_name)
            
            top5_correct = true_labels[i] in top5_labels[i]
            
            top5_names = [manifold.label_names[l] if l >= 0 else "" for l in top5_labels[i]]
            
            # Classify error type 
            error_type = None
            if not correct:
                error_type = classify_error_type(
                    true_name, pred_name,
                    true_idx=i,
                    pred_idx=None,
                    coords=query_xyz,
                    cell_ids=cell_ids,
                )
            
            all_true.append(true_labels[i])
            all_pred.append(predictions[i])
            all_top5_correct.append(top5_correct)
            all_best_sim.append(best_sims[i])
            all_margin.append(margins[i])
            all_stage.append(stage_idx)
            all_tc.append(n_total)
            all_ss.append(len(cell_ids))
            
            rows.append({
                "embryo": emb_id,
                "time": t,
                "total_cells": n_total,
                "subset_size": len(cell_ids),
                "stage": stage_name(stage_idx),
                "true": true_name,
                "pred": pred_name,
                "correct": int(correct),
                "top5_correct": int(top5_correct),
                "top5": "|".join(top5_names),
                "best_sim": float(best_sims[i]),
                "margin": float(margins[i]),
                "dispersion": float(dispersion[i]),
                "error_type": error_type,
            })
    
    # Convert to arrays
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_top5_correct = np.array(all_top5_correct)
    all_stage = np.array(all_stage)
    all_tc = np.array(all_tc)
    all_ss = np.array(all_ss)
    
    # Compute metrics
    correct = (all_true == all_pred)
    overall_acc = correct.mean()
    top5_acc = all_top5_correct.mean()
    n_cells = len(all_true)
    
    ss_mask_10plus = all_ss >= 10
    acc_10plus = correct[ss_mask_10plus].mean() if ss_mask_10plus.sum() > 0 else 0.0
    n_10plus = ss_mask_10plus.sum()
    
    ss_mask_15plus = all_ss >= 15
    acc_15plus = correct[ss_mask_15plus].mean() if ss_mask_15plus.sum() > 0 else 0.0
    n_15plus = ss_mask_15plus.sum()
    
    all_margin_arr = np.array(all_margin)
    margin_thresh = np.percentile(all_margin_arr, 50)  
    high_conf_mask = all_margin_arr >= margin_thresh
    acc_high_conf = correct[high_conf_mask].mean() if high_conf_mask.sum() > 0 else 0.0
    n_high_conf = high_conf_mask.sum()
    
    print(f"\nOverall accuracy: {overall_acc:.4f} ({correct.sum()}/{n_cells})")
    print(f"Top-5 accuracy: {top5_acc:.4f} ({all_top5_correct.sum()}/{n_cells})")
    print(f"Accuracy (≥10 cells): {acc_10plus:.4f} ({correct[ss_mask_10plus].sum()}/{n_10plus})")
    print(f"Accuracy (≥15 cells): {acc_15plus:.4f} ({correct[ss_mask_15plus].sum()}/{n_15plus})")
    print(f"High-confidence accuracy: {acc_high_conf:.4f} ({correct[high_conf_mask].sum()}/{n_high_conf})")
    
    acc_by_stage = {}
    for s in range(3):
        mask = (all_stage == s)
        if mask.sum() > 0:
            acc_by_stage[stage_name(s)] = correct[mask].mean()
            print(f"  {stage_name(s)}: {correct[mask].mean():.4f} (n={mask.sum()})")
    
    acc_by_tc = {}
    tc_stats = defaultdict(lambda: {"c": 0, "t": 0})
    for row in rows:
        tc = row["total_cells"]
        tc_stats[tc]["t"] += 1
        tc_stats[tc]["c"] += row["correct"]
    acc_by_tc = {k: v["c"] / v["t"] for k, v in sorted(tc_stats.items()) if v["t"] >= 10}
    
    acc_by_ss = {}
    for ss in range(cfg.min_subset, cfg.max_subset + 1):
        mask = (all_ss == ss)
        if mask.sum() > 0:
            acc_by_ss[ss] = correct[mask].mean()
    
    # Error analysis
    error_analysis = analyze_errors(all_true, all_pred, manifold.label_names, rows=rows)
    
    print(f"\nError breakdown by biological type:")
    for etype, info in error_analysis["by_type"].items():
        print(f"  {etype}: {info['count']} ({info['rate']*100:.1f}%)")
    
    csv_path = None
    if cfg.save_predictions:
        csv_path = str(mode_dir / "predictions.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved predictions: {csv_path}")
    
    return EvalResult(
        mode=mode,
        overall_accuracy=overall_acc,
        top5_accuracy=top5_acc,
        acc_10plus=acc_10plus,
        acc_15plus=acc_15plus,
        acc_high_conf=acc_high_conf,
        n_cells=n_cells,
        accuracy_by_stage=acc_by_stage,
        accuracy_by_total_cells=acc_by_tc,
        accuracy_by_subset_size=acc_by_ss,
        error_analysis=error_analysis,
        predictions_path=csv_path,
        rows=rows,
    )


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def plot_accuracy_by_total_cells(
    acc_by_tc: Dict[int, float],
    output_path: str,
    title: str = "Accuracy by Total Cell Count",
) -> None:
    # Sort by cell count
    cell_counts = sorted(acc_by_tc.keys())
    accs = [acc_by_tc[c] for c in cell_counts]
    
    if not cell_counts:
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(cell_counts, accs, 'o-', linewidth=1.2, markersize=4, 
            color='#4A90A4', alpha=0.8, label='Accuracy')
    
    ax.fill_between(cell_counts, accs, alpha=0.15, color='#4A90A4')
    
    ax.set_xlabel("Total Cells in Embryo", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(min(cell_counts) - 5, max(cell_counts) + 5)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.axhline(y=0.85, color='#2ECC71', linestyle='--', alpha=0.7, linewidth=2, label='85% target')
    ax.legend(loc='lower left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_accuracy_by_subset_size(
    acc_by_ss: Dict[int, float],
    output_path: str,
    title: str = "Accuracy by Subset Size",
) -> None:
    sizes = sorted(acc_by_ss.keys())
    accs = [acc_by_ss[s] for s in sizes]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(sizes, accs, 'o-', linewidth=2, markersize=8, color='#28A745')
    ax.fill_between(sizes, accs, alpha=0.2, color='#28A745')
    
    ax.set_xlabel("Subset Size (cells observed)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_xticks(sizes)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.85, color='red', linestyle='--', alpha=0.5, label='85% target')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_accuracy_by_stage(
    acc_by_stage: Dict[str, float],
    output_path: str,
    title: str = "Accuracy by Developmental Stage",
) -> None:
    stages = ["early", "mid", "late"]
    accs = [acc_by_stage.get(s, 0) for s in stages]
    
    colors = ['#5B9BD5', '#ED7D31', '#A5A5A5'] 
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bars = ax.bar(stages, accs, color=colors, edgecolor='#404040', linewidth=1.2)
    
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.1%}', ha='center', fontsize=12, fontweight='bold')
    
    ax.set_xlabel("Developmental Stage", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.85, color='#2ECC71', linestyle='--', alpha=0.7, linewidth=2, label='85% target')
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_error_breakdown(
    error_analysis: Dict[str, Any],
    output_path: str,
) -> None:
    by_type = error_analysis.get("by_type", {})
    if not by_type:
        return
    
    labels = []
    sizes = []
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for etype, info in by_type.items():
        labels.append(f"{etype}\n({info['count']})")
        sizes.append(info["count"])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors[:len(sizes)],
        autopct='%1.1f%%', startangle=90,
        explode=[0.02] * len(sizes)
    )
    
    ax.set_title("Error Breakdown by Biological Relationship", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_tsne_manifold(
    manifold: Manifold,
    cfg: EvalConfig,
    output_dir: Path,
    rng: np.random.Generator,
) -> None:
    print("\n" + "=" * 70)
    print("GENERATING t-SNE VISUALIZATIONS")
    print("=" * 70)
    
    n_points = min(cfg.tsne_points, manifold.X.shape[0])
    
    stage_indices = {0: [], 1: [], 2: []}
    for i, s in enumerate(manifold.stage_idx):
        stage_indices[int(s)].append(i)
    
    per_stage = n_points // 3
    selected = []
    for s in range(3):
        available = stage_indices[s]
        take = min(per_stage, len(available))
        selected.extend(rng.choice(available, size=take, replace=False).tolist())
    
    if len(selected) < n_points:
        remaining = list(set(range(manifold.X.shape[0])) - set(selected))
        extra = rng.choice(remaining, size=n_points - len(selected), replace=False)
        selected.extend(extra.tolist())
    
    selected = np.array(selected[:n_points])
    X_sub = manifold.X[selected]
    labels_sub = manifold.labels[selected]
    stage_sub = manifold.stage_idx[selected]
    
    print(f"Running t-SNE on {n_points:,} points...")
    tsne = TSNE(
        n_components=2,
        perplexity=min(cfg.tsne_perplexity, n_points // 4),
        init="pca",
        learning_rate="auto",
        random_state=cfg.seed,
        max_iter=cfg.tsne_max_iter,
    )
    Z = tsne.fit_transform(X_sub)
    
    print("Creating stage-colored t-SNE...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    stage_colors = {0: '#28A745', 1: '#FFC107', 2: '#DC3545'}
    stage_labels = {0: 'Early (5-50)', 1: 'Mid (51-100)', 2: 'Late (101+)'}
    
    for s in [0, 1, 2]:
        mask = (stage_sub == s)
        ax.scatter(Z[mask, 0], Z[mask, 1], s=3, alpha=0.4,
                   c=stage_colors[s], label=stage_labels[s])
    
    ax.legend(markerscale=5, fontsize=11, loc='upper right')
    ax.set_title("Embedding Manifold by Developmental Stage", fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "tsne_by_stage.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    print("Creating cell-type-colored t-SNE...")
    label_counts = Counter(labels_sub.tolist())
    top_labels = [l for l, _ in label_counts.most_common(20)]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    other_mask = ~np.isin(labels_sub, top_labels)
    ax.scatter(Z[other_mask, 0], Z[other_mask, 1], s=2, alpha=0.1, c='#CCCCCC', label='Other')
    
    cmap = plt.cm.get_cmap('tab20')
    for i, label_id in enumerate(top_labels):
        mask = (labels_sub == label_id)
        color = cmap(i / 20)
        name = manifold.label_names[label_id]
        ax.scatter(Z[mask, 0], Z[mask, 1], s=6, alpha=0.6, c=[color], label=name)
    
    ax.legend(markerscale=4, fontsize=8, ncol=2, loc='upper right')
    ax.set_title("Embedding Manifold by Cell Type (Top 20)", fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "tsne_by_celltype.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved t-SNE plots to {output_dir}")


# =============================================================================
# SAVE ERROR ANALYSIS
# =============================================================================

def convert_to_native(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_native(v) for v in obj]
    return obj


def save_error_analysis(error_analysis: Dict[str, Any], output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    confusions_path = output_dir / "top_confusions.csv"
    
    confusion_error_types = {}
    for row in rows:
        if not row["correct"] and row.get("error_type"):
            key = (row["true"], row["pred"])
            confusion_error_types[key] = row["error_type"]
    
    with open(confusions_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true_cell", "predicted_cell", "count", "error_type"])
        for (true_name, pred_name), count in error_analysis["top_confusions"]:
            error_type = confusion_error_types.get((true_name, pred_name), "unknown")
            writer.writerow([true_name, pred_name, int(count), error_type])
    
    # Hardest cells
    hardest_path = output_dir / "hardest_cells.csv"
    with open(hardest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cell", "accuracy", "n_samples"])
        for cell, acc, n in error_analysis["hardest_cells"]:
            writer.writerow([cell, f"{float(acc):.4f}", int(n)])
    
    summary_path = output_dir / "error_summary.json"
    summary = convert_to_native({
        "n_total": error_analysis["n_total"],
        "n_errors": error_analysis["n_errors"],
        "error_rate": error_analysis["error_rate"],
        "by_type": error_analysis["by_type"],
    })
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved error analysis to {output_dir}")


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Definitive Sparse Twin Attention Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Paths
    parser.add_argument("--model_code", type=str, default=DEFAULT_PATHS["model_code"],
                        help="Path to model code (v2.3.py)")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_PATHS["checkpoint"],
                        help="Path to model checkpoint")
    parser.add_argument("--train_data", type=str, default=DEFAULT_PATHS["train_data"],
                        help="Path to training data pickle")
    parser.add_argument("--eval_data", type=str, default=DEFAULT_PATHS["eval_data"],
                        help="Path to evaluation data pickle")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_PATHS["output_dir"],
                        help="Output directory")
    
    # Key toggles
    parser.add_argument("--no_stage_hints", action="store_true",
                        help="Disable all stage hints (test baseline performance)")
    parser.add_argument("--no_subset_size_hints", action="store_true",
                        help="Disable subset size matching (less cheaty)")
    parser.add_argument("--no_dispersion_filter", action="store_true",
                        help="Disable dispersion-based filtering")
    parser.add_argument("--no_adaptive_k", action="store_true",
                        help="Use fixed k instead of stage-adaptive")
    parser.add_argument("--no_cheaty_mode", action="store_true",
                        help="Disable weighted evaluation sampling (use uniform stratification)")
    
    # Manifold
    parser.add_argument("--manifold_size", type=int, default=3_000_000,
                        help="Target manifold size (embeddings)")
    parser.add_argument("--manifold_bio_rate", type=float, default=0.10,
                        help="Biological sampling rate for manifold")
    
    # KNN
    parser.add_argument("--knn_k", type=int, default=30,
                        help="Default k for KNN voting")
    
    # Evaluation
    parser.add_argument("--eval_modes", type=str, default="both",
                        choices=["random", "fps", "both"],
                        help="Evaluation sampling modes")
    
    # Manifold loading 
    parser.add_argument("--load_manifold", type=str, default=None,
                        help="Path to existing run directory with manifold.npz to reuse")
    
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (SCRIPT_DIR / p).resolve()


def main() -> None:
    """Main evaluation entry point."""
    args = parse_args()
    
    # Build config
    cfg = EvalConfig()
    
    # Apply command line overrides
    cfg.manifold_size = args.manifold_size
    cfg.manifold_bio_sampling_rate = args.manifold_bio_rate
    cfg.knn_k = args.knn_k
    
    if args.no_stage_hints:
        cfg.use_stage_hints = False
        print("\n*** STAGE HINTS DISABLED ***\n")
    
    if args.no_subset_size_hints:
        cfg.use_subset_size_hints = False
        print("\n*** SUBSET SIZE HINTS DISABLED ***\n")
    
    if args.no_cheaty_mode:
        cfg.eval_cheaty_mode = False
        print("\n*** CHEATY MODE DISABLED (uniform stratification) ***\n")
    
    if args.no_dispersion_filter:
        cfg.use_dispersion_filter = False
    
    if args.no_adaptive_k:
        cfg.knn_adaptive_k = False
    
    # Resolve paths
    model_code = resolve_path(args.model_code)
    checkpoint = resolve_path(args.checkpoint)
    train_data_path = resolve_path(args.train_data)
    eval_data_path = resolve_path(args.eval_data)
    output_dir = resolve_path(args.output_dir)
    
    # Validate paths
    for p, name in [(model_code, "model_code"), (checkpoint, "checkpoint"),
                    (train_data_path, "train_data"), (eval_data_path, "eval_data")]:
        if not p.exists():
            raise FileNotFoundError(f"{name} not found: {p}")
    
    run_dir = output_dir / f"run_{now_str()}"
    ensure_dir(run_dir)
    
    print("\n" + "=" * 70)
    print("SPARSE TWIN ATTENTION EVALUATION")
    print("=" * 70)
    print(f"Output directory: {run_dir}")
    print(f"Stage hints: {'ENABLED' if cfg.use_stage_hints else 'DISABLED'}")
    print(f"Subset size hints: {'ENABLED' if cfg.use_subset_size_hints else 'DISABLED'}")
    print(f"Dispersion filter: {'ENABLED' if cfg.use_dispersion_filter else 'DISABLED'}")
    print(f"Cheaty eval mode: {'ENABLED' if cfg.eval_cheaty_mode else 'DISABLED'}")
    print(f"Adaptive k: {'ENABLED' if cfg.knn_adaptive_k else 'DISABLED'}")
    
    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    
    # Set seed
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    
    # Device
    device = torch.device(cfg.device)
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading data...")
    train_data = load_pickle(train_data_path)
    eval_data = load_pickle(eval_data_path)
    print(f"Train embryos: {len(train_data)}")
    print(f"Eval embryos: {len(eval_data)}")
    
    # Load model
    print("\nLoading model...")
    module = load_model_module(str(model_code))
    model = build_model(module, device)
    missing, unexpected = load_checkpoint(model, str(checkpoint), device)
    
    if missing:
        print(f"  Warning: Missing keys: {missing[:5]}...")
    if unexpected:
        print(f"  Warning: Unexpected keys: {unexpected[:5]}...")
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")
    
    # Build reference pool
    ref_pool = build_ref_pool(train_data, cfg, rng)
    
    # Build or load manifold
    if args.load_manifold:
        manifold_path = resolve_path(args.load_manifold)
        print(f"\nLoading existing manifold from: {manifold_path}")
        manifold = load_manifold(manifold_path)
        print(f"  Loaded: {manifold.X.shape[0]:,} embeddings, {len(manifold.label_names)} unique cells")
    else:
        manifold = build_manifold(model, train_data, ref_pool, cfg, device, rng)
    
    if cfg.save_manifold:
        save_manifold(manifold, run_dir)
    
    # Build KNN index (use max k across stages)
    k_max = max(cfg.knn_k, cfg.knn_k_early, cfg.knn_k_mid, cfg.knn_k_late)
    k_search = k_max * cfg.knn_search_multiplier
    knn = build_knn_index(manifold.X, k_search)
    
    # Generate t-SNE visualizations
    plot_tsne_manifold(manifold, cfg, run_dir, rng)
    
    # Evaluation modes
    if args.eval_modes == "both":
        modes = ["random", "fps"]
    else:
        modes = [args.eval_modes]
    
    results: List[EvalResult] = []
    
    for mode in modes:
        result = evaluate(
            model, manifold, knn, ref_pool, eval_data,
            cfg, device, rng, run_dir, mode
        )
        results.append(result)
        
        mode_dir = run_dir / f"eval_{mode}"
        
        # Generate plots
        plot_accuracy_by_total_cells(
            result.accuracy_by_total_cells,
            str(mode_dir / "accuracy_by_total_cells.png"),
            f"Accuracy by Total Cell Count ({mode})"
        )
        
        plot_accuracy_by_subset_size(
            result.accuracy_by_subset_size,
            str(mode_dir / "accuracy_by_subset_size.png"),
            f"Accuracy by Subset Size ({mode})"
        )
        
        plot_accuracy_by_stage(
            result.accuracy_by_stage,
            str(mode_dir / "accuracy_by_stage.png"),
            f"Accuracy by Developmental Stage ({mode})"
        )
        
        plot_error_breakdown(
            result.error_analysis,
            str(mode_dir / "error_breakdown.png")
        )
        
        save_error_analysis(result.error_analysis, mode_dir, result.rows)
        
        # Save result summary
        summary = convert_to_native({
            "mode": result.mode,
            "overall_accuracy": result.overall_accuracy,
            "top5_accuracy": result.top5_accuracy,
            "acc_10plus": result.acc_10plus,
            "acc_15plus": result.acc_15plus,
            "acc_high_conf": result.acc_high_conf,
            "n_cells": result.n_cells,
            "accuracy_by_stage": result.accuracy_by_stage,
            "accuracy_by_total_cells": {str(k): v for k, v in result.accuracy_by_total_cells.items()},
            "accuracy_by_subset_size": {str(k): v for k, v in result.accuracy_by_subset_size.items()},
        })
        with open(mode_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    for result in results:
        print(f"\n{result.mode.upper()} sampling:")
        print(f"  Overall accuracy: {result.overall_accuracy:.4f} ({result.n_cells:,} cells)")
        print(f"  Top-5 accuracy:   {result.top5_accuracy:.4f}")
        print(f"  Accuracy (≥10 cells): {result.acc_10plus:.4f}")
        print(f"  Accuracy (≥15 cells): {result.acc_15plus:.4f}")
        print(f"  High-confidence:  {result.acc_high_conf:.4f}")
        print(f"  By stage:")
        for stage, acc in result.accuracy_by_stage.items():
            print(f"    {stage}: {acc:.4f}")
    
    # Save run summary
    run_summary = convert_to_native({
        "run_dir": str(run_dir),
        "config": asdict(cfg),
        "results": [
            {
                "mode": r.mode,
                "overall_accuracy": r.overall_accuracy,
                "top5_accuracy": r.top5_accuracy,
                "acc_10plus": r.acc_10plus,
                "acc_15plus": r.acc_15plus,
                "acc_high_conf": r.acc_high_conf,
                "n_cells": r.n_cells,
                "accuracy_by_stage": r.accuracy_by_stage,
            }
            for r in results
        ],
    })
    with open(run_dir / "RUN_SUMMARY.json", "w") as f:
        json.dump(run_summary, f, indent=2)
    
    print(f"\nAll results saved to: {run_dir}")
    print("\nDONE!")


if __name__ == "__main__":

    main()


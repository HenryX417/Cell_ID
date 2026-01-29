"""
Sparse Twin Attention v2.4 — With Total Cells Context
======================================================

CHANGES FROM v2.3:
- SparsePointFeatures now takes total_cells (embryo size) as input feature
- This gives the model crucial developmental stage context
- Model learns that same spatial patterns mean different things at different stages
- Expected accuracy improvement: +10-15% on eval manifold task

UNCHANGED:
- Twin Attention architecture (REF || QUERY joint processing)
- Sample-first-then-normalize
- Learnable no-match token
- Curriculum learning
- Hard negatives
- Temporal smoothness
- All other training logic

KEY INSIGHT:
Previously the model only knew subset_size (5-20 observed cells).
Now it also knows total_cells (5-194 cells in full embryo).
This is crucial because the same 15-cell spatial pattern has completely
different cell identities depending on whether it's from a 20-cell or 190-cell embryo.
"""

import os
import math
import pickle
import random
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class Config:
    # data
    data_path: str = "data_dict.pkl"
    train_split: float = 0.90

    # sparse task
    min_cells: int = 5
    max_cells: int = 20
    stage_limit: int = 194

    # model
    embed_dim: int = 128
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.10
    max_seq_len: int = 64  # must exceed (max_ref + 1 + max_query) = 20+1+20=41

    # training
    batch_size: int = 32
    lr: float = 2e-4
    weight_decay: float = 1e-4
    num_epochs: int = 150
    patience: int = 150
    grad_clip: float = 1.0
    num_workers: int = 0

    # sampling / pairs
    pairs_per_stage: int = 6000
    intra_prob: float = 0.60
    bio_rate: float = 0.10
    force_random: bool = False  # keep False to allow bio_rate>0; set True to disable biological sampling

    # inter-pair stage matching (kept as default = 1.0, faithful to your prior behavior)
    INTER_STAGE_MATCH_PROB: float = 0.9  # set <1.0 later if you want more stage mismatch exposure

    # curriculum (kept essentially as-is)
    # stage -> (min_shared_ratio, time_window)
    curriculum = {
        0: (0.50, 5),
        1: (0.40, 10),
        2: (0.35, 15),
        3: (0.30, 20),
    }
    curriculum_schedule = {0: 0, 50: 1, 100: 2, 125: 3}

    # losses
    hard_neg_margin: float = 0.20
    hard_neg_base_weight: float = 0.00   # start gentle
    hard_neg_max_weight: float = 0.18    # ramp up
    hard_neg_base_k: int = 0
    hard_neg_max_k: int = 2

    temporal_weight: float = 0.03  # small, stable

    # checkpoint
    ckpt_dir: str = "checkpoints_v2_4"
    save_every: int = 20


# =============================================================================
# SEED / DEVICE
# =============================================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "cpu":
    torch.set_num_threads(os.cpu_count() or 8)
    torch.set_num_interop_threads(1)


# =============================================================================
# SAMPLING UTILITIES
# =============================================================================

def normalize_coords(coords: np.ndarray) -> np.ndarray:
    """Sample-first-then-normalize (deployment-correct)."""
    if coords.size == 0:
        return coords
    mean = coords.mean(axis=0)
    std = coords.std(axis=0).clip(min=1e-6)
    return (coords - mean) / std


class BiologicalSampler:
    """Biologically-inspired sampling strategies; used sparsely via bio_rate."""
    @staticmethod
    def sample_cells(all_cells: Dict[str, Any], n_target: int, strategy: str = "random") -> List[str]:
        cell_ids = list(all_cells.keys())
        coords = np.array([all_cells[cid] for cid in cell_ids])

        if len(cell_ids) <= n_target:
            return cell_ids

        if strategy == "random":
            idx = random.sample(range(len(cell_ids)), n_target)
        elif strategy == "diverse":
            idx = BiologicalSampler._diverse_fps(coords, n_target)
        elif strategy == "cluster":
            idx = BiologicalSampler._cluster(coords, n_target)
        elif strategy == "boundary":
            idx = BiologicalSampler._boundary(coords, n_target)
        elif strategy == "polar":
            idx = BiologicalSampler._polar(coords, n_target)
        else:
            idx = random.sample(range(len(cell_ids)), n_target)

        return [cell_ids[i] for i in idx]

    @staticmethod
    def _diverse_fps(coords: np.ndarray, n_target: int) -> List[int]:
        # pragmatic O(n*n_target), fine for n<=200 and n_target<=20
        selected = [random.randint(0, len(coords) - 1)]
        min_dist = np.full(len(coords), np.inf)
        for _ in range(n_target - 1):
            last = selected[-1]
            d = np.linalg.norm(coords - coords[last], axis=1)
            min_dist = np.minimum(min_dist, d)
            min_dist[selected] = -np.inf
            selected.append(int(np.argmax(min_dist)))
        return selected

    @staticmethod
    def _cluster(coords: np.ndarray, n_target: int) -> List[int]:
        seed = random.randint(0, len(coords) - 1)
        d = np.linalg.norm(coords - coords[seed], axis=1)
        return np.argsort(d)[:n_target].tolist()

    @staticmethod
    def _boundary(coords: np.ndarray, n_target: int) -> List[int]:
        # can be slower; bio_rate is low so ok
        try:
            hull = ConvexHull(coords)
            boundary_idx = np.unique(hull.simplices.flatten())
            if len(boundary_idx) >= n_target:
                return np.random.choice(boundary_idx, n_target, replace=False).tolist()
            interior = [i for i in range(len(coords)) if i not in boundary_idx]
            need = n_target - len(boundary_idx)
            if need <= 0:
                return boundary_idx[:n_target].tolist()
            return np.concatenate([
                boundary_idx,
                np.random.choice(interior, need, replace=False)
            ]).tolist()
        except Exception:
            return random.sample(range(len(coords)), n_target)

    @staticmethod
    def _polar(coords: np.ndarray, n_target: int) -> List[int]:
        centered = coords - coords.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        proj = centered @ vt[0]
        order = np.argsort(proj)
        step = max(1, len(coords) // n_target)
        pick = [order[i * step] for i in range(n_target)]
        if len(pick) < n_target:
            rem = [i for i in range(len(coords)) if i not in pick]
            pick += random.sample(rem, n_target - len(pick))
        return pick[:n_target]


# =============================================================================
# SPARSE GEOMETRIC FEATURES (v2.4 - RELATIONAL + TOTAL CELLS)
# =============================================================================

class SparsePointFeatures(nn.Module):
    """
    Relational geometric encoding for sparse point clouds.
    
    v2.4: Biologically-grounded features focused on RELATIONAL information.
    
    Key insight from user: We can't assume anything about WHERE in the embryo 
    the subset came from (cluster vs spread?). All features must be purely 
    relational - how cells relate to EACH OTHER.
    
    Biology: Sibling cells (hardest errors) are spatially close and in similar
    local environments. Features should capture this.
    
    Features (computed efficiently from single distance matrix):
    1. Relative position to centroid      (3 -> 20)  - local spatial structure
    2. Centroid distance (normalized)     (1 -> 16)  - how peripheral in subset
    3. Subset size embedding              (N -> 16)  - context about observation
    4. Total cells embedding              (T -> 24)  - CRUCIAL developmental stage
    5. Pairwise relational features       (5 -> 52)  - from distance matrix:
       - Local density (k-NN mean)        - crowding around this cell
       - Mean dist to all others          - overall centrality  
       - Min dist (nearest neighbor)      - KEY for siblings!
       - Std of distances                 - uniformity of surroundings
       - Centrality rank (0-1)            - most central to most peripheral
    
    Total: 128 dims
    
    CPU-optimized: Distance matrix computed ONCE, all features derived from it.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        # Feature dimensions (sum to embed_dim=128)
        d_rel = 20       # relative position
        d_cdist = 16     # centroid distance  
        d_count = 16     # subset size
        d_total = 24     # total cells (most important!)
        d_pairwise = 52  # all pairwise relational features
        
        assert d_rel + d_cdist + d_count + d_total + d_pairwise == embed_dim, \
            f"Feature dims must sum to {embed_dim}"
        
        self.rel = nn.Linear(3, d_rel)
        self.cdist = nn.Linear(1, d_cdist)
        self.count = nn.Embedding(64, d_count)      # subset size: 0-63
        self.total = nn.Embedding(256, d_total)     # total cells: 0-255
        self.pairwise = nn.Linear(5, d_pairwise)    # 5 relational stats

    def forward(
        self, 
        points: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        total_cells: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            points: [B, N, 3] - normalized coordinates
            mask: [B, N] - 1 for valid, 0 for padding
            total_cells: [B] - total cells in embryo (5-194)
            
        Returns:
            features: [B, N, embed_dim]
        """
        B, N, _ = points.shape
        device = points.device

        if mask is not None:
            m = mask.float().unsqueeze(-1)  # [B,N,1]
            pts = points * m
            n_valid = mask.sum(dim=1, keepdim=True).clamp(min=1).float()  # [B,1]
            centroid = pts.sum(dim=1, keepdim=True) / n_valid.unsqueeze(-1)  # [B,1,3]
        else:
            centroid = points.mean(dim=1, keepdim=True)
            n_valid = torch.full((B, 1), float(N), device=device)
            m = torch.ones(B, N, 1, device=device)

        # === 1. Relative position ===
        rel = points - centroid
        f_rel = self.rel(rel)
        
        # === 2. Centroid distance (normalized by max in subset) ===
        cdist = torch.norm(rel, dim=-1, keepdim=True)  # [B,N,1]
        cdist_max = (cdist * m).reshape(B, -1).max(dim=1, keepdim=True)[0].unsqueeze(-1).clamp(min=1e-6)
        cdist_norm = cdist / cdist_max
        f_cdist = self.cdist(cdist_norm)

        # === 3. Subset size (observed cells) ===
        n_idx = n_valid.clamp(max=63).long().squeeze(1)  # [B]
        f_count = self.count(n_idx).unsqueeze(1).expand(-1, N, -1)

        # === 4. Total cells (developmental stage) ===
        if total_cells is not None:
            total_idx = total_cells.clamp(min=0, max=255).long()  # [B]
        else:
            total_idx = n_idx  # Fallback
        f_total = self.total(total_idx).unsqueeze(1).expand(-1, N, -1)

        # === 5. Pairwise relational features (single dist matrix computation) ===
        pairwise_vals = self._compute_pairwise_features(points, mask)
        f_pairwise = self.pairwise(pairwise_vals)

        return torch.cat([f_rel, f_cdist, f_count, f_total, f_pairwise], dim=-1)

    def _compute_pairwise_features(
        self, 
        points: torch.Tensor, 
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute ALL relational features from a single distance matrix.
        
        Returns:
            features: [B, N, 5] containing:
                [0] local_density - mean k-NN distance (normalized)
                [1] mean_dist - mean distance to all others (centrality)
                [2] min_dist - nearest neighbor distance (KEY for siblings!)
                [3] std_dist - std of distances to others (uniformity)
                [4] rank - centrality percentile (0=most central, 1=most peripheral)
        """
        B, N, _ = points.shape
        device = points.device
        
        out = torch.zeros(B, N, 5, device=device, dtype=points.dtype)
        
        for b in range(B):
            if mask is not None:
                valid = mask[b].bool()
                n = int(valid.sum().item())
            else:
                valid = torch.ones(N, device=device, dtype=torch.bool)
                n = N
            
            if n <= 1:
                continue
            
            pts = points[b, valid]  # [n, 3]
            
            # === Single distance matrix computation (CPU-optimized) ===
            # ||a-b||² = ||a||² + ||b||² - 2*a·b
            pts_sq = (pts ** 2).sum(dim=1, keepdim=True)  # [n,1]
            d_sq = pts_sq + pts_sq.T - 2 * torch.mm(pts, pts.T)  # [n,n]
            dist_mat = torch.sqrt(d_sq.clamp(min=0))  # [n,n]
            
            # Normalize by max distance for scale invariance
            max_dist = dist_mat.max().clamp(min=1e-6)
            
            # Mask for excluding self-distances
            eye = torch.eye(n, device=device, dtype=torch.bool)
            dist_no_self = dist_mat.masked_fill(eye, float('inf'))
            
            # [0] Local density: mean of k=3 nearest neighbors
            k = min(3, n - 1)
            knn_dists, _ = torch.topk(dist_no_self, k=k, largest=False, dim=1)
            local_dens = knn_dists.mean(dim=1) / max_dist  # [n]
            
            # [1] Mean distance to all others (centrality)
            # Lower = more central in the point cloud
            mean_d = dist_mat.sum(dim=1) / (n - 1) / max_dist  # [n]
            
            # [2] Min distance (nearest neighbor) - KEY for distinguishing siblings
            min_d = dist_no_self.min(dim=1)[0] / max_dist  # [n]
            
            # [3] Std of distances (uniformity of surroundings)
            # Mask self, compute std per row
            std_d = torch.zeros(n, device=device)
            for i in range(n):
                others = dist_mat[i, ~eye[i]]
                std_d[i] = others.std() if n > 2 else 0.0
            std_d = std_d / max_dist
            
            # [4] Centrality rank (0 = most central, 1 = most peripheral)
            # Based on mean distance
            if n > 1:
                ranks = mean_d.argsort().argsort().float()  # double argsort = rank
                rank = ranks / (n - 1)
            else:
                rank = torch.zeros(n, device=device)
            
            # Store results
            out[b, valid, 0] = local_dens
            out[b, valid, 1] = mean_d
            out[b, valid, 2] = min_d
            out[b, valid, 3] = std_d
            out[b, valid, 4] = rank
        
        return out



# =============================================================================
# MODEL (TWIN ATTENTION - v2.4 WITH TOTAL CELLS)
# =============================================================================

class SparseTwinAttentionEncoder(nn.Module):
    """
    Twin Attention style: joint transformer over [REF || QUERY].
    
    v2.4 CHANGE: forward() now accepts total_cells_ref and total_cells_query
    to provide developmental stage context to the feature encoder.
    
    API:
        forward(ref_points, query_points, ref_mask, query_mask, 
                total_cells_ref=None, total_cells_query=None, epoch=0)
    """
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        max_seq_len: int = 64,
        use_sparse_features: bool = True,
        use_uncertainty: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_sparse_features = use_sparse_features
        self.use_uncertainty = use_uncertainty

        if use_sparse_features:
            self.feat = SparsePointFeatures(embed_dim)
            self.proj = nn.Linear(embed_dim, embed_dim)
        else:
            self.proj = nn.Linear(3, embed_dim)

        # positional + token type embeddings (ref vs query)
        self.pos = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)
        self.type_emb = nn.Embedding(2, embed_dim)  # 0=ref, 1=query

        # learnable no-match token (appended to REF)
        self.no_match = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        if use_uncertainty:
            self.out_mean = nn.Linear(embed_dim, embed_dim)
            self.out_logvar = nn.Linear(embed_dim, embed_dim)
        else:
            self.out = nn.Linear(embed_dim, embed_dim)

        self.log_temperature = nn.Parameter(torch.tensor(0.0))

    def encode_tokens(
        self, 
        pts: torch.Tensor, 
        mask: Optional[torch.Tensor],
        total_cells: Optional[torch.Tensor] = None,  # NEW in v2.4
    ) -> torch.Tensor:
        """Encode point cloud to tokens with optional total_cells context."""
        if self.use_sparse_features:
            z = self.feat(pts, mask, total_cells)  # Pass total_cells to features
            z = self.proj(z)
        else:
            z = self.proj(pts)
        return z

    def forward(
        self,
        ref_points: torch.Tensor,
        query_points: torch.Tensor,
        ref_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
        total_cells_ref: Optional[torch.Tensor] = None,    # NEW in v2.4
        total_cells_query: Optional[torch.Tensor] = None,  # NEW in v2.4
        epoch: int = 0
    ):
        """
        Args:
            ref_points: [B, Nr, 3]
            query_points: [B, Nq, 3]
            ref_mask: [B, Nr]
            query_mask: [B, Nq]
            total_cells_ref: [B] - total cells in ref embryo (NEW)
            total_cells_query: [B] - total cells in query embryo (NEW)
            epoch: current training epoch
            
        Returns:
            ref_out: embeddings for ref + no-match token
            query_out: embeddings for query cells
            temperature: learned temperature scalar
        """
        B, Nr, _ = ref_points.shape
        _, Nq, _ = query_points.shape

        # Encode with total_cells context (v2.4 change)
        zr = self.encode_tokens(ref_points, ref_mask, total_cells_ref)
        zq = self.encode_tokens(query_points, query_mask, total_cells_query)

        # append no-match to ref side
        nm = self.no_match.expand(B, -1, -1)  # [B,1,D]
        zr = torch.cat([zr, nm], dim=1)  # [B,Nr+1,D]

        # build joint sequence: [REF(with no-match) || QUERY]
        z = torch.cat([zr, zq], dim=1)  # [B, (Nr+1+Nq), D]
        seq_len = z.shape[1]
        if seq_len > self.pos.shape[1]:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.pos.shape[1]}")

        # token types: ref=0 for first (Nr+1), query=1 for rest
        type_ids = torch.cat([
            torch.zeros(B, Nr + 1, device=z.device, dtype=torch.long),
            torch.ones(B, Nq, device=z.device, dtype=torch.long)
        ], dim=1)
        z = z + self.pos[:, :seq_len, :] + self.type_emb(type_ids)

        # key padding mask: True = pad positions to ignore
        if ref_mask is None:
            ref_mask = torch.ones(B, Nr, device=z.device)
        if query_mask is None:
            query_mask = torch.ones(B, Nq, device=z.device)

        # ref_mask does NOT include no-match; add it as valid
        ref_mask_nm = torch.cat([ref_mask.float(), torch.ones(B, 1, device=z.device)], dim=1)  # [B,Nr+1]
        joint_valid = torch.cat([ref_mask_nm, query_mask.float()], dim=1)  # [B, seq]
        key_padding = (joint_valid < 0.5)  # True = ignore

        z = self.tr(z, src_key_padding_mask=key_padding)

        zr_out = z[:, :Nr + 1, :]
        zq_out = z[:, Nr + 1:, :]

        if self.use_uncertainty:
            r_mean = F.normalize(self.out_mean(zr_out), p=2, dim=-1)
            q_mean = F.normalize(self.out_mean(zq_out), p=2, dim=-1)
            r_logv = torch.clamp(self.out_logvar(zr_out), -10, 2)
            q_logv = torch.clamp(self.out_logvar(zq_out), -10, 2)
            ref = (r_mean, r_logv)
            query = (q_mean, q_logv)
        else:
            ref = F.normalize(self.out(zr_out), p=2, dim=-1)
            query = F.normalize(self.out(zq_out), p=2, dim=-1)

        temp = torch.exp(self.log_temperature).clamp(0.01, 10.0)
        return ref, query, temp


# =============================================================================
# DATASET (v2.4 - RETURNS TOTAL CELLS)
# =============================================================================

class SparsePairDataset(Dataset):
    """
    Produces (ref_pc, query_pc, ref_mask, query_mask, match_indices, temporal_pairs, info)

    v2.4 CHANGE: info dict now includes 'total_cells_ref' and 'total_cells_query'
    for passing developmental stage context to the model.
    
    match_indices are for QUERY cells:
      - values in [0, Nr-1] -> matched ref index
      - value == Nr          -> no-match (points to ref no-match token)
    """

    STAGE_BINS = {
        "early": (5, 50),
        "mid": (51, 100),
        "late": (101, 194),
    }

    def __init__(
        self,
        data_dict: Dict,
        cfg: Config,
        augment: bool = True,
        curriculum_stage: int = 0,
    ):
        super().__init__()
        self.data = data_dict
        self.cfg = cfg
        self.augment = augment
        self.curriculum_stage = curriculum_stage

        self.stage_data = self._organize_by_stage()
        self.pairs = self._generate_pairs()

        print("Dataset created:")
        for s in ["early", "mid", "late"]:
            print(f"  {s}: {len(self.stage_data[s])} timepoints")
        print(f"  Total pairs: {len(self.pairs)}")
        print(f"  bio_rate={cfg.bio_rate}  force_random={cfg.force_random}  intra_prob={cfg.intra_prob}")

    def update_curriculum(self, stage: int):
        self.curriculum_stage = stage
        self.pairs = self._generate_pairs()
        print(f"Updated curriculum -> stage {stage}")

    def _organize_by_stage(self) -> Dict[str, List[Dict]]:
        out = {s: [] for s in self.STAGE_BINS}
        for run, tps in self.data.items():
            for t, cells in sorted(tps.items()):
                n = len(cells)
                if n < self.cfg.min_cells or n > self.cfg.stage_limit:
                    continue
                stage = None
                for name, (lo, hi) in self.STAGE_BINS.items():
                    if lo <= n <= hi:
                        stage = name
                        break
                if stage is None:
                    continue
                out[stage].append({
                    "run": run,
                    "time": int(t),
                    "cells": cells,
                    "n_cells": n,
                    "stage": stage,
                })
        return out

    def _generate_pairs(self) -> List[Dict]:
        pairs = []
        stage_mult = {"early": 1, "mid": 2, "late": 3}
        min_shared_ratio, time_window = self.cfg.curriculum.get(
            self.curriculum_stage, self.cfg.curriculum[3]
        )
        self._min_shared_ratio = float(min_shared_ratio)
        self._time_window = int(time_window)

        for stage_name, items in self.stage_data.items():
            if len(items) < 2:
                continue
            target = int(self.cfg.pairs_per_stage * stage_mult[stage_name])
            got = 0
            attempts = 0
            max_attempts = target * 20

            while got < target and attempts < max_attempts:
                attempts += 1
                anchor = random.choice(items)
                if random.random() < self.cfg.intra_prob:
                    pair = self._make_intra(anchor, items)
                else:
                    pair = self._make_inter(anchor, items)
                if pair is not None:
                    pair["stage"] = stage_name
                    pairs.append(pair)
                    got += 1

            print(f"  Generated {got} pairs for {stage_name}")

        random.shuffle(pairs)
        return pairs

    def _make_intra(self, anchor: Dict, items: List[Dict]) -> Optional[Dict]:
        same = [it for it in items if it["run"] == anchor["run"] and it["time"] != anchor["time"]]
        if not same:
            return None
        nearby = [it for it in same if abs(it["time"] - anchor["time"]) <= self._time_window]
        if not nearby:
            nearby = same
        comp = random.choice(nearby)
        is_adjacent = (abs(comp["time"] - anchor["time"]) == 1)
        return {"anchor": anchor, "comp": comp, "pair_type": "intra", "is_adjacent": is_adjacent}

    def _make_inter(self, anchor: Dict, items: List[Dict]) -> Optional[Dict]:
        other = []
        for s, lst in self.stage_data.items():
            other.extend([it for it in lst if it["run"] != anchor["run"]])
        if not other:
            return None

        if random.random() < self.cfg.INTER_STAGE_MATCH_PROB:
            tol = max(10, int(anchor["n_cells"] * 0.2))
            cand = [it for it in other if abs(it["n_cells"] - anchor["n_cells"]) <= tol]
            if cand:
                comp = random.choice(cand)
            else:
                comp = random.choice(other)
        else:
            comp = random.choice(other)

        return {"anchor": anchor, "comp": comp, "pair_type": "inter", "is_adjacent": False}

    def _augment(self, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if a.size == 0 or b.size == 0:
            return a, b
        noise_scale = [0.0, 0.01, 0.02, 0.03][min(self.curriculum_stage, 3)]
        rot_max = [np.pi/36, np.pi/24, np.pi/18, np.pi/12][min(self.curriculum_stage, 3)]

        if random.random() < 0.5:
            angle = random.uniform(-rot_max, rot_max)
            axis = np.random.randn(3)
            axis = axis / (np.linalg.norm(axis) + 1e-8)
            rot = R.from_rotvec(angle * axis).as_matrix()
            a = (a - a.mean(axis=0)) @ rot.T + a.mean(axis=0)
            b = (b - b.mean(axis=0)) @ rot.T + b.mean(axis=0)

        if noise_scale > 0:
            a = a + np.random.normal(0, noise_scale, a.shape)
            b = b + np.random.normal(0, noise_scale, b.shape)

        return a, b

    def _sample_pair(self, cells_ref: Dict, cells_query: Dict) -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
        shared = list(set(cells_ref.keys()) & set(cells_query.keys()))
        unique_r = list(set(cells_ref.keys()) - set(cells_query.keys()))
        unique_q = list(set(cells_query.keys()) - set(cells_ref.keys()))

        n_target = random.randint(self.cfg.min_cells, self.cfg.max_cells)

        min_shared = max(2, int(n_target * self._min_shared_ratio))
        max_shared = min(len(shared), n_target - 1)
        if max_shared < min_shared:
            n_shared = min(len(shared), n_target - 1)
        else:
            n_shared = random.randint(min_shared, max_shared)

        if len(shared) <= n_shared:
            pick_shared = shared
        else:
            pick_shared = random.sample(shared, n_shared)

        n_r = n_target - len(pick_shared)
        n_q = n_target - len(pick_shared)

        if self.augment and (not self.cfg.force_random) and (random.random() < self.cfg.bio_rate):
            strat = random.choice(["diverse", "cluster", "polar", "boundary"])
            if unique_r and n_r > 0:
                uniq_dict = {cid: cells_ref[cid] for cid in unique_r}
                pick_unique_r = BiologicalSampler.sample_cells(uniq_dict, min(n_r, len(unique_r)), strat)
            else:
                pick_unique_r = []
        else:
            pick_unique_r = random.sample(unique_r, min(n_r, len(unique_r))) if unique_r and n_r > 0 else []

        pick_unique_q = random.sample(unique_q, min(n_q, len(unique_q))) if unique_q and n_q > 0 else []

        ref_ids = list(pick_shared) + list(pick_unique_r)
        query_ids = list(pick_shared) + list(pick_unique_q)

        ref_raw = np.array([cells_ref[cid] for cid in ref_ids])
        qry_raw = np.array([cells_query[cid] for cid in query_ids])

        ridx = list(range(len(ref_ids)))
        qidx = list(range(len(query_ids)))
        random.shuffle(ridx)
        random.shuffle(qidx)
        ref_ids = [ref_ids[i] for i in ridx]
        query_ids = [query_ids[i] for i in qidx]
        ref_raw = ref_raw[ridx]
        qry_raw = qry_raw[qidx]

        ref = normalize_coords(ref_raw)
        qry = normalize_coords(qry_raw)

        return ref_ids, ref, query_ids, qry

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        pair = self.pairs[idx % len(self.pairs)]
        a = pair["anchor"]
        c = pair["comp"]

        # IMPORTANT: REF = comparison, QUERY = anchor (twin attention uses ref as "atlas")
        ref_ids, ref_xyz, query_ids, query_xyz = self._sample_pair(c["cells"], a["cells"])

        if self.augment:
            ref_xyz, query_xyz = self._augment(ref_xyz, query_xyz)

        match = []
        ref_index = {cid: j for j, cid in enumerate(ref_ids)}
        for cid in query_ids:
            match.append(ref_index.get(cid, -1))
        match = np.array(match, dtype=np.int64)

        temporal_pairs = []
        if pair["pair_type"] == "intra" and pair["is_adjacent"]:
            shared_ids = [cid for cid in query_ids if cid in ref_index]
            for cid in shared_ids:
                temporal_pairs.append((query_ids.index(cid), ref_index[cid]))

        # v2.4 CHANGE: Include total_cells for both ref and query embryos
        info = {
            "pair_type": pair["pair_type"],
            "stage": pair["stage"],
            "run_query": a["run"],
            "time_query": a["time"],
            "run_ref": c["run"],
            "time_ref": c["time"],
            "n_shared": int(len(set(query_ids) & set(ref_ids))),
            "total_cells_ref": c["n_cells"],      # NEW: total cells in ref embryo
            "total_cells_query": a["n_cells"],    # NEW: total cells in query embryo
        }

        return (
            torch.from_numpy(ref_xyz).float(),
            torch.from_numpy(query_xyz).float(),
            torch.from_numpy(match).long(),
            temporal_pairs,
            info,
            ref_ids,
            query_ids,
        )


# =============================================================================
# COLLATE (v2.4 - EXTRACTS TOTAL CELLS)
# =============================================================================

def collate_fn(batch):
    """
    v2.4 CHANGE: Also extracts and batches total_cells_ref and total_cells_query
    from info dicts into tensors.
    """
    ref_list, qry_list, match_list, temporal_list, info_list, ref_ids_list, qry_ids_list = zip(*batch)

    B = len(ref_list)
    max_r = max(x.shape[0] for x in ref_list)
    max_q = max(x.shape[0] for x in qry_list)

    ref_pad = torch.zeros(B, max_r, 3)
    qry_pad = torch.zeros(B, max_q, 3)
    ref_mask = torch.zeros(B, max_r)
    qry_mask = torch.zeros(B, max_q)

    match_pad = torch.full((B, max_q), fill_value=max_r, dtype=torch.long)
    temporal_pairs = []

    # v2.4: Extract total_cells into tensors
    total_cells_ref = torch.zeros(B, dtype=torch.long)
    total_cells_query = torch.zeros(B, dtype=torch.long)

    for b in range(B):
        r = ref_list[b]
        q = qry_list[b]
        m = match_list[b]

        nr = r.shape[0]
        nq = q.shape[0]

        ref_pad[b, :nr] = r
        qry_pad[b, :nq] = q
        ref_mask[b, :nr] = 1.0
        qry_mask[b, :nq] = 1.0

        if torch.is_tensor(m):
            m_fixed = m.clone().long()
        else:
            m_fixed = torch.tensor(m, dtype=torch.long)

        m_fixed[m_fixed < 0] = max_r
        match_pad[b, :nq] = m_fixed[:nq]

        for (q_idx, r_idx) in temporal_list[b]:
            if q_idx < max_q and r_idx < max_r:
                temporal_pairs.append((b, q_idx, r_idx))

        # v2.4: Extract total_cells from info
        total_cells_ref[b] = info_list[b].get("total_cells_ref", nr)
        total_cells_query[b] = info_list[b].get("total_cells_query", nq)

    return ref_pad, qry_pad, ref_mask, qry_mask, match_pad, temporal_pairs, info_list, total_cells_ref, total_cells_query



# =============================================================================
# LOSS (UNCHANGED FROM v2.3)
# =============================================================================

class MatchingLoss(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    @staticmethod
    def _safe_mean(vals: List[torch.Tensor], device):
        return torch.stack(vals).mean() if len(vals) > 0 else torch.tensor(0.0, device=device)

    def forward(
        self,
        ref_out,
        qry_out,
        temperature: torch.Tensor,
        match_indices: torch.Tensor,
        ref_mask: torch.Tensor,
        qry_mask: torch.Tensor,
        temporal_pairs: List[Tuple[int, int, int]],
        coords_ref: Optional[torch.Tensor] = None,
        epoch: int = 0
    ):
        if isinstance(ref_out, tuple):
            ref_mean, _ = ref_out
            qry_mean, _ = qry_out
        else:
            ref_mean = ref_out
            qry_mean = qry_out

        B, R_pad = ref_mask.shape
        _, Q_pad = qry_mask.shape

        sim = torch.matmul(qry_mean, ref_mean.transpose(-1, -2)) / temperature

        ref_valid = torch.cat(
            [ref_mask.float(), torch.ones(B, 1, device=sim.device)], dim=1
        )

        sim = sim.masked_fill(ref_valid.unsqueeze(1) < 0.5, -1e9)

        losses = []
        correct = 0
        total = 0

        ramp_end = int(self.cfg.num_epochs * 0.70)
        t = min(1.0, epoch / max(1, ramp_end))
        hard_w = self.cfg.hard_neg_base_weight + t * (self.cfg.hard_neg_max_weight - self.cfg.hard_neg_base_weight)
        hard_k = int(round(self.cfg.hard_neg_base_k + t * (self.cfg.hard_neg_max_k - self.cfg.hard_neg_base_k)))
        hard_k = max(0, min(hard_k, 6))

        hard_neg_mask = None
        if (coords_ref is not None) and (hard_k > 0):
            hard_neg_mask = self._find_hard_negatives(coords_ref, match_indices, ref_mask, hard_k)

        for b in range(B):
            for i in range(Q_pad):
                if qry_mask[b, i] < 0.5:
                    continue

                y = match_indices[b, i].item()
                logp = F.log_softmax(sim[b, i], dim=0)

                losses.append(-logp[y])
                total += 1
                if int(torch.argmax(sim[b, i]).item()) == int(y):
                    correct += 1

                if hard_neg_mask is not None and y < R_pad:
                    hn_idx = hard_neg_mask[b, i].nonzero(as_tuple=True)[0]
                    for j in hn_idx:
                        j = int(j.item())
                        if j == y:
                            continue
                        h = torch.clamp(logp[j] - logp[y] + self.cfg.hard_neg_margin, min=0.0)
                        losses.append(hard_w * h)

        match_loss = self._safe_mean(losses, sim.device)

        temporal_terms = []
        if self.cfg.temporal_weight > 0 and temporal_pairs:
            for (b, q_idx, r_idx) in temporal_pairs:
                if qry_mask[b, q_idx] < 0.5:
                    continue
                if ref_mask[b, r_idx] < 0.5:
                    continue
                qv = qry_mean[b, q_idx]
                rv = ref_mean[b, r_idx]
                cs = F.cosine_similarity(qv.unsqueeze(0), rv.unsqueeze(0), dim=1)
                temporal_terms.append(1.0 - cs.squeeze(0))
        temporal_loss = self.cfg.temporal_weight * self._safe_mean(temporal_terms, sim.device)

        total_loss = match_loss + temporal_loss

        metrics = {
            "loss": float(total_loss.detach().cpu().item()),
            "match_loss": float(match_loss.detach().cpu().item()),
            "temporal_loss": float(temporal_loss.detach().cpu().item()),
            "acc": correct / total if total > 0 else 0.0,
            "hard_w": float(hard_w),
            "hard_k": int(hard_k),
            "temp": float(temperature.detach().cpu().item()),
        }
        return total_loss, metrics

    @staticmethod
    def _find_hard_negatives(coords_ref: torch.Tensor, match_indices: torch.Tensor, ref_mask: torch.Tensor, k: int):
        B, R, _ = coords_ref.shape
        _, Q = match_indices.shape
        out = torch.zeros(B, Q, R, dtype=torch.bool, device=coords_ref.device)

        for b in range(B):
            valid_ref = ref_mask[b].bool()
            ref_pts = coords_ref[b]
            for i in range(Q):
                y = int(match_indices[b, i].item())
                if y >= R:
                    continue
                if not valid_ref[y]:
                    continue

                center = ref_pts[y]
                d = torch.norm(ref_pts - center, dim=-1)
                d[~valid_ref] = float("inf")
                d[y] = float("inf")

                kk = min(k, int((d < float("inf")).sum().item()))
                if kk <= 0:
                    continue
                idx = torch.topk(d, kk, largest=False).indices
                out[b, i, idx] = True

        return out


# =============================================================================
# TRAINER (v2.4 - PASSES TOTAL CELLS TO MODEL)
# =============================================================================

class Trainer:
    def __init__(self, cfg: Config, model: nn.Module, train_ds: Dataset, val_ds: Dataset):
        self.cfg = cfg
        self.model = model.to(device)
        self.train_ds = train_ds
        self.val_ds = val_ds

        os.makedirs(cfg.ckpt_dir, exist_ok=True)

        self.train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.999),
        )

        total_steps = len(self.train_loader) * cfg.num_epochs
        self.sched = torch.optim.lr_scheduler.OneCycleLR(
            self.opt,
            max_lr=cfg.lr,
            total_steps=total_steps,
            pct_start=0.10,
            anneal_strategy="cos",
        )

        self.loss_fn = MatchingLoss(cfg)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

        self.best_val = 0.0
        self.history = defaultdict(list)

    def save_ckpt(self, epoch: int, val_acc: float, is_best: bool):
        name = "best_model.pth" if is_best else f"ckpt_epoch_{epoch}.pth"
        path = os.path.join(self.cfg.ckpt_dir, name)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.opt.state_dict(),
            "scheduler_state_dict": self.sched.state_dict(),
            "best_val": self.best_val,
            "cfg": self.cfg.__dict__,
            "history": dict(self.history),
        }, path)
        if is_best:
            print(f"  ✓ Saved best: {path}")

    @torch.no_grad()
    def validate(self, epoch: int):
        self.model.eval()
        correct = 0
        total = 0
        stage_stats = {s: {"c": 0, "t": 0} for s in ["early", "mid", "late"]}

        # v2.4: Unpack total_cells from collate_fn
        for ref, qry, ref_m, qry_m, match, temporal_pairs, infos, tc_ref, tc_qry in self.val_loader:
            ref = ref.to(device)
            qry = qry.to(device)
            ref_m = ref_m.to(device)
            qry_m = qry_m.to(device)
            match = match.to(device)
            tc_ref = tc_ref.to(device)    # v2.4
            tc_qry = tc_qry.to(device)    # v2.4

            # v2.4: Pass total_cells to model
            ref_out, qry_out, temp = self.model(ref, qry, ref_m, qry_m, tc_ref, tc_qry, epoch=epoch)
            
            if isinstance(ref_out, tuple):
                ref_mean = ref_out[0]
                qry_mean = qry_out[0]
            else:
                ref_mean = ref_out
                qry_mean = qry_out

            sim = torch.matmul(qry_mean, ref_mean.transpose(-1, -2)) / temp

            ref_valid = torch.cat(
                [ref_m.float(), torch.ones(ref_m.shape[0], 1, device=ref_m.device)], dim=1
            )
            sim = sim.masked_fill(ref_valid.unsqueeze(1) < 0.5, -1e9)

            pred = sim.argmax(dim=-1)

            B, Q = pred.shape
            for b in range(B):
                stage = infos[b].get("stage", "mid")
                n_valid = int(qry_m[b].sum().item())
                for i in range(n_valid):
                    if int(pred[b, i].item()) == int(match[b, i].item()):
                        correct += 1
                        stage_stats[stage]["c"] += 1
                    stage_stats[stage]["t"] += 1
                    total += 1

        overall = correct / total if total > 0 else 0.0
        out = {"overall_acc": overall}
        for s in ["early", "mid", "late"]:
            t = stage_stats[s]["t"]
            out[f"{s}_acc"] = stage_stats[s]["c"] / t if t > 0 else 0.0
        return out

    def train(self):
        print("\n" + "=" * 80)
        print("TRAINING: Sparse Twin Attention v2.4 (with total_cells)")
        print("=" * 80)
        patience = 0

        for epoch in range(self.cfg.num_epochs):
            # curriculum update
            if epoch in self.cfg.curriculum_schedule:
                st = self.cfg.curriculum_schedule[epoch]
                self.train_ds.update_curriculum(st)
                self.val_ds.update_curriculum(st)
                patience = 0

            self.model.train()
            run_loss = 0.0
            run_match = 0.0
            run_temp = 0.0
            run_acc = 0.0
            n_batches = 0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.cfg.num_epochs}")

            # v2.4: Unpack total_cells from collate_fn
            for ref, qry, ref_m, qry_m, match, temporal_pairs, infos, tc_ref, tc_qry in pbar:
                ref = ref.to(device)
                qry = qry.to(device)
                ref_m = ref_m.to(device)
                qry_m = qry_m.to(device)
                match = match.to(device)
                tc_ref = tc_ref.to(device)    # v2.4
                tc_qry = tc_qry.to(device)    # v2.4

                self.opt.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    # v2.4: Pass total_cells to model
                    ref_out, qry_out, temp = self.model(ref, qry, ref_m, qry_m, tc_ref, tc_qry, epoch=epoch)
                    loss, metrics = self.loss_fn(
                        ref_out, qry_out, temp,
                        match,
                        ref_m, qry_m,
                        temporal_pairs=temporal_pairs,
                        coords_ref=ref,
                        epoch=epoch,
                    )

                if torch.isnan(loss) or torch.isinf(loss):
                    raise RuntimeError("Loss is NaN/Inf — check data/normalization.")

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.opt)
                self.scaler.update()
                self.sched.step()

                lr = self.opt.param_groups[0]["lr"]

                run_loss += metrics["loss"]
                run_match += metrics["match_loss"]
                run_temp += metrics["temporal_loss"]
                run_acc += metrics["acc"]
                n_batches += 1

                pbar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "acc": f"{metrics['acc']:.3f}",
                    "hn_w": f"{metrics['hard_w']:.2f}",
                    "hn_k": f"{metrics['hard_k']}",
                    "t": f"{metrics['temp']:.2f}",
                    "lr": f"{lr:.1e}",
                })

            train_metrics = {
                "train_loss": run_loss / max(1, n_batches),
                "train_match_loss": run_match / max(1, n_batches),
                "train_temporal_loss": run_temp / max(1, n_batches),
                "train_acc": run_acc / max(1, n_batches),
                "lr": self.opt.param_groups[0]["lr"],
            }

            val_metrics = self.validate(epoch)

            for k, v in {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}.items():
                self.history[k].append(v)

            print(
                f"\nEpoch {epoch+1} | "
                f"Train loss {train_metrics['train_loss']:.4f} "
                f"(match {train_metrics['train_match_loss']:.4f}, temp {train_metrics['train_temporal_loss']:.4f}) "
                f"acc {train_metrics['train_acc']:.3f} | "
                f"Val overall {val_metrics['overall_acc']:.3f} "
                f"E {val_metrics['early_acc']:.3f} M {val_metrics['mid_acc']:.3f} L {val_metrics['late_acc']:.3f}"
            )

            # model selection
            if val_metrics["overall_acc"] > self.best_val:
                self.best_val = val_metrics["overall_acc"]
                self.save_ckpt(epoch, self.best_val, is_best=True)
                patience = 0
            else:
                patience += 1

            if patience >= self.cfg.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

            if (epoch + 1) % self.cfg.save_every == 0:
                self.save_ckpt(epoch, val_metrics["overall_acc"], is_best=False)

        return dict(self.history)

def preflight_check(model, loader, loss_fn, steps=2):
    model.eval()
    with torch.no_grad():
        for k, batch in enumerate(loader):
            if k >= steps:
                break
            # v2.4: Unpack total_cells
            ref, qry, ref_m, qry_m, match, temporal_pairs, infos, tc_ref, tc_qry = batch
            ref = ref.to(device); qry = qry.to(device)
            ref_m = ref_m.to(device); qry_m = qry_m.to(device)
            match = match.to(device)
            tc_ref = tc_ref.to(device); tc_qry = tc_qry.to(device)

            # v2.4: Pass total_cells to model
            ref_out, qry_out, temp = model(ref, qry, ref_m, qry_m, tc_ref, tc_qry, epoch=0)
            loss, metrics = loss_fn(
                ref_out, qry_out, temp,
                match,
                ref_m, qry_m,
                temporal_pairs=temporal_pairs,
                coords_ref=ref,
                epoch=0,
            )
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError("Preflight: Loss is NaN/Inf.")
    print("✓ Preflight passed (forward + loss on a couple batches).")


# =============================================================================
# MAIN
# =============================================================================

def main():
    cfg = Config()

    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    with open(cfg.data_path, "rb") as f:
        data = pickle.load(f)

    embryo_ids = list(data.keys())
    random.shuffle(embryo_ids)
    n_train = int(cfg.train_split * len(embryo_ids))
    train_ids = embryo_ids[:n_train]
    val_ids = embryo_ids[n_train:]

    train_data = {k: data[k] for k in train_ids}
    val_data = {k: data[k] for k in val_ids}

    print(f"Embryos: {len(train_ids)} train, {len(val_ids)} val")
    print(f"min_cells={cfg.min_cells} max_cells={cfg.max_cells} pairs_per_stage={cfg.pairs_per_stage}")

    # Datasets
    train_ds = SparsePairDataset(train_data, cfg, augment=True, curriculum_stage=0)
    val_ds = SparsePairDataset(val_data, cfg, augment=False, curriculum_stage=0)

    # Model
    model = SparseTwinAttentionEncoder(
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        max_seq_len=cfg.max_seq_len,
        use_sparse_features=True,
        use_uncertainty=True,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n" + "=" * 80)
    print(f"MODEL: {n_params:,} trainable parameters")
    print("=" * 80)
    print("v2.4 CHANGE: Model now receives total_cells context for stage-aware embeddings")
    print("API: forward(ref_points, query_points, ref_mask, query_mask, total_cells_ref, total_cells_query, epoch)")

    trainer = Trainer(cfg, model, train_ds, val_ds)

    preflight_check(trainer.model, trainer.train_loader, trainer.loss_fn, steps=2)
    preflight_check(trainer.model, trainer.val_loader, trainer.loss_fn, steps=1)

    history = trainer.train()

    # Save final weights
    final_path = os.path.join(cfg.ckpt_dir, "final_weights.pth")
    torch.save(model.state_dict(), final_path)
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best val pairwise acc: {trainer.best_val:.4f}")
    print(f"Saved final weights: {final_path}")

    return history


if __name__ == "__main__":
    main()
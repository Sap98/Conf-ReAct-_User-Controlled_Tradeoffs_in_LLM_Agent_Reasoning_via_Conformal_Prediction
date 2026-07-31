"""
Conformal Predictor for WebShop action selection.

Approach: similarity-based conditional conformal prediction via the dual LP
(Gibbs et al. 2023, eq 4.2).  No quantile fallback — only dual LP is used.

The calibration pool has two sources:
  1. Val-split states from WebShop_ScoreFunc_Train_Data.csv  (loaded once at startup)
  2. States explored during live gameplay                     (added via commit_step)

Architecture differences from ALFWorld conformal_predictor.py:
  - No 'location' field → state_raw is [1536] not [2304]
  - Action logits are raw token log-probs (List[List[float]]) not pre-binned histograms
  - Uses ScoreFunctionWebShop instead of ScoreFunction
  - Only dual_lp method is supported (no quantile method)

At each game step:
  1. Compute state_raw = BERT(instruction) || mean(BERT(prev_actions))
  2. Rank all calibration states by L2 distance to state_raw_test
  3. Select the most similar ones (top-k or distance threshold)
  4. Solve the dual LP (Gibbs et al. 2023, eq 4.2) on the selected states:
       maximize  Σ_i η_i S_i + η_{n+1} S_test
       s.t.      −α ≤ η_i ≤ 1−α
                 Φ^T η = 0
     Φ is a low-dim projection of the selected state_raws
     (computed via SVD so d < n_sel, keeping LP feasible).
     Action is in the prediction set iff η_{n+1} < 1 − α.

Gameplay loop:
    pred_set, scores, n_sel = predictor.get_prediction_set(...)
    action = pick_action(pred_set)
    env.step(action)
    predictor.commit_step(action)   # adds this state+score to the pool
"""

import pickle
import numpy as np
import torch
from scipy.optimize import linprog
from typing import List, Dict, Optional, Tuple

from score_model_webshop import ScoreFunctionWebShop, BertEmbeddingCache, BERT_DIM


class ConformalPredictorWebShop:

    def __init__(
        self,
        score_model: ScoreFunctionWebShop,  # score function model
        bert_cache: BertEmbeddingCache,     # BERT model to use the precomputed embeddings
        alpha: float = 0.1,                 # alpha = (1-coverage), if alpha = 0.1, coverage is 90%
    ):
        self.score_model = score_model
        self.bert_cache  = bert_cache
        self.alpha       = alpha

        # Calibration pool — grows over time
        self.cal_scores:     Optional[np.ndarray] = None   # [n]        nonconformity scores of optimal actions
        self.cal_state_raws: Optional[np.ndarray] = None   # [n, 1536]  corresponding state embeddings

        # Optional human-readable metadata for each calibration state.
        # List of dicts with keys: instruction, prev_actions, optimal_action.
        # Populated externally (e.g. from calibration script after calibrate_from_records).
        # Online states added via commit_step won't have metadata entries.
        self.cal_metadata: Optional[List[Dict]] = None     # [n]  metadata per pool state

        # Pending state from the most recent get_prediction_set() call.
        # Consumed by commit_step().
        self._pending_state_raw:     Optional[np.ndarray]       = None   # current state you just made a prediction on
        self._pending_action_scores: Optional[Dict[str, float]] = None   # scores for all candidate actions in that state

    # ─── CALIBRATION SETUP ───────────────────────────────────────────────────────

    def calibrate_from_records(
        self,
        val_records: list,
        device: torch.device,
    ):
        """
        Populate the initial calibration pool from val-split records.
        Uses pre-processed tensors from train_score_webshop.build_records().
        """
        scores, state_raws = [], []

        for rec in val_records:
            state_raw    = rec['state_raw'].to(device)
            action_embs  = rec['action_embs'].to(device)
            softmax_bins = rec['softmax_bins'].to(device)
            optimal_idx  = rec['optimal_idx']

            nc = self.score_model.compute_nonconformity_scores(
                state_raw, action_embs, softmax_bins
            )  # [K]  nonconformity scores for all admissible actions

            scores.append(nc[optimal_idx].item())               # store score for the optimal action only
            state_raws.append(rec['state_raw'].cpu().numpy())   # store corresponding state embedding

        self.cal_scores     = np.array(scores,     dtype=np.float64)   # [S1, S2, ..., Sn]
        self.cal_state_raws = np.array(state_raws, dtype=np.float32)   # [n, 1536]

        print(
            f"[Calibration]  pool size={len(scores)}  α={self.alpha}"
            f"  full-pool marginal S*={self._full_pool_threshold():.4f}"
            f"  mean={self.cal_scores.mean():.4f}"
            f"  std={self.cal_scores.std():.4f}"
        )

    def commit_step(self, executed_action: str, metadata: dict = None):
        """
        Add the current state to the calibration pool using the nonconformity
        score of the action that was actually executed.

        Call this immediately after env.step(action) at every game step
        (after the warm-up period ends).

        Args:
            executed_action: the action that was executed.
            metadata: optional dict with keys instruction, prev_actions,
                      optimal_action — appended to cal_metadata if provided.
        """
        assert self._pending_state_raw is not None, \
            "commit_step() called before get_prediction_set()."
        assert executed_action in self._pending_action_scores, \
            f"executed_action '{executed_action}' not found in last action scores."

        score = self._pending_action_scores[executed_action]

        self.cal_scores     = np.append(self.cal_scores, float(score))
        self.cal_state_raws = np.vstack([
            self.cal_state_raws,
            self._pending_state_raw.reshape(1, -1),
        ])

        if self.cal_metadata is not None:
            self.cal_metadata.append(metadata if metadata is not None else {
                'instruction':    '',
                'prev_actions':   [],
                'optimal_action': executed_action,
            })

        # Clear pending state
        self._pending_state_raw     = None
        self._pending_action_scores = None

    def save(self, path: str):
        """Store the current calibration pool into a file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'cal_scores':     self.cal_scores,
                'cal_state_raws': self.cal_state_raws,
                'alpha':          self.alpha,
            }, f)
        print(f"[ConformalPredictorWebShop] saved → {path}  (pool size={len(self.cal_scores)})")

    def load(self, path: str):
        """Read a saved file and restore the calibration state."""
        with open(path, 'rb') as f:
            d = pickle.load(f)
        self.cal_scores     = d['cal_scores']
        self.cal_state_raws = d['cal_state_raws']
        self.alpha          = d['alpha']
        print(
            f"[ConformalPredictorWebShop] loaded  pool size={len(self.cal_scores)}"
            f"  α={self.alpha}"
            f"  full-pool marginal S*={self._full_pool_threshold():.4f}"
        )

    # ─── SIMILARITY SELECTION ────────────────────────────────────────────────────

    def _select_similar(
        self,
        state_raw_test: np.ndarray,
        select: str                    = 'topk',
        k: int                         = 50,
        sim_threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Rank all calibration states by squared L2 distance to state_raw_test.
        Return the selected subset's scores, state_raws, pool indices, and distances.

        Returns:
            selected_scores     : [n_sel]        nonconformity scores of optimal actions
            selected_state_raws : [n_sel, 1536]  state embeddings
            selected_indices    : [n_sel]         indices into cal_scores / cal_state_raws
            selected_dists      : [n_sel]         squared L2 distances (ascending order)
        """
        diff    = self.cal_state_raws - state_raw_test[None, :]  # [n, 1536]
        sq_dist = (diff ** 2).sum(axis=1)                        # [n]
        sorted_idx = np.argsort(sq_dist)                         # ascending

        if select == 'topk':
            selected_idx = sorted_idx[:min(k, len(sorted_idx))]
        else:  # 'threshold'
            assert sim_threshold is not None, \
                "sim_threshold must be set when select='threshold'"
            mask         = sq_dist[sorted_idx] <= sim_threshold
            selected_idx = sorted_idx[mask]

        return (
            self.cal_scores[selected_idx],
            self.cal_state_raws[selected_idx],
            selected_idx,
            sq_dist[selected_idx],
        )

    # ─── DUAL LP ─────────────────────────────────────────────────────────────────

    def _full_pool_threshold(self) -> float:
        """Marginal quantile on the entire calibration pool (used for diagnostics
        and as the empty-selection fallback)."""
        n     = len(self.cal_scores)
        level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        return float(np.quantile(self.cal_scores, level))

    def _build_phi(
        self,
        selected_state_raws: np.ndarray,   # [n_sel, 1536]
        state_raw_test: np.ndarray,        # [1536]
        n_components: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project the selected calibration state_raws and the test state_raw
        to a low-dimensional space (dim = n_components) using SVD.

        This ensures the LP has d = n_components equality constraints and
        n_sel + 1 variables, keeping it feasible (d < n_sel).

        The SVD is fitted on the selected calibration states so the directions
        captured are the ones that vary most among the similar states.

        Returns:
            Phi_cal  : [n_sel, d]  feature matrix for selected calibration states
            Phi_test : [d]         feature vector for the test state
        """
        n_sel  = selected_state_raws.shape[0]
        n_comp = min(n_components, n_sel - 1, selected_state_raws.shape[1])

        # Fit SVD on selected calibration states
        mean     = selected_state_raws.mean(axis=0)                  # [1536]
        centered = (selected_state_raws - mean).astype(np.float64)   # [n_sel, 1536]
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)      # top directions
        V = Vt[:n_comp].T                                            # [1536, n_comp]

        # Project both calibration states and test state into this space
        Phi_cal  = selected_state_raws.astype(np.float64) @ V       # [n_sel, n_comp]
        Phi_test = state_raw_test.astype(np.float64)       @ V      # [n_comp]

        return Phi_cal, Phi_test

    def _dual_eta(
        self,
        S_test: float,
        selected_scores: np.ndarray,   # [n_sel]
        Phi_cal: np.ndarray,           # [n_sel, d]
        Phi_test: np.ndarray,          # [d]
    ) -> np.ndarray:
        """
        Solve the dual LP (Gibbs et al. 2023, eq 4.2) on the selected states:

            maximize   Σ_i η_i S_i + η_{n+1} S_test
            subject to −α ≤ η_i ≤ 1−α   for all i = 1…n_sel+1
                       Φ^T η = 0

        where Φ = [Phi_cal; Phi_test]  (shape [n_sel+1, d]).

        Returns the FULL η vector [n_sel+1]:
            η[0]…η[n_sel-1]  — weights for each calibration state's optimal action score
            η[-1]            — weight for the test action score
        Action is in prediction set iff η[-1] < 1 − α.
        """
        n_sel  = len(selected_scores)

        # Objective: negate for minimization
        c      = -np.append(selected_scores, S_test)            # [n_sel+1]
        bounds = [(-self.alpha, 1 - self.alpha)] * (n_sel + 1)  # η_i ∈ [-α, 1-α]

        # Equality constraint: Φ^T η = 0
        Phi_all = np.vstack([Phi_cal, Phi_test.reshape(1, -1)])  # [n_sel+1, d]
        A_eq    = Phi_all.T                                       # [d, n_sel+1]
        b_eq    = np.zeros(A_eq.shape[0])

        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if res.success:
            return res.x   # full vector [n_sel+1]

        # LP failed — fallback: include action iff its score is below the full-pool threshold
        S_star  = self._full_pool_threshold()
        eta     = np.full(n_sel + 1, -self.alpha)
        eta[-1] = (1 - self.alpha) if S_test > S_star else -self.alpha
        return eta

    # ─── MAIN INTERFACE ──────────────────────────────────────────────────────────

    def get_prediction_set(
        self,
        instruction: str,
        previous_actions: List[str],
        admissible_actions: List[str],
        logits_from_llm: List[List[float]],
        select: str                    = 'topk',
        k: int                         = 50,
        sim_threshold: Optional[float] = None,
        n_components: int              = 10,
    ) -> Tuple[List[str], Dict[str, float], int, List[Dict]]:
        """
        Build the prediction set for the current WebShop state using dual LP.

        Args:
            instruction        : task description string
            previous_actions   : list of actions taken so far (excluding 'reset')
            admissible_actions : candidate actions from the environment
            logits_from_llm    : list of K lists; each inner list is token log-probs
                                 for the corresponding admissible action (aligned
                                 positionally with admissible_actions)

            select        : 'topk'      — use k nearest calibration states.
                            'threshold' — use all states with sq_dist ≤ sim_threshold.
            k             : number of nearest neighbours (select='topk').
            sim_threshold : squared-L2 cutoff            (select='threshold').
            n_components  : dimensionality of Φ projection for the dual LP.
                            Must be < k (or < number of selected states).

        Returns:
            prediction_set : actions with η_{n+1} < 1−α, sorted ascending by
                             nonconformity score.
            action_scores  : {action: nonconformity_score} for all admissible actions.
            n_sel          : number of calibration states that contributed.
            similar_info   : list of dicts (one per selected state), each with:
                               rank, dist, score, eta, and metadata fields
                               (instruction, prev_actions, optimal_action)
                               if cal_metadata was set.
        """
        assert self.cal_scores is not None, \
            "Call calibrate_from_records() or load() first."

        # ── Step 1: nonconformity score for every admissible action ──────────    # Given a state and its admissible actions, it returns the non-conformity scores for all the actions
        action_scores = self.score_model.get_nonconformity_scores(
            instruction=instruction,
            previous_actions=previous_actions,
            admissible_actions=admissible_actions,
            logits_from_llm=logits_from_llm,
            bert_cache=self.bert_cache,
        )

        # ── Step 2: build state_raw for the current state ─────────────────────   # creating a raw embedding for my current state
        instr_emb = self.bert_cache.get(instruction)
        prev_pool = (
            self.bert_cache.get_batch(previous_actions).mean(dim=0)
            if previous_actions
            else torch.zeros(BERT_DIM)
        )
        state_raw_test = torch.cat([instr_emb, prev_pool]).numpy()  # [1536]

        # ── Step 3: store pending state so commit_step() can use it ──────────
        self._pending_state_raw     = state_raw_test.astype(np.float32)
        self._pending_action_scores = action_scores

        # ── Step 4: select similar calibration states ─────────────────────────   # selecting k similar states from the calibration pool of data
        selected_scores, selected_state_raws, selected_indices, selected_dists = self._select_similar(state_raw_test, select=select, k=k, sim_threshold=sim_threshold)

        # ── Step 5: build similar_info for human-readable printing ────────────
        similar_info = []
        for rank, (idx, dist, score) in enumerate(
            zip(selected_indices, selected_dists, selected_scores), start=1
        ):
            entry = {
                'rank':  rank,
                'dist':  float(dist),
                'score': float(score),
            }
            if self.cal_metadata is not None and int(idx) < len(self.cal_metadata):
                entry.update(self.cal_metadata[int(idx)])
            else:
                entry['instruction']    = '(online state — no metadata)'
                entry['prev_actions']   = []
                entry['optimal_action'] = ''
            similar_info.append(entry)

        # Fallback: if no states were selected, include all actions
        if len(selected_scores) == 0:
            in_set = sorted(admissible_actions, key=lambda a: action_scores[a])
            return in_set, action_scores, 0, []

        n_sel = len(selected_scores)

        # ── Step 6: solve dual LP for each admissible action ─────────────────
        Phi_cal, Phi_test = self._build_phi(
            selected_state_raws, state_raw_test, n_components
        )

        eta_vectors = {
            a: self._dual_eta(S, selected_scores, Phi_cal, Phi_test)
            for a, S in action_scores.items()
        }

        # for action, S in action_scores.items():
        #     eta = self._dual_eta(S, selected_scores, Phi_cal, Phi_test)
        #     print(f"  Action: {action!r}  |  Score: {S:.4f}  |  eta[-1]: {eta[-1]:.4f}  |  in_set: {eta[-1] < 1 - self.alpha}")
        #     if S < 0.3:
        #         print(f"    All eta values: {eta}")
        #         print(f"    Selected scores: {selected_scores}")
        # input("Press Enter to continue...")

        in_set = [a for a, eta in eta_vectors.items() if eta[-1] < 1 - self.alpha]

        # Attach calibration etas to similar_info using the best-scoring action's LP
        best_action = min(action_scores, key=action_scores.__getitem__)
        cal_etas    = eta_vectors[best_action][:-1]   # [n_sel]
        for i, entry in enumerate(similar_info):
            entry['eta'] = float(cal_etas[i])

        in_set.sort(key=lambda a: action_scores[a])
        return in_set, action_scores, n_sel, similar_info

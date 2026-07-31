"""
Conformal Predictor for ALFWorld action selection.

Approach: similarity-based conditional conformal prediction.

The calibration pool has two sources:
  1. Val-split states from training_data_2.pkl  (loaded once at startup)
  2. States explored during live gameplay        (added via commit_step)

At each game step:
  1. Compute state_raw = BERT(task) || BERT(location) || mean(BERT(prev_actions))
  2. Rank all calibration states by L2 distance to state_raw_test
  3. Select the most similar ones (top-k or distance threshold)
  4. Compute the threshold S* for the current state using one of two methods:

     method='quantile'  — standard conformal quantile on the selected scores
                          S* = ceil((n_sel+1)(1-α))/n_sel quantile
                          Action in set iff score ≤ S*

     method='dual_lp'   — solve the dual LP (Gibbs et al. 2023, eq 4.2)
                          on the selected states:
                            maximize  Σ_i η_i S_i + η_{n+1} S_test
                            s.t.      −α ≤ η_i ≤ 1−α
                                      Φ^T η = 0
                          Φ is a low-dim projection of the selected state_raws
                          (computed via SVD so d < n_sel, keeping LP feasible).
                          Action in set iff η_{n+1} < 1 − α.

Gameplay loop:
    pred_set, scores, S_star, n_sel = predictor.get_prediction_set(
        ..., method='quantile'   # or 'dual_lp'
    )
    action = pick_action(pred_set)
    env.step(action)
    predictor.commit_step(action)   # adds this state+score to the pool
"""

import pickle
import numpy as np
import torch
from scipy.optimize import linprog
from typing import List, Dict, Optional, Tuple

from score_model import ScoreFunction, BertEmbeddingCache, BERT_DIM


class ConformalPredictor:

    def __init__(
        self,
        score_model: ScoreFunction,         # score function model
        bert_cache: BertEmbeddingCache,     # BERT model to use the precomputed embeddings
        alpha: float = 0.1,                 # alpha = (1-coverage), if alpha = 0.1, coverage is 90%
    ):
        self.score_model = score_model
        self.bert_cache  = bert_cache
        self.alpha       = alpha

        # Calibration pool — grows over time
        self.cal_scores:     Optional[np.ndarray] = None   # [n]         # non-conformity scores from validation set
        self.cal_state_raws: Optional[np.ndarray] = None   # [n, 2304]   # corresponding states

        # Optional human-readable metadata for each calibration state.
        # List of dicts with keys: task_name, location, prev_actions, optimal_action.
        # Populated externally (e.g. from test script after calibrate_from_records).
        # Online states added via commit_step won't have metadata entries.
        self.cal_metadata: Optional[List[Dict]] = None     # [n]  metadata per pool state

        # Pending state from the most recent get_prediction_set() call.
        # Consumed by commit_step().
        self._pending_state_raw:     Optional[np.ndarray]      = None    # The current state you just made a prediction on
        self._pending_action_scores: Optional[Dict[str, float]] = None   # The scores for all candidate actions in that current state

    # ─── CALIBRATION SETUP ───────────────────────────────────────────────────────

    def calibrate_from_records(                 # given the val_records, it computes the non-confirmty scores of all optimal actions, stores it and its corresponding state embeddings.
        self,
        val_records: list,
        device: torch.device,
    ):
        """
        Populate the initial calibration pool from val-split records.
        Uses pre-processed tensors from train_score.build_records().
        """
        scores, state_raws, metadata = [], [], []

        for rec in val_records:
            state_raw    = rec['state_raw'].to(device)
            action_embs  = rec['action_embs'].to(device)
            softmax_bins = rec['softmax_bins'].to(device)
            # Multi-label aware: build_records emits `optimal_idxs` (list);
            # fall back to the legacy single-label `optimal_idx`.
            optimal_idxs = rec.get('optimal_idxs')
            if optimal_idxs is None:
                optimal_idxs = [rec['optimal_idx']]

            nc = self.score_model.compute_nonconformity_scores(
                state_raw, action_embs, softmax_bins
            )  # [K]   # given the state and all its admissible actions, it s returns non-conformity scores for all the actions.

            # Coverage = at least one optimal action lands in the prediction set
            # (action in set iff nc ≤ S*), so the state's calibration score is
            # the MIN nonconformity over the acceptable optimal actions.
            best_i   = min(optimal_idxs, key=lambda i: nc[i].item())   # optimal action with the lowest nonconformity
            scores.append(nc[best_i].item())   # store the optimal-action nonconformity for this state.
            state_raws.append(rec['state_raw'].cpu().numpy())   # stores the corresponding state embeddings.

            # Diagnostic metadata for this pool entry (which optimal action the
            # stored score belongs to, plus the state it came from).
            meta = dict(rec.get('meta', {}))
            adm  = meta.get('admissible_actions', [])
            meta['optimal_action'] = adm[best_i] if best_i < len(adm) else ''
            metadata.append(meta)

        self.cal_scores     = np.array(scores,     dtype=np.float64)     # [S1, S2, S3, ....Sn] -> non-conformity scores of all optimal actions
        self.cal_state_raws = np.array(state_raws, dtype=np.float32)     # Contains all corresponding states for the optimal actions.
        self.cal_metadata   = metadata                                  # [n]  readable info per pool entry (for _select_similar debug)

        print(
            f"[Calibration]  pool size={len(scores)}  α={self.alpha}"
            f"  full-pool S*={self._full_pool_threshold():.4f}"
            f"  mean={self.cal_scores.mean():.4f}"
            f"  std={self.cal_scores.std():.4f}"
        )

    def commit_step(self, executed_action: str, metadata: dict = None):         # adding current state to the calibration data or pool for future use
        """
        Add the current state to the calibration pool using the nonconformity
        score of the action that was actually executed.

        Call this immediately after env.step(action) at every game step
        (after the warm-up period ends).

        Args:
            executed_action: the action that was executed.
            metadata: optional dict with keys task_name, location, prev_actions,
                      optimal_action — appended to cal_metadata if provided.
        """
        assert self._pending_state_raw is not None, \
            "commit_step() called before get_prediction_set()."
        assert executed_action in self._pending_action_scores, \
            f"executed_action '{executed_action}' not found in last action scores."

        score = self._pending_action_scores[executed_action]

        self.cal_scores     = np.append(self.cal_scores, float(score))       # adding the non-conformity score for the optimal action.
        self.cal_state_raws = np.vstack([
            self.cal_state_raws,
            self._pending_state_raw.reshape(1, -1),
        ])                           # appends the current state

        if self.cal_metadata is not None:
            self.cal_metadata.append(metadata if metadata is not None else {
                'task_name':      '',
                'location':       '',
                'prev_actions':   [],
                'optimal_action': executed_action,
            })

        # Clear pending state
        self._pending_state_raw     = None
        self._pending_action_scores = None

    def save(self, path: str):   # Stores your current calibration pool into a file.
        with open(path, 'wb') as f:
            pickle.dump({
                'cal_scores':     self.cal_scores,
                'cal_state_raws': self.cal_state_raws,
                'cal_metadata':   self.cal_metadata,
                'alpha':          self.alpha,
            }, f)
        print(f"[ConformalPredictor] saved → {path}  (pool size={len(self.cal_scores)})")

    def load(self, path: str):   #  Reads the saved file and restores your calibration state.
        with open(path, 'rb') as f:
            d = pickle.load(f)
        self.cal_scores     = d['cal_scores']
        self.cal_state_raws = d['cal_state_raws']
        self.cal_metadata   = d.get('cal_metadata')   # may be absent in older pickles
        self.alpha          = d['alpha']
        print(
            f"[ConformalPredictor] loaded  pool size={len(self.cal_scores)}"
            f"  α={self.alpha}"
            f"  full-pool S*={self._full_pool_threshold():.4f}"
        )

    # ─── SIMILARITY SELECTION ────────────────────────────────────────────────────

    def _select_similar(                                         # Given a state, it return the scores for the optimal actions of the top-k similar states
        self,
        state_raw_test: np.ndarray,
        select: str             = 'topk',
        k: int                  = 50,
        sim_threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Rank all calibration states by squared L2 distance to state_raw_test.
        Return the selected subset's scores, state_raws, pool indices, and distances.

        Returns:
            selected_scores     : [n_sel]       nonconformity scores of optimal actions
            selected_state_raws : [n_sel, 2304] state embeddings
            selected_indices    : [n_sel]       indices into cal_scores / cal_state_raws
            selected_dists      : [n_sel]       squared L2 distances (ascending order)
        """
        diff    = self.cal_state_raws - state_raw_test[None, :]  # [n, 2304]
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

    # ─── METHOD 1: QUANTILE ON SELECTED STATES ───────────────────────────────────

    def _quantile_threshold(
        self,
        selected_scores: np.ndarray,
    ) -> float:
        """
        Standard conformal quantile on the selected subset:
            S* = ceil((n_sel + 1)(1 - α)) / n_sel  quantile
        """
        n_sel  = len(selected_scores)
        level  = min(np.ceil((n_sel + 1) * (1 - self.alpha)) / n_sel, 1.0)
        return float(np.quantile(selected_scores, level))

    # ─── METHOD 2: DUAL LP ON SELECTED STATES ────────────────────────────────────

    def _build_phi(
        self,
        selected_state_raws: np.ndarray,   # [n_sel, 2304]
        state_raw_test: np.ndarray,        # [2304]
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
        n_sel   = selected_state_raws.shape[0]                                            # number of selected calibration samples (states). 
        n_comp  = min(n_components, n_sel - 1, selected_state_raws.shape[1])              # number of dimensions (features) you keep after SVD projection.

        # Fit SVD on selected calibration states
        mean    = selected_state_raws.mean(axis=0)               # [2304]              # compute the average state across selected samples
        centered = (selected_state_raws - mean).astype(np.float64)  # [n_sel, 2304]     # Data is centered, so mean is substracted from all selected states
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)                         # Vt contains the principal directions of my data
        V = Vt[:n_comp].T                                         # [2304, n_comp]       # projection matrix which contains the top part of Vt.

        # Project both calibration states and test state into this space
        Phi_cal  = selected_state_raws.astype(np.float64) @ V    # [n_sel, n_comp]
        Phi_test = state_raw_test.astype(np.float64)       @ V   # [n_comp]

        return Phi_cal, Phi_test

    def _dual_eta(
        self,
        S_test: float,                                       # score of the candidate action
        selected_scores: np.ndarray,   # [n_sel]             # scores of optimal actions from similar states
        Phi_cal: np.ndarray,           # [n_sel, d]          # features of similar states.
        Phi_test: np.ndarray,          # [d]                 # features of current state.
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
        c      = -np.append(selected_scores, S_test)           # [n_sel+1]
        bounds = [(-self.alpha, 1 - self.alpha)] * (n_sel + 1) # η_i ∈ [-α, 1-α]

        # Equality constraint: Φ^T η = 0
        Phi_all = np.vstack([Phi_cal, Phi_test.reshape(1, -1)]) # [n_sel+1, d]
        A_eq    = Phi_all.T                                      # [d, n_sel+1]
        b_eq    = np.zeros(A_eq.shape[0])

        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        # res.x = [η₁, η₂, ..., η_{n_sel}, η_{test}]

        if res.success:
            # print("Eta values are: ", res.x[-1])
            # exit(0)
            return res.x   # full vector [n_sel+1]

        # LP failed — fallback: set all cal etas to -α, test eta by quantile rule
        S_star   = self._quantile_threshold(selected_scores)
        eta      = np.full(n_sel + 1, -self.alpha)
        eta[-1]  = (1 - self.alpha) if S_test > S_star else -self.alpha
        return eta

    # ─── FALLBACK ────────────────────────────────────────────────────────────────

    def _full_pool_threshold(self) -> float:       # calculating marginal threshold on the entire calibration pool
        n     = len(self.cal_scores)
        level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        return float(np.quantile(self.cal_scores, level))

    # ─── MAIN INTERFACE ──────────────────────────────────────────────────────────

    def get_prediction_set(
        self,
        task: str,
        previous_actions: List[str],
        location: str,
        admissible_actions: List[str],
        softmax_values: Dict[str, list],
        method: str             = 'quantile',
        select: str             = 'topk',
        k: int                  = 50,
        sim_threshold: Optional[float] = None,
        n_components: int       = 10,
        optimal_action: str     = '',
    ) -> Tuple[List[str], Dict[str, float], float, int, List[Dict]]:
        """
        Build the prediction set for the current ALFWorld state.

        Args:
            task, previous_actions, location : current state
            admissible_actions : candidate actions from the environment
            softmax_values     : {action: [10 bins]} LLM distribution

            method        : 'quantile' — threshold = conformal quantile of
                                         selected calibration scores.
                            'dual_lp'  — solve dual LP on selected states;
                                         action included iff η_{n+1} < 1-α.

            select        : 'topk'      — use k nearest calibration states.
                            'threshold' — use all states with sq_dist ≤ sim_threshold.
            k             : number of nearest neighbours (select='topk').
            sim_threshold : squared-L2 cutoff            (select='threshold').
            n_components  : dimensionality of Φ projection for method='dual_lp'.
                            Must be < k (or < number of selected states).

        Returns:
            prediction_set : actions with score ≤ S* (quantile) or η<1-α (LP),
                             sorted ascending by nonconformity score.
            action_scores  : {action: nonconformity_score} for all actions.
            S_star         : threshold used (quantile method) or np.nan (LP method).
            n_sel          : number of calibration states that contributed.
            similar_info   : list of dicts (one per selected state), each with:
                               rank, dist, score, and metadata fields
                               (task_name, location, prev_actions, optimal_action)
                               if cal_metadata was set.
        """
        assert self.cal_scores is not None, \
            "Call calibrate_from_records() or load() first."

        # ── Step 1: nonconformity score for every admissible action ──────────
        action_scores = self.score_model.get_nonconformity_scores(
            task=task,
            previous_actions=previous_actions,
            location=location,
            admissible_actions=admissible_actions,
            softmax_values=softmax_values,
            bert_cache=self.bert_cache,
        )
        

        # ── Step 2: build state_raw for the current state ─────────────────────
        task_emb  = self.bert_cache.get(task)
        loc_emb   = self.bert_cache.get(location)
        prev_pool = (
            self.bert_cache.get_batch(previous_actions).mean(dim=0)
            if previous_actions
            else torch.zeros(BERT_DIM)
        )
        state_raw_test = torch.cat([task_emb, loc_emb, prev_pool]).numpy()

        # ── Step 3: store pending state so commit_step() can use it ──────────
        self._pending_state_raw     = state_raw_test.astype(np.float32)
        self._pending_action_scores = action_scores

        # ── Step 4: select similar calibration states ─────────────────────────
        selected_scores, selected_state_raws, selected_indices, selected_dists = \
            self._select_similar(state_raw_test, select=select, k=k, sim_threshold=sim_threshold)

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
                entry['task_name']      = '(online state — no metadata)'
                entry['location']       = ''
                entry['prev_actions']   = []
                entry['optimal_action'] = ''
            similar_info.append(entry)

        # ── DEBUG: print selected similar states (set predictor.debug_similar
        #    = True to enable; off by default — 50 lines per node bloats logs) ──
        if getattr(self, 'debug_similar', False):
            print(f"\n[DEBUG _select_similar] {len(selected_scores)} similar states "
                  f"(select={select}, k={k}, sim_threshold={sim_threshold})")
            for e in similar_info:
                print(f"  rank {e['rank']:>3}  dist={e['dist']:.4f}  score={e['score']:.4f}"
                      f"  optimal_action={e.get('optimal_action', '')!r}"
                      f"  task={e.get('task_name', '')!r}  loc={e.get('location', '')!r}"
                      f"  prev_actions={e.get('prev_actions', [])}")

        # Fallback to full pool if nothing selected
        if len(selected_scores) == 0:
            S_star = self._full_pool_threshold()
            in_set = [a for a, S in action_scores.items() if S <= S_star]
            in_set.sort(key=lambda a: action_scores[a])
            return in_set, action_scores, S_star, 0, []

        n_sel = len(selected_scores)

        # ── Step 6: build prediction set using chosen method ──────────────────
        if method == 'dual_lp':
            Phi_cal, Phi_test = self._build_phi(
                selected_state_raws, state_raw_test, n_components
            )

            # Solve LP once per admissible action; collect full η vectors
            # for a,S in action_scores.items():
            #     with open("eta_input.pkl","wb") as f:
            #         data_to_save = {
            #             "S": S, "all_admissible_action_scores": action_scores, "optimal_action": optimal_action, "selected_scores": selected_scores, "Phi_cal": Phi_cal, "Phi_test": Phi_test
            #         }
            #         pickle.dump(data_to_save,f)
            #     break
            # print("pkl file saved......")
            # exit(0)
            eta_vectors = {
                a: self._dual_eta(S, selected_scores, Phi_cal, Phi_test)
                for a, S in action_scores.items()
            }
            # for a, S in action_scores.items():
            #     a = self._dual_eta(S, selected_scores, Phi_cal, Phi_test)
            #     print(f"Score: {S}, eta_values: {a[-1]}")
            #     if S < 0.3:
            #         print(f"All eta values: {a}")
            #         print(f"Selected scores: {selected_scores}")
            # input("Press Enter to continue...")

            in_set = [a for a, eta in eta_vectors.items() if eta[-1] < 1 - self.alpha]
            # print("In-set: ", [(a, eta[-1]) for a, eta in in_set])
            # exit(0)
            S_star = float('nan')

            # Attach calibration etas (η_1…η_{n_sel}) to similar_info.
            # η values are action-dependent; we use the best-scoring action's LP
            # solution as the representative (this is the action most likely to
            # be executed, so its calibration weights are the most relevant).
            best_action = min(action_scores, key=action_scores.__getitem__)
            cal_etas    = eta_vectors[best_action][:-1]   # [n_sel]
            for i, entry in enumerate(similar_info):
                entry['eta'] = float(cal_etas[i])

        else:  # 'quantile'
            S_star = self._quantile_threshold(selected_scores)
            in_set = [a for a, S in action_scores.items() if S <= S_star]
            for entry in similar_info:
                entry['eta'] = None   # not applicable for quantile method

        in_set.sort(key=lambda a: action_scores[a])
        return in_set, action_scores, S_star, n_sel, similar_info

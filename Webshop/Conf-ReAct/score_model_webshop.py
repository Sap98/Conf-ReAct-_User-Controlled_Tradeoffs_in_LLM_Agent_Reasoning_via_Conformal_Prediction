"""
Score Model for Conformal Prediction in WebShop.

Input:  instruction = task description string
        previous_actions = list of actions taken so far (excluding 'reset')
        admissible_actions = list of candidate action strings
        logits_from_llm = list of K lists, each being token log-probs for one action

Output: nonconformity score for each admissible action

Internally uses a frozen BERT to embed all text, then trainable MLPs to
project and score each (state, action) pair.

Architecture differences from ALFWorld score_model.py:
  - No 'location' field → STATE_RAW_DIM = 2 * 768 = 1536 (instruction + prev_pool)
  - Same 10-bin softmax histogram as ALFWorld → ACTION_RAW_DIM = 768 + 10 = 778
"""

import re
import pickle
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Lone/paired UTF-16 surrogate code points (U+D800–U+DFFF) crash the HuggingFace
# fast tokenizer ("TextEncodeInput must be ..."). They show up in degenerate LLM
# generations (e.g. a garbage action full of repeated glyphs + a broken emoji).
# Strip them before tokenizing so BERT never sees them. Applied only to the text
# fed to the tokenizer — cache keys keep the original string, so lookups by the
# raw action text are unaffected.
_SURROGATE_RE = re.compile(r'[\ud800-\udfff]')


def _strip_surrogates(text: str) -> str:
    return _SURROGATE_RE.sub('', text)

BERT_MODEL_NAME = "bert-base-uncased"
BERT_DIM = 768
SOFTMAX_BINS = 10
STATE_RAW_DIM = 2 * BERT_DIM              # 1536: concat(instruction_emb, prev_pool)
ACTION_RAW_DIM = BERT_DIM + SOFTMAX_BINS  # 778:  concat(action_emb, softmax_bins[10])


# ═══════════════════════════════════════════════════════════════════════════════
# BERT EMBEDDING CACHE
# ═══════════════════════════════════════════════════════════════════════════════

class BertEmbeddingCache:
    """Caches BERT CLS embeddings so each unique text is embedded only once."""

    def __init__(self, model_name: str = BERT_MODEL_NAME, device: torch.device = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name).to(device)
        self.bert.eval()
        self.device = device
        self.cache: Dict[str, torch.Tensor] = {}

    def _compute_cls(self, texts: List[str], batch_size: int = 64) -> List[torch.Tensor]:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = [_strip_surrogates(t) for t in texts[i: i + batch_size]]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=64, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                out = self.bert(**encoded)
                cls = out.last_hidden_state[:, 0, :].cpu()  # CLS token embedding
            for j in range(cls.size(0)):
                # .clone() is REQUIRED: cls[j] is a view into the [B, L, 768]
                # last_hidden_state storage, and pickling a view serialises the
                # ENTIRE underlying storage. Without it each 3 KB embedding is
                # written as ~12.6 MB, bloating the cache ~4000x (49 GB instead
                # of ~14 MB) and making it unloadable in RAM.
                embeddings.append(cls[j].clone())
        return embeddings

    def precompute(self, all_texts: List[str]):
        """Embed all unique texts once and store in cache."""
        new = [t for t in set(all_texts) if t not in self.cache]
        if not new:
            return
        embs = self._compute_cls(new)
        for text, emb in zip(new, embs):
            self.cache[text] = emb

    def get(self, text: str) -> torch.Tensor:
        """Return CLS embedding for a single text (compute and cache if not seen)."""
        if text not in self.cache:
            self.cache[text] = self._compute_cls([text])[0]
        return self.cache[text]

    def get_batch(self, texts: List[str]) -> torch.Tensor:
        """Return stacked CLS embeddings for a list of texts. Shape: [N, 768]."""
        return torch.stack([self.get(t) for t in texts])

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(dict(self.cache), f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            self.cache = pickle.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# SOFTMAX BIN HELPER  (mirrors compute_softmax_bins_10.py used for ALFWorld)
# ═══════════════════════════════════════════════════════════════════════════════

import math

def _logprobs_to_softmax_bins(token_logprobs: List[float], num_bins: int = SOFTMAX_BINS) -> List[int]:
    """
    Convert a list of token log-probs to a 10-bin softmax histogram.

    Steps:
      1. exp(logprob) → softmax probability per token  (value in [0, 1])
      2. bin each value into one of num_bins equal-width buckets
      3. return bin counts [num_bins]

    If token_logprobs is empty, returns all-zero bins.
    """
    if not token_logprobs:
        return [0] * num_bins
    bin_width = 1.0 / num_bins  # 0.1 for 10 bins
    bins = [0] * num_bins
    for lp in token_logprobs:
        val = math.exp(lp)
        bin_idx = int(val / bin_width)
        if bin_idx >= num_bins:
            bin_idx = num_bins - 1
        bins[bin_idx] += 1
    return bins


# ═══════════════════════════════════════════════════════════════════════════════
# SCORE FUNCTION MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class ScoreFunctionWebShop(nn.Module):
    """
    Takes a WebShop state and candidate actions, returns a nonconformity score
    per action.

    Internal architecture:
      State  = concat(BERT(instruction)[768], mean(BERT(prev_actions))[768])
               -> MLP_state -> [d_proj]
      Action = concat(BERT(action)[768], softmax_bins[10])
               -> MLP_action -> [d_proj]
      Logit  = MLP_head(concat(state_proj, action_proj)) -> scalar

      nonconformity_score = 1 - sigmoid(logit)
        Small -> model thinks action fits the state (likely optimal)
        Large -> model thinks action does not fit
    """

    def __init__(self, d_proj: int = 256, hidden: int = 128, dropout: float = 0.4):
        super().__init__()
        self.d_proj = d_proj
        self.hidden = hidden

        # State MLP: 1536 -> d_proj*2 -> d_proj
        self.mlp_state = nn.Sequential(
            nn.Linear(STATE_RAW_DIM, d_proj * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_proj * 2, d_proj),
            nn.LayerNorm(d_proj),
        )

        # Action MLP: 778 -> d_proj*2 -> d_proj
        self.mlp_action = nn.Sequential(
            nn.Linear(ACTION_RAW_DIM, d_proj * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_proj * 2, d_proj),
            nn.LayerNorm(d_proj),
        )

        # Score head: 2*d_proj -> hidden -> 1
        self.score_head = nn.Sequential(
            nn.Linear(2 * d_proj, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        state_raw: torch.Tensor,     # [batch, 1536]
        action_raw: torch.Tensor,    # [batch, 768]
        softmax_bins: torch.Tensor,  # [batch, 10]  10-bin softmax histogram
    ) -> torch.Tensor:
        """Raw logits [batch, 1]. Used during training with BCEWithLogitsLoss."""
        state_proj = self.mlp_state(state_raw)
        action_input = torch.cat([action_raw, softmax_bins], dim=-1)  # [batch, 778]
        action_proj = self.mlp_action(action_input)
        combined = torch.cat([state_proj, action_proj], dim=-1)    # [batch, 2*d_proj]
        return self.score_head(combined)                           # [batch, 1]

    @torch.no_grad()
    def compute_nonconformity_scores(
        self,
        state_raw: torch.Tensor,     # [1536]
        action_embs: torch.Tensor,   # [K, 768]
        softmax_bins: torch.Tensor,  # [K, 10]
    ) -> torch.Tensor:
        """
        Given precomputed tensors, return nonconformity scores [K].
        Used during evaluation (Recall@K).

        nonconformity = 1 - sigmoid(logit)
          Small -> model thinks action fits the state (likely optimal)
          Large -> model thinks action does not fit
        """
        K = action_embs.size(0)
        state_expanded = state_raw.unsqueeze(0).expand(K, -1)  # [K, 1536]
        logits = self.forward(state_expanded, action_embs, softmax_bins).squeeze(-1)  # [K]
        return 1.0 - torch.sigmoid(logits)

    @torch.no_grad()
    def get_nonconformity_scores(
        self,
        instruction: str,
        previous_actions: List[str],
        admissible_actions: List[str],
        logits_from_llm: List[List[float]],
        bert_cache: BertEmbeddingCache,
    ) -> Dict[str, float]:
        """
        Main inference interface — pass raw text in, get nonconformity scores out.

        Args:
            instruction:        task description string
            previous_actions:   list of actions taken so far (excluding 'reset')
            admissible_actions: list of candidate action strings
            logits_from_llm:    list of K lists; each inner list is token log-probs
                                for the corresponding admissible action
            bert_cache:         BertEmbeddingCache instance

        Returns:
            {action_str: nonconformity_score} for each admissible action
              Small score -> likely optimal
              Large score -> unlikely optimal
        """
        device = next(self.parameters()).device

        # ── Embed state ──
        instr_emb = bert_cache.get(instruction)              # [768]
        if previous_actions:
            prev_pool = bert_cache.get_batch(previous_actions).mean(dim=0)  # [768]
        else:
            prev_pool = torch.zeros(BERT_DIM)
        state_raw = torch.cat([instr_emb, prev_pool])        # [1536]

        # ── Embed actions ──
        action_embs = bert_cache.get_batch(admissible_actions)  # [K, 768]

        # Convert token log-probs → softmax probabilities → 10-bin histogram
        bins = [_logprobs_to_softmax_bins(lp) for lp in logits_from_llm]
        softmax_bins = torch.tensor(bins, dtype=torch.float32)   # [K, 10]

        # ── Score ──
        K = len(admissible_actions)
        state_expanded = state_raw.unsqueeze(0).expand(K, -1).to(device)
        action_embs  = action_embs.to(device)
        softmax_bins = softmax_bins.to(device)

        logits = self.forward(state_expanded, action_embs, softmax_bins).squeeze(-1)  # [K]
        nonconformity = (1.0 - torch.sigmoid(logits)).cpu()

        return {act: nonconformity[i].item() for i, act in enumerate(admissible_actions)}

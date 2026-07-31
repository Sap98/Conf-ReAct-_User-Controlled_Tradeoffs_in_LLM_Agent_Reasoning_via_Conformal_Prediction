"""
Score Model for Conformal Prediction in ALFWorld.

Input:  state = {task, previous_actions, location}
        admissible_actions = list of candidate action strings
        softmax_values = {action_str: [10 bin counts]}

Output: nonconformity score for each admissible action

Internally uses a frozen BERT to embed all text, then trainable MLPs to
project and score each (state, action) pair.
"""

import pickle
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

BERT_MODEL_NAME = "bert-base-uncased"
BERT_DIM = 768
SOFTMAX_BINS = 10
STATE_RAW_DIM = 3 * BERT_DIM              # 2304
ACTION_RAW_DIM = BERT_DIM + SOFTMAX_BINS   # 773


# ═══════════════════════════════════════════════════════════════════════════════
# BERT EMBEDDING CACHE
# ═══════════════════════════════════════════════════════════════════════════════

class BertEmbeddingCache:
    """Caches BERT CLS embeddings so each unique text is embedded only once."""

    def __init__(self, model_name: str = BERT_MODEL_NAME, device: torch.device = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.bert = AutoModel.from_pretrained(model_name, local_files_only=True).to(device)
        self.bert.eval()
        self.device = device
        self.cache: Dict[str, torch.Tensor] = {}

    def _compute_cls(self, texts: List[str], batch_size: int = 64) -> List[torch.Tensor]:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=64, return_tensors="pt"
            ).to(self.device)   # encoded stores, input_ids, attention_mask and token_type_ids in a json format
            with torch.no_grad():
                out = self.bert(**encoded)
                cls = out.last_hidden_state[:, 0, :].cpu()   # only CLS token's embedding is taken 
            for j in range(cls.size(0)):
                embeddings.append(cls[j])  # attaches all K embeddings to the list.
        return embeddings

    def precompute(self, all_texts: List[str]):  # Compute its CLS embedding once. Store it in self.cache. Reuse it forever
        new = [t for t in set(all_texts) if t not in self.cache]
        if not new:                            
            return
        embs = self._compute_cls(new)
        for text, emb in zip(new, embs):
            self.cache[text] = emb

    def get(self, text: str) -> torch.Tensor:  # if text is not already cached, compute it once, store it, return it
        if text not in self.cache:
            self.cache[text] = self._compute_cls([text])[0]
        return self.cache[text]

    def get_batch(self, texts: List[str]) -> torch.Tensor:  # # Calls get() for each text. Ensures it's cached. Stacks embeddings into tensor
        return torch.stack([self.get(t) for t in texts])

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(dict(self.cache), f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            self.cache = pickle.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# SCORE FUNCTION MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class ScoreFunction(nn.Module):
    """
    Takes a state and candidate actions, returns a nonconformity score per action.

    Internal architecture:
      State  = concat(BERT(task)[768], BERT(location)[768], mean(BERT(prev_actions))[768])
               -> MLP_state -> [d_proj]
      Action = concat(BERT(action)[768], softmax_bins[10])
               -> MLP_action -> [d_proj]
      Logit  = MLP_head(concat(state_proj, action_proj)) -> scalar

      nonconformity_score = 1 - sigmoid(logit)
        Small -> model thinks action fits the state (likely optimal)
        Large -> model thinks action does not fit
    """

    def __init__(self, d_proj: int = 256, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.d_proj = d_proj
        self.hidden = hidden

        self.mlp_state = nn.Sequential(
            nn.Linear(STATE_RAW_DIM, d_proj * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_proj * 2, d_proj),
            nn.LayerNorm(d_proj),
        )   # this is the state_mlp of 2 layers MLP which returns a learned embedding of each state in the batch

        self.mlp_action = nn.Sequential(
            nn.Linear(ACTION_RAW_DIM, d_proj * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_proj * 2, d_proj),
            nn.LayerNorm(d_proj),
        )  # this is the action_mlp of 2 layers MLP which returns a learned embedding of each action in the batch

        self.score_head = nn.Sequential(
            nn.Linear(2 * d_proj, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )  # the states' and actions' learned embeddings are concatenated and passed to this 2-layer MLP for a final confirmty score

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        state_raw: torch.Tensor,       # [batch, 2304]
        action_raw: torch.Tensor,      # [batch, 768]
        softmax_bins: torch.Tensor,    # [batch, 10]
    ) -> torch.Tensor:
        """Raw logits [batch, 1]. Used during training with BCEWithLogitsLoss."""
        state_proj = self.mlp_state(state_raw)  
        action_input = torch.cat([action_raw, softmax_bins], dim=-1)
        action_proj = self.mlp_action(action_input)
        combined = torch.cat([state_proj, action_proj], dim=-1)
        return self.score_head(combined)

    @torch.no_grad()
    def compute_nonconformity_scores(
        self,
        state_raw: torch.Tensor,       # [2304]
        action_embs: torch.Tensor,     # [K, 768]
        softmax_bins: torch.Tensor,    # [K, 10]
    ) -> torch.Tensor:
        """
        Given precomputed tensors, return nonconformity scores [K].
        Used during training evaluation (Recall@K).

        nonconformity = 1 - sigmoid(logit)
          Small -> model thinks action fits the state (likely optimal)
          Large -> model thinks action does not fit
        """
        K = action_embs.size(0)  # calculates total no of admissible actions per state
        state_expanded = state_raw.unsqueeze(0).expand(K, -1) # it returns K identical copies of same state
        logits = self.forward(state_expanded, action_embs, softmax_bins).squeeze(-1)  # confirmity scores are returned by forward()
        return 1.0 - torch.sigmoid(logits)   # non-confirmity scores finally returned from this function

    @torch.no_grad()
    def get_nonconformity_scores(
        self,
        task: str,
        previous_actions: List[str],
        location: str,
        admissible_actions: List[str],
        softmax_values: Dict[str, list],
        bert_cache: BertEmbeddingCache,
    ) -> Dict[str, float]:
        """
        Main interface — pass raw text in, get nonconformity scores out.

        Args:
            task:               task description string (e.g. "put a mug in desk.")
            previous_actions:   list of actions taken so far
            location:           current location string
            admissible_actions: list of candidate action strings
            softmax_values:     {action_str: [10 bin counts]} from the LLM
            bert_cache:         BertEmbeddingCache instance

        Returns:
            {action_str: nonconformity_score} for each admissible action
              Small score -> likely optimal
              Large score -> unlikely optimal
        """
        device = next(self.parameters()).device

        # ── Embed state ──
        task_emb = bert_cache.get(task)        # task is embedded
        loc_emb = bert_cache.get(location)     # location is embedded
        if previous_actions:
            prev_pool = bert_cache.get_batch(previous_actions).mean(dim=0)     # previous actions are embedded if there exist any
        else:
            prev_pool = torch.zeros(BERT_DIM)                 # if there are no previous actions all zeros are taken in a tensor
        state_raw = torch.cat([task_emb, loc_emb, prev_pool])   # [2304]    # state embedding is created by concatenation of task, location, previous actions

        # ── Embed actions ──
        action_embs = bert_cache.get_batch(admissible_actions)   # [K, 768]
        bins = [softmax_values.get(a, [0] * SOFTMAX_BINS) for a in admissible_actions]  # .get(key, default value) gives the value from a dict of the given key otherwise it returns the default value
        softmax_bins = torch.tensor(bins, dtype=torch.float32)   # [K, 10]  # now, bins got converted to tensor from python data

        # ── Score ──
        K = len(admissible_actions)
        state_expanded = state_raw.unsqueeze(0).expand(K, -1).to(device)  # .expands() gets me K identical copies of same state
        # so that my scorer can compute f(s,a1​),f(s,a2​),…,f(s,aK​)

        action_embs = action_embs.to(device)
        softmax_bins = softmax_bins.to(device)

        logits = self.forward(state_expanded, action_embs, softmax_bins).squeeze(-1)  # [K]
        nonconformity = (1.0 - torch.sigmoid(logits)).cpu() # it is a tensor of size K containing non-conformity scores for each admissible action

        return {act: nonconformity[i].item() for i, act in enumerate(admissible_actions)}  # it just returns a dict having all admissible actions as their key and its scores as its respective values.

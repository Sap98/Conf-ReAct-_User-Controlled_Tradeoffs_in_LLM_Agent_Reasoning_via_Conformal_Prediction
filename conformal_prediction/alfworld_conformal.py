from typing import List, Tuple, Dict, Any
from transformers import AutoTokenizer, AutoModel
from alfworld.agents.environment import get_environment
import pickle
import torch
import warnings
from state_encoder import create_state_encoder

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-uncased"
MAX_ACTIONS = 30       # number of action slots for transformer input
CLS_MAX_LEN = 64       # max tokens per action/location
BATCH_SIZE = 32

# Initialize BERT for embeddings
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
bert.eval() 

def get_cls_embeddings(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device = DEVICE,
    max_length: int = CLS_MAX_LEN,
    batch_size: int = BATCH_SIZE
) -> torch.FloatTensor:
    """
    Returns CLS embeddings for each string in texts.
    Output tensor shape: (len(texts), hidden_size)
    """
    if len(texts) == 0:
        return torch.zeros((0, model.config.hidden_size), dtype=torch.float)

    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out = model(**encoded)
            last_hidden = out.last_hidden_state   # (B, seq_len, hidden)
            cls_emb = last_hidden[:, 0, :].cpu()  # (B, hidden) -> move to cpu for easier accumulation
        all_embs.append(cls_emb)

    embeddings = torch.cat(all_embs, dim=0)   # (N_texts, hidden)
    return embeddings 


def pad_action_embeddings(action_embs: torch.FloatTensor, max_actions: int = MAX_ACTIONS, hidden_size: int = 768):
    """
    action_embs: (num_actions, hidden) or empty (0, hidden)
    Returns:
      padded_actions: (max_actions, hidden)
      action_mask: (max_actions,)  tensor of 1/0 (1 -> real action, 0 -> pad)
    """
    if action_embs.shape[0] == 0:
        padded = torch.zeros((max_actions, hidden_size), dtype=torch.float)
        mask = torch.zeros((max_actions,), dtype=torch.long)
        return padded, mask

    num_actions = action_embs.shape[0]
    if num_actions >= max_actions:
        # keep the *last* max_actions actions (common choice). Change to [:max_actions] if you prefer first.
        trimmed = action_embs[-max_actions:]
        mask = torch.ones((max_actions,), dtype=torch.long)
        return trimmed, mask
    else:
        pad_len = max_actions - num_actions
        pad_tensor = torch.zeros((pad_len, hidden_size), dtype=torch.float)
        padded = torch.cat([action_embs, pad_tensor], dim=0)
        mask = torch.cat([torch.ones((num_actions,), dtype=torch.long),
                          torch.zeros((pad_len,), dtype=torch.long)], dim=0)
        return padded, mask



def parse_game_tuple(game_tuple: Tuple[str, Dict[str, Any]]):
    """
    Input like:
      (path_str, {'game_name': ..., 'steps': [ (prev_actions, location, action, next_location), ... ]})
    Returns list of steps where each step is (prev_actions_list, location_str)
    """
    _, game_dict = game_tuple
    steps = game_dict.get("steps", [])
    parsed = []
    for step in steps:
        # your steps appear to be tuples of length 4:
        # (prev_actions_list, location, action_str, next_location)
        if len(step) >= 2:
            prev_actions = step[0] if isinstance(step[0], list) else []
            location = step[1] if isinstance(step[1], str) else ""
            parsed.append((prev_actions, location))
        else:
            # fallback: skip or handle differently
            parsed.append(([], ""))
    return parsed


def build_state_sequences_from_game(
    state_to_optimal_mapping: Dict,
    # game_tuple: Tuple[str, Dict[str, Any]],
    tokenizer,
    model,
    device: torch.device = DEVICE,
    max_actions: int = MAX_ACTIONS
) -> List[Dict[str, torch.Tensor]]:
    """
    For each step returns a dict with:
      - 'state_sequence' : Tensor (1 + max_actions, hidden)  (location CLS first, then action CLSs padded)
      - 'state_mask'     : Tensor (1 + max_actions,)  (1 for location and real actions, 0 for padded actions)
      - 'num_actions'    : int  (original number of prev actions)
      - 'location_str'   : original location string (for debugging)
    """
    final_states_dict = []
    for key, value in state_to_optimal_mapping.items():
        game_tuple = (key, value)
        # print(game_tuple)

        parsed_steps = parse_game_tuple(game_tuple)      
        # To be simple but decently efficient: collect all texts to embed in two buckets:
        #  - all locations (one per step)
        #  - all actions flattened (we will embed per-step to preserve ordering)
        location_texts = [loc for (_, loc) in parsed_steps]
        # We'll compute location embeddings in one batch
        location_embs = get_cls_embeddings(location_texts, tokenizer, model, device=device)

        state_results = []
        for idx, (prev_actions, location) in enumerate(parsed_steps):
            # get location embedding (already computed)
            loc_emb = location_embs[idx]  # (hidden,)

            # compute embeddings for all prev_actions (can be empty)
            action_embs = get_cls_embeddings(prev_actions, tokenizer, model, device=device)  # (num_actions, hidden) or (0, hidden)

            # pad/truncate
            padded_actions, action_mask = pad_action_embeddings(action_embs, max_actions=max_actions)
            # construct full sequence: [location_emb] + padded_actions  -> shape (1 + max_actions, hidden)
            state_sequence = torch.cat([loc_emb.unsqueeze(0), padded_actions], dim=0)  # CPU tensor
            # state mask: first element is 1 (location), then action_mask (1/0)
            state_mask = torch.cat([torch.tensor([1], dtype=torch.long), action_mask], dim=0)

            state_results.append({
                "state_sequence": state_sequence,      # (1+max_actions, hidden)
                "state_mask": state_mask,              # (1+max_actions,)
                "num_actions": action_embs.shape[0],
                "location_str": location,
                "prev_actions": prev_actions
            })
        final_states_dict.append(state_results)
        # print(final_states_dict)
        # exit(0)
    return final_states_dict


def encode_states_with_transformer(
    state_dicts: List[Dict[str, torch.Tensor]],
    state_encoder,
    device: torch.device = DEVICE,
    batch_size: int = 32
) -> torch.FloatTensor:
    """
    Process state sequences through the transformer encoder to get final state embeddings.

    Args:
        state_dicts: List of dicts from build_state_sequences_from_game, each with:
                    - 'state_sequence': (seq_len, hidden)
                    - 'state_mask': (seq_len,)
        state_encoder: The StateEncoder model
        device: Device to run on
        batch_size: Batch size for processing

    Returns:
        state_embeddings: (num_states, hidden) - Final state embeddings
    """
    all_embeddings = []

    # Process in batches
    for i in range(0, len(state_dicts), batch_size):
        batch_dicts = state_dicts[i : i + batch_size]

        # Stack sequences and masks
        batch_sequences = torch.stack([s["state_sequence"] for s in batch_dicts], dim=0)  # (batch, seq_len, hidden)
        batch_masks = torch.stack([s["state_mask"] for s in batch_dicts], dim=0)  # (batch, seq_len)

        # Move to device
        batch_sequences = batch_sequences.to(device)
        batch_masks = batch_masks.to(device)

        # Forward pass through transformer
        with torch.no_grad():
            batch_embeddings = state_encoder(batch_sequences, batch_masks)  # (batch, hidden)

        # Move back to CPU for accumulation
        all_embeddings.append(batch_embeddings.cpu())

    # Concatenate all batches
    final_embeddings = torch.cat(all_embeddings, dim=0)  # (num_states, hidden)
    return final_embeddings


def main():

    with open('/home/saptarshi/alfworld/ReAct/react_sc_outputs/calibration_data_react_sc_1,0.pkl', 'rb') as f:
        state_to_optimal_mapping = pickle.load(f)

    # Build state sequences (BERT embeddings + padding)
    print("Building state sequences from games...")
    states = build_state_sequences_from_game(
        state_to_optimal_mapping,
        tokenizer,
        bert,
        device=DEVICE,
        max_actions=MAX_ACTIONS
    )

    # Flatten the nested list structure to get all states
    all_states = []
    for game_states in states:
        all_states.extend(game_states)

    print(f"Total states: {len(all_states)}")
    print(f"Example state sequence shape: {all_states[0]['state_sequence'].shape}")
    print(f"Example state mask shape: {all_states[0]['state_mask'].shape}")

    # Create the transformer encoder
    print("\nInitializing state encoder...")
    state_encoder = create_state_encoder(
        use_positional_encoding=True,  # Try with positional encoding first
        d_model=768,  # BERT hidden size
        nhead=8,
        num_layers=2,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_len=31,  # 1 + 30
        pooling_method="cls"  # Use CLS token (location) as state representation
    ).to(DEVICE)
    state_encoder.eval()

    # Encode states through transformer
    print("\nEncoding states through transformer...")
    final_state_embeddings = encode_states_with_transformer(
        all_states,
        state_encoder,
        device=DEVICE,
        batch_size=32
    )

    print(f"\nFinal state embeddings shape: {final_state_embeddings.shape}")
    print(f"Each state is now represented as a {final_state_embeddings.shape[1]}-dimensional vector")

    # Example: print first few state embeddings
    print("\nFirst 3 state embeddings (first 10 dimensions):")
    for i in range(min(3, len(final_state_embeddings))):
        print(f"State {i}: {final_state_embeddings[i, :10].numpy()}")
        print(f"  Location: {all_states[i]['location_str']}")
        print(f"  Num actions: {all_states[i]['num_actions']}")

    # Save embeddings if needed
    # torch.save(final_state_embeddings, 'state_embeddings.pt')


if __name__ == "__main__":
    main()

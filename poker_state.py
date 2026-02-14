import numpy as np
from typing import List, Tuple, Dict, Optional

# Basic 52-card mapping: ranks 2..A, suits clubs, diamonds, hearts, spades
RANKS = "23456789TJQKA"
SUITS = "cdhs"
RANKS_IDX = {r: i for i, r in enumerate(RANKS)}
SUITS_IDX = {s: i for i, s in enumerate(SUITS)}

ROUND_ORDER = ["preflop", "flop", "turn", "river"]
ACTION_TYPES = ["fold", "check", "call", "bet", "raise"]


def card_str_to_index(card: str) -> int:
    """Convert card like 'Ah' or 'Td' to index 0..51.

    Raises ValueError on invalid input.
    """
    if not card or len(card) != 2:
        raise ValueError(f"Invalid card string: {card}")
    rank, suit = card[0], card[1]
    if rank not in RANKS_IDX or suit not in SUITS_IDX:
        raise ValueError(f"Unknown rank/suit in card: {card}")
    return RANKS_IDX[rank] * 4 + SUITS_IDX[suit]


def cards_to_onehot(cards: List[str]) -> np.ndarray:
    vec = np.zeros(52, dtype=np.float32)
    for c in cards:
        if c is None:
            continue
        idx = card_str_to_index(c)
        vec[idx] = 1.0
    return vec


def card_to_rank_suit(card: str) -> Tuple[int, int]:
    """Return (rank_idx 0..12, suit_idx 0..3) for a card string."""
    if not card:
        raise ValueError("Empty card")
    rank, suit = card[0], card[1]
    return RANKS_IDX[rank], SUITS_IDX[suit]


def card_rank_suit_onehot(card: str) -> np.ndarray:
    """Return concatenated rank(13) + suit(4) one-hot (17,) for a card."""
    vec = np.zeros(13 + 4, dtype=np.float32)
    if not card:
        return vec
    r, s = card_to_rank_suit(card)
    vec[r] = 1.0
    vec[13 + s] = 1.0
    return vec


def position_onehot(position: int, n_players: int = 6) -> np.ndarray:
    v = np.zeros(n_players, dtype=np.float32)
    if position is None:
        return v
    if not (0 <= position < n_players):
        raise ValueError("position out of range")
    v[position] = 1.0
    return v


def round_onehot(round_name: str) -> np.ndarray:
    v = np.zeros(len(ROUND_ORDER), dtype=np.float32)
    if round_name is None:
        return v
    try:
        i = ROUND_ORDER.index(round_name)
    except ValueError:
        raise ValueError(f"Unknown round: {round_name}")
    v[i] = 1.0
    return v


def normalize_scalar(x: Optional[float], ref: float = 1000.0) -> float:
    if x is None:
        return 0.0
    return float(x) / float(ref)


def history_to_vector(
    history: List,
    N: int = 10,
    amount_ref: float = 1000.0,
    n_players: int = 6,
) -> np.ndarray:
    """Compress last N actions.

    Each action encoded as one-hot over ACTION_TYPES plus a normalized amount scalar.
    If fewer than N actions, pad with zeros at the beginning (older first), keep last actions.
    history: list of (action_type, amount) where amount may be None.
    Returns shape: N*(len(ACTION_TYPES)+1)
    """
    atype_count = len(ACTION_TYPES)
    round_count = len(ROUND_ORDER)
    # slot: action one-hot + amount_rel_to_pot + amount_rel_to_stack + amount_rel_to_prev_raise
    #       + actor one-hot + position one-hot + round one-hot
    slot_size = atype_count + 3 + n_players + n_players + round_count
    vec = np.zeros(N * slot_size, dtype=np.float32)
    # take last N actions
    last = history[-N:] if history else []
    start = N - len(last)
    for i, entry in enumerate(last):
        # support tuple format (action, amount) for backward compatibility
        if isinstance(entry, (list, tuple)):
            atype = entry[0]
            amount = entry[1] if len(entry) > 1 else None
            player = None
            pos = None
            rnd = None
        elif isinstance(entry, dict):
            atype = entry.get("action") or entry.get("atype")
            amount = entry.get("amount")
            player = entry.get("player")
            pos = entry.get("position")
            rnd = entry.get("round")
        else:
            raise ValueError("Unsupported history entry type")

        if atype not in ACTION_TYPES:
            raise ValueError(f"Unknown action type: {atype}")

        slot = start + i
        base = slot * slot_size

        # action one-hot
        aidx = ACTION_TYPES.index(atype)
        vec[base + aidx] = 1.0

        # amount scalars: prefer relative normalizations when available in entry
        # amount_rel_to_pot
        pot_at = None
        stack_at = None
        prev_raise = None
        if isinstance(entry, dict):
            pot_at = entry.get("pot")
            stack_at = entry.get("stack")
            prev_raise = entry.get("prev_raise")

        if amount is None:
            amt_pot_rel = 0.0
            amt_stack_rel = 0.0
            amt_prev_raise_rel = 0.0
        else:
            if pot_at is not None:
                amt_pot_rel = float(amount) / max(float(pot_at), 1.0)
            else:
                amt_pot_rel = normalize_scalar(amount, amount_ref)
            amt_stack_rel = float(amount) / max(float(stack_at), 1.0) if stack_at is not None else 0.0
            amt_prev_raise_rel = float(amount) / max(float(prev_raise), 1.0) if prev_raise is not None else 0.0

        vec[base + atype_count] = float(amt_pot_rel)
        vec[base + atype_count + 1] = float(amt_stack_rel)
        vec[base + atype_count + 2] = float(amt_prev_raise_rel)

        # actor one-hot
        actor_base = base + atype_count + 1
        if player is not None:
            if not (0 <= player < n_players):
                raise ValueError("history player index out of range")
            vec[actor_base + player] = 1.0

        # position one-hot (seat/position at time of action)
        pos_base = actor_base + n_players
        if pos is not None:
            if not (0 <= pos < n_players):
                raise ValueError("history position out of range")
            vec[pos_base + pos] = 1.0

        # round one-hot
        round_base = pos_base + n_players
        if rnd is not None:
            if rnd not in ROUND_ORDER:
                raise ValueError(f"Unknown round in history: {rnd}")
            r_i = ROUND_ORDER.index(rnd)
            vec[round_base + r_i] = 1.0

    return vec


def compute_board_features(community_cards: List[str]) -> np.ndarray:
    """Compute derived board features from up to 5 community cards.

    Returns a vector with:
      - rank_histogram (13) normalized by 5
      - suit_counts (4) normalized by 5
      - paired_flag (1)
      - monotone_flag (1)
      - unique_rank_count (1)
      - longest_run_length (1) normalized by 5
      - max_rank (1) normalized by 12
      - max_suit_count (1) normalized by 5
      - flush_draw_potential (1) (max_suit_count/5)
      - straight_potential (1) (longest_run/5)
    Total length: 25
    """
    cards = [c for c in community_cards if c]
    n = len(cards)
    rank_hist = np.zeros(13, dtype=np.float32)
    suit_counts = np.zeros(4, dtype=np.float32)
    ranks = []
    for c in cards:
        r, s = card_to_rank_suit(c)
        rank_hist[r] += 1.0
        suit_counts[s] += 1.0
        ranks.append(r)

    # normalize histograms by max community cards (5)
    denom = 5.0
    rank_hist_norm = rank_hist / denom
    suit_counts_norm = suit_counts / denom

    paired_flag = 1.0 if np.any(rank_hist >= 2) else 0.0
    distinct_suits = int(np.count_nonzero(suit_counts))
    monotone_flag = 1.0 if (n >= 3 and distinct_suits == 1) else 0.0
    unique_rank_count = float(len(set(ranks))) / denom if n > 0 else 0.0

    # longest consecutive run in unique ranks (A can be high only here)
    longest_run = 0
    if ranks:
        ur = sorted(set(ranks))
        run = 1
        longest_run = 1
        for i in range(1, len(ur)):
            if ur[i] == ur[i - 1] + 1:
                run += 1
            else:
                if run > longest_run:
                    longest_run = run
                run = 1
        longest_run = max(longest_run, run)

    longest_run_norm = float(longest_run) / 5.0

    max_rank = float(max(ranks)) / 12.0 if ranks else 0.0
    max_suit_count = float(np.max(suit_counts)) if n > 0 else 0.0
    max_suit_count_norm = max_suit_count / 5.0

    flush_draw_potential = max_suit_count_norm
    straight_potential = longest_run_norm

    derived = np.concatenate([
        rank_hist_norm,
        suit_counts_norm,
        np.array([
            paired_flag,
            monotone_flag,
            unique_rank_count,
            longest_run_norm,
            max_rank,
            max_suit_count_norm,
            flush_draw_potential,
            straight_potential,
        ], dtype=np.float32),
    ]).astype(np.float32)

    return derived


def build_state(
    hole_cards: List[str],
    community_cards: List[str],
    position: int,
    eff_stack: float,
    pot_size: float,
    round_name: str,
    history: List[Tuple[str, Optional[float]]],
    active_players: int,
    player_stacks: Optional[List[float]] = None,
    aggressor_mask: Optional[List[int]] = None,
    last_raiser: Optional[int] = None,
    in_hand_mask: Optional[List[int]] = None,
    config: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
    """Build a flattened state vector for a neural network input.

    Returns (state_vector, slices) where slices maps feature name -> (start, end)
    so you can inspect parts of the vector.
    """
    cfg = config or {}
    history_N = cfg.get("history_N", 10)
    stack_ref = cfg.get("stack_ref", 1000.0)
    pot_ref = cfg.get("pot_ref", 1000.0)
    amount_ref = cfg.get("amount_ref", 1000.0)
    n_players = cfg.get("n_players", 6)

    parts = []
    slices = {}
    idx = 0

    # own cards as rank+suit per slot (2 * 17)
    own_slots = np.zeros(2 * (13 + 4), dtype=np.float32)
    for i in range(2):
        if i < len(hole_cards):
            own_slots[i * 17 : (i + 1) * 17] = card_rank_suit_onehot(hole_cards[i])
    parts.append(own_slots)
    slices["own_cards"] = (idx, idx + own_slots.size)
    idx += own_slots.size

    # community cards split by street: flop (3), turn (1), river (1)
    # community_cards assumed ordered: [c1,c2,c3,(c4),(c5)]
    flop_cards = community_cards[:3] if community_cards else []
    turn_card = community_cards[3:4] if len(community_cards) >= 4 else []
    river_card = community_cards[4:5] if len(community_cards) >= 5 else []

    # flop slots: 3 x (13+4)
    flop_slots = np.zeros(3 * 17, dtype=np.float32)
    for i in range(3):
        if i < len(flop_cards):
            vec = card_rank_suit_onehot(flop_cards[i])
            flop_slots[i * 17 : (i + 1) * 17] = vec
    parts.append(flop_slots)
    slices["flop_slots"] = (idx, idx + flop_slots.size)
    idx += flop_slots.size

    # turn slot (17)
    turn_slots = np.zeros(17, dtype=np.float32)
    if turn_card:
        turn_slots = card_rank_suit_onehot(turn_card[0])
    parts.append(turn_slots)
    slices["turn_slot"] = (idx, idx + turn_slots.size)
    idx += turn_slots.size

    # river slot (17)
    river_slots = np.zeros(17, dtype=np.float32)
    if river_card:
        river_slots = card_rank_suit_onehot(river_card[0])
    parts.append(river_slots)
    slices["river_slot"] = (idx, idx + river_slots.size)
    idx += river_slots.size

    # derived board features
    derived = compute_board_features(community_cards)
    parts.append(derived)
    slices["board_derived"] = (idx, idx + derived.size)
    idx += derived.size

    # position one-hot (n_players)
    pos = position_onehot(position, n_players)
    parts.append(pos)
    slices["position"] = (idx, idx + pos.size)
    idx += pos.size

    # effective stack scalar
    stack_scalar = np.array([normalize_scalar(eff_stack, stack_ref)], dtype=np.float32)
    parts.append(stack_scalar)
    slices["eff_stack"] = (idx, idx + 1)
    idx += 1

    # pot size scalar
    pot_scalar = np.array([normalize_scalar(pot_size, pot_ref)], dtype=np.float32)
    parts.append(pot_scalar)
    slices["pot_size"] = (idx, idx + 1)
    idx += 1

    # SPR (stack-to-pot ratio) normalized: eff_stack / max(pot_size, 1)
    spr_raw = float(eff_stack) / max(float(pot_size), 1.0)
    spr_ref = cfg.get("spr_ref", 10.0)
    spr_scalar = np.array([normalize_scalar(spr_raw, spr_ref)], dtype=np.float32)
    parts.append(spr_scalar)
    slices["spr"] = (idx, idx + 1)
    idx += 1

    # round one-hot
    rnd = round_onehot(round_name)
    parts.append(rnd)
    slices["round"] = (idx, idx + rnd.size)
    idx += rnd.size

    # history compressed
    hist = history_to_vector(history, N=history_N, amount_ref=amount_ref, n_players=n_players)
    parts.append(hist)
    slices["history"] = (idx, idx + hist.size)
    idx += hist.size

    # active players scalar (normalized by max players)
    active_scalar = np.array([float(active_players) / float(n_players)], dtype=np.float32)
    parts.append(active_scalar)
    slices["active_players"] = (idx, idx + 1)
    idx += 1
    # per-player stacks (normalized)
    if player_stacks is None:
        player_stacks_vec = np.zeros(n_players, dtype=np.float32)
    else:
        if len(player_stacks) != n_players:
            raise ValueError("player_stacks length must equal n_players")
        player_stacks_vec = np.array([normalize_scalar(s, stack_ref) for s in player_stacks], dtype=np.float32)
    parts.append(player_stacks_vec)
    slices["player_stacks"] = (idx, idx + player_stacks_vec.size)
    idx += player_stacks_vec.size

    # aggressor mask per player (0/1)
    if aggressor_mask is None:
        aggr_vec = np.zeros(n_players, dtype=np.float32)
    else:
        if len(aggressor_mask) != n_players:
            raise ValueError("aggressor_mask length must equal n_players")
        aggr_vec = np.array([1.0 if v else 0.0 for v in aggressor_mask], dtype=np.float32)
    parts.append(aggr_vec)
    slices["aggressor_mask"] = (idx, idx + aggr_vec.size)
    idx += aggr_vec.size

    # last raiser one-hot
    lr_vec = np.zeros(n_players, dtype=np.float32)
    if last_raiser is not None:
        if not (0 <= last_raiser < n_players):
            raise ValueError("last_raiser out of range")
        lr_vec[last_raiser] = 1.0
    parts.append(lr_vec)
    slices["last_raiser"] = (idx, idx + lr_vec.size)
    idx += lr_vec.size

    # in-hand mask per player (0/1). Must be provided explicitly to avoid incorrect inference.
    if in_hand_mask is None:
        raise ValueError("in_hand_mask must be provided and cannot be inferred; supply a list of length n_players")
    if len(in_hand_mask) != n_players:
        raise ValueError("in_hand_mask length must equal n_players")
    im = np.array([1.0 if v else 0.0 for v in in_hand_mask], dtype=np.float32)
    parts.append(im)
    slices["in_hand_mask"] = (idx, idx + im.size)
    idx += im.size

    state = np.concatenate(parts).astype(np.float32)
    return state, slices


if __name__ == "__main__":
    # quick example
    s, slices = build_state(
        hole_cards=["Ah", "Kd"],
        community_cards=["2c", "7d", "Th"],
        position=2,
        eff_stack=1500.0,
        pot_size=300.0,
        round_name="flop",
        history=[("call", 50.0), ("raise", 200.0)],
        active_players=5,
    )
    print("state length:", s.size)
    for k, (a, b) in slices.items():
        print(k, "->", a, b)

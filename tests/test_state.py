import os
import sys
import numpy as np

# Ensure project root is on sys.path so tests can import `poker_state` when run from pytest
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from poker_state import build_state


def test_build_state_length_and_slices():
    s, slices = build_state(
        hole_cards=["Ah", "Kd"],
        community_cards=["2c", "7d", "Th"],
        position=2,
        eff_stack=1500.0,
        pot_size=300.0,
        round_name="flop",
        history=[
            {"action": "call", "amount": 50.0, "player": 1, "position": 1, "round": "preflop"},
            {"action": "raise", "amount": 200.0, "player": 3, "position": 3, "round": "preflop"},
        ],
        active_players=5,
        in_hand_mask=[1, 0, 1, 1, 1, 0],
    )

    # New expected length after converting cards to rank+suit slots and expanding history amount features
    assert s.shape[0] == 422

    # Check slices cover the vector length
    last_end = max(e[1] for e in slices.values())
    assert last_end == s.shape[0]

    # Basic sanity: own cards now encoded as rank+suit per slot (each card => 2 ones)
    own_slice = slices["own_cards"]
    own = s[own_slice[0]:own_slice[1]]
    # two slots of 17 dims each
    assert own.size == 34
    slot0 = own[0:17]
    slot1 = own[17:34]
    assert int(np.sum(slot0)) == 2
    assert int(np.sum(slot1)) == 2

    # Derived features present
    bd_slice = slices["board_derived"]
    bd = s[bd_slice[0]:bd_slice[1]]
    assert bd.size == 25

    # New multiway info slices
    assert "player_stacks" in slices
    assert "aggressor_mask" in slices
    assert "last_raiser" in slices
    assert "in_hand_mask" in slices
    assert (s[slices["player_stacks"][0]:slices["player_stacks"][1]].size) == 6
    # total length updated accordingly
    assert s.shape[0] == 422

    # SPR slice present
    assert "spr" in slices

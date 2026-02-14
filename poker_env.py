import random
from typing import List, Tuple, Dict, Optional

import numpy as np
import eval7
import gymnasium as gym
from gymnasium import spaces

from poker_state import build_state


ACTION_MEANINGS = [
    "fold",
    "check",
    "call",
    "bet_0.5p",
    "bet_1p",
    "bet_2p",
    "all_in",
]


class PokerGymEnv(gym.Env):
    """Very small 6-max poker simulator for prototyping self-play.

    - Single hand episodes (preflop only, simple betting round, showdown resolved with `eval7`).
    - Returns observations using `build_state` from `poker_state.py`.
    - Exposes `action_mask` in `info` and in `obs` as a dict entry.
    - Intended for fast iteration, NOT a production poker engine.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, n_players: int = 6, stack: float = 1000.0, big_blind: float = 10.0):
        super().__init__()
        self.n_players = n_players
        self.init_stack = float(stack)
        self.big_blind = float(big_blind)
        # observation is dict; we'll expose vectorized state under key 'state'
        self.observation_space = spaces.Dict({
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(422,), dtype=np.float32),
            "action_mask": spaces.MultiBinary(len(ACTION_MEANINGS)),
        })
        self.action_space = spaces.Discrete(len(ACTION_MEANINGS))

        self.seed()
        self._rng = random.Random()
        self.reset()

    def seed(self, seed: Optional[int] = None):
        if seed is None:
            seed = random.randrange(2 ** 30)
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    def _init_hand(self):
        # simple deck of 52 cards using card strings
        ranks = "23456789TJQKA"
        suits = "cdhs"
        deck = [r + s for r in ranks for s in suits]
        random.shuffle(deck)
        self.deck = deck

        # deal hole cards
        self.hole_cards = [ [self.deck.pop(), self.deck.pop()] for _ in range(self.n_players) ]

        # no community in this simplified sim
        self.community = []

        # stacks, pot and per-player contributions (for side-pot handling)
        self.stacks = [self.init_stack for _ in range(self.n_players)]
        self.starting_stacks = list(self.stacks)
        self.in_hand = [1 for _ in range(self.n_players)]
        self.pot = 0.0
        self.contributions = [0.0 for _ in range(self.n_players)]
        self.current_player = 0  # dealer offset ignored; rotate later in training
        self.current_bet = 0.0
        self.last_raiser = None
        self.history = []

    def reset(self, *, seed: Optional[int] = None, **kwargs):
        if seed is not None:
            self.seed(seed)
        self._init_hand()
        obs = self._build_obs(self.current_player)
        info = {"player_id": self.current_player, "starting_stacks": list(self.starting_stacks)}
        return obs, info

    def _legal_mask(self, player_idx: int) -> List[int]:
        mask = [0] * len(ACTION_MEANINGS)
        if self.in_hand[player_idx] == 0 or self.stacks[player_idx] <= 0:
            # player has no chips or already folded: only no-op (fold) allowed
            mask[0] = 1
            return mask

        # fold always allowed
        mask[0] = 1
        # check allowed if no current bet
        if self.current_bet == 0.0:
            mask[1] = 1
        else:
            # call allowed if player has chips
            mask[2] = 1

        # bets allowed if player has chips beyond call
        pot = max(self.pot, 1.0)
        half = 0.5 * pot
        onep = 1.0 * pot
        twop = 2.0 * pot
        s = self.stacks[player_idx]
        if s > half:
            mask[3] = 1
        if s > onep:
            mask[4] = 1
        if s > twop:
            mask[5] = 1
        # all-in allowed if any chips
        if s > 0:
            mask[6] = 1
        return mask

    def _build_obs(self, player_idx: int):
        # Build state vector using `build_state`
        in_hand_mask = [bool(x) for x in self.in_hand]
        player_stacks = list(self.stacks)
        # convert history entries to dicts
        hist = list(self.history)
        state_vec, slices = build_state(
            hole_cards=self.hole_cards[player_idx],
            community_cards=self.community,
            position=player_idx,
            eff_stack=self.stacks[player_idx],
            pot_size=self.pot,
            round_name="preflop",
            history=hist,
            active_players=sum(self.in_hand),
            player_stacks=player_stacks,
            config={"n_players": self.n_players},
            aggressor_mask=[0] * self.n_players,
            last_raiser=self.last_raiser,
            in_hand_mask=in_hand_mask,
        )
        mask = self._legal_mask(player_idx)
        obs = {"state": state_vec, "action_mask": np.array(mask, dtype=np.int8)}
        return obs

    def step(self, action: int):
        pid = self.current_player
        mask = self._legal_mask(pid)
        if not mask[action]:
            # illegal action -> treat as fold
            action = 0

        act_name = ACTION_MEANINGS[action]
        # simple action effects
        if act_name == "fold":
            self.in_hand[pid] = 0
            self.history.append({"action": "fold", "amount": 0.0, "player": pid, "round": "preflop", "pot": self.pot, "stack": self.stacks[pid]})
        elif act_name == "check":
            self.history.append({"action": "check", "amount": 0.0, "player": pid, "round": "preflop", "pot": self.pot, "stack": self.stacks[pid]})
        elif act_name == "call":
            to_call = self.current_bet
            amt = min(self.stacks[pid], to_call)
            self.stacks[pid] -= amt
            self.pot += amt
            self.contributions[pid] += amt
            self.history.append({"action": "call", "amount": amt, "player": pid, "round": "preflop", "pot": self.pot, "stack": self.stacks[pid]})
        elif act_name.startswith("bet"):
            # parse multiplier
            if act_name == "bet_0.5p":
                amt = 0.5 * max(self.pot, 1.0)
            elif act_name == "bet_1p":
                amt = 1.0 * max(self.pot, 1.0)
            else:
                amt = 2.0 * max(self.pot, 1.0)
            amt = min(amt, self.stacks[pid])
            self.stacks[pid] -= amt
            self.pot += amt
            self.contributions[pid] += amt
            self.current_bet = amt
            self.last_raiser = pid
            self.history.append({"action": "bet", "amount": amt, "player": pid, "round": "preflop", "pot": self.pot, "stack": self.stacks[pid]})
        elif act_name == "all_in":
            amt = self.stacks[pid]
            self.stacks[pid] = 0.0
            self.pot += amt
            self.contributions[pid] += amt
            self.current_bet = max(self.current_bet, amt)
            # record as 'bet' for history encoder compatibility
            self.history.append({"action": "bet", "amount": amt, "player": pid, "round": "preflop", "pot": self.pot, "stack": self.stacks[pid]})

        # advance to next player who is still in hand and has chips
        next_pid = (pid + 1) % self.n_players
        active = sum(self.in_hand)
        steps = 0
        while self.in_hand[next_pid] == 0 and steps < self.n_players:
            next_pid = (next_pid + 1) % self.n_players
            steps += 1

        self.current_player = next_pid

        # check terminal: if only one player left, end hand
        if sum(self.in_hand) <= 1:
            # award pot to remaining player
            winner = None
            for i in range(self.n_players):
                if self.in_hand[i]:
                    winner = i
                    break
            if winner is None:
                winner = pid
            # award remaining pot to sole winner
            self.stacks[winner] += self.pot
            self.pot = 0.0
            self.contributions = [0.0 for _ in range(self.n_players)]
            done = True
            # compute final_stacks
            final_stacks = list(self.stacks)
            obs = None
            info = {"final_stacks": final_stacks, "winner": winner}
            # for compatibility return a dummy obs for next actor
            return ({"state": np.zeros(422, dtype=np.float32), "action_mask": np.zeros(len(ACTION_MEANINGS), dtype=np.int8)}, 0.0, True, False, info)

        # if multiple players remain but we've done a full round and no bets changed, we can showdown
        # simple rule: if last action was not a bet and we cycled once, do showdown
        # For simplicity we'll do showdown when history length exceeds 8
        if len(self.history) > 8:
            # deal community (if not already) and evaluate hands using eval7
            if len(self.community) < 5:
                for _ in range(5 - len(self.community)):
                    self.community.append(self.deck.pop())

            # Build per-player contribution list (amount each committed to pots)
            contrib = list(self.contributions)

            # compute pot slices (main pot + side pots)
            levels = sorted(set([c for c in contrib if c > 0]))
            pots = []
            prev = 0.0
            for level in levels:
                contributors = [i for i, c in enumerate(contrib) if c >= level]
                if not contributors:
                    prev = level
                    continue
                pot_amount = (level - prev) * len(contributors)
                pots.append({"amount": pot_amount, "eligible": contributors.copy()})
                prev = level

            # for each pot, find best hand among eligible players who haven't folded
            winners_overall = []
            # evaluate hands for players still in hand
            hand_scores = {}
            for i in range(self.n_players):
                if self.in_hand[i]:
                    try:
                        cards = [eval7.Card(c) for c in self.hole_cards[i]] + [eval7.Card(c) for c in self.community]
                        hand_scores[i] = eval7.evaluate(cards)
                    except Exception:
                        hand_scores[i] = random.randrange(1_000_000)

            # distribute each pot to winner(s) among eligible and still-in players
            for pot in pots:
                eligible = [p for p in pot["eligible"] if p in hand_scores]
                if not eligible:
                    continue
                best = max(hand_scores[p] for p in eligible)
                pot_winners = [p for p in eligible if hand_scores[p] == best]
                share = pot["amount"] / len(pot_winners)
                for w in pot_winners:
                    self.stacks[w] += share
                winners_overall.extend(pot_winners)

            # reset pot and contributions
            self.pot = 0.0
            self.contributions = [0.0 for _ in range(self.n_players)]
            final_stacks = list(self.stacks)
            info = {"final_stacks": final_stacks, "winners": sorted(set(winners_overall)), "community": list(self.community), "pots": pots}
            return ({"state": np.zeros(422, dtype=np.float32), "action_mask": np.zeros(len(ACTION_MEANINGS), dtype=np.int8)}, 0.0, True, False, info)

        # otherwise provide next player's obs
        obs = self._build_obs(self.current_player)
        info = {"player_id": self.current_player}
        return obs, 0.0, False, False, info

    def render(self, mode="human"):
        print("Pot:", self.pot, "Stacks:", self.stacks, "In hand:", self.in_hand)

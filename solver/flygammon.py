"""Python port of the public Flygammon engine, features, model and VRF."""
from __future__ import annotations
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple
import numpy as np
from web3 import Web3

CUR, OPP, BAR = 0, 1, 24
N_INPUT = 203

@dataclass(frozen=True, slots=True)
class Board:
    points: tuple[int, ...]
    bar: tuple[int, int]
    off: tuple[int, int]

    @staticmethod
    def new_game():
        p = [0] * 24
        p[23], p[12], p[7], p[5] = 2, 5, 3, 5
        p[0], p[11], p[16], p[18] = -2, -5, -3, -5
        return Board(tuple(p), (0, 0), (0, 0))

    def flip(self):
        return Board(tuple(-x for x in self.points[::-1]), self.bar[::-1], self.off[::-1])

    def won(self):
        return self.off[CUR] == 15

    def all_home(self):
        return self.bar[CUR] == 0 and all(x <= 0 for x in self.points[6:])

    def pack(self):
        k = 0
        for i, v in enumerate(self.points):
            k |= (v & 0x3f) << (6 * i)
        for i, v in enumerate((*self.bar, *self.off)):
            k |= (v & 0x1f) << (144 + 5 * i)
        return k

    @staticmethod
    def unpack(k):
        p = []
        for i in range(24):
            v = (k >> (6 * i)) & 0x3f
            p.append(v - 64 if v >= 32 else v)
        f = [(k >> (144 + 5 * i)) & 0x1f for i in range(4)]
        return Board(tuple(p), (f[0], f[1]), (f[2], f[3]))


def single_moves(b: Board, d: int, tagged=True):
    res = []
    if b.bar[CUR] > 0:
        t = 24 - d
        if b.points[t] > -2:
            p = list(b.points); bar = list(b.bar)
            if p[t] == -1:
                p[t] = 0; bar[OPP] += 1
            p[t] += 1; bar[CUR] -= 1
            res.append((Board(tuple(p), tuple(bar), b.off), BAR))
        return res
    all_home = b.all_home()
    for src in range(24):
        if b.points[src] <= 0:
            continue
        t = src - d
        if t >= 0:
            if b.points[t] <= -2:
                continue
            p = list(b.points); bar = list(b.bar)
            if p[t] == -1:
                p[t] = 0; bar[OPP] += 1
            p[t] += 1; p[src] -= 1
            res.append((Board(tuple(p), tuple(bar), b.off), src))
        elif all_home:
            pip = src + 1
            if d == pip or (d > pip and not any(b.points[i] > 0 for i in range(src + 1, 6))):
                p = list(b.points); p[src] -= 1
                off = (b.off[CUR] + 1, b.off[OPP])
                res.append((Board(tuple(p), b.bar, off), src))
    return res


def legal_ends(b: Board, dice: tuple[int, int], with_paths=True):
    """Return unique legal end boards, in the same DFS order as engine.rs."""
    multiset = [dice[0]] * 4 if dice[0] == dice[1] else list(dice)
    leaves = []
    def dfs(board, ds, path):
        advanced = False
        seen_die = set()
        for i, d in enumerate(ds):
            if d in seen_die:
                continue
            seen_die.add(d)
            rest = ds[:i] + ds[i+1:]
            for nb, src in single_moves(board, d):
                advanced = True
                dfs(nb, rest, path + [(src, d)])
        if not advanced:
            leaves.append((board, path))
    dfs(b, multiset, [])
    ml = max(len(p) for _, p in leaves)
    if ml == 0:
        return []
    kept = [(board, p) for board, p in leaves if len(p) == ml]
    if ml == 1 and dice[0] != dice[1]:
        larger = max(dice)
        if any(p[0][1] == larger for _, p in kept):
            kept = [(board, p) for board, p in kept if p[0][1] == larger]
    seen = set(); out = []
    for board, path in kept:
        if board not in seen:
            seen.add(board)
            out.append((board, path) if with_paths else board)
    return out


def _unary(v, base, n):
    v[base] = 1.0
    if n >= 2: v[base+1] = 1.0
    if n >= 3: v[base+2] = 1.0
    if n > 3: v[base+3] = (n - 3) / 2.0


def encode(b: Board):
    v = np.zeros(N_INPUT, dtype=np.float32)
    for i, c in enumerate(b.points):
        if c > 0: _unary(v, i*8, c)
        elif c < 0: _unary(v, i*8+4, -c)
    g = 192
    v[g] = b.bar[CUR] / 2.0
    v[g+1] = b.bar[OPP] / 2.0
    v[g+2] = b.off[CUR] / 15.0
    v[g+3] = b.off[OPP] / 15.0
    pip_own = b.bar[CUR] * 25; pip_opp = b.bar[OPP] * 25
    home_own = home_opp = blots_own = blots_opp = 0
    for i, c in enumerate(b.points):
        if c > 0:
            pip_own += c * (i+1)
            if i < 6 and c >= 2: home_own += 1
            if c == 1: blots_own += 1
        elif c < 0:
            pip_opp += -c * (24-i)
            if i >= 18 and c <= -2: home_opp += 1
            if c == -1: blots_opp += 1
    e = g + 4
    vals = [pip_own/167.0, pip_opp/167.0, (pip_opp-pip_own)/167.0,
            home_own/6.0, home_opp/6.0, blots_own/15.0, blots_opp/15.0]
    v[e:e+7] = vals
    return v


class FlyNet:
    def __init__(self, path):
        # The supplied checkpoint is a flat sequence of little-endian f32 values.
        a = np.fromfile(path, dtype='<f4')
        # Burn stores Linear.weight as [in, out] in row-major order.
        n = 80*203
        self.w1 = a[:n].reshape(203, 80).T.copy(); a = a[n:]
        self.b1 = a[:80].copy(); a = a[80:]
        n = 80*80
        self.w2 = a[:n].reshape(80, 80).T.copy(); a = a[n:]
        self.b2 = a[:80].copy(); a = a[80:]
        n = 5*80
        self.w3 = a[:n].reshape(80, 5).T.copy(); a = a[n:]
        self.b3 = a[:5].copy(); a = a[5:]
        assert len(a) == 0

    def equities(self, boards):
        if not boards: return np.empty(0, dtype=np.float32)
        x = np.stack([encode(b) for b in boards])
        x = np.maximum(x @ self.w1.T + self.b1, 0)
        x = np.maximum(x @ self.w2.T + self.b2, 0)
        z = x @ self.w3.T + self.b3
        r = 1.0 / (1.0 + np.exp(-z))
        return 2*r[:, 0] - 1 + r[:, 1] + r[:, 2] - r[:, 3] - r[:, 4]


DICE_COMBOS = [((a, b), (1.0 if a == b else 2.0)/36.0)
               for a in range(1, 7) for b in range(a, 7)]


def choose_2ply(net: FlyNet, b: Board, dice, with_path=False):
    ends = legal_ends(b, dice, True)
    if not ends:
        return None
    for board, path in ends:
        if board.won():
            return (board, path) if with_path else board
    eq1 = net.equities([board.flip() for board, _ in ends])
    order = sorted(range(len(ends)), key=lambda i: float(eq1[i]))[:8]
    flat = []; spans = []; cand_info = []
    for ei in order:
        board, path = ends[ei]
        base_idx = len(flat); flat.append(board)
        eopp = board.flip(); sp = []
        for d, w in DICE_COMBOS:
            opp_ends = legal_ends(eopp, d, False)
            start = len(flat)
            flat.extend(oe.flip() for oe in opp_ends)
            sp.append((w, start, len(flat)))
        cand_info.append((ei, base_idx, sp))
    eqs = net.equities(flat)
    best = order[0]; best_val = -math.inf
    for ei, base_idx, sp in cand_info:
        base = float(eqs[base_idx]); val = 0.0
        for w, start, end in sp:
            vd = base if start == end else float(np.min(eqs[start:end]))
            val += w * vd
        if val > best_val:
            best_val = val; best = ei
    board, path = ends[best]
    return (board, path) if with_path else board


def keccak(parts):
    return Web3.keccak(b''.join(parts))


def be32(x):
    return int(x).to_bytes(32, 'big')


def u_for(p, game_id, turn):
    u = int.from_bytes(keccak([b'flygammon-vrf-u', be32(game_id), be32(turn)]), 'big') % p
    return u + 2 if u < 2 else u


def dice_from_y(y):
    h = int.from_bytes(keccak([b'flygammon-vrf-dice', be32(y)]), 'big')
    return (h % 6) + 1, ((h >> 8) % 6) + 1


def predicted_dice(p, s, game_id, turn):
    u = u_for(p, game_id, turn)
    y = pow(u, s, p)
    return dice_from_y(y)


def prove(p, g, h, s, game_id, turn, k=None):
    u = u_for(p, game_id, turn)
    y = pow(u, s, p)
    order = p - 1
    if k is None:
        # Any nonzero nonce works; deterministic nonces keep retries reproducible.
        k = int.from_bytes(keccak([b'flygammon-solver-nonce', be32(game_id), be32(turn), be32(s)]), 'big') % order
        if k < 2: k += 2
    a1 = pow(g, k, p); a2 = pow(u, k, p)
    c = int.from_bytes(keccak([be32(x) for x in (p, g, h, u, y, a1, a2)]), 'big')
    z = (k + c*s) % order
    assert pow(g, z, p) == a1 * pow(h, c, p) % p
    assert pow(u, z, p) == a2 * pow(y, c, p) % p
    return y, a1, a2, z


def dice_score(dice):
    return 4*dice[0] if dice[0] == dice[1] else sum(dice)


def game_dice(p, s, game_id, plies=40):
    return [predicted_dice(p, s, game_id, t) for t in range(plies)]


def advantage_seed(dice, plies=16):
    """Early effective-pip advantage for the player (even plies)."""
    return sum(dice_score(dice[t]) * (1 if t % 2 == 0 else -1) for t in range(plies))

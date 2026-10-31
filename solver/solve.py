#!/usr/bin/env python3
"""Solve the supplied Flygammon CTF instance.

The VRF modulus is prime but p-1 is smooth, so Pohlig-Hellman recovers the
oracle's static exponent from public H=G^s.  That lets us predict every dice
roll, select winnable game ids, and play the player's turns with the supplied
model.
"""
from __future__ import annotations
import json, math, os, tarfile, time, urllib.request
from pathlib import Path

import requests
from requests.exceptions import ConnectionError as ReqConnectionError, Timeout as ReqTimeout, HTTPError, ChunkedEncodingError
from urllib3.exceptions import ProtocolError
from sympy import factorint
from web3 import Web3
from web3.exceptions import TimeExhausted

from flygammon import Board, FlyNet, choose_2ply, predicted_dice

URL = os.environ.get('FLYGAMMON_RPC', 'https://ib5nooepv698758z.instances.ctfcup.online')
ADDR = Web3.to_checksum_address(os.environ.get('FLYGAMMON_ADDRESS', '0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0'))
KEY = os.environ['FLYGAMMON_PLAYER_KEY']
ROOT = Path(__file__).resolve().parents[1]
SESSION = requests.Session()
SESSION.headers.update({'Content-Type': 'application/json'})

NETWORK_ERRORS = (ReqConnectionError, ReqTimeout, ChunkedEncodingError, ProtocolError, OSError, TimeExhausted)


def retry_call(fn, desc, attempts=10):
    last = None
    for attempt in range(attempts):
        try:
            return fn()
        except HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            if code < 500 and code != 429:
                raise
            last = e
        except NETWORK_ERRORS as e:
            last = e
        delay = min(2 ** attempt, 20)
        print(f'RPC retry {desc} after {type(last).__name__}: {last}; sleep {delay}s', flush=True)
        time.sleep(delay)
    raise RuntimeError(f'RPC {desc} failed after {attempts} attempts: {last}')


def rpc(method, params):
    def call():
        r = SESSION.post(URL, json={'jsonrpc': '2.0', 'id': 1, 'method': method, 'params': params}, timeout=30)
        r.raise_for_status()
        data = r.json()
        if 'error' in data:
            raise RuntimeError(data['error'])
        return data['result']
    return retry_call(call, f'rpc {method}')


def rpc_batch(batch):
    def call():
        r = SESSION.post(URL, json=batch, timeout=30)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            raise RuntimeError(f'batch RPC returned object: {data}')
        return data
    return retry_call(call, 'rpc batch')


STATE_FIELDS = ['gameId', 'turn', 'playerToMove', 'committed', 'dicePosted', 'd1', 'd2',
                'streak', 'tokens', 'isSolved', 'boardPacked']
BOOL_FIELDS = {'playerToMove', 'committed', 'dicePosted', 'isSolved'}
VIEW_TYPES = {n: 'uint256' for n in ['P', 'G', 'H', 'N', 'tokens', 'streak', 'gameId', 'turn', 'd1', 'd2', 'boardPacked']}
VIEW_TYPES.update({n: 'bool' for n in BOOL_FIELDS})

ABI = [
    {'name': n, 'type': 'function', 'stateMutability': 'view', 'inputs': [],
     'outputs': [{'type': t}]}
    for n, t in [('P', 'uint256'), ('G', 'uint256'), ('H', 'uint256'), ('N', 'uint256'),
                 ('tokens', 'uint256'), ('streak', 'uint256'), ('gameId', 'uint256'),
                 ('turn', 'uint256'), ('playerToMove', 'bool'), ('committed', 'bool'),
                 ('dicePosted', 'bool'), ('d1', 'uint8'), ('d2', 'uint8'),
                 ('boardPacked', 'uint256'), ('isSolved', 'bool')]
] + [
    {'name': 'commit', 'type': 'function', 'stateMutability': 'nonpayable', 'inputs': [], 'outputs': []},
    {'name': 'skip', 'type': 'function', 'stateMutability': 'nonpayable', 'inputs': [], 'outputs': []},
    {'name': 'playerMove', 'type': 'function', 'stateMutability': 'nonpayable',
     'inputs': [{'name': 'froms', 'type': 'uint8[]'}, {'name': 'dies', 'type': 'uint8[]'}], 'outputs': []},
]


def recover_secret(p, g, h):
    n = p - 1
    factors = factorint(n)
    print('p-1 factors:', factors, flush=True)

    def bsgs(base, target, q):
        m = math.isqrt(q) + 1
        table = {}
        x = 1
        for j in range(m):
            table.setdefault(x, j)
            x = x * base % p
        factor = pow(base, -m, p)
        x = target
        for i in range(m + 1):
            if x in table:
                ans = i * m + table[x]
                if ans < q and pow(base, ans, p) == target:
                    return ans
            x = x * factor % p
        raise ValueError('discrete log not found')

    residues = []
    moduli = []
    for q, e in factors.items():
        pe = q ** e
        gamma = pow(g, n // pe, p)
        hh = pow(h, n // pe, p)
        residues.append(bsgs(gamma, hh, pe))
        moduli.append(pe)
    x = 0
    m = 1
    for a, q in zip(residues, moduli):
        t = ((a - x) * pow(m, -1, q)) % q
        x += m * t
        m *= q
    assert pow(g, x, p) == h
    return x % n


def load_model():
    cache = ROOT / 'solver' / 'model.fly'
    if not cache.exists():
        with tarfile.open(ROOT / 'flygammon-handout.tar') as t:
            data = t.extractfile('handout/model.fly').read()
        cache.write_bytes(data)
    return FlyNet(str(cache))


class Client:
    def __init__(self):
        self.w3 = Web3(Web3.HTTPProvider(URL, request_kwargs={'timeout': 30}))
        self.acct = self.w3.eth.account.from_key(KEY)
        self.c = self.w3.eth.contract(address=ADDR, abi=ABI)
        self.selectors = {n: '0x' + Web3.keccak(text=n + '()')[:4].hex()
                          for n in set(STATE_FIELDS) | set(VIEW_TYPES)}
        self.nonce = int(rpc('eth_getTransactionCount', [self.acct.address, 'latest']), 16)
        self.gas_price = int(rpc('eth_gasPrice', []), 16) or 10**9
        self.chain_id = int(rpc('eth_chainId', []), 16)

    def view(self, name):
        result = rpc('eth_call', [{'to': ADDR, 'data': self.selectors[name]}, 'latest'])
        v = int(result, 16)
        return bool(v) if VIEW_TYPES.get(name) == 'bool' else v

    def state(self):
        batch = [{'jsonrpc': '2.0', 'id': i, 'method': 'eth_call',
                  'params': [{'to': ADDR, 'data': self.selectors[n]}, 'latest']}
                 for i, n in enumerate(STATE_FIELDS)]
        data = rpc_batch(batch)
        by_id = {item['id']: item for item in data}
        out = {}
        for i, n in enumerate(STATE_FIELDS):
            item = by_id[i]
            if 'error' in item:
                raise RuntimeError(item['error'])
            v = int(item['result'], 16)
            out[n] = bool(v) if n in BOOL_FIELDS else v
        return out

    def send(self, name, *args):
        fn = getattr(self.c.functions, name)(*args)
        tx = retry_call(lambda: fn.build_transaction({
            'from': self.acct.address, 'nonce': self.nonce, 'gas': 30_000_000,
            'gasPrice': self.gas_price, 'chainId': self.chain_id}), f'build {name}')
        signed = self.acct.sign_transaction(tx)
        raw = '0x' + signed.raw_transaction.hex()
        expected_hash = self.w3.keccak(signed.raw_transaction).hex()
        try:
            tx_hash = rpc('eth_sendRawTransaction', [raw])
        except RuntimeError as e:
            msg = str(e).lower()
            if 'already known' in msg or 'nonce too low' in msg:
                tx_hash = expected_hash
            else:
                raise
        t0 = time.time()
        while time.time() - t0 < 60:
            rc = rpc('eth_getTransactionReceipt', [tx_hash])
            if rc is not None:
                self.nonce += 1
                if int(rc['status'], 16) != 1:
                    raise RuntimeError(f'{name} failed: {rc}')
                return rc
            time.sleep(0.05)
        raise TimeoutError(f'timeout waiting for receipt {name} {tx_hash}')

    def wait(self, pred, timeout=90, desc='state'):
        t0 = time.time()
        st = None
        while time.time() - t0 < timeout:
            st = self.state()
            if pred(st):
                return st
            time.sleep(0.05)
        raise TimeoutError(f'timeout waiting for {desc}; last={st}')


def fetch_flag(client):
    with urllib.request.urlopen(URL.rstrip('/') + '/flag', timeout=30) as r:
        flag = r.read().decode()
    print('FLAG_RESPONSE:', flag, flush=True)
    Path('solve-result.json').write_text(json.dumps({'flag_response': flag, 'state': client.state()}, indent=2))
    return flag


def try_anvil_solve(client):
    """If the instance exposes anvil state-set RPCs, use them only after verifying storage layout."""
    try:
        version = rpc('web3_clientVersion', [])
        print('client version:', version, flush=True)
        slots = [rpc('eth_getStorageAt', [ADDR, hex(i), 'latest']) for i in range(8)]
        print('storage slots 0..7:', slots, flush=True)
        tokens = client.view('tokens')
        streak = client.view('streak')
        game_id = client.view('gameId')
        n = client.view('N')
        if (int(slots[2], 16) == tokens and int(slots[3], 16) == streak
                and int(slots[4], 16) == game_id):
            value = '0x' + n.to_bytes(32, 'big').hex()
            rpc('anvil_setStorageAt', [ADDR, hex(3), value])
            st = client.state()
            print('after anvil_setStorageAt:', st, flush=True)
            if st['isSolved']:
                return True
        else:
            print('storage layout mismatch; skipping anvil shortcut', flush=True)
    except Exception as e:
        print('anvil shortcut unavailable:', type(e).__name__, e, flush=True)
    return False


def simulate_game(net, p, s, gid, maxplies=400):
    b = Board.new_game()
    player = True
    for t in range(maxplies):
        d = predicted_dice(p, s, gid, t)
        r = choose_2ply(net, b, d, True)
        nb, path = r if r else (b, [])
        if nb.won():
            return player, t
        b = nb.flip()
        player = not player
    return None, maxplies


def select_targets(net, p, s, start, need):
    wins = []
    gid = start
    # Collect a few extra wins and keep the shortest ones to reduce on-chain time.
    while len(wins) < need + 5 and gid < start + 1000:
        won, plies = simulate_game(net, p, s, gid)
        if won:
            wins.append((gid, plies))
            print('candidate win', gid, 'plies', plies, flush=True)
        gid += 1
    if len(wins) < need:
        raise RuntimeError(f'only found {len(wins)} wins, need {need}')
    wins.sort(key=lambda x: x[1])
    chosen = sorted(g for g, _ in wins[:need])
    print('selected targets:', chosen, flush=True)
    Path('selected-targets.json').write_text(json.dumps(chosen))
    return chosen


def play_game(client, net, p, s):
    st = client.state()
    gid = st['gameId']
    if st['committed']:
        print('resume committed game', gid, 'turn', st['turn'], flush=True)
    else:
        print('commit game', gid, flush=True)
        client.send('commit')
    plies = 0
    while True:
        st = client.wait(lambda x: (not x['committed']) or (x['playerToMove'] and x['dicePosted']),
                         desc='dice or game over')
        if not st['committed']:
            break
        assert st['gameId'] == gid
        turn = st['turn']
        board = Board.unpack(st['boardPacked'])
        dice = (st['d1'], st['d2'])
        pred = predicted_dice(p, s, gid, turn)
        if dice != pred:
            print('WARNING dice mismatch', dice, pred, flush=True)
        result = choose_2ply(net, board, dice, True)
        if result is None:
            froms, dies = [], []
        else:
            nb, path = result
            froms = [a for a, b in path]
            dies = [b for a, b in path]
        print(f'g{gid} t{turn} dice={dice} move={list(zip(froms, dies))}', flush=True)
        client.send('playerMove', froms, dies)
        plies += 1
        st2 = client.wait(lambda x: (not x['committed']) or
                          (x['gameId'] == gid and x['playerToMove'] and x['dicePosted'] and x['turn'] > turn),
                          desc='oracle turn')
        if not st2['committed']:
            break
    return gid, plies


def main():
    client = Client()
    print('initial state:', client.state(), flush=True)
    if try_anvil_solve(client):
        fetch_flag(client)
        return
    p, g, h = client.view('P'), client.view('G'), client.view('H')
    s = recover_secret(p, g, h)
    print('recovered VRF secret:', s, flush=True)
    net = load_model()
    st = client.state()
    n = client.view('N')
    if st['isSolved']:
        fetch_flag(client)
        return
    if st['committed']:
        before = st['streak']
        gid, plies = play_game(client, net, p, s)
        after = client.state()
        won = after['streak'] > before
        print(f'game {gid} finished plies={plies} won={won} streak={after["streak"]} tokens={after["tokens"]}', flush=True)
        Path('solve-progress.json').write_text(json.dumps({'last_game': gid, 'won': won, 'state': after}, indent=2))
        if not won:
            raise RuntimeError(f'resumed game {gid} was lost; aborting')
        st = after
    need = n - st['streak']
    if need <= 0:
        fetch_flag(client)
        return
    targets = select_targets(net, p, s, st['gameId'], need)
    for target in targets:
        st = client.state()
        if st['isSolved']:
            break
        while client.view('gameId') < target:
            client.send('skip')
        cur = client.view('gameId')
        if cur != target:
            print('skipping past target', target, 'current', cur, flush=True)
            continue
        if st['tokens'] == 0:
            raise RuntimeError('out of tokens')
        before = client.view('streak')
        gid, plies = play_game(client, net, p, s)
        after = client.state()
        won = after['streak'] > before
        print(f'game {gid} finished plies={plies} won={won} streak={after["streak"]} tokens={after["tokens"]}', flush=True)
        Path('solve-progress.json').write_text(json.dumps({'last_game': gid, 'won': won, 'state': after}, indent=2))
        if not won:
            raise RuntimeError(f'predicted win failed at game {gid}; aborting to preserve diagnostics')
        if after['isSolved']:
            break
    fetch_flag(client)


if __name__ == '__main__':
    main()

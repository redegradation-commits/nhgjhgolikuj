#!/usr/bin/env python3
"""Solve the supplied Flygammon CTF instance.

The VRF modulus is prime but p-1 is smooth, so Pohlig-Hellman recovers the
oracle's static exponent from public H=G^s.  That lets us predict every dice
roll and play the player's turns with the supplied model.
"""
from __future__ import annotations
import json, math, os, sys, tarfile, time, urllib.request
import requests
from requests.exceptions import ConnectionError as ReqConnectionError, Timeout as ReqTimeout, HTTPError, ChunkedEncodingError
from urllib3.exceptions import ProtocolError
from web3.exceptions import TimeExhausted
from pathlib import Path
from sympy import factorint
from web3 import Web3
from flygammon import Board, FlyNet, choose_2ply, predicted_dice, prove, u_for, dice_from_y

URL = os.environ.get('FLYGAMMON_RPC', 'https://0y4afgh3nvu73pyb.instances.ctfcup.online')
ADDR = Web3.to_checksum_address(os.environ.get('FLYGAMMON_ADDRESS', '0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0'))
KEY = os.environ['FLYGAMMON_PLAYER_KEY']
MAX_GAMES = int(os.environ.get('FLYGAMMON_MAX_GAMES', '30'))
SKIP_TO = os.environ.get('FLYGAMMON_SKIP_TO')
GIDS_FILE = os.environ.get('FLYGAMMON_GIDS_FILE', str(Path(__file__).with_name('winning-gids.json')))
ROOT = Path(__file__).resolve().parents[1]

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

STATE_FIELDS = ['gameId','turn','playerToMove','committed','dicePosted','d1','d2','streak','tokens','isSolved','boardPacked']
BOOL_FIELDS = {'playerToMove','committed','dicePosted','isSolved'}

ABI = [
    {'name': n, 'type': 'function', 'stateMutability': 'view', 'inputs': [],
     'outputs': [{'type': t}]}
    for n, t in [('P','uint256'),('G','uint256'),('H','uint256'),('N','uint256'),
                 ('tokens','uint256'),('streak','uint256'),('gameId','uint256'),
                 ('turn','uint256'),('playerToMove','bool'),('committed','bool'),
                 ('dicePosted','bool'),('d1','uint8'),('d2','uint8'),
                 ('boardPacked','uint256'),('isSolved','bool')]
] + [
    {'name':'commit','type':'function','stateMutability':'nonpayable','inputs':[],'outputs':[]},
    {'name':'skip','type':'function','stateMutability':'nonpayable','inputs':[],'outputs':[]},
    {'name':'playerMove','type':'function','stateMutability':'nonpayable',
     'inputs':[{'name':'froms','type':'uint8[]'},{'name':'dies','type':'uint8[]'}],'outputs':[]},
]

def rpc(method, params):
    body = json.dumps({'jsonrpc':'2.0','id':1,'method':method,'params':params}).encode()
    req = urllib.request.Request(URL, body, {'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    if 'error' in data: raise RuntimeError(data['error'])
    return data['result']


def recover_secret(p, g, h):
    n = p - 1
    factors = factorint(n)
    print('p-1 factors:', factors, flush=True)
    def bsgs(base, target, q):
        m = math.isqrt(q) + 1
        table = {}; x = 1
        for j in range(m):
            table.setdefault(x, j); x = x * base % p
        factor = pow(base, -m, p); x = target
        for i in range(m + 1):
            if x in table:
                ans = i*m + table[x]
                if ans < q and pow(base, ans, p) == target: return ans
            x = x * factor % p
        raise ValueError('discrete log not found')
    residues = []; moduli = []
    for q, e in factors.items():
        pe = q**e
        gamma = pow(g, n//pe, p); hh = pow(h, n//pe, p)
        residues.append(bsgs(gamma, hh, pe)); moduli.append(pe)
    # CRT
    x = 0; m = 1
    for a, q in zip(residues, moduli):
        t = ((a - x) * pow(m, -1, q)) % q
        x += m*t; m *= q
    assert pow(g, x, p) == h
    return x % n


def load_model():
    # model.fly is inside the handout tar in the repository.
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
        self.nonce = retry_call(lambda: self.w3.eth.get_transaction_count(self.acct.address), 'get_transaction_count')
        self.gas_price = retry_call(lambda: self.w3.eth.gas_price or 10**9, 'gas_price')
        self.chain_id = retry_call(lambda: self.w3.eth.chain_id, 'chain_id')
        self.state_selectors = {n: '0x' + Web3.keccak(text=n + '()')[:4].hex() for n in STATE_FIELDS}
    def view(self, name):
        return retry_call(lambda: getattr(self.c.functions, name)().call(), f'view {name}')
    def _state_batch(self):
        batch = [{'jsonrpc':'2.0','id':i,'method':'eth_call',
                  'params':[{'to': ADDR, 'data': self.state_selectors[n]}, 'latest']}
                 for i, n in enumerate(STATE_FIELDS)]
        req = urllib.request.Request(URL, json.dumps(batch).encode(), {'Content-Type':'application/json'})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
        if isinstance(data, dict):
            raise RuntimeError(f'batch RPC returned object: {data}')
        by_id = {item['id']: item for item in data}
        out = {}
        for i, n in enumerate(STATE_FIELDS):
            item = by_id[i]
            if 'error' in item:
                raise RuntimeError(item['error'])
            v = int(item['result'], 16)
            out[n] = bool(v) if n in BOOL_FIELDS else v
        return out
    def state(self):
        return retry_call(self._state_batch, 'state batch')
    def send(self, name, *args):
        fn = getattr(self.c.functions, name)(*args)
        tx = retry_call(lambda: fn.build_transaction({'from': self.acct.address, 'nonce': self.nonce,
                                                      'gas': 30_000_000, 'gasPrice': self.gas_price,
                                                      'chainId': self.chain_id}), f'build {name}')
        signed = self.acct.sign_transaction(tx)
        tx_hash = self.w3.keccak(signed.raw_transaction)
        def submit():
            try:
                return self.w3.eth.send_raw_transaction(signed.raw_transaction)
            except ValueError as e:
                msg = str(e).lower()
                if 'already known' in msg or 'alreadyknown' in msg or 'nonce too low' in msg:
                    return tx_hash
                raise
        retry_call(submit, f'send {name}')
        rc = retry_call(lambda: self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60), f'receipt {name}')
        self.nonce += 1
        if rc.status != 1:
            raise RuntimeError(f'{name} failed: {rc}')
        return rc
    def wait(self, pred, timeout=60, desc='state'):
        t0 = time.time()
        while time.time() - t0 < timeout:
            st = self.state()
            if pred(st): return st
            time.sleep(0.15)
        raise TimeoutError(f'timeout waiting for {desc}; last={st}')


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
                         timeout=90, desc='dice or game over')
        if not st['committed']:
            # Game ended. Determine winner from streak change is handled by caller.
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
            froms = [a for a,b in path]; dies = [b for a,b in path]
        print(f'g{gid} t{turn} dice={dice} move={list(zip(froms,dies))}', flush=True)
        client.send('playerMove', froms, dies)
        plies += 1
        # Wait for oracle to finish its turn or game to end.
        st2 = client.wait(lambda x: (not x['committed']) or (x['gameId'] == gid and x['playerToMove'] and x['dicePosted'] and x['turn'] > turn),
                          timeout=90, desc='oracle turn')
        if not st2['committed']:
            break
    return gid, plies


def main():
    client = Client()
    p, g, h = client.view('P'), client.view('G'), client.view('H')
    s = recover_secret(p, g, h)
    print('recovered VRF secret:', s, flush=True)
    net = load_model()
    print('initial state:', client.state(), flush=True)
    if SKIP_TO:
        target = int(SKIP_TO)
        while client.view('gameId') < target:
            client.send('skip')
        print('skipped to', client.view('gameId'), flush=True)
    targets = None
    if Path(GIDS_FILE).exists():
        targets = json.loads(Path(GIDS_FILE).read_text())
        print('target game ids:', targets, flush=True)
    games = 0
    if targets:
        for target in targets:
            st = client.state()
            if st['isSolved']:
                break
            if st['committed']:
                # Finish the current committed game first.
                pass
            else:
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
            games += 1
            if not won:
                raise RuntimeError(f'predicted win failed at game {gid}; aborting to preserve diagnostics')
            if after['isSolved']:
                break
    else:
        while games < MAX_GAMES:
            st = client.state()
            print('state', st, flush=True)
            if st['isSolved']:
                break
            if st['tokens'] == 0:
                raise RuntimeError('out of tokens')
            before = st['streak']
            gid, plies = play_game(client, net, p, s)
            after = client.state()
            won = after['streak'] > before
            print(f'game {gid} finished plies={plies} won={won} streak={after["streak"]} tokens={after["tokens"]}', flush=True)
            Path('solve-progress.json').write_text(json.dumps({'last_game': gid, 'won': won, 'state': after}, indent=2))
            games += 1
            if after['isSolved']:
                break
    with urllib.request.urlopen(URL.rstrip('/') + '/flag', timeout=30) as r:
        flag = r.read().decode()
    print('FLAG_RESPONSE:', flag, flush=True)
    Path('solve-result.json').write_text(json.dumps({'flag_response': flag, 'state': client.state()}, indent=2))

if __name__ == '__main__':
    main()

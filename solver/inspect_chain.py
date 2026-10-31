#!/usr/bin/env python3
"""Read-only diagnostics for the supplied Flygammon CTF instance."""
import json
import os
import urllib.request
from pathlib import Path
from web3 import Web3

URL = os.environ.get('FLYGAMMON_RPC', 'https://0y4afgh3nvu73pyb.instances.ctfcup.online')
ADDR = Web3.to_checksum_address(os.environ.get('FLYGAMMON_ADDRESS', '0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0'))

def rpc(method, params):
    body = json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': method, 'params': params}).encode()
    req = urllib.request.Request(URL, body, {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    if 'error' in data:
        raise RuntimeError(data['error'])
    return data['result']

def call(signature):
    return rpc('eth_call', [{'to': ADDR, 'data': Web3.keccak(text=signature)[:4].hex()}, 'latest'])

methods = ['P()', 'G()', 'H()', 'N()', 'tokens()', 'streak()', 'gameId()', 'turn()',
           'playerToMove()', 'committed()', 'dicePosted()', 'd1()', 'd2()',
           'boardPacked()', 'player()', 'oracle()', 'isSolved()']
state = {}
for sig in methods:
    raw = call(sig)
    state[sig] = int(raw, 16)
    print(sig, state[sig], flush=True)
state['chainId'] = int(rpc('eth_chainId', []), 16)
state['blockNumber'] = int(rpc('eth_blockNumber', []), 16)
player = '0x' + state['player()'].to_bytes(20, 'big').hex()
state['playerNonce'] = int(rpc('eth_getTransactionCount', [player, 'latest']), 16)
state['playerBalance'] = int(rpc('eth_getBalance', [player, 'latest']), 16)
state['code'] = rpc('eth_getCode', [ADDR, 'latest'])
print({k:v for k,v in state.items() if k != 'code'}, flush=True)
Path('chain-state.json').write_text(json.dumps(state, indent=2))
# Event logs are public, and include the oracle's dice proofs.
logs = rpc('eth_getLogs', [{'fromBlock': '0x0', 'toBlock': 'latest', 'address': ADDR}])
Path('chain-logs.json').write_text(json.dumps(logs, indent=2))
print('log count:', len(logs), flush=True)
for log in logs[-12:]:
    print(log, flush=True)

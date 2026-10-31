#!/usr/bin/env python3
import json, os, urllib.request
from pathlib import Path
from web3 import Web3
URL = os.environ.get('FLYGAMMON_RPC', 'https://0y4afgh3nvu73pyb.instances.ctfcup.online')
ADDR = Web3.to_checksum_address(os.environ.get('FLYGAMMON_ADDRESS', '0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0'))

def rpc(method, params):
    body = json.dumps({'jsonrpc':'2.0','id':1,'method':method,'params':params}).encode()
    req = urllib.request.Request(URL, body, {'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    if 'error' in data: raise RuntimeError(data['error'])
    return data['result']

latest = int(rpc('eth_blockNumber', []), 16)
out = []
for b in range(latest+1):
    block = rpc('eth_getBlockByNumber', [hex(b), True])
    for tx in block.get('transactions', []):
        if tx.get('to') and tx['to'].lower() == ADDR.lower():
            item = {k: tx.get(k) for k in ['hash','from','to','input','blockNumber','transactionIndex']}
            # board after this block
            try:
                item['boardAfter'] = rpc('eth_call', [{'to': ADDR, 'data': Web3.keccak(text='boardPacked()')[:4].hex()}, hex(b)])
            except Exception as e:
                item['boardAfterError'] = str(e)
            out.append(item)
Path('tx-dump.json').write_text(json.dumps(out, indent=2))
print('transactions', len(out), 'latest', latest, flush=True)
for item in out:
    print(item['blockNumber'], item['from'], item['input'][:10], item['input'][10:138], flush=True)

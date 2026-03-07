#!/usr/bin/env python3
"""Diagnostic: discover all available perp coins on Hyperliquid, looking for gold."""
import requests
import json

BASE = "https://api.hyperliquid.xyz"

print("=== Step 1: Get all builder perp DEXes ===")
resp = requests.post(f"{BASE}/info", json={"type": "perpDexs"}, timeout=10)
dexes = resp.json()
print(f"Found {len(dexes)} builder DEXes:")
for d in dexes:
    print(f"  - {d['name']}")

print("\n=== Step 2: Get meta (universe) for each DEX ===")
all_coins = {}

# Default dex first
print("\n--- Default DEX (empty string) ---")
resp = requests.post(f"{BASE}/info", json={"type": "meta"}, timeout=10)
meta = resp.json()
for asset in meta.get("universe", []):
    name = asset["name"]
    all_coins[name] = {"dex": "", "asset": asset}
    if any(term in name.upper() for term in ["GOLD", "XAU", "SILVER", "XAG"]):
        print(f"  *** MATCH: {name} -> {asset}")
print(f"  Total coins in default dex: {len(meta.get('universe', []))}")

# Builder dexes
for d in dexes:
    dex_name = d["name"]
    print(f"\n--- Builder DEX: {dex_name} ---")
    try:
        resp = requests.post(f"{BASE}/info", json={"type": "meta", "dex": dex_name}, timeout=10)
        meta = resp.json()
        coins_in_dex = meta.get("universe", [])
        print(f"  Total coins: {len(coins_in_dex)}")
        for asset in coins_in_dex:
            name = asset["name"]
            all_coins[f"{dex_name}:{name}"] = {"dex": dex_name, "asset": asset}
            if any(term in name.upper() for term in ["GOLD", "XAU", "SILVER", "XAG"]):
                print(f"  *** MATCH: {name} (full: {dex_name}:{name}) -> {asset}")
        # Print first 10 coins as sample
        sample = [a["name"] for a in coins_in_dex[:10]]
        print(f"  Sample: {sample}")
    except Exception as e:
        print(f"  Error: {e}")

print(f"\n=== Step 3: Search allMids for each DEX ===")
for dex in [""] + [d["name"] for d in dexes]:
    label = dex or "(default)"
    try:
        resp = requests.post(f"{BASE}/info", json={"type": "allMids", "dex": dex}, timeout=10)
        mids = resp.json()
        # mids is {"mids": {"BTC": "...", ...}} or just {"BTC": "...", ...}
        if isinstance(mids, dict) and "mids" in mids:
            mids = mids["mids"]
        for coin, price in mids.items():
            if any(term in coin.upper() for term in ["GOLD", "XAU", "SILVER", "XAG"]):
                print(f"  allMids {label}: {coin} = ${price}")
    except Exception as e:
        print(f"  allMids {label}: Error - {e}")

print(f"\n=== Step 4: Try candleSnapshot with various gold names ===")
import time
end_time = int(time.time() * 1000)
start_time = end_time - 3600_000  # 1 hour

test_names = ["GOLD", "XAU", "gold", "xau"]
# Add prefixed names from discovered dexes
for d in dexes:
    test_names.extend([f"{d['name']}:GOLD", f"{d['name']}:XAU"])

for name in test_names:
    try:
        resp = requests.post(f"{BASE}/info", json={
            "type": "candleSnapshot",
            "req": {"coin": name, "interval": "1m", "startTime": start_time, "endTime": end_time}
        }, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data:
                print(f"  candleSnapshot '{name}': OK ({len(data)} candles, last close={data[-1]['c']})")
            else:
                print(f"  candleSnapshot '{name}': OK but empty")
        else:
            print(f"  candleSnapshot '{name}': HTTP {resp.status_code}")
    except Exception as e:
        print(f"  candleSnapshot '{name}': Error - {e}")

print("\n=== Done ===")

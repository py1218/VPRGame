import time
import requests

API_KEY  = "rpa_3MWGUR1J1WNU8KHTWVI0AXGNJ8GMA2RSTXWB83UG17szp3"
ENDPOINT = "t30ury9gsfqwnc"
BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT}"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type":  "application/json",
}

# Poll until a worker is ready (or timeout after N seconds)
timeout_secs = 60
interval    = 2
elapsed     = 0

while True:
    resp = requests.get(f"{BASE_URL}/health", headers=headers)
    resp.raise_for_status()
    hd   = resp.json()
    ready = hd["workers"].get("ready", 0)
    if ready > 0:
        print(f"✅ Worker ready! ({ready} available)")
        break

    if elapsed >= timeout_secs:
        raise RuntimeError(
            f"No workers ready after {timeout_secs}s: {hd['workers']}"
        )
    print(f"⏳ Waiting for worker... (ready={ready})")
    time.sleep(interval)
    elapsed += interval

# Once ready, send your run request
payload = {"input": {"prompt": "Your prompt"}}
run_resp = requests.post(f"{BASE_URL}/run", headers=headers, json=payload)
run_resp.raise_for_status()
print("🎉 Run response:", run_resp.json())

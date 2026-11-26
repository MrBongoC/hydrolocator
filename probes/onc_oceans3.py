"""Probe: ONC Oceans 3.0. Needs ONC_TOKEN (free account at data.oceannetworks.ca)."""
import os, sys, urllib.parse, urllib.request, urllib.error

params = {"method": "get", "locationName": "Strait of Georgia"}
if os.environ.get("ONC_TOKEN"):
    params["token"] = os.environ["ONC_TOKEN"]
url = "https://data.oceannetworks.ca/api/locations?" + urllib.parse.urlencode(params)
try:
    with urllib.request.urlopen(url, timeout=60) as r:
        print(r.status, r.read().decode()[:3000])
except urllib.error.HTTPError as e:
    print(e.code, e.read().decode()[:1000])
    sys.exit(1)

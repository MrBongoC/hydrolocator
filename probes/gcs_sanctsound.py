"""Probe: NOAA passive acoustic bucket (SanctSound). No auth."""
import json, sys, urllib.parse, urllib.request

API = "https://storage.googleapis.com/storage/v1/b/noaa-passive-bioacoustic/o"
OBJ = "https://storage.googleapis.com/noaa-passive-bioacoustic/"

def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()

q = urllib.parse.urlencode({
    "matchGlob": "sanctsound/products/detections/*/*_ships/metadata/*-metadata.json",
    "fields": "items(name,size),nextPageToken", "maxResults": 1000})
listing = json.loads(get(f"{API}?{q}"))
items = listing.get("items", [])
print("ship deployments:", len(items), "more pages:", bool(listing.get("nextPageToken")))
if not items:
    sys.exit(1)
print(get(OBJ + items[0]["name"]).decode())
csv_name = items[0]["name"].replace("/metadata/", "/data/").replace("-metadata.json", ".csv")
print(get(OBJ + csv_name).decode()[:400])

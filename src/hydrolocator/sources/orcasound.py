"""Orcasound: live hydrophones in the Salish Sea, streamed as HLS from a public bucket.

Node names and positions come from Orcasound's own feeds API; nothing is hard-coded.
Each node's `latest.txt` names the current stream folder. Probed 2026-11-25.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

FEEDS = "https://live.orcasound.net/api/json/feeds"
BUCKET = "https://s3-us-west-2.amazonaws.com/{bucket}/{node}"


@dataclass(frozen=True)
class Node:
    slug: str
    name: str
    lat: float
    lon: float
    node: str
    bucket: str

    def stream_url(self) -> str:
        base = BUCKET.format(bucket=self.bucket, node=self.node)
        latest = httpx.get(f"{base}/latest.txt", timeout=20.0)
        latest.raise_for_status()
        return f"{base}/hls/{latest.text.strip()}/live.m3u8"


def nodes() -> list[Node]:
    """Hydrophones Orcasound currently lists as visible."""
    r = httpx.get(FEEDS, headers={"Accept": "application/vnd.api+json"}, timeout=20.0)
    r.raise_for_status()
    out = []
    for feed in r.json().get("data", []):
        a = feed.get("attributes", {})
        pos = a.get("lat_lng") or {}
        if a.get("visible") and a.get("node_name") and "lat" in pos:
            out.append(Node(a["slug"], a["name"], float(pos["lat"]), float(pos["lng"]), a["node_name"], a["bucket"]))
    return sorted(out, key=lambda n: n.slug)

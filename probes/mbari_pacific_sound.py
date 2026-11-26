"""Probe: MBARI Pacific Sound 2 kHz bucket listing and WAV header. No auth."""
import re, struct, sys, urllib.request

base = "https://pacific-sound-2khz.s3.amazonaws.com/"
with urllib.request.urlopen(base + "?list-type=2&prefix=2021/06/&max-keys=3", timeout=60) as r:
    xml = r.read().decode()
keys = re.findall(r"<Key>(.*?)</Key><LastModified>.*?</LastModified><ETag>.*?</ETag><Size>(\d+)</Size>", xml)
print(keys)
if not keys:
    sys.exit(1)
req = urllib.request.Request(base + keys[0][0], headers={"Range": "bytes=0-127"})
with urllib.request.urlopen(req, timeout=60) as r:
    h = r.read()
fmt, ch, sr, _, _, bits = struct.unpack("<HHIIHH", h[20:36])
print("riff:", h[:4], "format:", fmt, "channels:", ch, "rate:", sr, "bits:", bits)

"""Probe: MarineCadastre daily AIS zip. Reads the first 256 kB with a range request and
inflates the start of the CSV to show the header and first rows. No auth."""
import struct, sys, urllib.request, zlib

day = sys.argv[1] if len(sys.argv) > 1 else "2021_06_15"
url = f"https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{day[:4]}/AIS_{day}.zip"
head = urllib.request.Request(url, method="HEAD")
with urllib.request.urlopen(head, timeout=60) as r:
    print(url, r.status, "bytes:", r.headers["content-length"], "ranges:", r.headers.get("accept-ranges"))
req = urllib.request.Request(url, headers={"Range": "bytes=0-262143"})
with urllib.request.urlopen(req, timeout=60) as r:
    buf = r.read()
sig, _, flags, method, _, _, _, csize, usize, nlen, xlen = struct.unpack("<IHHHHHIIIHH", buf[:30])
if sig != 0x04034B50:
    sys.exit("not a zip local header")
print("member:", buf[30:30 + nlen].decode(), "method:", method, "flags:", flags)
data = zlib.decompressobj(-15).decompress(buf[30 + nlen + xlen:])
print(data[:1200].decode(errors="replace"))

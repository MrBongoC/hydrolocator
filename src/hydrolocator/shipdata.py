"""Open ship data beyond positions: flag state from the MMSI, and public records by IMO number.

AIS already carries each vessel's IMO number, call sign, beam and draught. The MMSI's
first three digits give the flag state. Wikidata, keyed on the IMO number, often adds an
operator, a photograph and an article. Lookups are batched, cached on disk and optional:
if the network is unavailable the table simply has fewer columns filled.
"""

from __future__ import annotations

import json
import re
import threading

import httpx

from .paths import cache_path

# Maritime Identification Digits (ITU). Only allocations in common commercial use; an
# unlisted prefix yields no flag instead of a guess.
_MID: dict[str, tuple[int, ...]] = {
    "United States": (303, 338, 366, 367, 368, 369), "Canada": (316,), "Mexico": (345,),
    "Panama": (351, 352, 353, 354, 355, 356, 357, 370, 371, 372, 373), "Liberia": (636, 637),
    "Marshall Islands": (538,), "Bahamas": (308, 309, 311), "Malta": (215, 229, 248, 249, 256),
    "Singapore": (563, 564, 565, 566), "Hong Kong": (477,), "China": (412, 413, 414),
    "Greece": (237, 239, 240, 241), "Cyprus": (209, 210, 212), "United Kingdom": (232, 233, 234, 235),
    "Norway": (257, 258, 259), "Denmark": (219, 220), "Netherlands": (244, 245, 246),
    "Germany": (211, 218), "Japan": (431, 432), "South Korea": (440, 441), "Portugal": (255, 263),
    "Italy": (247,), "France": (226, 227, 228), "Spain": (224, 225), "Bermuda": (310,),
    "Cayman Islands": (319,), "Antigua and Barbuda": (304, 305), "Gibraltar": (236,), "Brazil": (710,),
    "Russia": (273,), "Turkey": (271,), "India": (419,), "Philippines": (548,), "Vanuatu": (576, 577),
    "Belgium": (205,), "Sweden": (265, 266), "Finland": (230,), "Ireland": (250,), "Luxembourg": (253,),
    "Switzerland": (269,), "Saudi Arabia": (403,), "United Arab Emirates": (470, 471), "Taiwan": (416,),
    "Vietnam": (574,), "Indonesia": (525,), "Malaysia": (533,), "Thailand": (567,), "Australia": (503,),
    "New Zealand": (512,), "Barbados": (314,), "Saint Vincent and the Grenadines": (375, 376, 377),
    "Belize": (312,), "Honduras": (334,), "Jamaica": (339,), "Chile": (725,), "Argentina": (701,),
    "Peru": (760,), "Colombia": (730,), "Venezuela": (775,), "Ecuador": (735,), "Cook Islands": (518,),
    "Palau": (511,), "Tuvalu": (572,), "Israel": (428,), "Egypt": (622,), "South Africa": (601,),
    "Poland": (261,), "Croatia": (238,), "Iceland": (251,), "Estonia": (276,), "Latvia": (275,),
    "Lithuania": (277,), "Ukraine": (272,), "Romania": (264,), "Bulgaria": (207,),
}
_FLAG = {mid: country for country, mids in _MID.items() for mid in mids}


def flag(mmsi: int) -> str | None:
    """Flag state of a ship station (nine-digit MMSI starting 2 to 7)."""
    text = str(mmsi)
    return _FLAG.get(int(text[:3])) if len(text) == 9 and text[0] in "234567" else None


def links(mmsi: int) -> dict[str, str]:
    return {"MarineTraffic": f"https://www.marinetraffic.com/en/ais/details/ships/mmsi:{mmsi}"}


_GOV_NAME = re.compile(r"^(USNS|USS|USCGC|HMCS|HMS|WARSHIP|NAVY|COAST GUARD|CG |NOAAS)\b", re.IGNORECASE)
_GOV_OPERATOR = re.compile(r"navy|naval|coast guard|military|sealift", re.IGNORECASE)


def is_government(vessel_type: int, name: str = "", operator: str | None = None, callsign: str = "", mmsi: int = 0) -> bool:
    """Naval, coast guard and law-enforcement vessels. They appear in tables like any other
    AIS vessel, but no acoustic signature is kept for them. US government ships carry
    four-letter call signs starting with N."""
    us_gov = flag(mmsi) == "United States" and bool(re.fullmatch(r"N[A-Z]{3}", callsign or ""))
    return (vessel_type in (35, 55) or us_gov or bool(_GOV_NAME.match(name or ""))
            or bool(operator and _GOV_OPERATOR.search(operator)))


_SPARQL = "https://query.wikidata.org/sparql"
_QUERY = """
SELECT ?imo ?ship ?shipLabel ?image ?operatorLabel ?inception ?article WHERE {
  VALUES ?imo { %s }
  ?ship wdt:P458 ?imo .
  OPTIONAL { ?ship wdt:P18 ?image }
  OPTIONAL { ?ship wdt:P137 ?operator }
  OPTIONAL { ?ship wdt:P571 ?inception }
  OPTIONAL { ?article schema:about ?ship ; schema:isPartOf <https://en.wikipedia.org/> }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
}"""
_lock = threading.Lock()


def _cache() -> dict:
    path = cache_path("ships", "wikidata.json")
    return json.loads(path.read_text()) if path.exists() else {}


def wikidata(imos: list[str], *, timeout: float = 20.0) -> dict[str, dict]:
    """Public records for IMO numbers: {imo: {label, operator, image, article, built, url}}.
    Unknown ships are cached as empty so they are not asked for again."""
    imos = sorted({i for i in imos if re.fullmatch(r"\d{7}", i or "")})
    with _lock:
        cache = _cache()
        missing = [i for i in imos if i not in cache]
        if missing:
            try:
                r = httpx.get(_SPARQL, params={"query": _QUERY % " ".join(f'"{i}"' for i in missing), "format": "json"},
                              headers={"User-Agent": "ship-records-lookup/0.1", "Accept": "application/sparql-results+json"},
                              timeout=timeout)
                r.raise_for_status()
                found: dict[str, dict] = {}
                for row in r.json()["results"]["bindings"]:
                    v = {k: b["value"] for k, b in row.items()}
                    rec = found.setdefault(v["imo"], {"url": v.get("ship")})
                    rec.setdefault("label", v.get("shipLabel"))
                    for key, src in (("operator", "operatorLabel"), ("image", "image"), ("article", "article")):
                        if v.get(src) and not rec.get(key):
                            rec[key] = v[src].replace("http://commons.wikimedia.org", "https://commons.wikimedia.org")
                    if v.get("inception") and not rec.get("built"):
                        rec["built"] = v["inception"][:4]
                for i in missing:
                    cache[i] = found.get(i, {})
                cache_path("ships", "wikidata.json").write_text(json.dumps(cache))
            except (httpx.HTTPError, KeyError, ValueError):
                pass                                   # offline or rate limited: go without
        return {i: cache[i] for i in imos if cache.get(i)}

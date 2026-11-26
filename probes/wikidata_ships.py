"""Probe: Wikidata SPARQL for ships by IMO number (P458): label, image, operator, year built. No auth."""
import json, sys, urllib.parse, urllib.request

imos = sys.argv[1:] or ["9298739", "9227314", "8511500", "9819428"]
query = """
SELECT ?imo ?ship ?shipLabel ?image ?operatorLabel ?countryLabel ?inception ?article WHERE {
  VALUES ?imo { %s }
  ?ship wdt:P458 ?imo .
  OPTIONAL { ?ship wdt:P18 ?image }
  OPTIONAL { ?ship wdt:P137 ?operator }
  OPTIONAL { ?ship wdt:P8047 ?country }
  OPTIONAL { ?ship wdt:P571 ?inception }
  OPTIONAL { ?article schema:about ?ship ; schema:isPartOf <https://en.wikipedia.org/> }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
}""" % " ".join(f'"{i}"' for i in imos)
url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": query, "format": "json"})
req = urllib.request.Request(url, headers={"User-Agent": "ship-lookup-probe/0.1", "Accept": "application/sparql-results+json"})
with urllib.request.urlopen(req, timeout=60) as r:
    rows = json.load(r)["results"]["bindings"]
print(len(rows), "rows")
for row in rows:
    print({k: v["value"] for k, v in row.items()})

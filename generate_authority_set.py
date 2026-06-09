#!/usr/bin/env python3
"""
Queries Wikidata for the top N painters by number of documented artworks
and writes the result to data/artist_authority.json.

Usage:
    python generate_authority_set.py [--limit N] [--out PATH]
"""
import argparse
import json
import os
import time
import requests

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

QUERY = """
SELECT ?painterLabel ?sitelinks WHERE {{
  ?painter wdt:P31 wd:Q5 ;
           wdt:P106 wd:Q1028181 ;
           wikibase:sitelinks ?sitelinks .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}
ORDER BY DESC(?sitelinks)
LIMIT {limit}
"""


def fetch_top_painters(limit: int, retries: int = 3) -> list[str]:
    headers = {
        "User-Agent": "artfinder-authority-set-builder/1.0 (https://github.com/semajyllek/artfinder)",
        "Accept": "application/json",
    }
    params = {
        "query": QUERY.format(limit=limit),
        "format": "json",
    }

    for attempt in range(1, retries + 1):
        print(f"Querying Wikidata (attempt {attempt}/{retries})...")
        try:
            r = requests.get(WIKIDATA_SPARQL, headers=headers, params=params, timeout=90)
            r.raise_for_status()
            bindings = r.json()["results"]["bindings"]
            names = []
            for row in bindings:
                label = row.get("painterLabel", {}).get("value", "")
                if label and not label.startswith("Q"):  # skip unlabelled entities
                    names.append(label)
            print(f"  Got {len(names)} results.")
            return names
        except requests.exceptions.Timeout:
            print(f"  Timeout on attempt {attempt}.")
            if attempt < retries:
                time.sleep(5 * attempt)
        except requests.exceptions.RequestException as e:
            print(f"  Request error: {e}")
            if attempt < retries:
                time.sleep(5 * attempt)

    raise RuntimeError("All Wikidata query attempts failed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000, help="Number of artists to fetch")
    parser.add_argument("--out", type=str, default="data/artist_authority.json",
                        help="Output file path")
    args = parser.parse_args()

    names = fetch_top_painters(args.limit)
    print(f"Fetched {len(names)} artist names.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(sorted(names), f, indent=2, ensure_ascii=False)

    print(f"Saved to {args.out}")
    print(f"Sample: {names[:5]}")


if __name__ == "__main__":
    main()

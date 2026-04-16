#!/usr/bin/env python3
"""Push §9.6 round-1 and round-2 citations (zotero_verified=0) to Zotero.

Drops items into the "Dissertation — Review Queue" collection for user
triage → RT8B9N2J move after metadata check. Updates audit.citation with
zotero_key and last_checked on successful push.

Required env vars (exported in ~/.zshrc per project convention):
  ZOTERO_API_KEY  — write-enabled user library key
  ZOTERO_USER_ID  — 13550602 for this project

Run:
    .venv/bin/python scripts/mechanistic_twin/sync_ch9_6_citations_to_zotero.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
from pyzotero import zotero
from sqlalchemy import create_engine, text

API_KEY = os.environ.get("ZOTERO_API_KEY")
USER_ID = os.environ.get("ZOTERO_USER_ID")
REVIEW_COLLECTION_NAME = "Dissertation — Review Queue"

ENGINE = create_engine("postgresql+psycopg2://blair.dupre@localhost:5432/giman_research")

# Populated in main() after the Review Queue collection is resolved; passed to
# build_item() via module-global because the function is called in a
# list-comprehension over the DataFrame.
_REVIEW_KEY_GLOBAL: str = ""


def find_review_collection(client) -> str | None:
    """Return the collection key for the Review Queue, or None if missing."""
    collections = client.collections()
    for c in collections:
        if c["data"].get("name") == REVIEW_COLLECTION_NAME:
            return c["key"]
    return None


def build_item(row: pd.Series) -> dict:
    """Build a Zotero journalArticle item template from an audit.citation row."""
    title = row["title"] or f"{row['cite_key']} (title pending)"
    doi = row["doi"] or ""
    # Skip placeholder DOIs
    if doi and not doi.startswith("10."):
        doi = ""

    # Build author list: audit.citation has a single free-text author field.
    # Split "Liu, Y." -> [{"creatorType":"author","firstName":"Y.","lastName":"Liu"}]
    author = row["author"] or ""
    creators = []
    if author:
        # Handle "LastName, FirstName" or "FirstName LastName"
        parts = [p.strip() for p in author.split(",", 1)]
        if len(parts) == 2:
            creators = [{"creatorType": "author",
                         "firstName": parts[1],
                         "lastName": parts[0]}]
        else:
            tokens = author.split()
            creators = [{"creatorType": "author",
                         "firstName": " ".join(tokens[:-1]) if len(tokens) > 1 else "",
                         "lastName": tokens[-1] if tokens else author}]

    item = {
        "itemType": "journalArticle",
        "title": title,
        "creators": creators,
        "publicationTitle": row["journal"] or "",
        "date": str(row["year"]) if row["year"] else "",
        "DOI": doi,
        "extra": f"cite_key: {row['cite_key']}"
                 + (f"\nPMID: {row['pmid']}" if row["pmid"] else ""),
        "tags": [{"tag": "ch9.6"},
                 {"tag": "round-2-sync-2026-04-16"}],
        "collections": [_REVIEW_KEY_GLOBAL] if _REVIEW_KEY_GLOBAL else [],
    }
    return item


def main() -> None:
    if not API_KEY or not USER_ID:
        print("ERROR: ZOTERO_API_KEY and/or ZOTERO_USER_ID not set in environment.",
              file=sys.stderr)
        print("  Run: source ~/.zshrc  then re-run this script.", file=sys.stderr)
        sys.exit(1)

    client = zotero.Zotero(USER_ID, "user", API_KEY)

    review_key = find_review_collection(client)
    if not review_key:
        print(f"Creating collection '{REVIEW_COLLECTION_NAME}' ...")
        resp = client.create_collections([{"name": REVIEW_COLLECTION_NAME}])
        review_key = next(iter(resp.get("success", {}).values()))
    print(f"Review Queue collection key: {review_key}")

    # Make review_key visible to build_item() so new items land in the
    # collection at creation time (avoids the fragile addto_collection call).
    global _REVIEW_KEY_GLOBAL
    _REVIEW_KEY_GLOBAL = review_key

    df = pd.read_sql(
        text(
            "SELECT cite_key, author, year, title, journal, doi, pmid "
            "FROM audit.citation "
            "WHERE zotero_verified = 0 "
            "  AND (doi <> '' OR pmid <> '') "
            "ORDER BY last_checked DESC NULLS LAST"
        ),
        ENGINE,
    )
    # Skip placeholder DOIs
    df = df[df["doi"].str.startswith("10.") | (df["pmid"].str.len() > 0)].copy()
    print(f"Found {len(df)} unverified citations with real DOI/PMID to push")

    if df.empty:
        return

    items = [build_item(r) for _, r in df.iterrows()]

    # Push in chunks of 50 (Zotero API limit)
    zotero_keys_by_index = {}
    CHUNK = 50
    for i in range(0, len(items), CHUNK):
        chunk = items[i:i + CHUNK]
        resp = client.create_items(chunk)
        time.sleep(0.5)  # Light rate limiting
        # Map chunk index → Zotero key from the success map
        success_map = resp.get("success", {})
        for local_idx, zkey in success_map.items():
            zotero_keys_by_index[i + int(local_idx)] = zkey
        failed = resp.get("failed", {}) or {}
        if failed:
            print(f"  WARN: {len(failed)} items failed in chunk {i}: {failed}",
                  file=sys.stderr)
        print(f"  Chunk {i}-{i+len(chunk)}: {len(success_map)} succeeded, {len(failed)} failed")

    # Items were created with `collections: [review_key]` in the payload,
    # so no separate collection-assignment call is needed. pyzotero's
    # addto_collection expects a single-item payload with a "key" field,
    # which doesn't match the list-based batch pattern.
    keys_to_add = list(zotero_keys_by_index.values())
    print(f"{len(keys_to_add)} items created directly in Review Queue ({review_key})")

    # Update audit.citation with the returned zotero_keys
    with ENGINE.begin() as conn:
        for idx, zkey in zotero_keys_by_index.items():
            cite_key = df.iloc[idx]["cite_key"]
            conn.execute(
                text(
                    "UPDATE audit.citation SET zotero_key = :zkey, "
                    "last_checked = CURRENT_DATE "
                    "WHERE cite_key = :ck"
                ),
                {"zkey": zkey, "ck": cite_key},
            )
    print(f"\nUpdated {len(zotero_keys_by_index)} audit.citation rows with zotero_key.")
    print(f"Next: manually triage items in Zotero '{REVIEW_COLLECTION_NAME}' → "
          f"move confirmed ones to RT8B9N2J, then set zotero_verified=1.")


if __name__ == "__main__":
    main()

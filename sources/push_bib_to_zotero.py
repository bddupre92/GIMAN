#!/usr/bin/env python3
"""Push missing .bib entries to Zotero's 'Dissertation — Review Queue' collection.

Reads the consolidated sources/zotero_import_missing.bib, creates items
via pyzotero, and places them in a review queue collection for user
verification before tagging into RT8B9N2J.

Run:
    source ~/.zshrc && .venv/bin/python sources/push_bib_to_zotero.py
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import bibtexparser
from pyzotero import zotero

API_KEY = os.environ.get("ZOTERO_API_KEY")
USER_ID = os.environ.get("ZOTERO_USER_ID")
BIB_FILE = Path("/Users/blair.dupre/Projects/CSCI-FALL-2025/sources/zotero_import_missing.bib")
REVIEW_COLLECTION_NAME = "Dissertation — Review Queue"

# BibTeX entry type → Zotero item type
TYPE_MAP = {
    "article": "journalArticle",
    "book": "book",
    "inbook": "bookSection",
    "incollection": "bookSection",
    "inproceedings": "conferencePaper",
    "proceedings": "conferencePaper",
    "conference": "conferencePaper",
    "phdthesis": "thesis",
    "mastersthesis": "thesis",
    "techreport": "report",
    "manual": "report",
    "misc": "document",
    "online": "webpage",
    "unpublished": "manuscript",
}


def clean(s: str) -> str:
    """Strip LaTeX-ish wrapping braces and collapse whitespace."""
    if not s:
        return ""
    s = re.sub(r"\{+", "", s)
    s = re.sub(r"\}+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_authors(author_field: str) -> list[dict]:
    """Convert BibTeX author string to Zotero creator list."""
    if not author_field:
        return []
    creators = []
    # BibTeX uses " and " as separator
    for name in re.split(r"\s+and\s+", author_field):
        name = clean(name)
        if not name:
            continue
        if "," in name:
            last, first = [p.strip() for p in name.split(",", 1)]
        else:
            parts = name.rsplit(" ", 1)
            first = parts[0] if len(parts) > 1 else ""
            last = parts[-1]
        creators.append({
            "creatorType": "author",
            "firstName": first,
            "lastName": last,
        })
    return creators


def bib_to_zotero(entry: dict, item_templates: dict) -> dict | None:
    """Convert a parsed BibTeX entry to a Zotero item dict."""
    btype = entry.get("ENTRYTYPE", "misc").lower()
    ztype = TYPE_MAP.get(btype, "document")

    template = item_templates.get(ztype)
    if not template:
        print(f"  WARN: no template for type '{ztype}' (bib type '{btype}')")
        return None

    # Shallow copy
    item = {k: v for k, v in template.items() if k != "creators"}
    item["itemType"] = ztype
    item["creators"] = parse_authors(entry.get("author", ""))

    # Map common fields
    item["title"] = clean(entry.get("title", ""))
    item["date"] = clean(entry.get("year", "") or entry.get("date", ""))
    item["DOI"] = clean(entry.get("doi", ""))
    item["url"] = clean(entry.get("url", ""))
    item["abstractNote"] = clean(entry.get("abstract", "") or entry.get("abstractnote", ""))

    if ztype == "journalArticle":
        item["publicationTitle"] = clean(entry.get("journal", "") or entry.get("journaltitle", ""))
        item["volume"] = clean(entry.get("volume", ""))
        item["issue"] = clean(entry.get("number", "") or entry.get("issue", ""))
        item["pages"] = clean(entry.get("pages", ""))
        item["ISSN"] = clean(entry.get("issn", ""))
    elif ztype == "conferencePaper":
        item["proceedingsTitle"] = clean(entry.get("booktitle", ""))
        item["conferenceName"] = clean(entry.get("conference", ""))
        item["pages"] = clean(entry.get("pages", ""))
    elif ztype == "book":
        item["publisher"] = clean(entry.get("publisher", ""))
        item["place"] = clean(entry.get("address", ""))
        item["ISBN"] = clean(entry.get("isbn", ""))
    elif ztype == "bookSection":
        item["bookTitle"] = clean(entry.get("booktitle", ""))
        item["publisher"] = clean(entry.get("publisher", ""))
        item["pages"] = clean(entry.get("pages", ""))
    elif ztype == "thesis":
        item["university"] = clean(entry.get("school", "") or entry.get("institution", ""))
        item["thesisType"] = "Ph.D. thesis" if btype == "phdthesis" else "Master's thesis"
    elif ztype == "report":
        item["institution"] = clean(entry.get("institution", ""))
        item["reportNumber"] = clean(entry.get("number", ""))

    # Use citekey as tag for traceability
    item["tags"] = [{"tag": f"bibkey:{entry['ID']}"}]

    return item


def main():
    if not API_KEY or not USER_ID:
        print("ERROR: ZOTERO_API_KEY and ZOTERO_USER_ID must be set")
        sys.exit(1)
    if not BIB_FILE.exists():
        print(f"ERROR: {BIB_FILE} not found. Run the gap analysis first.")
        sys.exit(1)

    print(f"Connecting to Zotero (user={USER_ID[:2]}***)...")
    zot = zotero.Zotero(USER_ID, "user", API_KEY)

    # Find or create review queue collection
    print(f"\nLooking for '{REVIEW_COLLECTION_NAME}' collection...")
    existing = zot.collections()
    review_coll = None
    for c in existing:
        if c["data"]["name"] == REVIEW_COLLECTION_NAME:
            review_coll = c
            print(f"  Found: key={c['key']}, items={c['meta']['numItems']}")
            break

    if not review_coll:
        print(f"  Creating new collection...")
        result = zot.create_collections([{"name": REVIEW_COLLECTION_NAME}])
        review_coll = result["successful"]["0"]
        print(f"  Created: key={review_coll['key']}")

    review_key = review_coll["key"]

    # Parse bib
    print(f"\nParsing {BIB_FILE.name}...")
    with BIB_FILE.open(encoding="utf-8") as f:
        bib_db = bibtexparser.load(f)
    print(f"  Parsed {len(bib_db.entries)} entries")

    # Get item templates (cached)
    print("\nFetching Zotero item templates...")
    templates = {}
    for ztype in set(TYPE_MAP.values()):
        try:
            templates[ztype] = zot.item_template(ztype)
        except Exception as e:
            print(f"  WARN: failed template for {ztype}: {e}")

    # Convert entries
    items_to_create = []
    skipped = 0
    for entry in bib_db.entries:
        item = bib_to_zotero(entry, templates)
        if item is None:
            skipped += 1
            continue
        items_to_create.append(item)

    print(f"\nReady to create {len(items_to_create)} items (skipped {skipped})")
    print(f"Target collection: {REVIEW_COLLECTION_NAME} ({review_key})")

    # Create in batches of 50 (Zotero limit)
    created_keys = []
    failed = []
    BATCH = 50
    for i in range(0, len(items_to_create), BATCH):
        batch = items_to_create[i:i+BATCH]
        print(f"\n  Batch {i//BATCH + 1}/{(len(items_to_create)-1)//BATCH + 1} ({len(batch)} items)...", end=" ", flush=True)
        try:
            result = zot.create_items(batch)
            successful = result.get("successful", {})
            unchanged = result.get("unchanged", {})
            failures = result.get("failed", {})

            for idx, item_data in successful.items():
                created_keys.append(item_data["key"])
            for idx, err in failures.items():
                failed.append((batch[int(idx)].get("title", "?")[:60], err))

            print(f"created={len(successful)}, unchanged={len(unchanged)}, failed={len(failures)}")
            time.sleep(1)  # be nice to the API
        except Exception as e:
            print(f"ERROR: {e}")
            failed.append(("BATCH", str(e)))

    # Add all created items to the review queue collection
    if created_keys:
        print(f"\nAdding {len(created_keys)} items to '{REVIEW_COLLECTION_NAME}'...")
        for i in range(0, len(created_keys), 50):
            chunk = created_keys[i:i+50]
            try:
                # Fetch, update, save — pyzotero way
                for key in chunk:
                    item = zot.item(key)
                    if review_key not in item["data"].get("collections", []):
                        item["data"]["collections"] = list(set(item["data"].get("collections", []) + [review_key]))
                        zot.update_item(item)
                print(f"  Chunk {i//50 + 1}/{(len(created_keys)-1)//50 + 1}: tagged {len(chunk)} items")
                time.sleep(1)
            except Exception as e:
                print(f"  ERROR tagging chunk: {e}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Created:  {len(created_keys)}")
    print(f"  Failed:   {len(failed)}")
    print(f"  Skipped:  {skipped}")
    print(f"\n  Review Queue: {REVIEW_COLLECTION_NAME}")
    print(f"  Open in Zotero: zotero://select/library/collections/{review_key}")

    if failed:
        print("\nFailures:")
        for title, err in failed[:10]:
            print(f"  - {title}: {err}")


if __name__ == "__main__":
    main()

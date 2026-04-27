-- Phase B End-to-End Dissertation Audit — SQLite Claim-Lineage Schema
-- Created 2026-04-13 per Docs/superpowers/plans/2026-04-13-phase-b-detailed-audit-plan.md
--
-- NOTE: This is the AUDIT METADATA database (12 tables) that tracks the audit
-- process itself. It is SEPARATE from the source-of-truth giman_research
-- PostgreSQL DB (112 tables across 10 schemas) which holds the actual
-- scientific data. PostgreSQL tables are referenced from this audit DB via
-- data_source.source_type = 'sql_table' + sql_schema + sql_table_name.

-- Chapters (1-15 + Appendix D)
CREATE TABLE chapter (
  chapter_id INTEGER PRIMARY KEY,
  chapter_number INTEGER NOT NULL,
  title TEXT NOT NULL,
  tex_path TEXT NOT NULL,
  paper_label TEXT,
  audited_at TEXT,
  defensibility_score TEXT CHECK (defensibility_score IN ('green','yellow','red','pending'))
);

-- Canonical claim inventory: one row per distinct load-bearing claim
CREATE TABLE claim (
  claim_id INTEGER PRIMARY KEY AUTOINCREMENT,
  chapter_id INTEGER NOT NULL REFERENCES chapter(chapter_id),
  claim_type TEXT CHECK (claim_type IN ('numerical','literature','method','conclusion')),
  claim_text TEXT NOT NULL,
  source_line INTEGER,
  verdict TEXT CHECK (verdict IN ('verified','partial','unverified','contradicted','pending')),
  verdict_notes TEXT
);

-- Citation keys (from \bibitem) — joined to Zotero + ezproxy verification states
CREATE TABLE citation (
  cite_key TEXT PRIMARY KEY,
  author TEXT,
  year INTEGER,
  title TEXT,
  journal TEXT,
  doi TEXT,
  pmid TEXT,
  zotero_key TEXT,
  zotero_verified INTEGER CHECK (zotero_verified IN (0,1)),
  ezproxy_verified INTEGER CHECK (ezproxy_verified IN (0,1)),
  last_checked TEXT
);

-- Citation use within claim (many-to-many)
CREATE TABLE citation_use (
  claim_id INTEGER NOT NULL REFERENCES claim(claim_id),
  cite_key TEXT NOT NULL REFERENCES citation(cite_key),
  PRIMARY KEY (claim_id, cite_key)
);

-- Numerical claim inventory (subset of claim with extracted values)
CREATE TABLE numerical_claim (
  claim_id INTEGER PRIMARY KEY REFERENCES claim(claim_id),
  value REAL,
  unit TEXT,
  ci_low REAL,
  ci_high REAL,
  producing_artifact TEXT,
  artifact_key TEXT
);

-- Script artifacts that produce claims
CREATE TABLE code_artifact (
  script_path TEXT PRIMARY KEY,
  language TEXT,
  last_modified TEXT,
  python_reviewer_verdict TEXT,
  silent_failure_verdict TEXT,
  review_notes TEXT
);

-- Script produces claim (many-to-many)
CREATE TABLE code_artifact_link (
  claim_id INTEGER NOT NULL REFERENCES claim(claim_id),
  script_path TEXT NOT NULL REFERENCES code_artifact(script_path),
  PRIMARY KEY (claim_id, script_path)
);

-- Test assertions that cover a claim
CREATE TABLE test_link (
  claim_id INTEGER NOT NULL REFERENCES claim(claim_id),
  test_path TEXT NOT NULL,
  test_function TEXT,
  PRIMARY KEY (claim_id, test_path, test_function)
);

-- Data source: CSV / parquet / JSON / HDF5 / PostgreSQL table (giman_research)
CREATE TABLE data_source (
  source_path TEXT PRIMARY KEY,
  source_type TEXT CHECK (source_type IN ('csv','parquet','json','hdf5','sql_table','pth','other')),
  schema_or_columns TEXT,
  row_count INTEGER,
  size_bytes INTEGER,
  sql_schema TEXT,
  sql_table_name TEXT
);

CREATE TABLE data_source_link (
  claim_id INTEGER NOT NULL REFERENCES claim(claim_id),
  source_path TEXT NOT NULL REFERENCES data_source(source_path),
  PRIMARY KEY (claim_id, source_path)
);

-- Mempalace memory anchors (diary entries, KG facts, closets, drawers)
CREATE TABLE mempalace_link (
  claim_id INTEGER NOT NULL REFERENCES claim(claim_id),
  mempalace_kind TEXT CHECK (mempalace_kind IN ('diary','kg_fact','closet','drawer')),
  mempalace_ref TEXT NOT NULL,
  PRIMARY KEY (claim_id, mempalace_ref)
);

-- Reviewer flags (queue for defense prep + ezproxy queue)
CREATE TABLE reviewer_flag (
  flag_id INTEGER PRIMARY KEY AUTOINCREMENT,
  chapter_id INTEGER NOT NULL REFERENCES chapter(chapter_id),
  claim_id INTEGER REFERENCES claim(claim_id),
  flag_type TEXT CHECK (flag_type IN ('zotero_unverified','ezproxy_pending','missing_test','missing_script','missing_data','mempalace_gap','numeric_mismatch','contradicted')),
  severity TEXT CHECK (severity IN ('critical','major','minor','info')),
  description TEXT,
  resolution TEXT,
  resolved INTEGER CHECK (resolved IN (0,1))
);

-- ========= Views =========

CREATE VIEW v_chapter_scorecard AS
SELECT c.chapter_number, c.title, c.defensibility_score,
       COUNT(DISTINCT cl.claim_id) AS total_claims,
       SUM(CASE WHEN cl.verdict = 'verified' THEN 1 ELSE 0 END) AS verified,
       SUM(CASE WHEN cl.verdict = 'partial' THEN 1 ELSE 0 END) AS partial,
       SUM(CASE WHEN cl.verdict IN ('unverified','contradicted') THEN 1 ELSE 0 END) AS unverified
FROM chapter c LEFT JOIN claim cl ON cl.chapter_id = c.chapter_id
GROUP BY c.chapter_id;

CREATE VIEW v_critical_flags AS
SELECT c.chapter_number, rf.*
FROM reviewer_flag rf JOIN chapter c ON c.chapter_id = rf.chapter_id
WHERE rf.severity = 'critical' AND rf.resolved = 0;

CREATE VIEW v_ezproxy_queue AS
SELECT ci.cite_key, ci.author, ci.year, ci.title, ci.doi,
       COUNT(DISTINCT cu.claim_id) AS claims_using
FROM citation ci
LEFT JOIN citation_use cu ON cu.cite_key = ci.cite_key
WHERE ci.ezproxy_verified IS NULL OR ci.ezproxy_verified = 0
GROUP BY ci.cite_key
ORDER BY claims_using DESC;

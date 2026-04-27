# Appendix E.1a — PostgreSQL Data Dictionary

Auto-generated from the live `giman_research` PostgreSQL 17 database (connection: `postgresql+psycopg2://blair.dupre@localhost:5432/giman_research`). Regenerate via `scripts/appendix_e/generate_data_dictionary.py`.

## Schema overview

| Schema | Tables | Description |
|---|---|---|
| `ppmi_raw` | 25 | PPMI clinical/imaging raw tables (AMP-PD v4 BigQuery + LONI IDA) |
| `biofind_raw` | 23 | BioFIND external validation cohort (Russo 2025 replication source) |
| `pdbp_raw` | 52 | PDBP external prediction cohort (includes April 2026 LONI expansion) |
| `hbs_raw` | 11 | HBS external prediction cohort (low-data deployment test) |
| `staging` | 3 | NSD-ISS staging results per cohort |
| `features` | 4 | Assembled ML feature matrices for Papers 1-3 + cross-cohort |
| `longitudinal` | 4 | Paper 3 longitudinal NSD-ISS staging + transition events |
| `paper3` | 1 | Paper 3 per-visit longitudinal feature vectors |
| `ledd` | 2 | Levodopa-equivalent daily dose + concomitant PD medication |
| `mechanistic` | 21 | Phase 1-4 mechanistic twin outputs (posteriors, LOO, counterfactuals) |

---

## Schema `ppmi_raw` (25 tables)

_PPMI clinical/imaging raw tables (AMP-PD v4 BigQuery + LONI IDA)_

### `ppmi_raw.conclusion_of_study_participation` — 943 rows, 22 columns

| Column | Type | Nullable |
|---|---|---|
| `rec_id` | text | YES |
| `patno` | bigint | YES |
| `event_id` | text | YES |
| `sub_event_id` | text | YES |
| `pag_name` | text | YES |
| `complt` | double precision | YES |
| `wddt` | text | YES |
| `wdae` | double precision | YES |
| `wdcmplt` | double precision | YES |
| `wddeath` | double precision | YES |
| `wdfamily` | double precision | YES |
| `wdltfu` | double precision | YES |
| `wdnoncomp` | double precision | YES |
| `wdtransport` | double precision | YES |
| `wdburden` | double precision | YES |
| `wdhealth` | double precision | YES |
| `wdhealthpd` | double precision | YES |
| `wdsite` | double precision | YES |
| `wddisint` | double precision | YES |
| `wdother` | double precision | YES |
| `orig_entry` | text | YES |
| `last_update` | text | YES |

### `ppmi_raw.concomitant_medication_log` — 59,909 rows, 22 columns

| Column | Type | Nullable |
|---|---|---|
| `rec_id` | text | YES |
| `patno` | bigint | YES |
| `event_id` | text | YES |
| `pag_name` | text | YES |
| `cmtrt` | text | YES |
| `cmdose` | double precision | YES |
| `cmdosu` | text | YES |
| `cmdosfrq` | text | YES |
| `route` | double precision | YES |
| `startdt` | text | YES |
| `stopdt` | text | YES |
| `ongoing` | double precision | YES |
| `cmindc` | double precision | YES |
| `cmindc_text` | text | YES |
| `totddose` | double precision | YES |
| `recno` | double precision | YES |
| `seqno1` | double precision | YES |
| `seqno2` | double precision | YES |
| `whodrug` | text | YES |
| `exclmed` | text | YES |
| `orig_entry` | text | YES |
| `last_update` | text | YES |

### `ppmi_raw.datscan_imaging` — 13,860 rows, 17 columns

| Column | Type | Nullable |
|---|---|---|
| `rec_id` | text | YES |
| `patno` | bigint | YES |
| `event_id` | text | YES |
| `sub_event_id` | text | YES |
| `pag_name` | text | YES |
| `infodt` | text | YES |
| `off_schedule` | double precision | YES |
| `datscan` | bigint | YES |
| `datscantrc` | double precision | YES |
| `prevdatdt` | text | YES |
| `scnloc` | double precision | YES |
| `scninjct` | double precision | YES |
| `vsintrpt` | text | YES |
| `vsrptelg` | double precision | YES |
| `diffloc` | double precision | YES |
| `orig_entry` | text | YES |
| `last_update` | text | YES |

### `ppmi_raw.datscan_sbr_analysis` — 4,184 rows, 14 columns

| Column | Type | Nullable |
|---|---|---|
| `protocol` | text | YES |
| `patno` | bigint | YES |
| `event_id` | text | YES |
| `datscan_ligand` | text | YES |
| `datscan_date` | text | YES |
| `datscan_caudate_r` | double precision | YES |
| `datscan_caudate_l` | double precision | YES |
| `datscan_putamen_r` | double precision | YES |
| `datscan_putamen_l` | double precision | YES |
| `datscan_putamen_r_ant` | double precision | YES |
| `datscan_putamen_l_ant` | double precision | YES |
| `datscan_analyzed` | text | YES |
| `datscan_not_analyzed_reason` | text | YES |
| `datscan_other_specify` | text | YES |

### `ppmi_raw.demographics` — 8,038 rows, 29 columns

| Column | Type | Nullable |
|---|---|---|
| `rec_id` | text | YES |
| `patno` | bigint | YES |
| `event_id` | text | YES |
| `pag_name` | text | YES |
| `infodt` | text | YES |
| `aficberb` | double precision | YES |
| `ashkjew` | double precision | YES |
| `basque` | double precision | YES |
| `birthdt` | text | YES |
| `sex` | double precision | YES |
| `chldbear` | double precision | YES |
| `howlive` | double precision | YES |
| `gayles` | double precision | YES |
| `hetero` | double precision | YES |
| `bisexual` | double precision | YES |
| `pansexual` | double precision | YES |
| `asexual` | double precision | YES |
| `othsexuality` | double precision | YES |
| `handed` | double precision | YES |
| `hisplat` | double precision | YES |
| `raasian` | double precision | YES |
| `rablack` | double precision | YES |
| `rahawopi` | double precision | YES |
| `raindals` | double precision | YES |
| `ranos` | double precision | YES |
| `rawhite` | double precision | YES |
| `raunknown` | bigint | YES |
| `orig_entry` | text | YES |
| `last_update` | text | YES |

### `ppmi_raw.deprecated_biospecimen_analysis_results` — 5,433 rows, 13 columns

| Column | Type | Nullable |
|---|---|---|
| `patno` | bigint | YES |
| `sex` | text | YES |
| `cohort` | text | YES |
| `clinical_event` | text | YES |
| `type` | text | YES |
| `testname` | text | YES |
| `testvalue` | text | YES |
| `units` | text | YES |
| `rundate` | text | YES |
| `projectid` | bigint | YES |
| `pi_name` | text | YES |
| `pi_institution` | text | YES |
| `update_stamp` | text | YES |

### `ppmi_raw.epworth_sleepiness_scale` — 19,612 rows, 16 columns

| Column | Type | Nullable |
|---|---|---|
| `rec_id` | text | YES |
| `patno` | bigint | YES |
| `event_id` | text | YES |
| `pag_name` | text | YES |
| `infodt` | text | YES |
| `ptcgboth` | double precision | YES |
| `ess1` | bigint | YES |
| `ess2` | double precision | YES |
| `ess3` | double precision | YES |
| `ess4` | double precision | YES |
| `ess5` | double precision | YES |
| `ess6` | double precision | YES |
| `ess7` | double precision | YES |
| `ess8` | double precision | YES |
| `orig_entry` | text | YES |
| `last_update` | text | YES |

### `ppmi_raw.epworth_sleepiness_scale_online` — 62,950 rows, 16 columns

| Column | Type | Nullable |
|---|---|---|
| `respondent_id` | bigint | YES |
| `patno` | bigint | YES |
| `event_id` | text | YES |
| `mod_instance_id` | bigint | YES |
| `modified_at` | text | YES |
| `created_at` | text | YES |
| `response_status` | bigint | YES |
| `ptcgboth_ol` | bigint | YES |
| `ess1_ol` | double precision | YES |
| `ess2_ol` | double precision | YES |
| `ess3_ol` | double precision | YES |
| `ess4_ol` | double precision | YES |
| `ess5_ol` | double precision | YES |
| `ess6_ol` | double precision | YES |
| `ess7_ol` | double precision | YES |
| `ess8_ol` | double precision | YES |

### `ppmi_raw.fs7_aparc_cth` — 1,716 rows, 72 columns

| Column | Type | Nullable |
|---|---|---|
| `patno` | bigint | YES |
| `event_id` | text | YES |
| `lh_bankssts` | double precision | YES |
| `lh_caudalanteriorcingulate` | double precision | YES |
| `lh_caudalmiddlefrontal` | double precision | YES |
| `lh_cuneus` | double precision | YES |
| `lh_entorhinal` | double precision | YES |
| `lh_fusiform` | double precision | YES |
| `lh_inferiorparietal` | double precision | YES |
| `lh_inferiortemporal` | double precision | YES |
| `lh_isthmuscingulate` | double precision | YES |
| `lh_lateraloccipital` | double precision | YES |
| `lh_lateralorbitofrontal` | double precision | YES |
| `lh_lingual` | double precision | YES |
| `lh_medialorbitofrontal` | double precision | YES |
| `lh_middletemporal` | double precision | YES |
| `lh_parahippocampal` | double precision | YES |
| `lh_paracentral` | double precision | YES |
| `lh_parsopercularis` | double precision | YES |
| `lh_parsorbitalis` | double precision | YES |
| `lh_parstriangularis` | double precision | YES |
| `lh_pericalcarine` | double precision | YES |
| `lh_postcentral` | double precision | YES |
| `lh_posteriorcingulate` | double precision | YES |
| `lh_precentral` | double precision | YES |
| `lh_precuneus` | double precision | YES |
| `lh_rostralanteriorcingulate` | double precision | YES |
| `lh_rostralmiddlefrontal` | double precision | YES |
| `lh_superiorfrontal` | double precision | YES |
| `lh_superiorparietal` | double precision | YES |
| `lh_superiortemporal` | double precision | YES |
| `lh_supramarginal` | double precision | YES |
| `lh_frontalpole` | double precision | YES |
| `lh_temporalpole` | double precision | YES |
| `lh_transversetemporal` | double precision | YES |
| `lh_insula` | double precision | YES |
| `lh_meanthickness` | double precision | YES |
| `rh_bankssts` | double precision | YES |
| `rh_caudalanteriorcingulate` | double precision | YES |
| `rh_caudalmiddlefrontal` | double precision | YES |
| `rh_cuneus` | double precision | YES |
| `rh_entorhinal` | double precision | YES |
| `rh_fusiform` | double precision | YES |
| `rh_inferiorparietal` | double precision | YES |
| `rh_inferiortemporal` | double precision | YES |
| `rh_isthmuscingulate` | double precision | YES |
| `rh_lateraloccipital` | double precision | YES |
| `rh_lateralorbitofrontal` | double precision | YES |
| `rh_lingual` | double precision | YES |
| `rh_medialorbitofrontal` | double precision | YES |
| `rh_middletemporal` | double precision | YES |
| `rh_parahippocampal` | double precision | YES |
| `rh_paracentral` | double precision | YES |
| `rh_parsopercularis` | double precision | YES |
| `rh_parsorbitalis` | double precision | YES |
| `rh_parstriangularis` | double precision | YES |
| `rh_pericalcarine` | double precision | YES |
| `rh_postcentral` | double precision | YES |
| `rh_posteriorcingulate` | double precision | YES |
| `rh_precentral` | double precision | YES |
| `rh_precuneus` | double precision | YES |
| `rh_rostralanteriorcingulate` | double precision | YES |
| `rh_rostralmiddlefrontal` | double precision | YES |
| `rh_superiorfrontal` | double precision | YES |
| `rh_superiorparietal` | double precision | YES |
| `rh_superiortemporal` | double precision | YES |
| `rh_supramarginal` | double precision | YES |
| `rh_frontalpole` | double precision | YES |
| `rh_temporalpole` | double precision | YES |
| `rh_transversetemporal` | double precision | YES |
| `rh_insula` | double precision | YES |
| `rh_meanthickness` | double precision | YES |

### `ppmi_raw.fs7_aparc_sa` — 1,716 rows, 72 columns

| Column | Type | Nullable |
|---|---|---|
| `patno` | bigint | YES |
| `event_id` | text | YES |
| `lh_bankssts` | double precision | YES |
| `lh_caudalanteriorcingulate` | double precision | YES |
| `lh_caudalmiddlefrontal` | double precision | YES |
| `lh_cuneus` | double precision | YES |
| `lh_entorhinal` | double precision | YES |
| `lh_fusiform` | double precision | YES |
| `lh_inferiorparietal` | double precision | YES |
| `lh_inferiortemporal` | double precision | YES |
| `lh_isthmuscingulate` | double precision | YES |
| `lh_lateraloccipital` | double precision | YES |
| `lh_lateralorbitofrontal` | double precision | YES |
| `lh_lingual` | double precision | YES |
| `lh_medialorbitofrontal` | double precision | YES |
| `lh_middletemporal` | double precision | YES |
| `lh_parahippocampal` | double precision | YES |
| `lh_paracentral` | double precision | YES |
| `lh_parsopercularis` | double precision | YES |
| `lh_parsorbitalis` | double precision | YES |
| `lh_parstriangularis` | double precision | YES |
| `lh_pericalcarine` | double precision | YES |
| `lh_postcentral` | double precision | YES |
| `lh_posteriorcingulate` | double precision | YES |
| `lh_precentral` | double precision | YES |
| `lh_precuneus` | double precision | YES |
| `lh_rostralanteriorcingulate` | double precision | YES |
| `lh_rostralmiddlefrontal` | double precision | YES |
| `lh_superiorfrontal` | double precision | YES |
| `lh_superiorparietal` | double precision | YES |
| `lh_superiortemporal` | double precision | YES |
| `lh_supramarginal` | double precision | YES |
| `lh_frontalpole` | double precision | YES |
| `lh_temporalpole` | double precision | YES |
| `lh_transversetemporal` | double precision | YES |
| `lh_insula` | double precision | YES |
| `lh_whitesurfarea` | double precision | YES |
| `rh_bankssts` | double precision | YES |
| `rh_caudalanteriorcingulate` | double precision | YES |
| `rh_caudalmiddlefrontal` | double precision | YES |
| `rh_cuneus` | double precision | YES |
| `rh_entorhinal` | double precision | YES |
| `rh_fusiform` | double precision | YES |
| `rh_inferiorparietal` | double precision | YES |
| `rh_inferiortemporal` | double precision | YES |
| `rh_isthmuscingulate` | double precision | YES |
| `rh_lateraloccipital` | double precision | YES |
| `rh_lateralorbitofrontal` | double precision | YES |
| `rh_lingual` | double precision | YES |
| `rh_medialorbitofrontal` | double precision | YES |
| `rh_middletemporal` | double precision | YES |
| `rh_parahippocampal` | double precision | YES |
| `rh_paracentral` | double precision | YES |
| `rh_parsopercularis` | double precision | YES |
| `rh_parsorbitalis` | double precision | YES |
| `rh_parstriangularis` | double precision | YES |
| `rh_pericalcarine` | double precision | YES |
| `rh_postcentral` | double precision | YES |
| `rh_posteriorcingulate` | double precision | YES |
| `rh_precentral` | double precision | YES |
| `rh_precuneus` | double precision | YES |
| `rh_rostralanteriorcingulate` | double precision | YES |
| `rh_rostralmiddlefrontal` | double precision | YES |
| `rh_superiorfrontal` | double precision | YES |
| `rh_superiorparietal` | double precision | YES |
| `rh_superiortemporal` | double precision | YES |
| `rh_supramarginal` | double precision | YES |
| `rh_frontalpole` | double precision | YES |
| `rh_temporalpole` | double precision | YES |
| `rh_transversetemporal` | double precision | YES |
| `rh_insula` | double precision | YES |
| `rh_whitesurfarea` | double precision | YES |

### `ppmi_raw.fs7_aseg_vol` — 1,713 rows, 66 columns

| Column | Type | Nullable |
|---|---|---|
| `patno` | bigint | YES |
| `event_id` | text | YES |
| `left_wm_hypointensities` | double precision | YES |
| `brain_stem` | double precision | YES |
| `left_non_wm_hypointensities` | double precision | YES |
| `optic_chiasm` | double precision | YES |
| `right_wm_hypointensities` | double precision | YES |
| `brainsegvol` | double precision | YES |
| `right_lateral_ventricle` | double precision | YES |
| `cc_central` | double precision | YES |
| `col_5th_ventricle` | double precision | YES |
| `right_choroid_plexus` | double precision | YES |
| `right_cerebellum_white_matter` | double precision | YES |
| `left_vessel` | double precision | YES |
| `left_cerebellum_cortex` | double precision | YES |
| `maskvol_to_etiv` | double precision | YES |
| `maskvol` | double precision | YES |
| `totalgrayvol` | double precision | YES |
| `left_choroid_plexus` | double precision | YES |
| `right_inf_lat_vent` | double precision | YES |
| `left_pallidum` | double precision | YES |
| `left_thalamus` | double precision | YES |
| `right_ventraldc` | double precision | YES |
| `rhcortexvol` | double precision | YES |
| `right_non_wm_hypointensities` | double precision | YES |
| `brainsegvol_to_etiv` | double precision | YES |
| `right_amygdala` | double precision | YES |
| `left_amygdala` | double precision | YES |
| `estimatedtotalintracranialvol` | double precision | YES |
| `col_4th_ventricle` | double precision | YES |
| `left_inf_lat_vent` | double precision | YES |
| `cortexvol` | double precision | YES |
| `right_pallidum` | double precision | YES |
| `lhcortexvol` | double precision | YES |
| `cc_anterior` | double precision | YES |
| `cc_posterior` | double precision | YES |
| `left_accumbens_area` | double precision | YES |
| `right_vessel` | double precision | YES |
| `right_cerebellum_cortex` | double precision | YES |
| `left_putamen` | double precision | YES |
| `col_3rd_ventricle` | double precision | YES |
| `non_wm_hypointensities` | double precision | YES |
| `right_caudate` | double precision | YES |
| `cc_mid_posterior` | double precision | YES |
| `lhsurfaceholes` | double precision | YES |
| `left_hippocampus` | double precision | YES |
| `right_hippocampus` | double precision | YES |
| `left_caudate` | double precision | YES |
| `rhcerebralwhitemattervol` | double precision | YES |
| `right_putamen` | double precision | YES |
| `wm_hypointensities` | double precision | YES |
| `right_accumbens_area` | double precision | YES |
| `rhsurfaceholes` | double precision | YES |
| `lhcerebralwhitemattervol` | double precision | YES |
| `brainsegvolnotvent` | double precision | YES |
| `left_cerebellum_white_matter` | double precision | YES |
| `left_ventraldc` | double precision | YES |
| `supratentorialvolnotvent` | double precision | YES |
| `cc_mid_anterior` | double precision | YES |
| `supratentorialvol` | double precision | YES |
| `subcortgrayvol` | double precision | YES |
| `right_thalamus` | double precision | YES |
| `left_lateral_ventricle` | double precision | YES |
| `csf` | double precision | YES |
| `surfaceholes` | double precision | YES |
| `cerebralwhitemattervol` | double precision | YES |

### `ppmi_raw.inclusion_exclusion` — 9,163 rows, 45 columns

| Column | Type | Nullable |
|---|---|---|
| `rec_id` | text | YES |
| `patno` | bigint | YES |
| `event_id` | text | YES |
| `pag_name` | text | YES |
| `infodt` | text | YES |
| `exfampd` | double precision | YES |
| `exneurcurr` | double precision | YES |
| `inage30` | double precision | YES |
| `inlrrk2` | double precision | YES |
| `inlrrk2gba` | double precision | YES |
| `inlrrk2gbacore` | double precision | YES |
| `inhy1or2` | double precision | YES |
| `ex60dypdrx` | double precision | YES |
| `ex90dypdrx` | double precision | YES |
| `exabcond` | double precision | YES |
| `exantcoag` | double precision | YES |
| `exatyppd` | double precision | YES |
| `excurpdrx` | double precision | YES |
| `exdarx6mo` | double precision | YES |
| `exmeddbs` | double precision | YES |
| `exdemntdx` | double precision | YES |
| `exneurmri` | double precision | YES |
| `exunsaflp` | double precision | YES |
| `exothrsn` | double precision | YES |
| `in2cardpd` | double precision | YES |
| `in2yrpd` | double precision | YES |
| `in7yrpd` | double precision | YES |
| `incnst` | double precision | YES |
| `inholdrx` | double precision | YES |
| `innomed6mo` | double precision | YES |
| `inpregnt` | double precision | YES |
| `expddemdx` | double precision | YES |
| `inage4030` | double precision | YES |
| `inage6030` | double precision | YES |
| `indatscn` | double precision | YES |
| `inprescrn` | double precision | YES |
| `insaa` | double precision | YES |
| `inhy1to3` | double precision | YES |
| `inpdsc` | double precision | YES |
| `inpddxsc` | double precision | YES |
| `insncapark` | double precision | YES |
| `insncaparkcore` | double precision | YES |
| `inupsit` | double precision | YES |
| `orig_entry` | text | YES |
| `last_update` | text | YES |

### `ppmi_raw.iusm_catalog` — 131,616 rows, 19 columns

| Column | Type | Nullable |
|---|---|---|
| `specimen_no` | double precision | YES |
| `alias_id` | bigint | YES |
| `cohort` | bigint | YES |
| `clinical_event` | text | YES |
| `type` | text | YES |
| `num_available` | bigint | YES |
| `quantity` | double precision | YES |
| `quantity_units` | text | YES |
| `addtl_stock_avail_on_req` | text | YES |
| `mass_ug` | double precision | YES |
| `concentration` | double precision | YES |
| `ratio_260_280` | double precision | YES |
| `rin` | double precision | YES |
| `rin_robot` | text | YES |
| `qc_class_rna` | double precision | YES |
| `qc_status_dna_rna` | double precision | YES |
| `clotting` | text | YES |
| `turbidity` | text | YES |
| `average_hemoglobin` | double precision | YES |

### `ppmi_raw.montreal_cognitive_assessment_moca` — 18,422 rows, 35 columns

| Column | Type | Nullable |
|---|---|---|
| `rec_id` | text | YES |
| `patno` | bigint | YES |
| `event_id` | text | YES |
| `pag_name` | text | YES |
| `infodt` | text | YES |
| `mcaalttm` | double precision | YES |
| `mcacube` | double precision | YES |
| `mcaclckc` | double precision | YES |
| `mcaclckn` | double precision | YES |
| `mcaclckh` | double precision | YES |
| `mcalion` | double precision | YES |
| `mcarhino` | double precision | YES |
| `mcacamel` | double precision | YES |
| `mcafds` | double precision | YES |
| `mcabds` | double precision | YES |
| `mcavigil` | double precision | YES |
| `mcaser7` | double precision | YES |
| `mcasntnc` | double precision | YES |
| `mcavfnum` | double precision | YES |
| `mcavf` | double precision | YES |
| `mcaabstr` | double precision | YES |
| `mcarec1` | double precision | YES |
| `mcarec2` | double precision | YES |
| `mcarec3` | double precision | YES |
| `mcarec4` | double precision | YES |
| `mcarec5` | double precision | YES |
| `mcadate` | double precision | YES |
| `mcamonth` | double precision | YES |
| `mcayr` | double precision | YES |
| `mcaday` | double precision | YES |
| `mcaplace` | double precision | YES |
| `mcacity` | double precision | YES |
| `mcatot` | double precision | YES |
| `orig_entry` | text | YES |
| `last_update` | text | YES |

### `ppmi_raw.participant_status` — 8,121 rows, 30 columns

| Column | Type | Nullable |
|---|---|---|
| `patno` | bigint | YES |
| `cohort` | bigint | YES |
| `cohort_definition` | text | YES |
| `enroll_date` | text | YES |
| `enroll_status` | text | YES |
| `status_date` | text | YES |
| `screenedam` | double precision | YES |
| `enroll_age` | double precision | YES |
| `inexpage` | text | YES |
| `av133stdy` | double precision | YES |
| `taustdy` | double precision | YES |
| `gaitstdy` | double precision | YES |
| `pistdy` | double precision | YES |
| `sv2astdy` | double precision | YES |
| `nxtaustdy` | double precision | YES |
| `dppdstdy` | double precision | YES |
| `dpprostdy` | double precision | YES |
| `fd4stdy` | double precision | YES |
| `datelig` | double precision | YES |
| `ppmi_online_enroll` | text | YES |
| `enrlpink1` | double precision | YES |
| `enrlprkn` | double precision | YES |
| `enrlsrdc` | double precision | YES |
| `enrlnorm` | double precision | YES |
| `enrlothgv` | double precision | YES |
| `enrlhpsm` | bigint | YES |
| `enrlrbd` | bigint | YES |
| `enrllrrk2` | bigint | YES |
| `enrlsnca` | bigint | YES |
| `enrlgba` | bigint | YES |

### `ppmi_raw.pet_sbr_analysis` — 214 rows, 28 columns

| Column | Type | Nullable |
|---|---|---|
| `protocol` | text | YES |
| `patno` | bigint | YES |
| `event_id` | text | YES |
| `pet_ligand` | text | YES |
| `pet_scan_date` | text | YES |
| `pet_rcaud_s` | double precision | YES |
| `pet_rputant_s` | double precision | YES |
| `pet_rputpost_s` | double precision | YES |
| `pet_lcaud_s` | double precision | YES |
| `pet_lputant_s` | double precision | YES |
| `pet_lputpost_s` | double precision | YES |
| `pet_lprecaud` | double precision | YES |
| `pet_rprecaud` | double precision | YES |
| `pet_lputpredors` | double precision | YES |
| `pet_rputpredors` | double precision | YES |
| `pet_lputprevent` | double precision | YES |
| `pet_rputprevent` | double precision | YES |
| `pet_lputpostdors` | double precision | YES |
| `pet_rputpostdors` | double precision | YES |
| `pet_lputpostvent` | double precision | YES |
| `pet_rputpostvent` | double precision | YES |
| `pet_lcaudpost` | double precision | YES |
| `pet_rcaudpost` | double precision | YES |
| `pet_cbm` | double precision | YES |
| `pet_occip` | double precision | YES |
| `pet_analyzed` | text | YES |
| `pet_not_analyzed_reason` | text | YES |
| `pet_other_specify` | double precision | YES |

### `ppmi_raw.pilot_biospecimen_analysis_results` — 6,095 rows, 13 columns

| Column | Type | Nullable |
|---|---|---|
| `patno` | bigint | YES |
| `sex` | text | YES |
| `cohort` | text | YES |
| `clinical_event` | text | YES |
| `type` | text | YES |
| `testname` | text | YES |
| `testvalue` | text | YES |
| `units` | text | YES |
| `rundate` | text | YES |
| `projectid` | bigint | YES |
| `pi_name` | text | YES |
| `pi_institution` | text | YES |
| `update_stamp` | text | YES |

### `ppmi_raw.primary_clinical_diagnosis` — 27,605 rows, 11 columns

| Column | Type | Nullable |
|---|---|---|
| `rec_id` | text | YES |
| `patno` | bigint | YES |
| `event_id` | text | YES |
| `pag_name` | text | YES |
| `infodt` | text | YES |
| `primdiag` | bigint | YES |
| `newdiagexp` | text | YES |
| `othneuro` | text | YES |
| `dxlvl` | double precision | YES |
| `orig_entry` | text | YES |
| `last_update` | text | YES |

### `ppmi_raw.rem_sleep_behavior_disorder_questionnaire` — 19,622 rows, 29 columns

| Column | Type | Nullable |
|---|---|---|
| `rec_id` | text | YES |
| `patno` | bigint | YES |
| `event_id` | text | YES |
| `pag_name` | text | YES |
| `infodt` | text | YES |
| `ptcgboth` | double precision | YES |
| `drmvivid` | double precision | YES |
| `drmagrac` | bigint | YES |
| `drmnoctb` | double precision | YES |
| `slplmbmv` | double precision | YES |
| `slpinjur` | double precision | YES |
| `drmverbl` | double precision | YES |
| `drmfight` | double precision | YES |
| `drmumv` | double precision | YES |
| `drmobjfl` | double precision | YES |
| `mvawaken` | double precision | YES |
| `drmremem` | double precision | YES |
| `slpdstrb` | double precision | YES |
| `stroke` | double precision | YES |
| `hetra` | double precision | YES |
| `parkism` | double precision | YES |
| `rls` | double precision | YES |
| `narclpsy` | double precision | YES |
| `deprs` | double precision | YES |
| `epilepsy` | double precision | YES |
| `brninfm` | double precision | YES |
| `cnsoth` | double precision | YES |
| `orig_entry` | text | YES |
| `last_update` | text | YES |

### `ppmi_raw.research_biospecimens` — 28,054 rows, 55 columns

| Column | Type | Nullable |
|---|---|---|
| `rec_id` | text | YES |
| `patno` | bigint | YES |
| `event_id` | text | YES |
| `pag_name` | text | YES |
| `infodt` | text | YES |
| `off_schedule` | double precision | YES |
| `lmdt` | text | YES |
| `lmtm` | text | YES |
| `faststat` | double precision | YES |
| `pdmedyn` | double precision | YES |
| `pdmeddt` | text | YES |
| `pdmedtm` | text | YES |
| `uaspec` | double precision | YES |
| `uaspecdt` | text | YES |
| `ut1tm` | text | YES |
| `ut1spntm` | text | YES |
| `ut1spnrt` | double precision | YES |
| `ut1spndr` | double precision | YES |
| `ut1cfrg` | double precision | YES |
| `ut1ftm` | text | YES |
| `bldwhl` | double precision | YES |
| `bldwhltm` | text | YES |
| `bldwhlvl` | double precision | YES |
| `bldwhlstortm` | text | YES |
| `bldwhltmp` | double precision | YES |
| `blddrdt` | text | YES |
| `bldrna` | double precision | YES |
| `bldrnatm` | text | YES |
| `bldrnavl` | double precision | YES |
| `rnafdt` | text | YES |
| `rnaftm` | text | YES |
| `rnasttmp` | double precision | YES |
| `bldplas` | double precision | YES |
| `plastm` | text | YES |
| `plaspntm` | text | YES |
| `plaspnrt` | double precision | YES |
| `plaspndr` | double precision | YES |
| `plascfrg` | double precision | YES |
| `plasvaft` | double precision | YES |
| `plaalqn` | double precision | YES |
| `plasftm` | text | YES |
| `plasttmp` | double precision | YES |
| `bldser` | double precision | YES |
| `bldsertm` | text | YES |
| `bsspntm` | text | YES |
| `bsspnrt` | double precision | YES |
| `bsspndr` | double precision | YES |
| `bscfrg` | double precision | YES |
| `bsvaft` | double precision | YES |
| `bsalqn` | double precision | YES |
| `bsftm` | text | YES |
| `bssttmp` | double precision | YES |
| `plasbfct` | double precision | YES |
| `orig_entry` | text | YES |
| `last_update` | text | YES |

### `ppmi_raw.scopa_aut` — 19,589 rows, 43 columns

| Column | Type | Nullable |
|---|---|---|
| `rec_id` | text | YES |
| `patno` | bigint | YES |
| `event_id` | text | YES |
| `pag_name` | text | YES |
| `infodt` | text | YES |
| `ptcgboth` | double precision | YES |
| `scau1` | double precision | YES |
| `scau2` | double precision | YES |
| `scau3` | double precision | YES |
| `scau4` | double precision | YES |
| `scau5` | double precision | YES |
| `scau6` | double precision | YES |
| `scau7` | double precision | YES |
| `scau8` | double precision | YES |
| `scau9` | double precision | YES |
| `scau10` | double precision | YES |
| `scau11` | double precision | YES |
| `scau12` | double precision | YES |
| `scau13` | double precision | YES |
| `scau14` | double precision | YES |
| `scau15` | double precision | YES |
| `scau16` | double precision | YES |
| `scau17` | double precision | YES |
| `scau18` | double precision | YES |
| `scau19` | double precision | YES |
| `scau20` | double precision | YES |
| `scau21` | double precision | YES |
| `scau22` | double precision | YES |
| `scau23` | double precision | YES |
| `scau23a` | double precision | YES |
| `scau23at` | text | YES |
| `scau24` | double precision | YES |
| `scau25` | double precision | YES |
| `scau26a` | double precision | YES |
| `scau26at` | text | YES |
| `scau26b` | double precision | YES |
| `scau26bt` | text | YES |
| `scau26c` | double precision | YES |
| `scau26ct` | text | YES |
| `scau26d` | double precision | YES |
| `scau26dt` | text | YES |
| `orig_entry` | text | YES |
| `last_update` | text | YES |

### `ppmi_raw.screening_demographics` — 2,254 rows, 29 columns

| Column | Type | Nullable |
|---|---|---|
| `rec_id` | bigint | YES |
| `f_status` | text | YES |
| `patno` | bigint | YES |
| `event_id` | text | YES |
| `pag_name` | text | YES |
| `signcnst` | double precision | YES |
| `consntdt` | text | YES |
| `apprdx` | double precision | YES |
| `current_apprdx` | double precision | YES |
| `p3grp` | double precision | YES |
| `birthdt` | double precision | YES |
| `gender` | double precision | YES |
| `hisplat` | double precision | YES |
| `raindals` | double precision | YES |
| `raasian` | double precision | YES |
| `rablack` | double precision | YES |
| `rahawopi` | double precision | YES |
| `rawhite` | double precision | YES |
| `ranos` | double precision | YES |
| `prjenrdt` | text | YES |
| `referral` | double precision | YES |
| `declined` | double precision | YES |
| `rsndec` | double precision | YES |
| `excluded` | double precision | YES |
| `rsnexc` | double precision | YES |
| `orig_entry` | text | YES |
| `last_update` | text | YES |
| `query` | double precision | YES |
| `site_aprv` | text | YES |

### `ppmi_raw.university_of_pennsylvania_smell_identification_test_upsit` — 9,017 rows, 95 columns

| Column | Type | Nullable |
|---|---|---|
| `rec_id` | text | YES |
| `patno` | bigint | YES |
| `event_id` | text | YES |
| `pag_name` | text | YES |
| `infodt` | text | YES |
| `scent_01_correct` | double precision | YES |
| `scent_01_response` | double precision | YES |
| `scent_02_correct` | double precision | YES |
| `scent_02_response` | double precision | YES |
| `scent_03_correct` | double precision | YES |
| `scent_03_response` | double precision | YES |
| `scent_04_correct` | double precision | YES |
| `scent_04_response` | double precision | YES |
| `scent_05_correct` | double precision | YES |
| `scent_05_response` | double precision | YES |
| `scent_06_correct` | double precision | YES |
| `scent_06_response` | double precision | YES |
| `scent_07_correct` | double precision | YES |
| `scent_07_response` | double precision | YES |
| `scent_08_correct` | double precision | YES |
| `scent_08_response` | double precision | YES |
| `scent_09_correct` | double precision | YES |
| `scent_09_response` | double precision | YES |
| `scent_10_correct` | double precision | YES |
| `scent_10_response` | double precision | YES |
| `scent_11_correct` | double precision | YES |
| `scent_11_response` | double precision | YES |
| `scent_12_correct` | double precision | YES |
| `scent_12_response` | double precision | YES |
| `scent_13_correct` | double precision | YES |
| `scent_13_response` | double precision | YES |
| `scent_14_correct` | double precision | YES |
| `scent_14_response` | double precision | YES |
| `scent_15_correct` | double precision | YES |
| `scent_15_response` | double precision | YES |
| `scent_16_correct` | double precision | YES |
| `scent_16_response` | double precision | YES |
| `scent_17_correct` | double precision | YES |
| `scent_17_response` | double precision | YES |
| `scent_18_correct` | double precision | YES |
| `scent_18_response` | double precision | YES |
| `scent_19_correct` | double precision | YES |
| `scent_19_response` | double precision | YES |
| `scent_20_correct` | double precision | YES |
| `scent_20_response` | double precision | YES |
| `scent_21_correct` | double precision | YES |
| `scent_21_response` | double precision | YES |
| `scent_22_correct` | double precision | YES |
| `scent_22_response` | double precision | YES |
| `scent_23_correct` | double precision | YES |
| `scent_23_response` | double precision | YES |
| `scent_24_correct` | double precision | YES |
| `scent_24_response` | double precision | YES |
| `scent_25_correct` | double precision | YES |
| `scent_25_response` | double precision | YES |
| `scent_26_correct` | double precision | YES |
| `scent_26_response` | double precision | YES |
| `scent_27_correct` | double precision | YES |
| `scent_27_response` | double precision | YES |
| `scent_28_correct` | double precision | YES |
| `scent_28_response` | double precision | YES |
| `scent_29_correct` | double precision | YES |
| `scent_29_response` | double precision | YES |
| `scent_30_correct` | double precision | YES |
| `scent_30_response` | double precision | YES |
| `scent_31_correct` | double precision | YES |
| `scent_31_response` | double precision | YES |
| `scent_32_correct` | double precision | YES |
| `scent_32_response` | double precision | YES |
| `scent_33_correct` | double precision | YES |
| `scent_33_response` | double precision | YES |
| `scent_34_correct` | double precision | YES |
| `scent_34_response` | double precision | YES |
| `scent_35_correct` | double precision | YES |
| `scent_35_response` | double precision | YES |
| `scent_36_correct` | double precision | YES |
| `scent_36_response` | double precision | YES |
| `scent_37_correct` | double precision | YES |
| `scent_37_response` | double precision | YES |
| `scent_38_correct` | double precision | YES |
| `scent_38_response` | double precision | YES |
| `scent_39_correct` | double precision | YES |
| `scent_39_response` | double precision | YES |
| `scent_40_correct` | double precision | YES |
| `scent_40_response` | double precision | YES |
| `total_correct` | double precision | YES |
| `upsitorder` | double precision | YES |
| `upsitform` | bigint | YES |
| `upsit_prcntge` | double precision | YES |
| `upsit_prctver` | text | YES |
| `imputed_data` | boolean | YES |
| `upsit_source` | text | YES |
| `upsitlangcntr` | text | YES |
| `orig_entry` | text | YES |
| `last_update` | text | YES |

### `ppmi_raw.use_of_pd_medication` — 8,613 rows, 21 columns

| Column | Type | Nullable |
|---|---|---|
| `rec_id` | bigint | YES |
| `f_status` | text | YES |
| `patno` | bigint | YES |
| `event_id` | text | YES |
| `pag_name` | text | YES |
| `infodt` | text | YES |
| `pdmedyn` | bigint | YES |
| `onldopa` | double precision | YES |
| `ondopag` | double precision | YES |
| `onother` | double precision | YES |
| `fulnupdr` | text | YES |
| `pdmeddt` | text | YES |
| `pdmedtm` | text | YES |
| `nupdrtm` | text | YES |
| `nupdwprf` | double precision | YES |
| `nounfrsn` | double precision | YES |
| `nuposmas` | double precision | YES |
| `orig_entry` | text | YES |
| `last_update` | text | YES |
| `query` | text | YES |
| `site_aprv` | text | YES |

### `ppmi_raw.xing_core_lab_quant_sbr` — 3,553 rows, 42 columns

| Column | Type | Nullable |
|---|---|---|
| `protocol` | bigint | YES |
| `patno` | bigint | YES |
| `event_id` | text | YES |
| `previously_acquired` | text | YES |
| `datscan_ligand` | text | YES |
| `datscan_date` | text | YES |
| `datscan_analyzed` | text | YES |
| `datscan_not_analyzed_reason` | double precision | YES |
| `datscan_other_specify` | text | YES |
| `striatum_ref_cwm` | double precision | YES |
| `caudate_ref_cwm` | double precision | YES |
| `putamen_ref_cwm` | double precision | YES |
| `precaudate_ref_cwm` | double precision | YES |
| `poscaudate_ref_cwm` | double precision | YES |
| `precommissural_putamen_ref_cwm` | double precision | YES |
| `poscommissural_putamen_ref_cwm` | double precision | YES |
| `predorsalputamen_ref_cwm` | double precision | YES |
| `preventralputamen_ref_cwm` | double precision | YES |
| `posdorsalputamen_ref_cwm` | double precision | YES |
| `posventralputamen_ref_cwm` | double precision | YES |
| `striatum_l_ref_cwm` | double precision | YES |
| `caudate_l_ref_cwm` | double precision | YES |
| `putamen_l_ref_cwm` | double precision | YES |
| `precaudate_l_ref_cwm` | double precision | YES |
| `poscaudate_l_ref_cwm` | double precision | YES |
| `precommissural_putamen_l_ref_cwm` | double precision | YES |
| `poscommissural_putamen_l_ref_cwm` | double precision | YES |
| `predorsalputamen_l_ref_cwm` | double precision | YES |
| `preventralputamen_l_ref_cwm` | double precision | YES |
| `posdorsalputamen_l_ref_cwm` | double precision | YES |
| `posventralputamen_l_ref_cwm` | double precision | YES |
| `striatum_r_ref_cwm` | double precision | YES |
| `caudate_r_ref_cwm` | double precision | YES |
| `putamen_r_ref_cwm` | double precision | YES |
| `precaudate_r_ref_cwm` | double precision | YES |
| `poscaudate_r_ref_cwm` | double precision | YES |
| `precommissural_putamen_r_ref_cwm` | double precision | YES |
| `poscommissural_putamen_r_ref_cwm` | double precision | YES |
| `predorsalputamen_r_ref_cwm` | double precision | YES |
| `preventralputamen_r_ref_cwm` | double precision | YES |
| `posdorsalputamen_r_ref_cwm` | double precision | YES |
| `posventralputamen_r_ref_cwm` | double precision | YES |

## Schema `biofind_raw` (23 tables)

_BioFIND external validation cohort (Russo 2025 replication source)_

### `biofind_raw.amp_pd_case_control` — 213 rows, 5 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `diagnosis_at_baseline` | text | YES |
| `diagnosis_latest` | text | YES |
| `case_control_other_at_baseline` | text | YES |
| `case_control_other_latest` | text | YES |

### `biofind_raw.biofind_project_104_normalized_log_intensities_all_mjf` — 60,650 rows, 61 columns

| Column | Type | Nullable |
|---|---|---|
| `unnamed_0` | bigint | YES |
| `col_1052` | double precision | YES |
| `col_1051` | double precision | YES |
| `col_1205` | double precision | YES |
| `col_1009` | double precision | YES |
| `col_1057` | double precision | YES |
| `col_1207` | double precision | YES |
| `col_1104` | double precision | YES |
| `col_1059` | double precision | YES |
| `col_1101` | double precision | YES |
| `col_1001` | double precision | YES |
| `col_1102` | double precision | YES |
| `col_1002` | double precision | YES |
| `col_1103` | double precision | YES |
| `col_1053` | double precision | YES |
| `col_1201` | double precision | YES |
| `col_1005` | double precision | YES |
| `col_1006` | double precision | YES |
| `col_1055` | double precision | YES |
| `col_1204` | double precision | YES |
| `col_1153` | double precision | YES |
| `col_1152` | double precision | YES |
| `col_1208` | double precision | YES |
| `col_1058` | double precision | YES |
| `col_1012` | double precision | YES |
| `col_1209` | double precision | YES |
| `col_1105` | double precision | YES |
| `col_1106` | double precision | YES |
| `col_1013` | double precision | YES |
| `col_1014` | double precision | YES |
| `col_1210` | double precision | YES |
| `col_1064` | double precision | YES |
| `col_1066` | double precision | YES |
| `col_1016` | double precision | YES |
| `col_1017` | double precision | YES |
| `col_1062` | double precision | YES |
| `col_1212` | double precision | YES |
| `col_1023` | double precision | YES |
| `col_1022` | double precision | YES |
| `col_1024` | double precision | YES |
| `col_1019` | double precision | YES |
| `col_1028` | double precision | YES |
| `col_1071` | double precision | YES |
| `col_1216` | double precision | YES |
| `col_1020` | double precision | YES |
| `col_1041` | double precision | YES |
| `col_1158` | double precision | YES |
| `col_1042` | double precision | YES |
| `col_1068` | double precision | YES |
| `col_1159` | double precision | YES |
| `col_1076` | double precision | YES |
| `col_1109` | double precision | YES |
| `col_1030` | double precision | YES |
| `col_1075` | double precision | YES |
| `col_1218` | double precision | YES |
| `col_1031` | double precision | YES |
| `col_1034` | double precision | YES |
| `col_1033` | double precision | YES |
| `col_1032` | double precision | YES |
| `col_1036` | double precision | YES |
| `col_1070` | double precision | YES |

### `biofind_raw.biofind_saa_consensus` — 194 rows, 9 columns

| Column | Type | Nullable |
|---|---|---|
| `patno` | bigint | YES |
| `saa_bc` | text | YES |
| `saa_ag` | text | YES |
| `saa_cs` | text | YES |
| `score_ag` | double precision | YES |
| `score_bc` | double precision | YES |
| `score_cs` | double precision | YES |
| `saa_score` | double precision | YES |
| `saa_result` | boolean | YES |

### `biofind_raw.biospecimen_analysis_results` — 492,841 rows, 13 columns

| Column | Type | Nullable |
|---|---|---|
| `patno` | bigint | YES |
| `gender` | text | YES |
| `diagnosis` | text | YES |
| `clinical_event` | text | YES |
| `type` | text | YES |
| `testname` | text | YES |
| `testvalue` | text | YES |
| `units` | text | YES |
| `rundate` | text | YES |
| `projectid` | bigint | YES |
| `pi_name` | text | YES |
| `pi_institution` | text | YES |
| `update_stamp` | text | YES |

### `biofind_raw.biospecimen_csf_abeta_tau` — 574 rows, 8 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `sample_type` | text | YES |
| `test_name` | text | YES |
| `test_value` | double precision | YES |
| `test_units` | text | YES |

### `biofind_raw.biospecimen_csf_beta_glucocerebrosidase` — 278 rows, 8 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `sample_type` | text | YES |
| `test_name` | text | YES |
| `test_value` | double precision | YES |
| `test_units` | text | YES |

### `biofind_raw.biospecimen_somalogic_plasma` — 26,100 rows, 8 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `sample_type` | text | YES |
| `test_name` | text | YES |
| `test_value` | double precision | YES |
| `test_units` | text | YES |

### `biofind_raw.data_dictionary` — 997 rows, 11 columns

| Column | Type | Nullable |
|---|---|---|
| `mod_name` | text | YES |
| `itm_name` | text | YES |
| `seq_no` | double precision | YES |
| `dscr` | text | YES |
| `itm_type` | text | YES |
| `fld_len` | double precision | YES |
| `decml` | double precision | YES |
| `min_len` | double precision | YES |
| `max_len` | double precision | YES |
| `codelist` | text | YES |
| `update_stamp` | text | YES |

### `biofind_raw.demographics` — 213 rows, 9 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `age_at_baseline` | bigint | YES |
| `sex` | text | YES |
| `ethnicity` | text | YES |
| `race` | text | YES |
| `education_level_years` | text | YES |

### `biofind_raw.enrollment` — 213 rows, 8 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `enrollment_months_after_baseline` | double precision | YES |
| `informed_consent_months_after_baseline` | double precision | YES |
| `prodromal_category` | text | YES |
| `study_arm` | text | YES |

### `biofind_raw.family_history_pd` — 213 rows, 7 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `biological_mother_with_pd` | text | YES |
| `biological_father_with_pd` | text | YES |
| `other_relative_with_pd` | text | YES |

### `biofind_raw.mds_updrs_part_i` — 119 rows, 35 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `mds_updrs_part_i_primary_info_source` | text | YES |
| `code_upd2101_cognitive_impairment` | bigint | YES |
| `code_upd2102_hallucinations_and_psychosis` | bigint | YES |
| `code_upd2103_depressed_mood` | bigint | YES |
| `code_upd2104_anxious_mood` | bigint | YES |
| `code_upd2105_apathy` | bigint | YES |
| `code_upd2106_dopamine_dysregulation_syndrome_features` | bigint | YES |
| `upd2101_cognitive_impairment` | text | YES |
| `upd2102_hallucinations_and_psychosis` | text | YES |
| `upd2103_depressed_mood` | text | YES |
| `upd2104_anxious_mood` | text | YES |
| `upd2105_apathy` | text | YES |
| `upd2106_dopamine_dysregulation_syndrome_features` | text | YES |
| `mds_updrs_part_i_sub_score` | bigint | YES |
| `mds_updrs_part_i_pat_quest_primary_info_source` | text | YES |
| `code_upd2107_pat_quest_sleep_problems` | bigint | YES |
| `code_upd2108_pat_quest_daytime_sleepiness` | bigint | YES |
| `code_upd2109_pat_quest_pain_and_other_sensations` | bigint | YES |
| `code_upd2110_pat_quest_urinary_problems` | bigint | YES |
| `code_upd2111_pat_quest_constipation_problems` | bigint | YES |
| `code_upd2112_pat_quest_lightheadedness_on_standing` | bigint | YES |
| `code_upd2113_pat_quest_fatigue` | bigint | YES |
| `upd2107_pat_quest_sleep_problems` | text | YES |
| `upd2108_pat_quest_daytime_sleepiness` | text | YES |
| `upd2109_pat_quest_pain_and_other_sensations` | text | YES |
| `upd2110_pat_quest_urinary_problems` | text | YES |
| `upd2111_pat_quest_constipation_problems` | text | YES |
| `upd2112_pat_quest_lightheadedness_on_standing` | text | YES |
| `upd2113_pat_quest_fatigue` | text | YES |
| `mds_updrs_part_i_pat_quest_sub_score` | bigint | YES |
| `mds_updrs_part_i_summary_score` | bigint | YES |

### `biofind_raw.mds_updrs_part_ii` — 119 rows, 32 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `mds_updrs_part_ii_primary_info_source` | text | YES |
| `code_upd2201_speech` | bigint | YES |
| `code_upd2202_saliva_and_drooling` | bigint | YES |
| `code_upd2203_chewing_and_swallowing` | bigint | YES |
| `code_upd2204_eating_tasks` | bigint | YES |
| `code_upd2205_dressing` | bigint | YES |
| `code_upd2206_hygiene` | bigint | YES |
| `code_upd2207_handwriting` | bigint | YES |
| `code_upd2208_doing_hobbies_and_other_activities` | bigint | YES |
| `code_upd2209_turning_in_bed` | bigint | YES |
| `code_upd2210_tremor` | bigint | YES |
| `code_upd2211_get_out_of_bed_car_or_deep_chair` | bigint | YES |
| `code_upd2212_walking_and_balance` | bigint | YES |
| `code_upd2213_freezing` | bigint | YES |
| `upd2201_speech` | text | YES |
| `upd2202_saliva_and_drooling` | text | YES |
| `upd2203_chewing_and_swallowing` | text | YES |
| `upd2204_eating_tasks` | text | YES |
| `upd2205_dressing` | text | YES |
| `upd2206_hygiene` | text | YES |
| `upd2207_handwriting` | text | YES |
| `upd2208_doing_hobbies_and_other_activities` | text | YES |
| `upd2209_turning_in_bed` | text | YES |
| `upd2210_tremor` | text | YES |
| `upd2211_get_out_of_bed_car_or_deep_chair` | text | YES |
| `upd2212_walking_and_balance` | text | YES |
| `upd2213_freezing` | text | YES |
| `mds_updrs_part_ii_summary_score` | bigint | YES |

### `biofind_raw.mds_updrs_part_iii` — 331 rows, 77 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `code_upd2301_speech_problems` | bigint | YES |
| `code_upd2302_facial_expression` | bigint | YES |
| `code_upd2303a_rigidity_neck` | bigint | YES |
| `code_upd2303b_rigidity_rt_upper_extremity` | bigint | YES |
| `code_upd2303c_rigidity_left_upper_extremity` | bigint | YES |
| `code_upd2303d_rigidity_rt_lower_extremity` | bigint | YES |
| `code_upd2303e_rigidity_left_lower_extremity` | bigint | YES |
| `code_upd2304a_right_finger_tapping` | bigint | YES |
| `code_upd2304b_left_finger_tapping` | bigint | YES |
| `code_upd2305a_right_hand_movements` | bigint | YES |
| `code_upd2305b_left_hand_movements` | bigint | YES |
| `code_upd2306a_pron_sup_movement_right_hand` | bigint | YES |
| `code_upd2306b_pron_sup_movement_left_hand` | bigint | YES |
| `code_upd2307a_right_toe_tapping` | bigint | YES |
| `code_upd2307b_left_toe_tapping` | bigint | YES |
| `code_upd2308a_right_leg_agility` | bigint | YES |
| `code_upd2308b_left_leg_agility` | bigint | YES |
| `code_upd2309_arising_from_chair` | bigint | YES |
| `code_upd2310_gait` | bigint | YES |
| `code_upd2311_freezing_of_gait` | bigint | YES |
| `code_upd2312_postural_stability` | bigint | YES |
| `code_upd2313_posture` | bigint | YES |
| `code_upd2314_body_bradykinesia` | bigint | YES |
| `code_upd2315a_postural_tremor_of_right_hand` | bigint | YES |
| `code_upd2315b_postural_tremor_of_left_hand` | bigint | YES |
| `code_upd2316a_kinetic_tremor_of_right_hand` | bigint | YES |
| `code_upd2316b_kinetic_tremor_of_left_hand` | bigint | YES |
| `code_upd2317a_rest_tremor_amplitude_right_upper_extremity` | bigint | YES |
| `code_upd2317b_rest_tremor_amplitude_left_upper_extremity` | bigint | YES |
| `code_upd2317c_rest_tremor_amplitude_right_lower_extremity` | bigint | YES |
| `code_upd2317d_rest_tremor_amplitude_left_lower_extremity` | bigint | YES |
| `code_upd2317e_rest_tremor_amplitude_lip_or_jaw` | bigint | YES |
| `code_upd2318_consistency_of_rest_tremor` | bigint | YES |
| `upd2301_speech_problems` | text | YES |
| `upd2302_facial_expression` | text | YES |
| `upd2303a_rigidity_neck` | text | YES |
| `upd2303b_rigidity_rt_upper_extremity` | text | YES |
| `upd2303c_rigidity_left_upper_extremity` | text | YES |
| `upd2303d_rigidity_rt_lower_extremity` | text | YES |
| `upd2303e_rigidity_left_lower_extremity` | text | YES |
| `upd2304a_right_finger_tapping` | text | YES |
| `upd2304b_left_finger_tapping` | text | YES |
| `upd2305a_right_hand_movements` | text | YES |
| `upd2305b_left_hand_movements` | text | YES |
| `upd2306a_pron_sup_movement_right_hand` | text | YES |
| `upd2306b_pron_sup_movement_left_hand` | text | YES |
| `upd2307a_right_toe_tapping` | text | YES |
| `upd2307b_left_toe_tapping` | text | YES |
| `upd2308a_right_leg_agility` | text | YES |
| `upd2308b_left_leg_agility` | text | YES |
| `upd2309_arising_from_chair` | text | YES |
| `upd2310_gait` | text | YES |
| `upd2311_freezing_of_gait` | text | YES |
| `upd2312_postural_stability` | text | YES |
| `upd2313_posture` | text | YES |
| `upd2314_body_bradykinesia` | text | YES |
| `upd2315a_postural_tremor_of_right_hand` | text | YES |
| `upd2315b_postural_tremor_of_left_hand` | text | YES |
| `upd2316a_kinetic_tremor_of_right_hand` | text | YES |
| `upd2316b_kinetic_tremor_of_left_hand` | text | YES |
| `upd2317a_rest_tremor_amplitude_right_upper_extremity` | text | YES |
| `upd2317b_rest_tremor_amplitude_left_upper_extremity` | text | YES |
| `upd2317c_rest_tremor_amplitude_right_lower_extremity` | text | YES |
| `upd2317d_rest_tremor_amplitude_left_lower_extremity` | text | YES |
| `upd2317e_rest_tremor_amplitude_lip_or_jaw` | text | YES |
| `upd2318_consistency_of_rest_tremor` | text | YES |
| `upd2da_dyskinesias_during_exam` | text | YES |
| `upd2db_movements_interfere_with_ratings` | text | YES |
| `code_upd2hy_hoehn_and_yahr_stage` | double precision | YES |
| `upd2hy_hoehn_and_yahr_stage` | text | YES |
| `upd23a_medication_for_pd` | double precision | YES |
| `upd23b_clinical_state_on_medication` | double precision | YES |
| `mds_updrs_part_iii_summary_score` | bigint | YES |

### `biofind_raw.mds_updrs_part_iv` — 133 rows, 17 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `code_upd2401_time_spent_with_dyskinesias` | bigint | YES |
| `code_upd2402_functional_impact_of_dyskinesias` | bigint | YES |
| `code_upd2403_time_spent_in_the_off_state` | bigint | YES |
| `code_upd2404_functional_impact_of_fluctuations` | bigint | YES |
| `code_upd2405_complexity_of_motor_fluctuations` | bigint | YES |
| `code_upd2406_painful_off_state_dystonia` | bigint | YES |
| `upd2401_time_spent_with_dyskinesias` | text | YES |
| `upd2402_functional_impact_of_dyskinesias` | text | YES |
| `upd2403_time_spent_in_the_off_state` | text | YES |
| `upd2404_functional_impact_of_fluctuations` | text | YES |
| `upd2405_complexity_of_motor_fluctuations` | text | YES |
| `upd2406_painful_off_state_dystonia` | text | YES |
| `mds_updrs_part_iv_summary_score` | bigint | YES |

### `biofind_raw.moca` — 213 rows, 43 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `moca01_alternating_trail_making` | bigint | YES |
| `moca02_visuoconstr_skills_cube` | bigint | YES |
| `moca03_visuoconstr_skills_clock_cont` | bigint | YES |
| `moca04_visuoconstr_skills_clock_num` | bigint | YES |
| `moca05_visuoconstr_skills_clock_hands` | bigint | YES |
| `moca_visuospatial_executive_subscore` | bigint | YES |
| `moca06_naming_lion` | bigint | YES |
| `moca07_naming_rhino` | bigint | YES |
| `moca08_naming_camel` | bigint | YES |
| `moca_naming_subscore` | bigint | YES |
| `moca09_attention_forward_digit_span` | bigint | YES |
| `moca10_attention_backward_digit_span` | bigint | YES |
| `moca_attention_digits_subscore` | bigint | YES |
| `moca11_attention_vigilance` | bigint | YES |
| `moca12_attention_serial_7s` | bigint | YES |
| `moca13_sentence_repetition` | bigint | YES |
| `moca14_verbal_fluency_number_of_words` | bigint | YES |
| `moca15_verbal_fluency` | bigint | YES |
| `moca_language_subscore` | bigint | YES |
| `moca16_abstraction` | bigint | YES |
| `moca_abstraction_subscore` | bigint | YES |
| `moca17_delayed_recall_face` | bigint | YES |
| `moca18_delayed_recall_velvet` | bigint | YES |
| `moca19_delayed_recall_church` | bigint | YES |
| `moca20_delayed_recall_daisy` | bigint | YES |
| `moca21_delayed_recall_red` | bigint | YES |
| `moca_delayed_recall_subscore` | bigint | YES |
| `moca_delayed_recall_subscore_optnl_cat_cue` | double precision | YES |
| `moca_delayed_recall_subscore_optnl_mult_choice` | double precision | YES |
| `moca22_orientation_date_score` | double precision | YES |
| `moca23_orientation_month_score` | double precision | YES |
| `moca24_orientation_year_score` | double precision | YES |
| `moca25_orientation_day_score` | double precision | YES |
| `moca26_orientation_place_score` | double precision | YES |
| `moca27_orientation_city_score` | double precision | YES |
| `moca_orientation_subscore` | bigint | YES |
| `code_education_12years_complete` | bigint | YES |
| `education_12years_complete` | text | YES |
| `moca_total_score` | bigint | YES |

### `biofind_raw.modified_schwab_england_adl` — 118 rows, 6 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `mod_schwab_england_pct_adl_score` | bigint | YES |
| `mod_schwab_england_on_off_med` | double precision | YES |

### `biofind_raw.pd_features` — 123 rows, 21 columns

| Column | Type | Nullable |
|---|---|---|
| `rec_id` | bigint | YES |
| `f_status` | text | YES |
| `patno` | bigint | YES |
| `event_id` | text | YES |
| `pag_name` | text | YES |
| `infodt` | text | YES |
| `sxmo` | double precision | YES |
| `sxyear` | bigint | YES |
| `pddxdt` | text | YES |
| `pddxest` | text | YES |
| `dxtremor` | text | YES |
| `dxrigid` | text | YES |
| `dxbrady` | text | YES |
| `dxposins` | text | YES |
| `dxothsx` | text | YES |
| `dxothcm` | text | YES |
| `domside` | bigint | YES |
| `orig_entry` | text | YES |
| `last_update` | text | YES |
| `query` | double precision | YES |
| `site_aprv` | text | YES |

### `biofind_raw.pd_medical_history` — 424 rows, 20 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `diagnosis` | text | YES |
| `initial_diagnosis` | double precision | YES |
| `most_recent_diagnosis` | double precision | YES |
| `change_in_diagnosis` | double precision | YES |
| `change_in_diagnosis_months_after_baseline` | double precision | YES |
| `surgery_for_parkinson_disease` | double precision | YES |
| `pd_diagnosis_months_after_baseline` | double precision | YES |
| `age_at_diagnosis` | double precision | YES |
| `pd_medication_initiation_months_after_baseline` | double precision | YES |
| `pd_medication_start_months_after_baseline` | double precision | YES |
| `use_of_pd_medication` | text | YES |
| `pd_medication_recent_use_months_after_baseline` | double precision | YES |
| `on_levodopa` | text | YES |
| `on_dopamine_agonist` | text | YES |
| `on_other_pd_medications` | text | YES |
| `diagnosis_type` | double precision | YES |

### `biofind_raw.rbd_stiasny_kolster` — 213 rows, 51 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `code_rbd_info_source` | bigint | YES |
| `code_rbd01_vivid_dreams` | bigint | YES |
| `code_rbd02_aggressive_or_action_packed_dreams` | bigint | YES |
| `code_rbd03_nocturnal_behaviour` | bigint | YES |
| `code_rbd04_move_arms_legs_during_sleep` | bigint | YES |
| `code_rbd05_hurt_bed_partner` | bigint | YES |
| `code_rbd06_1_speaking_in_sleep` | bigint | YES |
| `code_rbd06_2_sudden_limb_movements` | bigint | YES |
| `code_rbd06_3_complex_movements` | bigint | YES |
| `code_rbd06_4_things_fell_down` | bigint | YES |
| `code_rbd07_my_movements_awake_me` | bigint | YES |
| `code_rbd08_remember_dreams` | bigint | YES |
| `code_rbd09_sleep_is_disturbed` | bigint | YES |
| `code_rbd10a_stroke` | bigint | YES |
| `code_rbd10b_head_trauma` | bigint | YES |
| `code_rbd10c_parkinsonism` | bigint | YES |
| `code_rbd10d_rls` | bigint | YES |
| `code_rbd10e_narcolepsy` | bigint | YES |
| `code_rbd10f_depression` | bigint | YES |
| `code_rbd10g_epilepsy` | bigint | YES |
| `code_rbd10h_brain_inflammatory_disease` | bigint | YES |
| `code_rbd10i_other` | bigint | YES |
| `code_rbd10_nervous_system_disease` | double precision | YES |
| `rbd_info_source` | text | YES |
| `rbd01_vivid_dreams` | text | YES |
| `rbd02_aggressive_or_action_packed_dreams` | text | YES |
| `rbd03_nocturnal_behaviour` | text | YES |
| `rbd04_move_arms_legs_during_sleep` | text | YES |
| `rbd05_hurt_bed_partner` | text | YES |
| `rbd06_1_speaking_in_sleep` | text | YES |
| `rbd06_2_sudden_limb_movements` | text | YES |
| `rbd06_3_complex_movements` | text | YES |
| `rbd06_4_things_fell_down` | text | YES |
| `rbd07_my_movements_awake_me` | text | YES |
| `rbd08_remember_dreams` | text | YES |
| `rbd09_sleep_is_disturbed` | text | YES |
| `rbd10a_stroke` | text | YES |
| `rbd10b_head_trauma` | text | YES |
| `rbd10c_parkinsonism` | text | YES |
| `rbd10d_rls` | text | YES |
| `rbd10e_narcolepsy` | text | YES |
| `rbd10f_depression` | text | YES |
| `rbd10g_epilepsy` | text | YES |
| `rbd10h_brain_inflammatory_disease` | text | YES |
| `rbd10i_other` | text | YES |
| `rbd10_nervous_system_disease` | double precision | YES |
| `rbd_summary_score` | bigint | YES |

### `biofind_raw.rem_sleep_stiasny_kolster` — 213 rows, 51 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `code_rbd_info_source` | bigint | YES |
| `code_rbd01_vivid_dreams` | bigint | YES |
| `code_rbd02_aggressive_or_action_packed_dreams` | bigint | YES |
| `code_rbd03_nocturnal_behaviour` | bigint | YES |
| `code_rbd04_move_arms_legs_during_sleep` | bigint | YES |
| `code_rbd05_hurt_bed_partner` | bigint | YES |
| `code_rbd06_1_speaking_in_sleep` | bigint | YES |
| `code_rbd06_2_sudden_limb_movements` | bigint | YES |
| `code_rbd06_3_complex_movements` | bigint | YES |
| `code_rbd06_4_things_fell_down` | bigint | YES |
| `code_rbd07_my_movements_awake_me` | bigint | YES |
| `code_rbd08_remember_dreams` | bigint | YES |
| `code_rbd09_sleep_is_disturbed` | bigint | YES |
| `code_rbd10a_stroke` | bigint | YES |
| `code_rbd10b_head_trauma` | bigint | YES |
| `code_rbd10c_parkinsonism` | bigint | YES |
| `code_rbd10d_rls` | bigint | YES |
| `code_rbd10e_narcolepsy` | bigint | YES |
| `code_rbd10f_depression` | bigint | YES |
| `code_rbd10g_epilepsy` | bigint | YES |
| `code_rbd10h_brain_inflammatory_disease` | bigint | YES |
| `code_rbd10i_other` | bigint | YES |
| `code_rbd10_nervous_system_disease` | double precision | YES |
| `rbd_info_source` | text | YES |
| `rbd01_vivid_dreams` | text | YES |
| `rbd02_aggressive_or_action_packed_dreams` | text | YES |
| `rbd03_nocturnal_behaviour` | text | YES |
| `rbd04_move_arms_legs_during_sleep` | text | YES |
| `rbd05_hurt_bed_partner` | text | YES |
| `rbd06_1_speaking_in_sleep` | text | YES |
| `rbd06_2_sudden_limb_movements` | text | YES |
| `rbd06_3_complex_movements` | text | YES |
| `rbd06_4_things_fell_down` | text | YES |
| `rbd07_my_movements_awake_me` | text | YES |
| `rbd08_remember_dreams` | text | YES |
| `rbd09_sleep_is_disturbed` | text | YES |
| `rbd10a_stroke` | text | YES |
| `rbd10b_head_trauma` | text | YES |
| `rbd10c_parkinsonism` | text | YES |
| `rbd10d_rls` | text | YES |
| `rbd10e_narcolepsy` | text | YES |
| `rbd10f_depression` | text | YES |
| `rbd10g_epilepsy` | text | YES |
| `rbd10h_brain_inflammatory_disease` | text | YES |
| `rbd10i_other` | text | YES |
| `rbd10_nervous_system_disease` | double precision | YES |
| `rbd_summary_score` | bigint | YES |

### `biofind_raw.smoking_and_alcohol_history` — 51 rows, 33 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `tobacco_ever_used` | text | YES |
| `tobacco_current_use` | text | YES |
| `smoked_100_more_cigarettes` | text | YES |
| `alcohol_ever_used` | text | YES |
| `alcohol_current_use` | text | YES |
| `tobacco_recent_use` | double precision | YES |
| `tobacco_prior_use` | double precision | YES |
| `tobacco_start_age` | double precision | YES |
| `tobacco_stop_age` | double precision | YES |
| `tobacco_product_type` | double precision | YES |
| `cigarettes_per_day` | double precision | YES |
| `cigarettes_packs_per_day` | double precision | YES |
| `alcohol_recent_use` | double precision | YES |
| `alcohol_prior_use` | double precision | YES |
| `alcohol_start_age` | double precision | YES |
| `alcohol_stop_age` | double precision | YES |
| `alcohol_use_frequency` | double precision | YES |
| `alcohol_drinks_daily_range` | double precision | YES |
| `alcohol_six_more_drinks_frequency` | double precision | YES |
| `alcohol_related_hospitalization` | double precision | YES |
| `cigarettes_per_day_current` | double precision | YES |
| `cigarettes_per_day_past` | double precision | YES |
| `smoke_exposure_home` | double precision | YES |
| `smoke_exposure_work` | double precision | YES |
| `smoke_exposure_other_areas` | double precision | YES |
| `alcohol_drinks_day` | double precision | YES |
| `alcohol_consumed_years` | double precision | YES |
| `alcohol_consumption_change` | double precision | YES |
| `alcohol_inc_dec` | double precision | YES |

### `biofind_raw.use_of_pd_medication` — 222 rows, 19 columns

| Column | Type | Nullable |
|---|---|---|
| `rec_id` | bigint | YES |
| `f_status` | text | YES |
| `patno` | bigint | YES |
| `event_id` | text | YES |
| `pag_name` | text | YES |
| `infodt` | text | YES |
| `pdmedyn` | bigint | YES |
| `onldopa` | double precision | YES |
| `ondopag` | double precision | YES |
| `onamantd` | double precision | YES |
| `onmaobih` | double precision | YES |
| `onother` | double precision | YES |
| `pdmeddt` | text | YES |
| `pdmedtm` | text | YES |
| `nupdrtm` | text | YES |
| `orig_entry` | text | YES |
| `last_update` | text | YES |
| `query` | double precision | YES |
| `site_aprv` | text | YES |

## Schema `pdbp_raw` (52 tables)

_PDBP external prediction cohort (includes April 2026 LONI expansion)_

### `pdbp_raw.adverseevents` — 538 rows, 20 columns

| Column | Type | Nullable |
|---|---|---|
| `study_id` | double precision | YES |
| `dataset` | text | YES |
| `adverseevents_required_fields_sitename` | text | YES |
| `adverseevents_required_fields_visittyppdbp` | text | YES |
| `adverseevents_required_fields_visitdate` | double precision | YES |
| `adverseevents_required_fields_guid` | text | YES |
| `adverseevents_required_fields_associated_guid` | double precision | YES |
| `adverseevents_required_fields_ageyrs` | text | YES |
| `adverseevents_required_fields_ageremaindrmonths` | double precision | YES |
| `adverseevents_required_fields_ageval` | double precision | YES |
| `adverseevents_adverse_event_indicator_advrsevntduringstudyind` | text | YES |
| `adverseevents_adverse_event_description_advrsevntstartdatetime` | text | YES |
| `adverseevents_adverse_event_description_advrsevntseverscale` | text | YES |
| `adverseevents_adverse_event_description_advevntrelatednessscale` | text | YES |
| `adverseevents_adverse_event_description_advevntstdyintrvntactta` | text | YES |
| `adverseevents_adverse_event_description_advevntothractiontakent` | text | YES |
| `adverseevents_adverse_event_description_advrsevntoutcomstatus` | text | YES |
| `adverseevents_adverse_event_description_seriousadvrsevntind` | text | YES |
| `adverseevents_adverse_event_description_advrsevntverbatimtermte` | text | YES |
| `adverseevents_adverse_event_description_adverseevntenddatetime` | text | YES |

### `pdbp_raw.amp_pd_case_control` — 1,610 rows, 5 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `diagnosis_at_baseline` | text | YES |
| `diagnosis_latest` | text | YES |
| `case_control_other_at_baseline` | text | YES |
| `case_control_other_latest` | text | YES |

### `pdbp_raw.behavioralhistory` — 255 rows, 30 columns

| Column | Type | Nullable |
|---|---|---|
| `study_id` | bigint | YES |
| `dataset` | text | YES |
| `behavioralhistory_required_fields_sitename` | text | YES |
| `behavioralhistory_required_fields_visittyppdbp` | text | YES |
| `behavioralhistory_required_fields_visitdate` | double precision | YES |
| `behavioralhistory_required_fields_guid` | text | YES |
| `behavioralhistory_required_fields_associated_guid` | double precision | YES |
| `behavioralhistory_required_fields_ageyrs` | text | YES |
| `behavioralhistory_required_fields_ageremaindrmonths` | double precision | YES |
| `behavioralhistory_required_fields_ageval` | double precision | YES |
| `behavioralhistory_smoking_history_everusedtobaccoind` | text | YES |
| `behavioralhistory_smoking_history_tobcorcntuseindpdbp` | text | YES |
| `behavioralhistory_smoking_history_tobcoprioruseind` | text | YES |
| `behavioralhistory_smoking_history_tobcousestrtageval` | double precision | YES |
| `behavioralhistory_smoking_history_tobcousecurntind` | text | YES |
| `behavioralhistory_smoking_history_tobcousestopageval` | double precision | YES |
| `behavioralhistory_smoking_history_tobcoprodctusedtyppdbp` | text | YES |
| `behavioralhistory_smoking_history_tobcocigaretsmokeddlyavgnum` | text | YES |
| `behavioralhistory_alcohol_history_everusedalcoholind` | text | YES |
| `behavioralhistory_alcohol_history_alcrcntuseindpdbp` | text | YES |
| `behavioralhistory_alcohol_history_alcprioruseind` | text | YES |
| `behavioralhistory_alcohol_history_alcusestrtageval` | double precision | YES |
| `behavioralhistory_alcohol_history_alccurntuseind` | text | YES |
| `behavioralhistory_alcohol_history_alcusestopageval` | double precision | YES |
| `behavioralhistory_alcohol_history_alcusefreq` | text | YES |
| `behavioralhistory_alcohol_history_alcdrinkingdayavgdrinks` | text | YES |
| `behavioralhistory_alcohol_history_alcconsume6moredrinkfreq` | text | YES |
| `behavioralhistory_alcohol_history_alcuserelatedhospind` | text | YES |
| `behavioralhistory_drug_history_drgsubstcurrntillicitusecat` | text | YES |
| `behavioralhistory_drug_history_drgsubillctusecatpdbp` | text | YES |

### `pdbp_raw.biospecimen_csf_abeta_tau` — 3 rows, 8 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `sample_type` | text | YES |
| `test_name` | text | YES |
| `test_value` | double precision | YES |
| `test_units` | text | YES |

### `pdbp_raw.biospecimen_other` — 4,985 rows, 8 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `sample_type` | text | YES |
| `test_name` | text | YES |
| `test_value` | double precision | YES |
| `test_units` | text | YES |

### `pdbp_raw.csfcollfollowupphonecall` — 375 rows, 15 columns

| Column | Type | Nullable |
|---|---|---|
| `study_id` | bigint | YES |
| `dataset` | text | YES |
| `csfcollfollowupphonecall_required_fields_sitename` | text | YES |
| `csfcollfollowupphonecall_required_fields_visittyppdbp` | text | YES |
| `csfcollfollowupphonecall_required_fields_visitdate` | double precision | YES |
| `csfcollfollowupphonecall_required_fields_guid` | text | YES |
| `csfcollfollowupphonecall_required_fields_associated_guid` | double precision | YES |
| `csfcollfollowupphonecall_required_fields_ageyrs` | text | YES |
| `csfcollfollowupphonecall_required_fields_ageremaindrmonths` | double precision | YES |
| `csfcollfollowupphonecall_required_fields_ageval` | double precision | YES |
| `csfcollfollowupphonecall_phone_interview_contactmadeonphoneind` | text | YES |
| `csfcollfollowupphonecall_phone_interview_sampcollunuslsymptmaft` | text | YES |
| `csfcollfollowupphonecall_phone_interview_csfcollunuslsymptmmedc` | text | YES |
| `csfcollfollowupphonecall_phone_interview_seriousadvrsevntind` | text | YES |
| `csfcollfollowupphonecall_phone_interview_advrsevntverbatimtermt` | text | YES |

### `pdbp_raw.csfdatacollectionform` — 415 rows, 53 columns

| Column | Type | Nullable |
|---|---|---|
| `study_id` | bigint | YES |
| `dataset` | text | YES |
| `csfdatacollectionform_required_fields_sitename` | text | YES |
| `csfdatacollectionform_required_fields_visittyppdbp` | text | YES |
| `csfdatacollectionform_required_fields_visitdate` | double precision | YES |
| `csfdatacollectionform_required_fields_guid` | text | YES |
| `csfdatacollectionform_required_fields_associated_guid` | double precision | YES |
| `csfdatacollectionform_required_fields_ageyrs` | text | YES |
| `csfdatacollectionform_required_fields_ageremaindrmonths` | double precision | YES |
| `csfdatacollectionform_required_fields_ageval` | double precision | YES |
| `csfdatacollectionform_sample_collection_data_sampcolldatetim` | text | YES |
| `csfdatacollectionform_sample_collection_data_sampcoll12hrsfa` | text | YES |
| `csfdatacollectionform_sample_collection_data_lstmealdatetime` | text | YES |
| `csfdatacollectionform_sample_collection_data_lstmealtyp` | text | YES |
| `csfdatacollectionform_sample_collection_data_sampcolltyneedl` | text | YES |
| `csfdatacollectionform_sample_collection_data_csfsampcollmthd` | text | YES |
| `csfdatacollectionform_sample_collection_data_csfcolllumbarpu` | text | YES |
| `csfdatacollectionform_sample_collection_data_csfcolllumbarpu_1` | text | YES |
| `csfdatacollectionform_sample_collection_data_sampcollvol` | double precision | YES |
| `csfdatacollectionform_sample_collection_data_sampcollcentrif` | text | YES |
| `csfdatacollectionform_sample_collection_data_centrifgrate` | double precision | YES |
| `csfdatacollectionform_sample_collection_data_sampcolltempcen` | double precision | YES |
| `csfdatacollectionform_sample_collection_data_sampcolldatetim_1` | text | YES |
| `csfdatacollectionform_sample_collection_data_sampcolltotalvo` | double precision | YES |
| `csfdatacollectionform_sample_collection_data_sampcolltotalnu` | double precision | YES |
| `csfdatacollectionform_sample_collection_data_sampcollpartdis` | text | YES |
| `csfdatacollectionform_sample_collection_data_sampplacedfreez` | text | YES |
| `csfdatacollectionform_sample_collection_data_tempmeasrfreeze` | double precision | YES |
| `csfdatacollectionform_sample_collection_data_sampcollsentloc` | text | YES |
| `csfdatacollectionform_white_blood_count_wbc_locallabcsfcolls` | text | YES |
| `csfdatacollectionform_white_blood_count_wbc_sampobtind` | text | YES |
| `csfdatacollectionform_white_blood_count_wbc_csfcolllocallabr` | double precision | YES |
| `csfdatacollectionform_white_blood_count_wbc_sampcollbloodcel` | text | YES |
| `csfdatacollectionform_white_blood_count_wbc_csfcollresltind` | text | YES |
| `csfdatacollectionform_white_blood_count_wbc_labtestcomment` | text | YES |
| `csfdatacollectionform_red_blood_count_rbc_locallabcsfcollstd` | text | YES |
| `csfdatacollectionform_red_blood_count_rbc_sampobtind` | text | YES |
| `csfdatacollectionform_red_blood_count_rbc_csfcolllocallabres` | double precision | YES |
| `csfdatacollectionform_red_blood_count_rbc_sampcollbloodcellc` | text | YES |
| `csfdatacollectionform_red_blood_count_rbc_csfcollresltind` | text | YES |
| `csfdatacollectionform_red_blood_count_rbc_labtestcomment` | text | YES |
| `csfdatacollectionform_total_protein_locallabcsfcollstdytyp` | text | YES |
| `csfdatacollectionform_total_protein_sampobtind` | text | YES |
| `csfdatacollectionform_total_protein_csfcolllocallabreslt` | double precision | YES |
| `csfdatacollectionform_total_protein_sampcolltotalproteinuom` | text | YES |
| `csfdatacollectionform_total_protein_csfcollresltind` | text | YES |
| `csfdatacollectionform_total_protein_labtestcomment` | text | YES |
| `csfdatacollectionform_total_glucose_locallabcsfcollstdytyp` | text | YES |
| `csfdatacollectionform_total_glucose_sampobtind` | text | YES |
| `csfdatacollectionform_total_glucose_csfcolllocallabreslt` | double precision | YES |
| `csfdatacollectionform_total_glucose_sampcolltotalglucoseuom` | text | YES |
| `csfdatacollectionform_total_glucose_csfcollresltind` | text | YES |
| `csfdatacollectionform_total_glucose_labtestcomment` | text | YES |

### `pdbp_raw.demographics` — 1,610 rows, 9 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `age_at_baseline` | bigint | YES |
| `sex` | text | YES |
| `ethnicity` | text | YES |
| `race` | text | YES |
| `education_level_years` | text | YES |

### `pdbp_raw.earlyterminationquest` — 24 rows, 16 columns

| Column | Type | Nullable |
|---|---|---|
| `study_id` | bigint | YES |
| `dataset` | text | YES |
| `earlyterminationquest_required_fields_sitename` | text | YES |
| `earlyterminationquest_required_fields_visittyppdbp` | text | YES |
| `earlyterminationquest_required_fields_visitdate` | double precision | YES |
| `earlyterminationquest_required_fields_guid` | text | YES |
| `earlyterminationquest_required_fields_associated_guid` | double precision | YES |
| `earlyterminationquest_required_fields_ageyrs` | text | YES |
| `earlyterminationquest_required_fields_ageremaindrmonths` | double precision | YES |
| `earlyterminationquest_required_fields_ageval` | double precision | YES |
| `earlyterminationquest_questions_earlyterm_prtpwhy` | text | YES |
| `earlyterminationquest_questions_earlyterm_importntfactor` | text | YES |
| `earlyterminationquest_questions_earlyterm_stopprtpwhy` | text | YES |
| `earlyterminationquest_questions_earlyterm_stopfactor` | text | YES |
| `earlyterminationquest_questions_earlyterm_stdysatisfctn` | text | YES |
| `earlyterminationquest_questions_earlyterm_expectns` | text | YES |

### `pdbp_raw.enrollment` — 1,592 rows, 8 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `enrollment_months_after_baseline` | double precision | YES |
| `informed_consent_months_after_baseline` | double precision | YES |
| `prodromal_category` | text | YES |
| `study_arm` | text | YES |

### `pdbp_raw.epworth_sleepiness_scale` — 2,991 rows, 22 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `ess_info_source` | double precision | YES |
| `code_ess0101_sitting_and_reading` | bigint | YES |
| `code_ess0102_watching_tv` | bigint | YES |
| `code_ess0103_sitting_inactive_in_public_place` | bigint | YES |
| `code_ess0104_passenger_in_car_for_hour` | bigint | YES |
| `code_ess0105_lying_down_to_rest_in_afternoon` | bigint | YES |
| `code_ess0106_sitting_and_talking_to_someone` | bigint | YES |
| `code_ess0107_sitting_after_lunch` | bigint | YES |
| `code_ess0108_car_stopped_in_traffic` | bigint | YES |
| `ess0101_sitting_and_reading` | text | YES |
| `ess0102_watching_tv` | text | YES |
| `ess0103_sitting_inactive_in_public_place` | text | YES |
| `ess0104_passenger_in_car_for_hour` | text | YES |
| `ess0105_lying_down_to_rest_in_afternoon` | text | YES |
| `ess0106_sitting_and_talking_to_someone` | text | YES |
| `ess0107_sitting_after_lunch` | text | YES |
| `ess0108_car_stopped_in_traffic` | text | YES |
| `ess_summary_score` | bigint | YES |

### `pdbp_raw.epworthsleepinessscale` — 628 rows, 19 columns

| Column | Type | Nullable |
|---|---|---|
| `study_id` | bigint | YES |
| `dataset` | text | YES |
| `epworthsleepinessscale_required_fields_sitename` | text | YES |
| `epworthsleepinessscale_required_fields_visittyppdbp` | text | YES |
| `epworthsleepinessscale_required_fields_visitdate` | double precision | YES |
| `epworthsleepinessscale_required_fields_guid` | text | YES |
| `epworthsleepinessscale_required_fields_associated_guid` | double precision | YES |
| `epworthsleepinessscale_required_fields_ageyrs` | text | YES |
| `epworthsleepinessscale_required_fields_ageremaindrmonths` | double precision | YES |
| `epworthsleepinessscale_required_fields_ageval` | double precision | YES |
| `epworthsleepinessscale_ess_ess_sittingreading` | bigint | YES |
| `epworthsleepinessscale_ess_ess_watchingtv` | bigint | YES |
| `epworthsleepinessscale_ess_ess_sittinginactive` | bigint | YES |
| `epworthsleepinessscale_ess_ess_passengerincar` | bigint | YES |
| `epworthsleepinessscale_ess_ess_lyingdowntorest` | bigint | YES |
| `epworthsleepinessscale_ess_ess_sittingtalking` | bigint | YES |
| `epworthsleepinessscale_ess_ess_sittinglunchnoalc` | bigint | YES |
| `epworthsleepinessscale_ess_ess_dozingintraffc` | bigint | YES |
| `epworthsleepinessscale_ess_ess_totalscore` | bigint | YES |

### `pdbp_raw.family_history_pd` — 1,585 rows, 7 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `biological_mother_with_pd` | text | YES |
| `biological_father_with_pd` | text | YES |
| `other_relative_with_pd` | text | YES |

### `pdbp_raw.familyhistory` — 279 rows, 82 columns

| Column | Type | Nullable |
|---|---|---|
| `study_id` | double precision | YES |
| `dataset` | text | YES |
| `familyhistory_required_fields_sitename` | text | YES |
| `familyhistory_required_fields_visittyppdbp` | text | YES |
| `familyhistory_required_fields_visitdate` | double precision | YES |
| `familyhistory_required_fields_guid` | text | YES |
| `familyhistory_required_fields_associated_guid` | double precision | YES |
| `familyhistory_required_fields_ageyrs` | text | YES |
| `familyhistory_required_fields_ageremaindrmonths` | double precision | YES |
| `familyhistory_required_fields_ageval` | double precision | YES |
| `familyhistory_alzheimer_s_disease_famhistmedclcondtyp` | text | YES |
| `familyhistory_alzheimer_s_disease_famhistmedclcondind` | text | YES |
| `familyhistory_alzheimer_s_disease_famhistmedclcondreltvtyp` | text | YES |
| `familyhistory_amytrophic_lateral_sclerosis_famhistmedclcondtyp` | text | YES |
| `familyhistory_amytrophic_lateral_sclerosis_famhistmedclcondind` | text | YES |
| `familyhistory_amytrophic_lateral_sclerosis_famhistmedclcondrelt` | text | YES |
| `familyhistory_ataxia_famhistmedclcondtyp` | text | YES |
| `familyhistory_ataxia_famhistmedclcondind` | text | YES |
| `familyhistory_ataxia_famhistmedclcondreltvtyp` | text | YES |
| `familyhistory_autism_famhistmedclcondtyp` | text | YES |
| `familyhistory_autism_famhistmedclcondind` | text | YES |
| `familyhistory_autism_famhistmedclcondreltvtyp` | text | YES |
| `familyhistory_bi_polar_disorder_famhistmedclcondtyp` | text | YES |
| `familyhistory_bi_polar_disorder_famhistmedclcondind` | text | YES |
| `familyhistory_bi_polar_disorder_famhistmedclcondreltvtyp` | text | YES |
| `familyhistory_brain_aneurysm_famhistmedclcondtyp` | text | YES |
| `familyhistory_brain_aneurysm_famhistmedclcondind` | text | YES |
| `familyhistory_brain_aneurysm_famhistmedclcondreltvtyp` | text | YES |
| `familyhistory_cancer_famhistmedclcondtyp` | text | YES |
| `familyhistory_cancer_famhistmedclcondind` | text | YES |
| `familyhistory_cancer_famhistmedclcondreltvtyp` | text | YES |
| `familyhistory_dementia_famhistmedclcondtyp` | text | YES |
| `familyhistory_dementia_famhistmedclcondind` | text | YES |
| `familyhistory_dementia_famhistmedclcondreltvtyp` | text | YES |
| `familyhistory_depression_famhistmedclcondtyp` | text | YES |
| `familyhistory_depression_famhistmedclcondind` | text | YES |
| `familyhistory_depression_famhistmedclcondreltvtyp` | text | YES |
| `familyhistory_diabetes_mellitus_famhistmedclcondtyp` | text | YES |
| `familyhistory_diabetes_mellitus_famhistmedclcondind` | text | YES |
| `familyhistory_diabetes_mellitus_famhistmedclcondreltvtyp` | text | YES |
| `familyhistory_dystonia_famhistmedclcondtyp` | text | YES |
| `familyhistory_dystonia_famhistmedclcondind` | text | YES |
| `familyhistory_dystonia_famhistmedclcondreltvtyp` | text | YES |
| `familyhistory_epilepsy_famhistmedclcondtyp` | text | YES |
| `familyhistory_epilepsy_famhistmedclcondind` | text | YES |
| `familyhistory_epilepsy_famhistmedclcondreltvtyp` | text | YES |
| `familyhistory_heart_disease_famhistmedclcondtyp` | text | YES |
| `familyhistory_heart_disease_famhistmedclcondind` | text | YES |
| `familyhistory_heart_disease_famhistmedclcondreltvtyp` | text | YES |
| `familyhistory_hypertension_famhistmedclcondtyp` | text | YES |
| `familyhistory_hypertension_famhistmedclcondind` | text | YES |
| `familyhistory_hypertension_famhistmedclcondreltvtyp` | text | YES |
| `familyhistory_memory_loss_famhistmedclcondtyp` | text | YES |
| `familyhistory_memory_loss_famhistmedclcondind` | text | YES |
| `familyhistory_memory_loss_famhistmedclcondreltvtyp` | text | YES |
| `familyhistory_migraines_famhistmedclcondtyp` | text | YES |
| `familyhistory_migraines_famhistmedclcondind` | text | YES |
| `familyhistory_migraines_famhistmedclcondreltvtyp` | text | YES |
| `familyhistory_multiple_sclerosis_famhistmedclcondtyp` | text | YES |
| `familyhistory_multiple_sclerosis_famhistmedclcondind` | text | YES |
| `familyhistory_multiple_sclerosis_famhistmedclcondreltvtyp` | text | YES |
| `familyhistory_muscle_disease_famhistmedclcondtyp` | text | YES |
| `familyhistory_muscle_disease_famhistmedclcondind` | text | YES |
| `familyhistory_muscle_disease_famhistmedclcondreltvtyp` | text | YES |
| `familyhistory_parkinson_s_disease_famhistmedclcondtyp` | text | YES |
| `familyhistory_parkinson_s_disease_famhistmedclcondind` | text | YES |
| `familyhistory_parkinson_s_disease_famhistmedclcondreltvtyp` | text | YES |
| `familyhistory_schizophrenia_famhistmedclcondtyp` | text | YES |
| `familyhistory_schizophrenia_famhistmedclcondind` | text | YES |
| `familyhistory_schizophrenia_famhistmedclcondreltvtyp` | text | YES |
| `familyhistory_stroke_famhistmedclcondtyp` | text | YES |
| `familyhistory_stroke_famhistmedclcondind` | text | YES |
| `familyhistory_stroke_famhistmedclcondreltvtyp` | text | YES |
| `familyhistory_suicide_or_suicide_attempt_famhistmedclcondtyp` | text | YES |
| `familyhistory_suicide_or_suicide_attempt_famhistmedclcondind` | text | YES |
| `familyhistory_suicide_or_suicide_attempt_famhistmedclcondreltvt` | text | YES |
| `familyhistory_tourette_syndrome_famhistmedclcondtyp` | text | YES |
| `familyhistory_tourette_syndrome_famhistmedclcondind` | text | YES |
| `familyhistory_tourette_syndrome_famhistmedclcondreltvtyp` | text | YES |
| `familyhistory_additional_conditions_famhistmedclcondtyp` | text | YES |
| `familyhistory_additional_conditions_famhistmedclcondind` | text | YES |
| `familyhistory_additional_conditions_famhistmedclcondreltvtyp` | text | YES |

### `pdbp_raw.ham_a` — 533 rows, 25 columns

| Column | Type | Nullable |
|---|---|---|
| `study_id` | bigint | YES |
| `dataset` | text | YES |
| `ham_a_required_fields_sitename` | text | YES |
| `ham_a_required_fields_visittyppdbp` | text | YES |
| `ham_a_required_fields_visitdate` | double precision | YES |
| `ham_a_required_fields_guid` | text | YES |
| `ham_a_required_fields_associated_guid` | double precision | YES |
| `ham_a_required_fields_ageyrs` | text | YES |
| `ham_a_required_fields_ageremaindrmonths` | double precision | YES |
| `ham_a_required_fields_ageval` | double precision | YES |
| `ham_a_ham_a_hamaanxiousmoodscore` | bigint | YES |
| `ham_a_ham_a_hamatensionscore` | bigint | YES |
| `ham_a_ham_a_hamafearscore` | bigint | YES |
| `ham_a_ham_a_hamainsomniascore` | bigint | YES |
| `ham_a_ham_a_hamaintellectualscore` | bigint | YES |
| `ham_a_ham_a_hamadepressedmoodscore` | bigint | YES |
| `ham_a_ham_a_hamasomaticmuscularscore` | bigint | YES |
| `ham_a_ham_a_hamasomaticsensoryscore` | bigint | YES |
| `ham_a_ham_a_hamacardiovascularsymptomscore` | bigint | YES |
| `ham_a_ham_a_hamarespiratorysymptomscore` | bigint | YES |
| `ham_a_ham_a_hamagastrointestinalsymptomscr` | bigint | YES |
| `ham_a_ham_a_hamagenitourinarysymptomscore` | bigint | YES |
| `ham_a_ham_a_hamaautonomicsymptomscore` | bigint | YES |
| `ham_a_ham_a_hamabehaviorinterviewscore` | bigint | YES |
| `ham_a_ham_a_ham_a_totalscore` | bigint | YES |

### `pdbp_raw.hdrs` — 524 rows, 29 columns

| Column | Type | Nullable |
|---|---|---|
| `study_id` | bigint | YES |
| `dataset` | text | YES |
| `hdrs_required_fields_sitename` | text | YES |
| `hdrs_required_fields_visittyppdbp` | text | YES |
| `hdrs_required_fields_visitdate` | double precision | YES |
| `hdrs_required_fields_guid` | text | YES |
| `hdrs_required_fields_associated_guid` | double precision | YES |
| `hdrs_required_fields_ageyrs` | text | YES |
| `hdrs_required_fields_ageremaindrmonths` | double precision | YES |
| `hdrs_required_fields_ageval` | double precision | YES |
| `hdrs_hdrs_hdrsdeprsdmdind` | bigint | YES |
| `hdrs_hdrs_hdrsgltmndind` | bigint | YES |
| `hdrs_hdrs_hdrsucdind` | bigint | YES |
| `hdrs_hdrs_hdrserlyngtinsmnind` | bigint | YES |
| `hdrs_hdrs_hdrsmddlngtinsmnind` | bigint | YES |
| `hdrs_hdrs_hdrserlymorninsmnind` | bigint | YES |
| `hdrs_hdrs_hdrswrkactdifcltind` | bigint | YES |
| `hdrs_hdrs_hdrsretrdtnind` | bigint | YES |
| `hdrs_hdrs_hdrsagttnind` | bigint | YES |
| `hdrs_hdrs_hdrsanxpsycdifcltind` | bigint | YES |
| `hdrs_hdrs_hdrsanxsomtcind` | bigint | YES |
| `hdrs_hdrs_hdrssomtcsymptmind` | bigint | YES |
| `hdrs_hdrs_hdrsgenrlsomtcsymptmind` | bigint | YES |
| `hdrs_hdrs_hdrsgentlsymptmind` | bigint | YES |
| `hdrs_hdrs_hdrshypchdssind` | bigint | YES |
| `hdrs_hdrs_hdrswgtlospatind` | double precision | YES |
| `hdrs_hdrs_hdrswgtlosmeasrind` | double precision | YES |
| `hdrs_hdrs_hdrsinsgtind` | bigint | YES |
| `hdrs_hdrs_hdrstotscore` | double precision | YES |

### `pdbp_raw.informedconsent` — 258 rows, 16 columns

| Column | Type | Nullable |
|---|---|---|
| `study_id` | bigint | YES |
| `dataset` | text | YES |
| `informedconsent_required_fields_sitename` | text | YES |
| `informedconsent_required_fields_visittyppdbp` | text | YES |
| `informedconsent_required_fields_visitdate` | double precision | YES |
| `informedconsent_required_fields_guid` | text | YES |
| `informedconsent_required_fields_associated_guid` | double precision | YES |
| `informedconsent_required_fields_ageyrs` | text | YES |
| `informedconsent_required_fields_ageremaindrmonths` | double precision | YES |
| `informedconsent_required_fields_ageval` | double precision | YES |
| `informedconsent_informed_consent_infconsntobtind` | text | YES |
| `informedconsent_informed_consent_informconsntobtndatetime` | text | YES |
| `informedconsent_enrollment_enrldstdyind` | text | YES |
| `informedconsent_enrollment_enrldstdydatetime` | text | YES |
| `informedconsent_randomization_randomizedind` | text | YES |
| `informedconsent_randomization_randomizeddatetime` | text | YES |

### `pdbp_raw.labtesttracking` — 1,772 rows, 24 columns

| Column | Type | Nullable |
|---|---|---|
| `study_id` | double precision | YES |
| `dataset` | text | YES |
| `labtesttracking_required_fields_sitename` | text | YES |
| `labtesttracking_required_fields_visittyppdbp` | text | YES |
| `labtesttracking_required_fields_visitdate` | double precision | YES |
| `labtesttracking_required_fields_guid` | text | YES |
| `labtesttracking_required_fields_associated_guid` | double precision | YES |
| `labtesttracking_required_fields_ageyrs` | text | YES |
| `labtesttracking_required_fields_ageremaindrmonths` | double precision | YES |
| `labtesttracking_required_fields_ageval` | double precision | YES |
| `labtesttracking_lab_panel_status_labtestperfind` | text | YES |
| `labtesttracking_lab_panel_status_labspecmncolldatetime` | text | YES |
| `labtesttracking_basic_metabolic_labs_metaboliclabtstpfmdtype` | text | YES |
| `labtesttracking_basic_metabolic_labs_labtestresltval` | text | YES |
| `labtesttracking_basic_metabolic_labs_labtestresltuom` | text | YES |
| `labtesttracking_basic_metabolic_labs_labtestresltstatuspdbp` | text | YES |
| `labtesttracking_liver_function_and_other_metabolic_labs_live` | text | YES |
| `labtesttracking_liver_function_and_other_metabolic_labs_labt` | text | YES |
| `labtesttracking_liver_function_and_other_metabolic_labs_labt_1` | text | YES |
| `labtesttracking_liver_function_and_other_metabolic_labs_labt_2` | text | YES |
| `labtesttracking_hematology_labs_hematologylabtstpfmdtype` | text | YES |
| `labtesttracking_hematology_labs_labtestresltval` | text | YES |
| `labtesttracking_hematology_labs_labtestresltuom` | text | YES |
| `labtesttracking_hematology_labs_labtestresltstatuspdbp` | text | YES |

### `pdbp_raw.lbd_npi` — 619 rows, 149 columns

| Column | Type | Nullable |
|---|---|---|
| `study_id` | bigint | YES |
| `dataset` | text | YES |
| `lbd_npi_required_sitename` | text | YES |
| `lbd_npi_required_visittyppdbp` | text | YES |
| `lbd_npi_required_guid` | text | YES |
| `lbd_npi_required_associated_guid` | double precision | YES |
| `lbd_npi_required_visitdate` | double precision | YES |
| `lbd_npi_required_ageyrs` | text | YES |
| `lbd_npi_required_ageremaindrmonths` | double precision | YES |
| `lbd_npi_required_ageval` | double precision | YES |
| `lbd_npi_delusions_npicontscreendelusind` | text | YES |
| `lbd_npi_delusions_npidangerind` | text | YES |
| `lbd_npi_delusions_npistealingind` | text | YES |
| `lbd_npi_delusions_npispouseaffairind` | text | YES |
| `lbd_npi_delusions_npiguestsind` | text | YES |
| `lbd_npi_delusions_npinotwhosayind` | text | YES |
| `lbd_npi_delusions_npihousenothomeind` | text | YES |
| `lbd_npi_delusions_npifamilyabandonind` | text | YES |
| `lbd_npi_delusions_npifictioncharacterind` | text | YES |
| `lbd_npi_delusions_npiothrdelusionsind` | text | YES |
| `lbd_npi_delusions_npisymptomfrequencyscale` | double precision | YES |
| `lbd_npi_delusions_npidelusseverityscale` | double precision | YES |
| `lbd_npi_delusions_npicaregivemotdistressscore` | double precision | YES |
| `lbd_npi_hallucinations_npicontscreenhallucinind` | text | YES |
| `lbd_npi_hallucinations_npihearsvoicesind` | text | YES |
| `lbd_npi_hallucinations_npitalksimaginaryind` | text | YES |
| `lbd_npi_hallucinations_npiseesimaginaryind` | text | YES |
| `lbd_npi_hallucinations_npismellsimaginaryind` | text | YES |
| `lbd_npi_hallucinations_npifeelsimaginaryind` | text | YES |
| `lbd_npi_hallucinations_npitastesimaginaryind` | text | YES |
| `lbd_npi_hallucinations_npiothrhallucinationsind` | text | YES |
| `lbd_npi_hallucinations_npisymptomfrequencyscale` | double precision | YES |
| `lbd_npi_hallucinations_npihallucinationseverityscale` | double precision | YES |
| `lbd_npi_hallucinations_npicaregivemotdistressscore` | double precision | YES |
| `lbd_npi_agitation_aggression_npicontscreenaggressionind` | text | YES |
| `lbd_npi_agitation_aggression_npiresistactivitiesind` | text | YES |
| `lbd_npi_agitation_aggression_npistubbornind` | text | YES |
| `lbd_npi_agitation_aggression_npiuncooperativeind` | text | YES |
| `lbd_npi_agitation_aggression_npihardhandleind` | text | YES |
| `lbd_npi_agitation_aggression_npicursesind` | text | YES |
| `lbd_npi_agitation_aggression_npislamdoorind` | text | YES |
| `lbd_npi_agitation_aggression_npihurthitothrind` | text | YES |
| `lbd_npi_agitation_aggression_npiothraggressionind` | text | YES |
| `lbd_npi_agitation_aggression_npisymptomfrequencyscale` | double precision | YES |
| `lbd_npi_agitation_aggression_npiaggressionseverityscale` | double precision | YES |
| `lbd_npi_agitation_aggression_npicaregivemotdistressscore` | double precision | YES |
| `lbd_npi_depression_dysphoria_npicontscreendepressionind` | text | YES |
| `lbd_npi_depression_dysphoria_npitearfulind` | text | YES |
| `lbd_npi_depression_dysphoria_npisadind` | text | YES |
| `lbd_npi_depression_dysphoria_npifailureind` | text | YES |
| `lbd_npi_depression_dysphoria_npibadpersonind` | text | YES |
| `lbd_npi_depression_dysphoria_npidiscouragedind` | text | YES |
| `lbd_npi_depression_dysphoria_npiburdenfamilyind` | text | YES |
| `lbd_npi_depression_dysphoria_npisuicideind` | text | YES |
| `lbd_npi_depression_dysphoria_npiothrdepressionind` | text | YES |
| `lbd_npi_depression_dysphoria_npisymptomcontfrequencyscale` | double precision | YES |
| `lbd_npi_depression_dysphoria_npidepressionseverityscale` | double precision | YES |
| `lbd_npi_depression_dysphoria_npicaregivemotdistressscore` | double precision | YES |
| `lbd_npi_anxiety_npicontscreenanxietyind` | text | YES |
| `lbd_npi_anxiety_npiworryplaneventind` | text | YES |
| `lbd_npi_anxiety_npifeelshakyind` | text | YES |
| `lbd_npi_anxiety_npisighnervousind` | text | YES |
| `lbd_npi_anxiety_npipoundheartind` | text | YES |
| `lbd_npi_anxiety_npiavoidplacesind` | text | YES |
| `lbd_npi_anxiety_npiclingyind` | text | YES |
| `lbd_npi_anxiety_npiothranxietyind` | text | YES |
| `lbd_npi_anxiety_npisymptomfrequencyscale` | double precision | YES |
| `lbd_npi_anxiety_npianxietyseverityscale` | double precision | YES |
| `lbd_npi_anxiety_npicaregivemotdistressscore` | double precision | YES |
| `lbd_npi_elation_euphoria_npicontscreenelationind` | text | YES |
| `lbd_npi_elation_euphoria_npiexcesshappyind` | text | YES |
| `lbd_npi_elation_euphoria_npifunnynotfunnyind` | text | YES |
| `lbd_npi_elation_euphoria_npichildhumorind` | text | YES |
| `lbd_npi_elation_euphoria_npitellsbadjokesind` | text | YES |
| `lbd_npi_elation_euphoria_npichildpranksind` | text | YES |
| `lbd_npi_elation_euphoria_npitalkbigind` | text | YES |
| `lbd_npi_elation_euphoria_npiothrelationind` | text | YES |
| `lbd_npi_elation_euphoria_npisymptomcontfrequencyscale` | double precision | YES |
| `lbd_npi_elation_euphoria_npielationseverityscale` | double precision | YES |
| `lbd_npi_elation_euphoria_npicaregivemotdistressscore` | double precision | YES |
| `lbd_npi_apathy_indifference_npicontscreenapathyind` | text | YES |
| `lbd_npi_apathy_indifference_npilessspontaneousind` | text | YES |
| `lbd_npi_apathy_indifference_npiinitiateconvoind` | text | YES |
| `lbd_npi_apathy_indifference_npilessaffectionateind` | text | YES |
| `lbd_npi_apathy_indifference_npichoresind` | text | YES |
| `lbd_npi_apathy_indifference_npidisinterestind` | text | YES |
| `lbd_npi_apathy_indifference_npinofriendfamilyind` | text | YES |
| `lbd_npi_apathy_indifference_npilessenthusiasmind` | text | YES |
| `lbd_npi_apathy_indifference_npiothrapathyind` | text | YES |
| `lbd_npi_apathy_indifference_npiapathyfrequencyscale` | double precision | YES |
| `lbd_npi_apathy_indifference_npiapathyseverityscale` | double precision | YES |
| `lbd_npi_apathy_indifference_npicaregivemotdistressscore` | double precision | YES |
| `lbd_npi_disinhibition_npicontscreendisinhibitionind` | text | YES |
| `lbd_npi_disinhibition_npiactimpulsiveind` | text | YES |
| `lbd_npi_disinhibition_npitalkstrangersind` | text | YES |
| `lbd_npi_disinhibition_npiinsensitiveremarksind` | text | YES |
| `lbd_npi_disinhibition_npisexualremarksind` | text | YES |
| `lbd_npi_disinhibition_npipersonalpublicind` | text | YES |
| `lbd_npi_disinhibition_npitouchhugind` | text | YES |
| `lbd_npi_disinhibition_npiothrdisinhibitionind` | text | YES |
| `lbd_npi_disinhibition_npisymptomcontfrequencyscale` | double precision | YES |
| `lbd_npi_disinhibition_npidisinhibitionseverityscale` | double precision | YES |
| `lbd_npi_disinhibition_npicaregivemotdistressscore` | double precision | YES |
| `lbd_npi_irritability_lability_npicontscreenirritabilityind` | text | YES |
| `lbd_npi_irritability_lability_npibadtemperind` | text | YES |
| `lbd_npi_irritability_lability_npirapidmoodshiftind` | text | YES |
| `lbd_npi_irritability_lability_npiflashangerind` | text | YES |
| `lbd_npi_irritability_lability_npiimpatientind` | text | YES |
| `lbd_npi_irritability_lability_npicrankyind` | text | YES |
| `lbd_npi_irritability_lability_npiargumentativeind` | text | YES |
| `lbd_npi_irritability_lability_npiothrirritabilityind` | text | YES |
| `lbd_npi_irritability_lability_npisymptomcontfrequencyscale` | double precision | YES |
| `lbd_npi_irritability_lability_npiirritabilityseverityscale` | double precision | YES |
| `lbd_npi_irritability_lability_npicaregivemotdistressscore` | double precision | YES |
| `lbd_npi_aberrant_motor_behavior_npicontscreenmotorbehaviorind` | text | YES |
| `lbd_npi_aberrant_motor_behavior_npipacesind` | text | YES |
| `lbd_npi_aberrant_motor_behavior_npirummagesind` | text | YES |
| `lbd_npi_aberrant_motor_behavior_npiclothesoffonind` | text | YES |
| `lbd_npi_aberrant_motor_behavior_npirepetitivehabitsind` | text | YES |
| `lbd_npi_aberrant_motor_behavior_npirepetitiveactivitiesind` | text | YES |
| `lbd_npi_aberrant_motor_behavior_npifidgetsind` | text | YES |
| `lbd_npi_aberrant_motor_behavior_npiothrmotorbehaviorind` | text | YES |
| `lbd_npi_aberrant_motor_behavior_npisymptomcontfrequencyscale` | double precision | YES |
| `lbd_npi_aberrant_motor_behavior_npimotorbehavseverityscale` | double precision | YES |
| `lbd_npi_aberrant_motor_behavior_npicaregivemotdistressscore` | double precision | YES |
| `lbd_npi_sleep_npicontscreennightbehaviorind` | text | YES |
| `lbd_npi_sleep_npihardfallasleepind` | text | YES |
| `lbd_npi_sleep_npigetupnightind` | text | YES |
| `lbd_npi_sleep_npiinappropriateactivityind` | text | YES |
| `lbd_npi_sleep_npiawakeguardianind` | text | YES |
| `lbd_npi_sleep_npiwakenightthinkdayind` | text | YES |
| `lbd_npi_sleep_npiawakeearlyind` | text | YES |
| `lbd_npi_sleep_npisleepdayind` | text | YES |
| `lbd_npi_sleep_npiothrnightbehaviorind` | text | YES |
| `lbd_npi_sleep_npinightbehavfrequencyscale` | double precision | YES |
| `lbd_npi_sleep_npinightbehavseverityscale` | double precision | YES |
| `lbd_npi_sleep_npicaregivemotdistressscore` | double precision | YES |
| `lbd_npi_appetite_and_eating_disorders_npicontscreenappetiteind` | text | YES |
| `lbd_npi_appetite_and_eating_disorders_npiappetitlossind` | text | YES |
| `lbd_npi_appetite_and_eating_disorders_npiappetitgainind` | text | YES |
| `lbd_npi_appetite_and_eating_disorders_npiwgtlossind` | text | YES |
| `lbd_npi_appetite_and_eating_disorders_npiwgtgainind` | text | YES |
| `lbd_npi_appetite_and_eating_disorders_npichangeeatbehavind` | text | YES |
| `lbd_npi_appetite_and_eating_disorders_npichangefoodprefind` | text | YES |
| `lbd_npi_appetite_and_eating_disorders_npiunusualeatbehavind` | text | YES |
| `lbd_npi_appetite_and_eating_disorders_npiothrappetiteind` | text | YES |
| `lbd_npi_appetite_and_eating_disorders_npiappetitefrequencyscale` | double precision | YES |
| `lbd_npi_appetite_and_eating_disorders_npiappetiteseverityscale` | double precision | YES |
| `lbd_npi_appetite_and_eating_disorders_npicaregivemotdistresssco` | double precision | YES |

### `pdbp_raw.mayofluctuationscale` — 169 rows, 14 columns

| Column | Type | Nullable |
|---|---|---|
| `study_id` | bigint | YES |
| `dataset` | text | YES |
| `mayofluctuationscale_required_fields_sitename` | text | YES |
| `mayofluctuationscale_required_fields_visittyppdbp` | text | YES |
| `mayofluctuationscale_required_fields_visitdate` | double precision | YES |
| `mayofluctuationscale_required_fields_guid` | text | YES |
| `mayofluctuationscale_required_fields_associated_guid` | double precision | YES |
| `mayofluctuationscale_required_fields_ageyrs` | text | YES |
| `mayofluctuationscale_required_fields_ageremaindrmonths` | double precision | YES |
| `mayofluctuationscale_required_fields_ageval` | double precision | YES |
| `mayofluctuationscale_mayo_fluctuation_scale_mayofluctlethargicp` | text | YES |
| `mayofluctuationscale_mayo_fluctuation_scale_mayofluctsleeppdbp` | text | YES |
| `mayofluctuationscale_mayo_fluctuation_scale_mayofluctdisorganiz` | text | YES |
| `mayofluctuationscale_mayo_fluctuation_scale_mayofluctstarepdbp` | text | YES |

### `pdbp_raw.mayosleepquestionnaire` — 618 rows, 31 columns

| Column | Type | Nullable |
|---|---|---|
| `study_id` | bigint | YES |
| `dataset` | text | YES |
| `mayosleepquestionnaire_required_fields_sitename` | text | YES |
| `mayosleepquestionnaire_required_fields_visittyppdbp` | text | YES |
| `mayosleepquestionnaire_required_fields_visitdate` | double precision | YES |
| `mayosleepquestionnaire_required_fields_guid` | text | YES |
| `mayosleepquestionnaire_required_fields_associated_guid` | double precision | YES |
| `mayosleepquestionnaire_required_fields_ageyrs` | text | YES |
| `mayosleepquestionnaire_required_fields_ageremaindrmonths` | double precision | YES |
| `mayosleepquestionnaire_required_fields_ageval` | double precision | YES |
| `mayosleepquestionnaire_interviewee_reportertyp` | text | YES |
| `mayosleepquestionnaire_interviewee_msq_livewithsubjind` | text | YES |
| `mayosleepquestionnaire_interviewee_msq_sleepsameroomind` | text | YES |
| `mayosleepquestionnaire_interviewee_msq_sleepbehavrind` | text | YES |
| `mayosleepquestionnaire_questionnaire_msq_dreamsactind` | text | YES |
| `mayosleepquestionnaire_questionnaire_msq_dreamsactnumyr` | double precision | YES |
| `mayosleepquestionnaire_questionnaire_msq_dreamsactnummo` | double precision | YES |
| `mayosleepquestionnaire_questionnaire_msq_patientinjind` | text | YES |
| `mayosleepquestionnaire_questionnaire_msq_bedpartnerinjind` | text | YES |
| `mayosleepquestionnaire_questionnaire_msq_dreamsattackedind` | text | YES |
| `mayosleepquestionnaire_questionnaire_msq_dreamsmatchdetailsind` | text | YES |
| `mayosleepquestionnaire_questionnaire_msq_legjerkind` | text | YES |
| `mayosleepquestionnaire_questionnaire_msq_restlesslegind` | text | YES |
| `mayosleepquestionnaire_questionnaire_msq_legsensationdecreasein` | text | YES |
| `mayosleepquestionnaire_questionnaire_msq_legsensationworst` | text | YES |
| `mayosleepquestionnaire_questionnaire_msq_sleepwalk` | text | YES |
| `mayosleepquestionnaire_questionnaire_msq_snortawakeind` | text | YES |
| `mayosleepquestionnaire_questionnaire_msq_stopbreathingind` | text | YES |
| `mayosleepquestionnaire_questionnaire_msq_treatedstopbreathingin` | text | YES |
| `mayosleepquestionnaire_questionnaire_msq_legcrampsind` | text | YES |
| `mayosleepquestionnaire_questionnaire_msq_ratealert` | double precision | YES |

### `pdbp_raw.mds_updrs` — 700 rows, 98 columns

| Column | Type | Nullable |
|---|---|---|
| `study_id` | bigint | YES |
| `dataset` | text | YES |
| `mds_updrs_required_fields_sitename` | text | YES |
| `mds_updrs_required_fields_visittyppdbp` | text | YES |
| `mds_updrs_required_fields_visitdate` | double precision | YES |
| `mds_updrs_required_fields_guid` | text | YES |
| `mds_updrs_required_fields_associated_guid` | double precision | YES |
| `mds_updrs_required_fields_ageyrs` | text | YES |
| `mds_updrs_required_fields_ageremaindrmonths` | double precision | YES |
| `mds_updrs_required_fields_ageval` | double precision | YES |
| `mds_updrs_part_i_nm_edl_mdsupdrsprimrysrcinfotyp` | text | YES |
| `mds_updrs_part_i_nm_edl_mdsupdrsrcntcogimprmntscore` | double precision | YES |
| `mds_updrs_part_i_nm_edl_mdsupdrshallucpsychosscore` | double precision | YES |
| `mds_updrs_part_i_nm_edl_mdsupdrsdrpssmoodscore` | double precision | YES |
| `mds_updrs_part_i_nm_edl_mdsupdrsanxsmoodscore` | double precision | YES |
| `mds_updrs_part_i_nm_edl_mdsupdrsapathyscore` | double precision | YES |
| `mds_updrs_part_i_nm_edl_mdsupdrsdopmndysregsyndscore` | double precision | YES |
| `mds_updrs_part_i_nm_edl_mdsupdrsqstnnreinfoprovdrtyp` | text | YES |
| `mds_updrs_part_i_nm_edl_mdsupdrssleepprobscore` | double precision | YES |
| `mds_updrs_part_i_nm_edl_mdsupdrsdaytmsleepscore` | double precision | YES |
| `mds_updrs_part_i_nm_edl_mdsupdrspainothrsensscore` | double precision | YES |
| `mds_updrs_part_i_nm_edl_mdsupdrsurnryprobscore` | double precision | YES |
| `mds_updrs_part_i_nm_edl_mdsupdrsconstipprobscore` | double precision | YES |
| `mds_updrs_part_i_nm_edl_mdsupdrsliteheadstndngscore` | double precision | YES |
| `mds_updrs_part_i_nm_edl_mdsupdrsfatiguescore` | double precision | YES |
| `mds_updrs_part_i_nm_edl_mdsupdrs_partiscore` | double precision | YES |
| `mds_updrs_part_ii_m_edl_mdsupdrsspeechscore` | double precision | YES |
| `mds_updrs_part_ii_m_edl_mdsupdrsslivadroolscore` | double precision | YES |
| `mds_updrs_part_ii_m_edl_mdsupdrschwngswllwngscore` | double precision | YES |
| `mds_updrs_part_ii_m_edl_mdsupdrseatingtskscore` | double precision | YES |
| `mds_updrs_part_ii_m_edl_mdsupdrsdressingscore` | double precision | YES |
| `mds_updrs_part_ii_m_edl_mdsupdrshygienescore` | double precision | YES |
| `mds_updrs_part_ii_m_edl_mdsupdrshandwritingscore` | double precision | YES |
| `mds_updrs_part_ii_m_edl_mdsupdrshobbieothractscore` | double precision | YES |
| `mds_updrs_part_ii_m_edl_mdsupdrsturngbedscore` | double precision | YES |
| `mds_updrs_part_ii_m_edl_mdsupdrstremorscore` | double precision | YES |
| `mds_updrs_part_ii_m_edl_mdsupdrsgttngoutbedscore` | double precision | YES |
| `mds_updrs_part_ii_m_edl_mdsupdrswlkngbalancescore` | double precision | YES |
| `mds_updrs_part_ii_m_edl_mdsupdrsfreezingscore` | double precision | YES |
| `mds_updrs_part_ii_m_edl_mdsupdrs_partiiscore` | double precision | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrsptntprknsnmedind` | text | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrsptclinstateprknsnm` | text | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrsptntuseldopaind` | text | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrslstldopadosetm` | double precision | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrsfreeflowspeechscor` | bigint | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrsfacialexprscore` | bigint | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrsneckrigidscore` | bigint | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrsruerigidscore` | bigint | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrsluerigidscore` | bigint | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrsrlerigidscore` | bigint | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrsllerigidscore` | bigint | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrsfingertppngrtehnds` | double precision | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrsfingertppnglfthnds` | bigint | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrsrtehndscore` | bigint | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrslfthndscore` | bigint | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrsprontsupnrthndmvmt` | bigint | YES |
| `mds_updrs_part_iii_motor_examination_pronatsupinlfthndmvmntscor` | bigint | YES |
| `mds_updrs_part_iii_motor_examination_rtefttoetppngscore` | bigint | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrslftfttoetppngscore` | bigint | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrslegagiltyrtelegsco` | bigint | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrslegagiltylftlegsco` | bigint | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrsarisingfrmchrscore` | bigint | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrsgaitscore` | bigint | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrsfreezinggaitscore` | bigint | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrspostrlstabltyscore` | bigint | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrsposturescore` | bigint | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrsglblspontntymvmnts` | bigint | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrspostrltremorrthnds` | bigint | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrspostrltremrlfthnds` | bigint | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrskinetictremrrthnds` | bigint | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrskinetictremrlfthnd` | bigint | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrsresttremorampruesc` | bigint | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrsresttremorampluesc` | bigint | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrsresttremoramprlesc` | bigint | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrsresttremorampllesc` | bigint | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrsresttremramplipjaw` | bigint | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrsconstncyresttremrs` | bigint | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrsdyskchreadystnaprs` | text | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrsmvmntintrfrncescor` | text | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrshoehnyahrstagescor` | double precision | YES |
| `mds_updrs_part_iii_motor_examination_mdsupdrs_partiiiscore` | double precision | YES |
| `mds_updrs_part_iv_motor_complications_mdsupdrstmspntdyskscore` | bigint | YES |
| `mds_updrs_part_iv_motor_complications_mdsupdrsttlhrawkdysknum` | double precision | YES |
| `mds_updrs_part_iv_motor_complications_mdsupdrsttlhrdysknum` | double precision | YES |
| `mds_updrs_part_iv_motor_complications_mdsupdrsprcntdyskval` | double precision | YES |
| `mds_updrs_part_iv_motor_complications_mdsupdrsfuncimpactdysksco` | double precision | YES |
| `mds_updrs_part_iv_motor_complications_mdsupdrsttlhrawkoffstaten` | double precision | YES |
| `mds_updrs_part_iv_motor_complications_mdsupdrsttlhroffnum` | double precision | YES |
| `mds_updrs_part_iv_motor_complications_mdsupdrsprcntoffval` | double precision | YES |
| `mds_updrs_part_iv_motor_complications_mdsupdrstmspntoffstatesco` | double precision | YES |
| `mds_updrs_part_iv_motor_complications_mdsupdrsfuncimpactfluctsc` | double precision | YES |
| `mds_updrs_part_iv_motor_complications_mdsupdrscomplxtymtrflucts` | double precision | YES |
| `mds_updrs_part_iv_motor_complications_mdsupdrspainfloffstatdyst` | double precision | YES |
| `mds_updrs_part_iv_motor_complications_mdsupdrsttlhroffdemndystn` | double precision | YES |
| `mds_updrs_part_iv_motor_complications_mdsupdrsttlhroffwdystnian` | double precision | YES |
| `mds_updrs_part_iv_motor_complications_mdsupdrsprcntoffdystniava` | double precision | YES |
| `mds_updrs_part_iv_motor_complications_mdsupdrs_partivscore` | double precision | YES |
| `mds_updrs_total_score_mdsupdrs_totalscore` | double precision | YES |

### `pdbp_raw.mds_updrs_part_i` — 4,746 rows, 35 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `mds_updrs_part_i_primary_info_source` | text | YES |
| `code_upd2101_cognitive_impairment` | double precision | YES |
| `code_upd2102_hallucinations_and_psychosis` | double precision | YES |
| `code_upd2103_depressed_mood` | double precision | YES |
| `code_upd2104_anxious_mood` | double precision | YES |
| `code_upd2105_apathy` | double precision | YES |
| `code_upd2106_dopamine_dysregulation_syndrome_features` | double precision | YES |
| `upd2101_cognitive_impairment` | text | YES |
| `upd2102_hallucinations_and_psychosis` | text | YES |
| `upd2103_depressed_mood` | text | YES |
| `upd2104_anxious_mood` | text | YES |
| `upd2105_apathy` | text | YES |
| `upd2106_dopamine_dysregulation_syndrome_features` | text | YES |
| `mds_updrs_part_i_sub_score` | double precision | YES |
| `mds_updrs_part_i_pat_quest_primary_info_source` | text | YES |
| `code_upd2107_pat_quest_sleep_problems` | double precision | YES |
| `code_upd2108_pat_quest_daytime_sleepiness` | double precision | YES |
| `code_upd2109_pat_quest_pain_and_other_sensations` | double precision | YES |
| `code_upd2110_pat_quest_urinary_problems` | double precision | YES |
| `code_upd2111_pat_quest_constipation_problems` | double precision | YES |
| `code_upd2112_pat_quest_lightheadedness_on_standing` | double precision | YES |
| `code_upd2113_pat_quest_fatigue` | double precision | YES |
| `upd2107_pat_quest_sleep_problems` | text | YES |
| `upd2108_pat_quest_daytime_sleepiness` | text | YES |
| `upd2109_pat_quest_pain_and_other_sensations` | text | YES |
| `upd2110_pat_quest_urinary_problems` | text | YES |
| `upd2111_pat_quest_constipation_problems` | text | YES |
| `upd2112_pat_quest_lightheadedness_on_standing` | text | YES |
| `upd2113_pat_quest_fatigue` | text | YES |
| `mds_updrs_part_i_pat_quest_sub_score` | double precision | YES |
| `mds_updrs_part_i_summary_score` | bigint | YES |

### `pdbp_raw.mds_updrs_part_ii` — 5,026 rows, 32 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `mds_updrs_part_ii_primary_info_source` | double precision | YES |
| `code_upd2201_speech` | double precision | YES |
| `code_upd2202_saliva_and_drooling` | double precision | YES |
| `code_upd2203_chewing_and_swallowing` | double precision | YES |
| `code_upd2204_eating_tasks` | double precision | YES |
| `code_upd2205_dressing` | double precision | YES |
| `code_upd2206_hygiene` | double precision | YES |
| `code_upd2207_handwriting` | double precision | YES |
| `code_upd2208_doing_hobbies_and_other_activities` | double precision | YES |
| `code_upd2209_turning_in_bed` | double precision | YES |
| `code_upd2210_tremor` | double precision | YES |
| `code_upd2211_get_out_of_bed_car_or_deep_chair` | double precision | YES |
| `code_upd2212_walking_and_balance` | double precision | YES |
| `code_upd2213_freezing` | double precision | YES |
| `upd2201_speech` | text | YES |
| `upd2202_saliva_and_drooling` | text | YES |
| `upd2203_chewing_and_swallowing` | text | YES |
| `upd2204_eating_tasks` | text | YES |
| `upd2205_dressing` | text | YES |
| `upd2206_hygiene` | text | YES |
| `upd2207_handwriting` | text | YES |
| `upd2208_doing_hobbies_and_other_activities` | text | YES |
| `upd2209_turning_in_bed` | text | YES |
| `upd2210_tremor` | text | YES |
| `upd2211_get_out_of_bed_car_or_deep_chair` | text | YES |
| `upd2212_walking_and_balance` | text | YES |
| `upd2213_freezing` | text | YES |
| `mds_updrs_part_ii_summary_score` | double precision | YES |

### `pdbp_raw.mds_updrs_part_iii` — 5,025 rows, 77 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `code_upd2301_speech_problems` | bigint | YES |
| `code_upd2302_facial_expression` | bigint | YES |
| `code_upd2303a_rigidity_neck` | bigint | YES |
| `code_upd2303b_rigidity_rt_upper_extremity` | bigint | YES |
| `code_upd2303c_rigidity_left_upper_extremity` | bigint | YES |
| `code_upd2303d_rigidity_rt_lower_extremity` | bigint | YES |
| `code_upd2303e_rigidity_left_lower_extremity` | double precision | YES |
| `code_upd2304a_right_finger_tapping` | bigint | YES |
| `code_upd2304b_left_finger_tapping` | bigint | YES |
| `code_upd2305a_right_hand_movements` | bigint | YES |
| `code_upd2305b_left_hand_movements` | bigint | YES |
| `code_upd2306a_pron_sup_movement_right_hand` | bigint | YES |
| `code_upd2306b_pron_sup_movement_left_hand` | bigint | YES |
| `code_upd2307a_right_toe_tapping` | double precision | YES |
| `code_upd2307b_left_toe_tapping` | bigint | YES |
| `code_upd2308a_right_leg_agility` | bigint | YES |
| `code_upd2308b_left_leg_agility` | bigint | YES |
| `code_upd2309_arising_from_chair` | bigint | YES |
| `code_upd2310_gait` | bigint | YES |
| `code_upd2311_freezing_of_gait` | bigint | YES |
| `code_upd2312_postural_stability` | bigint | YES |
| `code_upd2313_posture` | bigint | YES |
| `code_upd2314_body_bradykinesia` | bigint | YES |
| `code_upd2315a_postural_tremor_of_right_hand` | bigint | YES |
| `code_upd2315b_postural_tremor_of_left_hand` | bigint | YES |
| `code_upd2316a_kinetic_tremor_of_right_hand` | bigint | YES |
| `code_upd2316b_kinetic_tremor_of_left_hand` | bigint | YES |
| `code_upd2317a_rest_tremor_amplitude_right_upper_extremity` | bigint | YES |
| `code_upd2317b_rest_tremor_amplitude_left_upper_extremity` | bigint | YES |
| `code_upd2317c_rest_tremor_amplitude_right_lower_extremity` | double precision | YES |
| `code_upd2317d_rest_tremor_amplitude_left_lower_extremity` | bigint | YES |
| `code_upd2317e_rest_tremor_amplitude_lip_or_jaw` | double precision | YES |
| `code_upd2318_consistency_of_rest_tremor` | double precision | YES |
| `upd2301_speech_problems` | text | YES |
| `upd2302_facial_expression` | text | YES |
| `upd2303a_rigidity_neck` | text | YES |
| `upd2303b_rigidity_rt_upper_extremity` | text | YES |
| `upd2303c_rigidity_left_upper_extremity` | text | YES |
| `upd2303d_rigidity_rt_lower_extremity` | text | YES |
| `upd2303e_rigidity_left_lower_extremity` | text | YES |
| `upd2304a_right_finger_tapping` | text | YES |
| `upd2304b_left_finger_tapping` | text | YES |
| `upd2305a_right_hand_movements` | text | YES |
| `upd2305b_left_hand_movements` | text | YES |
| `upd2306a_pron_sup_movement_right_hand` | text | YES |
| `upd2306b_pron_sup_movement_left_hand` | text | YES |
| `upd2307a_right_toe_tapping` | text | YES |
| `upd2307b_left_toe_tapping` | text | YES |
| `upd2308a_right_leg_agility` | text | YES |
| `upd2308b_left_leg_agility` | text | YES |
| `upd2309_arising_from_chair` | text | YES |
| `upd2310_gait` | text | YES |
| `upd2311_freezing_of_gait` | text | YES |
| `upd2312_postural_stability` | text | YES |
| `upd2313_posture` | text | YES |
| `upd2314_body_bradykinesia` | text | YES |
| `upd2315a_postural_tremor_of_right_hand` | text | YES |
| `upd2315b_postural_tremor_of_left_hand` | text | YES |
| `upd2316a_kinetic_tremor_of_right_hand` | text | YES |
| `upd2316b_kinetic_tremor_of_left_hand` | text | YES |
| `upd2317a_rest_tremor_amplitude_right_upper_extremity` | text | YES |
| `upd2317b_rest_tremor_amplitude_left_upper_extremity` | text | YES |
| `upd2317c_rest_tremor_amplitude_right_lower_extremity` | text | YES |
| `upd2317d_rest_tremor_amplitude_left_lower_extremity` | text | YES |
| `upd2317e_rest_tremor_amplitude_lip_or_jaw` | text | YES |
| `upd2318_consistency_of_rest_tremor` | text | YES |
| `upd2da_dyskinesias_during_exam` | text | YES |
| `upd2db_movements_interfere_with_ratings` | text | YES |
| `code_upd2hy_hoehn_and_yahr_stage` | double precision | YES |
| `upd2hy_hoehn_and_yahr_stage` | text | YES |
| `upd23a_medication_for_pd` | text | YES |
| `upd23b_clinical_state_on_medication` | text | YES |
| `mds_updrs_part_iii_summary_score` | double precision | YES |

### `pdbp_raw.mds_updrs_part_iv` — 4,745 rows, 17 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `code_upd2401_time_spent_with_dyskinesias` | bigint | YES |
| `code_upd2402_functional_impact_of_dyskinesias` | double precision | YES |
| `code_upd2403_time_spent_in_the_off_state` | double precision | YES |
| `code_upd2404_functional_impact_of_fluctuations` | double precision | YES |
| `code_upd2405_complexity_of_motor_fluctuations` | bigint | YES |
| `code_upd2406_painful_off_state_dystonia` | double precision | YES |
| `upd2401_time_spent_with_dyskinesias` | text | YES |
| `upd2402_functional_impact_of_dyskinesias` | text | YES |
| `upd2403_time_spent_in_the_off_state` | text | YES |
| `upd2404_functional_impact_of_fluctuations` | text | YES |
| `upd2405_complexity_of_motor_fluctuations` | text | YES |
| `upd2406_painful_off_state_dystonia` | text | YES |
| `mds_updrs_part_iv_summary_score` | bigint | YES |

### `pdbp_raw.mds_updrs_virtualvisit` — 40 rows, 98 columns

| Column | Type | Nullable |
|---|---|---|
| `study_id` | bigint | YES |
| `dataset` | text | YES |
| `mds_updrs_virtualvisit_required_fields_sitename` | text | YES |
| `mds_updrs_virtualvisit_required_fields_visittyppdbp` | text | YES |
| `mds_updrs_virtualvisit_required_fields_visitdate` | double precision | YES |
| `mds_updrs_virtualvisit_required_fields_guid` | text | YES |
| `mds_updrs_virtualvisit_required_fields_associated_guid` | double precision | YES |
| `mds_updrs_virtualvisit_required_fields_ageyrs` | text | YES |
| `mds_updrs_virtualvisit_required_fields_ageremaindrmonths` | double precision | YES |
| `mds_updrs_virtualvisit_required_fields_ageval` | double precision | YES |
| `mds_updrs_virtualvisit_part_i_nm_edl_mdsupdrsprimrysrcinfoty` | text | YES |
| `mds_updrs_virtualvisit_part_i_nm_edl_mdsupdrsrcntcogimprmnts` | double precision | YES |
| `mds_updrs_virtualvisit_part_i_nm_edl_mdsupdrshallucpsychossc` | double precision | YES |
| `mds_updrs_virtualvisit_part_i_nm_edl_mdsupdrsdrpssmoodscore` | double precision | YES |
| `mds_updrs_virtualvisit_part_i_nm_edl_mdsupdrsanxsmoodscore` | double precision | YES |
| `mds_updrs_virtualvisit_part_i_nm_edl_mdsupdrsapathyscore` | double precision | YES |
| `mds_updrs_virtualvisit_part_i_nm_edl_mdsupdrsdopmndysregsynd` | double precision | YES |
| `mds_updrs_virtualvisit_part_i_nm_edl_mdsupdrsqstnnreinfoprov` | text | YES |
| `mds_updrs_virtualvisit_part_i_nm_edl_mdsupdrssleepprobscore` | double precision | YES |
| `mds_updrs_virtualvisit_part_i_nm_edl_mdsupdrsdaytmsleepscore` | double precision | YES |
| `mds_updrs_virtualvisit_part_i_nm_edl_mdsupdrspainothrsenssco` | double precision | YES |
| `mds_updrs_virtualvisit_part_i_nm_edl_mdsupdrsurnryprobscore` | double precision | YES |
| `mds_updrs_virtualvisit_part_i_nm_edl_mdsupdrsconstipprobscor` | double precision | YES |
| `mds_updrs_virtualvisit_part_i_nm_edl_mdsupdrsliteheadstndngs` | double precision | YES |
| `mds_updrs_virtualvisit_part_i_nm_edl_mdsupdrsfatiguescore` | double precision | YES |
| `mds_updrs_virtualvisit_part_i_nm_edl_mdsupdrs_partiscore` | double precision | YES |
| `mds_updrs_virtualvisit_part_ii_m_edl_mdsupdrsspeechscore` | double precision | YES |
| `mds_updrs_virtualvisit_part_ii_m_edl_mdsupdrsslivadroolscore` | double precision | YES |
| `mds_updrs_virtualvisit_part_ii_m_edl_mdsupdrschwngswllwngsco` | double precision | YES |
| `mds_updrs_virtualvisit_part_ii_m_edl_mdsupdrseatingtskscore` | double precision | YES |
| `mds_updrs_virtualvisit_part_ii_m_edl_mdsupdrsdressingscore` | double precision | YES |
| `mds_updrs_virtualvisit_part_ii_m_edl_mdsupdrshygienescore` | double precision | YES |
| `mds_updrs_virtualvisit_part_ii_m_edl_mdsupdrshandwritingscor` | double precision | YES |
| `mds_updrs_virtualvisit_part_ii_m_edl_mdsupdrshobbieothractsc` | double precision | YES |
| `mds_updrs_virtualvisit_part_ii_m_edl_mdsupdrsturngbedscore` | double precision | YES |
| `mds_updrs_virtualvisit_part_ii_m_edl_mdsupdrstremorscore` | double precision | YES |
| `mds_updrs_virtualvisit_part_ii_m_edl_mdsupdrsgttngoutbedscor` | double precision | YES |
| `mds_updrs_virtualvisit_part_ii_m_edl_mdsupdrswlkngbalancesco` | double precision | YES |
| `mds_updrs_virtualvisit_part_ii_m_edl_mdsupdrsfreezingscore` | double precision | YES |
| `mds_updrs_virtualvisit_part_ii_m_edl_mdsupdrs_partiiscore` | double precision | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrspt` | text | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrspt_1` | text | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrspt_2` | text | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrsls` | double precision | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrsfr` | double precision | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrsfa` | double precision | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrsne` | double precision | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrsru` | double precision | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrslu` | double precision | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrsrl` | double precision | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrsll` | double precision | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrsfi` | double precision | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrsfi_1` | double precision | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrsrt` | double precision | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrslf` | double precision | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrspr` | double precision | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_pronatsupi` | double precision | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_rtefttoetp` | double precision | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrslf_1` | double precision | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrsle` | double precision | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrsle_1` | double precision | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrsar` | double precision | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrsga` | double precision | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrsfr_1` | double precision | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrspo` | double precision | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrspo_1` | double precision | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrsgl` | double precision | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrspo_2` | double precision | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrspo_3` | double precision | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrski` | double precision | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrski_1` | double precision | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrsre` | double precision | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrsre_1` | double precision | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrsre_2` | double precision | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrsre_3` | double precision | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrsre_4` | double precision | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrsco` | double precision | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrsdy` | text | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrsmv` | text | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrsho` | double precision | YES |
| `mds_updrs_virtualvisit_part_iii_motor_examination_mdsupdrs_p` | double precision | YES |
| `mds_updrs_virtualvisit_part_iv_motor_complications_mdsupdrst` | double precision | YES |
| `mds_updrs_virtualvisit_part_iv_motor_complications_mdsupdrst_1` | double precision | YES |
| `mds_updrs_virtualvisit_part_iv_motor_complications_mdsupdrst_2` | double precision | YES |
| `mds_updrs_virtualvisit_part_iv_motor_complications_mdsupdrsp` | double precision | YES |
| `mds_updrs_virtualvisit_part_iv_motor_complications_mdsupdrsf` | double precision | YES |
| `mds_updrs_virtualvisit_part_iv_motor_complications_mdsupdrst_3` | double precision | YES |
| `mds_updrs_virtualvisit_part_iv_motor_complications_mdsupdrst_4` | double precision | YES |
| `mds_updrs_virtualvisit_part_iv_motor_complications_mdsupdrsp_1` | double precision | YES |
| `mds_updrs_virtualvisit_part_iv_motor_complications_mdsupdrst_5` | double precision | YES |
| `mds_updrs_virtualvisit_part_iv_motor_complications_mdsupdrsf_1` | double precision | YES |
| `mds_updrs_virtualvisit_part_iv_motor_complications_mdsupdrsc` | double precision | YES |
| `mds_updrs_virtualvisit_part_iv_motor_complications_mdsupdrsp_2` | double precision | YES |
| `mds_updrs_virtualvisit_part_iv_motor_complications_mdsupdrst_6` | double precision | YES |
| `mds_updrs_virtualvisit_part_iv_motor_complications_mdsupdrst_7` | double precision | YES |
| `mds_updrs_virtualvisit_part_iv_motor_complications_mdsupdrsp_3` | double precision | YES |
| `mds_updrs_virtualvisit_part_iv_motor_complications_mdsupdrs_` | double precision | YES |
| `mds_updrs_virtualvisit_total_score_mdsupdrs_totalscore` | double precision | YES |

### `pdbp_raw.moca` — 3,267 rows, 43 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `moca01_alternating_trail_making` | double precision | YES |
| `moca02_visuoconstr_skills_cube` | double precision | YES |
| `moca03_visuoconstr_skills_clock_cont` | double precision | YES |
| `moca04_visuoconstr_skills_clock_num` | double precision | YES |
| `moca05_visuoconstr_skills_clock_hands` | double precision | YES |
| `moca_visuospatial_executive_subscore` | double precision | YES |
| `moca06_naming_lion` | double precision | YES |
| `moca07_naming_rhino` | double precision | YES |
| `moca08_naming_camel` | double precision | YES |
| `moca_naming_subscore` | bigint | YES |
| `moca09_attention_forward_digit_span` | double precision | YES |
| `moca10_attention_backward_digit_span` | double precision | YES |
| `moca_attention_digits_subscore` | double precision | YES |
| `moca11_attention_vigilance` | double precision | YES |
| `moca12_attention_serial_7s` | double precision | YES |
| `moca13_sentence_repetition` | double precision | YES |
| `moca14_verbal_fluency_number_of_words` | double precision | YES |
| `moca15_verbal_fluency` | double precision | YES |
| `moca_language_subscore` | double precision | YES |
| `moca16_abstraction` | double precision | YES |
| `moca_abstraction_subscore` | double precision | YES |
| `moca17_delayed_recall_face` | double precision | YES |
| `moca18_delayed_recall_velvet` | double precision | YES |
| `moca19_delayed_recall_church` | double precision | YES |
| `moca20_delayed_recall_daisy` | double precision | YES |
| `moca21_delayed_recall_red` | double precision | YES |
| `moca_delayed_recall_subscore` | double precision | YES |
| `moca_delayed_recall_subscore_optnl_cat_cue` | double precision | YES |
| `moca_delayed_recall_subscore_optnl_mult_choice` | double precision | YES |
| `moca22_orientation_date_score` | double precision | YES |
| `moca23_orientation_month_score` | double precision | YES |
| `moca24_orientation_year_score` | double precision | YES |
| `moca25_orientation_day_score` | double precision | YES |
| `moca26_orientation_place_score` | double precision | YES |
| `moca27_orientation_city_score` | double precision | YES |
| `moca_orientation_subscore` | double precision | YES |
| `code_education_12years_complete` | double precision | YES |
| `education_12years_complete` | text | YES |
| `moca_total_score` | bigint | YES |

### `pdbp_raw.moca_virtualvisit` — 23 rows, 25 columns

| Column | Type | Nullable |
|---|---|---|
| `study_id` | bigint | YES |
| `dataset` | text | YES |
| `moca_virtualvisit_required_fields_sitename` | text | YES |
| `moca_virtualvisit_required_fields_visittyppdbp` | text | YES |
| `moca_virtualvisit_required_fields_visitdate` | double precision | YES |
| `moca_virtualvisit_required_fields_guid` | text | YES |
| `moca_virtualvisit_required_fields_associated_guid` | double precision | YES |
| `moca_virtualvisit_required_fields_ageyrs` | text | YES |
| `moca_virtualvisit_required_fields_ageremaindrmonths` | double precision | YES |
| `moca_virtualvisit_required_fields_ageval` | double precision | YES |
| `moca_virtualvisit_moca_moca_visuospatialexec` | double precision | YES |
| `moca_virtualvisit_moca_moca_naming` | double precision | YES |
| `moca_virtualvisit_moca_moca_digits` | bigint | YES |
| `moca_virtualvisit_moca_moca_letters` | bigint | YES |
| `moca_virtualvisit_moca_moca_serial7` | bigint | YES |
| `moca_virtualvisit_moca_moca_langrepeat` | bigint | YES |
| `moca_virtualvisit_moca_moca_langfluency` | double precision | YES |
| `moca_virtualvisit_moca_moca_abstraction` | bigint | YES |
| `moca_virtualvisit_moca_moca_delydrecall` | bigint | YES |
| `moca_virtualvisit_moca_moca_delydrecalloptnlcatcue` | double precision | YES |
| `moca_virtualvisit_moca_moca_delydrecaloptnlmultchoice` | double precision | YES |
| `moca_virtualvisit_moca_moca_orient` | bigint | YES |
| `moca_virtualvisit_moca_moca_imageresponse` | double precision | YES |
| `moca_virtualvisit_moca_moca_eduind` | double precision | YES |
| `moca_virtualvisit_moca_moca_total` | bigint | YES |

### `pdbp_raw.modified_schwab_england_adl` — 3,003 rows, 6 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `mod_schwab_england_pct_adl_score` | bigint | YES |
| `mod_schwab_england_on_off_med` | double precision | YES |

### `pdbp_raw.modschwabandenglandscale` — 539 rows, 11 columns

| Column | Type | Nullable |
|---|---|---|
| `study_id` | bigint | YES |
| `dataset` | text | YES |
| `modschwabandenglandscale_required_fields_sitename` | text | YES |
| `modschwabandenglandscale_required_fields_visittyppdbp` | text | YES |
| `modschwabandenglandscale_required_fields_visitdate` | double precision | YES |
| `modschwabandenglandscale_required_fields_guid` | text | YES |
| `modschwabandenglandscale_required_fields_associated_guid` | double precision | YES |
| `modschwabandenglandscale_required_fields_ageyrs` | text | YES |
| `modschwabandenglandscale_required_fields_ageremaindrmonths` | double precision | YES |
| `modschwabandenglandscale_required_fields_ageval` | double precision | YES |
| `modschwabandenglandscale_scale_score_updrstscaleschengdallivscl` | double precision | YES |

### `pdbp_raw.neurologicalexam` — 606 rows, 117 columns

| Column | Type | Nullable |
|---|---|---|
| `study_id` | bigint | YES |
| `dataset` | text | YES |
| `neurologicalexam_required_fields_sitename` | text | YES |
| `neurologicalexam_required_fields_visittyppdbp` | text | YES |
| `neurologicalexam_required_fields_visitdate` | double precision | YES |
| `neurologicalexam_required_fields_guid` | text | YES |
| `neurologicalexam_required_fields_associated_guid` | double precision | YES |
| `neurologicalexam_required_fields_ageyrs` | text | YES |
| `neurologicalexam_required_fields_ageremaindrmonths` | double precision | YES |
| `neurologicalexam_required_fields_ageval` | double precision | YES |
| `neurologicalexam_neurological_examination_inclusnxclusncntrlind` | text | YES |
| `neurologicalexam_neurological_examination_neuroexamprimarydiagn` | text | YES |
| `neurologicalexam_neurological_examination_agediagnosyrs` | bigint | YES |
| `neurologicalexam_neurological_examination_agediagnosremaindrmon` | bigint | YES |
| `neurologicalexam_neurological_examination_secondarydxind` | text | YES |
| `neurologicalexam_neurological_examination_secondarydxcontributi` | text | YES |
| `neurologicalexam_neurological_examination_secondarydxnoncontrib` | text | YES |
| `neurologicalexam_neurological_examination_secondarydxftldoth` | double precision | YES |
| `neurologicalexam_neurological_examination_secondarydxftldsubtyp` | double precision | YES |
| `neurologicalexam_neurological_examination_neuroexamfind` | text | YES |
| `neurologicalexam_neurological_examination_neuroexamabnrmlfinddo` | text | YES |
| `neurologicalexam_mental_status_mentlstatexamfind` | text | YES |
| `neurologicalexam_mental_status_mentlstatexamcogabnrmlytxt` | text | YES |
| `neurologicalexam_mental_status_mentlstatexammoodabnrmlytxt` | text | YES |
| `neurologicalexam_cranial_nerves_cranlnervexamfind` | text | YES |
| `neurologicalexam_cranial_nerves_cranlnervvisualacuityfind` | text | YES |
| `neurologicalexam_cranial_nerves_cranlnervvisualacuityabnrmly` | text | YES |
| `neurologicalexam_cranial_nerves_cranlnervpapilledemafind` | text | YES |
| `neurologicalexam_cranial_nerves_cranlnervdiscpallorfind` | text | YES |
| `neurologicalexam_cranial_nerves_cranlnervpupilsfind` | text | YES |
| `neurologicalexam_cranial_nerves_cranlnervpupilsabnrmlyfind` | text | YES |
| `neurologicalexam_cranial_nerves_cranlnervptosislocfind` | text | YES |
| `neurologicalexam_cranial_nerves_cranlnervvisualfieldsfind` | text | YES |
| `neurologicalexam_cranial_nerves_cranlnervhemianopialoc` | text | YES |
| `neurologicalexam_cranial_nerves_cranlnervquadrantanopialoc` | text | YES |
| `neurologicalexam_cranial_nerves_cranlnervcentralscotomaloc` | text | YES |
| `neurologicalexam_cranial_nerves_cranlnerveyemovmntfind` | text | YES |
| `neurologicalexam_cranial_nerves_cranlnerveyemovmntlatfind` | text | YES |
| `neurologicalexam_cranial_nerves_cranlnerveyemovmntmedialfind` | text | YES |
| `neurologicalexam_cranial_nerves_cranlnerveyemovmntupfind` | text | YES |
| `neurologicalexam_cranial_nerves_cranlnerveyemovmntdownfind` | text | YES |
| `neurologicalexam_cranial_nerves_cranlnerveyemovmntsaccadesfind` | text | YES |
| `neurologicalexam_cranial_nerves_cranlnerveyemovmntsmoothpur` | text | YES |
| `neurologicalexam_cranial_nerves_cranlnervfaclnumbnessfind` | text | YES |
| `neurologicalexam_cranial_nerves_cranlnervfaclnumbnessloc` | text | YES |
| `neurologicalexam_cranial_nerves_cranlnervmassetweakfind` | text | YES |
| `neurologicalexam_cranial_nerves_cranlnervfaclstrngthfind` | text | YES |
| `neurologicalexam_cranial_nerves_cranlnervfaclstrngthrt` | text | YES |
| `neurologicalexam_cranial_nerves_cranlnervfaclstrngthlft` | text | YES |
| `neurologicalexam_cranial_nerves_cranlnervhearingfind` | text | YES |
| `neurologicalexam_cranial_nerves_cranlnervhearingabnrmlyfind` | text | YES |
| `neurologicalexam_cranial_nerves_cranlnervwebertestrsltfind` | text | YES |
| `neurologicalexam_cranial_nerves_cranlnervrinnetestrslt` | text | YES |
| `neurologicalexam_cranial_nerves_cranlnervpalatalelevatnfind` | text | YES |
| `neurologicalexam_cranial_nerves_cranlnervgagrflxfind` | text | YES |
| `neurologicalexam_cranial_nerves_cranlnervshouldshrugweakfind` | text | YES |
| `neurologicalexam_cranial_nerves_cranlnervscmweakfind` | text | YES |
| `neurologicalexam_cranial_nerves_cranlnervtongueweakfind` | text | YES |
| `neurologicalexam_cranial_nerves_cranlnervtongueothrfind` | text | YES |
| `neurologicalexam_motor_strength_motorstrgthexamfind` | text | YES |
| `neurologicalexam_motor_strength_motorstrgthdeltoidrtassmt` | double precision | YES |
| `neurologicalexam_motor_strength_motorstrgthbicepsrtassmt` | double precision | YES |
| `neurologicalexam_motor_strength_motorstrgthtricepsrtassmt` | double precision | YES |
| `neurologicalexam_motor_strength_motorstrgthextcarpiradrtassmt` | double precision | YES |
| `neurologicalexam_motor_strength_motorstrgthflexcarpiradrtassmt` | double precision | YES |
| `neurologicalexam_motor_strength_motorstrgthinterosrtassmt` | double precision | YES |
| `neurologicalexam_motor_strength_motorstrgthlliopsoasrtassmt` | double precision | YES |
| `neurologicalexam_motor_strength_motorstrgthquadcpsrtassmt` | double precision | YES |
| `neurologicalexam_motor_strength_motorstrgthhamstringrtassmt` | double precision | YES |
| `neurologicalexam_motor_strength_motorstrgthgastrocnmusrtassmt` | double precision | YES |
| `neurologicalexam_motor_strength_motorstrgthtblanterrtassmt` | double precision | YES |
| `neurologicalexam_motor_strength_motorstrgthdeltoidlftassmt` | double precision | YES |
| `neurologicalexam_motor_strength_motorstrgthbicepslftassmt` | double precision | YES |
| `neurologicalexam_motor_strength_motorstrgthtricepslftassmt` | double precision | YES |
| `neurologicalexam_motor_strength_motorstrgthextcarpiradlftassmt` | double precision | YES |
| `neurologicalexam_motor_strength_motorstrgthflexcarpiradltassmt` | double precision | YES |
| `neurologicalexam_motor_strength_motorstrgthinteroslftassmt` | double precision | YES |
| `neurologicalexam_motor_strength_motorstrgthlliopsoaslftassmt` | double precision | YES |
| `neurologicalexam_motor_strength_motorstrgthquadcpslftassmt` | double precision | YES |
| `neurologicalexam_motor_strength_motorstrgthhamstringlftassmt` | double precision | YES |
| `neurologicalexam_motor_strength_motorstrgthgastrocnmuslftassmt` | double precision | YES |
| `neurologicalexam_motor_strength_motorstrgthtblanterlftassmt` | double precision | YES |
| `neurologicalexam_motor_strength_motorstrgthatrophind` | text | YES |
| `neurologicalexam_motor_strength_motorstrgthfasciculationfind` | text | YES |
| `neurologicalexam_sensation_sensationassmtexamfind` | text | YES |
| `neurologicalexam_sensation_sensationproxmlarmrtfind` | text | YES |
| `neurologicalexam_sensation_sensationdistlarmrtfind` | text | YES |
| `neurologicalexam_sensation_sensationproxmllegrtfind` | text | YES |
| `neurologicalexam_sensation_sensationdistllegrtfind` | text | YES |
| `neurologicalexam_sensation_sensationproxmlarmlftfind` | text | YES |
| `neurologicalexam_sensation_sensationdistlarmlftfind` | text | YES |
| `neurologicalexam_sensation_sensationproxmlleglftfind` | text | YES |
| `neurologicalexam_sensation_sensationdistlleglftfind` | text | YES |
| `neurologicalexam_sensation_sensationstockingglovefind` | text | YES |
| `neurologicalexam_reflexes_reflexind` | text | YES |
| `neurologicalexam_reflexes_reflexbicepsrtassmt` | text | YES |
| `neurologicalexam_reflexes_reflextricepsrtassmt` | text | YES |
| `neurologicalexam_reflexes_reflexbrachioradrtassmt` | text | YES |
| `neurologicalexam_reflexes_reflexpatellarrtassmt` | text | YES |
| `neurologicalexam_reflexes_reflexachillesrtassmt` | text | YES |
| `neurologicalexam_reflexes_reflexbicepslftassmt` | text | YES |
| `neurologicalexam_reflexes_reflextricepslftassmt` | text | YES |
| `neurologicalexam_reflexes_reflexbrachioradltassmt` | text | YES |
| `neurologicalexam_reflexes_reflexpatellarlftassmt` | text | YES |
| `neurologicalexam_reflexes_reflexachilleslftassmt` | text | YES |
| `neurologicalexam_reflexes_reflexbabinskyind` | text | YES |
| `neurologicalexam_reflexes_reflexbabinskyloc` | text | YES |
| `neurologicalexam_reflexes_reflexothrtxt` | text | YES |
| `neurologicalexam_gait_gait` | text | YES |
| `neurologicalexam_other_findings_othrmovmntind` | text | YES |
| `neurologicalexam_other_findings_othrmovmntataxialoc` | text | YES |
| `neurologicalexam_other_findings_othrmovmntchorealoc` | text | YES |
| `neurologicalexam_other_findings_othrmovmntballismusloc` | text | YES |
| `neurologicalexam_other_findings_othrmovmntticsloc` | text | YES |
| `neurologicalexam_other_findings_othrmovmntathetosisloc` | text | YES |
| `neurologicalexam_other_findings_othrmovmntmyoclonusloc` | text | YES |
| `neurologicalexam_other_findings_othrexamfindtxt` | text | YES |

### `pdbp_raw.parkinsonism_meds` — 1,432 rows, 26 columns

| Column | Type | Nullable |
|---|---|---|
| `study_id` | double precision | YES |
| `dataset` | text | YES |
| `parkinsonism_meds_required_sitename` | text | YES |
| `parkinsonism_meds_required_visittyppdbp` | text | YES |
| `parkinsonism_meds_required_visitdate` | double precision | YES |
| `parkinsonism_meds_required_guid` | text | YES |
| `parkinsonism_meds_required_associated_guid` | double precision | YES |
| `parkinsonism_meds_required_ageyrs` | text | YES |
| `parkinsonism_meds_required_ageremaindrmonths` | double precision | YES |
| `parkinsonism_meds_required_ageval` | double precision | YES |
| `parkinsonism_meds_medications_medctnpriorconcompdbptyp1` | text | YES |
| `parkinsonism_meds_medications_medctnpriorconcomdose` | text | YES |
| `parkinsonism_meds_medications_medctnpriorconcomdoseuo` | text | YES |
| `parkinsonism_meds_medications_medctnfreqpdbpnum` | double precision | YES |
| `parkinsonism_meds_medications_medctnfreqpdbppillnum` | double precision | YES |
| `parkinsonism_meds_medications_medctnpriorconcomrtetyp` | text | YES |
| `parkinsonism_meds_medications_medctnpriorconcomhrslstdose` | double precision | YES |
| `parkinsonism_meds_medications_medctnpriorconcomminslstdose` | double precision | YES |
| `parkinsonism_meds_other_medications_specific_to_neurologic_d` | text | YES |
| `parkinsonism_meds_other_medications_specific_to_neurologic_d_1` | double precision | YES |
| `parkinsonism_meds_other_medications_specific_to_neurologic_d_2` | text | YES |
| `parkinsonism_meds_other_medications_specific_to_neurologic_d_3` | double precision | YES |
| `parkinsonism_meds_other_medications_specific_to_neurologic_d_4` | double precision | YES |
| `parkinsonism_meds_other_medications_specific_to_neurologic_d_5` | text | YES |
| `parkinsonism_meds_other_medications_specific_to_neurologic_d_6` | double precision | YES |
| `parkinsonism_meds_other_medications_specific_to_neurologic_d_7` | double precision | YES |

### `pdbp_raw.pd_medical_history` — 5,080 rows, 20 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `diagnosis` | text | YES |
| `initial_diagnosis` | text | YES |
| `most_recent_diagnosis` | text | YES |
| `change_in_diagnosis` | text | YES |
| `change_in_diagnosis_months_after_baseline` | double precision | YES |
| `surgery_for_parkinson_disease` | double precision | YES |
| `pd_diagnosis_months_after_baseline` | double precision | YES |
| `age_at_diagnosis` | double precision | YES |
| `pd_medication_initiation_months_after_baseline` | double precision | YES |
| `pd_medication_start_months_after_baseline` | double precision | YES |
| `use_of_pd_medication` | double precision | YES |
| `pd_medication_recent_use_months_after_baseline` | double precision | YES |
| `on_levodopa` | text | YES |
| `on_dopamine_agonist` | text | YES |
| `on_other_pd_medications` | text | YES |
| `diagnosis_type` | double precision | YES |

### `pdbp_raw.pdbp_neuropath_naccv11` — 16 rows, 165 columns

| Column | Type | Nullable |
|---|---|---|
| `study_id` | bigint | YES |
| `dataset` | text | YES |
| `pdbp_neuropath_naccv11_main_group_sitename` | text | YES |
| `pdbp_neuropath_naccv11_main_group_visittyppdbp` | text | YES |
| `pdbp_neuropath_naccv11_main_group_guid` | text | YES |
| `pdbp_neuropath_naccv11_main_group_associated_guid` | double precision | YES |
| `pdbp_neuropath_naccv11_main_group_ptid` | text | YES |
| `pdbp_neuropath_naccv11_main_group_pdbpdateformcompleted` | text | YES |
| `pdbp_neuropath_naccv11_main_group_npid` | text | YES |
| `pdbp_neuropath_naccv11_main_group_npsex` | text | YES |
| `pdbp_neuropath_naccv11_main_group_npdage` | bigint | YES |
| `pdbp_neuropath_naccv11_main_group_pdbpdateofdeath` | text | YES |
| `pdbp_neuropath_naccv11_brain_autopsy_nppmih` | text | YES |
| `pdbp_neuropath_naccv11_brain_autopsy_npfix` | bigint | YES |
| `pdbp_neuropath_naccv11_brain_autopsy_npfixx` | double precision | YES |
| `pdbp_neuropath_naccv11_gross_findings_and_overall_impression` | bigint | YES |
| `pdbp_neuropath_naccv11_gross_findings_and_overall_impression_1` | bigint | YES |
| `pdbp_neuropath_naccv11_gross_findings_and_overall_impression_2` | double precision | YES |
| `pdbp_neuropath_naccv11_gross_findings_and_overall_impression_3` | double precision | YES |
| `pdbp_neuropath_naccv11_gross_findings_and_overall_impression_4` | double precision | YES |
| `pdbp_neuropath_naccv11_gross_findings_and_overall_impression_5` | double precision | YES |
| `pdbp_neuropath_naccv11_gross_findings_and_overall_impression_6` | double precision | YES |
| `pdbp_neuropath_naccv11_gross_findings_and_overall_impression_7` | double precision | YES |
| `pdbp_neuropath_naccv11_methods_used_for_scoring_case_nptan` | double precision | YES |
| `pdbp_neuropath_naccv11_methods_used_for_scoring_case_nptanx` | double precision | YES |
| `pdbp_neuropath_naccv11_methods_used_for_scoring_case_npaban` | double precision | YES |
| `pdbp_neuropath_naccv11_methods_used_for_scoring_case_npabanx` | text | YES |
| `pdbp_neuropath_naccv11_methods_used_for_scoring_case_npasan` | double precision | YES |
| `pdbp_neuropath_naccv11_methods_used_for_scoring_case_npasanx` | double precision | YES |
| `pdbp_neuropath_naccv11_methods_used_for_scoring_case_nptdpan` | double precision | YES |
| `pdbp_neuropath_naccv11_methods_used_for_scoring_case_nptdpan_1` | double precision | YES |
| `pdbp_neuropath_naccv11_methods_used_for_scoring_case_nphismb` | double precision | YES |
| `pdbp_neuropath_naccv11_methods_used_for_scoring_case_nphisg` | double precision | YES |
| `pdbp_neuropath_naccv11_methods_used_for_scoring_case_nphisss` | double precision | YES |
| `pdbp_neuropath_naccv11_methods_used_for_scoring_case_nphist` | double precision | YES |
| `pdbp_neuropath_naccv11_methods_used_for_scoring_case_nphiso` | double precision | YES |
| `pdbp_neuropath_naccv11_methods_used_for_scoring_case_nphisox` | text | YES |
| `pdbp_neuropath_naccv11_alzheimer_s_disease_npthal` | double precision | YES |
| `pdbp_neuropath_naccv11_alzheimer_s_disease_npbraak` | bigint | YES |
| `pdbp_neuropath_naccv11_alzheimer_s_disease_npneur` | bigint | YES |
| `pdbp_neuropath_naccv11_alzheimer_s_disease_npadnc` | double precision | YES |
| `pdbp_neuropath_naccv11_alzheimer_s_disease_npdiff` | double precision | YES |
| `pdbp_neuropath_naccv11_alzheimer_s_disease_npamy` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_npinf` | bigint | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_npinf1a` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_npinf1b` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_npinf1d` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_npinf1f` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_npinf2a` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_npinf2b` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_npinf2d` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_npinf2f` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_npinf3a` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_npinf3b` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_npinf3d` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_npinf3f` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_npinf4a` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_npinf4b` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_npinf4d` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_npinf4f` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_nphemo` | bigint | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_nphemo1` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_nphemo2` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_nphemo3` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_npold` | bigint | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_npold1` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_npold2` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_npold3` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_npold4` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_npoldd` | bigint | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_npoldd1` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_npoldd2` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_npoldd3` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_npoldd4` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_nparter` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_npwmr` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_nppath` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_npnec` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_nppath2` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_nppath3` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_nppath4` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_nppath5` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_nppath6` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_nppath7` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_nppath8` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_nppath9` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_nppath10` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_nppath11` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_nppatho` | double precision | YES |
| `pdbp_neuropath_naccv11_cerebrovascular_disease_cvd_nppathox` | double precision | YES |
| `pdbp_neuropath_naccv11_lewy_body_pathology_nplbod` | bigint | YES |
| `pdbp_neuropath_naccv11_neuron_loss_in_the_substantia_nigra_n` | bigint | YES |
| `pdbp_neuropath_naccv11_hippocampal_sclerosis_nphipscl` | double precision | YES |
| `pdbp_neuropath_naccv11_distribution_of_tdp_43_nptdpa` | double precision | YES |
| `pdbp_neuropath_naccv11_distribution_of_tdp_43_nptdpb` | bigint | YES |
| `pdbp_neuropath_naccv11_distribution_of_tdp_43_nptdpc` | bigint | YES |
| `pdbp_neuropath_naccv11_distribution_of_tdp_43_nptdpd` | double precision | YES |
| `pdbp_neuropath_naccv11_distribution_of_tdp_43_nptdpe` | double precision | YES |
| `pdbp_neuropath_naccv11_frontotemporal_lobar_degeneration_and` | double precision | YES |
| `pdbp_neuropath_naccv11_frontotemporal_lobar_degeneration_and_1` | double precision | YES |
| `pdbp_neuropath_naccv11_frontotemporal_lobar_degeneration_and_2` | double precision | YES |
| `pdbp_neuropath_naccv11_frontotemporal_lobar_degeneration_and_3` | double precision | YES |
| `pdbp_neuropath_naccv11_frontotemporal_lobar_degeneration_and_4` | double precision | YES |
| `pdbp_neuropath_naccv11_frontotemporal_lobar_degeneration_and_5` | double precision | YES |
| `pdbp_neuropath_naccv11_frontotemporal_lobar_degeneration_and_6` | double precision | YES |
| `pdbp_neuropath_naccv11_frontotemporal_lobar_degeneration_and_7` | double precision | YES |
| `pdbp_neuropath_naccv11_frontotemporal_lobar_degeneration_and_8` | double precision | YES |
| `pdbp_neuropath_naccv11_frontotemporal_lobar_degeneration_and_9` | double precision | YES |
| `pdbp_neuropath_naccv11_frontotemporal_lobar_degeneration_and_10` | double precision | YES |
| `pdbp_neuropath_naccv11_frontotemporal_lobar_degeneration_and_11` | double precision | YES |
| `pdbp_neuropath_naccv11_frontotemporal_lobar_degeneration_and_12` | double precision | YES |
| `pdbp_neuropath_naccv11_frontotemporal_lobar_degeneration_and_13` | double precision | YES |
| `pdbp_neuropath_naccv11_frontotemporal_lobar_degeneration_and_14` | double precision | YES |
| `pdbp_neuropath_naccv11_frontotemporal_lobar_degeneration_and_15` | double precision | YES |
| `pdbp_neuropath_naccv11_frontotemporal_lobar_degeneration_and_16` | double precision | YES |
| `pdbp_neuropath_naccv11_frontotemporal_lobar_degeneration_and_17` | double precision | YES |
| `pdbp_neuropath_naccv11_frontotemporal_lobar_degeneration_and_18` | double precision | YES |
| `pdbp_neuropath_naccv11_aging_related_tau_astrogliopathy_arta` | double precision | YES |
| `pdbp_neuropath_naccv11_aging_related_tau_astrogliopathy_arta_1` | double precision | YES |
| `pdbp_neuropath_naccv11_aging_related_tau_astrogliopathy_arta_2` | double precision | YES |
| `pdbp_neuropath_naccv11_aging_related_tau_astrogliopathy_arta_3` | double precision | YES |
| `pdbp_neuropath_naccv11_aging_related_tau_astrogliopathy_arta_4` | double precision | YES |
| `pdbp_neuropath_naccv11_aging_related_tau_astrogliopathy_arta_5` | double precision | YES |
| `pdbp_neuropath_naccv11_aging_related_tau_astrogliopathy_arta_6` | double precision | YES |
| `pdbp_neuropath_naccv11_aging_related_tau_astrogliopathy_arta_7` | double precision | YES |
| `pdbp_neuropath_naccv11_aging_related_tau_astrogliopathy_arta_8` | double precision | YES |
| `pdbp_neuropath_naccv11_aging_related_tau_astrogliopathy_arta_9` | double precision | YES |
| `pdbp_neuropath_naccv11_aging_related_tau_astrogliopathy_arta_10` | double precision | YES |
| `pdbp_neuropath_naccv11_aging_related_tau_astrogliopathy_arta_11` | double precision | YES |
| `pdbp_neuropath_naccv11_aging_related_tau_astrogliopathy_arta_12` | double precision | YES |
| `pdbp_neuropath_naccv11_other_pathologic_diagnoses_nppdxa` | double precision | YES |
| `pdbp_neuropath_naccv11_other_pathologic_diagnoses_nppdxb` | double precision | YES |
| `pdbp_neuropath_naccv11_other_pathologic_diagnoses_nppdxc` | double precision | YES |
| `pdbp_neuropath_naccv11_other_pathologic_diagnoses_nppdxd` | double precision | YES |
| `pdbp_neuropath_naccv11_other_pathologic_diagnoses_nppdxe` | double precision | YES |
| `pdbp_neuropath_naccv11_other_pathologic_diagnoses_nppdxf` | double precision | YES |
| `pdbp_neuropath_naccv11_other_pathologic_diagnoses_nppdxg` | double precision | YES |
| `pdbp_neuropath_naccv11_other_pathologic_diagnoses_nppdxh` | double precision | YES |
| `pdbp_neuropath_naccv11_other_pathologic_diagnoses_nppdxi` | double precision | YES |
| `pdbp_neuropath_naccv11_other_pathologic_diagnoses_nppdxj` | double precision | YES |
| `pdbp_neuropath_naccv11_other_pathologic_diagnoses_nppdxk` | double precision | YES |
| `pdbp_neuropath_naccv11_other_pathologic_diagnoses_nppdxl` | double precision | YES |
| `pdbp_neuropath_naccv11_other_pathologic_diagnoses_nppdxm` | double precision | YES |
| `pdbp_neuropath_naccv11_other_pathologic_diagnoses_nppdxn` | double precision | YES |
| `pdbp_neuropath_naccv11_other_pathologic_diagnoses_nppdxo` | double precision | YES |
| `pdbp_neuropath_naccv11_other_pathologic_diagnoses_nppdxp` | double precision | YES |
| `pdbp_neuropath_naccv11_other_pathologic_diagnoses_nppdxq` | double precision | YES |
| `pdbp_neuropath_naccv11_other_pathologic_diagnoses_nppdxr` | double precision | YES |
| `pdbp_neuropath_naccv11_other_pathologic_diagnoses_nppdxrx` | double precision | YES |
| `pdbp_neuropath_naccv11_other_pathologic_diagnoses_nppdxs` | double precision | YES |
| `pdbp_neuropath_naccv11_other_pathologic_diagnoses_nppdxsx` | double precision | YES |
| `pdbp_neuropath_naccv11_other_pathologic_diagnoses_nppdxt` | double precision | YES |
| `pdbp_neuropath_naccv11_other_pathologic_diagnoses_nppdxtx` | double precision | YES |
| `pdbp_neuropath_naccv11_banked_biospecimens_npbnka` | double precision | YES |
| `pdbp_neuropath_naccv11_banked_biospecimens_npbnkb` | double precision | YES |
| `pdbp_neuropath_naccv11_banked_biospecimens_npbnkc` | double precision | YES |
| `pdbp_neuropath_naccv11_banked_biospecimens_npbnkd` | double precision | YES |
| `pdbp_neuropath_naccv11_banked_biospecimens_npbnke` | double precision | YES |
| `pdbp_neuropath_naccv11_banked_biospecimens_npbnkf` | double precision | YES |
| `pdbp_neuropath_naccv11_banked_biospecimens_npbnkg` | double precision | YES |
| `pdbp_neuropath_naccv11_banked_biospecimens_npfaut` | double precision | YES |
| `pdbp_neuropath_naccv11_banked_biospecimens_npfaut1` | double precision | YES |
| `pdbp_neuropath_naccv11_banked_biospecimens_npfaut2` | double precision | YES |
| `pdbp_neuropath_naccv11_banked_biospecimens_npfaut3` | double precision | YES |
| `pdbp_neuropath_naccv11_banked_biospecimens_npfaut4` | double precision | YES |

### `pdbp_raw.pdbp_npi` — 6 rows, 386 columns

| Column | Type | Nullable |
|---|---|---|
| `study_id` | bigint | YES |
| `dataset` | text | YES |
| `pdbp_npi_required_fields_sitename` | text | YES |
| `pdbp_npi_required_fields_visittyppdbp` | text | YES |
| `pdbp_npi_required_fields_visitdate` | double precision | YES |
| `pdbp_npi_required_fields_guid` | text | YES |
| `pdbp_npi_required_fields_associated_guid` | double precision | YES |
| `pdbp_npi_required_fields_ageyrs` | text | YES |
| `pdbp_npi_required_fields_ageremaindrmonths` | double precision | YES |
| `pdbp_npi_required_fields_ageval` | double precision | YES |
| `pdbp_npi_delusions_npiscreendelusind` | text | YES |
| `pdbp_npi_delusions_danger_npidangerind` | text | YES |
| `pdbp_npi_delusions_danger_npisymptomfrequencyrate` | double precision | YES |
| `pdbp_npi_delusions_danger_npisymptomsseverityscore` | double precision | YES |
| `pdbp_npi_delusions_danger_npicaregivemotdistressscore` | double precision | YES |
| `pdbp_npi_delusions_stealing_npistealingind` | text | YES |
| `pdbp_npi_delusions_stealing_npisymptomfrequencyrate` | double precision | YES |
| `pdbp_npi_delusions_stealing_npisymptomsseverityscore` | double precision | YES |
| `pdbp_npi_delusions_stealing_npicaregivemotdistressscore` | double precision | YES |
| `pdbp_npi_delusions_affair_npispouseaffairind` | text | YES |
| `pdbp_npi_delusions_affair_npisymptomfrequencyrate` | double precision | YES |
| `pdbp_npi_delusions_affair_npisymptomsseverityscore` | double precision | YES |
| `pdbp_npi_delusions_affair_npicaregivemotdistressscore` | double precision | YES |
| `pdbp_npi_delusions_guest_in_house_npiguestsind` | text | YES |
| `pdbp_npi_delusions_guest_in_house_npisymptomfrequencyrate` | double precision | YES |
| `pdbp_npi_delusions_guest_in_house_npisymptomsseverityscore` | double precision | YES |
| `pdbp_npi_delusions_guest_in_house_npicaregivemotdistressscor` | double precision | YES |
| `pdbp_npi_delusions_claiming_to_be_someone_else_npinotwhosayi` | text | YES |
| `pdbp_npi_delusions_claiming_to_be_someone_else_npisymptomfre` | double precision | YES |
| `pdbp_npi_delusions_claiming_to_be_someone_else_npisymptomsse` | double precision | YES |
| `pdbp_npi_delusions_claiming_to_be_someone_else_npicaregivemo` | double precision | YES |
| `pdbp_npi_delusions_not_in_their_house_npihousenothomeind` | text | YES |
| `pdbp_npi_delusions_not_in_their_house_npisymptomfrequencyrat` | double precision | YES |
| `pdbp_npi_delusions_not_in_their_house_npisymptomsseveritysco` | double precision | YES |
| `pdbp_npi_delusions_not_in_their_house_npicaregivemotdistress` | double precision | YES |
| `pdbp_npi_delusions_being_abandon_by_family_npifamilyabandoni` | text | YES |
| `pdbp_npi_delusions_being_abandon_by_family_npisymptomfrequen` | double precision | YES |
| `pdbp_npi_delusions_being_abandon_by_family_npisymptomsseveri` | double precision | YES |
| `pdbp_npi_delusions_being_abandon_by_family_npicaregivemotdis` | double precision | YES |
| `pdbp_npi_delusions_fictional_characters_npifictioncharacteri` | text | YES |
| `pdbp_npi_delusions_fictional_characters_npisymptomfrequencyr` | double precision | YES |
| `pdbp_npi_delusions_fictional_characters_npisymptomsseveritys` | double precision | YES |
| `pdbp_npi_delusions_fictional_characters_npicaregivemotdistre` | double precision | YES |
| `pdbp_npi_delusions_other_npiothrdelusionsind` | text | YES |
| `pdbp_npi_delusions_other_npisymptomfrequencyrate` | double precision | YES |
| `pdbp_npi_delusions_other_npisymptomsseverityscore` | double precision | YES |
| `pdbp_npi_delusions_other_npicaregivemotdistressscore` | double precision | YES |
| `pdbp_npi_hallucinations_npiscreenhallucinationsind` | text | YES |
| `pdbp_npi_hallucinations_hearing_voices_npihearsvoicesind` | text | YES |
| `pdbp_npi_hallucinations_hearing_voices_npisymptomfrequencyra` | double precision | YES |
| `pdbp_npi_hallucinations_hearing_voices_npisymptomsseveritysc` | double precision | YES |
| `pdbp_npi_hallucinations_hearing_voices_npicaregivemotdistres` | double precision | YES |
| `pdbp_npi_hallucinations_talks_to_imaginary_people_npitalksim` | text | YES |
| `pdbp_npi_hallucinations_talks_to_imaginary_people_npisymptom` | double precision | YES |
| `pdbp_npi_hallucinations_talks_to_imaginary_people_npisymptom_1` | double precision | YES |
| `pdbp_npi_hallucinations_talks_to_imaginary_people_npicaregiv` | double precision | YES |
| `pdbp_npi_hallucinations_sees_imaginary_objects_or_people_npi` | text | YES |
| `pdbp_npi_hallucinations_sees_imaginary_objects_or_people_npi_1` | double precision | YES |
| `pdbp_npi_hallucinations_sees_imaginary_objects_or_people_npi_2` | double precision | YES |
| `pdbp_npi_hallucinations_sees_imaginary_objects_or_people_npi_3` | double precision | YES |
| `pdbp_npi_hallucinations_smells_imaginary_orders_npismellsima` | text | YES |
| `pdbp_npi_hallucinations_smells_imaginary_orders_npisymptomfr` | double precision | YES |
| `pdbp_npi_hallucinations_smells_imaginary_orders_npisymptomss` | double precision | YES |
| `pdbp_npi_hallucinations_smells_imaginary_orders_npicaregivem` | double precision | YES |
| `pdbp_npi_hallucinations_touching_skin_npifeelsimaginaryind` | text | YES |
| `pdbp_npi_hallucinations_touching_skin_npisymptomfrequencyrat` | double precision | YES |
| `pdbp_npi_hallucinations_touching_skin_npisymptomsseveritysco` | double precision | YES |
| `pdbp_npi_hallucinations_touching_skin_npicaregivemotdistress` | double precision | YES |
| `pdbp_npi_hallucinations_imaginary_taste_npitastesimaginaryin` | text | YES |
| `pdbp_npi_hallucinations_imaginary_taste_npisymptomfrequencyr` | double precision | YES |
| `pdbp_npi_hallucinations_imaginary_taste_npisymptomsseveritys` | double precision | YES |
| `pdbp_npi_hallucinations_imaginary_taste_npicaregivemotdistre` | double precision | YES |
| `pdbp_npi_hallucinations_other_npiothrhallucinationsind` | text | YES |
| `pdbp_npi_hallucinations_other_npisymptomfrequencyrate` | double precision | YES |
| `pdbp_npi_hallucinations_other_npisymptomsseverityscore` | double precision | YES |
| `pdbp_npi_hallucinations_other_npicaregivemotdistressscore` | double precision | YES |
| `pdbp_npi_agitation_aggression_npiscreenaggressionind` | text | YES |
| `pdbp_npi_agitation_aggression_resist_activities_npiresistact` | text | YES |
| `pdbp_npi_agitation_aggression_resist_activities_npisymptomfr` | double precision | YES |
| `pdbp_npi_agitation_aggression_resist_activities_npisymptomss` | double precision | YES |
| `pdbp_npi_agitation_aggression_resist_activities_npicaregivem` | double precision | YES |
| `pdbp_npi_agitation_aggression_stubborn_npistubbornind` | text | YES |
| `pdbp_npi_agitation_aggression_stubborn_npisymptomfrequencyra` | double precision | YES |
| `pdbp_npi_agitation_aggression_stubborn_npisymptomsseveritysc` | double precision | YES |
| `pdbp_npi_agitation_aggression_stubborn_npicaregivemotdistres` | double precision | YES |
| `pdbp_npi_agitation_aggression_uncooperative_npiuncooperative` | text | YES |
| `pdbp_npi_agitation_aggression_uncooperative_npisymptomfreque` | double precision | YES |
| `pdbp_npi_agitation_aggression_uncooperative_npisymptomssever` | double precision | YES |
| `pdbp_npi_agitation_aggression_uncooperative_npicaregivemotdi` | double precision | YES |
| `pdbp_npi_agitation_aggression_hard_to_handle_npihardhandlein` | text | YES |
| `pdbp_npi_agitation_aggression_hard_to_handle_npisymptomfrequ` | double precision | YES |
| `pdbp_npi_agitation_aggression_hard_to_handle_npisymptomsseve` | double precision | YES |
| `pdbp_npi_agitation_aggression_hard_to_handle_npicaregivemotd` | double precision | YES |
| `pdbp_npi_agitation_aggression_angry_npicursesind` | text | YES |
| `pdbp_npi_agitation_aggression_angry_npisymptomfrequencyrate` | double precision | YES |
| `pdbp_npi_agitation_aggression_angry_npisymptomsseverityscore` | double precision | YES |
| `pdbp_npi_agitation_aggression_angry_npicaregivemotdistresssc` | double precision | YES |
| `pdbp_npi_agitation_aggression_other_behaviors_npislamdoorind` | text | YES |
| `pdbp_npi_agitation_aggression_other_behaviors_npisymptomfreq` | double precision | YES |
| `pdbp_npi_agitation_aggression_other_behaviors_npisymptomssev` | double precision | YES |
| `pdbp_npi_agitation_aggression_other_behaviors_npicaregivemot` | double precision | YES |
| `pdbp_npi_agitation_aggression_hurt_or_hit_others_npihurthito` | text | YES |
| `pdbp_npi_agitation_aggression_hurt_or_hit_others_npisymptomf` | double precision | YES |
| `pdbp_npi_agitation_aggression_hurt_or_hit_others_npisymptoms` | double precision | YES |
| `pdbp_npi_agitation_aggression_hurt_or_hit_others_npicaregive` | double precision | YES |
| `pdbp_npi_agitation_aggression_aggressive_or_agitated_behavio` | text | YES |
| `pdbp_npi_agitation_aggression_aggressive_or_agitated_behavio_1` | double precision | YES |
| `pdbp_npi_agitation_aggression_aggressive_or_agitated_behavio_2` | double precision | YES |
| `pdbp_npi_agitation_aggression_aggressive_or_agitated_behavio_3` | double precision | YES |
| `pdbp_npi_depression_dysphoria_npiscreendepressionind` | text | YES |
| `pdbp_npi_depression_dysphoria_tearful_npitearfulind` | text | YES |
| `pdbp_npi_depression_dysphoria_tearful_npisymptomfrequencyrat` | double precision | YES |
| `pdbp_npi_depression_dysphoria_tearful_npisymptomsseveritysco` | double precision | YES |
| `pdbp_npi_depression_dysphoria_tearful_npicaregivemotdistress` | double precision | YES |
| `pdbp_npi_depression_dysphoria_low_spirits_npisadind` | text | YES |
| `pdbp_npi_depression_dysphoria_low_spirits_npisymptomfrequenc` | double precision | YES |
| `pdbp_npi_depression_dysphoria_low_spirits_npisymptomsseverit` | double precision | YES |
| `pdbp_npi_depression_dysphoria_low_spirits_npicaregivemotdist` | double precision | YES |
| `pdbp_npi_depression_dysphoria_failure_npifailureind` | text | YES |
| `pdbp_npi_depression_dysphoria_failure_npisymptomfrequencyrat` | double precision | YES |
| `pdbp_npi_depression_dysphoria_failure_npisymptomsseveritysco` | double precision | YES |
| `pdbp_npi_depression_dysphoria_failure_npicaregivemotdistress` | double precision | YES |
| `pdbp_npi_depression_dysphoria_punished_npibadpersonind` | text | YES |
| `pdbp_npi_depression_dysphoria_punished_npisymptomfrequencyra` | double precision | YES |
| `pdbp_npi_depression_dysphoria_punished_npisymptomsseveritysc` | double precision | YES |
| `pdbp_npi_depression_dysphoria_punished_npicaregivemotdistres` | double precision | YES |
| `pdbp_npi_depression_dyphoria_discouraged_npidiscouragedind` | text | YES |
| `pdbp_npi_depression_dyphoria_discouraged_npisymptomfrequency` | double precision | YES |
| `pdbp_npi_depression_dyphoria_discouraged_npisymptomsseverity` | double precision | YES |
| `pdbp_npi_depression_dyphoria_discouraged_npicaregivemotdistr` | double precision | YES |
| `pdbp_npi_depression_dysphoria_burden_to_family_npiburdenfami` | text | YES |
| `pdbp_npi_depression_dysphoria_burden_to_family_npisymptomfre` | double precision | YES |
| `pdbp_npi_depression_dysphoria_burden_to_family_npisymptomsse` | double precision | YES |
| `pdbp_npi_depression_dysphoria_burden_to_family_npicaregivemo` | double precision | YES |
| `pdbp_npi_depression_dysphoria_suicide_npisuicideind` | text | YES |
| `pdbp_npi_depression_dysphoria_suicide_npisymptomfrequencyrat` | double precision | YES |
| `pdbp_npi_depression_dysphoria_suicide_npisymptomsseveritysco` | double precision | YES |
| `pdbp_npi_depression_dysphoria_suicide_npicaregivemotdistress` | double precision | YES |
| `pdbp_npi_depression_dysphoria_other_npiothrdepressionind` | text | YES |
| `pdbp_npi_depression_dysphoria_other_npisymptomfrequencyrate` | double precision | YES |
| `pdbp_npi_depression_dysphoria_other_npisymptomsseverityscore` | double precision | YES |
| `pdbp_npi_depression_dysphoria_other_npicaregivemotdistresssc` | double precision | YES |
| `pdbp_npi_anxiety_npiscreenanxietyind` | text | YES |
| `pdbp_npi_anxiety_worried_npiworryplaneventind` | text | YES |
| `pdbp_npi_anxiety_worried_npisymptomfrequencyrate` | double precision | YES |
| `pdbp_npi_anxiety_worried_npisymptomsseverityscore` | double precision | YES |
| `pdbp_npi_anxiety_worried_npicaregivemotdistressscore` | double precision | YES |
| `pdbp_npi_anxiety_shaky_npifeelshakyind` | text | YES |
| `pdbp_npi_anxiety_shaky_npisymptomfrequencyrate` | double precision | YES |
| `pdbp_npi_anxiety_shaky_npisymptomsseverityscore` | double precision | YES |
| `pdbp_npi_anxiety_shaky_npicaregivemotdistressscore` | double precision | YES |
| `pdbp_npi_anxiety_shortness_of_breath_npisighnervousind` | text | YES |
| `pdbp_npi_anxiety_shortness_of_breath_npisymptomfrequencyrate` | double precision | YES |
| `pdbp_npi_anxiety_shortness_of_breath_npisymptomsseverityscor` | double precision | YES |
| `pdbp_npi_anxiety_shortness_of_breath_npicaregivemotdistresss` | double precision | YES |
| `pdbp_npi_anxiety_other_signs_of_nervousness_npipoundheartind` | text | YES |
| `pdbp_npi_anxiety_other_signs_of_nervousness_npisymptomfreque` | double precision | YES |
| `pdbp_npi_anxiety_other_signs_of_nervousness_npisymptomssever` | double precision | YES |
| `pdbp_npi_anxiety_other_signs_of_nervousness_npicaregivemotdi` | double precision | YES |
| `pdbp_npi_anxiety_avoiding_certain_places_npiavoidplacesind` | text | YES |
| `pdbp_npi_anxiety_avoiding_certain_places_npisymptomfrequency` | double precision | YES |
| `pdbp_npi_anxiety_avoiding_certain_places_npisymptomsseverity` | double precision | YES |
| `pdbp_npi_anxiety_avoiding_certain_places_npicaregivemotdistr` | double precision | YES |
| `pdbp_npi_anxiety_separated_from_caregiver_or_spouse_npicling` | text | YES |
| `pdbp_npi_anxiety_separated_from_caregiver_or_spouse_npisympt` | double precision | YES |
| `pdbp_npi_anxiety_separated_from_caregiver_or_spouse_npisympt_1` | double precision | YES |
| `pdbp_npi_anxiety_separated_from_caregiver_or_spouse_npicareg` | double precision | YES |
| `pdbp_npi_anxiety_other_npiothranxietyind` | text | YES |
| `pdbp_npi_anxiety_other_npisymptomfrequencyrate` | double precision | YES |
| `pdbp_npi_anxiety_other_npisymptomsseverityscore` | double precision | YES |
| `pdbp_npi_anxiety_other_npicaregivemotdistressscore` | double precision | YES |
| `pdbp_npi_elation_euphoria_npiscreenelationind` | text | YES |
| `pdbp_npi_elation_euphoria_excess_happy_npiexcesshappyind` | text | YES |
| `pdbp_npi_elation_euphoria_excess_happy_npisymptomfrequencyra` | double precision | YES |
| `pdbp_npi_elation_euphoria_excess_happy_npisymptomsseveritysc` | double precision | YES |
| `pdbp_npi_elation_euphoria_excess_happy_npicaregivemotdistres` | double precision | YES |
| `pdbp_npi_elation_euphoria_not_funny_to_others_npifunnynotfun` | text | YES |
| `pdbp_npi_elation_euphoria_not_funny_to_others_npisymptomfreq` | double precision | YES |
| `pdbp_npi_elation_euphoria_not_funny_to_others_npisymptomssev` | double precision | YES |
| `pdbp_npi_elation_euphoria_not_funny_to_others_npicaregivemot` | double precision | YES |
| `pdbp_npi_elation_euphoria_childish_humor_npichildhumorind` | text | YES |
| `pdbp_npi_elation_euphoria_childish_humor_npisymptomfrequency` | double precision | YES |
| `pdbp_npi_elation_euphoria_childish_humor_npisymptomsseverity` | double precision | YES |
| `pdbp_npi_elation_euphoria_childish_humor_npicaregivemotdistr` | double precision | YES |
| `pdbp_npi_elation_euphoria_bad_jokes_or_remarks_npitellsbadjo` | text | YES |
| `pdbp_npi_elation_euphoria_bad_jokes_or_remarks_npisymptomfre` | double precision | YES |
| `pdbp_npi_elation_euphoria_bad_jokes_or_remarks_npisymptomsse` | double precision | YES |
| `pdbp_npi_elation_euphoria_bad_jokes_or_remarks_npicaregivemo` | double precision | YES |
| `pdbp_npi_elation_euphoria_childish_pranks_npichildpranksind` | text | YES |
| `pdbp_npi_elation_euphoria_childish_pranks_npisymptomfrequenc` | double precision | YES |
| `pdbp_npi_elation_euphoria_childish_pranks_npisymptomsseverit` | double precision | YES |
| `pdbp_npi_elation_euphoria_childish_pranks_npicaregivemotdist` | double precision | YES |
| `pdbp_npi_elation_euphoria_wealth_npitalkbigind` | text | YES |
| `pdbp_npi_elation_euphoria_wealth_npisymptomfrequencyrate` | double precision | YES |
| `pdbp_npi_elation_euphoria_wealth_npisymptomsseverityscore` | double precision | YES |
| `pdbp_npi_elation_euphoria_wealth_npicaregivemotdistressscore` | double precision | YES |
| `pdbp_npi_elation_euphoria_other_npiothrelationind` | text | YES |
| `pdbp_npi_elation_euphoria_other_npisymptomfrequencyrate` | double precision | YES |
| `pdbp_npi_elation_euphoria_other_npisymptomsseverityscore` | double precision | YES |
| `pdbp_npi_elation_euphoria_other_npicaregivemotdistressscore` | double precision | YES |
| `pdbp_npi_apathy_indifference_npiscreenapathyind` | text | YES |
| `pdbp_npi_apathy_indifference_less_active_npilessspontaneousi` | text | YES |
| `pdbp_npi_apathy_indifference_less_active_npisymptomfrequency` | double precision | YES |
| `pdbp_npi_apathy_indifference_less_active_npisymptomsseverity` | double precision | YES |
| `pdbp_npi_apathy_indifference_less_active_npicaregivemotdistr` | double precision | YES |
| `pdbp_npi_apathy_indifference_conversation_npiinitiateconvoin` | text | YES |
| `pdbp_npi_apathy_indifference_conversation_npisymptomfrequenc` | double precision | YES |
| `pdbp_npi_apathy_indifference_conversation_npisymptomsseverit` | double precision | YES |
| `pdbp_npi_apathy_indifference_conversation_npicaregivemotdist` | double precision | YES |
| `pdbp_npi_apathy_indifference_lacking_in_emotions_npilessaffe` | text | YES |
| `pdbp_npi_apathy_indifference_lacking_in_emotions_npisymptomf` | double precision | YES |
| `pdbp_npi_apathy_indifference_lacking_in_emotions_npisymptoms` | double precision | YES |
| `pdbp_npi_apathy_indifference_lacking_in_emotions_npicaregive` | double precision | YES |
| `pdbp_npi_apathy_indifference_household_chores_npichoresind` | text | YES |
| `pdbp_npi_apathy_indifference_household_chores_npisymptomfreq` | double precision | YES |
| `pdbp_npi_apathy_indifference_household_chores_npisymptomssev` | double precision | YES |
| `pdbp_npi_apathy_indifference_household_chores_npicaregivemot` | double precision | YES |
| `pdbp_npi_apathy_indifference_interested_in_activities_npidis` | text | YES |
| `pdbp_npi_apathy_indifference_interested_in_activities_npisym` | double precision | YES |
| `pdbp_npi_apathy_indifference_interested_in_activities_npisym_1` | double precision | YES |
| `pdbp_npi_apathy_indifference_interested_in_activities_npicar` | double precision | YES |
| `pdbp_npi_apathy_indifference_interest_in_friends_or_family_n` | text | YES |
| `pdbp_npi_apathy_indifference_interest_in_friends_or_family_n_1` | double precision | YES |
| `pdbp_npi_apathy_indifference_interest_in_friends_or_family_n_2` | double precision | YES |
| `pdbp_npi_apathy_indifference_interest_in_friends_or_family_n_3` | double precision | YES |
| `pdbp_npi_apathy_indifference_own_interests_npilessenthusiasm` | text | YES |
| `pdbp_npi_apathy_indifference_own_interests_npisymptomfrequen` | double precision | YES |
| `pdbp_npi_apathy_indifference_own_interests_npisymptomsseveri` | double precision | YES |
| `pdbp_npi_apathy_indifference_own_interests_npicaregivemotdis` | double precision | YES |
| `pdbp_npi_apathy_indifference_other_npiothrapathyind` | text | YES |
| `pdbp_npi_apathy_indifference_other_npisymptomfrequencyrate` | double precision | YES |
| `pdbp_npi_apathy_indifference_other_npisymptomsseverityscore` | double precision | YES |
| `pdbp_npi_apathy_indifference_other_npicaregivemotdistresssco` | double precision | YES |
| `pdbp_npi_disinhibition_npiscreendisinhibitionind` | text | YES |
| `pdbp_npi_disinhibition_acting_impulsively_npiactimpulsiveind` | text | YES |
| `pdbp_npi_disinhibition_acting_impulsively_npisymptomfrequenc` | double precision | YES |
| `pdbp_npi_disinhibition_acting_impulsively_npisymptomsseverit` | double precision | YES |
| `pdbp_npi_disinhibition_acting_impulsively_npicaregivemotdist` | double precision | YES |
| `pdbp_npi_disinhibition_talking_to_strangers_npitalkstrangers` | text | YES |
| `pdbp_npi_disinhibition_talking_to_strangers_npisymptomfreque` | double precision | YES |
| `pdbp_npi_disinhibition_talking_to_strangers_npisymptomssever` | double precision | YES |
| `pdbp_npi_disinhibition_talking_to_strangers_npicaregivemotdi` | double precision | YES |
| `pdbp_npi_disinhibition_insensitive_npiinsensitiveremarksind` | text | YES |
| `pdbp_npi_disinhibition_insensitive_npisymptomfrequencyrate` | double precision | YES |
| `pdbp_npi_disinhibition_insensitive_npisymptomsseverityscore` | double precision | YES |
| `pdbp_npi_disinhibition_insensitive_npicaregivemotdistresssco` | double precision | YES |
| `pdbp_npi_disinhibition_sexual_remarks_npisexualremarksind` | text | YES |
| `pdbp_npi_disinhibition_sexual_remarks_npisymptomfrequencyrat` | double precision | YES |
| `pdbp_npi_disinhibition_sexual_remarks_npisymptomsseveritysco` | double precision | YES |
| `pdbp_npi_disinhibition_sexual_remarks_npicaregivemotdistress` | double precision | YES |
| `pdbp_npi_disinhibition_personal_matters_npipersonalpublicind` | text | YES |
| `pdbp_npi_disinhibition_personal_matters_npisymptomfrequencyr` | double precision | YES |
| `pdbp_npi_disinhibition_personal_matters_npisymptomsseveritys` | double precision | YES |
| `pdbp_npi_disinhibition_personal_matters_npicaregivemotdistre` | double precision | YES |
| `pdbp_npi_disinhibition_touching_others_npitouchhugind` | text | YES |
| `pdbp_npi_disinhibition_touching_others_npisymptomfrequencyra` | double precision | YES |
| `pdbp_npi_disinhibition_touching_others_npisymptomsseveritysc` | double precision | YES |
| `pdbp_npi_disinhibition_touching_others_npicaregivemotdistres` | double precision | YES |
| `pdbp_npi_disinhibition_other_npiothrdisinhibitionind` | text | YES |
| `pdbp_npi_disinhibition_other_npisymptomfrequencyrate` | double precision | YES |
| `pdbp_npi_disinhibition_other_npisymptomsseverityscore` | double precision | YES |
| `pdbp_npi_disinhibition_other_npicaregivemotdistressscore` | double precision | YES |
| `pdbp_npi_irritability_lability_npiscreenirritabilityind` | text | YES |
| `pdbp_npi_irritability_lability_bad_temper_npibadtemperind` | text | YES |
| `pdbp_npi_irritability_lability_bad_temper_npisymptomfrequenc` | double precision | YES |
| `pdbp_npi_irritability_lability_bad_temper_npisymptomsseverit` | double precision | YES |
| `pdbp_npi_irritability_lability_bad_temper_npicaregivemotdist` | double precision | YES |
| `pdbp_npi_irritability_lability_mood_changes_npirapidmoodshif` | text | YES |
| `pdbp_npi_irritability_lability_mood_changes_npisymptomfreque` | double precision | YES |
| `pdbp_npi_irritability_lability_mood_changes_npisymptomssever` | double precision | YES |
| `pdbp_npi_irritability_lability_mood_changes_npicaregivemotdi` | double precision | YES |
| `pdbp_npi_irritability_lability_flashes_of_anger_npiflashange` | text | YES |
| `pdbp_npi_irritability_lability_flashes_of_anger_npisymptomfr` | double precision | YES |
| `pdbp_npi_irritability_lability_flashes_of_anger_npisymptomss` | double precision | YES |
| `pdbp_npi_irritability_lability_flashes_of_anger_npicaregivem` | double precision | YES |
| `pdbp_npi_irritability_lability_impatient_npiimpatientind` | text | YES |
| `pdbp_npi_irritability_lability_impatient_npisymptomfrequency` | double precision | YES |
| `pdbp_npi_irritability_lability_impatient_npisymptomsseverity` | double precision | YES |
| `pdbp_npi_irritability_lability_impatient_npicaregivemotdistr` | double precision | YES |
| `pdbp_npi_irritability_lability_irritable_npicrankyind` | text | YES |
| `pdbp_npi_irritability_lability_irritable_npisymptomfrequency` | double precision | YES |
| `pdbp_npi_irritability_lability_irritable_npisymptomsseverity` | double precision | YES |
| `pdbp_npi_irritability_lability_irritable_npicaregivemotdistr` | double precision | YES |
| `pdbp_npi_irritability_lability_arguementative_npiargumentati` | text | YES |
| `pdbp_npi_irritability_lability_arguementative_npisymptomfreq` | double precision | YES |
| `pdbp_npi_irritability_lability_arguementative_npisymptomssev` | double precision | YES |
| `pdbp_npi_irritability_lability_arguementative_npicaregivemot` | double precision | YES |
| `pdbp_npi_irritability_lability_other_npiothrirritabilityind` | text | YES |
| `pdbp_npi_irritability_lability_other_npisymptomfrequencyrate` | double precision | YES |
| `pdbp_npi_irritability_lability_other_npisymptomsseverityscor` | double precision | YES |
| `pdbp_npi_irritability_lability_other_npicaregivemotdistresss` | double precision | YES |
| `pdbp_npi_aberrant_motor_behavior_npiscreenmotorbehaviorind` | text | YES |
| `pdbp_npi_aberrant_motor_behavior_pace_around_the_house_npipa` | text | YES |
| `pdbp_npi_aberrant_motor_behavior_pace_around_the_house_npisy` | double precision | YES |
| `pdbp_npi_aberrant_motor_behavior_pace_around_the_house_npisy_1` | double precision | YES |
| `pdbp_npi_aberrant_motor_behavior_pace_around_the_house_npica` | double precision | YES |
| `pdbp_npi_aberrant_motor_behavior_rummaging_npirummagesind` | text | YES |
| `pdbp_npi_aberrant_motor_behavior_rummaging_npisymptomfrequen` | double precision | YES |
| `pdbp_npi_aberrant_motor_behavior_rummaging_npisymptomsseveri` | double precision | YES |
| `pdbp_npi_aberrant_motor_behavior_rummaging_npicaregivemotdis` | double precision | YES |
| `pdbp_npi_aberrant_motor_behavior_clothing_npiclothesoffonind` | text | YES |
| `pdbp_npi_aberrant_motor_behavior_clothing_npisymptomfrequenc` | double precision | YES |
| `pdbp_npi_aberrant_motor_behavior_clothing_npisymptomsseverit` | double precision | YES |
| `pdbp_npi_aberrant_motor_behavior_clothing_npicaregivemotdist` | double precision | YES |
| `pdbp_npi_aberrant_motor_behavior_repetitive_activities_npire` | text | YES |
| `pdbp_npi_aberrant_motor_behavior_repetitive_activities_npisy` | double precision | YES |
| `pdbp_npi_aberrant_motor_behavior_repetitive_activities_npisy_1` | double precision | YES |
| `pdbp_npi_aberrant_motor_behavior_repetitive_activities_npica` | double precision | YES |
| `pdbp_npi_aberrant_motor_behavior_engage_in_activities_npirep` | text | YES |
| `pdbp_npi_aberrant_motor_behavior_engage_in_activities_npisym` | double precision | YES |
| `pdbp_npi_aberrant_motor_behavior_engage_in_activities_npisym_1` | double precision | YES |
| `pdbp_npi_aberrant_motor_behavior_engage_in_activities_npicar` | double precision | YES |
| `pdbp_npi_aberrant_motor_behavior_fidget_excessively_npifidge` | text | YES |
| `pdbp_npi_aberrant_motor_behavior_fidget_excessively_npisympt` | double precision | YES |
| `pdbp_npi_aberrant_motor_behavior_fidget_excessively_npisympt_1` | double precision | YES |
| `pdbp_npi_aberrant_motor_behavior_fidget_excessively_npicareg` | double precision | YES |
| `pdbp_npi_aberrant_motor_behavior_other_npiothrmotorbehaviori` | text | YES |
| `pdbp_npi_aberrant_motor_behavior_other_npisymptomfrequencyra` | double precision | YES |
| `pdbp_npi_aberrant_motor_behavior_other_npisymptomsseveritysc` | double precision | YES |
| `pdbp_npi_aberrant_motor_behavior_other_npicaregivemotdistres` | double precision | YES |
| `pdbp_npi_sleep_npiscreennightbehaviorind` | text | YES |
| `pdbp_npi_sleep_difficulty_falling_alseep_npihardfallasleepin` | text | YES |
| `pdbp_npi_sleep_difficulty_falling_alseep_npisymptomfrequency` | double precision | YES |
| `pdbp_npi_sleep_difficulty_falling_alseep_npisymptomsseverity` | double precision | YES |
| `pdbp_npi_sleep_difficulty_falling_alseep_npicaregivemotdistr` | double precision | YES |
| `pdbp_npi_sleep_get_up_during_the_night_npigetupnightind` | text | YES |
| `pdbp_npi_sleep_get_up_during_the_night_npisymptomfrequencyra` | double precision | YES |
| `pdbp_npi_sleep_get_up_during_the_night_npisymptomsseveritysc` | double precision | YES |
| `pdbp_npi_sleep_get_up_during_the_night_npicaregivemotdistres` | double precision | YES |
| `pdbp_npi_sleep_involved_in_appropriate_activities_npiinappro` | text | YES |
| `pdbp_npi_sleep_involved_in_appropriate_activities_npisymptom` | double precision | YES |
| `pdbp_npi_sleep_involved_in_appropriate_activities_npisymptom_1` | double precision | YES |
| `pdbp_npi_sleep_involved_in_appropriate_activities_npicaregiv` | double precision | YES |
| `pdbp_npi_sleep_wakeup_others_at_night_npiawakeguardianind` | text | YES |
| `pdbp_npi_sleep_wakeup_others_at_night_npisymptomfrequencyrat` | double precision | YES |
| `pdbp_npi_sleep_wakeup_others_at_night_npisymptomsseveritysco` | double precision | YES |
| `pdbp_npi_sleep_wakeup_others_at_night_npicaregivemotdistress` | double precision | YES |
| `pdbp_npi_sleep_awake_at_night_npiwakenightthinkdayind` | text | YES |
| `pdbp_npi_sleep_awake_at_night_npisymptomfrequencyrate` | double precision | YES |
| `pdbp_npi_sleep_awake_at_night_npisymptomsseverityscore` | double precision | YES |
| `pdbp_npi_sleep_awake_at_night_npicaregivemotdistressscore` | double precision | YES |
| `pdbp_npi_sleep_awake_early_npiawakeearlyind` | text | YES |
| `pdbp_npi_sleep_awake_early_npisymptomfrequencyrate` | double precision | YES |
| `pdbp_npi_sleep_awake_early_npisymptomsseverityscore` | double precision | YES |
| `pdbp_npi_sleep_awake_early_npicaregivemotdistressscore` | double precision | YES |
| `pdbp_npi_sleep_sleeing_during_the_day_npisleepdayind` | text | YES |
| `pdbp_npi_sleep_sleeing_during_the_day_npisymptomfrequencyrat` | double precision | YES |
| `pdbp_npi_sleep_sleeing_during_the_day_npisymptomsseveritysco` | double precision | YES |
| `pdbp_npi_sleep_sleeing_during_the_day_npicaregivemotdistress` | double precision | YES |
| `pdbp_npi_sleep_other_nightime_behavior_npiothrnightbehaviori` | text | YES |
| `pdbp_npi_sleep_other_nightime_behavior_npisymptomfrequencyra` | double precision | YES |
| `pdbp_npi_sleep_other_nightime_behavior_npisymptomsseveritysc` | double precision | YES |
| `pdbp_npi_sleep_other_nightime_behavior_npicaregivemotdistres` | double precision | YES |
| `pdbp_npi_appetite_and_eating_disorders_npiscreenappetiteind` | text | YES |
| `pdbp_npi_appetite_and_eating_disorders_loss_of_appetite_npia` | text | YES |
| `pdbp_npi_appetite_and_eating_disorders_loss_of_appetite_npis` | double precision | YES |
| `pdbp_npi_appetite_and_eating_disorders_loss_of_appetite_npis_1` | double precision | YES |
| `pdbp_npi_appetite_and_eating_disorders_loss_of_appetite_npic` | double precision | YES |
| `pdbp_npi_appetite_and_eating_disorders_increase_in_appetite_` | text | YES |
| `pdbp_npi_appetite_and_eating_disorders_increase_in_appetite__1` | double precision | YES |
| `pdbp_npi_appetite_and_eating_disorders_increase_in_appetite__2` | double precision | YES |
| `pdbp_npi_appetite_and_eating_disorders_increase_in_appetite__3` | double precision | YES |
| `pdbp_npi_appetite_and_eating_disorders_loss_of_weight_npiwgt` | text | YES |
| `pdbp_npi_appetite_and_eating_disorders_loss_of_weight_npisym` | double precision | YES |
| `pdbp_npi_appetite_and_eating_disorders_loss_of_weight_npisym_1` | double precision | YES |
| `pdbp_npi_appetite_and_eating_disorders_loss_of_weight_npicar` | double precision | YES |
| `pdbp_npi_appetite_and_eating_disorders_weight_gain_npiwgtgai` | text | YES |
| `pdbp_npi_appetite_and_eating_disorders_weight_gain_npisympto` | double precision | YES |
| `pdbp_npi_appetite_and_eating_disorders_weight_gain_npisympto_1` | double precision | YES |
| `pdbp_npi_appetite_and_eating_disorders_weight_gain_npicaregi` | double precision | YES |
| `pdbp_npi_appetite_and_eating_disorders_eating_behaviors_npic` | text | YES |
| `pdbp_npi_appetite_and_eating_disorders_eating_behaviors_npis` | double precision | YES |
| `pdbp_npi_appetite_and_eating_disorders_eating_behaviors_npis_1` | double precision | YES |
| `pdbp_npi_appetite_and_eating_disorders_eating_behaviors_npic_1` | double precision | YES |
| `pdbp_npi_appetite_and_eating_behaviors_kind_of_food_npichang` | text | YES |
| `pdbp_npi_appetite_and_eating_behaviors_kind_of_food_npisympt` | double precision | YES |
| `pdbp_npi_appetite_and_eating_behaviors_kind_of_food_npisympt_1` | double precision | YES |
| `pdbp_npi_appetite_and_eating_behaviors_kind_of_food_npicareg` | double precision | YES |
| `pdbp_npi_appetite_and_eating_disorders_same_foods_npiunusual` | text | YES |
| `pdbp_npi_appetite_and_eating_disorders_same_foods_npisymptom` | double precision | YES |
| `pdbp_npi_appetite_and_eating_disorders_same_foods_npisymptom_1` | double precision | YES |
| `pdbp_npi_appetite_and_eating_disorders_same_foods_npicaregiv` | double precision | YES |
| `pdbp_npi_appetite_and_eating_disorders_other_npiothrappetite` | text | YES |
| `pdbp_npi_appetite_and_eating_disorders_other_npisymptomfrequ` | double precision | YES |
| `pdbp_npi_appetite_and_eating_disorders_other_npisymptomsseve` | double precision | YES |
| `pdbp_npi_appetite_and_eating_disorders_other_npicaregivemotd` | double precision | YES |

### `pdbp_raw.pdbp_sat1` — 617 rows, 13 columns

| Column | Type | Nullable |
|---|---|---|
| `study_id` | bigint | YES |
| `dataset` | text | YES |
| `pdbp_sat1_required_sitename` | text | YES |
| `pdbp_sat1_required_visittyppdbp` | text | YES |
| `pdbp_sat1_required_visitdate` | double precision | YES |
| `pdbp_sat1_required_guid` | text | YES |
| `pdbp_sat1_required_associated_guid` | double precision | YES |
| `pdbp_sat1_required_ageyrs` | text | YES |
| `pdbp_sat1_required_ageremaindrmonths` | double precision | YES |
| `pdbp_sat1_required_ageval` | double precision | YES |
| `pdbp_sat1_word_task_speededattnwordrawscore` | bigint | YES |
| `pdbp_sat1_color_task_speededattncolorrawscore` | bigint | YES |
| `pdbp_sat1_color_word_task_speededattncolorwordrawscore` | bigint | YES |

### `pdbp_raw.pdbp_special_attributes` — 182 rows, 20 columns

| Column | Type | Nullable |
|---|---|---|
| `study_id` | bigint | YES |
| `dataset` | text | YES |
| `pdbp_special_attributes_main_group_sitename` | text | YES |
| `pdbp_special_attributes_main_group_visittyppdbp` | text | YES |
| `pdbp_special_attributes_main_group_visitdate` | double precision | YES |
| `pdbp_special_attributes_main_group_guid` | text | YES |
| `pdbp_special_attributes_main_group_associated_guid` | double precision | YES |
| `pdbp_special_attributes_main_group_ageyrs` | text | YES |
| `pdbp_special_attributes_main_group_ageremaindrmonths` | double precision | YES |
| `pdbp_special_attributes_main_group_ageval` | double precision | YES |
| `pdbp_special_attributes_special_attributes_pdbpgene_mutations_i` | double precision | YES |
| `pdbp_special_attributes_special_attributes_pdbp_hyposmiascore` | text | YES |
| `pdbp_special_attributes_special_attributes_rem_behavior_disorde` | text | YES |
| `pdbp_special_attributes_special_attributes_pdbp_datscan` | text | YES |
| `pdbp_special_attributes_deep_brain_stimulation_pdbp_dbs_imp` | text | YES |
| `pdbp_special_attributes_deep_brain_stimulation_pdbp_dbs_loc` | double precision | YES |
| `pdbp_special_attributes_deep_brain_stimulation_pdbp_dbs_yr` | double precision | YES |
| `pdbp_special_attributes_deep_brain_stimulation_pdbp_dbs_mth` | double precision | YES |
| `pdbp_special_attributes_deep_brain_stimulation_pdbp_dbs_day` | double precision | YES |
| `pdbp_special_attributes_deep_brain_stimulation_pdbp_dbs_model` | double precision | YES |

### `pdbp_raw.pdbpbloodcollection` — 680 rows, 38 columns

| Column | Type | Nullable |
|---|---|---|
| `study_id` | bigint | YES |
| `dataset` | text | YES |
| `pdbpbloodcollection_required_fields_sitename` | text | YES |
| `pdbpbloodcollection_required_fields_visittyppdbp` | text | YES |
| `pdbpbloodcollection_required_fields_visitdate` | double precision | YES |
| `pdbpbloodcollection_required_fields_guid` | text | YES |
| `pdbpbloodcollection_required_fields_associated_guid` | double precision | YES |
| `pdbpbloodcollection_required_fields_ageyrs` | text | YES |
| `pdbpbloodcollection_required_fields_ageremaindrmonths` | double precision | YES |
| `pdbpbloodcollection_required_fields_ageval` | double precision | YES |
| `pdbpbloodcollection_date_sample_was_shipped_sampleshippeddateti` | text | YES |
| `pdbpbloodcollection_dna_sample_bloodcollectind` | text | YES |
| `pdbpbloodcollection_dna_sample_bloodcollectdatetime` | text | YES |
| `pdbpbloodcollection_dna_sample_bloodcollectvol` | double precision | YES |
| `pdbpbloodcollection_dna_sample_sampleshippeddatetime` | text | YES |
| `pdbpbloodcollection_whole_blood_sample_bloodcollectind` | text | YES |
| `pdbpbloodcollection_whole_blood_sample_bloodcollectdatetime` | text | YES |
| `pdbpbloodcollection_rna_sample_bloodcollectind` | text | YES |
| `pdbpbloodcollection_rna_sample_bloodcollectdatetime` | text | YES |
| `pdbpbloodcollection_rna_sample_sampplacedfreezerdatetime` | text | YES |
| `pdbpbloodcollection_rna_sample_tempmeasrfreezerpdbp` | double precision | YES |
| `pdbpbloodcollection_plasma_sample_bloodcollectind` | text | YES |
| `pdbpbloodcollection_plasma_sample_bloodcollectdatetime` | text | YES |
| `pdbpbloodcollection_plasma_sample_bloodcollcentrifdatetime` | text | YES |
| `pdbpbloodcollection_plasma_sample_centrifgrate` | double precision | YES |
| `pdbpbloodcollection_plasma_sample_bloodcollectcentrifugationdur` | double precision | YES |
| `pdbpbloodcollection_plasma_sample_samplecentriftempval` | double precision | YES |
| `pdbpbloodcollection_plasma_sample_sampplacedfreezerdatetime` | text | YES |
| `pdbpbloodcollection_plasma_sample_tempmeasrfreezerpdbp` | double precision | YES |
| `pdbpbloodcollection_serum_sample_bloodcollectind` | text | YES |
| `pdbpbloodcollection_serum_sample_bloodcollectdatetime` | text | YES |
| `pdbpbloodcollection_serum_sample_bloodcollcentrifdatetime` | text | YES |
| `pdbpbloodcollection_serum_sample_centrifgrate` | double precision | YES |
| `pdbpbloodcollection_serum_sample_bloodcollectcentrifugationdur` | double precision | YES |
| `pdbpbloodcollection_serum_sample_samplecentriftempval` | double precision | YES |
| `pdbpbloodcollection_serum_sample_sampplacedfreezerdatetime` | text | YES |
| `pdbpbloodcollection_serum_sample_tempmeasrfreezerpdbp` | double precision | YES |
| `pdbpbloodcollection_csf_sample_csfcollectedind` | text | YES |

### `pdbp_raw.pdbpchangediagnosis` — 236 rows, 18 columns

| Column | Type | Nullable |
|---|---|---|
| `study_id` | bigint | YES |
| `dataset` | text | YES |
| `pdbpchangediagnosis_required_sitename` | text | YES |
| `pdbpchangediagnosis_required_visittyppdbp` | text | YES |
| `pdbpchangediagnosis_required_visitdate` | double precision | YES |
| `pdbpchangediagnosis_required_guid` | text | YES |
| `pdbpchangediagnosis_required_associated_guid` | double precision | YES |
| `pdbpchangediagnosis_required_ageyrs` | text | YES |
| `pdbpchangediagnosis_required_ageremaindrmonths` | double precision | YES |
| `pdbpchangediagnosis_required_ageval` | double precision | YES |
| `pdbpchangediagnosis_diagnosis_diagnoschangeind` | text | YES |
| `pdbpchangediagnosis_diagnosis_diagnosprimarychangedate` | text | YES |
| `pdbpchangediagnosis_diagnosis_diagnosinitialpdbptyp` | text | YES |
| `pdbpchangediagnosis_diagnosis_diagnosinitialpdbptypoth` | double precision | YES |
| `pdbpchangediagnosis_diagnosis_diagnosmostrecentpdbptyp` | text | YES |
| `pdbpchangediagnosis_diagnosis_diagnosmostrecentpdbptypoth` | text | YES |
| `pdbpchangediagnosis_diagnosis_diagnoschngrsntyp` | text | YES |
| `pdbpchangediagnosis_diagnosis_diagnoschngrsntypoth` | double precision | YES |

### `pdbp_raw.pdbplbd_inclusnxclusn` — 250 rows, 14 columns

| Column | Type | Nullable |
|---|---|---|
| `study_id` | bigint | YES |
| `dataset` | text | YES |
| `pdbplbd_inclusnxclusn_required_sitename` | text | YES |
| `pdbplbd_inclusnxclusn_required_visittyppdbp` | text | YES |
| `pdbplbd_inclusnxclusn_required_visitdate` | double precision | YES |
| `pdbplbd_inclusnxclusn_required_guid` | text | YES |
| `pdbplbd_inclusnxclusn_required_associated_guid` | double precision | YES |
| `pdbplbd_inclusnxclusn_required_ageyrs` | text | YES |
| `pdbplbd_inclusnxclusn_required_ageremaindrmonths` | double precision | YES |
| `pdbplbd_inclusnxclusn_required_ageval` | double precision | YES |
| `pdbplbd_inclusnxclusn_control_subject_subjectcntrlind` | text | YES |
| `pdbplbd_inclusnxclusn_control_subject_pdbplbdinclusncritcntrlsu` | text | YES |
| `pdbplbd_inclusnxclusn_case_subject_subjectcaseind` | text | YES |
| `pdbplbd_inclusnxclusn_case_subject_pdbplbdinclusncritcasesubjty` | text | YES |

### `pdbp_raw.pdbpnoisepareidoliatask` — 694 rows, 35 columns

| Column | Type | Nullable |
|---|---|---|
| `study_id` | bigint | YES |
| `dataset` | text | YES |
| `pdbpnoisepareidoliatask_required_fields_sitename` | text | YES |
| `pdbpnoisepareidoliatask_required_fields_visittyppdbp` | text | YES |
| `pdbpnoisepareidoliatask_required_fields_visitdate` | double precision | YES |
| `pdbpnoisepareidoliatask_required_fields_guid` | text | YES |
| `pdbpnoisepareidoliatask_required_fields_associated_guid` | double precision | YES |
| `pdbpnoisepareidoliatask_required_fields_ageyrs` | text | YES |
| `pdbpnoisepareidoliatask_required_fields_ageremaindrmonths` | double precision | YES |
| `pdbpnoisepareidoliatask_required_fields_ageval` | double precision | YES |
| `pdbpnoisepareidoliatask_noise_pareidolia_task_performance_in` | text | YES |
| `pdbpnoisepareidoliatask_subject_response_1_10_nptstimulusfac` | text | YES |
| `pdbpnoisepareidoliatask_subject_response_1_10_nptstimulusnoi` | text | YES |
| `pdbpnoisepareidoliatask_subject_response_1_10_nptstimulusnoi_1` | text | YES |
| `pdbpnoisepareidoliatask_subject_response_1_10_nptstimulusfac_1` | text | YES |
| `pdbpnoisepareidoliatask_subject_response_1_10_nptstimulusnoi_2` | text | YES |
| `pdbpnoisepareidoliatask_subject_response_1_10_nptstimulusnoi_3` | text | YES |
| `pdbpnoisepareidoliatask_subject_response_1_10_nptstimulusnoi_4` | text | YES |
| `pdbpnoisepareidoliatask_subject_response_1_10_nptstimulusfac_2` | text | YES |
| `pdbpnoisepareidoliatask_subject_response_1_10_nptstimulusnoi_5` | text | YES |
| `pdbpnoisepareidoliatask_subject_response_1_10_nptstimulusfac_3` | text | YES |
| `pdbpnoisepareidoliatask_subject_response_11_20_nptstimulusno` | text | YES |
| `pdbpnoisepareidoliatask_subject_response_11_20_nptstimulusno_1` | text | YES |
| `pdbpnoisepareidoliatask_subject_response_11_20_nptstimulusfa` | text | YES |
| `pdbpnoisepareidoliatask_subject_response_11_20_nptstimulusno_2` | text | YES |
| `pdbpnoisepareidoliatask_subject_response_11_20_nptstimulusno_3` | text | YES |
| `pdbpnoisepareidoliatask_subject_response_11_20_nptstimulusfa_1` | text | YES |
| `pdbpnoisepareidoliatask_subject_response_11_20_nptstimulusno_4` | text | YES |
| `pdbpnoisepareidoliatask_subject_response_11_20_nptstimulusfa_2` | text | YES |
| `pdbpnoisepareidoliatask_subject_response_11_20_nptstimulusno_5` | text | YES |
| `pdbpnoisepareidoliatask_subject_response_11_20_nptstimulusno_6` | text | YES |
| `pdbpnoisepareidoliatask_scoring_nptcorrectfacerespnsscore` | double precision | YES |
| `pdbpnoisepareidoliatask_scoring_nptcorrectnoiserespnsscore` | double precision | YES |
| `pdbpnoisepareidoliatask_scoring_npttotalcorrectscore` | double precision | YES |
| `pdbpnoisepareidoliatask_scoring_nptnoiserespnsscore` | double precision | YES |

### `pdbp_raw.pdbpparkinsonismvitalsigns` — 495 rows, 26 columns

| Column | Type | Nullable |
|---|---|---|
| `study_id` | bigint | YES |
| `dataset` | text | YES |
| `pdbpparkinsonismvitalsigns_required_fields_sitename` | text | YES |
| `pdbpparkinsonismvitalsigns_required_fields_visittyppdbp` | text | YES |
| `pdbpparkinsonismvitalsigns_required_fields_visitdate` | double precision | YES |
| `pdbpparkinsonismvitalsigns_required_fields_guid` | text | YES |
| `pdbpparkinsonismvitalsigns_required_fields_associated_guid` | double precision | YES |
| `pdbpparkinsonismvitalsigns_required_fields_ageyrs` | text | YES |
| `pdbpparkinsonismvitalsigns_required_fields_ageremaindrmonths` | double precision | YES |
| `pdbpparkinsonismvitalsigns_required_fields_ageval` | double precision | YES |
| `pdbpparkinsonismvitalsigns_vital_signs_vitalsgndatetime` | text | YES |
| `pdbpparkinsonismvitalsigns_vital_signs_resprate` | double precision | YES |
| `pdbpparkinsonismvitalsigns_vital_signs_tempmeasr` | double precision | YES |
| `pdbpparkinsonismvitalsigns_vital_signs_tempmeasrantmicsite` | text | YES |
| `pdbpparkinsonismvitalsigns_vital_signs_wgtmeasr` | double precision | YES |
| `pdbpparkinsonismvitalsigns_vital_signs_hgtmeasr` | double precision | YES |
| `pdbpparkinsonismvitalsigns_orthostatic_assessment_orthostat3` | double precision | YES |
| `pdbpparkinsonismvitalsigns_orthostatic_assessment_orthostat3_1` | double precision | YES |
| `pdbpparkinsonismvitalsigns_orthostatic_assessment_orthostat3_2` | double precision | YES |
| `pdbpparkinsonismvitalsigns_orthostatic_assessment_orthostat3_3` | double precision | YES |
| `pdbpparkinsonismvitalsigns_orthostatic_assessment_orthostat3_4` | double precision | YES |
| `pdbpparkinsonismvitalsigns_orthostatic_assessment_orthostat3_5` | double precision | YES |
| `pdbpparkinsonismvitalsigns_orthostatic_assessment_orthostat3_6` | double precision | YES |
| `pdbpparkinsonismvitalsigns_orthostatic_assessment_orthostat3_7` | double precision | YES |
| `pdbpparkinsonismvitalsigns_orthostatic_assessment_orthostat3_8` | double precision | YES |
| `pdbpparkinsonismvitalsigns_orthostatic_assessment_orthostat3_9` | text | YES |

### `pdbp_raw.pdq39` — 529 rows, 57 columns

| Column | Type | Nullable |
|---|---|---|
| `study_id` | bigint | YES |
| `dataset` | text | YES |
| `pdq39_required_fields_sitename` | text | YES |
| `pdq39_required_fields_visittyppdbp` | text | YES |
| `pdq39_required_fields_visitdate` | double precision | YES |
| `pdq39_required_fields_guid` | text | YES |
| `pdq39_required_fields_associated_guid` | double precision | YES |
| `pdq39_required_fields_ageyrs` | text | YES |
| `pdq39_required_fields_ageremaindrmonths` | double precision | YES |
| `pdq39_required_fields_ageval` | double precision | YES |
| `pdq39_pdq_39_pdq_39_leisure` | text | YES |
| `pdq39_pdq_39_pdq_39_housework` | text | YES |
| `pdq39_pdq_39_pdq_39_grocerybags` | text | YES |
| `pdq39_pdq_39_pdq_39_walkinghalfmile` | text | YES |
| `pdq39_pdq_39_pdq_39_walkingblock` | text | YES |
| `pdq39_pdq_39_pdq_39_house` | text | YES |
| `pdq39_pdq_39_pdq_39_publicplaces` | text | YES |
| `pdq39_pdq_39_pdq_39_outside` | text | YES |
| `pdq39_pdq_39_pdq_39_falling` | text | YES |
| `pdq39_pdq_39_pdq_39_confined` | text | YES |
| `pdq39_pdq_39_pdq_39_showering` | text | YES |
| `pdq39_pdq_39_pdq_39_dressing` | text | YES |
| `pdq39_pdq_39_pdq_39_buttons` | text | YES |
| `pdq39_pdq_39_pdq_39_writing` | text | YES |
| `pdq39_pdq_39_pdq_39_cuttingfood` | text | YES |
| `pdq39_pdq_39_pdq_39_holdingdrink` | text | YES |
| `pdq39_pdq_39_pdq_39_depressed` | text | YES |
| `pdq39_pdq_39_pdq_39_lonely` | text | YES |
| `pdq39_pdq_39_pdq_39_tearful` | text | YES |
| `pdq39_pdq_39_pdq_39_angry` | text | YES |
| `pdq39_pdq_39_pdq_39_anxious` | text | YES |
| `pdq39_pdq_39_pdq_39_worried` | text | YES |
| `pdq39_pdq_39_pdq_39_hidepd` | text | YES |
| `pdq39_pdq_39_pdq_39_eatingpublic` | text | YES |
| `pdq39_pdq_39_pdq_39_embarrassedpublic` | text | YES |
| `pdq39_pdq_39_pdq_39_worriedreaction` | text | YES |
| `pdq39_pdq_39_pdq_39_relationships` | text | YES |
| `pdq39_pdq_39_pdq_39_lackofsuprtprtnr` | text | YES |
| `pdq39_pdq_39_pdq_39_lackofsuprtfmly` | text | YES |
| `pdq39_pdq_39_pdq_39_unexpctdsleep` | text | YES |
| `pdq39_pdq_39_pdq_39_concentration` | text | YES |
| `pdq39_pdq_39_pdq_39_memory` | text | YES |
| `pdq39_pdq_39_pdq_39_dreams` | text | YES |
| `pdq39_pdq_39_pdq_39_speaking` | text | YES |
| `pdq39_pdq_39_pdq_39_communicate` | text | YES |
| `pdq39_pdq_39_pdq_39_ignored` | text | YES |
| `pdq39_pdq_39_pdq_39_crampsspasms` | text | YES |
| `pdq39_pdq_39_pdq_39_achespains` | text | YES |
| `pdq39_pdq_39_pdq_39_hotcold` | text | YES |
| `pdq39_scale_scores_pdq_39_totalscore_mobility` | double precision | YES |
| `pdq39_scale_scores_pdq_39_totalscore_adl` | double precision | YES |
| `pdq39_scale_scores_pdq_39_totalscore_emotional` | double precision | YES |
| `pdq39_scale_scores_pdq_39_totalscore_stigma` | double precision | YES |
| `pdq39_scale_scores_pdq_39_totalscore_socialsuprt` | double precision | YES |
| `pdq39_scale_scores_pdq_39_totalscore_cogimpairmnt` | double precision | YES |
| `pdq39_scale_scores_pdq_39_totalscore_communcation` | double precision | YES |
| `pdq39_scale_scores_pdq_39_totalscore_boddiscomfrt` | double precision | YES |

### `pdbp_raw.pdq_39` — 2,988 rows, 51 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `pdq39_01_doing_leisure_activity` | text | YES |
| `pdq39_02_looking_after_home` | text | YES |
| `pdq39_03_carrying_shopping_bags` | text | YES |
| `pdq39_04_walking_half_mile` | text | YES |
| `pdq39_05_walking_100_yards` | text | YES |
| `pdq39_06_getting_around_house` | text | YES |
| `pdq39_07_getting_around_in_public` | text | YES |
| `pdq39_08_need_someone_to_accompany` | text | YES |
| `pdq39_09_worried_about_falling` | text | YES |
| `pdq39_10_confined_to_house` | text | YES |
| `pdq39_11_showering` | text | YES |
| `pdq39_12_dressing` | text | YES |
| `pdq39_13_buttons_and_shoelaces` | text | YES |
| `pdq39_14_writing` | text | YES |
| `pdq39_15_cutting_food` | text | YES |
| `pdq39_16_spill_drink` | text | YES |
| `pdq39_17_depressed` | text | YES |
| `pdq39_18_lonely` | text | YES |
| `pdq39_19_weepy` | text | YES |
| `pdq39_20_angry` | text | YES |
| `pdq39_21_anxious` | text | YES |
| `pdq39_22_worried_about_future` | text | YES |
| `pdq39_23_hide_pd_from_people` | text | YES |
| `pdq39_24_avoid_eat_drink_in_public` | text | YES |
| `pdq39_25_embarassed_in_public` | text | YES |
| `pdq39_26_worried_about_reactions` | text | YES |
| `pdq39_27_close_personal_relations` | text | YES |
| `pdq39_28_support_from_spouse` | text | YES |
| `pdq39_29_support_from_family` | text | YES |
| `pdq39_30_sleep_in_day` | text | YES |
| `pdq39_31_problem_with_concentration` | text | YES |
| `pdq39_32_memory_is_failing` | text | YES |
| `pdq39_33_hallucinations` | text | YES |
| `pdq39_34_speaking` | text | YES |
| `pdq39_35_unable_to_communicate` | text | YES |
| `pdq39_36_felt_ignored` | text | YES |
| `pdq39_37_muscle_cramps` | text | YES |
| `pdq39_38_joint_pains` | text | YES |
| `pdq39_39_hot_or_cold` | text | YES |
| `pdq39_mobility_score` | double precision | YES |
| `pdq39_adl_score` | double precision | YES |
| `pdq39_emotional_score` | double precision | YES |
| `pdq39_stigma_score` | double precision | YES |
| `pdq39_social_score` | double precision | YES |
| `pdq39_cognition_score` | double precision | YES |
| `pdq39_communication_score` | double precision | YES |
| `pdq39_discomfort_score` | double precision | YES |

### `pdbp_raw.priorandconcomitantmeds` — 311 rows, 23 columns

| Column | Type | Nullable |
|---|---|---|
| `study_id` | double precision | YES |
| `dataset` | text | YES |
| `priorandconcomitantmeds_required_fields_sitename` | text | YES |
| `priorandconcomitantmeds_required_fields_visittyppdbp` | text | YES |
| `priorandconcomitantmeds_required_fields_visitdate` | double precision | YES |
| `priorandconcomitantmeds_required_fields_guid` | text | YES |
| `priorandconcomitantmeds_required_fields_associated_guid` | double precision | YES |
| `priorandconcomitantmeds_required_fields_ageyrs` | text | YES |
| `priorandconcomitantmeds_required_fields_ageremaindrmonths` | double precision | YES |
| `priorandconcomitantmeds_required_fields_ageval` | double precision | YES |
| `priorandconcomitantmeds_parkinson_s_disease_medications_medc` | text | YES |
| `priorandconcomitantmeds_parkinson_s_disease_medications_medc_1` | text | YES |
| `priorandconcomitantmeds_parkinson_s_disease_medications_medc_2` | text | YES |
| `priorandconcomitantmeds_parkinson_s_disease_medications_medc_3` | text | YES |
| `priorandconcomitantmeds_parkinson_s_disease_medications_medc_4` | text | YES |
| `priorandconcomitantmeds_parkinson_s_disease_medications_medc_5` | double precision | YES |
| `priorandconcomitantmeds_parkinson_s_disease_medications_medc_6` | double precision | YES |
| `priorandconcomitantmeds_other_medications_medctnpriorconcomn` | text | YES |
| `priorandconcomitantmeds_other_medications_medctnpriorconcomi` | text | YES |
| `priorandconcomitantmeds_other_medications_medctnpriorconcomd` | text | YES |
| `priorandconcomitantmeds_other_medications_medctnpriorconcomd_1` | text | YES |
| `priorandconcomitantmeds_other_medications_medctnpriorconcomf` | text | YES |
| `priorandconcomitantmeds_other_medications_medctnpriorconcomr` | text | YES |

### `pdbp_raw.protocoldeviations` — 430 rows, 13 columns

| Column | Type | Nullable |
|---|---|---|
| `study_id` | double precision | YES |
| `dataset` | text | YES |
| `protocoldeviations_required_fields_sitename` | text | YES |
| `protocoldeviations_required_fields_visittyppdbp` | text | YES |
| `protocoldeviations_required_fields_visitdate` | double precision | YES |
| `protocoldeviations_required_fields_guid` | text | YES |
| `protocoldeviations_required_fields_associated_guid` | double precision | YES |
| `protocoldeviations_required_fields_ageyrs` | text | YES |
| `protocoldeviations_required_fields_ageremaindrmonths` | double precision | YES |
| `protocoldeviations_required_fields_ageval` | double precision | YES |
| `protocoldeviations_deviation_indicator_protocoldeviatnoccurind` | text | YES |
| `protocoldeviations_deviations_protocoldeviatndescriptxt` | text | YES |
| `protocoldeviations_deviations_protocoldeviatnoccurdatetime` | text | YES |

### `pdbp_raw.rem_sleep_behavior_disorder` — 2,944 rows, 25 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `msq_info_source` | text | YES |
| `msq_interviewee_live_with_subject` | text | YES |
| `msq_interviewee_sleep_same_room` | text | YES |
| `msq_distracting_sleep_behaviors` | text | YES |
| `msq01_act_out_dreams` | text | YES |
| `msq01a_act_out_years` | double precision | YES |
| `msq01a_act_out_months` | double precision | YES |
| `msq01b_patient_injured` | text | YES |
| `msq01c_bedpartner_injured` | text | YES |
| `msq01d_told_dreams` | text | YES |
| `msq01e_dream_details_match` | text | YES |
| `msq02_legs_jerk` | text | YES |
| `msq03_restless_legs` | text | YES |
| `msq03a_leg_sensations_decrease` | text | YES |
| `msq03b_time_leg_sensations_worst` | text | YES |
| `msq04b_walked_asleep` | text | YES |
| `msq05_snorted_awake` | text | YES |
| `msq06_stop_breathing` | text | YES |
| `msq06a_treated_for_stop_breathing` | text | YES |
| `msq07_leg_cramps` | text | YES |
| `msq08_rate_of_alertness` | double precision | YES |

### `pdbp_raw.rem_sleep_stiasny_kolster` — 59 rows, 51 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `code_rbd_info_source` | double precision | YES |
| `code_rbd01_vivid_dreams` | bigint | YES |
| `code_rbd02_aggressive_or_action_packed_dreams` | bigint | YES |
| `code_rbd03_nocturnal_behaviour` | bigint | YES |
| `code_rbd04_move_arms_legs_during_sleep` | bigint | YES |
| `code_rbd05_hurt_bed_partner` | bigint | YES |
| `code_rbd06_1_speaking_in_sleep` | bigint | YES |
| `code_rbd06_2_sudden_limb_movements` | bigint | YES |
| `code_rbd06_3_complex_movements` | bigint | YES |
| `code_rbd06_4_things_fell_down` | bigint | YES |
| `code_rbd07_my_movements_awake_me` | bigint | YES |
| `code_rbd08_remember_dreams` | bigint | YES |
| `code_rbd09_sleep_is_disturbed` | bigint | YES |
| `code_rbd10a_stroke` | double precision | YES |
| `code_rbd10b_head_trauma` | double precision | YES |
| `code_rbd10c_parkinsonism` | double precision | YES |
| `code_rbd10d_rls` | double precision | YES |
| `code_rbd10e_narcolepsy` | double precision | YES |
| `code_rbd10f_depression` | double precision | YES |
| `code_rbd10g_epilepsy` | double precision | YES |
| `code_rbd10h_brain_inflammatory_disease` | double precision | YES |
| `code_rbd10i_other` | double precision | YES |
| `code_rbd10_nervous_system_disease` | bigint | YES |
| `rbd_info_source` | double precision | YES |
| `rbd01_vivid_dreams` | text | YES |
| `rbd02_aggressive_or_action_packed_dreams` | text | YES |
| `rbd03_nocturnal_behaviour` | text | YES |
| `rbd04_move_arms_legs_during_sleep` | text | YES |
| `rbd05_hurt_bed_partner` | text | YES |
| `rbd06_1_speaking_in_sleep` | text | YES |
| `rbd06_2_sudden_limb_movements` | text | YES |
| `rbd06_3_complex_movements` | text | YES |
| `rbd06_4_things_fell_down` | text | YES |
| `rbd07_my_movements_awake_me` | text | YES |
| `rbd08_remember_dreams` | text | YES |
| `rbd09_sleep_is_disturbed` | text | YES |
| `rbd10a_stroke` | double precision | YES |
| `rbd10b_head_trauma` | double precision | YES |
| `rbd10c_parkinsonism` | double precision | YES |
| `rbd10d_rls` | double precision | YES |
| `rbd10e_narcolepsy` | double precision | YES |
| `rbd10f_depression` | double precision | YES |
| `rbd10g_epilepsy` | double precision | YES |
| `rbd10h_brain_inflammatory_disease` | double precision | YES |
| `rbd10i_other` | double precision | YES |
| `rbd10_nervous_system_disease` | text | YES |
| `rbd_summary_score` | bigint | YES |

### `pdbp_raw.univofpennsmellidentest` — 544 rows, 16 columns

| Column | Type | Nullable |
|---|---|---|
| `study_id` | bigint | YES |
| `dataset` | text | YES |
| `univofpennsmellidentest_required_fields_sitename` | text | YES |
| `univofpennsmellidentest_required_fields_visittyppdbp` | text | YES |
| `univofpennsmellidentest_required_fields_visitdate` | double precision | YES |
| `univofpennsmellidentest_required_fields_guid` | text | YES |
| `univofpennsmellidentest_required_fields_associated_guid` | double precision | YES |
| `univofpennsmellidentest_required_fields_ageyrs` | text | YES |
| `univofpennsmellidentest_required_fields_ageremaindrmonths` | double precision | YES |
| `univofpennsmellidentest_required_fields_ageval` | double precision | YES |
| `univofpennsmellidentest_upsit_upennsitind` | text | YES |
| `univofpennsmellidentest_upsit_upennsitbk1scr` | double precision | YES |
| `univofpennsmellidentest_upsit_upennsitbk2scr` | double precision | YES |
| `univofpennsmellidentest_upsit_upennsitbk3scr` | double precision | YES |
| `univofpennsmellidentest_upsit_upennsitbk4scr` | double precision | YES |
| `univofpennsmellidentest_upsit_upennsittotal` | double precision | YES |

### `pdbp_raw.upsit` — 2,955 rows, 10 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `upsit_performed` | text | YES |
| `score_from_booklet_1` | double precision | YES |
| `score_from_booklet_2` | double precision | YES |
| `score_from_booklet_3` | double precision | YES |
| `score_from_booklet_4` | double precision | YES |
| `upsit_total_score` | double precision | YES |

### `pdbp_raw.vitalsigns` — 401 rows, 20 columns

| Column | Type | Nullable |
|---|---|---|
| `study_id` | double precision | YES |
| `dataset` | text | YES |
| `vitalsigns_required_fields_sitename` | text | YES |
| `vitalsigns_required_fields_visittyppdbp` | text | YES |
| `vitalsigns_required_fields_visitdate` | double precision | YES |
| `vitalsigns_required_fields_guid` | text | YES |
| `vitalsigns_required_fields_associated_guid` | double precision | YES |
| `vitalsigns_required_fields_ageyrs` | text | YES |
| `vitalsigns_required_fields_ageremaindrmonths` | double precision | YES |
| `vitalsigns_required_fields_ageval` | double precision | YES |
| `vitalsigns_vital_signs_vitalsgndatetime` | text | YES |
| `vitalsigns_vital_signs_heartrate` | double precision | YES |
| `vitalsigns_vital_signs_resprate` | double precision | YES |
| `vitalsigns_vital_signs_tempmeasr` | double precision | YES |
| `vitalsigns_vital_signs_tempmeasrantmicsite` | text | YES |
| `vitalsigns_vital_signs_wgtmeasr` | double precision | YES |
| `vitalsigns_vital_signs_hgtmeasr` | double precision | YES |
| `vitalsigns_blood_pressure_bldpressrmeasrpositiontyp` | text | YES |
| `vitalsigns_blood_pressure_bldpressrsystmeasr` | double precision | YES |
| `vitalsigns_blood_pressure_bldpressrdiastlmeasr` | double precision | YES |

## Schema `hbs_raw` (11 tables)

_HBS external prediction cohort (low-data deployment test)_

### `hbs_raw.amp_pd_case_control` — 1,189 rows, 5 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `diagnosis_at_baseline` | text | YES |
| `diagnosis_latest` | text | YES |
| `case_control_other_at_baseline` | text | YES |
| `case_control_other_latest` | text | YES |

### `hbs_raw.biospecimen_other` — 1,240 rows, 8 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `sample_type` | text | YES |
| `test_name` | text | YES |
| `test_value` | double precision | YES |
| `test_units` | text | YES |

### `hbs_raw.demographics` — 1,189 rows, 9 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `age_at_baseline` | bigint | YES |
| `sex` | text | YES |
| `ethnicity` | text | YES |
| `race` | text | YES |
| `education_level_years` | text | YES |

### `hbs_raw.enrollment` — 1,189 rows, 8 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `enrollment_months_after_baseline` | double precision | YES |
| `informed_consent_months_after_baseline` | double precision | YES |
| `prodromal_category` | text | YES |
| `study_arm` | text | YES |

### `hbs_raw.family_history_pd` — 1,189 rows, 7 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `biological_mother_with_pd` | text | YES |
| `biological_father_with_pd` | text | YES |
| `other_relative_with_pd` | text | YES |

### `hbs_raw.mds_updrs_part_ii` — 649 rows, 32 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `mds_updrs_part_ii_primary_info_source` | double precision | YES |
| `code_upd2201_speech` | double precision | YES |
| `code_upd2202_saliva_and_drooling` | double precision | YES |
| `code_upd2203_chewing_and_swallowing` | double precision | YES |
| `code_upd2204_eating_tasks` | double precision | YES |
| `code_upd2205_dressing` | double precision | YES |
| `code_upd2206_hygiene` | double precision | YES |
| `code_upd2207_handwriting` | double precision | YES |
| `code_upd2208_doing_hobbies_and_other_activities` | double precision | YES |
| `code_upd2209_turning_in_bed` | double precision | YES |
| `code_upd2210_tremor` | double precision | YES |
| `code_upd2211_get_out_of_bed_car_or_deep_chair` | double precision | YES |
| `code_upd2212_walking_and_balance` | double precision | YES |
| `code_upd2213_freezing` | double precision | YES |
| `upd2201_speech` | text | YES |
| `upd2202_saliva_and_drooling` | double precision | YES |
| `upd2203_chewing_and_swallowing` | double precision | YES |
| `upd2204_eating_tasks` | text | YES |
| `upd2205_dressing` | text | YES |
| `upd2206_hygiene` | double precision | YES |
| `upd2207_handwriting` | double precision | YES |
| `upd2208_doing_hobbies_and_other_activities` | double precision | YES |
| `upd2209_turning_in_bed` | double precision | YES |
| `upd2210_tremor` | text | YES |
| `upd2211_get_out_of_bed_car_or_deep_chair` | double precision | YES |
| `upd2212_walking_and_balance` | double precision | YES |
| `upd2213_freezing` | double precision | YES |
| `mds_updrs_part_ii_summary_score` | double precision | YES |

### `hbs_raw.mds_updrs_part_iii` — 646 rows, 77 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `code_upd2301_speech_problems` | double precision | YES |
| `code_upd2302_facial_expression` | double precision | YES |
| `code_upd2303a_rigidity_neck` | double precision | YES |
| `code_upd2303b_rigidity_rt_upper_extremity` | double precision | YES |
| `code_upd2303c_rigidity_left_upper_extremity` | double precision | YES |
| `code_upd2303d_rigidity_rt_lower_extremity` | double precision | YES |
| `code_upd2303e_rigidity_left_lower_extremity` | double precision | YES |
| `code_upd2304a_right_finger_tapping` | double precision | YES |
| `code_upd2304b_left_finger_tapping` | double precision | YES |
| `code_upd2305a_right_hand_movements` | double precision | YES |
| `code_upd2305b_left_hand_movements` | double precision | YES |
| `code_upd2306a_pron_sup_movement_right_hand` | double precision | YES |
| `code_upd2306b_pron_sup_movement_left_hand` | double precision | YES |
| `code_upd2307a_right_toe_tapping` | double precision | YES |
| `code_upd2307b_left_toe_tapping` | double precision | YES |
| `code_upd2308a_right_leg_agility` | double precision | YES |
| `code_upd2308b_left_leg_agility` | double precision | YES |
| `code_upd2309_arising_from_chair` | double precision | YES |
| `code_upd2310_gait` | double precision | YES |
| `code_upd2311_freezing_of_gait` | double precision | YES |
| `code_upd2312_postural_stability` | double precision | YES |
| `code_upd2313_posture` | double precision | YES |
| `code_upd2314_body_bradykinesia` | double precision | YES |
| `code_upd2315a_postural_tremor_of_right_hand` | double precision | YES |
| `code_upd2315b_postural_tremor_of_left_hand` | double precision | YES |
| `code_upd2316a_kinetic_tremor_of_right_hand` | double precision | YES |
| `code_upd2316b_kinetic_tremor_of_left_hand` | double precision | YES |
| `code_upd2317a_rest_tremor_amplitude_right_upper_extremity` | double precision | YES |
| `code_upd2317b_rest_tremor_amplitude_left_upper_extremity` | double precision | YES |
| `code_upd2317c_rest_tremor_amplitude_right_lower_extremity` | double precision | YES |
| `code_upd2317d_rest_tremor_amplitude_left_lower_extremity` | double precision | YES |
| `code_upd2317e_rest_tremor_amplitude_lip_or_jaw` | double precision | YES |
| `code_upd2318_consistency_of_rest_tremor` | double precision | YES |
| `upd2301_speech_problems` | text | YES |
| `upd2302_facial_expression` | double precision | YES |
| `upd2303a_rigidity_neck` | double precision | YES |
| `upd2303b_rigidity_rt_upper_extremity` | double precision | YES |
| `upd2303c_rigidity_left_upper_extremity` | double precision | YES |
| `upd2303d_rigidity_rt_lower_extremity` | double precision | YES |
| `upd2303e_rigidity_left_lower_extremity` | double precision | YES |
| `upd2304a_right_finger_tapping` | double precision | YES |
| `upd2304b_left_finger_tapping` | double precision | YES |
| `upd2305a_right_hand_movements` | double precision | YES |
| `upd2305b_left_hand_movements` | double precision | YES |
| `upd2306a_pron_sup_movement_right_hand` | double precision | YES |
| `upd2306b_pron_sup_movement_left_hand` | double precision | YES |
| `upd2307a_right_toe_tapping` | double precision | YES |
| `upd2307b_left_toe_tapping` | double precision | YES |
| `upd2308a_right_leg_agility` | double precision | YES |
| `upd2308b_left_leg_agility` | double precision | YES |
| `upd2309_arising_from_chair` | text | YES |
| `upd2310_gait` | double precision | YES |
| `upd2311_freezing_of_gait` | double precision | YES |
| `upd2312_postural_stability` | double precision | YES |
| `upd2313_posture` | double precision | YES |
| `upd2314_body_bradykinesia` | text | YES |
| `upd2315a_postural_tremor_of_right_hand` | double precision | YES |
| `upd2315b_postural_tremor_of_left_hand` | double precision | YES |
| `upd2316a_kinetic_tremor_of_right_hand` | double precision | YES |
| `upd2316b_kinetic_tremor_of_left_hand` | double precision | YES |
| `upd2317a_rest_tremor_amplitude_right_upper_extremity` | double precision | YES |
| `upd2317b_rest_tremor_amplitude_left_upper_extremity` | double precision | YES |
| `upd2317c_rest_tremor_amplitude_right_lower_extremity` | double precision | YES |
| `upd2317d_rest_tremor_amplitude_left_lower_extremity` | double precision | YES |
| `upd2317e_rest_tremor_amplitude_lip_or_jaw` | double precision | YES |
| `upd2318_consistency_of_rest_tremor` | double precision | YES |
| `upd2da_dyskinesias_during_exam` | double precision | YES |
| `upd2db_movements_interfere_with_ratings` | double precision | YES |
| `code_upd2hy_hoehn_and_yahr_stage` | double precision | YES |
| `upd2hy_hoehn_and_yahr_stage` | text | YES |
| `upd23a_medication_for_pd` | double precision | YES |
| `upd23b_clinical_state_on_medication` | double precision | YES |
| `mds_updrs_part_iii_summary_score` | double precision | YES |

### `hbs_raw.mds_updrs_part_iv` — 639 rows, 17 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `code_upd2401_time_spent_with_dyskinesias` | bigint | YES |
| `code_upd2402_functional_impact_of_dyskinesias` | double precision | YES |
| `code_upd2403_time_spent_in_the_off_state` | double precision | YES |
| `code_upd2404_functional_impact_of_fluctuations` | double precision | YES |
| `code_upd2405_complexity_of_motor_fluctuations` | double precision | YES |
| `code_upd2406_painful_off_state_dystonia` | double precision | YES |
| `upd2401_time_spent_with_dyskinesias` | text | YES |
| `upd2402_functional_impact_of_dyskinesias` | double precision | YES |
| `upd2403_time_spent_in_the_off_state` | text | YES |
| `upd2404_functional_impact_of_fluctuations` | double precision | YES |
| `upd2405_complexity_of_motor_fluctuations` | double precision | YES |
| `upd2406_painful_off_state_dystonia` | double precision | YES |
| `mds_updrs_part_iv_summary_score` | double precision | YES |

### `hbs_raw.pd_medical_history` — 1,189 rows, 20 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `diagnosis` | text | YES |
| `initial_diagnosis` | double precision | YES |
| `most_recent_diagnosis` | double precision | YES |
| `change_in_diagnosis` | double precision | YES |
| `change_in_diagnosis_months_after_baseline` | double precision | YES |
| `surgery_for_parkinson_disease` | text | YES |
| `pd_diagnosis_months_after_baseline` | double precision | YES |
| `age_at_diagnosis` | double precision | YES |
| `pd_medication_initiation_months_after_baseline` | double precision | YES |
| `pd_medication_start_months_after_baseline` | double precision | YES |
| `use_of_pd_medication` | text | YES |
| `pd_medication_recent_use_months_after_baseline` | double precision | YES |
| `on_levodopa` | text | YES |
| `on_dopamine_agonist` | text | YES |
| `on_other_pd_medications` | text | YES |
| `diagnosis_type` | double precision | YES |

### `hbs_raw.pdq_39` — 22 rows, 51 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `pdq39_01_doing_leisure_activity` | text | YES |
| `pdq39_02_looking_after_home` | text | YES |
| `pdq39_03_carrying_shopping_bags` | text | YES |
| `pdq39_04_walking_half_mile` | text | YES |
| `pdq39_05_walking_100_yards` | text | YES |
| `pdq39_06_getting_around_house` | text | YES |
| `pdq39_07_getting_around_in_public` | text | YES |
| `pdq39_08_need_someone_to_accompany` | text | YES |
| `pdq39_09_worried_about_falling` | text | YES |
| `pdq39_10_confined_to_house` | text | YES |
| `pdq39_11_showering` | text | YES |
| `pdq39_12_dressing` | text | YES |
| `pdq39_13_buttons_and_shoelaces` | text | YES |
| `pdq39_14_writing` | text | YES |
| `pdq39_15_cutting_food` | text | YES |
| `pdq39_16_spill_drink` | text | YES |
| `pdq39_17_depressed` | text | YES |
| `pdq39_18_lonely` | text | YES |
| `pdq39_19_weepy` | text | YES |
| `pdq39_20_angry` | text | YES |
| `pdq39_21_anxious` | text | YES |
| `pdq39_22_worried_about_future` | text | YES |
| `pdq39_23_hide_pd_from_people` | text | YES |
| `pdq39_24_avoid_eat_drink_in_public` | text | YES |
| `pdq39_25_embarassed_in_public` | text | YES |
| `pdq39_26_worried_about_reactions` | text | YES |
| `pdq39_27_close_personal_relations` | text | YES |
| `pdq39_28_support_from_spouse` | text | YES |
| `pdq39_29_support_from_family` | text | YES |
| `pdq39_30_sleep_in_day` | text | YES |
| `pdq39_31_problem_with_concentration` | text | YES |
| `pdq39_32_memory_is_failing` | text | YES |
| `pdq39_33_hallucinations` | text | YES |
| `pdq39_34_speaking` | text | YES |
| `pdq39_35_unable_to_communicate` | text | YES |
| `pdq39_36_felt_ignored` | text | YES |
| `pdq39_37_muscle_cramps` | text | YES |
| `pdq39_38_joint_pains` | text | YES |
| `pdq39_39_hot_or_cold` | text | YES |
| `pdq39_mobility_score` | double precision | YES |
| `pdq39_adl_score` | double precision | YES |
| `pdq39_emotional_score` | double precision | YES |
| `pdq39_stigma_score` | double precision | YES |
| `pdq39_social_score` | double precision | YES |
| `pdq39_cognition_score` | double precision | YES |
| `pdq39_communication_score` | double precision | YES |
| `pdq39_discomfort_score` | double precision | YES |

### `hbs_raw.rem_sleep_stiasny_kolster` — 467 rows, 51 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `guid` | text | YES |
| `visit_name` | text | YES |
| `visit_month` | double precision | YES |
| `code_rbd_info_source` | double precision | YES |
| `code_rbd01_vivid_dreams` | bigint | YES |
| `code_rbd02_aggressive_or_action_packed_dreams` | bigint | YES |
| `code_rbd03_nocturnal_behaviour` | double precision | YES |
| `code_rbd04_move_arms_legs_during_sleep` | bigint | YES |
| `code_rbd05_hurt_bed_partner` | double precision | YES |
| `code_rbd06_1_speaking_in_sleep` | bigint | YES |
| `code_rbd06_2_sudden_limb_movements` | double precision | YES |
| `code_rbd06_3_complex_movements` | double precision | YES |
| `code_rbd06_4_things_fell_down` | double precision | YES |
| `code_rbd07_my_movements_awake_me` | bigint | YES |
| `code_rbd08_remember_dreams` | double precision | YES |
| `code_rbd09_sleep_is_disturbed` | bigint | YES |
| `code_rbd10a_stroke` | double precision | YES |
| `code_rbd10b_head_trauma` | double precision | YES |
| `code_rbd10c_parkinsonism` | double precision | YES |
| `code_rbd10d_rls` | double precision | YES |
| `code_rbd10e_narcolepsy` | double precision | YES |
| `code_rbd10f_depression` | double precision | YES |
| `code_rbd10g_epilepsy` | double precision | YES |
| `code_rbd10h_brain_inflammatory_disease` | double precision | YES |
| `code_rbd10i_other` | double precision | YES |
| `code_rbd10_nervous_system_disease` | bigint | YES |
| `rbd_info_source` | double precision | YES |
| `rbd01_vivid_dreams` | text | YES |
| `rbd02_aggressive_or_action_packed_dreams` | text | YES |
| `rbd03_nocturnal_behaviour` | text | YES |
| `rbd04_move_arms_legs_during_sleep` | text | YES |
| `rbd05_hurt_bed_partner` | text | YES |
| `rbd06_1_speaking_in_sleep` | text | YES |
| `rbd06_2_sudden_limb_movements` | text | YES |
| `rbd06_3_complex_movements` | text | YES |
| `rbd06_4_things_fell_down` | text | YES |
| `rbd07_my_movements_awake_me` | text | YES |
| `rbd08_remember_dreams` | text | YES |
| `rbd09_sleep_is_disturbed` | text | YES |
| `rbd10a_stroke` | double precision | YES |
| `rbd10b_head_trauma` | double precision | YES |
| `rbd10c_parkinsonism` | double precision | YES |
| `rbd10d_rls` | double precision | YES |
| `rbd10e_narcolepsy` | double precision | YES |
| `rbd10f_depression` | double precision | YES |
| `rbd10g_epilepsy` | double precision | YES |
| `rbd10h_brain_inflammatory_disease` | double precision | YES |
| `rbd10i_other` | double precision | YES |
| `rbd10_nervous_system_disease` | text | YES |
| `rbd_summary_score` | bigint | YES |

## Schema `staging` (3 tables)

_NSD-ISS staging results per cohort_

### `staging.biofind_nsd_iss_staging` — 103 rows, 19 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `saa_result` | boolean | YES |
| `saa_score` | double precision | YES |
| `np1cog` | double precision | YES |
| `p1tot_raw` | double precision | YES |
| `p1tot` | double precision | YES |
| `p2tot` | double precision | YES |
| `p3tot` | double precision | YES |
| `mcatot` | double precision | YES |
| `pdmedyn` | double precision | YES |
| `rbdsq_total` | double precision | YES |
| `rbd_status` | bigint | YES |
| `nsd_iss_stage` | bigint | YES |
| `cognitive_stage` | double precision | YES |
| `motor_stage` | double precision | YES |
| `nonmotor_stage` | double precision | YES |
| `target_binary` | bigint | YES |
| `target_3class` | double precision | YES |
| `target_nsd_positive` | double precision | YES |

### `staging.nsd_iss_staging_enriched` — 2,201 rows, 16 columns

| Column | Type | Nullable |
|---|---|---|
| `patno` | bigint | YES |
| `nsd_iss_stage` | text | YES |
| `nsd_iss_stage_numeric` | double precision | YES |
| `nsd_iss_stage_ordinal` | bigint | YES |
| `s_positive` | boolean | YES |
| `d_positive` | boolean | YES |
| `has_clinical_signs` | boolean | YES |
| `has_functional_impairment` | boolean | YES |
| `functional_impairment_level` | text | YES |
| `staging_confidence` | text | YES |
| `n_missing_anchors` | bigint | YES |
| `missing_anchors` | text | YES |
| `target_binary` | bigint | YES |
| `target_3class` | bigint | YES |
| `target_full_ordinal` | bigint | YES |
| `target_nsd_positive` | bigint | YES |

### `staging.nsd_iss_staging_results` — 2,201 rows, 12 columns

| Column | Type | Nullable |
|---|---|---|
| `patno` | bigint | YES |
| `nsd_iss_stage` | text | YES |
| `nsd_iss_stage_numeric` | double precision | YES |
| `nsd_iss_stage_ordinal` | bigint | YES |
| `s_positive` | boolean | YES |
| `d_positive` | boolean | YES |
| `has_clinical_signs` | boolean | YES |
| `has_functional_impairment` | boolean | YES |
| `functional_impairment_level` | text | YES |
| `staging_confidence` | text | YES |
| `n_missing_anchors` | bigint | YES |
| `missing_anchors` | text | YES |

## Schema `features` (4 tables)

_Assembled ML feature matrices for Papers 1-3 + cross-cohort_

### `features.biofind_features` — 118 rows, 16 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `age_at_baseline` | bigint | YES |
| `sex` | bigint | YES |
| `education_years` | bigint | YES |
| `updrs1_total` | bigint | YES |
| `updrs2_total` | bigint | YES |
| `updrs3_tremor` | double precision | YES |
| `updrs3_rigidity` | double precision | YES |
| `updrs3_bradykinesia` | double precision | YES |
| `updrs3_axial` | double precision | YES |
| `updrs4_total` | double precision | YES |
| `moca_total` | bigint | YES |
| `rbd_total` | bigint | YES |
| `family_history_pd` | bigint | YES |
| `cohort` | text | YES |
| `diagnosis` | text | YES |

### `features.hbs_features` — 649 rows, 14 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `age_at_baseline` | bigint | YES |
| `sex` | bigint | YES |
| `education_years` | bigint | YES |
| `updrs2_total` | double precision | YES |
| `updrs3_tremor` | double precision | YES |
| `updrs3_rigidity` | double precision | YES |
| `updrs3_bradykinesia` | double precision | YES |
| `updrs3_axial` | double precision | YES |
| `updrs4_total` | double precision | YES |
| `rbd_total` | double precision | YES |
| `family_history_pd` | bigint | YES |
| `cohort` | text | YES |
| `diagnosis` | text | YES |

### `features.paper1_features_with_targets` — 2,201 rows, 38 columns

| Column | Type | Nullable |
|---|---|---|
| `patno` | bigint | YES |
| `nsd_iss_stage` | text | YES |
| `nsd_iss_stage_numeric` | double precision | YES |
| `nsd_iss_stage_ordinal` | bigint | YES |
| `s_positive` | boolean | YES |
| `d_positive` | boolean | YES |
| `has_clinical_signs` | boolean | YES |
| `has_functional_impairment` | boolean | YES |
| `functional_impairment_level` | text | YES |
| `staging_confidence` | text | YES |
| `n_missing_anchors` | bigint | YES |
| `missing_anchors` | text | YES |
| `target_binary` | bigint | YES |
| `target_3class` | bigint | YES |
| `target_full_ordinal` | bigint | YES |
| `target_nsd_positive` | bigint | YES |
| `sex` | double precision | YES |
| `handed` | double precision | YES |
| `age_at_baseline` | double precision | YES |
| `updrs1_total` | double precision | YES |
| `updrs2_total` | double precision | YES |
| `updrs3_tremor` | double precision | YES |
| `updrs3_rigidity` | double precision | YES |
| `updrs3_bradykinesia` | double precision | YES |
| `updrs3_axial` | double precision | YES |
| `updrs4_total` | double precision | YES |
| `moca_total` | double precision | YES |
| `rbd_total` | double precision | YES |
| `ess_total` | double precision | YES |
| `scopa_aut_total` | double precision | YES |
| `caudate_r_sbr` | double precision | YES |
| `caudate_l_sbr` | double precision | YES |
| `caudate_mean_sbr` | double precision | YES |
| `caudate_asymmetry` | double precision | YES |
| `caudate_putamen_ratio` | double precision | YES |
| `lrrk2_carrier` | double precision | YES |
| `gba_carrier` | double precision | YES |
| `apoe_e4_carrier` | double precision | YES |

### `features.pdbp_features` — 893 rows, 18 columns

| Column | Type | Nullable |
|---|---|---|
| `participant_id` | text | YES |
| `age_at_baseline` | bigint | YES |
| `sex` | bigint | YES |
| `education_years` | double precision | YES |
| `updrs1_total` | double precision | YES |
| `updrs2_total` | double precision | YES |
| `updrs3_tremor` | double precision | YES |
| `updrs3_rigidity` | double precision | YES |
| `updrs3_bradykinesia` | double precision | YES |
| `updrs3_axial` | double precision | YES |
| `updrs4_total` | double precision | YES |
| `moca_total` | double precision | YES |
| `upsit_total` | double precision | YES |
| `rbd_total` | double precision | YES |
| `ess_total` | double precision | YES |
| `family_history_pd` | double precision | YES |
| `cohort` | text | YES |
| `diagnosis` | text | YES |

## Schema `longitudinal` (4 tables)

_Paper 3 longitudinal NSD-ISS staging + transition events_

### `longitudinal.censored_patients` — 1,900 rows, 12 columns

| Column | Type | Nullable |
|---|---|---|
| `patno` | bigint | YES |
| `cohort` | text | YES |
| `baseline_stage` | text | YES |
| `last_observed_stage` | text | YES |
| `total_follow_up_months` | double precision | YES |
| `total_follow_up_years` | double precision | YES |
| `n_visits` | bigint | YES |
| `had_any_transition` | boolean | YES |
| `had_forward_transition` | boolean | YES |
| `had_backward_transition` | boolean | YES |
| `n_forward_transitions` | bigint | YES |
| `n_backward_transitions` | bigint | YES |

### `longitudinal.longitudinal_nsd_iss` — 16,699 rows, 17 columns

| Column | Type | Nullable |
|---|---|---|
| `patno` | bigint | YES |
| `event_id` | text | YES |
| `months_from_baseline` | double precision | YES |
| `age_at_visit` | double precision | YES |
| `nsd_stage` | text | YES |
| `nsd_stage_numeric` | double precision | YES |
| `s_positive` | boolean | YES |
| `d_positive` | boolean | YES |
| `hy_stage` | double precision | YES |
| `updrs3_total` | double precision | YES |
| `updrs2_total` | double precision | YES |
| `impairment_level` | text | YES |
| `has_clinical_signs` | boolean | YES |
| `pdmedyn` | double precision | YES |
| `datscan_event` | text | YES |
| `confidence` | text | YES |
| `cohort` | text | YES |

### `longitudinal.stage_episodes` — 4,759 rows, 10 columns

| Column | Type | Nullable |
|---|---|---|
| `patno` | bigint | YES |
| `cohort` | text | YES |
| `stage` | text | YES |
| `entry_time_months` | double precision | YES |
| `exit_time_months` | double precision | YES |
| `duration_months` | double precision | YES |
| `duration_years` | double precision | YES |
| `event` | bigint | YES |
| `exit_to` | text | YES |
| `exit_direction` | text | YES |

### `longitudinal.transition_events` — 2,859 rows, 15 columns

| Column | Type | Nullable |
|---|---|---|
| `patno` | bigint | YES |
| `cohort` | text | YES |
| `source_stage` | text | YES |
| `dest_stage` | text | YES |
| `source_stage_numeric` | double precision | YES |
| `dest_stage_numeric` | double precision | YES |
| `direction` | text | YES |
| `is_skip_transition` | boolean | YES |
| `time_interval_months` | double precision | YES |
| `time_interval_years` | double precision | YES |
| `months_from_baseline_src` | double precision | YES |
| `months_from_baseline_dst` | double precision | YES |
| `event_id_from` | text | YES |
| `event_id_to` | text | YES |
| `age_at_transition` | double precision | YES |

## Schema `paper3` (1 tables)

_Paper 3 per-visit longitudinal feature vectors_

### `paper3.longitudinal_features` — 16,699 rows, 48 columns

| Column | Type | Nullable |
|---|---|---|
| `patno` | bigint | YES |
| `event_id` | text | YES |
| `months_from_baseline` | double precision | YES |
| `age_at_visit` | double precision | YES |
| `nsd_stage` | text | YES |
| `nsd_stage_numeric` | double precision | YES |
| `s_positive` | boolean | YES |
| `d_positive` | boolean | YES |
| `hy_stage` | double precision | YES |
| `updrs3_total` | double precision | YES |
| `updrs2_total` | double precision | YES |
| `impairment_level` | text | YES |
| `has_clinical_signs` | boolean | YES |
| `pdmedyn` | double precision | YES |
| `datscan_event` | text | YES |
| `confidence` | text | YES |
| `cohort` | text | YES |
| `sex` | double precision | YES |
| `handed` | double precision | YES |
| `lrrk2_carrier` | double precision | YES |
| `gba_carrier` | double precision | YES |
| `snca_carrier` | double precision | YES |
| `apoe_e4` | double precision | YES |
| `updrs1_total` | double precision | YES |
| `updrs4_total` | double precision | YES |
| `moca_total` | double precision | YES |
| `ess_total` | double precision | YES |
| `rbd_total` | double precision | YES |
| `scopa_aut_total` | double precision | YES |
| `upsit_total` | double precision | YES |
| `caudate_r_sbr` | double precision | YES |
| `caudate_l_sbr` | double precision | YES |
| `putamen_r_sbr` | double precision | YES |
| `putamen_l_sbr` | double precision | YES |
| `caudate_mean_sbr` | double precision | YES |
| `putamen_mean_sbr` | double precision | YES |
| `caudate_asymmetry` | double precision | YES |
| `time_in_current_stage_months` | double precision | YES |
| `delta_updrs3_total` | double precision | YES |
| `delta_updrs2_total` | double precision | YES |
| `delta_updrs1_total` | double precision | YES |
| `delta_updrs4_total` | double precision | YES |
| `delta_moca_total` | double precision | YES |
| `delta_ess_total` | double precision | YES |
| `delta_hy_stage` | double precision | YES |
| `delta_scopa_aut_total` | double precision | YES |
| `visit_number` | bigint | YES |
| `age_at_baseline` | double precision | YES |

## Schema `ledd` (2 tables)

_Levodopa-equivalent daily dose + concomitant PD medication_

### `ledd.concomitant_medication_ledd` — 9,583 rows, 15 columns

| Column | Type | Nullable |
|---|---|---|
| `rec_id` | text | YES |
| `patno` | bigint | YES |
| `event_id` | text | YES |
| `pag_name` | text | YES |
| `ledtrt` | text | YES |
| `totdda` | double precision | YES |
| `leddstrmg` | double precision | YES |
| `leddosstr` | text | YES |
| `leddose` | double precision | YES |
| `leddosfrq` | double precision | YES |
| `startdt` | text | YES |
| `stopdt` | text | YES |
| `ledd` | text | YES |
| `orig_entry` | text | YES |
| `last_update` | text | YES |

### `ledd.use_of_pd_medication` — 222 rows, 19 columns

| Column | Type | Nullable |
|---|---|---|
| `rec_id` | bigint | YES |
| `f_status` | text | YES |
| `patno` | bigint | YES |
| `event_id` | text | YES |
| `pag_name` | text | YES |
| `infodt` | text | YES |
| `pdmedyn` | bigint | YES |
| `onldopa` | double precision | YES |
| `ondopag` | double precision | YES |
| `onamantd` | double precision | YES |
| `onmaobih` | double precision | YES |
| `onother` | double precision | YES |
| `pdmeddt` | text | YES |
| `pdmedtm` | text | YES |
| `nupdrtm` | text | YES |
| `orig_entry` | text | YES |
| `last_update` | text | YES |
| `query` | double precision | YES |
| `site_aprv` | text | YES |

## Schema `mechanistic` (21 tables)

_Phase 1-4 mechanistic twin outputs (posteriors, LOO, counterfactuals)_

### `mechanistic.block4_counterfactual_combined_1065` — 1,065 rows, 8 columns

| Column | Type | Nullable |
|---|---|---|
| `patno` | bigint | YES |
| `wave` | text | YES |
| `t_50_untreated_median` | double precision | YES |
| `delay_50_a_minimal_median` | double precision | YES |
| `delay_50_b_moderate_median` | double precision | YES |
| `delay_50_c_optimistic_median` | double precision | YES |
| `ess_frac` | double precision | YES |
| `pct_yr` | double precision | YES |

### `mechanistic.block4_s5_counterfactual` — 304 rows, 49 columns

| Column | Type | Nullable |
|---|---|---|
| `patno` | bigint | YES |
| `t_25pct_untreated_median_yr` | double precision | YES |
| `t_25pct_untreated_q025_yr` | double precision | YES |
| `t_25pct_untreated_q975_yr` | double precision | YES |
| `delay_25pct_a_minimal_median_yr` | double precision | YES |
| `delay_25pct_a_minimal_q025_yr` | double precision | YES |
| `delay_25pct_a_minimal_q975_yr` | double precision | YES |
| `pct_slower_25pct_a_minimal` | double precision | YES |
| `delay_25pct_b_pasadena_median_yr` | double precision | YES |
| `delay_25pct_b_pasadena_q025_yr` | double precision | YES |
| `delay_25pct_b_pasadena_q975_yr` | double precision | YES |
| `pct_slower_25pct_b_pasadena` | double precision | YES |
| `delay_25pct_c_optimistic_median_yr` | double precision | YES |
| `delay_25pct_c_optimistic_q025_yr` | double precision | YES |
| `delay_25pct_c_optimistic_q975_yr` | double precision | YES |
| `pct_slower_25pct_c_optimistic` | double precision | YES |
| `t_50pct_untreated_median_yr` | double precision | YES |
| `t_50pct_untreated_q025_yr` | double precision | YES |
| `t_50pct_untreated_q975_yr` | double precision | YES |
| `delay_50pct_a_minimal_median_yr` | double precision | YES |
| `delay_50pct_a_minimal_q025_yr` | double precision | YES |
| `delay_50pct_a_minimal_q975_yr` | double precision | YES |
| `pct_slower_50pct_a_minimal` | double precision | YES |
| `delay_50pct_b_pasadena_median_yr` | double precision | YES |
| `delay_50pct_b_pasadena_q025_yr` | double precision | YES |
| `delay_50pct_b_pasadena_q975_yr` | double precision | YES |
| `pct_slower_50pct_b_pasadena` | double precision | YES |
| `delay_50pct_c_optimistic_median_yr` | double precision | YES |
| `delay_50pct_c_optimistic_q025_yr` | double precision | YES |
| `delay_50pct_c_optimistic_q975_yr` | double precision | YES |
| `pct_slower_50pct_c_optimistic` | double precision | YES |
| `t_75pct_untreated_median_yr` | double precision | YES |
| `t_75pct_untreated_q025_yr` | double precision | YES |
| `t_75pct_untreated_q975_yr` | double precision | YES |
| `delay_75pct_a_minimal_median_yr` | double precision | YES |
| `delay_75pct_a_minimal_q025_yr` | double precision | YES |
| `delay_75pct_a_minimal_q975_yr` | double precision | YES |
| `pct_slower_75pct_a_minimal` | double precision | YES |
| `delay_75pct_b_pasadena_median_yr` | double precision | YES |
| `delay_75pct_b_pasadena_q025_yr` | double precision | YES |
| `delay_75pct_b_pasadena_q975_yr` | double precision | YES |
| `pct_slower_75pct_b_pasadena` | double precision | YES |
| `delay_75pct_c_optimistic_median_yr` | double precision | YES |
| `delay_75pct_c_optimistic_q025_yr` | double precision | YES |
| `delay_75pct_c_optimistic_q975_yr` | double precision | YES |
| `pct_slower_75pct_c_optimistic` | double precision | YES |
| `ess_frac` | double precision | YES |
| `has_csf` | boolean | YES |
| `pct_loss_per_yr_median` | double precision | YES |

### `mechanistic.block5_loo_combined_1065` — 644 rows, 7 columns

| Column | Type | Nullable |
|---|---|---|
| `patno` | bigint | YES |
| `n_train` | bigint | YES |
| `wave` | text | YES |
| `covered_95` | boolean | YES |
| `covered_50` | boolean | YES |
| `z_score` | double precision | YES |
| `has_csf` | boolean | YES |

### `mechanistic.dat_spect_longitudinal` — 3,109 rows, 10 columns

| Column | Type | Nullable |
|---|---|---|
| `patno` | bigint | YES |
| `t_years` | double precision | YES |
| `sbr_caudate_mean` | double precision | YES |
| `sbr_putamen_mean` | double precision | YES |
| `lrrk2` | bigint | YES |
| `gba` | bigint | YES |
| `baseline_age` | double precision | YES |
| `n_visits` | bigint | YES |
| `nsd_iss_stage` | double precision | YES |
| `wave` | text | YES |

### `mechanistic.dat_spect_phase3_calibration` — 2,267 rows, 17 columns

| Column | Type | Nullable |
|---|---|---|
| `patno` | bigint | YES |
| `t_years` | double precision | YES |
| `sbr_caudate_r` | double precision | YES |
| `sbr_caudate_l` | double precision | YES |
| `sbr_putamen_r` | double precision | YES |
| `sbr_putamen_l` | double precision | YES |
| `sbr_putamen_r_ant` | double precision | YES |
| `sbr_putamen_l_ant` | double precision | YES |
| `sbr_caudate_mean` | double precision | YES |
| `sbr_putamen_mean` | double precision | YES |
| `sbr_caudate_putamen_ratio` | double precision | YES |
| `lrrk2` | bigint | YES |
| `gba` | bigint | YES |
| `baseline_age` | double precision | YES |
| `n_visits` | bigint | YES |
| `nsd_iss_stage` | double precision | YES |
| `wave` | text | YES |

### `mechanistic.dat_spect_regional` — 3,109 rows, 17 columns

| Column | Type | Nullable |
|---|---|---|
| `patno` | bigint | YES |
| `t_years` | double precision | YES |
| `sbr_caudate_r` | double precision | YES |
| `sbr_caudate_l` | double precision | YES |
| `sbr_putamen_r` | double precision | YES |
| `sbr_putamen_l` | double precision | YES |
| `sbr_putamen_r_ant` | double precision | YES |
| `sbr_putamen_l_ant` | double precision | YES |
| `sbr_caudate_mean` | double precision | YES |
| `sbr_putamen_mean` | double precision | YES |
| `sbr_caudate_putamen_ratio` | double precision | YES |
| `lrrk2` | bigint | YES |
| `gba` | bigint | YES |
| `baseline_age` | double precision | YES |
| `n_visits` | bigint | YES |
| `nsd_iss_stage` | double precision | YES |
| `wave` | text | YES |

### `mechanistic.multi_observable_inventory` — 1,065 rows, 28 columns

| Column | Type | Nullable |
|---|---|---|
| `patno` | bigint | YES |
| `csf_asyn_median` | double precision | YES |
| `csf_asyn_mean` | double precision | YES |
| `csf_asyn_sd` | double precision | YES |
| `csf_asyn_n` | double precision | YES |
| `saa_sd50_median` | double precision | YES |
| `saa_sd50_n` | double precision | YES |
| `syntap_result` | text | YES |
| `saa_ttt_120` | double precision | YES |
| `saa_ttt_150` | double precision | YES |
| `saa_ttt_1400` | double precision | YES |
| `saa_ttt_1800` | double precision | YES |
| `saa_ttt_11600` | double precision | YES |
| `asyn_agg_frac_median` | double precision | YES |
| `asyn_agg_frac_n` | double precision | YES |
| `asyn_surf_frac_median` | double precision | YES |
| `nev_asyn_median` | double precision | YES |
| `nev_asyn_n` | double precision | YES |
| `nfl_median` | double precision | YES |
| `nfl_mean` | double precision | YES |
| `nfl_n` | double precision | YES |
| `has_csf` | boolean | YES |
| `has_saa_sd50` | boolean | YES |
| `has_saa_dilution` | boolean | YES |
| `has_asyn_agg` | boolean | YES |
| `has_nev` | boolean | YES |
| `has_nfl` | boolean | YES |
| `n_observables` | bigint | YES |

### `mechanistic.path1_its_hierarchical_summary` — 304 rows, 8 columns

| Column | Type | Nullable |
|---|---|---|
| `patno` | bigint | YES |
| `logk_mean` | double precision | YES |
| `loga_mean` | double precision | YES |
| `logk_sd` | double precision | YES |
| `loga_sd` | double precision | YES |
| `cor` | double precision | YES |
| `kn_median` | double precision | YES |
| `kn_ratio` | double precision | YES |

### `mechanistic.path23_comparison` — 70 rows, 13 columns

| Column | Type | Nullable |
|---|---|---|
| `patno` | bigint | YES |
| `ttt_obs` | double precision | YES |
| `saa_positive` | boolean | YES |
| `kn_a` | double precision | YES |
| `cor_a` | double precision | YES |
| `kn_ratio_a` | double precision | YES |
| `kn_b` | double precision | YES |
| `cor_b` | double precision | YES |
| `kn_ratio_b` | double precision | YES |
| `kn_c` | double precision | YES |
| `cor_c` | double precision | YES |
| `kn_ratio_c` | double precision | YES |
| `pct_yr_c` | double precision | YES |

### `mechanistic.phase25_saa_joint_summary` — 119 rows, 9 columns

| Column | Type | Nullable |
|---|---|---|
| `patno` | bigint | YES |
| `ess_frac` | double precision | YES |
| `has_csf` | boolean | YES |
| `saa_positive` | boolean | YES |
| `ttt_obs` | double precision | YES |
| `cor_logk_logalpha` | double precision | YES |
| `log_k_n_sd_prior_ratio` | double precision | YES |
| `log_alpha_tox_sd_prior_ratio` | double precision | YES |
| `pct_loss_per_yr_median` | double precision | YES |

### `mechanistic.phase4_assembled_data` — 22,270 rows, 27 columns

| Column | Type | Nullable |
|---|---|---|
| `patno` | text | YES |
| `event_id` | text | YES |
| `updrs3_off` | double precision | YES |
| `visit_dt` | timestamp without time zone | YES |
| `pdstate` | text | YES |
| `pdmedyn` | double precision | YES |
| `np4off` | double precision | YES |
| `np4wdysk` | double precision | YES |
| `np4tot` | double precision | YES |
| `months_from_baseline` | double precision | YES |
| `nsd_stage_numeric` | double precision | YES |
| `cohort` | text | YES |
| `years_from_baseline` | double precision | YES |
| `t_tox_median` | double precision | YES |
| `t_tox_q025` | double precision | YES |
| `t_tox_q975` | double precision | YES |
| `pct_loss_per_yr_median` | double precision | YES |
| `pct_loss_per_yr_q025` | double precision | YES |
| `pct_loss_per_yr_q975` | double precision | YES |
| `n_scans` | double precision | YES |
| `wave` | text | YES |
| `n_frac` | double precision | YES |
| `ledd_total` | double precision | YES |
| `n_meds_active` | bigint | YES |
| `wearing_off_any` | double precision | YES |
| `wearing_off_moderate` | double precision | YES |
| `has_posterior` | bigint | YES |

### `mechanistic.saa_kinetics_extracted` — 119 rows, 6 columns

| Column | Type | Nullable |
|---|---|---|
| `patno` | bigint | YES |
| `median_ttt` | double precision | YES |
| `median_fmax` | double precision | YES |
| `saa_positive` | boolean | YES |
| `in_dat_spect` | boolean | YES |
| `has_csf` | boolean | YES |

### `mechanistic.step_2_7_profile_likelihood` — 304 rows, 15 columns

| Column | Type | Nullable |
|---|---|---|
| `patno` | bigint | YES |
| `n_scans` | bigint | YES |
| `max_rhat` | double precision | YES |
| `converged` | boolean | YES |
| `n_samples` | bigint | YES |
| `k_n_ci_decades` | double precision | YES |
| `alpha_tox_ci_decades` | double precision | YES |
| `t_tox_ci_decades` | double precision | YES |
| `log_kn_alpha_correlation` | double precision | YES |
| `t_tox_median_hr` | double precision | YES |
| `t_tox_mean_hr` | double precision | YES |
| `pct_loss_per_yr_median` | double precision | YES |
| `pct_loss_per_yr_mean` | double precision | YES |
| `stiff_minus_sloppy_decades` | double precision | YES |
| `t_tox_informative` | boolean | YES |

### `mechanistic.step_2_7_v4_profile_likelihood_is` — 304 rows, 10 columns

| Column | Type | Nullable |
|---|---|---|
| `patno` | bigint | YES |
| `n_scans` | bigint | YES |
| `ess_frac` | double precision | YES |
| `k_n_ci_decades` | double precision | YES |
| `alpha_tox_ci_decades` | double precision | YES |
| `t_tox_ci_decades` | double precision | YES |
| `stiff_minus_sloppy_decades` | double precision | YES |
| `log_kn_alpha_correlation` | double precision | YES |
| `t_tox_median_hr` | double precision | YES |
| `pct_loss_per_yr_median` | double precision | YES |

### `mechanistic.step_2_7_v5_profile_likelihood_is` — 304 rows, 10 columns

| Column | Type | Nullable |
|---|---|---|
| `patno` | bigint | YES |
| `n_scans` | bigint | YES |
| `ess_frac` | double precision | YES |
| `k_n_ci_decades` | double precision | YES |
| `alpha_tox_ci_decades` | double precision | YES |
| `t_tox_ci_decades` | double precision | YES |
| `stiff_minus_sloppy_decades` | double precision | YES |
| `log_kn_alpha_correlation` | double precision | YES |
| `t_tox_median_hr` | double precision | YES |
| `pct_loss_per_yr_median` | double precision | YES |

### `mechanistic.step_2_8_v4_ppc` — 304 rows, 15 columns

| Column | Type | Nullable |
|---|---|---|
| `patno` | bigint | YES |
| `n_scans` | bigint | YES |
| `phase2_scans_covered` | bigint | YES |
| `phase2_scans_total` | bigint | YES |
| `phase2_patient_coverage` | double precision | YES |
| `phase2_scans_covered_50` | bigint | YES |
| `phase2_patient_coverage_50` | double precision | YES |
| `phase2_zratio` | double precision | YES |
| `ess_frac` | double precision | YES |
| `phase1_scans_covered` | bigint | YES |
| `phase1_scans_total` | bigint | YES |
| `phase1_patient_coverage` | double precision | YES |
| `phase1_scans_covered_50` | bigint | YES |
| `phase1_patient_coverage_50` | double precision | YES |
| `phase1_zratio` | double precision | YES |

### `mechanistic.step_2_8_v5_loo_forward` — 304 rows, 11 columns

| Column | Type | Nullable |
|---|---|---|
| `patno` | bigint | YES |
| `n_train_scans` | bigint | YES |
| `t_test_yr` | double precision | YES |
| `sbr_test_obs` | double precision | YES |
| `sbr_pred_median` | double precision | YES |
| `pi_025` | double precision | YES |
| `pi_975` | double precision | YES |
| `covered_95` | boolean | YES |
| `covered_50` | boolean | YES |
| `z_score` | double precision | YES |
| `has_csf` | boolean | YES |

### `mechanistic.step_2_8_v5_ppc` — 304 rows, 15 columns

| Column | Type | Nullable |
|---|---|---|
| `patno` | bigint | YES |
| `n_scans` | bigint | YES |
| `phase2_scans_covered` | bigint | YES |
| `phase2_scans_total` | bigint | YES |
| `phase2_patient_coverage` | double precision | YES |
| `phase2_scans_covered_50` | bigint | YES |
| `phase2_patient_coverage_50` | double precision | YES |
| `phase2_zratio` | double precision | YES |
| `ess_frac` | double precision | YES |
| `phase1_scans_covered` | bigint | YES |
| `phase1_scans_total` | bigint | YES |
| `phase1_patient_coverage` | double precision | YES |
| `phase1_scans_covered_50` | bigint | YES |
| `phase1_patient_coverage_50` | double precision | YES |
| `phase1_zratio` | double precision | YES |

### `mechanistic.step_2_9_s1_per_stage` — 6 rows, 7 columns

| Column | Type | Nullable |
|---|---|---|
| `baseline_stage` | text | YES |
| `n` | bigint | YES |
| `t_tox_median` | double precision | YES |
| `t_tox_mean` | double precision | YES |
| `pct_yr_median` | double precision | YES |
| `sojourn_years` | double precision | YES |
| `label` | text | YES |

### `mechanistic.step_2_9_v2_s1_alt_per_stage` — 7 rows, 7 columns

| Column | Type | Nullable |
|---|---|---|
| `stage_baseline` | double precision | YES |
| `n` | bigint | YES |
| `t_tox_median` | double precision | YES |
| `t_tox_iqr_lo` | double precision | YES |
| `t_tox_iqr_hi` | double precision | YES |
| `pct_yr_median` | double precision | YES |
| `subset` | text | YES |

### `mechanistic.track3_sensitivity_oat` — 26 rows, 6 columns

| Column | Type | Nullable |
|---|---|---|
| `label` | text | YES |
| `n` | bigint | YES |
| `cor_median` | double precision | YES |
| `kn_ratio_median` | double precision | YES |
| `pct_yr_median` | double precision | YES |
| `ess_median` | double precision | YES |

---

**Totals:** 146 tables, 1,233,732 rows.

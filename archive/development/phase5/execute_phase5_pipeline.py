#!/usr/bin/env python3
"""Phase 5 master pipeline executor (patched for current task APIs)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


class Phase5PipelineExecutor:
    """Orchestrate Task 5.1 -> 5.6 with current module interfaces."""

    def __init__(self, ppmi_data_path: str, output_dir: str = "results/phase5_execution"):
        self.ppmi_data_path = Path(ppmi_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.phase5_dir = Path(__file__).resolve().parent
        self.task_output_dir = self.output_dir / "task_outputs"
        self.task_output_dir.mkdir(parents=True, exist_ok=True)

        if str(self.phase5_dir) not in sys.path:
            sys.path.insert(0, str(self.phase5_dir))

        self.logger = self._setup_logging()
        self.results: dict = {
            "metadata": {
                "start_time": None,
                "end_time": None,
                "runtime_minutes": None,
                "ppmi_data_path": str(self.ppmi_data_path),
                "output_dir": str(self.output_dir),
            },
            "tasks": {},
        }

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("phase5_executor")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.output_dir / f"phase5_pipeline_{ts}.log"

        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(sh)
        return logger

    def _read_json_if_exists(self, path: Path) -> dict:
        if not path.exists():
            return {}
        with open(path) as f:
            return json.load(f)

    def run_task_5_1(self) -> tuple[Path, Path]:
        from task_5_1_prodromal_cohort_identification import ProdromalCohortIdentification

        out = self.task_output_dir
        identifier = ProdromalCohortIdentification(
            prodromal_cohort_path=str(self.ppmi_data_path),
            output_dir=str(out),
        )
        survival_df, cohort_stats = identifier.run_full_pipeline()

        survival_path = out / "prodromal_survival_data.csv"
        report_path = out / "prodromal_cohort_report.json"
        self.results["tasks"]["5.1"] = {
            "status": "completed",
            "n_rows": int(len(survival_df)),
            "n_converted": int(cohort_stats.get("phenoconversion", {}).get("n_converted", 0)),
            "survival_data": str(survival_path),
            "report": str(report_path),
        }
        return survival_path, report_path

    def run_task_5_2(self, survival_path: Path) -> Path:
        from task_5_2_time_varying_biomarkers import TimeVaryingBiomarkerExtraction

        out = self.task_output_dir
        extractor = TimeVaryingBiomarkerExtraction(
            survival_data_path=str(survival_path),
            prodromal_cohort_path=str(self.ppmi_data_path),
            output_dir=str(out),
        )
        tv_df = extractor.run_full_pipeline()

        tv_path = out / "time_varying_biomarkers.csv"
        self.results["tasks"]["5.2"] = {
            "status": "completed",
            "n_rows": int(len(tv_df)),
            "n_patients": int(tv_df["PATNO"].nunique()) if "PATNO" in tv_df.columns else 0,
            "time_varying_data": str(tv_path),
        }
        return tv_path

    def run_task_5_3(self, survival_path: Path, tv_path: Path) -> Path:
        from task_5_3_cox_proportional_hazards import CoxProportionalHazardsAnalysis

        out = self.task_output_dir
        analyzer = CoxProportionalHazardsAnalysis(
            survival_data_path=str(survival_path),
            time_varying_data_path=str(tv_path),
            output_dir=str(out),
        )
        analyzer.run_complete_analysis()

        cox_path = out / "cox_model_results.json"
        cox_data = self._read_json_if_exists(cox_path)
        self.results["tasks"]["5.3"] = {
            "status": "completed",
            "cox_results": str(cox_path),
            "baseline_c_index": cox_data.get("baseline_model", {}).get("concordance_index"),
            "time_varying_c_index": cox_data.get("time_varying_model", {}).get("concordance_index"),
        }
        return cox_path

    def run_task_5_4(self, survival_path: Path, tv_path: Path) -> Path:
        from task_5_4_deepsurv_neural_survival import DeepSurvAnalysis

        out = self.task_output_dir
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        analyzer = DeepSurvAnalysis(
            survival_data_path=str(survival_path),
            time_varying_data_path=str(tv_path),
            output_dir=str(out),
            device=device,
        )
        analyzer.run_complete_analysis()

        model_path = out / "deepsurv_model.pth"
        results_path = out / "deepsurv_results.json"
        ds_data = self._read_json_if_exists(results_path)
        self.results["tasks"]["5.4"] = {
            "status": "completed",
            "deepsurv_model": str(model_path),
            "deepsurv_results": str(results_path),
            "c_index": ds_data.get("model_performance", {}).get("c_index"),
        }
        return model_path

    def run_task_5_5(self, survival_path: Path, tv_path: Path) -> Path:
        from task_5_5_biomarker_thresholds import BiomarkerThresholdAnalysis

        out = self.task_output_dir
        analyzer = BiomarkerThresholdAnalysis(
            survival_data_path=str(survival_path),
            time_varying_data_path=str(tv_path),
            output_dir=str(out),
        )
        analyzer.run_complete_analysis()

        thresh_path = out / "biomarker_thresholds.json"
        threshold_data = self._read_json_if_exists(thresh_path)
        self.results["tasks"]["5.5"] = {
            "status": "completed",
            "thresholds": str(thresh_path),
            "n_threshold_groups": int(len(threshold_data.get("thresholds", {}))) if threshold_data else 0,
        }
        return thresh_path

    def run_task_5_6(self) -> Path:
        from task_5_6_risk_stratification_tool import (
            PhenoconversionRiskCalculator,
            RiskStratificationDashboard,
        )

        out = self.task_output_dir

        calculator = PhenoconversionRiskCalculator(models_dir=str(out))
        survival_df = pd.read_csv(out / "prodromal_survival_data.csv")
        tv_df = pd.read_csv(out / "time_varying_biomarkers.csv")

        slope_df = tv_df.groupby("PATNO")[["UPDRS_III_slope"]].last().reset_index()
        cohort_df = survival_df.merge(slope_df, on="PATNO", how="left")
        cohort_df["UPDRS_III_slope"] = cohort_df["UPDRS_III_slope"].fillna(0)

        if "baseline_updrs" in cohort_df.columns:
            cohort_df["baseline_updrs"] = cohort_df["baseline_updrs"].fillna(cohort_df["baseline_updrs"].median())
        if "baseline_moca" in cohort_df.columns:
            cohort_df["baseline_moca"] = cohort_df["baseline_moca"].fillna(cohort_df["baseline_moca"].median())

        scored = calculator.batch_assess_cohort(cohort_df)
        scored_path = out / "cohort_risk_stratification.csv"
        scored.to_csv(scored_path, index=False)

        dashboard = RiskStratificationDashboard(output_dir=str(out))
        dashboard.create_cohort_dashboard(scored)

        self.results["tasks"]["5.6"] = {
            "status": "completed",
            "risk_csv": str(scored_path),
            "n_scored": int(len(scored)),
            "risk_category_counts": scored["risk_category"].value_counts().to_dict()
            if "risk_category" in scored.columns
            else {},
        }
        return scored_path

    def _record_failure(self, task_id: str, exc: Exception):
        self.results["tasks"][task_id] = {
            "status": "failed",
            "error": str(exc),
        }

    def execute(self) -> dict:
        start = datetime.now()
        self.results["metadata"]["start_time"] = start.isoformat()

        self.logger.info("Starting Phase 5 pipeline execution")

        survival_path = None
        tv_path = None

        try:
            self.logger.info("Task 5.1")
            survival_path, _ = self.run_task_5_1()

            self.logger.info("Task 5.2")
            tv_path = self.run_task_5_2(survival_path)
        except Exception as exc:
            self.logger.error("Phase 5 failed in required setup tasks (5.1/5.2)", exc_info=True)
            failed_task = "5.1" if "5.1" not in self.results["tasks"] else "5.2"
            self._record_failure(failed_task, exc)
            raise

        # Non-blocking downstream tasks: keep going to maximize usable outputs.
        try:
            self.logger.info("Task 5.3")
            self.run_task_5_3(survival_path, tv_path)
        except Exception as exc:
            self.logger.warning("Task 5.3 failed; continuing with 5.4-5.6", exc_info=True)
            self._record_failure("5.3", exc)

        try:
            self.logger.info("Task 5.4")
            self.run_task_5_4(survival_path, tv_path)
        except Exception as exc:
            self.logger.warning("Task 5.4 failed; continuing with remaining tasks", exc_info=True)
            self._record_failure("5.4", exc)

        try:
            self.logger.info("Task 5.5")
            self.run_task_5_5(survival_path, tv_path)
        except Exception as exc:
            self.logger.warning("Task 5.5 failed; attempting task 5.6 if inputs exist", exc_info=True)
            self._record_failure("5.5", exc)

        try:
            self.logger.info("Task 5.6")
            self.run_task_5_6()
        except Exception as exc:
            self.logger.warning("Task 5.6 failed", exc_info=True)
            self._record_failure("5.6", exc)

        finally:
            end = datetime.now()
            self.results["metadata"]["end_time"] = end.isoformat()
            self.results["metadata"]["runtime_minutes"] = (end - start).total_seconds() / 60.0
            with open(self.output_dir / "phase5_master_results.json", "w") as f:
                json.dump(self.results, f, indent=2)

        self.logger.info("Phase 5 pipeline finished")
        return self.results


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute Phase 5 pipeline (patched)")
    parser.add_argument("--ppmi-data", required=True, help="Path to prodromal cohort CSV")
    parser.add_argument(
        "--output-dir", default="results/phase5_execution", help="Output directory"
    )
    args = parser.parse_args()

    executor = Phase5PipelineExecutor(ppmi_data_path=args.ppmi_data, output_dir=args.output_dir)

    try:
        results = executor.execute()
        print("\nPhase 5 completed")
        for task_id, payload in results.get("tasks", {}).items():
            print(f"- Task {task_id}: {payload.get('status')}")
        print(f"Master results: {Path(args.output_dir) / 'phase5_master_results.json'}")
        return 0
    except Exception as exc:
        print(f"\nPhase 5 failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

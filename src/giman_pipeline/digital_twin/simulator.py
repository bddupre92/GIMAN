from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from .state import CounterfactualSpec, TwinSimulationResult, TwinState


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_models(in_features: int, device: torch.device):
    root = _repo_root()
    phase8_dir = root / "archive" / "development" / "phase8" / "subphase8_2_dynamic_endpoints"
    if str(phase8_dir) not in sys.path:
        sys.path.append(str(phase8_dir))
    if str(root) not in sys.path:
        sys.path.append(str(root))

    from train_final_giman_survival import GIMANSurvivalGAT
    from archive.development.phase9.neuro_fuzzy import NeuroFuzzyGIMAN

    phase8_ckpt = root / "outputs" / "phase8_2_final_training_sota_run" / "giman_survival_final.pth"
    phase9_ckpt = root / "outputs" / "phase9_neuro_fuzzy_sota_run_from50ckpt" / "neuro_fuzzy_best.pth"

    surv = GIMANSurvivalGAT(in_features=in_features, hidden_dim=128).to(device)
    surv_state = torch.load(phase8_ckpt, map_location=device, weights_only=False)
    surv.load_state_dict(surv_state["model_state_dict"])
    surv.eval()

    gat = GIMANSurvivalGAT(in_features=in_features, hidden_dim=128)
    nf = NeuroFuzzyGIMAN(gat, num_classes=2, num_rules=32).to(device)
    nf.load_state_dict(torch.load(phase9_ckpt, map_location=device, weights_only=False))
    nf.eval()
    return surv, nf


class DataDrivenTwinSimulator:
    def __init__(self, data_path: Path, metadata_path: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = torch.load(data_path, weights_only=False).to(self.device)
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.feature_names: list[str] = self.metadata.get("feature_names", [])
        self.survival_model, self.neuro_fuzzy_model = _load_models(
            in_features=int(self.data.x.shape[1]),
            device=self.device,
        )

    def _predict(self, x_override: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        d = self.data.clone()
        if x_override is not None:
            d.x = torch.tensor(x_override, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            risk = self.survival_model(d).detach().cpu().numpy()
            logits, _ = self.neuro_fuzzy_model(d)
            saa = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        return risk, saa

    def simulate_patient(
        self,
        patient_idx: int,
        horizons: list[int] | None = None,
    ) -> list[TwinState]:
        if horizons is None:
            horizons = [0, 6, 12, 18, 24]

        x_np = self.data.x.detach().cpu().numpy()
        patno = int(self.data.patno[patient_idx].detach().cpu().item()) if hasattr(self.data, "patno") else int(patient_idx)

        risk_all, saa_all = self._predict(x_override=x_np)
        risk0 = float(risk_all[patient_idx])
        saa0 = float(saa_all[patient_idx])

        states: list[TwinState] = []
        for h in horizons:
            # Data-driven v1 approximation: monotone horizon scaling of risk
            h_scale = 1.0 + (h / 24.0) * 0.25
            risk_h = float(risk0 * h_scale)
            saa_h = float(np.clip(saa0 + (h / 24.0) * 0.05 * (saa0 - 0.5), 0.0, 1.0))
            unc = max(0.03, 0.12 * saa_h)
            states.append(
                TwinState(
                    patno=patno,
                    t_month=int(h),
                    feature_vector=x_np[patient_idx].tolist(),
                    risk_survival=risk_h,
                    risk_saa=saa_h,
                    uncertainty_low=float(max(0.0, saa_h - unc)),
                    uncertainty_high=float(min(1.0, saa_h + unc)),
                )
            )
        return states

    def simulate_counterfactual(
        self,
        patient_idx: int,
        specs: list[CounterfactualSpec],
        horizons: list[int] | None = None,
    ) -> TwinSimulationResult:
        baseline = self.simulate_patient(patient_idx, horizons)

        x_np = self.data.x.detach().cpu().numpy()
        base_vector = x_np[patient_idx].copy()

        cf_paths: dict[str, list[TwinState]] = {}
        delta_risk: dict[str, float] = {}
        ci_map: dict[str, tuple[float, float]] = {}
        attribution: dict[str, dict[str, float]] = {}

        for spec in specs:
            if spec.feature_name not in self.feature_names:
                continue
            idx = self.feature_names.index(spec.feature_name)
            cf_vector = base_vector.copy()
            cf_vector[idx] = cf_vector[idx] + spec.delta
            if spec.bounds is not None:
                cf_vector[idx] = float(np.clip(cf_vector[idx], spec.bounds[0], spec.bounds[1]))

            x_cf = x_np.copy()
            x_cf[patient_idx] = cf_vector
            risk_all, saa_all = self._predict(x_override=x_cf)
            risk0 = float(risk_all[patient_idx])
            saa0 = float(saa_all[patient_idx])

            key = f"{spec.feature_name}:{spec.delta:+.3f}"
            states: list[TwinState] = []
            for h in (horizons or [0, 6, 12, 18, 24]):
                h_scale = 1.0 + (h / 24.0) * 0.25
                risk_h = float(risk0 * h_scale)
                saa_h = float(np.clip(saa0 + (h / 24.0) * 0.05 * (saa0 - 0.5), 0.0, 1.0))
                unc = max(0.03, 0.12 * saa_h)
                states.append(
                    TwinState(
                        patno=int(self.data.patno[patient_idx].detach().cpu().item()) if hasattr(self.data, "patno") else int(patient_idx),
                        t_month=int(h),
                        feature_vector=cf_vector.tolist(),
                        risk_survival=risk_h,
                        risk_saa=saa_h,
                        uncertainty_low=float(max(0.0, saa_h - unc)),
                        uncertainty_high=float(min(1.0, saa_h + unc)),
                    )
                )

            cf_paths[key] = states
            delta_risk[key] = float(states[-1].risk_saa - baseline[-1].risk_saa)
            ci_map[key] = (states[-1].uncertainty_low, states[-1].uncertainty_high)
            attribution[key] = {
                "feature_index": float(idx),
                "feature_delta": float(spec.delta),
            }

        return TwinSimulationResult(
            baseline_path=baseline,
            counterfactual_paths=cf_paths,
            delta_risk=delta_risk,
            confidence_interval=ci_map,
            attribution=attribution,
        )


def save_simulation_result(result: TwinSimulationResult, output_path: Path) -> None:
    payload = {
        "baseline_path": [asdict(x) for x in result.baseline_path],
        "counterfactual_paths": {
            k: [asdict(v) for v in vals] for k, vals in result.counterfactual_paths.items()
        },
        "delta_risk": result.delta_risk,
        "confidence_interval": {
            k: [v[0], v[1]] for k, v in result.confidence_interval.items()
        },
        "attribution": result.attribution,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    root = _repo_root()
    sim = DataDrivenTwinSimulator(
        data_path=root / "data" / "03_prodromal" / "final_pyg_data_sota_run" / "test_data.pt",
        metadata_path=root / "data" / "03_prodromal" / "final_pyg_data_sota_run" / "pyg_data_metadata.json",
    )
    result = sim.simulate_counterfactual(
        patient_idx=0,
        specs=[CounterfactualSpec(feature_name="UPDRS_I", delta=-0.5)],
    )
    save_simulation_result(result, root / "outputs" / "digital_twin" / "patient_0_twin.json")

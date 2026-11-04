"""Связывает расширенные проверки в один воспроизводимый прогон поверх engine этапа 5."""
from __future__ import annotations

import math
from dataclasses import dataclass, fields, is_dataclass

import numpy as np
import pandas as pd

from experimentlab.analysis.bootstrap import (
    bootstrap_conversion_diff,
    bootstrap_conversion_relative_lift,
)
from experimentlab.analysis.decision import FinalDecision, decide
from experimentlab.analysis.engine import AnalysisResult, run_analysis
from experimentlab.analysis.experiment_quality import QualityReport, assess_experiment_quality
from experimentlab.analysis.guardrails import evaluate_guardrails
from experimentlab.analysis.segments import SegmentAnalysis, analyze_segments, significant_levels
from experimentlab.analysis.sensitivity import SensitivityResult, run_sensitivity
from experimentlab.analysis.srm import SRMReport, srm_by_segment, srm_report


def to_jsonable(o):
    """Рекурсивно приводит результаты к типам, допустимым в JSON."""
    if is_dataclass(o) and not isinstance(o, type):
        return {f.name: to_jsonable(getattr(o, f.name)) for f in fields(o)}
    if isinstance(o, pd.DataFrame):
        return to_jsonable(o.to_dict(orient="records"))
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return None if math.isnan(v) or math.isinf(v) else v
    if o is None or isinstance(o, str):
        return o
    return str(o)


@dataclass
class AdvancedResult:
    base: AnalysisResult
    srm: SRMReport
    srm_segments: pd.DataFrame
    bootstrap: dict
    segments: SegmentAnalysis
    guardrails: list
    sensitivity: SensitivityResult
    quality: QualityReport
    decision: FinalDecision

    def to_dict(self) -> dict:
        out = {f.name: to_jsonable(getattr(self, f.name)) for f in fields(self) if f.name != "base"}
        out["base"] = to_jsonable(self.base.to_dict())
        return out


def run_advanced(df: pd.DataFrame, cfg: dict, source: str = "") -> AdvancedResult:
    adv, a = cfg["advanced"], cfg["analysis"]
    alpha, seed = float(a["alpha"]), int(a["bootstrap_seed"])
    share = float(cfg["treatment_share"])
    d = df.assign(group_name=df["group_name"].astype(str))

    base = run_analysis(d, cfg, source)
    srm = srm_report(base.meta["n_control"], base.meta["n_treatment"], share, float(a["srm_alpha"]))
    dims = list(adv["segment_dimensions"])
    srm_seg = srm_by_segment(d, dims + ["signup_date"], share, float(adv["srm_segment_alpha"]))

    c, nb = base.conversion, int(adv["n_bootstrap_conversion"])
    boot = {
        "conversion": bootstrap_conversion_diff(
            c.x_control, c.n_control, c.x_treatment, c.n_treatment, nb, alpha, seed),
        "conversion_relative_lift": bootstrap_conversion_relative_lift(
            c.x_control, c.n_control, c.x_treatment, c.n_treatment, nb, alpha, seed),
        "arpu": base.arpu_bootstrap,  # тот же bootstrap, что и в этапе 5
    }

    segs = analyze_segments(d, dims, alpha, int(adv["segment_min_users"]), float(adv["fdr_q"]))
    guards = evaluate_guardrails(d, adv["guardrails"], float(adv["guardrail_alpha"]))
    sens = run_sensitivity(d, alpha, int(a["n_bootstrap"]), nb, seed, list(adv["winsor_percentiles"]))
    quality = assess_experiment_quality(d, cfg, base, srm)

    harm_holm = significant_levels(segs.table, "holm", "negative")
    harm_bh = [s for s in significant_levels(segs.table, "bh", "negative") if s not in harm_holm]
    decision = decide(
        srm_passed=srm.passed, quality_findings=quality.findings,
        conversion=base.conversion_verdict, guardrails=guards,
        conversion_robust=sens.conversion_robust, segment_harm_holm=harm_holm,
        segment_harm_bh=harm_bh, data_sufficient=base.sufficiency["data_sufficient"],
        criteria=adv.get("success_criteria"))
    return AdvancedResult(base, srm, srm_seg, boot, segs, guards, sens, quality, decision)

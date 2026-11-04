import copy
import hashlib
import json

import pandas as pd

from experimentlab.analysis.engine import run_analysis
from experimentlab.config import load_config
from experimentlab.data.generator import config_fingerprint, generate_users, run_generation


def _cfg(n=5000):
    cfg = load_config()
    cfg["n_users"] = n
    cfg["analysis"]["n_bootstrap"] = 100
    return cfg


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generation_is_byte_identical_across_runs(tmp_path):
    cfg = _cfg()
    for name in ("a", "b"):
        run_generation(cfg, tmp_path / f"{name}.csv", tmp_path / f"{name}.json")
    assert _sha(tmp_path / "a.csv") == _sha(tmp_path / "b.csv")
    assert (tmp_path / "a.json").read_text() == (tmp_path / "b.json").read_text()


def test_fingerprint_reacts_to_every_generation_parameter():
    cfg = _cfg()
    base = config_fingerprint(cfg)
    assert config_fingerprint(copy.deepcopy(cfg)) == base
    for key, value in (("seed", 1), ("n_users", 6000), ("scenario", "negative"), ("n_days", 14)):
        changed = copy.deepcopy(cfg)
        changed[key] = value
        assert config_fingerprint(changed) != base, key


def test_control_group_is_identical_across_scenarios():
    """Сценарий меняет только treatment: control при том же seed остаётся прежним."""
    cfg = _cfg()
    frames = {}
    for scenario in ("positive", "no_effect", "negative"):
        df, _ = generate_users({**cfg, "scenario": scenario})
        frames[scenario] = df[df["group"] == "control"].reset_index(drop=True)
    pd.testing.assert_frame_equal(frames["positive"], frames["no_effect"])
    pd.testing.assert_frame_equal(frames["positive"], frames["negative"])


def test_analysis_output_is_deterministic():
    from experimentlab.analysis import eda

    cfg = _cfg(20_000)
    df, _ = generate_users(cfg)
    frame = eda._derive_from_csv(df)
    first = json.dumps(run_analysis(frame, cfg).to_dict(), default=str, sort_keys=True)
    second = json.dumps(run_analysis(frame, cfg).to_dict(), default=str, sort_keys=True)
    assert first == second

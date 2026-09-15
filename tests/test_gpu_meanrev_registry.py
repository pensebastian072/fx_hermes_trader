"""Registry discipline: pre-registration required, ledger drives n_trials.

Same sandbox-fixture pattern as alpaca_gpu_lab/tests/test_registry.py.
"""
import pytest

from gpu_meanrev.experiments import registry


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "REGISTERED", tmp_path / "registered")
    monkeypatch.setattr(registry, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(registry, "RESULTS_MD", tmp_path / "RESULTS.md")
    return tmp_path


def test_unregistered_battery_raises(sandbox):
    with pytest.raises(registry.UnregisteredBattery):
        registry.load_registration("B99_never_registered")


def test_register_then_load(sandbox):
    registry.register("B01_test", "reversion happens", "fx_1min_meanrev",
                      ["EURUSD"], {"entry_z": [1.5]}, n_trials=1)
    rec = registry.load_registration("B01_test")
    assert rec["hypothesis"]
    assert rec["n_trials"] == 1


def test_registrations_append_only(sandbox):
    registry.register("B02_once", "h", "fx_1min_meanrev", ["EURUSD"], {}, 1)
    with pytest.raises(FileExistsError):
        registry.register("B02_once", "changed my mind", "fx_1min_meanrev",
                          ["EURUSD"], {}, 1)


def test_ledger_drives_n_trials(sandbox):
    fam = "fx_1min_meanrev"
    assert registry.n_trials_for(fam) == 1  # empty ledger, seed 0 -> floor 1
    for i in range(3):
        registry.log_trial("B01_test", fam, {"entry_z": 1.5 + i}, ["EURUSD"])
    assert registry.n_trials_for(fam) == 3


def test_fx_family_seeds_at_zero(sandbox):
    # genuinely new ground -- no prior 1-minute FX price-action battery on this box
    assert registry.FAMILY_SEEDS["fx_1min_meanrev"] == 0


def test_results_row_appends(sandbox):
    verdict = {"passes": False, "n_trades": 120, "profit_factor": 0.97,
               "sharpe": -0.02, "pbo": 0.55,
               "deflated_sharpe": {"ratio": -3.1, "n_trials": 6}}
    registry.append_result_row("B01_test", verdict, note="baseline")
    text = registry.RESULTS_MD.read_text(encoding="utf-8")
    assert "B01_test" in text and "FAIL" in text

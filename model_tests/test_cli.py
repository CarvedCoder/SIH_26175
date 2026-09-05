"""Tests for the unified CLI: registry integrity + lazy dispatch.

Guards the refactor's core promises:
  * every registered command's module matches the registry NAME/HELP
  * model.py builds help WITHOUT importing torch (lazy design)
  * each command's --help parses (imports only that module)
  * unknown commands fail cleanly
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from depthwizard.cli.registry import COMMANDS, SPECS, main


def test_registry_matches_command_modules():
    """Registry table and command modules must never drift apart."""
    for mod_name, name, help_text in SPECS:
        mod = __import__(f"depthwizard.cli.{mod_name}", fromlist=["run"])
        assert mod.NAME == name, f"{mod_name}.NAME={mod.NAME} != registry {name}"
        assert mod.HELP == help_text, f"{mod_name}.HELP drifted from registry"
        assert callable(mod.run) and callable(mod.add_parser)


def test_skeleton_help_never_imports_torch(monkeypatch):
    """`python model.py --help` must work with torch uninstalled/missing."""
    monkeypatch.setitem(sys.modules, "torch", None)     # poison the import
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code in (0, None)


def test_no_args_prints_help(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "inspect" in out and "evaluate" in out and "serve" in out


def test_command_help_parses_for_every_command():
    for _mod, name, _h in SPECS:
        with pytest.raises(SystemExit) as exc:
            main([name, "--help"])
        assert exc.value.code in (0, None), f"`{name} --help` failed"


def test_unknown_command_exits_with_error():
    with pytest.raises(SystemExit) as exc:
        main(["definitely-not-a-command"])
    assert exc.value.code == 2


def test_main_py_file_dispatch():
    """The physical entry point `python model.py <cmd> --help` works."""
    r = subprocess.run([sys.executable, str(ROOT / "model.py"), "infer", "--help"],
                       capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert r.returncode == 0
    assert "--anchor-dem" in r.stdout


def test_all_commands_cover_old_numbered_scripts():
    """Migration map: every numbered script from the old layout has exactly
    one modular home. If you add a command, update this map."""
    old_to_new = {
        "01_inspect_dataset.py": "inspect",
        "02_make_splits.py": "splits",
        "03_precompute_depth.py": "depth",
        "04_fit_baseline.py": "fit-baseline",
        "05_eval_baseline.py": "eval-baseline",
        "06_dummy_baselines.py": "dummies",
        "07_reference_table.py": "reference",
        "08_train_calibration.py": "train",
        "09_eval_calibration.py": "evaluate",
        "10_infer_single.py": "infer",
        "gt_check.py": "gt-check",
        "diag_rgb_arm.py": "diag",
    }
    for _old, new in old_to_new.items():
        assert new in COMMANDS
    # eval-scene + serve are new capabilities with no old counterpart
    assert {"eval-scene", "serve"} <= set(COMMANDS)

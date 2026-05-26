"""Wave 6.2 — schema + behaviour tests for ``mql5-task-graph-gen``."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from vibecodekit_mql5.step_gen import task_graph_gen as tgg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_CANONICAL_CONTRACT = """\
# Contract for `SmokeTrendEA.mq5`

## DELIVERABLES

- Item 1
- Item 2

## EXCLUSIONS

- Hedging modes
- Multi-symbol baskets

## TECH STACK

- Module diagram cells:
  - `CPipNormalizer`
  - `CRiskGuard`

## INVARIANTS — copied verbatim from BLUEPRINT

- Per-trade risk stays \u2264 0.5% of equity.
- The filter chain blackout-window blocks signals after 20:00.
- The backtest range covers at least 24 months on EURUSD.

## TASK GRAPH SUMMARY

- TIP-001 \u2014 scaffold `SmokeTrendEA.mq5` from `trend/netting` archetype
- TIP-002 \u2014 wire risk guard (0.5% per trade, 2.0% daily-loss cap)
- TIP-003 \u2014 implement signal block on `EURUSD` / `H1`
- TIP-004 \u2014 implement signal `macd` with declared params
- TIP-005 \u2014 implement filter chain (time_window)
- TIP-006 \u2014 produce backtest + walk-forward XML on declared range
- TIP-007 \u2014 pass `mql5-permission --mode personal` gate

## ACCEPTANCE OVERVIEW

- REQ-COVERAGE

## CONFIRM

CONFIRM by smoke at 2025-11-13
"""


@pytest.fixture()
def contract(tmp_path: Path) -> Path:
    p = tmp_path / "contract.md"
    p.write_text(_CANONICAL_CONTRACT, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_contract_extracts_seven_tips(contract: Path) -> None:
    graph = tgg.parse_contract(contract)
    assert len(graph.nodes) == 7
    assert [n.tip_id for n in graph.nodes] == [
        "TIP-001",
        "TIP-002",
        "TIP-003",
        "TIP-004",
        "TIP-005",
        "TIP-006",
        "TIP-007",
    ]


def test_parse_contract_extracts_invariants(contract: Path) -> None:
    graph = tgg.parse_contract(contract)
    assert len(graph.invariants) == 3
    assert "Per-trade risk" in graph.invariants[0]


def test_parse_contract_sha_is_deterministic(contract: Path) -> None:
    g1 = tgg.parse_contract(contract)
    g2 = tgg.parse_contract(contract)
    assert g1.contract_sha256 == g2.contract_sha256
    assert len(g1.contract_sha256) == 64


def test_parse_contract_missing_section_raises(tmp_path: Path) -> None:
    p = tmp_path / "broken.md"
    p.write_text("# header only\n\nno task graph here.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="TASK GRAPH SUMMARY"):
        tgg.parse_contract(p)


def test_parse_contract_empty_section_raises(tmp_path: Path) -> None:
    p = tmp_path / "broken.md"
    p.write_text(
        "# x\n\n## TASK GRAPH SUMMARY\n\n_no bullets_\n\n## CONFIRM\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no `TIP-NNN"):
        tgg.parse_contract(p)


# ---------------------------------------------------------------------------
# Dependency resolution
# ---------------------------------------------------------------------------


def test_root_node_has_no_dependencies(contract: Path) -> None:
    graph = tgg.parse_contract(contract)
    root = next(n for n in graph.nodes if n.tip_id == "TIP-001")
    assert root.depends_on == ()


def test_risk_depends_on_scaffold(contract: Path) -> None:
    graph = tgg.parse_contract(contract)
    risk = next(n for n in graph.nodes if n.tip_id == "TIP-002")
    assert risk.depends_on == ("TIP-001",)


def test_signal_depends_on_risk(contract: Path) -> None:
    graph = tgg.parse_contract(contract)
    sig = next(n for n in graph.nodes if n.tip_id == "TIP-003")
    assert sig.depends_on == ("TIP-002",)


def test_filter_depends_on_all_signals(contract: Path) -> None:
    graph = tgg.parse_contract(contract)
    flt = next(n for n in graph.nodes if n.tip_id == "TIP-005")
    assert set(flt.depends_on) == {"TIP-003", "TIP-004"}


def test_backtest_depends_on_filters_and_signals(contract: Path) -> None:
    graph = tgg.parse_contract(contract)
    bt = next(n for n in graph.nodes if n.tip_id == "TIP-006")
    assert "TIP-005" in bt.depends_on
    assert set(bt.depends_on) >= {"TIP-005", "TIP-003", "TIP-004"}


def test_permission_depends_on_backtest(contract: Path) -> None:
    graph = tgg.parse_contract(contract)
    perm = next(n for n in graph.nodes if n.tip_id == "TIP-007")
    assert perm.depends_on == ("TIP-006",)


def test_dag_has_no_self_loop(contract: Path) -> None:
    graph = tgg.parse_contract(contract)
    for node in graph.nodes:
        assert node.tip_id not in node.depends_on


def test_dag_is_topologically_sortable(contract: Path) -> None:
    """Every dep must point at an earlier TIP index."""

    graph = tgg.parse_contract(contract)
    indices = {n.tip_id: n.index for n in graph.nodes}
    for node in graph.nodes:
        for dep in node.depends_on:
            assert indices[dep] < node.index, (
                f"{node.tip_id} depends on {dep} which comes later"
            )


def test_invariant_refs_match_signal_filter_keywords() -> None:
    invariants = (
        "blackout-window blocks signals after 20:00.",
        "backtest range covers at least 24 months",
    )
    matched = tgg._match_invariants("implement filter chain blackout", invariants)
    assert any("blackout" in inv for inv in matched)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_render_tip_includes_frontmatter(contract: Path) -> None:
    graph = tgg.parse_contract(contract)
    rendered = tgg.render_tip_file(graph.nodes[0], graph)
    assert rendered.startswith("---")
    assert "tip_id: TIP-001" in rendered
    assert "status: PENDING" in rendered
    assert "actor: tho-thi-cong" in rendered


def test_render_tip_with_deps_lists_them(contract: Path) -> None:
    graph = tgg.parse_contract(contract)
    flt = next(n for n in graph.nodes if n.tip_id == "TIP-005")
    rendered = tgg.render_tip_file(flt, graph)
    assert "depends_on: [TIP-003, TIP-004]" in rendered
    assert "## Dependencies" in rendered
    assert "- `TIP-003`" in rendered
    assert "- `TIP-004`" in rendered


def test_render_root_tip_dependencies_section_is_empty(contract: Path) -> None:
    graph = tgg.parse_contract(contract)
    root = graph.nodes[0]
    rendered = tgg.render_tip_file(root, graph)
    assert "_No predecessors" in rendered


def test_render_tip_completion_section_uses_tasks_prefix(contract: Path) -> None:
    """Regression: the suggested ``mql5-completion-report --tip`` invocation
    inside each rendered TIP file must point at ``tasks/TIP-NNN.md`` because
    that is the path ``_write_outputs`` actually writes to. The earlier
    rendering omitted the prefix and produced a "file not found" footgun
    when a Builder copy-pasted the command from the project root.
    """

    graph = tgg.parse_contract(contract)
    rendered = tgg.render_tip_file(graph.nodes[0], graph)
    assert "mql5-completion-report --tip tasks/TIP-001.md " in rendered
    assert "--tip TIP-001.md " not in rendered


def test_render_task_graph_uses_mermaid(contract: Path) -> None:
    graph = tgg.parse_contract(contract)
    rendered = tgg.render_task_graph(graph)
    assert "```mermaid" in rendered
    assert "graph TD" in rendered
    # Every node must appear as a box declaration.
    for node in graph.nodes:
        assert f'    {node.tip_id}["{node.tip_id}' in rendered
    # Every edge must appear.
    for node in graph.nodes:
        for dep in node.depends_on:
            assert f"    {dep} --> {node.tip_id}" in rendered


def test_render_task_graph_has_index_table(contract: Path) -> None:
    graph = tgg.parse_contract(contract)
    rendered = tgg.render_task_graph(graph)
    assert "## Index" in rendered
    assert "| TIP | Description |" in rendered
    for node in graph.nodes:
        assert f"`{node.tip_id}`" in rendered


def test_render_is_deterministic(contract: Path) -> None:
    graph = tgg.parse_contract(contract)
    a = tgg.render_task_graph(graph)
    b = tgg.render_task_graph(graph)
    assert a == b


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "vibecodekit_mql5.step_gen.task_graph_gen", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_writes_tip_files(contract: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    proc = _run_cli([str(contract), "--out-dir", str(out_dir)])
    assert proc.returncode == 0, proc.stderr
    tasks = sorted((out_dir / "tasks").glob("TIP-*.md"))
    assert len(tasks) == 7
    assert (out_dir / "task-graph.md").is_file()


def test_cli_refuses_overwrite(contract: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    first = _run_cli([str(contract), "--out-dir", str(out_dir)])
    assert first.returncode == 0
    second = _run_cli([str(contract), "--out-dir", str(out_dir)])
    assert second.returncode == 2
    assert "use --force" in second.stderr


def test_cli_force_overwrite(contract: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    assert _run_cli([str(contract), "--out-dir", str(out_dir)]).returncode == 0
    proc = _run_cli([str(contract), "--out-dir", str(out_dir), "--force"])
    assert proc.returncode == 0, proc.stderr


def test_cli_json_envelope(contract: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    proc = _run_cli(
        [str(contract), "--out-dir", str(out_dir), "--json"]
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["schema_version"] == "1"
    assert payload["tool"] == "mql5-task-graph-gen"
    assert payload["ok"] is True
    assert payload["data"]["contract"] == str(contract)
    assert len(payload["data"]["nodes"]) == 7
    assert payload["data"]["nodes"][0]["tip_id"] == "TIP-001"
    assert payload["data"]["nodes"][0]["depends_on"] == []


def test_cli_gate_report_writes_envelope(contract: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    gate = tmp_path / "gate.json"
    proc = _run_cli(
        [
            str(contract),
            "--out-dir",
            str(out_dir),
            "--gate-report",
            str(gate),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(gate.read_text(encoding="utf-8"))
    assert payload["tool"] == "mql5-task-graph-gen"
    assert payload["data"]["task_graph"].endswith("task-graph.md")


def test_cli_missing_contract_errors() -> None:
    proc = _run_cli(["/tmp/this/path/does/not/exist.md"])
    assert proc.returncode == 2
    assert "not found" in proc.stderr


def test_cli_broken_contract_errors(tmp_path: Path) -> None:
    bad = tmp_path / "bad.md"
    bad.write_text("# nope\n\nno task summary.\n", encoding="utf-8")
    proc = _run_cli([str(bad)])
    assert proc.returncode == 2
    assert "TASK GRAPH SUMMARY" in proc.stderr


# ---------------------------------------------------------------------------
# Console script entry
# ---------------------------------------------------------------------------


def test_console_script_registered(contract: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    proc = subprocess.run(
        ["mql5-task-graph-gen", str(contract), "--out-dir", str(out_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

"""Wave 6.2 — emit a per-TIP task graph from a Wave-6.1 contract.

The Wave-6.1 ``mql5-contract-gen`` produces a homeowner-facing
``contract.md`` that ends with a high-level ``## TASK GRAPH SUMMARY``
block of ``TIP-NNN — <description>`` lines. ``mql5-task-graph-gen``
takes that contract as input and expands the summary into a directory
of per-TIP markdown files plus a Mermaid DAG diagram at
``task-graph.md`` so the Builder can pick up tasks in dependency
order.

Outputs (deterministic given a fixed contract):

* ``tasks/TIP-001.md`` … ``tasks/TIP-NNN.md``
    One file per TIP. Each file declares YAML frontmatter
    (``tip_id``, ``title``, ``status``, ``depends_on``,
    ``invariant_refs``, ``actor``) and four Markdown sections
    (``## Goal``, ``## Acceptance criteria``, ``## Dependencies``,
    ``## Completion``).
* ``task-graph.md``
    A single Markdown file with a Mermaid ``graph TD`` diagram of
    the TIP dependency DAG plus a tabular index of the tasks.

The CLI is a deterministic Markdown emitter — no LLM, no network. The
DAG is derived structurally from the TIP order in the contract:

* ``TIP-001`` (scaffold) is the root.
* ``TIP-002`` (risk guard) depends on ``TIP-001``.
* Signal TIPs (``TIP-003`` onward, matched by the ``signal`` keyword
  in the description) depend on the risk-guard TIP.
* Filter TIPs depend on every signal TIP.
* The backtest + walk-forward TIP depends on every signal + filter
  TIP.
* The final permission-gate TIP depends on the backtest TIP.

This keeps the kit honest: the DAG is a pure function of the contract,
so two invocations on the same contract bit-identically reproduce the
same per-TIP files + diagram.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .._agent_io import (
    Envelope,
    add_gate_report_flag,
    add_json_flag,
    maybe_emit,
)

TOOL = "mql5-task-graph-gen"

_TIP_LINE = re.compile(r"^\s*-\s+(TIP-(\d{3}))\s+[\u2014\-]\s+(.+?)\s*$")
"""Match a single TIP bullet inside ``## TASK GRAPH SUMMARY``.

Accepts both em-dash (U+2014, the canonical Wave-6.1 emitter) and a
plain hyphen so an operator who hand-edited the contract isn't locked
out.
"""

_TASK_SUMMARY_HEADER = re.compile(r"^##\s+TASK GRAPH SUMMARY", re.IGNORECASE)
_INVARIANTS_HEADER = re.compile(r"^##\s+INVARIANTS", re.IGNORECASE)
_CONFIRM_HEADER = re.compile(r"^##\s+CONFIRM", re.IGNORECASE)
_NEXT_HEADER = re.compile(r"^##\s+\S")


@dataclass(frozen=True)
class TIPNode:
    """One TIP parsed from the contract."""

    tip_id: str  # "TIP-001"
    index: int  # 1-based ordinal
    description: str
    actor: str = "tho-thi-cong"
    depends_on: tuple[str, ...] = field(default_factory=tuple)
    invariant_refs: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TaskGraph:
    """Parsed + dependency-resolved graph derived from a contract."""

    contract_path: Path
    contract_sha256: str
    nodes: tuple[TIPNode, ...]
    invariants: tuple[str, ...]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _slice_section(body: str, header_re: re.Pattern[str]) -> list[str]:
    """Return the lines of the section starting at ``header_re`` (exclusive
    of the header itself, up to the next ``##`` heading).
    """

    lines = body.splitlines()
    out: list[str] = []
    in_section = False
    for line in lines:
        if not in_section:
            if header_re.match(line):
                in_section = True
            continue
        if _NEXT_HEADER.match(line):
            break
        out.append(line)
    return out


def parse_contract(path: Path) -> TaskGraph:
    """Parse ``contract.md`` into a :class:`TaskGraph`.

    Raises ``ValueError`` if the contract has no ``## TASK GRAPH
    SUMMARY`` section or every bullet inside it fails to match the
    canonical ``TIP-NNN — <desc>`` shape.
    """

    body = path.read_text(encoding="utf-8")
    sha = hashlib.sha256(body.encode("utf-8")).hexdigest()

    summary_lines = _slice_section(body, _TASK_SUMMARY_HEADER)
    if not summary_lines:
        raise ValueError(
            f"{path}: no `## TASK GRAPH SUMMARY` section found (Wave-6.1 "
            "contract-gen output is required)"
        )

    raw_tips: list[tuple[int, str]] = []
    for line in summary_lines:
        m = _TIP_LINE.match(line)
        if not m:
            continue
        raw_tips.append((int(m.group(2)), m.group(3).strip()))
    if not raw_tips:
        raise ValueError(
            f"{path}: `## TASK GRAPH SUMMARY` had no `TIP-NNN — <desc>` bullets"
        )

    # Invariants — best-effort cross-link. We don't fail if absent.
    invariant_lines = _slice_section(body, _INVARIANTS_HEADER)
    invariants: list[str] = []
    for line in invariant_lines:
        stripped = line.strip()
        if stripped.startswith("- ") and len(stripped) > 2:
            invariants.append(stripped[2:].strip())

    nodes = _resolve_dependencies(raw_tips, tuple(invariants))
    return TaskGraph(
        contract_path=path,
        contract_sha256=sha,
        nodes=nodes,
        invariants=tuple(invariants),
    )


# ---------------------------------------------------------------------------
# Dependency resolution
# ---------------------------------------------------------------------------


_KEYWORD_SCAFFOLD = re.compile(r"\bscaffold\b", re.IGNORECASE)
_KEYWORD_RISK = re.compile(r"\brisk\s*guard\b|\brisk\s*per\b", re.IGNORECASE)
_KEYWORD_SIGNAL = re.compile(r"\bsignal\b", re.IGNORECASE)
_KEYWORD_FILTER = re.compile(r"\bfilter\b", re.IGNORECASE)
_KEYWORD_BACKTEST = re.compile(
    r"\bbacktest\b|\bwalk[-\s]?forward\b", re.IGNORECASE
)
_KEYWORD_PERMISSION = re.compile(r"\bpermission\b|\bgate\b", re.IGNORECASE)


def _classify(desc: str) -> str:
    """Return one of ``scaffold``, ``risk``, ``signal``, ``filter``,
    ``backtest``, ``permission``, ``other`` from a TIP description.
    """

    if _KEYWORD_SCAFFOLD.search(desc):
        return "scaffold"
    if _KEYWORD_RISK.search(desc):
        return "risk"
    if _KEYWORD_PERMISSION.search(desc):
        return "permission"
    if _KEYWORD_BACKTEST.search(desc):
        return "backtest"
    if _KEYWORD_FILTER.search(desc):
        return "filter"
    if _KEYWORD_SIGNAL.search(desc):
        return "signal"
    return "other"


def _resolve_dependencies(
    raw_tips: list[tuple[int, str]],
    invariants: tuple[str, ...],
) -> tuple[TIPNode, ...]:
    """Compute the DAG edges from the structural roles of the TIPs.

    The resolver is deterministic and does not consider line text
    beyond the keyword classifier above. Each TIP picks up a small
    list of invariant references by greedy keyword match against the
    BLUEPRINT invariants (so the per-TIP file can cite which line(s)
    of the contract it implements).
    """

    classes = {idx: _classify(desc) for idx, desc in raw_tips}

    scaffold_ids = [f"TIP-{i:03d}" for i, c in classes.items() if c == "scaffold"]
    risk_ids = [f"TIP-{i:03d}" for i, c in classes.items() if c == "risk"]
    signal_ids = [f"TIP-{i:03d}" for i, c in classes.items() if c == "signal"]
    filter_ids = [f"TIP-{i:03d}" for i, c in classes.items() if c == "filter"]
    backtest_ids = [f"TIP-{i:03d}" for i, c in classes.items() if c == "backtest"]

    nodes: list[TIPNode] = []
    for idx, desc in raw_tips:
        tip_id = f"TIP-{idx:03d}"
        kind = classes[idx]
        deps: list[str] = []
        if kind == "scaffold":
            pass  # root
        elif kind == "risk":
            deps.extend(scaffold_ids)
        elif kind == "signal":
            deps.extend(risk_ids or scaffold_ids)
        elif kind == "filter":
            deps.extend(signal_ids or risk_ids or scaffold_ids)
        elif kind == "backtest":
            deps.extend(filter_ids + signal_ids)
            if not deps:
                deps.extend(risk_ids or scaffold_ids)
        elif kind == "permission":
            deps.extend(backtest_ids or filter_ids + signal_ids)
            if not deps:
                deps.extend(risk_ids or scaffold_ids)
        else:
            # "other" depends on the immediate predecessor so the
            # graph stays connected.
            if nodes:
                deps.append(nodes[-1].tip_id)
        # Dedup + preserve order.
        seen: set[str] = set()
        ordered_deps: list[str] = []
        for d in deps:
            if d != tip_id and d not in seen:
                ordered_deps.append(d)
                seen.add(d)

        # Invariant cross-link: any invariant whose lowercased text
        # shares a >= 6-char alphabetic token with the TIP description.
        inv_refs = _match_invariants(desc, invariants)

        nodes.append(
            TIPNode(
                tip_id=tip_id,
                index=idx,
                description=desc,
                actor="tho-thi-cong",
                depends_on=tuple(ordered_deps),
                invariant_refs=inv_refs,
            )
        )
    return tuple(nodes)


_STOP_WORDS = {
    "block",
    "chain",
    "every",
    "implement",
    "produce",
    "report",
    "report.",
    "with",
    "without",
    "wire",
}


def _tokens(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[A-Za-z]{6,}", text.lower())
        if t not in _STOP_WORDS
    }


def _match_invariants(desc: str, invariants: tuple[str, ...]) -> tuple[str, ...]:
    """Return invariants that share a notable >=6-char token with ``desc``.

    Deliberately greedy; the operator can prune the list manually in
    the rendered TIP file.
    """

    desc_tokens = _tokens(desc)
    if not desc_tokens:
        return ()
    out: list[str] = []
    for inv in invariants:
        inv_tokens = _tokens(inv)
        if desc_tokens & inv_tokens:
            out.append(inv)
    return tuple(out)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_tip_file(node: TIPNode, graph: TaskGraph) -> str:
    """Render a single ``tasks/TIP-NNN.md`` file body."""

    lines: list[str] = [
        "---",
        f"tip_id: {node.tip_id}",
        f"title: {node.description}",
        "status: PENDING",
        f"actor: {node.actor}",
        f"depends_on: [{', '.join(node.depends_on)}]"
        if node.depends_on
        else "depends_on: []",
        "invariant_refs:",
    ]
    if node.invariant_refs:
        for inv in node.invariant_refs:
            lines.append(f"  - {inv!r}")
    else:
        lines.append("  []")
    lines += [
        f"contract_sha256_prefix: {graph.contract_sha256[:12]}",
        "---",
        "",
        f"# {node.tip_id} — {node.description}",
        "",
        "## Goal",
        "",
        node.description.rstrip(".") + ".",
        "",
        "## Acceptance criteria",
        "",
        (
            "- Code change compiles under MetaEditor build ≥ 5260 "
            "(`mql5-compile`)."
        ),
        "- `mql5-lint` reports no new ERROR-level findings.",
        "- `mql5-trader-check` keeps the existing PASS verdict.",
        (
            "- Every invariant cited in `invariant_refs:` is exercised "
            "by ≥ 1 test under `tests/`."
        ),
        "",
        "## Dependencies",
        "",
    ]
    if node.depends_on:
        lines.append(
            "This TIP must wait for the following predecessor TIP(s) to "
            "reach `status: DONE`:"
        )
        lines.append("")
        for dep in node.depends_on:
            lines.append(f"- `{dep}`")
    else:
        lines.append(
            "_No predecessors — this TIP is a root node of the DAG._"
        )
    lines += [
        "",
        "## Completion",
        "",
        (
            "When this TIP is complete, run `mql5-completion-report "
            "--tip "
            f"tasks/{node.tip_id}.md "
            "--gate-reports <dir> --out completion-"
            f"{node.tip_id.split('-')[1]}.md` to emit the per-TIP "
            "Completion Report. Then update `status:` above to `DONE`."
        ),
        "",
        f"_Tool: `{TOOL}` — contract sha256 prefix "
        f"`{graph.contract_sha256[:12]}…`_",
        "",
    ]
    return "\n".join(lines)


def render_task_graph(graph: TaskGraph) -> str:
    """Render the top-level ``task-graph.md`` (Mermaid DAG + index)."""

    lines: list[str] = [
        "# Task Graph",
        "",
        (
            f"_Generated by `{TOOL}` from "
            f"`{graph.contract_path.name}` "
            f"(sha256 prefix `{graph.contract_sha256[:12]}…`)._ "
        ),
        "",
        "## DAG",
        "",
        "```mermaid",
        "graph TD",
    ]
    for node in graph.nodes:
        label = node.description.replace('"', "'")
        if len(label) > 64:
            label = label[:61] + "..."
        lines.append(f'    {node.tip_id}["{node.tip_id} — {label}"]')
    for node in graph.nodes:
        for dep in node.depends_on:
            lines.append(f"    {dep} --> {node.tip_id}")
    lines += [
        "```",
        "",
        "## Index",
        "",
        "| TIP | Description | Actor | Depends on | Invariants linked |",
        "|---|---|---|---|---|",
    ]
    for node in graph.nodes:
        deps = ", ".join(f"`{d}`" for d in node.depends_on) or "—"
        inv_count = len(node.invariant_refs)
        lines.append(
            f"| `{node.tip_id}` | {node.description} | `{node.actor}` "
            f"| {deps} | {inv_count} |"
        )
    lines += [
        "",
        "## Status legend",
        "",
        "Each `tasks/TIP-NNN.md` carries a `status:` field in its YAML",
        "frontmatter. Valid values:",
        "",
        "- `PENDING` — not started.",
        "- `IN_PROGRESS` — Builder picked it up.",
        "- `DONE` — Builder filed a Completion Report and the predecessor",
        "  acceptance criteria are met.",
        "- `BLOCKED` — Builder filed an escalation (Wave-6.2b",
        "  `mql5-escalation`); pause downstream TIPs until resolved.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_outputs(
    graph: TaskGraph,
    out_dir: Path,
    *,
    force: bool,
) -> tuple[list[Path], Path]:
    """Write per-TIP files + ``task-graph.md`` under ``out_dir``.

    Returns the list of TIP file paths plus the task-graph path. Raises
    ``FileExistsError`` if any target file exists and ``force`` is
    false.
    """

    tasks_dir = out_dir / "tasks"
    graph_path = out_dir / "task-graph.md"
    written: list[Path] = []

    if not force:
        if graph_path.exists():
            raise FileExistsError(graph_path)
        for node in graph.nodes:
            candidate = tasks_dir / f"{node.tip_id}.md"
            if candidate.exists():
                raise FileExistsError(candidate)

    tasks_dir.mkdir(parents=True, exist_ok=True)
    for node in graph.nodes:
        path = tasks_dir / f"{node.tip_id}.md"
        path.write_text(render_tip_file(node, graph), encoding="utf-8")
        written.append(path)

    graph_path.write_text(render_task_graph(graph), encoding="utf-8")
    return written, graph_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=TOOL)
    parser.add_argument(
        "contract",
        type=Path,
        help="Path to contract.md (Wave-6.1 mql5-contract-gen output).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Directory to write tasks/ and task-graph.md into "
        "(default: current directory).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing tasks/TIP-*.md and task-graph.md files.",
    )
    add_json_flag(parser)
    add_gate_report_flag(parser)
    args = parser.parse_args(argv)

    if not args.contract.is_file():
        env = Envelope(
            tool=TOOL,
            ok=False,
            exit_code=2,
            summary=f"contract not found: {args.contract}",
            data={"contract": str(args.contract)},
        )
        maybe_emit(args, env)
        if not args.emit_json:
            sys.stderr.write(f"error: file not found: {args.contract}\n")
        return 2

    try:
        graph = parse_contract(args.contract)
    except ValueError as exc:
        env = Envelope(
            tool=TOOL,
            ok=False,
            exit_code=2,
            summary=str(exc),
            data={"contract": str(args.contract)},
        )
        maybe_emit(args, env)
        if not args.emit_json:
            sys.stderr.write(f"error: {exc}\n")
        return 2

    try:
        written, graph_path = _write_outputs(
            graph, args.out_dir, force=args.force
        )
    except FileExistsError as exc:
        env = Envelope(
            tool=TOOL,
            ok=False,
            exit_code=2,
            summary=f"refusing to overwrite {exc} (use --force)",
            data={"conflict": str(exc)},
        )
        maybe_emit(args, env)
        if not args.emit_json:
            sys.stderr.write(f"error: {exc} exists (use --force)\n")
        return 2

    if not args.emit_json:
        sys.stderr.write(
            f"wrote {len(written)} TIP file(s) + {graph_path}\n"
        )

    env = Envelope(
        tool=TOOL,
        ok=True,
        exit_code=0,
        summary=(
            f"emitted {len(graph.nodes)} TIP node(s) "
            f"+ task-graph.md for `{args.contract.name}`"
        ),
        data={
            "contract": str(args.contract),
            "contract_sha256_prefix": graph.contract_sha256[:12],
            "out_dir": str(args.out_dir),
            "tasks": [str(p) for p in written],
            "task_graph": str(graph_path),
            "nodes": [
                {
                    "tip_id": node.tip_id,
                    "description": node.description,
                    "depends_on": list(node.depends_on),
                    "actor": node.actor,
                    "invariant_refs": list(node.invariant_refs),
                }
                for node in graph.nodes
            ],
        },
        evidence=[str(args.contract)],
    )
    maybe_emit(args, env)
    return 0


__all__ = [
    "TaskGraph",
    "TIPNode",
    "main",
    "parse_contract",
    "render_task_graph",
    "render_tip_file",
]


if __name__ == "__main__":
    raise SystemExit(main())

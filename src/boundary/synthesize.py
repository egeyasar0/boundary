"""Deterministic candidate boundary synthesizer from profiler output (M10 — R13).

Rule R13 (Deterministic Synthesis):
`boundary synthesize <run_dir>` generates `variant_config.json` deterministically without
any LLM or network calls, relying strictly on `extract_subagent` opportunities in `profile.json`.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from boundary.models import ToolCall
from boundary.profile import Opportunity, ProfileSummary, profile_traces
from boundary.subagent import (
    DEFAULT_MAX_SUMMARY_TOKENS,
    DEFAULT_SUBAGENT_BUDGET,
    DEFAULT_SUBAGENT_MAX_STEPS,
)
from boundary.tools import TOOL_SCHEMAS
from boundary.traces import read_traces

BoundaryToolName = Literal["delegate_inspection", "delegate_search"]
DEFAULT_VARIANT_NAME = "subagent_extracted"
VARIANT_CONFIG_FILENAME = "variant_config.json"


def build_boundary_tool_schema(name: BoundaryToolName) -> dict[str, Any]:
    """Build OpenAI-compatible function schema for a synthesized sub-agent boundary tool."""
    description = (
        "Delegate file inspection to an isolated sub-agent that reads/greps the file and "
        "returns only a concise summary (<= 300 tokens) to protect parent context budget."
        if name == "delegate_inspection"
        else (
            "Delegate repository search to an isolated sub-agent that searches files and "
            "returns only a concise summary (<= 300 tokens) to protect parent context budget."
        )
    )
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative file or directory path to inspect.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Specific question, field name, or condition to extract.",
                    },
                },
                "required": ["path", "query"],
                "additionalProperties": False,
            },
        },
    }


class DelegationRule(BaseModel):
    """Per-task sub-agent boundary rule synthesized from `extract_subagent` opportunities."""

    task_id: str
    intercepted_tools: list[str]
    boundary_tool: BoundaryToolName = "delegate_inspection"
    target_paths: list[str] = Field(default_factory=list)
    source_opportunity_ids: list[str] = Field(default_factory=list)
    est_upper_bound_savings_tokens: int = 0


class VariantConfig(BaseModel):
    """Synthesized architectural variant configuration serialized to `variant_config.json`."""

    variant_name: str = DEFAULT_VARIANT_NAME
    subagent_max_steps: int = DEFAULT_SUBAGENT_MAX_STEPS
    subagent_budget: int = DEFAULT_SUBAGENT_BUDGET
    max_summary_tokens: int = DEFAULT_MAX_SUMMARY_TOKENS
    rules: dict[str, DelegationRule] = Field(default_factory=dict)

    def get_rule(self, task_id: str) -> DelegationRule | None:
        """Return the DelegationRule for `task_id`, or None if task uses baseline tools."""
        return self.rules.get(task_id)

    def should_delegate(self, task_id: str, tc: ToolCall) -> bool:
        """Return True if `tc` should be routed through the isolated SubagentRunner."""
        rule = self.rules.get(task_id)
        if rule is None:
            return False
        if tc.name in ("delegate_inspection", "delegate_search", rule.boundary_tool):
            return True
        if tc.name in rule.intercepted_tools:
            if not rule.target_paths:
                return True
            raw_path = tc.arguments.get("path")
            if not isinstance(raw_path, str) or not raw_path.strip():
                return True
            norm_call_path = raw_path.strip().replace("\\", "/").lstrip("./")
            return any(
                norm_call_path == p.replace("\\", "/").lstrip("./")
                for p in rule.target_paths
            )
        return False

    def get_tool_schemas(
        self,
        task_id: str,
        base_schemas: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Return tool schemas for `task_id`, replacing intercepted tools with the boundary tool."""
        schemas = list(base_schemas if base_schemas is not None else TOOL_SCHEMAS)
        rule = self.rules.get(task_id)
        if rule is None:
            return schemas

        intercepted = set(rule.intercepted_tools)
        filtered: list[dict[str, Any]] = [
            s
            for s in schemas
            if s.get("function", {}).get("name") not in intercepted
        ]
        filtered.append(build_boundary_tool_schema(rule.boundary_tool))
        return filtered


def _select_boundary_tool(intercepted_tools: list[str]) -> BoundaryToolName:
    """Choose `delegate_inspection` when `read_file` is intercepted, else `delegate_search`."""
    if "read_file" in intercepted_tools:
        return "delegate_inspection"
    return "delegate_search"


def synthesize_variant(
    profile: ProfileSummary,
    *,
    target_paths_by_call: dict[tuple[str, str], str] | None = None,
    variant_name: str = DEFAULT_VARIANT_NAME,
    subagent_max_steps: int = DEFAULT_SUBAGENT_MAX_STEPS,
    subagent_budget: int = DEFAULT_SUBAGENT_BUDGET,
    max_summary_tokens: int = DEFAULT_MAX_SUMMARY_TOKENS,
) -> VariantConfig:
    """Synthesize VariantConfig strictly from `extract_subagent` opportunities (R13)."""
    extract_by_task: dict[str, list[Opportunity]] = defaultdict(list)
    for opp in profile.opportunities:
        if opp.kind == "extract_subagent":
            extract_by_task[opp.task_id].append(opp)

    path_lookup = target_paths_by_call or {}
    rules: dict[str, DelegationRule] = {}

    for task_id in sorted(extract_by_task.keys()):
        task_opps = sorted(extract_by_task[task_id], key=lambda o: o.id)
        tools_set: set[str] = set()
        paths_set: set[str] = set()
        opp_ids: list[str] = []
        total_savings = 0

        for opp in task_opps:
            opp_ids.append(opp.id)
            total_savings += opp.est_upper_bound_savings_tokens or 0
            tool_name = str(opp.evidence.get("tool", "read_file"))
            if tool_name and tool_name != "unknown":
                tools_set.add(tool_name)
            trace_id = str(opp.evidence.get("trace_id", ""))
            call_id = str(opp.evidence.get("call_id", ""))
            if (trace_id, call_id) in path_lookup:
                paths_set.add(path_lookup[(trace_id, call_id)])

        intercepted_tools = sorted(tools_set) if tools_set else ["read_file"]
        boundary_tool = _select_boundary_tool(intercepted_tools)

        rules[task_id] = DelegationRule(
            task_id=task_id,
            intercepted_tools=intercepted_tools,
            boundary_tool=boundary_tool,
            target_paths=sorted(paths_set),
            source_opportunity_ids=opp_ids,
            est_upper_bound_savings_tokens=total_savings,
        )

    return VariantConfig(
        variant_name=variant_name,
        subagent_max_steps=subagent_max_steps,
        subagent_budget=subagent_budget,
        max_summary_tokens=max_summary_tokens,
        rules=rules,
    )


def load_or_build_profile_for_synthesis(
    run_path: Path | str,
) -> tuple[ProfileSummary, dict[tuple[str, str], str], Path]:
    """Load `profile.json` from `run_path` (or build from traces) and extract call paths."""
    target = Path(run_path)
    if target.is_file() and target.suffix.lower() == ".json":
        profile_file = target
        base_dir = target.parent
    elif target.is_file():
        base_dir = target.parent
        profile_file = base_dir / "profile.json"
    else:
        base_dir = target
        profile_file = base_dir / "profile.json"

    path_lookup: dict[tuple[str, str], str] = {}
    try:
        trace_src = (
            base_dir if (target.is_file() and target.suffix == ".json") else target
        )
        traces = read_traces(trace_src)
        for tr in traces:
            for step in tr.steps:
                for tc in step.tool_calls:
                    raw_p = tc.arguments.get("path")
                    if isinstance(raw_p, str) and raw_p.strip():
                        path_lookup[(tr.trace_id, tc.id)] = raw_p.strip()
    except FileNotFoundError:
        traces = []

    if profile_file.exists():
        profile = ProfileSummary.model_validate_json(
            profile_file.read_text(encoding="utf-8")
        )
        return profile, path_lookup, base_dir

    if traces:
        profile = profile_traces(traces)
        return profile, path_lookup, base_dir

    raise FileNotFoundError(
        f"No profile.json or trace JSONL found at '{run_path}'."
    )


def save_variant_config(config: VariantConfig, out_path: Path | str) -> Path:
    """Serialize VariantConfig deterministically to JSON."""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        config.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return target


def load_variant_config(path: Path | str) -> VariantConfig:
    """Load and validate a VariantConfig JSON file."""
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Variant config file not found at '{target}'.")
    return VariantConfig.model_validate_json(target.read_text(encoding="utf-8"))

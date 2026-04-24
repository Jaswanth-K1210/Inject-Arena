"""Task-success verifiers. One per target_category."""

from typing import Any, Callable, Dict

from .exfiltration import verify_exfiltration
from .forbidden_tool import verify_forbidden_tool
from .prompt_leak import verify_prompt_leak


_VERIFIERS: Dict[str, Callable[[str, Dict[str, Any]], bool]] = {
    "exfiltration": verify_exfiltration,
    "forbidden_tool": verify_forbidden_tool,
    "prompt_leak": verify_prompt_leak,
}


def verify(agent_output: str, scenario: Dict[str, Any]) -> bool:
    """Dispatch to the right verifier based on the scenario's target_category."""
    cat = scenario["target_category"]
    if cat not in _VERIFIERS:
        raise ValueError(f"unknown target_category: {cat}")
    return _VERIFIERS[cat](agent_output, scenario)


__all__ = [
    "verify",
    "verify_exfiltration",
    "verify_forbidden_tool",
    "verify_prompt_leak",
]

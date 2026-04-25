"""LlamaFirewall wrapper.

Runs user input through PromptGuard (injection detection) and agent
output through AgentAlignment (reasoning-trace auditing) via Meta's
official ``llamafirewall`` package.

Setup (once per Colab session):
    pip install llamafirewall
    llamafirewall configure

Docs: https://github.com/meta-llama/PurpleLlama/tree/main/LlamaFirewall
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .base import DefenseVerdict

logger = logging.getLogger(__name__)


class FirewallWrapper:
    """Thin wrapper around LlamaFirewall with our DefenseVerdict interface."""

    name = "llama_firewall"

    def __init__(self) -> None:
        self._fw = None

    def _ensure_loaded(self) -> None:
        if self._fw is not None:
            return
        try:
            from llamafirewall import LlamaFirewall, Role, ScannerType  # type: ignore

            logger.info("Initialising LlamaFirewall …")
            self._fw = LlamaFirewall(
                scanners={
                    Role.USER: [ScannerType.PROMPT_GUARD],
                    Role.ASSISTANT: [ScannerType.AGENT_ALIGNMENT],
                }
            )
            logger.info("LlamaFirewall ready.")
        except ImportError as exc:
            raise ImportError(
                "llamafirewall not installed. Run: pip install llamafirewall && "
                "llamafirewall configure"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"LlamaFirewall init failed: {exc}") from exc

    def scan(
        self,
        user_query: str,
        agent_output: str,
        tool_call: Optional[Any] = None,
    ) -> DefenseVerdict:
        """Scan both the user input and the agent's output.

        Returns flagged=True if either the PromptGuard or the
        AgentAlignment scanner blocks. Score is the max across both.
        """
        self._ensure_loaded()
        fw = self._fw

        user_flagged, user_score, user_reason = self._scan_user(fw, user_query)
        asst_flagged, asst_score, asst_reason = self._scan_assistant(fw, agent_output, tool_call)

        flagged = user_flagged or asst_flagged
        score = max(user_score, asst_score)
        reasons = [r for r in [user_reason, asst_reason] if r]
        reason = "; ".join(reasons) if reasons else None

        return DefenseVerdict(flagged=flagged, score=score, reason=reason)

    @staticmethod
    def _scan_user(fw: Any, text: str) -> tuple[bool, float, Optional[str]]:
        if not text or not text.strip():
            return False, 0.0, None
        try:
            from llamafirewall import ScanDecision, UserMessage  # type: ignore

            result = fw.scan(UserMessage(content=text))
            flagged = result.decision == ScanDecision.BLOCK
            score = float(getattr(result, "score", 1.0 if flagged else 0.0))
            reason = f"fw_user:{result.reason}" if flagged and getattr(result, "reason", None) else None
            return flagged, score, reason
        except Exception as exc:
            logger.warning("FirewallWrapper user scan error: %s", exc)
            return False, 0.0, None

    @staticmethod
    def _scan_assistant(fw: Any, text: str, tool_call: Optional[Any]) -> tuple[bool, float, Optional[str]]:
        content = text or ""
        if tool_call is not None:
            import json
            try:
                tc_str = json.dumps(tool_call) if not isinstance(tool_call, str) else tool_call
                content = f"{content}\n[tool_call]\n{tc_str}".strip()
            except Exception:
                pass

        if not content.strip():
            return False, 0.0, None
        try:
            from llamafirewall import AssistantMessage, ScanDecision  # type: ignore

            result = fw.scan(AssistantMessage(content=content))
            flagged = result.decision == ScanDecision.BLOCK
            score = float(getattr(result, "score", 1.0 if flagged else 0.0))
            reason = f"fw_asst:{result.reason}" if flagged and getattr(result, "reason", None) else None
            return flagged, score, reason
        except Exception as exc:
            logger.warning("FirewallWrapper assistant scan error: %s", exc)
            return False, 0.0, None

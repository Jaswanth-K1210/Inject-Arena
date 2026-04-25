"""LlamaFirewall wrapper.

Scans user input via the llamafirewall PromptGuard scanner.
AgentAlignment (output-side scanner) is loaded only when TOGETHER_API_KEY
is set in the environment; otherwise it is skipped gracefully.

llamafirewall 1.0.x uses async internally.  In Jupyter/Colab the event loop
is already running, so we patch it with nest_asyncio on first load.

Setup (once per Colab session):
    pip install llamafirewall nest_asyncio
    llamafirewall configure

Docs: https://github.com/meta-llama/PurpleLlama/tree/main/LlamaFirewall
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

from .base import DefenseVerdict

logger = logging.getLogger(__name__)


def _apply_nest_asyncio() -> None:
    """Allow asyncio.run() inside a running event loop (Jupyter/Colab)."""
    try:
        import nest_asyncio  # type: ignore
        nest_asyncio.apply()
    except ImportError:
        pass


def _run_coro(coro: Any) -> Any:
    """Run a coroutine or return a plain value, handling Jupyter's running loop."""
    if not asyncio.iscoroutine(coro):
        return coro
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


class FirewallWrapper:
    """Thin wrapper around LlamaFirewall with our DefenseVerdict interface."""

    name = "llama_firewall"

    def __init__(self, prompt_guard_fallback: Optional[Any] = None) -> None:
        self._fw = None
        self._has_agent_alignment = False
        # Reuse an existing PromptGuard instance when llamafirewall's internal
        # scanner fails (e.g. transformers version mismatch on Colab T4).
        self._pg2_fallback = prompt_guard_fallback
        # Circuit-breaker: after the first llamafirewall scan failure, skip
        # further attempts and route directly to PG2 to avoid per-step log noise.
        self._fw_user_broken = False

    def _ensure_loaded(self) -> None:
        if self._fw is not None:
            return
        _apply_nest_asyncio()
        try:
            from llamafirewall import LlamaFirewall, Role, ScannerType  # type: ignore

            has_together = bool(os.environ.get("TOGETHER_API_KEY", "").strip())
            scanners: dict = {Role.USER: [ScannerType.PROMPT_GUARD]}
            if has_together:
                scanners[Role.ASSISTANT] = [ScannerType.AGENT_ALIGNMENT]
                self._has_agent_alignment = True
            else:
                logger.warning(
                    "TOGETHER_API_KEY not set — AgentAlignment scanner disabled. "
                    "r_bypass_fw will reflect PromptGuard input-side scan only."
                )

            logger.info("Initialising LlamaFirewall (agent_alignment=%s) …", has_together)
            self._fw = LlamaFirewall(scanners=scanners)
            logger.info("LlamaFirewall ready.")
        except ImportError as exc:
            raise ImportError(
                "llamafirewall not installed. Run: pip install llamafirewall nest_asyncio && "
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
        """Scan user input (always) and agent output (if TOGETHER_API_KEY set)."""
        self._ensure_loaded()
        fw = self._fw

        user_flagged, user_score, user_reason = self._scan_user(fw, user_query)
        if self._has_agent_alignment:
            asst_flagged, asst_score, asst_reason = self._scan_assistant(fw, agent_output, tool_call)
        else:
            asst_flagged, asst_score, asst_reason = False, 0.0, None

        flagged = user_flagged or asst_flagged
        score = max(user_score, asst_score)
        reasons = [r for r in [user_reason, asst_reason] if r]
        reason = "; ".join(reasons) if reasons else None

        return DefenseVerdict(flagged=flagged, score=score, reason=reason)

    def _scan_user(self, fw: Any, text: str) -> tuple[bool, float, Optional[str]]:
        if not text or not text.strip():
            return False, 0.0, None

        # If llamafirewall's user scanner has failed before, go straight to PG2.
        if not self._fw_user_broken:
            try:
                from llamafirewall import ScanDecision, UserMessage  # type: ignore

                result = _run_coro(fw.scan(UserMessage(content=text)))
                flagged = result.decision == ScanDecision.BLOCK
                score = float(getattr(result, "score", 1.0 if flagged else 0.0))
                reason = f"fw_user:{result.reason}" if flagged and getattr(result, "reason", None) else None
                return flagged, score, reason
            except Exception as exc:
                logger.warning(
                    "FirewallWrapper user scan error: %s — switching to PG2 fallback for this session", exc
                )
                self._fw_user_broken = True

        # PG2 fallback path (used on first failure and all subsequent calls).
        if self._pg2_fallback is not None:
            try:
                verdict = self._pg2_fallback.scan(text)
                return verdict.flagged, verdict.score, f"fw_pg2fb:{verdict.reason}"
            except Exception as fb_exc:
                logger.warning("FirewallWrapper PG2 fallback also failed: %s", fb_exc)
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

            result = _run_coro(fw.scan(AssistantMessage(content=content)))
            flagged = result.decision == ScanDecision.BLOCK
            score = float(getattr(result, "score", 1.0 if flagged else 0.0))
            reason = f"fw_asst:{result.reason}" if flagged and getattr(result, "reason", None) else None
            return flagged, score, reason
        except Exception as exc:
            logger.warning("FirewallWrapper assistant scan error: %s", exc)
            return False, 0.0, None

"""Defense wrappers (GPU — instantiate on Colab only).

Imports here are lazy: the wrapper classes don't load any models until
their first .scan() / .run() call, so this module is safely importable
on Mac for type-checking.
"""

from .base import Defense, DefenseVerdict
from .llama_firewall import FirewallWrapper
from .prompt_guard import PromptGuard
from .secalign_agent import SecAlignAgent

__all__ = ["Defense", "DefenseVerdict", "PromptGuard", "SecAlignAgent", "FirewallWrapper"]

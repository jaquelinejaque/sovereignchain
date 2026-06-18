"""Quorum — multi-LLM consensus engine.

Apache 2.0 (code) + HSP Commercial Restrictions (evolution loops).
Patent Pending: PCT/US26/11908 (HSP Protocol).
"""

# Import-time side effect: enable HSP dev mode automatically for local /
# self-host researchers, leave it untouched in any hosted environment.
# See quorum/_local_mode.py for the rules.
from quorum import _local_mode  # noqa: F401

from quorum.core.consensus import ConsensusResult, consensus

__version__ = "0.1.5"
__all__ = ["consensus", "ConsensusResult"]

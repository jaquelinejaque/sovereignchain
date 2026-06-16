"""Quorum — multi-LLM consensus engine.

Apache 2.0 + HSP Commercial Restrictions.
Patent: PCT/US26/11908 (HSP Protocol).
"""

from quorum.core.consensus import ConsensusResult, consensus

__version__ = "0.1.0"
__all__ = ["consensus", "ConsensusResult"]

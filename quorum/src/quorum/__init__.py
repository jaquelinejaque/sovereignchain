"""Quorum — multi-LLM consensus engine.

Functional Source License 1.1 (Apache-2.0-future, 2-year sunset).
Source-available, commercial use requires a paid Pro key from quorum-ai.dev.
Patent Pending: PCT/US26/11908 (HSP Protocol).
"""

# License gate: blocks import unless QUORUM_LICENSE_KEY is set, the process
# is the official hosted container, or honour-system dev mode is on.
# See quorum/_license.py for the exact rules.
from quorum import _license  # noqa: F401
_license.check_license()

# Import-time side effect: enable HSP dev mode automatically for local /
# self-host researchers, leave it untouched in any hosted environment.
# See quorum/_local_mode.py for the rules.
from quorum import _local_mode  # noqa: F401

from quorum.core.consensus import ConsensusResult, consensus

__version__ = "0.2.1"
__all__ = ["consensus", "ConsensusResult"]

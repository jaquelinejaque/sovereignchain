"""
Verifier Plugin API for Quorum.

This module defines the public integration points for the private quorum-verifier
plugin, conforming to the architecture specified in A10.
"""

from typing import Protocol, Optional, Any, Dict, List
from dataclasses import dataclass, field
import importlib.metadata

@dataclass
class AuditRecord:
    epoch: str
    verdict: str
    claim_hash: str
    oracle_uri: str
    entry_hash: str

class ConsensusResult(Protocol):
    prompt: str
    model_outputs: List[str]
    final_text: str

@dataclass
class ConsensusError:
    kind: str
    details: Dict[str, Any] = field(default_factory=dict)

class VerifierHook(Protocol):
    def on_consensus(self, result: ConsensusResult) -> None:
        """Called immediately after the public consensus engine reaches a successful decision."""
        ...
        
    def on_consensus_error(self, err: ConsensusError) -> None:
        """Called when consensus fails (e.g., timeout, insufficient quorum)."""
        pass
        
    def audit_report(self) -> AuditRecord:
        """Generates the final audit record for the operation."""
        ...

def load_verifier() -> Optional[VerifierHook]:
    """
    Dynamically discovers and loads the private verifier plugin via entry points.
    Returns None if the plugin is not installed (open-core usage).
    """
    try:
        eps = importlib.metadata.entry_points(group='quorum.verifiers')
        if eps:
            verifier_class = list(eps)[0].load()
            return verifier_class()
    except Exception:
        pass
    return None

class FakeVerifier:
    """A fake verifier used for testing the public engine without the private plugin."""
    
    def __init__(self):
        self.results = []
        self.errors = []
        
    def on_consensus(self, result: ConsensusResult) -> None:
        self.results.append(result)
        
    def on_consensus_error(self, err: ConsensusError) -> None:
        self.errors.append(err)
        
    def audit_report(self) -> AuditRecord:
        return AuditRecord(
            epoch="2026-06-30T00:00:00Z",
            verdict="TEST_VALID",
            claim_hash="testclaimhash",
            oracle_uri="https://test.oracle",
            entry_hash="testentryhash"
        )

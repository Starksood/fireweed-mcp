"""Fireweed — deterministic memory substrate (source-available core).

Public surface of the MCP release. The full research surface lives in the private repo.
"""
__version__ = "16.0.0-alpha"
from .fabric import Fireweed, Session, Turn
from .retrieval import RetrievalResult, query_graph
from .grounding import claim_faithful, admissible, classify
from .receipts import Receipt, receipt_for, bind_document, verify
from .erasure import erase, Certificate, ErasureIncomplete

__all__ = [
    "Fireweed", "Session", "Turn", "RetrievalResult", "query_graph",
    "claim_faithful", "admissible", "classify",
    "Receipt", "receipt_for", "bind_document", "verify",
    "erase", "Certificate", "ErasureIncomplete",
]

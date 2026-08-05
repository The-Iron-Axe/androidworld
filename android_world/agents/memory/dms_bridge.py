"""DMS library bridge for the U2 episodic memory module.

Isolates all imports from others/darwinian_memory so the U2 wrapper
(episodic.py) depends only on this stable surface.  Keeping the heavy
DMS imports here means episodic.py can be reasoned about without
touching the DMS internals.
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _REPO_ROOT not in sys.path:
  sys.path.insert(0, _REPO_ROOT)

from others.darwinian_memory.config import DMSConfig
from others.darwinian_memory.embedding import (
    EmbeddingBackend,
    SentenceTransformerBackend,
    TFIDFBackend,
)
from others.darwinian_memory.memory_bank import MemoryBank, RetrievalResult
from others.darwinian_memory.memory_entry import MemoryEntry, ObsAct, Plan
from others.darwinian_memory.verifier import KVerifier

__all__ = [
    "DMSConfig",
    "EmbeddingBackend",
    "SentenceTransformerBackend",
    "TFIDFBackend",
    "MemoryBank",
    "RetrievalResult",
    "MemoryEntry",
    "ObsAct",
    "Plan",
    "KVerifier",
]

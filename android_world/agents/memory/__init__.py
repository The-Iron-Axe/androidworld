"""Memory infrastructure for GUI agents.

This package contains the five-class memory taxonomy (U1-U5) as pure data
modules.  They are NOT agents — they have no LLM calls, no environment
interaction, and no agent inheritance.

U1: Task State     — structured per-step task progress
U2: Episodic       — per-task trajectory records (DMS-powered)
U3: Environment    — app/page/element knowledge (PG-Agent page-graph RAG)
U4: Procedural     — abstracted multi-trajectory workflows
U5: Control        — memory operations controller

U1-U4 are the *data layer* (what to save).
U5 is the *control layer* (when/how to write/retrieve/update).
"""

from android_world.agents.memory.dms_bridge import (
    DMSConfig, MemoryBank, MemoryEntry, ObsAct, Plan, RetrievalResult,
)
from android_world.agents.memory.environment import EnvKnowledge, build_screen_summary
from android_world.agents.memory.episodic import EpisodicMemory, ObsAct
from android_world.agents.memory.page_graph import PageEdge, PageGraph, PageNode
from android_world.agents.memory.procedural import ProceduralMemory
from android_world.agents.memory.skill import Skill, SkillAction, SkillLibrary
from android_world.agents.memory.task_state import (
    TaskState,
    extract_app_from_elements,
    format_u1_context,
    init_task_state,
    update_task_state,
)

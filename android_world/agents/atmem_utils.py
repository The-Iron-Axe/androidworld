"""Utilities for ATMem (Active Task-Driving Memory) agent.

Based on "What Memory Do GUI Agents Really Need? From Passive Records to Active
Task-Driving States" (arXiv:2606.31612).
"""

import json
import re

# ---------------------------------------------------------------------------
# ATMem JSON schema (as defined in the paper's Box A.3 / Section 3.1)
# ---------------------------------------------------------------------------

ATMem = dict  # Type alias for readability

_ATMEM_SCHEMA_TEMPLATE = {
    "phase": "harvest",      # "harvest" | "execute"
    "remainingFiles": [],     # sources pending inspection
    "completedFiles": [],     # sources already processed
    "constraints": {          # task-derived filtering conditions
        "logic": "AND",       # "AND" | "OR"
        "conditions": []      # list of {"field": str, "op": str, "value": str}
    },
    "schema": {
        "fields": [],         # minimal actionable data unit attributes
        "locked": False       # freeze fields after construction
    },
    "items": {}              # "id": {"content": {field: value}, "status": "remaining|finished|skipped", "skipReason": ""}
}


def empty_atmem() -> ATMem:
    return {
        "phase": "harvest",
        "remainingFiles": [],
        "completedFiles": [],
        "constraints": {"logic": "AND", "conditions": []},
        "schema": {"fields": [], "locked": False},
        "items": {},
    }


# ---------------------------------------------------------------------------
# System prompt adaptation: ATMem-augmented agent prompt
# ---------------------------------------------------------------------------

ATMEM_SYSTEM_PROMPT_EXTENSION = """
## Active Task-Driving Memory (ATMem)

You maintain an active task memory to track your execution state across steps.
Output a <atmem> block after your action to update this memory.

### When to Use Memory
Use memory only when cross-step data tracking is needed (multi-item workflows,
cross-app data transfer, constraint filtering). Otherwise output {}.

### Memory Schema
```json
{
    "phase": "harvest|execute",
    "remainingFiles": [],
    "completedFiles": [],
    "constraints": {"logic": "AND", "conditions": [{"field": "...", "op": "eq|neq|contains", "value": "..."}]},
    "schema": {"fields": ["attr1", "attr2"], "locked": false},
    "items": {
        "1": {"content": {"attr1": "val1", "attr2": "val2"}, "status": "remaining|finished|skipped", "skipReason": ""},
        "2": {"content": {...}, "status": "finished"}
    }
}
```

### Memory Usage Protocol
- Start with phase="harvest" when collecting information; switch to phase="execute" when ready to perform operations.
- In harvest phase: populate items from observed screens, using status="remaining".
- In execute phase: process items one by one, updating each to status="finished" after successful operation.
- Use status="skipped" with a skipReason when an item doesn't satisfy constraints.
- Store only structured values; do NOT store raw text dumps or screenshots.
- The schema.fields define what constitutes one actionable data unit.
"""


# ---------------------------------------------------------------------------
# ATMem parsing
# ---------------------------------------------------------------------------

def parse_atmem_block(output_text: str) -> ATMem | None:
    """Extract and parse the <atmem> JSON block from an LLM response."""
    # Try <atmem>...</atmem> first
    match = re.search(r'<atmem>\s*(.*?)\s*</atmem>', output_text, re.DOTALL)
    if not match:
        # Fallback: try <memory>...</memory> (paper's original format)
        match = re.search(r'<memory>\s*(.*?)\s*</memory>', output_text, re.DOTALL)
    if not match:
        return None

    raw = match.group(1)
    # Try to extract the outermost JSON object from the captured text
    # (the LLM may have appended extra text after the closing brace)
    obj = _extract_json_object(raw)
    if obj is None:
        return None
    if "phase" in obj or "items" in obj:
        return obj
    return None


def _extract_json_object(text: str) -> dict | None:
    """Extract the first complete JSON object from text, handling nesting."""
    # Find the first '{'
    start = text.find('{')
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == '\\' and in_string:
            escape = True
            continue
        if c == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def format_atmem_for_prompt(atmem: ATMem) -> str:
    """Render ATMem state as a compact JSON string for prompt injection."""
    if not atmem or not atmem.get("items") and atmem.get("phase") != "harvest":
        return "{}"
    return json.dumps(atmem, ensure_ascii=False, indent=2)


def merge_atmem(old: ATMem | None, new: ATMem | None) -> ATMem:
    """Merge a new ATMem partial update into the existing state.

    The LLM may output only the changed fields; this merges them into old.
    """
    if old is None:
        old = empty_atmem()
    if new is None or not new:
        return old

    merged = dict(old)

    # Merge top-level scalar fields
    for key in ("phase",):
        if key in new and new[key]:
            merged[key] = new[key]

    # Merge list fields (replace entirely if provided non-empty)
    for key in ("remainingFiles", "completedFiles"):
        if key in new and new[key]:
            merged[key] = new[key]

    # Merge constraints (replace entirely if new conditions provided)
    if "constraints" in new and new["constraints"]:
        if new["constraints"].get("logic"):
            merged.setdefault("constraints", {})["logic"] = new["constraints"]["logic"]
        if new["constraints"].get("conditions"):
            merged.setdefault("constraints", {})["conditions"] = list(new["constraints"]["conditions"])

    # Merge schema
    if "schema" in new and new["schema"]:
        if "fields" in new["schema"] and new["schema"]["fields"]:
            merged.setdefault("schema", {})["fields"] = new["schema"]["fields"]
        if "locked" in new["schema"]:
            merged.setdefault("schema", {})["locked"] = new["schema"]["locked"]

    # Merge items (by id)
    if "items" in new and new["items"]:
        merged.setdefault("items", {})
        for item_id, item_data in new["items"].items():
            if item_id not in merged["items"]:
                merged["items"][item_id] = item_data
            else:
                # Update existing item
                existing = dict(merged["items"][item_id])
                if "content" in item_data:
                    existing["content"] = item_data["content"]
                if "status" in item_data:
                    existing["status"] = item_data["status"]
                if "skipReason" in item_data:
                    existing["skipReason"] = item_data["skipReason"]
                merged["items"][item_id] = existing

    return merged


def atmem_summary_for_history(atmem: ATMem) -> str:
    """Generate a short textual summary of ATMem for the step history."""
    if not atmem:
        return ""
    items = atmem.get("items", {})
    if not items:
        return "[ATMem: empty]"

    remaining = sum(1 for v in items.values() if v.get("status") == "remaining")
    finished = sum(1 for v in items.values() if v.get("status") == "finished")
    skipped = sum(1 for v in items.values() if v.get("status") == "skipped")
    phase = atmem.get("phase", "?")
    return f"[ATMem: {phase} | {finished} done, {remaining} remaining" + (f", {skipped} skipped]" if skipped else "]")


# ---------------------------------------------------------------------------
# Planner prompt — decomposes task and constructs initial ATMem
# ---------------------------------------------------------------------------

PLANNER_PROMPT_TEMPLATE = """You are a GUI task planner with active memory. Given a user goal and the current screenshot, you must:

1. Decompose the task into concrete subgoals.
2. Build an Active Task-Driving Memory (ATMem) structure to track data across the task.

## Task Goal: {goal}

## ATMem Structure
{atmem_schema}

## Rules
- Use phase="harvest" for data collection subgoals, phase="execute" for data operation subgoals.
- The schema.fields define the minimal actionable data unit for this task.
- Constraints encode task-level filtering rules independently of the schema.
- If the task involves multiple data items (contacts, recipes, files, etc.), populate items with observed data during harvest.

## Output Format
SUBGOALS:
1. <first subgoal>
2. <second subgoal>
...

ATMEM: {{valid JSON object following the schema above, or {{}} if no memory needed}}
"""


def parse_planner_subgoals(text: str) -> list[str]:
    """Parse numbered subgoals from planner output."""
    subgoals = []
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r'^\d+[\.\)\-]\s*(.+)$', line)
        if m:
            subgoals.append(m.group(1).strip())
        elif line and not subgoals and not line.lower().startswith(('#', 'sub', 'atm')):
            subgoals.append(line)
    return subgoals

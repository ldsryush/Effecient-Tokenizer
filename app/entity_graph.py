"""
Entity Graph & Conversation Memory
-----------------------------------
Models the conversation as a graph where:
  - Nodes  = conversation turns  +  named entity nodes
  - Edges  = references from a turn to entities it mentions

Entity nodes (people, files, tasks, code vars) are NEVER dropped during
compression — they are always carried forward in the summary.

Storage is in-process for single-node deployments.  The RedisStore adapter
in store.py provides the same interface for multi-node deployments.
"""
from __future__ import annotations
import re
import time
from typing import Optional
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Entity detection (regex-based, no heavy NLP dependency)
# ---------------------------------------------------------------------------

# Patterns that identify "load-bearing" entities
_ENTITY_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("file",     re.compile(r"\b[\w\-]+\.\w{1,6}\b")),                   # file.py, config.yaml
    ("function", re.compile(r"\b[a-z_][a-zA-Z0-9_]*\s*\(")),             # func_name(
    ("user",     re.compile(r"\b(?:user|username|@\w+|I am|my name is)\s+([A-Za-z]\w*)", re.I)),
    ("task",     re.compile(r"\b(?:task|ticket|issue|bug|pr|story)\s*[#:]?\s*\d+", re.I)),
    ("variable", re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")),                 # CONSTANTS
    ("url",      re.compile(r"https?://\S+")),
]


def extract_entities(text: str) -> dict[str, list[str]]:
    """Return {entity_type: [mention, ...]} found in *text*."""
    found: dict[str, list[str]] = {}
    for etype, pat in _ENTITY_PATTERNS:
        matches = pat.findall(text)
        if matches:
            # flatten: findall may return strings or tuples
            flat = [m if isinstance(m, str) else m[0] for m in matches]
            # deduplicate preserving order
            seen: set[str] = set()
            unique = [x for x in flat if not (x in seen or seen.add(x))]  # type: ignore[func-returns-value]
            found[etype] = unique
    return found


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TurnNode:
    turn_id: int
    role: str
    content: str
    entities: dict[str, list[str]] = field(default_factory=dict)
    relevance_score: float = 0.0
    compressed: bool = False
    created_at: float = field(default_factory=time.time)


@dataclass
class EntityNode:
    name: str
    etype: str
    first_seen_turn: int
    last_seen_turn: int
    mention_count: int = 1


@dataclass
class ConversationGraph:
    session_id: str
    turns: list[TurnNode] = field(default_factory=list)
    entities: dict[str, EntityNode] = field(default_factory=dict)   # name → EntityNode
    summary: str = ""
    summary_entity_snapshot: dict[str, list[str]] = field(default_factory=dict)
    next_turn_id: int = 0

    # ------------------------------------------------------------------
    def add_turn(self, role: str, content: str) -> TurnNode:
        ents = extract_entities(content)
        turn = TurnNode(
            turn_id=self.next_turn_id,
            role=role,
            content=content,
            entities=ents,
        )
        self.next_turn_id += 1
        self.turns.append(turn)
        self._index_entities(turn)
        return turn

    # ------------------------------------------------------------------
    def _index_entities(self, turn: TurnNode) -> None:
        tid = turn.turn_id
        for etype, names in turn.entities.items():
            for name in names:
                key = f"{etype}:{name}"
                if key in self.entities:
                    node = self.entities[key]
                    node.last_seen_turn = tid
                    node.mention_count += 1
                else:
                    self.entities[key] = EntityNode(
                        name=name, etype=etype,
                        first_seen_turn=tid, last_seen_turn=tid,
                    )

    # ------------------------------------------------------------------
    def entity_overlap(self, turn: TurnNode, query_entities: dict[str, list[str]]) -> float:
        """
        Fraction of query entities that appear in *turn*.
        Returns 0..1.
        """
        q_names = {n for names in query_entities.values() for n in names}
        if not q_names:
            return 0.0
        t_names = {n for names in turn.entities.values() for n in names}
        overlap = q_names & t_names
        return len(overlap) / len(q_names)

    # ------------------------------------------------------------------
    def load_bearing_turns(self, query_entities: dict[str, list[str]], threshold: float = 0.1) -> set[int]:
        """
        Return turn_ids that have entity-overlap >= threshold with the current query,
        PLUS the turns that first introduced any entity the query references.
        """
        q_names = {n for names in query_entities.values() for n in names}
        lb: set[int] = set()
        for turn in self.turns:
            if turn.compressed:
                continue
            if self.entity_overlap(turn, query_entities) >= threshold:
                lb.add(turn.turn_id)
        # Also add first-seen turns for query entities
        for key, ent in self.entities.items():
            if ent.name in q_names:
                lb.add(ent.first_seen_turn)
        return lb

    # ------------------------------------------------------------------
    def all_entities_snapshot(self) -> dict[str, list[str]]:
        """Flat dict {etype: [name, ...]} of all known entities."""
        snap: dict[str, list[str]] = {}
        for key, ent in self.entities.items():
            snap.setdefault(ent.etype, []).append(ent.name)
        return snap


# ---------------------------------------------------------------------------
# In-process session store (replaced by RedisStore for multi-node)
# ---------------------------------------------------------------------------

_graphs: dict[str, ConversationGraph] = {}


def get_graph(session_id: str) -> ConversationGraph:
    if session_id not in _graphs:
        _graphs[session_id] = ConversationGraph(session_id=session_id)
    return _graphs[session_id]


def delete_graph(session_id: str) -> None:
    _graphs.pop(session_id, None)


def all_session_ids() -> list[str]:
    return list(_graphs.keys())

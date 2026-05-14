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
import os
import re
import time
from typing import Any, Optional
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Entity detection (regex-based, no heavy NLP dependency)
# ---------------------------------------------------------------------------

# Patterns that identify "load-bearing" entities
_ENTITY_PATTERNS: list[tuple[str, re.Pattern]] = [
    # file.py, config.yaml (exclude single-char base, numeric extensions, v1.2)
    ("file",     re.compile(r"\b[A-Za-z0-9][A-Za-z0-9_\-]{1,64}\.[A-Za-z][A-Za-z0-9]{0,5}\b")),
    ("function", re.compile(r"\b[a-z_][a-zA-Z0-9_]*\s*\(")),             # func_name(
    ("module",   re.compile(r"\b([A-Za-z][A-Za-z0-9_\- ]{2,})\s+module\b", re.I)),
    ("class",    re.compile(r"\b[A-Z][A-Za-z0-9]+(?:[A-Z][A-Za-z0-9]+)+\b")),
    ("symbol",   re.compile(r"\b[a-z][a-z0-9]+_[a-z0-9_]+\b")),
    ("user",     re.compile(r"\b(?:user|username|@\w+|I am|my name is)\s+([A-Za-z]\w*)", re.I)),
    ("task",     re.compile(r"\b(?:task|ticket|issue|bug|pr|story)\s*[#:]?\s*\d+", re.I)),
    ("variable", re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")),                 # CONSTANTS
    ("url",      re.compile(r"https?://\S+")),
]

_GENERIC_TOKENS = {
    "module", "service", "class", "file", "function", "method", "handler",
    "controller", "manager", "utility", "util", "utils", "component",
}

_EXT_TOKENS = {
    "py", "js", "ts", "tsx", "jsx", "json", "yaml", "yml", "md", "txt",
    "csv", "html", "css", "toml", "ini", "cfg", "sh", "bat", "go", "rs",
    "java", "kt", "rb", "php", "cs", "cpp", "c", "h", "hpp",
}

_COREF_PATTERNS: list[tuple[Optional[str], re.Pattern]] = [
    ("function", re.compile(r"\b(that|this)\s+function\b", re.I)),
    ("file",     re.compile(r"\b(the\s+file\s+I\s+mentioned|that\s+file|this\s+file)\b", re.I)),
    ("module",   re.compile(r"\b(that|this)\s+module\b", re.I)),
    ("class",    re.compile(r"\b(that|this)\s+class\b", re.I)),
    ("task",     re.compile(r"\b(that|this)\s+(task|ticket|issue|bug|pr|story)\b", re.I)),
    ("variable", re.compile(r"\b(that|this)\s+variable\b", re.I)),
    ("url",      re.compile(r"\b(that|this)\s+url\b", re.I)),
    (None,        re.compile(r"\b(it|this)\b", re.I)),
]

_FALSE_FILE_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$"),           # IP address
    re.compile(r"^\$?\d+(?:\.\d+)+$"),                # currency or numeric dotted
    re.compile(r"^[A-Za-z](?:\.[A-Za-z])+\.?$"),        # U.S., U.K.
    re.compile(r"^[vV]?\d+(?:\.\d+){1,3}$"),           # version-like
]


def _split_camel(token: str) -> list[str]:
    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", token)
    return parts if parts else [token]


def _normalize_token(token: str) -> str:
    token = token.lower()
    if len(token) >= 8:
        return token[:4]
    return token


def _tokenize_entity_name(name: str) -> list[str]:
    raw = re.findall(r"[A-Za-z0-9]+", name)
    parts: list[str] = []
    for tok in raw:
        parts.extend(_split_camel(tok))
    cleaned: list[str] = []
    for tok in parts:
        tok = tok.lower()
        if tok in _GENERIC_TOKENS or tok in _EXT_TOKENS:
            continue
        tok = _normalize_token(tok)
        if len(tok) < 2:
            continue
        cleaned.append(tok)
    return cleaned


def _canonicalize_entity(name: str, etype: str) -> str:
    if etype == "url":
        return name.strip()
    tokens = _tokenize_entity_name(name)
    if not tokens:
        return name.strip().lower()
    return "_".join(tokens)


def _is_false_file_entity(name: str) -> bool:
    for pat in _FALSE_FILE_PATTERNS:
        if pat.match(name):
            return True
    return False


def extract_entities(text: str) -> dict[str, list[str]]:
    """Return {entity_type: [canonical_name, ...]} found in *text*."""
    found: dict[str, list[str]] = {}
    for etype, pat in _ENTITY_PATTERNS:
        matches = pat.findall(text)
        if matches:
            # flatten: findall may return strings or tuples
            flat = [m if isinstance(m, str) else m[0] for m in matches]
            # canonicalize and deduplicate preserving order
            seen: set[str] = set()
            unique: list[str] = []
            for raw in flat:
                if etype == "file" and _is_false_file_entity(raw):
                    continue
                canon = _canonicalize_entity(raw, etype)
                if canon and canon not in seen:
                    seen.add(canon)
                    unique.append(canon)
            if unique:
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
    aliases: list[str] = field(default_factory=list)


@dataclass
class ConversationGraph:
    session_id: str
    turns: list[TurnNode] = field(default_factory=list)
    entities: dict[str, EntityNode] = field(default_factory=dict)   # name → EntityNode
    summary: str = ""
    summary_entity_snapshot: dict[str, list[str]] = field(default_factory=dict)
    next_turn_id: int = 0
    _store_key: Optional[str] = field(default=None, repr=False)
    _store_ref: Optional[Any] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    def add_turn(self, role: str, content: str) -> TurnNode:
        ents = extract_entities(content)
        self._apply_coreferences(content, ents)
        turn = TurnNode(
            turn_id=self.next_turn_id,
            role=role,
            content=content,
            entities=ents,
        )
        self.next_turn_id += 1
        self.turns.append(turn)
        self._index_entities(turn)
        self.persist()
        return turn

    # ------------------------------------------------------------------
    def _index_entities(self, turn: TurnNode) -> None:
        tid = turn.turn_id
        for etype, names in turn.entities.items():
            for name in names:
                canonical = self._canonical_for_graph(name, etype)
                key = f"{etype}:{canonical}"
                if key in self.entities:
                    node = self.entities[key]
                    node.last_seen_turn = tid
                    node.mention_count += 1
                    if name != node.name and name not in node.aliases:
                        node.aliases.append(name)
                else:
                    self.entities[key] = EntityNode(
                        name=canonical, etype=etype,
                        first_seen_turn=tid, last_seen_turn=tid,
                    )
                    if name != canonical:
                        self.entities[key].aliases.append(name)

    def _canonical_for_graph(self, name: str, etype: str) -> str:
        tokens = set(_tokenize_entity_name(name))
        if not tokens:
            return _canonicalize_entity(name, etype)

        best_name = _canonicalize_entity(name, etype)
        best_score = 0.0

        for ent in self.entities.values():
            ent_candidates = [ent.name] + ent.aliases
            ent_tokens: set[str] = set()
            for cand in ent_candidates:
                ent_tokens.update(_tokenize_entity_name(cand))
            if not ent_tokens:
                continue
            overlap = len(tokens & ent_tokens) / max(1, max(len(tokens), len(ent_tokens)))
            if overlap >= 0.7 and overlap > best_score:
                best_score = overlap
                best_name = ent.name

        return best_name

    def _most_recent_entity(self, etype: Optional[str] = None) -> Optional[str]:
        best_name: Optional[str] = None
        best_turn = -1
        for ent in self.entities.values():
            if etype and ent.etype != etype:
                continue
            if ent.last_seen_turn > best_turn:
                best_turn = ent.last_seen_turn
                best_name = ent.name
        return best_name

    def _apply_coreferences(self, text: str, entities: dict[str, list[str]]) -> None:
        for etype, pattern in _COREF_PATTERNS:
            if not pattern.search(text):
                continue
            target = self._most_recent_entity(etype)
            if not target:
                continue
            use_type = etype or self._infer_type_for_name(target)
            if not use_type:
                continue
            entities.setdefault(use_type, [])
            if target not in entities[use_type]:
                entities[use_type].append(target)

    def _infer_type_for_name(self, name: str) -> Optional[str]:
        for ent in self.entities.values():
            if ent.name == name:
                return ent.etype
        return None

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turns": [
                {
                    "turn_id": t.turn_id,
                    "role": t.role,
                    "content": t.content,
                    "entities": t.entities,
                    "relevance_score": t.relevance_score,
                    "compressed": t.compressed,
                    "created_at": t.created_at,
                }
                for t in self.turns
            ],
            "entities": [
                {
                    "name": e.name,
                    "etype": e.etype,
                    "first_seen_turn": e.first_seen_turn,
                    "last_seen_turn": e.last_seen_turn,
                    "mention_count": e.mention_count,
                    "aliases": e.aliases,
                }
                for e in self.entities.values()
            ],
            "summary": self.summary,
            "summary_entity_snapshot": self.summary_entity_snapshot,
            "next_turn_id": self.next_turn_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationGraph":
        graph = cls(session_id=data.get("session_id", ""))
        graph.turns = [
            TurnNode(
                turn_id=t.get("turn_id", 0),
                role=t.get("role", ""),
                content=t.get("content", ""),
                entities=t.get("entities", {}),
                relevance_score=t.get("relevance_score", 0.0),
                compressed=t.get("compressed", False),
                created_at=t.get("created_at", time.time()),
            )
            for t in data.get("turns", [])
        ]
        graph.entities = {}
        for e in data.get("entities", []):
            key = f"{e.get('etype', '')}:{e.get('name', '')}"
            graph.entities[key] = EntityNode(
                name=e.get("name", ""),
                etype=e.get("etype", ""),
                first_seen_turn=e.get("first_seen_turn", 0),
                last_seen_turn=e.get("last_seen_turn", 0),
                mention_count=e.get("mention_count", 1),
                aliases=e.get("aliases", []) or [],
            )
        graph.summary = data.get("summary", "")
        graph.summary_entity_snapshot = data.get("summary_entity_snapshot", {})
        graph.next_turn_id = data.get("next_turn_id", len(graph.turns))
        return graph

    def persist(self, store: Optional[Any] = None) -> None:
        s = store or self._store_ref
        if s is None or self._store_key is None:
            return
        ttl_s = int(os.environ.get("ENTITY_GRAPH_TTL_S", "86400"))
        s.set(self._store_key, self.to_dict(), ttl_s=ttl_s)

    def attach_store(self, store: Any, key: str) -> None:
        self._store_ref = store
        self._store_key = key


# ---------------------------------------------------------------------------
# In-process session store (replaced by RedisStore for multi-node)
# ---------------------------------------------------------------------------

_graphs: dict[str, ConversationGraph] = {}


def _graph_key(session_id: str, user_id: str) -> str:
    return f"entity_graph:{user_id}:{session_id}"


def load_or_create(session_id: str, user_id: str, store: Any) -> ConversationGraph:
    if session_id in _graphs:
        return _graphs[session_id]

    key = _graph_key(session_id, user_id)
    raw = store.get(key)
    if isinstance(raw, dict):
        graph = ConversationGraph.from_dict(raw)
    else:
        graph = ConversationGraph(session_id=session_id)

    graph.attach_store(store, key)
    _graphs[session_id] = graph
    return graph


def get_graph(session_id: str, user_id: Optional[str] = None, store: Optional[Any] = None) -> ConversationGraph:
    from .store import store as _store
    uid = user_id or "anonymous"
    return load_or_create(session_id, uid, store or _store)


def delete_graph(session_id: str, user_id: Optional[str] = None, store: Optional[Any] = None) -> None:
    from .store import store as _store
    uid = user_id or "anonymous"
    s = store or _store
    key = _graph_key(session_id, uid)
    s.delete(key)
    _graphs.pop(session_id, None)


def all_session_ids() -> list[str]:
    return list(_graphs.keys())

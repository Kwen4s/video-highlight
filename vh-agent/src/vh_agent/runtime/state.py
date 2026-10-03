"""Durable information accounting, independent of model conversation history."""

import hashlib
import json
from dataclasses import dataclass, field


def identity(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


@dataclass
class ReadMemory:
    rows: dict = field(default_factory=dict)
    queries: dict = field(default_factory=dict)

    def record(self, arguments: dict, result: dict) -> dict:
        query = {k: v for k, v in arguments.items() if k not in {"offset", "limit"}}
        query["query"] = query["query"].casefold()
        key = identity(query)
        previous = self.queries.get(key, {"query": query, "row_ids": [], "pages": {}})
        new_ids = []
        for row in result["matches"]:
            row_id = row["row_id"]
            if row_id not in self.rows:
                new_ids.append(row_id)
            self.rows[row_id] = row
        returned = [r["row_id"] for r in result["matches"]]
        previous["row_ids"] = list(dict.fromkeys([*previous["row_ids"], *returned]))
        previous["pages"][str(arguments["offset"])] = {
            "row_ids": returned,
            "next_offset": result["next_offset"],
        }
        previous["total"] = result["total"]
        previous["complete"] = len(previous["row_ids"]) == result["total"]
        self.queries[key] = previous
        return {
            **result,
            "query_id": key,
            "new_row_ids": new_ids,
            "query_complete": previous["complete"],
            "reading_note": "本次有新材料。"
            if new_ids
            else "本次没有新增材料；已读内容可从 read_state 获取。",
        }

    def checkpoint(self):
        return {"rows": self.rows, "queries": self.queries}


@dataclass
class ProgressMemory:
    """Unique information and meaningful states; prose-only rewrites do not count."""

    seen: set[str] = field(default_factory=set)
    stagnant_steps: int = 0

    def record(self, facts: set[str], *, count_stagnation: bool = True) -> int:
        added = facts - self.seen
        self.seen.update(facts)
        if added:
            self.stagnant_steps = 0
        elif count_stagnation:
            self.stagnant_steps += 1
        return len(added)

    def checkpoint(self):
        return {"seen": sorted(self.seen), "stagnant_steps": self.stagnant_steps}

    @classmethod
    def restore(cls, value):
        return cls(set(value["seen"]), value["stagnant_steps"])

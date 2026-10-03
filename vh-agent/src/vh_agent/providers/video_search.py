"""Per-video exact vector search over original frames and source transcript text."""

import base64
import hashlib
import json
import math
from pathlib import Path
from uuid import uuid4

import httpx
import numpy as np

from ..preprocessing.scene_detection import detect_scenes
from ..storage import write_json

EMBED_INSTRUCTION = "Represent the user input."
QUERY_INSTRUCTION = "根据问题检索短剧视频中的相关画面、行动、物品、人物互动或台词。"


class QwenSearch:
    def __init__(
        self,
        evidence,
        embedding_url,
        reranker_url,
        transcript=None,
        *,
        segment_sec=8,
        frame_fps=1,
        timeout=300,
    ):
        self.evidence, self.transcript = evidence, transcript
        self.embedding_url, self.reranker_url = embedding_url, reranker_url
        self.segment_sec, self.frame_fps = segment_sec, frame_fps
        self._http = httpx.Client(trust_env=False, timeout=timeout)
        self.profile = {
            "version": 2,
            "media_id": evidence.media_id,
            "embedding_url": embedding_url,
            "reranker_url": reranker_url,
            "embedding_model": "Qwen3-VL-Embedding-2B",
            "reranker_model": "Qwen3-VL-Reranker-2B",
            "segment_sec": segment_sec,
            "frame_fps": frame_fps,
            "frame_width": 768,
            "instruction": EMBED_INSTRUCTION,
            "query_instruction": QUERY_INSTRUCTION,
            "transcript": transcript.profile if transcript else None,
        }
        for url, model in (
            (embedding_url, self.profile["embedding_model"]),
            (reranker_url, self.profile["reranker_model"]),
        ):
            if url:
                models = self._post_or_get(url + "/v1/models")
                entry = next((d for d in models["data"] if d["id"] == model), None)
                if entry is None:
                    raise ValueError(f"Qwen 服务未提供配置模型 {model}")
                root = Path(entry["root"])
                # Local deployed weights have stable revision material; never use restart time.
                if root.is_dir():
                    revision = [
                        (p.name, p.stat().st_size, p.stat().st_mtime_ns)
                        for p in sorted(root.glob("*.safetensors"))
                    ]
                    revision.append(hashlib.sha256((root / "config.json").read_bytes()).hexdigest())
                    self.profile[model + "_revision"] = hashlib.sha256(
                        json.dumps(revision).encode()
                    ).hexdigest()
        key = hashlib.sha256(json.dumps(self.profile, sort_keys=True).encode()).hexdigest()[:24]
        self.folder = evidence.output_dir / ("index_" + key)
        self.folder.mkdir(exist_ok=True)
        self.rows, self.vectors = [], None
        self.query_cache = {}

    def close(self):
        self._http.close()

    def _post_or_get(self, url, payload=None):
        try:
            response = (
                self._http.get(url) if payload is None else self._http.post(url, json=payload)
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Qwen 请求失败（{type(exc).__name__}），未生成替代检索结果。"
            ) from None

    @staticmethod
    def _content(row):
        if row["channel"] == "transcript":
            return [{"type": "text", "text": row["text"]}]
        content = []
        for frame in row["frames"]:
            content.append({"type": "text", "text": f"原片 {frame['timestamp_sec']:.3f} 秒"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,"
                        + base64.b64encode(Path(frame["path"]).read_bytes()).decode()
                    },
                }
            )
        return content

    def _embed(self, content, instruction):
        result = self._post_or_get(
            self.embedding_url + "/v1/embeddings",
            {
                "model": self.profile["embedding_model"],
                "encoding_format": "float",
                "messages": [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": content},
                ],
            },
        )
        vector = np.asarray(result["data"][0]["embedding"], dtype=np.float32)
        norm = np.linalg.norm(vector)
        if vector.ndim != 1 or len(vector) == 0 or not np.isfinite(vector).all() or norm <= 0:
            raise ValueError("Qwen 返回了无效向量。")
        return vector / norm

    def _manifest(self):
        path = self.folder / "materials.json"
        if path.is_file():
            return json.loads(path.read_text())
        rows = []
        for scene in detect_scenes(self.evidence.video_info.path):
            timeline = self.evidence._frame_times
            first = min(round(scene.start_sec * self.evidence.video_info.fps), len(timeline) - 1)
            last = round(scene.end_sec * self.evidence.video_info.fps)
            start_sec = timeline[first]
            end_sec = (
                timeline[last] if last < len(timeline) else self.evidence.video_info.duration_sec
            )
            count = max(1, math.ceil((end_sec - start_sec) / self.segment_sec))
            for i in range(count):
                start = start_sec + i * (end_sec - start_sec) / count
                end = start_sec + (i + 1) * (end_sec - start_sec) / count
                frame_count = max(1, math.ceil((end - start) * self.frame_fps))
                frames = self.evidence.frames(
                    [start + j * (end - start) / frame_count for j in range(frame_count)], width=768
                )
                rows.append(
                    {
                        "row_id": f"visual_{len(rows)}",
                        "channel": "visual",
                        "start_sec": start,
                        "end_sec": end,
                        "frames": frames,
                    }
                )
        if self.transcript:
            for i, row in enumerate(self.transcript.load()):
                rows.append({**row, "row_id": f"dialogue_{i}", "channel": "transcript"})
        write_json(path, rows)
        return rows

    def _ensure_index(self):
        if self.vectors is not None:
            return
        rows = self._manifest()
        vectors = []
        path = self.folder / "vectors.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as stored:
                recorded = json.loads(str(stored["row_ids"]))
                if recorded != [r["row_id"] for r in rows[: len(recorded)]]:
                    raise ValueError("检索向量与材料清单不一致。")
                vectors = list(stored["vectors"])
        for row in rows[len(vectors) :]:
            vectors.append(self._embed(self._content(row), EMBED_INSTRUCTION))
            temp = path.with_name(uuid4().hex + ".tmp")
            try:
                with temp.open("wb") as stream:
                    np.savez_compressed(
                        stream,
                        vectors=np.stack(vectors),
                        row_ids=json.dumps([r["row_id"] for r in rows[: len(vectors)]]),
                    )
                temp.replace(path)
            finally:
                temp.unlink(missing_ok=True)
        self.rows, self.vectors = rows, np.stack(vectors)
        write_json(self.folder / "manifest.json", {"profile": self.profile, "rows": rows})

    def search(self, query, start_sec, end_sec, offset, limit):
        if not query.strip():
            raise ValueError("semantic 检索需要具体问题；浏览字幕请使用 exact。")
        self._ensure_index()
        signature = json.dumps([query, start_sec, end_sec])
        if signature not in self.query_cache:
            vector = self._embed([{"type": "text", "text": query}], QUERY_INSTRUCTION)
            if self.vectors.shape[1] != len(vector):
                raise ValueError("查询向量与索引维度不同。")
            similarities = self.vectors @ vector
            indices = [
                i
                for i, row in enumerate(self.rows)
                if row["end_sec"] > start_sec and (end_sec is None or row["start_sec"] < end_sec)
            ]
            indices.sort(key=lambda i: float(similarities[i]), reverse=True)
            self.query_cache[signature] = [(i, float(similarities[i])) for i in indices]
        ranking = self.query_cache[signature]
        page = ranking[offset : offset + limit]
        rows = [
            {
                **{k: v for k, v in self.rows[i].items() if k != "frames"},
                "retrieval_similarity": score,
            }
            for i, score in page
        ]
        if rows and self.reranker_url:
            ranked = self._post_or_get(
                self.reranker_url + "/rerank",
                {
                    "model": self.profile["reranker_model"],
                    "query": query,
                    "documents": [{"content": self._content(self.rows[i])} for i, _ in page],
                    "top_n": len(page),
                    "instruction": QUERY_INSTRUCTION,
                },
            )["results"]
            if sorted(r["index"] for r in ranked) != list(range(len(page))):
                raise ValueError("Qwen 重排未返回本页全部材料。")
            rows = [
                {**rows[r["index"]], "retrieval_relevance": r["relevance_score"]} for r in ranked
            ]
        return {
            "matches": rows,
            "total": len(ranking),
            "next_offset": offset + limit if offset + limit < len(ranking) else None,
            "ordering": "全量向量排序；仅对当前返回页做多模态重排。未返回材料可继续分页。",
            "evidence_policy": "相关性用于定位材料，不能证明剧情事实或高光价值；观看原片确认。",
        }

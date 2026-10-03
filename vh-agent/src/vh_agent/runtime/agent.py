"""A single video ReAct controller, with independent viewing of rendered clips."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from uuid import uuid4

from pydantic import ValidationError

from ..prompts import ANALYSIS_PROMPT, DEFAULT_TASK, SCAN_QUESTION, SELECTION_PROMPT
from ..providers.gemini_client import GeminiClientError
from ..providers.openai_client import OpenAIClient, OpenAIClientError
from ..storage import write_json
from .context import Conversation, hydrate, image_part, text_part
from .evidence import EvidenceStore, VideoObservation
from .finalization import SelectionResult
from .state import ProgressMemory
from .tools import VideoTools

MAX_PAGE_BATCH_ITEMS = 2
MAX_PAGE_BATCH_ENCODED_BYTES = 36_000_000
MAX_REVIEW_WORKERS = 4


class VideoAgent:
    def __init__(
        self,
        client: OpenAIClient,
        evidence: EvidenceStore,
        output_dir: Path,
        *,
        perception,
        frame_fps: float = 0.5,
        search=None,
        frame_text=None,
        task: str = DEFAULT_TASK,
        max_highlights: int | None = 12,
        total_duration: float | None = None,
        allow_overlap: bool = True,
        max_clip_sec: float = 24,
        min_clip_sec: float = 3,
        video_fps: float = 4,
        on_progress: Callable[[str], None] | None = None,
        transcript=None,
        local_proposals=None,
        context_token_budget: int = 100_000,
        context_byte_budget: int = 18_000_000,
        max_stagnant_steps: int = 8,
        max_request_attempts: int = 3,
    ) -> None:
        if (
            (max_highlights is not None and max_highlights <= 0)
            or (total_duration is not None and total_duration <= 0)
            or not 0 < min_clip_sec <= max_clip_sec
            or video_fps <= 0
            or frame_fps <= 0
            or not all(math.isfinite(n) for n in (min_clip_sec, max_clip_sec, video_fps, frame_fps))
        ):
            raise ValueError("Invalid output or sampling constraints")
        self.client = client
        self.perception = perception
        self.frame_fps = frame_fps
        self._request_lock = Lock()
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tools = VideoTools(
            evidence,
            self.output_dir,
            max_highlights=max_highlights,
            max_clip_sec=max_clip_sec,
            min_clip_sec=min_clip_sec,
            total_duration=total_duration,
            allow_overlap=allow_overlap,
            transcript=transcript,
            local_proposals=local_proposals,
            search=search,
            frame_text=frame_text,
        )
        self.task = task
        self.video_fps = video_fps
        self.on_progress = on_progress or (lambda message: None)
        self.turn = 0
        self.model_calls = 0
        self.pending_media: list[VideoObservation] = []
        self.focus_event_ids: list[str] = []
        self.conversation = Conversation()
        self.progress_memory = ProgressMemory(seen=self.tools.information_facts())
        self.pending_calls: list[dict] = []
        self.reject_pending_batch = False
        self.call_results: list[dict] = []
        self.request_ready = False
        self.request_observation_ids: list[str] = []
        self.context_token_budget = context_token_budget
        self.context_byte_budget = context_byte_budget
        self.max_stagnant_steps = max_stagnant_steps
        self.max_request_attempts = max_request_attempts
        if (
            min(context_token_budget, context_byte_budget, max_stagnant_steps, max_request_attempts)
            <= 0
        ):
            raise ValueError("Context and progress budgets must be positive")
        self.profile = {
            "context_token_budget": context_token_budget,
            "context_byte_budget": context_byte_budget,
            "max_stagnant_steps": max_stagnant_steps,
            "max_request_attempts": max_request_attempts,
            "retrieval": transcript.profile if transcript else None,
            "local_proposals": local_proposals.profile if local_proposals else None,
            "implementation_hash": hashlib.sha256(
                b"".join(
                    (Path(__file__).parents[1] / name).read_bytes()
                    for name in (
                        "prompts.py",
                        "runtime/agent.py",
                        "runtime/tools.py",
                        "runtime/evidence.py",
                        "runtime/finalization.py",
                        "providers/gemini_client.py",
                        "providers/openai_client.py",
                        "providers/video_perception.py",
                        "providers/video_search.py",
                        "providers/frame_text.py",
                        "providers/retrieval.py",
                        "providers/local_proposals.py",
                        "storage.py",
                        "runtime/contracts.py",
                        "runtime/state.py",
                        "runtime/context.py",
                    )
                )
            ).hexdigest(),
            "model": client.model,
            "reasoning_effort": client.effort,
            "perception": perception.profile,
            "frame_fps": frame_fps,
            "semantic_search": search.profile if search else None,
            "frame_text": frame_text.profile if frame_text else None,
            "endpoint": client.endpoint,
            "video_fps": video_fps,
            "media_id": evidence.media_id,
            "task": task,
            "max_highlights": max_highlights,
            "total_duration": total_duration,
            "allow_overlap": allow_overlap,
            "min_clip_sec": min_clip_sec,
            "max_clip_sec": max_clip_sec,
            "pages": [p.model_dump(mode="json") for p in evidence.pages],
        }

    def _trace(self, kind: str, **data) -> None:
        with (self.output_dir / "trace.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {"timestamp": time.time(), "kind": kind, "turn": self.turn, **data},
                    ensure_ascii=False,
                )
                + "\n"
            )

    def _save(self) -> None:
        write_json(
            self.output_dir / "progress.json",
            {**self.tools.progress(), "model_calls": self.model_calls, "turn": self.turn},
        )
        write_json(
            self.output_dir / "state.json",
            {
                "profile": self.profile,
                "turn": self.turn,
                "model_calls": self.model_calls,
                "tools": self.tools.checkpoint(),
                "focus_event_ids": self.focus_event_ids,
                "pending_observation_ids": [o.observation_id for o in self.pending_media],
                "conversation": self.conversation.checkpoint(),
                "progress_memory": self.progress_memory.checkpoint(),
                "pending_calls": self.pending_calls,
                "reject_pending_batch": self.reject_pending_batch,
                "call_results": self.call_results,
                "request_ready": self.request_ready,
                "request_observation_ids": self.request_observation_ids,
            },
        )

    def _restore(self) -> None:
        state = json.loads((self.output_dir / "state.json").read_text(encoding="utf-8"))
        if state["profile"] != self.profile:
            raise ValueError("Resume requires the same video, model, prompt and task constraints")
        self.tools.restore(state["tools"])
        self.turn = state["turn"]
        self.model_calls = state["model_calls"]
        self.focus_event_ids = state["focus_event_ids"]
        self.pending_media = [self.tools.observations[k] for k in state["pending_observation_ids"]]
        self.conversation = Conversation(**state["conversation"])
        self.progress_memory = ProgressMemory.restore(state["progress_memory"])
        self.pending_calls = state["pending_calls"]
        self.reject_pending_batch = state["reject_pending_batch"]
        self.call_results = state["call_results"]
        self.request_ready = state["request_ready"]
        self.request_observation_ids = state["request_observation_ids"]

    def _context(self, transmitted_ids=()) -> dict:
        phase = (
            "selection"
            if self.conversation.phase == "selection" or self.tools.selection_ready()
            else "analysis"
        )
        candidates = self.tools.candidate_ledger()
        context = {
            "phase": phase,
            "task": self.task,
            "limits": {
                k: self.profile[k]
                for k in (
                    "max_highlights",
                    "min_clip_sec",
                    "max_clip_sec",
                    "total_duration",
                    "allow_overlap",
                    "video_fps",
                )
            },
            "progress": self.tools.progress(),
            "selection_ledger": {
                "candidates": candidates,
                "total": len(candidates),
                "choices": [c.model_dump(mode="json") for c in self.tools.selection.decisions]
                if self.tools.selection
                else [],
            },
            "last_tool_results": [
                {"tool": item["call"]["name"], "result": item["result"]}
                for item in self.call_results
            ]
            if phase == self.conversation.phase
            else [],
            "stagnant_steps": self.progress_memory.stagnant_steps,
            "story_so_far": self.tools.story_so_far,
            "unrecorded_readings": [
                self.tools.readings[o.observation_id]
                for o in self.tools.unacknowledged_observations()
                if o.observation_id in self.tools.readings
                and o.observation_id not in transmitted_ids
            ],
        }
        if phase == "selection":
            source_ids = dict.fromkeys(
                span["evidence_id"]
                for candidate in candidates
                for span in candidate["required_spans"]
            )
            context["source_notes"] = [
                {
                    "observation_id": key,
                    "src_start_sec": self.tools.observations[key].src_start_sec,
                    "src_end_sec": self.tools.observations[key].src_end_sec,
                    "uncertainties": self.tools.readings[key]["uncertainties"],
                }
                for key in source_ids
                if key in self.tools.readings and self.tools.readings[key]["uncertainties"]
            ]
            context["repair_events"] = [
                self.tools._event_view(self.tools.events[key])
                for key in self.tools.unresolved_events()[:20]
            ]
            checks = [
                {
                    "observation_id": key,
                    "src_start_sec": observation.src_start_sec,
                    "src_end_sec": observation.src_end_sec,
                    "record": record,
                }
                for key, record in self.tools.acknowledged.items()
                if (observation := self.tools.observations[key]).page_id is None
            ]
            context["inspection_memory"] = {"records": checks[-20:], "total": len(checks)}
            return context
        keys = list(dict.fromkeys([*self.tools.unresolved_events(), *self.focus_event_ids]))
        working_events = [
            self.tools._event_view(self.tools.events[key])
            for key in keys[:20]
            if key in self.tools.events
        ]
        evidence_ids = {
            span["evidence_id"]
            for item in working_events
            for span in item["event"]["required_spans"]
        }
        observations = [
            self.tools.observations[key] for key in evidence_ids if key in self.tools.observations
        ]
        rows = list(self.tools.read_memory.rows.values())
        relevant = [
            row
            for row in rows
            if any(
                row["end_sec"] > o.src_start_sec and row["start_sec"] < o.src_end_sec
                for o in observations
            )
        ]
        queries = [
            {
                "query_id": key,
                "query": value["query"],
                "read_count": len(value["row_ids"]),
                "total": value["total"],
                "complete": value["complete"],
            }
            for key, value in self.tools.read_memory.queries.items()
        ]
        return {
            **context,
            "working_events": working_events,
            "event_index": {
                "total": len(self.tools.events),
                "working_count": len(working_events),
                "remaining_count": max(0, len(self.tools.events) - len(working_events)),
            },
            "source_observations": [
                {
                    "observation_id": o.observation_id,
                    "src_start_sec": o.src_start_sec,
                    "src_end_sec": o.src_end_sec,
                    "record": self.tools.acknowledged.get(o.observation_id),
                }
                for o in observations
            ],
            "recent_observations": [
                {"observation_id": key, **value}
                for key, value in list(self.tools.acknowledged.items())[-20:]
            ],
            "transcript_memory": {
                "queries": queries[-20:],
                "query_count": len(queries),
                "rows": relevant[-100:],
                "total_rows_read": len(rows),
            },
        }

    def _remember_events(self, result: dict) -> None:
        """Keep records the model actually read or wrote, not a relevance-ranked subset."""
        views = [result] if "event" in result else result.get("events", [])
        if views:
            self.focus_event_ids = list(
                dict.fromkeys([*[view["event"]["id"] for view in views], *self.focus_event_ids])
            )[:20]

    def _media_parts(
        self, observations: list[VideoObservation]
    ) -> list[tuple[VideoObservation, list[dict]]]:
        def read(observation):
            key = observation.observation_id
            question = self.tools.inspections.get(key, {}).get("question", SCAN_QUESTION)
            if key not in self.tools.readings:
                reading = self.perception.observe(
                    observation,
                    question,
                    invoke=lambda *args, **kwargs: self._invoke_perception(key, *args, **kwargs),
                )
                with self._request_lock:
                    self.tools.readings[key] = reading
                    self._trace("video_observation", **reading)
                    self._save()
            return observation, self._observation_frames(observation)

        with ThreadPoolExecutor(max_workers=MAX_PAGE_BATCH_ITEMS) as pool:
            materials = list(pool.map(read, observations))
        batches = []
        for observation, frames in materials:
            transcript_rows = []
            if (
                self.tools.transcript is not None
                and self.tools.transcript.subtitle_path is not None
            ):
                arguments = {
                    "mode": "exact",
                    "query": "",
                    "start_sec": observation.src_start_sec,
                    "end_sec": observation.src_end_sec,
                    "offset": 0,
                    "limit": len(self.tools.transcript.load()),
                }
                result = self.tools.transcript.search(
                    **{key: value for key, value in arguments.items() if key != "mode"}
                )
                self.tools.read_memory.record(arguments, result)
                transcript_rows = result["matches"]
                self._trace(
                    "page_transcript",
                    observation_id=observation.observation_id,
                    row_ids=[row["row_id"] for row in transcript_rows],
                )
            parts = []
            parts.append(
                text_part(
                    json.dumps(
                        {
                            "observation_id": observation.observation_id,
                            "src_start_sec": observation.src_start_sec,
                            "src_end_sec": observation.src_end_sec,
                            "reading": self.tools.readings[observation.observation_id],
                            "transcript": transcript_rows,
                            "note": "音画观察是定位材料；原帧用于直接核对画面。时间均为原片秒数。",
                        },
                        ensure_ascii=False,
                    )
                )
            )
            for frame in frames:
                parts.extend(
                    [
                        text_part(f"原片帧 {frame['timestamp_sec']:.3f} 秒"),
                        image_part(frame["path"]),
                    ]
                )
            batches.append((observation, parts))
        return batches

    def _invoke_perception(self, observation_id, *args, **kwargs):
        started = time.perf_counter()
        reply = self._invoke(
            self.perception.client,
            *args,
            request_context={"phase": "video_observation", "observation_id": observation_id},
            **kwargs,
        )
        with self._request_lock:
            self._trace(
                "perception_model",
                observation_id=observation_id,
                usage=reply["usage"],
                model_version=reply["model_version"],
                thinking_returned=reply["thinking_returned"],
                elapsed_sec=time.perf_counter() - started,
            )
        return reply

    def _request_limits(self, context, observation_ids):
        instructions = SELECTION_PROMPT if context["phase"] == "selection" else ANALYSIS_PROMPT
        overhead = json.dumps(
            {"instructions": instructions, "tools": self.tools.declarations(observation_ids)},
            ensure_ascii=False,
        )
        return {
            "token_budget": self.context_token_budget - 16_384 - len(overhead),
            "byte_budget": self.context_byte_budget - len(overhead.encode()),
            "phase": context["phase"],
        }

    def _transmission_batch(self, observations, parts):
        """Pack complete requests, including readings, frames, state and tool schemas."""
        selected = []
        for observation, material in self._media_parts(observations):
            ids = [o.observation_id for o in [*selected, observation]]
            context = self._context(ids)
            if not self.conversation.fits(
                context, [*parts, *material], **self._request_limits(context, ids)
            ):
                if not selected:
                    raise ValueError("单段观察及当前状态超过请求预算，请缩短视频页面或提高预算。")
                break
            selected.append(observation)
            parts.extend(material)
        return selected

    def _observation_frames(self, observation):
        count = max(1, math.ceil(observation.duration_sec * self.frame_fps))
        times = [
            observation.src_start_sec + i * observation.duration_sec / count for i in range(count)
        ]
        return self.tools.evidence.frames(times, width=768)

    def _invoke(self, client, *args, request_context=None, **kwargs):
        """Journal every attempt; retry only the same request on a transient failure."""
        context = request_context or {"phase": self.conversation.phase}
        for attempt in range(1, self.max_request_attempts + 1):
            with self._request_lock:
                self.model_calls += 1
                model_call = self.model_calls
                self._save()
                self._trace(
                    "request_attempt",
                    model_call=model_call,
                    attempt=attempt,
                    model=client.model,
                    **context,
                )
            try:
                return client.generate(*args, **kwargs)
            except (OpenAIClientError, GeminiClientError) as exc:
                with self._request_lock:
                    self._trace(
                        "request_error",
                        model_call=model_call,
                        retryable=exc.retryable,
                        message=str(exc),
                        **context,
                    )
                if not exc.retryable or attempt == self.max_request_attempts:
                    raise
                self.on_progress(
                    f"请求暂时失败，重试同一请求（{attempt + 1}/{self.max_request_attempts}）。"
                )
                time.sleep(attempt)
        raise AssertionError("Request attempt budget must be positive")

    def _review_pending(self) -> None:
        plans = self.tools.pending_reviews()
        if not plans:
            return
        for plan in plans:
            self.on_progress(
                f"复核实际成片 {plan.event_id}: {plan.start_sec:.2f}–{plan.end_sec:.2f}s"
            )

        def invoke_review(plan):
            started = time.perf_counter()
            response = self.perception.review(
                plan.media_path,
                self.task,
                fps=self.video_fps,
                start_sec=plan.start_sec,
                end_sec=plan.end_sec,
                invoke=lambda *args, **kwargs: self._invoke(
                    self.perception.client,
                    *args,
                    request_context={"phase": "clip_review", "plan_id": plan.id},
                    **kwargs,
                ),
            )
            return response, time.perf_counter() - started

        failure = None
        with ThreadPoolExecutor(max_workers=min(MAX_REVIEW_WORKERS, len(plans))) as pool:
            futures = {pool.submit(invoke_review, plan): plan for plan in plans}
            for future in as_completed(futures):
                plan = futures[future]
                # Workers journal request attempts while the main thread commits results.
                # Serialize both writes, and preserve each success even if a peer fails.
                with self._request_lock:
                    try:
                        response, elapsed_sec = future.result()
                        review = response["review"]
                        self.tools.record_review(plan.event_id, review)
                        review_information = self.progress_memory.record(
                            self.tools.information_facts(), count_stagnation=False
                        )
                        self._trace(
                            "clip_review",
                            plan_id=plan.id,
                            media_path=str(plan.media_path),
                            review=review.model_dump(mode="json"),
                            usage=response["usage"],
                            model_version=response["model_version"],
                            thinking_returned=response["thinking_returned"],
                            elapsed_sec=elapsed_sec,
                            information_added=review_information,
                        )
                        self._save()
                    except Exception as exc:
                        self._trace(
                            "clip_review_error",
                            plan_id=plan.id,
                            error_type=type(exc).__name__,
                            message=str(exc),
                        )
                        if failure is None:
                            failure = exc
        if failure is not None:
            raise failure

    def _execute_pending_calls(self) -> None:
        # A received model batch is journaled before tools run. Each committed result
        # removes exactly one pending call, so a restart never replays that mutation.
        if self.reject_pending_batch:
            rejected = self.pending_calls
            message = "每轮只能调用一个工具。请根据当前状态选择唯一的下一步动作。"
            result = {
                "status": "invalid_tool_request",
                "message": message,
                "information_added": 0,
            }
            self.pending_calls = []
            self.reject_pending_batch = False
            self.call_results = [{"call": call, "result": result} for call in rejected]
            stagnant = self.progress_memory.record(self.tools.information_facts())
            for item in self.call_results:
                item["result"] = {
                    **item["result"],
                    "stagnant_steps": stagnant,
                }
            self._save()
            self._trace(
                "tool_batch_rejected",
                names=[call["name"] for call in rejected],
                result=self.call_results[0]["result"],
            )
        while self.pending_calls:
            call = self.pending_calls[0]
            name, args = call["name"], call.get("args", {})
            tool_started = time.perf_counter()
            labels = {
                "record_observations": "保存剧情观察与变化",
                "update_event": "整理候选片段与必要上下文",
                "read_state": "回查已保存的剧情记录",
                "read_frames": "查看原片画面细节",
                "read_text": "读取画面文字",
                "select_highlights": "比较全部候选并确定推荐",
                "propose_highlights": "核对本地模型的位置线索",
            }
            self.on_progress(
                args.get("question") or args.get("query") or labels.get(name, "核对视频材料")
            )
            try:
                result, observations = self.tools.execute(name, args)
                queued = {o.observation_id: o for o in [*self.pending_media, *observations]}
                self.pending_media = list(queued.values())
                self._remember_events(result)
                error = False
            except (ValueError, KeyError) as exc:
                if isinstance(exc, ValidationError):
                    result = {
                        "status": "invalid_tool_request",
                        "message": "参数未通过校验，请按 fields 修正后重试。",
                        "fields": [
                            {
                                "path": ".".join(map(str, issue["loc"])),
                                "code": issue["type"],
                                "message": issue["msg"],
                            }
                            for issue in exc.errors(
                                include_url=False,
                                include_context=False,
                                include_input=False,
                            )
                        ],
                    }
                else:
                    result = {"status": "invalid_tool_request", "message": str(exc)}
                error = True
            added = self.progress_memory.record(self.tools.information_facts())
            result = {
                **result,
                "information_added": added,
                "stagnant_steps": self.progress_memory.stagnant_steps,
            }
            self.call_results.append({"call": call, "result": result})
            self.pending_calls.pop(0)
            self._save()
            self._trace(
                "tool",
                call_id=call["id"],
                name=name,
                arguments=args,
                result=result,
                error=error,
                elapsed_sec=time.perf_counter() - tool_started,
            )

    def _prepare_request(self) -> list[dict]:
        if not self.request_ready:
            parts = [
                {
                    "type": "function_call_output",
                    "call_id": item["call"]["id"],
                    "output": json.dumps(item["result"], ensure_ascii=False),
                }
                for item in self.call_results
            ]
            # Every observed video is digested before another evidence action.
            outstanding = self.tools.unacknowledged_observations()
            for frame in self.tools.pending_frames:
                parts.extend(
                    [text_part(json.dumps(frame, ensure_ascii=False)), image_part(frame["path"])]
                )
            transmitted = self._transmission_batch(outstanding or self.pending_media, parts)
            ids = [o.observation_id for o in transmitted]
            context = self._context(ids)
            contents = self.conversation.prepare(
                context, parts, **self._request_limits(context, ids)
            )
            self.request_observation_ids = [o.observation_id for o in transmitted]
            self.request_ready = True
            self._save()
        else:
            contents = hydrate(self.conversation.contents)
        return contents

    def _controller_turn(self, contents: list[dict]) -> None:
        self.on_progress(
            f"Agent 第 {self.turn + 1} 轮：全片检查 {self.tools.progress()['scan_coverage']:.0%}"
        )
        started = time.perf_counter()
        required_ids = list(
            dict.fromkeys(
                [
                    *[o.observation_id for o in self.tools.unacknowledged_observations()],
                    *self.request_observation_ids,
                ]
            )
        )
        declarations = self.tools.declarations(required_ids or None)
        reply = self._invoke(
            self.client,
            SELECTION_PROMPT if self.conversation.phase == "selection" else ANALYSIS_PROMPT,
            contents,
            tools=declarations,
            max_output_tokens=16384,
        )
        self.turn += 1
        delivered_ids = list(self.request_observation_ids)
        delivered = [self.tools.observations[key] for key in delivered_ids]
        self.tools.mark_delivered(delivered)
        delivered_information = (
            self.progress_memory.record(self.tools.information_facts(), count_stagnation=False)
            if delivered
            else 0
        )
        self.pending_media = [
            o for o in self.pending_media if o.observation_id not in self.request_observation_ids
        ]
        self.conversation.contents.extend(reply["output"])
        self.tools.pending_frames = []
        self.pending_calls = list(reply["function_calls"])
        self.reject_pending_batch = len(self.pending_calls) > 1
        self.call_results = []
        self.request_ready = False
        self.request_observation_ids = []
        self._save()
        self._trace(
            "model",
            calls=reply["function_calls"],
            usage=reply["usage"],
            model_version=reply["model_version"],
            elapsed_sec=time.perf_counter() - started,
            phase=self.conversation.phase,
            reasoning_effort=self.client.effort,
            response_id=reply["response_id"],
            first_event_sec=reply["first_event_sec"],
            delivered_observation_ids=delivered_ids,
            delivered_information_added=delivered_information,
            available_tools=[item["name"] for item in declarations],
        )

    def run(self, *, resume: bool = False, max_turns: int | None = None) -> dict:
        if max_turns is not None and max_turns < 1:
            raise ValueError("max_turns must be positive when explicitly specified")
        if resume:
            self._restore()
        elif (self.output_dir / "state.json").exists():
            raise ValueError(
                "Output already contains a run; use --resume or a new output directory"
            )
        self._save()
        stop_reason = "complete"
        start_turn = self.turn
        try:
            while not self.tools.finished:
                self._execute_pending_calls()
                if self.tools.finished:
                    break
                self.tools.prepare_reviews()
                self._save()
                self._review_pending()
                if self.progress_memory.stagnant_steps >= self.max_stagnant_steps:
                    stop_reason = "repeated_action_without_progress"
                    break
                if max_turns is not None and self.turn - start_turn >= max_turns:
                    stop_reason = "turn_limit"
                    break
                if (
                    not self.pending_media
                    and not self.tools.unacknowledged_observations()
                    and len(self.tools.completed_pages) < len(self.tools.evidence.pages)
                ):
                    observations = self.tools.scan_next_observations(
                        max_items=MAX_PAGE_BATCH_ITEMS,
                        max_encoded_bytes=MAX_PAGE_BATCH_ENCODED_BYTES,
                        observation_size=self.perception.encoded_size,
                    )
                    self.pending_media.extend(observations)
                    self._trace(
                        "page_batch",
                        observation_ids=[item.observation_id for item in observations],
                    )
                    self._save()
                contents = self._prepare_request()
                self._controller_turn(contents)
                if not self.pending_calls:
                    self._trace("missing_tool_call")
                    stop_reason = "missing_tool_call"
                    break
        except KeyboardInterrupt:
            stop_reason = "user_stopped"
        except Exception as exc:
            # Provider errors exclude credentials and opaque request bodies.
            stop_reason = "execution_error"
            self._trace("execution_error", error_type=type(exc).__name__, message=str(exc))
            self.on_progress(f"运行中断：{type(exc).__name__}: {exc}")
        finally:
            self._save()
        return self._result(stop_reason)

    def _result(self, stop_reason: str) -> dict:
        selection = (
            self.tools.selection
            if self.tools.finished
            else SelectionResult(selected=[], decisions=[])
        )
        complete = self.tools.finished
        selection_choices = {
            decision.event_id: decision for decision in selection.decisions if decision.selected
        }
        selected = sorted(
            selection.selected,
            key=lambda plan: selection_choices[plan.event_id].score,
            reverse=True,
        )
        result = {
            "completion": "complete" if complete else "partial",
            "message": "已检查全片，推荐片段已生成，等待人工复核。"
            if complete
            else "分析尚未完成，可从检查点继续。",
            "video": {
                "video_id": self.tools.evidence.media_id,
                "duration_sec": self.tools.evidence.video_info.duration_sec,
            },
            "highlights": [
                {
                    "id": plan.id,
                    "start_sec": plan.start_sec,
                    "end_sec": plan.end_sec,
                    "score": selection_choices[plan.event_id].score,
                    "description": selection_choices[plan.event_id].description,
                    "reason": selection_choices[plan.event_id].reason,
                    "highlight_type": selection_choices[plan.event_id].highlight_type,
                    "review_status": "pending",
                    "clip_path": self._export_clip(plan),
                }
                for plan in selected
            ],
            "analysis": {
                **self.tools.progress(),
                "stop_reason": stop_reason,
                "turns": self.turn,
                "model_calls": self.model_calls,
            },
        }
        write_json(self.output_dir / "selection.json", selection.model_dump(mode="json"))
        return result

    def _export_clip(self, plan) -> str:
        folder = self.output_dir / "clips"
        folder.mkdir(exist_ok=True)
        name = hashlib.sha256(plan.id.encode()).hexdigest()[:24] + ".mp4"
        target = folder / name
        source_hash = hashlib.sha256(Path(plan.media_path).read_bytes()).hexdigest()
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != source_hash:
            temp = folder / (uuid4().hex + ".tmp")
            try:
                shutil.copyfile(plan.media_path, temp)
                if hashlib.sha256(temp.read_bytes()).hexdigest() != source_hash:
                    raise ValueError("交付片段与已核对媒体不一致。")
                temp.replace(target)
            finally:
                temp.unlink(missing_ok=True)
        return str(target.relative_to(self.output_dir))

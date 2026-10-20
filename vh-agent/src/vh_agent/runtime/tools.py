"""Tool execution and durable state: analysis -> selection -> review -> complete."""

from __future__ import annotations

import copy
from pathlib import Path

from .contracts import (
    TOOL_INPUTS,
    DetectInput,
    FramesInput,
    MemoryInput,
    SaveInput,
    SearchInput,
    SubmitInput,
    WatchInput,
    tool_declarations,
)
from .evidence import EvidenceStore, VideoObservation, VideoPage
from .finalization import (
    ClipPlan,
    EventRecord,
    ReviewResult,
    SelectionChoice,
    SelectionResult,
    attach_media,
    choose_clips,
    confirm_clip,
    draft_plan,
    select_clips,
)
from .state import ReadMemory, identity


class VideoTools:
    def __init__(
        self,
        evidence: EvidenceStore,
        output_dir: Path,
        *,
        max_highlights=12,
        total_duration=None,
        allow_overlap=True,
        max_clip_sec=24,
        min_clip_sec=3,
        transcript=None,
        detector=None,
        search=None,
        frame_text=None,
    ):
        self.evidence, self.output_dir = evidence, output_dir
        self.max_highlights, self.total_duration, self.allow_overlap = (
            max_highlights,
            total_duration,
            allow_overlap,
        )
        self.max_clip_sec, self.min_clip_sec = max_clip_sec, min_clip_sec
        self.transcript, self.detector, self.search, self.frame_text = (
            transcript,
            detector,
            search,
            frame_text,
        )
        self.story_so_far = ""
        self.inspections, self.readings, self.detections = {}, {}, {}
        self.pending_frames = []
        self.frame_readings = {}
        self.memory_reads = set()
        self.presented_reviews = set()
        self.confirmations = {}
        self.events: dict[str, EventRecord] = {}
        self.editorial = {}
        self.plans: dict[str, ClipPlan] = {}
        self.observations: dict[str, VideoObservation] = {}
        self.delivered: set[str] = set()
        self.completed_pages: set[str] = set()
        self.read_memory = ReadMemory()
        self.finished = False
        self.selection = None

    @property
    def phase(self):
        if self.finished:
            return "complete"
        if self.selection is not None:
            return (
                "selection"
                if any(self.plans[p.event_id].issues for p in self.selection.selected)
                else "review"
            )
        return "selection" if self.discovery_complete() else "analysis"

    def checkpoint(self):
        return {
            "story_so_far": self.story_so_far,
            "inspections": copy.deepcopy(self.inspections),
            "readings": copy.deepcopy(self.readings),
            "detections": copy.deepcopy(self.detections),
            "pending_frames": copy.deepcopy(self.pending_frames),
            "frame_readings": copy.deepcopy(self.frame_readings),
            "memory_reads": sorted(self.memory_reads),
            "presented_reviews": sorted(self.presented_reviews),
            "confirmations": copy.deepcopy(self.confirmations),
            "events": [e.model_dump(mode="json") for e in self.events.values()],
            "editorial": copy.deepcopy(self.editorial),
            "plans": [p.model_dump(mode="json") for p in self.plans.values()],
            "observations": [o.model_dump(mode="json") for o in self.observations.values()],
            "pages": [p.model_dump(mode="json") for p in self.evidence.pages],
            "delivered": sorted(self.delivered),
            "completed_pages": sorted(self.completed_pages),
            "read_memory": self.read_memory.checkpoint(),
            "selection": self.selection.model_dump(mode="json") if self.selection else None,
            "finished": self.finished,
        }

    def restore(self, state):
        self.story_so_far = state["story_so_far"]
        for key in ("inspections", "readings", "detections", "pending_frames", "editorial"):
            setattr(self, key, copy.deepcopy(state[key]))
        self.evidence.restore_pages([VideoPage.model_validate(p) for p in state["pages"]])
        self.events = {e.id: e for row in state["events"] if (e := EventRecord.model_validate(row))}
        self.plans = {
            p.event_id: p for row in state["plans"] if (p := ClipPlan.model_validate(row))
        }
        self.observations = {
            o.observation_id: o
            for row in state["observations"]
            if (o := VideoObservation.model_validate(row))
        }
        self.delivered, self.completed_pages = (
            set(state["delivered"]),
            set(state["completed_pages"]),
        )
        self.frame_readings = copy.deepcopy(state["frame_readings"])
        self.memory_reads = set(state["memory_reads"])
        self.presented_reviews = set(state["presented_reviews"])
        self.confirmations = copy.deepcopy(state["confirmations"])
        self.read_memory = ReadMemory(**state["read_memory"])
        self.selection = (
            SelectionResult.model_validate(state["selection"])
            if state["selection"] is not None
            else None
        )
        self.finished = state["finished"]

    def _covered_pages(self, observation_ids):
        watched = sorted(
            (self.observations[k].src_start_sec, self.observations[k].src_end_sec)
            for k in observation_ids
        )
        covered = set()
        for page in self.evidence.pages:
            end = page.core_start_sec
            for start, stop in watched:
                if start > end + 1e-6:
                    break
                end = max(end, stop)
            if end >= page.core_end_sec - 1e-6:
                covered.add(page.page_id)
        return covered

    def discovery_complete(self, additional_ids=()):
        visible = self.delivered | set(additional_ids)
        return len(self._covered_pages(visible)) == len(self.evidence.pages) and not (
            set(self.observations) - visible
        )

    def selection_ready(self):
        return self.discovery_complete()

    def progress(self):
        pages = self.evidence.pages
        pending = [p for p in pages if p.page_id not in self.completed_pages]
        completed = sum(
            p.core_end_sec - p.core_start_sec for p in pages if p.page_id in self.completed_pages
        )
        return {
            "phase": self.phase,
            "duration_sec": self.evidence.video_info.duration_sec,
            "scan_coverage": completed / self.evidence.video_info.duration_sec,
            "completed_page_count": len(self.completed_pages),
            "page_count": len(pages),
            "next_page": pending[0].model_dump(mode="json") if pending else None,
            "candidate_count": len(self.events),
            "event_count": len(self.events),
            "pending_event_count": sum(
                bool(
                    self.plans[p.event_id].issues
                    or (
                        self.plans[p.event_id].review is not None
                        and self.confirmations.get(p.event_id, {}).get("review_id")
                        != self._review_id(p.event_id)
                    )
                )
                for p in self.selection.selected
            )
            if self.selection is not None
            else 0,
            "pending_proposal_count": 0,
            "pending_observation_count": len(set(self.observations) - self.delivered),
            "pending_review_count": len(self.pending_reviews()),
            "capabilities": {
                "transcript": self.transcript is not None,
                "semantic_search": self.search is not None,
                "local_detector": self.detector is not None,
                "frame_text": self.frame_text is not None,
            },
            "next_action": "交付已完成。"
            if self.finished
            else (
                "继续观看 next_page 或查清当前问题，保存值得采用的候选。"
                if pending
                else "从完整候选池选择，提交粗剪；有复核问题时修订对应候选。"
            ),
        }

    def mark_delivered(self, observations):
        self.delivered.update(o.observation_id for o in observations)
        self.completed_pages = self._covered_pages(self.delivered)

    def declarations(self, required_observation_ids=None):
        result = []
        for item in tool_declarations():
            name, params = item["name"], item["parameters"]
            if name == "detect_highlights" and self.detector is None:
                continue
            if name == "search_video":
                modes = (["semantic"] if self.search is not None else []) + (
                    ["exact"] if self.transcript is not None else []
                )
                if not modes:
                    continue
                params["properties"]["mode"]["enum"] = modes
                params["properties"]["mode"]["default"] = modes[0]
            if name == "read_frames" and self.frame_text is None:
                params["properties"]["with_text"]["enum"] = [False]
                item["description"] = item["description"].replace(
                    "with_text=true 同时识别画面文字及位置。", "本任务返回原帧，文字可从图中核对。"
                )
            if name == "submit_highlights" and not self.discovery_complete(
                required_observation_ids or ()
            ):
                continue
            ids = sorted(self.delivered | set(required_observation_ids or []))
            definitions = params.get("$defs", {})
            if ids and "RequiredSpan" in definitions:
                definitions["RequiredSpan"]["properties"]["evidence_id"]["enum"] = ids
            result.append(item)
        return result

    def execute(self, name, arguments):
        if name not in TOOL_INPUTS:
            raise ValueError(f"工具 {name} 不存在，请使用当前提供的工具")
        args = TOOL_INPUTS[name].model_validate(arguments)
        if isinstance(args, DetectInput):
            if self.detector is None:
                raise ValueError("当前未配置本地检测服务")
            self._check_range(args.start_sec, args.end_sec)
            result = self.detector.detect(args.start_sec, args.end_sec)
            key = identity([args.start_sec, args.end_sec])
            self.detections[key] = {"start_sec": args.start_sec, "end_sec": args.end_sec, **result}
            rows = result["segments"]
            return {
                "segments": rows[args.offset : args.offset + args.limit],
                "total": len(rows),
                "next_offset": args.offset + args.limit
                if args.offset + args.limit < len(rows)
                else None,
                "invalid_segment_count": sum(len(w["invalid_segments"]) for w in result["windows"]),
            }, []
        if isinstance(args, SearchInput):
            end = (
                args.end_sec if args.end_sec is not None else self.evidence.video_info.duration_sec
            )
            self._check_range(args.start_sec, end)
            provider = self.search if args.mode == "semantic" else self.transcript
            if provider is None:
                raise ValueError(f"当前没有 {args.mode} 检索能力")
            return self.read_memory.record(
                args.model_dump(), provider.search(**args.model_dump(exclude={"mode"}))
            ), []
        if isinstance(args, WatchInput):
            self._check_range(args.start_sec, args.end_sec)
            observation = self.evidence.inspect(args.start_sec, args.end_sec)
            key = (
                "obs_"
                + identity([observation.observation_id, args.question, args.sampling_fps])[:24]
            )
            observation = observation.model_copy(
                update={"observation_id": key, "sampling_fps": args.sampling_fps}
            )
            self.inspections[key] = args.model_dump()
            self.observations[key] = observation
            return {
                "status": "ok",
                **observation.model_dump(mode="json", exclude={"path"}),
                "time_coordinate": "原片秒数",
            }, [observation]
        if isinstance(args, FramesInput):
            if args.with_text and self.frame_text is None:
                raise ValueError("当前未配置文字识别")
            frames = self.evidence.frames(args.times, args.region)
            result = self.frame_text.read(frames) if args.with_text else {"frames": frames}
            material = [
                {k: v for k, v in frame.items() if k != "requested_sec"}
                for frame in result["frames"]
            ]
            key = "frames_" + identity([material, args.region, args.with_text])[:24]
            self.frame_readings[key] = {"id": key, **result}
            self.pending_frames.extend(frames)
            return self.frame_readings[key], []
        if isinstance(args, SaveInput):
            return self._save_candidates(args), []
        if isinstance(args, MemoryInput):
            return self._read(args), []
        if isinstance(args, SubmitInput):
            return self._submit(args), []
        raise AssertionError(type(args).__name__)

    def _check_range(self, start, end):
        if not 0 <= start < end <= self.evidence.video_info.duration_sec:
            raise ValueError("范围必须位于原视频内")

    def _plan(self, event):
        plan = self.plans.get(event.id)
        if plan is None:
            plan = draft_plan(
                event,
                self.evidence.video_info.duration_sec,
                max_clip_sec=self.max_clip_sec,
                min_clip_sec=self.min_clip_sec,
            )
        return plan

    def candidate_ledger(self):
        return [self._candidate_view(key) for key in self.events]

    def _candidate_view(self, key):
        event = self.events[key]
        plan = self._plan(event)
        return {
            "id": key,
            "description": event.description,
            "reason": event.reason,
            **self.editorial[key],
            "start_sec": plan.start_sec,
            "end_sec": plan.end_sec,
            "evidence": [s.model_dump(mode="json") for s in event.required_spans],
            "issues": plan.issues,
            "review": plan.review.model_dump(mode="json") if plan.review else None,
        }

    def _save_candidates(self, args):
        snapshot = self.checkpoint()
        saved = []
        try:
            for candidate in args.candidates:
                self._check_range(candidate.start_sec, candidate.end_sec)
                if candidate.id is not None and candidate.id not in self.events:
                    raise ValueError("修订的候选 id 不存在；新候选 id 填 null")
                old = self.events.get(candidate.id)
                if not any(s.role == "decisive" for s in candidate.evidence):
                    raise ValueError("候选至少引用一段 decisive 核心依据")
                for span in candidate.evidence:
                    observation = self.observations.get(span.evidence_id)
                    if observation is None or span.evidence_id not in self.delivered:
                        raise ValueError("evidence 只能引用已经观看的 observation_id")
                    if (
                        not observation.src_start_sec
                        <= span.start_sec
                        < span.end_sec
                        <= observation.src_end_sec
                    ):
                        raise ValueError("依据时间必须位于所引用的观察范围内")
                    if (
                        not candidate.start_sec
                        <= span.start_sec
                        < span.end_sec
                        <= candidate.end_sec
                    ):
                        raise ValueError("候选粗剪必须完整包含所有必要依据")
                values = {
                    "description": candidate.description,
                    "reason": candidate.reason,
                    "status": "supported",
                    "required_spans": candidate.evidence,
                    "proposed_start_sec": candidate.start_sec,
                    "proposed_end_sec": candidate.end_sec,
                }
                if old:
                    values.update(id=old.id, version=old.version)
                event = EventRecord(**values)
                changed = old is None or old.model_dump(exclude={"version"}) != event.model_dump(
                    exclude={"version"}
                )
                if changed and old:
                    event = event.model_copy(update={"version": old.version + 1})
                if changed:
                    previous_plan = self.plans.get(event.id)
                    same_media = (
                        old is not None
                        and old.required_spans == event.required_spans
                        and old.proposed_start_sec == event.proposed_start_sec
                        and old.proposed_end_sec == event.proposed_end_sec
                    )
                    if previous_plan is not None and same_media:
                        self.plans[event.id] = previous_plan.model_copy(
                            update={"event_version": event.version, "status": "draft"}
                        )
                    else:
                        self.plans.pop(event.id, None)
                if (
                    changed
                    or self.editorial.get(event.id, {}).get("highlight_type")
                    != candidate.highlight_type
                ):
                    self.confirmations.pop(event.id, None)
                self.events[event.id] = event
                self.editorial[event.id] = {
                    "highlight_type": candidate.highlight_type,
                    "score": candidate.score,
                }
                saved.append(event.id)
            if args.story_so_far is not None:
                self.story_so_far = args.story_so_far
            if args.candidates:
                self.selection = None
                self.finished = False
        except ValueError:
            self.restore(snapshot)
            raise
        return {
            "status": "ok",
            "candidates": [self._candidate_view(k) for k in saved],
            "story_so_far": self.story_so_far,
        }

    def _read(self, args):
        if args.collection == "candidates":
            records = {k: self._candidate_view(k) for k in self.events}
        elif args.collection == "observations":
            records = {
                k: {
                    "observation_id": k,
                    "start_sec": o.src_start_sec,
                    "end_sec": o.src_end_sec,
                    "delivered": k in self.delivered,
                    "reading": self.readings.get(k),
                }
                for k, o in self.observations.items()
            }
        elif args.collection == "frames":
            records = self.frame_readings
        elif args.collection == "transcript":
            records = self.read_memory.rows
        else:
            records = {
                key: {"id": key, **{k: row[k] for k in ("start_sec", "end_sec", "segments")}}
                for key, row in self.detections.items()
            }
        if args.id is not None:
            if args.id not in records:
                raise ValueError("当前集合没有该 id")
            self.memory_reads.add(identity([args.collection, [args.id]]))
            return {"items": [records[args.id]], "total": 1, "next_offset": None}
        rows = list(records.values())
        read_ids = list(records)[args.offset : args.offset + args.limit]
        if read_ids:
            self.memory_reads.add(identity([args.collection, read_ids]))
        return {
            "items": rows[args.offset : args.offset + args.limit],
            "total": len(rows),
            "next_offset": args.offset + args.limit
            if args.offset + args.limit < len(rows)
            else None,
        }

    def _decisions(self, ids):
        selected = set(ids)
        return [
            SelectionChoice(
                event_id=k,
                selected=k in selected,
                score=self.editorial[k]["score"],
                description=e.description if k in selected else None,
                highlight_type=self.editorial[k]["highlight_type"] if k in selected else None,
                reason=e.reason if k in selected else "未列入本次交付",
            )
            for k, e in self.events.items()
        ]

    def _submit(self, args):
        if not self.discovery_complete():
            raise ValueError("请先看完原片，progress.next_page 给出下一段范围")
        ids = args.candidate_ids
        if len(ids) != len(set(ids)) or any(k not in self.events for k in ids):
            raise ValueError("candidate_ids 必须是已有且不重复的候选 ID")
        plans = [self._plan(event) for event in self.events.values()]
        decisions = self._decisions(ids)
        selection = choose_clips(
            plans,
            self.events,
            self.max_highlights,
            decisions=decisions,
            total_duration=self.total_duration,
            allow_overlap=self.allow_overlap,
        )
        selection = selection.model_copy(
            update={
                "selected": [
                    next(p for p in selection.selected if p.event_id == key) for key in ids
                ]
            }
        )
        by_id = {p.event_id: p for p in plans}
        acknowledgements = {}
        for confirmation in args.confirmations:
            key = confirmation.candidate_id
            if key not in ids or key in acknowledgements:
                raise ValueError("每份确认必须对应一个本次采用且未重复的候选")
            token = self._review_id(key, by_id[key])
            if (
                token is None
                or confirmation.review_id != token
                or token not in self.presented_reviews
            ):
                raise ValueError("请先读取当前候选的实际复核，再原样提交 review_id")
            if by_id[key].issues:
                raise ValueError("请先修复片段的边界问题")
            acknowledgements[key] = confirmation.model_dump()
        self.confirmations.update(acknowledgements)
        self.plans.update({p.event_id: p for p in plans})
        self.selection = selection
        self.finished = False
        return {"status": "reviewing", "candidate_ids": ids}

    def prepare_reviews(self):
        if self.selection is None:
            return
        for selected in self.selection.selected:
            plan = self.plans[selected.event_id]
            if plan.status != "draft" or plan.media_path is not None:
                continue
            media = self.evidence.render_clip(plan.start_sec, plan.end_sec)
            self.plans[plan.event_id] = attach_media(
                plan,
                start_sec=media.src_start_sec,
                end_sec=media.src_end_sec,
                media_path=media.path,
                video_duration=self.evidence.video_info.duration_sec,
                max_clip_sec=self.max_clip_sec,
            )

    def pending_reviews(self):
        if self.selection is None:
            return []
        return [
            self.plans[p.event_id]
            for p in self.selection.selected
            if self.plans[p.event_id].status == "draft"
            and self.plans[p.event_id].review is None
            and self.plans[p.event_id].media_path is not None
            and not self.plans[p.event_id].issues
        ]

    def record_review(self, event_id, review: ReviewResult):
        plan = self.plans[event_id]
        if any(
            issue.at_sec is not None and issue.at_sec > plan.end_sec - plan.start_sec
            for issue in review.issues
        ):
            raise ValueError("复核问题时间超出当前片段")
        self.plans[event_id] = plan.model_copy(update={"review": review})

    def _review_id(self, key, plan=None):
        plan = plan if plan is not None else self.plans.get(key)
        if plan is None or plan.review is None or plan.media_path is None:
            return None
        return (
            "review_"
            + identity(
                [
                    key,
                    self.events[key].model_dump(),
                    self.editorial[key]["highlight_type"],
                    str(plan.media_path),
                    plan.start_sec,
                    plan.end_sec,
                    plan.review.model_dump(),
                ]
            )[:24]
        )

    def mark_reviews_presented(self, report):
        if report and report["status"] == "awaiting_confirmation":
            self.presented_reviews.update(r["review_id"] for r in report["reviews"])

    def finish_submission(self):
        if self.selection is None:
            return None
        ids = [p.event_id for p in self.selection.selected]
        problems, reviews = [], []
        for key in ids:
            plan = self.plans[key]
            if plan.issues:
                problems.append({"candidate_id": key, "issues": plan.issues})
            elif plan.review is None or plan.media_path is None:
                return {"status": "reviewing", "candidate_ids": ids}
            elif self.confirmations.get(key, {}).get("review_id") != self._review_id(key):
                reviews.append(
                    {
                        "candidate_id": key,
                        "review_id": self._review_id(key),
                        "description": self.events[key].description,
                        "start_sec": plan.start_sec,
                        "end_sec": plan.end_sec,
                        **plan.review.model_dump(mode="json"),
                    }
                )
        if problems:
            return {"status": "needs_revision", "problems": problems}
        if reviews:
            return {"status": "awaiting_confirmation", "reviews": reviews}
        for key in ids:
            plan = self.plans[key]
            if plan.status != "ready":
                self.plans[key] = confirm_clip(plan, self.events[key], plan.review)
        selection = select_clips(
            list(self.plans.values()),
            self.events,
            self.max_highlights,
            decisions=self.selection.decisions,
            total_duration=self.total_duration,
            allow_overlap=self.allow_overlap,
        )
        self.selection = selection.model_copy(update={"selected": [self.plans[k] for k in ids]})
        self.finished = True
        return {"status": "complete", "candidate_ids": ids}

    def information_facts(self):
        facts = {f"text:{k}" for k in self.read_memory.rows} | {
            f"query:{k}" for k in self.read_memory.queries
        }
        facts.update(
            "video:"
            + identity(
                [
                    self.observations[k].src_start_sec,
                    self.observations[k].src_end_sec,
                    self.observations[k].sampling_fps,
                    self.inspections[k]["question"],
                ]
            )
            for k in self.delivered
        )
        facts.update(
            "candidate:"
            + identity([key, event.model_dump(exclude={"reason", "description", "version"})])
            for key, event in self.events.items()
        )
        facts.update("frames:" + k for k in self.frame_readings)
        facts.update("memory:" + k for k in self.memory_reads)
        facts.update(
            "confirmation:" + identity([k, value["review_id"]])
            for k, value in self.confirmations.items()
        )
        facts.update("detection:" + k for k in self.detections)
        facts.update(
            "review:" + identity([k, p.review.model_dump()])
            for k, p in self.plans.items()
            if p.review is not None
        )
        if self.selection:
            facts.add("selection:" + identity([p.event_id for p in self.selection.selected]))
        if self.finished:
            facts.add("finished")
        return facts

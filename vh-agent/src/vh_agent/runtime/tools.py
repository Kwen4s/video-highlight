"""Validated video tools and durable working state for the ReAct loop."""

from __future__ import annotations

import copy
from pathlib import Path

from .contracts import (
    TOOL_INPUTS,
    EventInput,
    FramesInput,
    InspectInput,
    ProposalInput,
    ReadInput,
    RecordInput,
    SearchInput,
    SelectInput,
    UpdateInput,
    tool_declarations,
)
from .evidence import EvidenceStore, VideoObservation, VideoPage
from .finalization import (
    ClipPlan,
    EventRecord,
    ReviewResult,
    SelectionResult,
    attach_media,
    choose_clips,
    confirm_clip,
    draft_plan,
    select_clips,
)
from .state import ReadMemory, identity

MAX_OBSERVATION_BYTES = 18_000_000  # Each audiovisual request stays below Gemini's 20 MB limit.


class VideoTools:
    """Coverage is earned by a successful model observation and explicit acknowledgement."""

    def __init__(
        self,
        evidence: EvidenceStore,
        output_dir: Path,
        *,
        max_highlights: int | None = 12,
        total_duration: float | None = None,
        allow_overlap: bool = True,
        max_clip_sec: float = 24,
        min_clip_sec: float = 3,
        transcript=None,
        local_proposals=None,
        search=None,
        frame_text=None,
    ) -> None:
        self.evidence = evidence
        self.output_dir = output_dir
        self.max_highlights = max_highlights
        self.total_duration = total_duration
        self.allow_overlap = allow_overlap
        self.transcript = transcript
        self.local_proposals = local_proposals
        self.search = search
        self.frame_text = frame_text
        self.story_so_far = ""
        self.inspections = {}
        self.readings = {}
        self.pending_frames = []
        self.proposals: dict[str, dict] = {}
        self.proposals_loaded = False
        self.max_clip_sec = max_clip_sec
        self.min_clip_sec = min_clip_sec
        self.events: dict[str, EventRecord] = {}
        self.plans: dict[str, ClipPlan] = {}
        self.observations: dict[str, VideoObservation] = {}
        self.delivered: set[str] = set()
        self.acknowledged: dict[str, dict] = {}
        self.read_memory = ReadMemory()
        self.completed_pages: set[str] = set()
        self.finished = False
        self.selection = None

    def checkpoint(self) -> dict:
        return {
            "story_so_far": self.story_so_far,
            "inspections": copy.deepcopy(self.inspections),
            "readings": copy.deepcopy(self.readings),
            "pending_frames": copy.deepcopy(self.pending_frames),
            "proposals": {k: dict(v) for k, v in self.proposals.items()},
            "proposals_loaded": self.proposals_loaded,
            "read_memory": self.read_memory.checkpoint(),
            "selection": self.selection.model_dump(mode="json") if self.selection else None,
            "events": [e.model_dump(mode="json") for e in self.events.values()],
            "pages": [p.model_dump(mode="json") for p in self.evidence.pages],
            "plans": [p.model_dump(mode="json") for p in self.plans.values()],
            "observations": [o.model_dump(mode="json") for o in self.observations.values()],
            "delivered": sorted(self.delivered),
            "acknowledged": dict(self.acknowledged),
            "completed_pages": sorted(self.completed_pages),
            "finished": self.finished,
        }

    def restore(self, state: dict) -> None:
        self.story_so_far = state["story_so_far"]
        self.inspections = copy.deepcopy(state["inspections"])
        self.readings = copy.deepcopy(state["readings"])
        self.pending_frames = copy.deepcopy(state["pending_frames"])
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
        self.proposals = {k: dict(v) for k, v in state["proposals"].items()}
        self.proposals_loaded = state["proposals_loaded"]
        self.selection = (
            SelectionResult.model_validate(state["selection"])
            if state["selection"] is not None
            else None
        )
        self.read_memory = ReadMemory(**state["read_memory"])
        self.delivered = set(state["delivered"])
        self.acknowledged = dict(state["acknowledged"])
        self.completed_pages = set(state["completed_pages"])
        self.finished = state["finished"]

    def progress(self) -> dict:
        pages = self.evidence.pages
        completed_sec = sum(
            p.core_end_sec - p.core_start_sec for p in pages if p.page_id in self.completed_pages
        )
        pending_pages = [p for p in pages if p.page_id not in self.completed_pages]
        unresolved = self.unresolved_events()
        candidates = self.candidate_ledger()
        unacknowledged = self.unacknowledged_observations()
        ready_to_select = self.selection_ready()
        return {
            "duration_sec": self.evidence.video_info.duration_sec,
            "scan_coverage": completed_sec / self.evidence.video_info.duration_sec,
            "completed_page_count": len(self.completed_pages),
            "page_count": len(pages),
            "next_page": pending_pages[0].model_dump(mode="json") if pending_pages else None,
            "event_count": len(self.events),
            "ready_event_ids": [key for key, plan in self.plans.items() if plan.status == "ready"],
            "candidate_count": len(candidates),
            "pending_proposal_count": sum(
                p["status"] == "pending" for p in self.proposals.values()
            ),
            "pending_review_count": len(self.pending_reviews()),
            "capabilities": {
                "transcript": self.transcript is not None,
                "semantic_search": self.search is not None,
                "read_frames": True,
                "read_text": self.frame_text is not None,
                "local_proposals": self.local_proposals is not None,
            },
            "pending_event_count": len(unresolved),
            "pending_observation_count": len(set(self.observations) - self.acknowledged.keys()),
            "pending_event_ids": unresolved[:20],
            "transcript_rows_read": len(self.read_memory.rows),
            "next_action": self._next_action(ready_to_select, unacknowledged),
            "unacknowledged_observations": [
                {
                    "observation_id": key,
                    "page_id": o.page_id,
                    "src_start_sec": o.src_start_sec,
                    "src_end_sec": o.src_end_sec,
                }
                for o in unacknowledged
                for key in [o.observation_id]
            ],
        }

    def unacknowledged_observations(self) -> list[VideoObservation]:
        return [
            observation
            for key, observation in self.observations.items()
            if key in self.delivered and key not in self.acknowledged
        ]

    def selection_ready(self) -> bool:
        return self.discovery_complete()

    def discovery_complete(self) -> bool:
        return (
            len(self.completed_pages) == len(self.evidence.pages)
            and not (set(self.observations) - self.acknowledged.keys())
            and not any(event.status == "pending" for event in self.events.values())
        )

    def _next_action(self, ready_to_select: bool, unacknowledged: list[VideoObservation]) -> str:
        if self.finished:
            return "分析已完成，最终取舍已保存，等待人工编辑。"
        if unacknowledged:
            return "用 record_observations 一次登记本轮收到的全部视频观察。"
        if set(self.observations) - self.acknowledged.keys():
            return "等待尚未送达的观察视频，观看后立即登记。"
        if len(self.completed_pages) < len(self.evidence.pages):
            return "系统将继续送入下一批视频页面。"
        if ready_to_select:
            if self.selection is None:
                return "用 select_highlights 从完整候选池初选；系统只制作并复核拟采用片段。"
            if self.unresolved_events():
                return "结合原片判断复核意见：核心看点受影响时修订；可交给编辑时确认采用，也可舍弃或替换。"
            return "核对拟采用片段的实际复核结果，再用 select_highlights 确认交付。"
        if self.unresolved_events():
            return "处理 pending 事件、失败复核或未制作片段。"
        return "等待系统完成实际片段的独立观看。"

    def unresolved_events(self) -> list[str]:
        pending = []
        selected = self._chosen_ids()
        for event in self.events.values():
            if event.status == "pending":
                pending.append(event.id)
            if event.status != "supported" or event.id not in selected:
                continue
            plan = self.plans.get(event.id)
            if (
                plan is None
                or plan.event_version != event.version
                or plan.review is None
                or plan.media_path is None
                or plan.issues
                or plan.status == "infeasible_duration"
                or (plan.status != "ready" and plan.review.issues)
            ):
                pending.append(event.id)
        return pending

    def _chosen_ids(self) -> set[str]:
        return (
            {c.event_id for c in self.selection.decisions if c.selected}
            if self.selection
            else set()
        )

    def _plan(self, event: EventRecord) -> ClipPlan:
        plan = self.plans.get(event.id)
        if plan is None or plan.event_version != event.version:
            plan = draft_plan(
                event,
                self.evidence.video_info.duration_sec,
                max_clip_sec=self.max_clip_sec,
                min_clip_sec=self.min_clip_sec,
            )
        return plan

    def mark_delivered(self, observations: list[VideoObservation]) -> None:
        self.delivered.update(o.observation_id for o in observations)

    def scan_next_observations(
        self,
        *,
        max_items: int,
        max_encoded_bytes: int,
        observation_size=None,
    ) -> list[VideoObservation]:
        """Prepare the next chronological page batch without spending an agent turn."""
        if max_items <= 0 or max_encoded_bytes <= 0:
            raise ValueError("Observation batch limits must be positive")
        existing_pages = {
            observation.page_id
            for observation in self.observations.values()
            if observation.page_id and observation.observation_id not in self.acknowledged
        }
        observations: list[VideoObservation] = []
        encoded_bytes = 0
        while len(observations) < max_items:
            page = next(
                (
                    item
                    for item in self.evidence.pages
                    if item.page_id not in self.completed_pages
                    and item.page_id not in existing_pages
                ),
                None,
            )
            if page is None:
                break
            observation = self.evidence.scan(page.page_id)
            size = (observation_size or self._encoded_size)(observation)
            if size > MAX_OBSERVATION_BYTES:
                self.evidence.split_page(page.page_id)
                continue
            if observations and encoded_bytes + size > max_encoded_bytes:
                break
            self._observation(observation)
            observations.append(observation)
            existing_pages.add(page.page_id)
            encoded_bytes += size
        return observations

    def prepare_reviews(self) -> None:
        """Only chosen candidates incur rendering and independent video review."""
        if not self.discovery_complete():
            return
        for key in self._chosen_ids():
            event = self.events.get(key)
            if event is None or event.status != "supported":
                continue
            plan = self._plan(event)
            self.plans[key] = plan
            if plan.media_path is not None or plan.status != "draft":
                continue
            media = self.evidence.render_clip(plan.start_sec, plan.end_sec)
            self.plans[key] = attach_media(
                plan,
                start_sec=media.src_start_sec,
                end_sec=media.src_end_sec,
                media_path=media.path,
                video_duration=self.evidence.video_info.duration_sec,
                max_clip_sec=self.max_clip_sec,
            )

    def execute(self, name: str, arguments: dict) -> tuple[dict, list[VideoObservation]]:
        if name not in TOOL_INPUTS:
            raise ValueError(f"工具 {name} 不存在；请使用当前提供的工具。")
        outstanding = self.unacknowledged_observations()
        if outstanding and name != "record_observations":
            raise ValueError(
                "请先用 record_observations 登记本轮收到的全部视频观察，再执行其他操作。"
            )
        args = TOOL_INPUTS[name].model_validate(arguments)
        if isinstance(args, SearchInput):
            payload = args.model_dump(exclude={"mode"})
            if args.mode == "semantic":
                if self.search is None:
                    raise ValueError("当前任务未配置语义检索，可使用 exact 查询字幕。")
                return self.read_memory.record(args.model_dump(), self.search.search(**payload)), []
            if self.transcript is None:
                raise ValueError("当前任务没有字幕检索工具，请直接观看视频。")
            return self.read_memory.record(args.model_dump(), self.transcript.search(**payload)), []
        if isinstance(args, ProposalInput):
            return self._propose(args), []
        if isinstance(args, InspectInput):
            observation = self.evidence.inspect(args.start_sec, args.end_sec)
            if args.event_id and args.event_id not in self.events:
                raise ValueError("关联事件不存在。")
            signature = [
                observation.observation_id,
                args.question,
                args.sampling_fps,
                args.event_id,
            ]
            key = "obs_" + identity(signature)[:24]
            observation = observation.model_copy(
                update={"observation_id": key, "sampling_fps": args.sampling_fps}
            )
            if self._encoded_size(observation) > MAX_OBSERVATION_BYTES:
                raise ValueError("视频区间过大，请缩短 start_sec 到 end_sec 的范围后重试。")
            self.inspections[key] = args.model_dump(mode="json")
            return self._observation(observation)
        if isinstance(args, FramesInput):
            if name == "read_text" and self.frame_text is None:
                raise ValueError("当前任务没有配置可用 OCR。")
            frames = self.evidence.frames(args.times, args.region)
            result = self.frame_text.read(frames) if name == "read_text" else {"frames": frames}
            self.pending_frames.extend(frames)
            return result, []
        if isinstance(args, RecordInput):
            return self._record(args), []
        if isinstance(args, ReadInput):
            return self._read(args), []
        if isinstance(args, UpdateInput):
            if args.event.id not in self.events:
                raise ValueError(
                    "新发现请通过 record_observations 登记；update_event 仅修订已有事件。"
                )
            return self._update(args), []
        if isinstance(args, SelectInput):
            return self._select(args), []
        raise AssertionError(f"Unhandled tool input: {type(args).__name__}")

    def declarations(self, required_observation_ids: list[str] | None = None):
        declarations = tool_declarations()
        evidence_ids = sorted(self.delivered | set(required_observation_ids or []))
        if evidence_ids:
            for declaration in declarations:
                definitions = declaration["parameters"].get("$defs", {})
                if "RequiredSpan" in definitions:
                    definitions["RequiredSpan"]["properties"]["evidence_id"]["enum"] = evidence_ids
        if required_observation_ids:
            record = next(d for d in declarations if d["name"] == "record_observations")
            schema = record["parameters"]
            schema["properties"]["observations"]["minItems"] = len(required_observation_ids)
            schema["properties"]["observations"]["maxItems"] = len(required_observation_ids)
            schema["$defs"]["ObservationRecord"]["properties"]["observation_id"]["enum"] = (
                required_observation_ids
            )
            record["description"] = (
                f"一次登记刚观看的全部观察：{', '.join(required_observation_ids)}。"
                "每个 observation_id 恰好提交一次；先如实保存发现或说明没有候选。"
            )
            return [record]

        # The full pool and independent reviews are already in current state.
        # Keep repair actions available even when technical reviews are clean:
        # the main agent may still find a semantic mismatch.
        if self.selection_ready():
            actions = {"select_highlights", "update_event", "inspect_video", "read_state"}
            actions.add("record_observations")
            if self.transcript is not None or self.search is not None:
                actions.add("search_video")
            actions.add("read_frames")
            if self.frame_text is not None:
                actions.add("read_text")
            if self.local_proposals is not None:
                actions.add("propose_highlights")
            return [d for d in declarations if d["name"] in actions]

        disabled = set()
        if self.transcript is None and self.search is None:
            disabled.add("search_video")
        if self.frame_text is None:
            disabled.add("read_text")
        if self.local_proposals is None:
            disabled.add("propose_highlights")
        disabled.add("select_highlights")
        declarations = [t for t in declarations if t["name"] not in disabled]
        return declarations

    def _propose(self, args: ProposalInput):
        if self.local_proposals is None:
            raise ValueError("当前任务没有本地模型位置线索，请直接观看视频。")
        if not self.proposals_loaded:
            self.proposals = {
                f"proposal_{i}": {**p, "status": "pending"}
                for i, p in enumerate(self.local_proposals())
            }
            self.proposals_loaded = True
            self.finished = False
        if args.action == "resolve":
            proposal = self.proposals.get(args.proposal_id)
            observations = [
                self.observations[key]
                for key in args.observation_ids
                if key in self.observations and key in self.delivered
            ]
            if (
                proposal is None
                or not observations
                or len(observations) != len(args.observation_ids)
            ):
                raise ValueError(
                    "resolve 需要有效的 proposal_id，以及已经观看的视频 observation_ids。"
                )
            if not args.reason or not args.reason.strip():
                raise ValueError("请在 reason 中说明根据视频作出的判断。")
            cursor = proposal["start_sec"]
            for observation in sorted(observations, key=lambda o: o.src_start_sec):
                if observation.src_start_sec > cursor + 1e-6:
                    break
                cursor = max(cursor, observation.src_end_sec)
            if cursor < proposal["end_sec"] - 1e-6:
                raise ValueError("请先观看覆盖整条位置线索的视频，再 resolve。")
            if args.event_id and args.event_id not in self.events:
                raise ValueError("关联的 event_id 不存在；请先登记事件，或检查事件 ID。")
            proposal.update(
                status="linked" if args.event_id else "rejected",
                event_id=args.event_id,
                reason=args.reason,
                observation_ids=args.observation_ids,
            )
        items = list(self.proposals.items())
        return {
            "proposals": [
                {"proposal_id": key, **value}
                for key, value in items[args.offset : args.offset + args.limit]
            ],
            "total": len(items),
            "next_offset": args.offset + args.limit
            if args.offset + args.limit < len(items)
            else None,
        }

    @staticmethod
    def _encoded_size(observation: VideoObservation) -> int:
        return 4 * ((observation.path.stat().st_size + 2) // 3)

    def _observation(self, observation: VideoObservation):
        self.observations[observation.observation_id] = observation
        if observation.observation_id not in self.acknowledged:
            self.finished = False
        public = observation.model_dump(mode="json", exclude={"path"})
        public["status"] = "ok"
        public["time_mapping"] = "原片秒数 = src_start_sec + 本视频内秒数"
        return public, [observation]

    def _read(self, args: ReadInput) -> dict:
        if args.event_id is not None:
            if args.collection != "events" or args.event_id not in self.events:
                raise ValueError("event_id 不存在或 collection 不是 events；请分页读取事件索引。")
            return self._event_view(self.events[args.event_id])
        if args.collection == "readings":
            rows = list(self.readings.values())
        elif args.collection == "events":
            rows = [self._event_view(event) for event in self.events.values()]
        elif args.collection == "candidates":
            rows = self.candidate_ledger()
        elif args.collection == "observations":
            rows = [
                {
                    **o.model_dump(mode="json", exclude={"path"}),
                    "record": self.acknowledged.get(o.observation_id),
                    "delivered": o.observation_id in self.delivered,
                }
                for o in self.observations.values()
            ]
        elif args.collection == "transcript":
            rows = list(self.read_memory.rows.values())
        elif args.collection == "queries":
            rows = [{"query_id": key, **value} for key, value in self.read_memory.queries.items()]
        else:
            rows = [page.model_dump(mode="json") for page in self.evidence.pages]
        end = args.offset + args.limit
        selected_rows = rows[args.offset : end]
        return {
            args.collection: selected_rows,
            "total": len(rows),
            "next_offset": end if end < len(rows) else None,
        }

    def candidate_ledger(self) -> list[dict]:
        rows = []
        for event in self.events.values():
            if event.status != "supported":
                continue
            plan = self._plan(event)
            rows.append(
                {
                    "event_id": event.id,
                    "event_version": event.version,
                    "description": event.description,
                    "reason": event.reason,
                    "required_spans": [
                        span.model_dump(mode="json") for span in event.required_spans
                    ],
                    "start_sec": plan.start_sec,
                    "end_sec": plan.end_sec,
                    "review": plan.review.model_dump(mode="json") if plan.review else None,
                    "issues": plan.issues,
                }
            )
        return rows

    def _record(self, args: RecordInput) -> dict:
        expected_ids = {item.observation_id for item in self.unacknowledged_observations()}
        submitted_ids = {item.observation_id for item in args.observations}
        if submitted_ids != expected_ids:
            raise ValueError("observations 必须恰好登记本轮收到的全部 observation_id。")
        snapshot = self.checkpoint()
        try:
            if args.story_so_far is not None:
                self.story_so_far = args.story_so_far
            recorded: dict[str, list[str]] = {}
            for item in args.observations:
                event_ids = []
                for event in item.findings:
                    # A boundary recheck can revise an earlier event without adding footage
                    # that belongs in its clip. _save_event validates all retained evidence.
                    view = self._save_event(event)
                    event_ids.append(view["event"]["id"])
                previous = self.acknowledged.get(item.observation_id, {})
                event_ids = list(dict.fromkeys([*previous.get("event_ids", []), *event_ids]))
                self.acknowledged[item.observation_id] = {
                    "event_ids": event_ids,
                    "no_event_reason": item.no_event_reason if not event_ids else None,
                }
                recorded[item.observation_id] = event_ids
            watched = sorted(
                (self.observations[key].src_start_sec, self.observations[key].src_end_sec)
                for key in self.acknowledged
            )
            for page in self.evidence.pages:
                covered_end = page.core_start_sec
                for start, end in watched:
                    if start > covered_end + 1e-6:
                        break
                    covered_end = max(covered_end, end)
                if covered_end >= page.core_end_sec - 1e-6:
                    self.completed_pages.add(page.page_id)
        except ValueError:
            self.restore(snapshot)
            raise
        return {
            "recorded": list(recorded),
            "events": [
                self._event_view(self.events[key])
                for key in dict.fromkeys(key for values in recorded.values() for key in values)
            ],
            "progress": self.progress(),
        }

    def _update(self, args: UpdateInput) -> dict:
        snapshot = self.checkpoint()
        try:
            return self._save_event(args.event)
        except ValueError:
            self.restore(snapshot)
            raise

    def _save_event(self, submitted: EventInput) -> dict:
        old = self.events.get(submitted.id)
        if old and submitted.expected_version != old.version:
            raise ValueError(
                f"事件版本冲突；请读取最新记录后修改，当前 expected_version 应为 {old.version}。"
            )
        if old is None and submitted.expected_version is not None:
            raise ValueError("新事件不填写 expected_version；请检查 event.id 是否正确。")
        values = submitted.model_dump(exclude={"expected_version"})
        values["version"] = old.version if old else 1
        event = EventRecord.model_validate(values)
        for span in event.required_spans:
            observation = self.observations.get(span.evidence_id)
            if not observation or span.evidence_id not in self.delivered:
                raise ValueError(
                    "required_spans 的 evidence_id 必须引用已经观看的视频 observation_id。"
                )
            if (
                span.start_sec < observation.src_start_sec - 1e-6
                or span.end_sec > observation.src_end_sec + 1e-6
            ):
                raise ValueError(
                    f"required_spans 中 {span.role} 的 {span.start_sec:g}–{span.end_sec:g}s "
                    f"超出 {span.evidence_id} 的 {observation.src_start_sec:g}–"
                    f"{observation.src_end_sec:g}s；引用覆盖该区间的观察，或补看后修正时间。"
                )
        if event.status == "rejected" and event.rejection_category == "clip_infeasible":
            plan = self._plan(old) if old and old.status == "supported" else None
            has_irreparable_review = bool(
                plan
                and plan.review
                and any(issue.category != "mixed_focus" for issue in plan.review.issues)
            )
            if plan is None or not (plan.status == "infeasible_duration" or has_irreparable_review):
                raise ValueError(
                    "clip_infeasible 只用于已生成且存在时长不可行或阻断复核缺陷的片段；"
                    "多个独立看点应拆成事件，价值较弱或边界可精简时保留事件。"
                )
        children = [e for e in self.events.values() if e.merged_into == event.id]
        if children:
            if event.status != "supported":
                raise ValueError(
                    "该事件仍被其他 merged 事件引用，请先修订这些关联，再排除或合并当前事件。"
                )
            for source in children:
                for span in source.required_spans:
                    if span.role == "decisive" and not any(
                        s.role == "decisive"
                        and s.start_sec <= span.start_sec
                        and s.end_sec >= span.end_sec
                        for s in event.required_spans
                    ):
                        raise ValueError(
                            "合并后保留的事件必须包含来源事件的 decisive 核心证据区间。"
                        )
        if event.status == "merged":
            target = self.events.get(event.merged_into)
            if target is None or target.id == event.id or target.status in {"merged", "rejected"}:
                raise ValueError("merged_into 应指向另一个尚未排除或合并的事件。")
            # The surviving event must retain the decisive evidence of the merged record.
            spans = [*(old.required_spans if old else []), *event.required_spans]
            for span in spans:
                if span.role == "decisive" and not any(
                    s.role == "decisive"
                    and s.start_sec <= span.start_sec
                    and s.end_sec >= span.end_sec
                    for s in target.required_spans
                ):
                    raise ValueError(
                        "请先更新合并目标，保留来源事件的 decisive 核心证据，再提交合并。"
                    )
            event = EventRecord.model_validate(
                {
                    **event.model_dump(),
                    "required_spans": list({s.model_dump_json(): s for s in spans}.values()),
                }
            )
        if old and old.model_dump(exclude={"version"}) == event.model_dump(exclude={"version"}):
            return self._event_view(old)
        values = event.model_dump()
        values["version"] = old.version + 1 if old else 1
        event = EventRecord.model_validate(values)
        self.events[event.id] = event
        plan = self.plans.get(event.id)
        excluded_fields = {"version", "description", "reason"}
        if (
            plan
            and old
            and old.model_dump(exclude=excluded_fields) == event.model_dump(exclude=excluded_fields)
        ):
            self.plans[event.id] = plan.model_copy(update={"event_version": event.version})
        else:
            self.plans.pop(event.id, None)
        self.finished = False
        return self._event_view(event)

    def _event_view(self, event: EventRecord) -> dict:
        plan = self.plans.get(event.id)
        preview = None
        if event.status == "supported" and plan is None:
            draft = draft_plan(
                event,
                self.evidence.video_info.duration_sec,
                max_clip_sec=self.max_clip_sec,
                min_clip_sec=self.min_clip_sec,
            )
            preview = {
                "start_sec": draft.start_sec,
                "end_sec": draft.end_sec,
                "duration_sec": draft.end_sec - draft.start_sec,
                "status": draft.status,
                "issues": draft.issues,
            }
        if event.status == "pending":
            next_step = "补看视频或查找上下文，再用 update_event 更新事件判断。"
        elif event.status in {"rejected", "merged"}:
            next_step = "该事件已排除；发现新证据时可用 update_event 修订。"
        elif event.id not in self._chosen_ids():
            next_step = "候选保留在完整池中，先判断采用价值；选中后系统才制作并复核。"
        elif preview and preview["status"] == "infeasible_duration":
            next_step = "建议边界当前不可成片；根据 draft_preview.issues 修订边界或必要证据，系统随后重新制作并复核。"
        elif plan is None:
            next_step = "候选已初选，系统将制作片段并独立复核。"
        elif plan.status == "ready":
            next_step = "片段已复核并冻结，已进入当前完整候选池。"
        elif plan.issues or plan.status == "infeasible_duration":
            next_step = "根据 issues 补看或修订事件，系统随后重新制作并复核；确认无法成片时设为 rejected 并说明理由。"
        elif plan.review is None:
            next_step = "片段等待系统独立复核。"
        elif plan.review.issues:
            next_step = "结合原片判断 review.issues 是否影响核心看点，决定修订、确认采用或舍弃；首尾微调交给编辑。"
        else:
            next_step = "依据 review.visible_event 在完整候选池中取舍；需要核对时补看并修订事件。"
        return {
            "event": event.model_dump(mode="json"),
            "draft_preview": preview,
            "clip": (
                {
                    **plan.model_dump(mode="json", exclude={"media_path"}),
                    "media_ready": plan.media_path is not None,
                }
                if plan
                else None
            ),
            "next_step": next_step,
        }

    def _select(self, args: SelectInput) -> dict:
        if not self.selection_ready():
            raise ValueError("分析尚未完成；请继续观看和登记原片，并处理待查事件。")
        snapshot = self.checkpoint()
        try:
            plans = [self._plan(e) for e in self.events.values() if e.status == "supported"]
            self.plans.update({p.event_id: p for p in plans})
            selection = choose_clips(
                plans,
                list(self.events.values()),
                self.max_highlights,
                decisions=args.decisions,
                total_duration=self.total_duration,
                allow_overlap=self.allow_overlap,
            )
            reviewed = all(
                p.review is not None and not p.issues and p.media_path is not None
                for p in selection.selected
            )
            if reviewed:
                for plan in selection.selected:
                    if plan.status == "draft":
                        self.plans[plan.event_id] = confirm_clip(
                            plan, self.events[plan.event_id], plan.review
                        )
                selection = select_clips(
                    [
                        p
                        for p in self.plans.values()
                        if self.events[p.event_id].status == "supported"
                    ],
                    list(self.events.values()),
                    self.max_highlights,
                    decisions=args.decisions,
                    total_duration=self.total_duration,
                    allow_overlap=self.allow_overlap,
                )
        except ValueError:
            self.restore(snapshot)
            raise
        self.selection = selection
        self.finished = reviewed
        return {
            **selection.model_dump(mode="json"),
            "completion": "complete" if reviewed else "review_pending",
        }

    def pending_reviews(self) -> list[ClipPlan]:
        return [
            p
            for p in self.plans.values()
            if p.event_id in self._chosen_ids()
            and p.status == "draft"
            and p.review is None
            and p.media_path
        ]

    def record_review(self, event_id: str, review: ReviewResult) -> None:
        plan = self.plans[event_id]
        duration = plan.end_sec - plan.start_sec
        if any(issue.at_sec is not None and issue.at_sec > duration for issue in review.issues):
            raise ValueError("复核缺陷时间超出了当前片段。")
        # Independent viewing supplies the facts used by the main agent's selection.
        self.plans[event_id] = ClipPlan.model_validate(
            {**plan.model_dump(), "review": review.model_dump()}
        )

    def information_facts(self) -> set[str]:
        facts = {f"text:{key}" for key in self.read_memory.rows}
        facts.update(f"query:{key}" for key in self.read_memory.queries)
        for key in self.delivered:
            o = self.observations[key]
            facts.add(
                "video:" + identity([o.media_id, o.src_start_sec, o.src_end_sec, o.sampling_fps])
            )
        facts.update(f"recorded:{key}" for key in self.acknowledged)
        for event in self.events.values():
            facts.add(
                "event:" + identity(event.model_dump(exclude={"reason", "description", "version"}))
            )
        for plan in self.plans.values():
            facts.add(
                "clip:"
                + identity(
                    [
                        plan.event_id,
                        plan.start_sec,
                        plan.end_sec,
                        plan.status,
                        plan.review.model_dump(exclude={"visible_event"}) if plan.review else None,
                    ]
                )
            )
        facts.update(
            "proposal:" + identity([key, p["status"], p.get("event_id")])
            for key, p in self.proposals.items()
        )
        facts.update(f"page:{page.page_id}" for page in self.evidence.pages)
        if self.selection:
            facts.add(
                "selection:"
                + identity(
                    [[c.event_id, c.selected, c.duplicate_of] for c in self.selection.decisions]
                )
            )
        if self.finished:
            facts.add("finished")
        return facts

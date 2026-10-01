"""Validated video tools and durable working state for the ReAct loop."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .contracts import (
    TOOL_INPUTS,
    EventInput,
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
    complete_review,
    draft_plan,
    select_clips,
)
from .state import ReadMemory, identity

MAX_OBSERVATION_BYTES = 8_000_000  # Base64 payload budget per observation.


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
    ) -> None:
        self.evidence = evidence
        self.output_dir = output_dir
        self.max_highlights = max_highlights
        self.total_duration = total_duration
        self.allow_overlap = allow_overlap
        self.transcript = transcript
        self.local_proposals = local_proposals
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
        candidates = self._candidate_rows()
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
                "local_proposals": self.local_proposals is not None,
            },
            "transcript_complete": self._transcript_complete(),
            "local_proposals_loaded": self.local_proposals is None or self.proposals_loaded,
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
        return self.discovery_complete() and not self.unresolved_events()

    def discovery_complete(self) -> bool:
        return (
            len(self.completed_pages) == len(self.evidence.pages)
            and self._transcript_complete()
            and (self.local_proposals is None or self.proposals_loaded)
            and not (set(self.observations) - self.acknowledged.keys())
            and not any(event.status == "pending" for event in self.events.values())
            and not any(proposal["status"] == "pending" for proposal in self.proposals.values())
        )

    def _transcript_complete(self) -> bool:
        if self.transcript is None:
            return True
        full_query = identity({"query": "", "start_sec": 0.0, "end_sec": None})
        return bool(self.read_memory.queries.get(full_query, {}).get("complete"))

    def _next_action(self, ready_to_select: bool, unacknowledged: list[VideoObservation]) -> str:
        if unacknowledged:
            return "用 record_observations 一次登记本轮收到的全部视频观察。"
        if set(self.observations) - self.acknowledged.keys():
            return "等待尚未送达的观察视频，观看后立即登记。"
        if len(self.completed_pages) < len(self.evidence.pages):
            return "系统将继续送入下一批视频页面。"
        if not self._transcript_complete():
            return "用 search_transcript 的空 query 分页读完全部可用台词。"
        if self.local_proposals is not None and not self.proposals_loaded:
            return "用 propose_highlights action=list 载入本地模型的完整位置线索池。"
        if self.unresolved_events():
            return "处理 pending 事件、失败复核或未制作片段。"
        if any(proposal["status"] == "pending" for proposal in self.proposals.values()):
            return "处理尚未核验的本地位置线索。"
        if not ready_to_select:
            return "处理尚未通过成片复核的事件。"
        return "用 select_highlights 提交每个待选片段的保留或舍弃决定。"

    def unresolved_events(self) -> list[str]:
        pending = []
        for event in self.events.values():
            if event.status in {"rejected", "merged"}:
                continue
            plan = self.plans.get(event.id)
            if not self._selectable_plan(event, plan):
                pending.append(event.id)
        return pending

    @staticmethod
    def _selectable_plan(event: EventRecord, plan: ClipPlan | None) -> bool:
        return bool(
            event.status == "supported"
            and plan is not None
            and plan.event_version == event.version
            and plan.status in {"draft", "ready"}
            and plan.review is not None
            and not plan.review.blocking_issues
            and not plan.issues
            and plan.media_path is not None
        )

    def mark_delivered(self, observations: list[VideoObservation]) -> None:
        self.delivered.update(o.observation_id for o in observations)

    def scan_next_observations(
        self,
        *,
        max_items: int,
        max_encoded_bytes: int,
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
            size = self._encoded_size(observation)
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

    def prepare_reviews(self) -> list[ClipPlan]:
        """Render every resolved supported event once discovery has finished."""
        if not self.discovery_complete():
            return []
        for event in self.events.values():
            if event.status != "supported":
                continue
            existing = self.plans.get(event.id)
            if existing and existing.event_version == event.version:
                continue
            plan = draft_plan(
                event,
                self.evidence.video_info.duration_sec,
                max_clip_sec=self.max_clip_sec,
                min_clip_sec=self.min_clip_sec,
            )
            if plan.status == "draft":
                media = self.evidence.render_clip(plan.start_sec, plan.end_sec)
                plan = attach_media(
                    plan,
                    start_sec=media.src_start_sec,
                    end_sec=media.src_end_sec,
                    media_path=media.path,
                    video_duration=self.evidence.video_info.duration_sec,
                    max_clip_sec=self.max_clip_sec,
                )
            self.plans[event.id] = plan
        return self.pending_reviews()

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
            if self.transcript is None:
                raise ValueError("当前任务没有字幕检索工具，请直接观看视频。")
            return self.read_memory.record(
                args.model_dump(), self.transcript.search(**args.model_dump())
            ), []
        if isinstance(args, ProposalInput):
            return self._propose(args), []
        if isinstance(args, InspectInput):
            observation = self.evidence.inspect(args.start_sec, args.end_sec)
            if args.sampling_fps is not None:
                identity = hashlib.sha256(
                    f"{observation.observation_id}:{args.sampling_fps}".encode()
                ).hexdigest()[:24]
                observation = observation.model_copy(
                    update={"observation_id": f"obs_{identity}", "sampling_fps": args.sampling_fps}
                )
            if self._encoded_size(observation) > MAX_OBSERVATION_BYTES:
                raise ValueError("视频区间过大，请缩短 start_sec 到 end_sec 的范围后重试。")
            return self._observation(observation)
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
        if required_observation_ids:
            record = next(d for d in declarations if d["name"] == "record_observations")
            schema = record["parametersJsonSchema"]
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
            actions = {"select_highlights", "update_event", "inspect_interval", "read_state"}
            if self.transcript is not None:
                actions.add("search_transcript")
            return [d for d in declarations if d["name"] in actions]

        disabled = set()
        disabled.add("record_observations")
        if self.transcript is None:
            disabled.add("search_transcript")
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
            self.selection = None
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
            self.selection = None
        public = observation.model_dump(mode="json", exclude={"path"})
        public["status"] = "ok"
        public["time_mapping"] = "原片秒数 = src_start_sec + 本视频内秒数"
        return public, [observation]

    def _read(self, args: ReadInput) -> dict:
        if args.event_id is not None:
            if args.collection != "events" or args.event_id not in self.events:
                raise ValueError("event_id 不存在或 collection 不是 events；请分页读取事件索引。")
            return self._event_view(self.events[args.event_id])
        if args.collection == "events":
            rows = [self._event_view(event) for event in self.events.values()]
        elif args.collection == "candidates":
            rows = self._candidate_rows()
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

    def _candidate_rows(self) -> list[dict]:
        rows = []
        for event in self.events.values():
            plan = self.plans.get(event.id)
            if not self._selectable_plan(event, plan):
                continue
            rows.append(
                {
                    "event_id": event.id,
                    "event_version": event.version,
                    "start_sec": plan.start_sec,
                    "end_sec": plan.end_sec,
                    "visible_event": plan.review.visible_event,
                }
            )
        return rows

    def candidate_ledger(self) -> list[dict]:
        return self._candidate_rows()

    def _record(self, args: RecordInput) -> dict:
        expected_ids = {item.observation_id for item in self.unacknowledged_observations()}
        submitted_ids = {item.observation_id for item in args.observations}
        if submitted_ids != expected_ids:
            raise ValueError("observations 必须恰好登记本轮收到的全部 observation_id。")
        snapshot = self.checkpoint()
        try:
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
                observation = self.observations[item.observation_id]
                if observation.page_id:
                    self.completed_pages.add(observation.page_id)
                recorded[item.observation_id] = event_ids
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
        return self._save_event(args.event, existing_only=True)

    def _save_event(self, submitted: EventInput, *, existing_only: bool = False) -> dict:
        old = self.events.get(submitted.id) if submitted.id is not None else None
        if existing_only and old is None:
            raise ValueError("update_event 仅能修改已登记事件。")
        if old and submitted.expected_version != old.version:
            raise ValueError(
                f"事件版本冲突；请读取最新记录后修改，当前 expected_version 应为 {old.version}。"
            )
        if old is None and submitted.expected_version is not None:
            raise ValueError("新事件不填写 expected_version；请检查 event.id 是否正确。")
        values = submitted.model_dump(exclude={"expected_version"})
        if values["id"] is None:
            values.pop("id")
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
                raise ValueError("证据区间超出了引用的观察范围；请补看对应视频，或修正原片时间。")
        if event.status == "rejected" and event.rejection_category == "clip_infeasible":
            plan = self.plans.get(event.id)
            has_irreparable_review = bool(
                plan
                and plan.review
                and any(issue.category != "mixed_focus" for issue in plan.review.blocking_issues)
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
        self.plans.pop(event.id, None)
        self.finished = False
        self.selection = None
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
            next_step = "该事件已处理；发现新证据时可用 update_event 修订。"
        elif preview and preview["status"] == "infeasible_duration":
            next_step = "建议边界当前不可成片；根据 draft_preview.issues 修订边界或必要证据，系统随后重新制作并复核。"
        elif plan is None:
            next_step = "全片取证完成后，系统会制作片段并独立复核。"
        elif plan.status == "ready":
            next_step = "片段已复核并冻结，已进入当前完整候选池。"
        elif plan.issues or plan.status == "infeasible_duration":
            next_step = "根据 issues 补看或修订事件，系统随后重新制作并复核；确认无法成片时设为 rejected 并说明理由。"
        elif plan.review is None:
            next_step = "片段等待系统独立复核。"
        elif plan.review.blocking_issues:
            if any(issue.category == "mixed_focus" for issue in plan.review.blocking_issues):
                next_step = "复核发现多个独立看点：收窄当前事件，补看并分别登记其他看点；系统随后重新制作并复核。"
            else:
                next_step = "复核发现具体剪辑缺陷：根据 blocking_issues 修订事件和边界，系统随后重新制作并复核；确认无法修复后排除事件。"
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
            raise ValueError(
                "分析尚未完成；请查看 progress，继续观看未完成页面、登记观察结论，并处理待查事件和位置线索。"
            )
        snapshot = self.checkpoint()
        try:
            for event in self.events.values():
                plan = self.plans.get(event.id)
                if (
                    plan is not None
                    and plan.status == "draft"
                    and self._selectable_plan(event, plan)
                ):
                    self.plans[event.id] = complete_review(plan, event, plan.review)
            selection = select_clips(
                list(self.plans.values()),
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
        self.finished = True
        return selection.model_dump(mode="json")

    def pending_reviews(self) -> list[ClipPlan]:
        return [
            p
            for p in self.plans.values()
            if p.status == "draft" and p.review is None and p.media_path
        ]

    def record_review(self, event_id: str, review: ReviewResult) -> None:
        plan = self.plans[event_id]
        duration = plan.end_sec - plan.start_sec
        if any(
            issue.at_sec is not None and issue.at_sec > duration for issue in review.blocking_issues
        ):
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
                        plan.review.model_dump(exclude={"visible_event", "highlight_type"})
                        if plan.review
                        else None,
                    ]
                )
            )
        facts.update(
            "proposal:" + identity([key, p["status"], p.get("event_id")])
            for key, p in self.proposals.items()
        )
        facts.update(f"page:{page.page_id}" for page in self.evidence.pages)
        if self.finished:
            facts.add("finished")
        return facts

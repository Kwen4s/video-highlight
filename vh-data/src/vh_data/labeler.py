"""Independent video labeling, focused model review and resumable evidence records."""

import asyncio
import base64
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import httpx
from filelock import FileLock
from openai import APIStatusError, AsyncOpenAI, OpenAIError
from vh_agent.runtime.evidence import EvidenceStore, VideoPage

from .config import Settings
from .dialogue import DialogueLoader
from .models import Annotation, Label, PageLabels, Span, digest, overlap
from .quality import overlapping_highlights, reconcile, subtract
from .store import Store

PROMPTS = Path(__file__).with_name("prompts")
GUIDE = (PROMPTS / "highlight_types.md").read_text(encoding="utf-8")
PROMPT = (PROMPTS / "label.md").read_text(encoding="utf-8") + "\n" + GUIDE
REVIEW_PROMPT = (PROMPTS / "review.md").read_text(encoding="utf-8") + "\n" + GUIDE
MAX_REQUEST_BYTES = 20 * 1024 * 1024
MAX_OUTPUT_TOKENS = 32768


def write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class PageNeedsSplit(ValueError):
    """The actual multimodal request needs a smaller video page."""


class Labeler:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.dialogues = DialogueLoader(settings)
        if not settings.model or not settings.api_key.get_secret_value():
            raise ValueError("先在 vh-data/.env 配置已验证的视频模型和 API 密钥")

    def close(self):
        self.dialogues.transcriber = None

    def _prompt(self, review=False):
        template = REVIEW_PROMPT if review else PROMPT
        return template.replace(
            "{preferred_highlight_sec}", f"{self.settings.preferred_highlight_sec:g}"
        )

    async def _response(self, prompt, parts):
        async with asyncio.timeout(self.settings.request_timeout_sec):
            async with AsyncOpenAI(
                api_key=self.settings.api_key.get_secret_value(),
                base_url=self.settings.base_url,
                timeout=self.settings.request_timeout_sec,
                max_retries=0,
            ) as client:
                result = None
                async with await client.responses.create(
                    model=self.settings.model,
                    instructions=prompt,
                    input=[{"role": "user", "content": parts}],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "PageLabels",
                            "strict": True,
                            "schema": PageLabels.model_json_schema(),
                        }
                    },
                    reasoning={"effort": self.settings.reasoning_effort},
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    store=False,
                    stream=True,
                ) as stream:
                    async for event in stream:
                        if event.type in {
                            "response.completed",
                            "response.incomplete",
                            "response.failed",
                        }:
                            result = event.response
                if result is None:
                    raise ValueError("视频标注流缺少完成事件")
                return result

    async def label(self, video: dict) -> tuple[Annotation, dict]:
        source = Path(video["path"])
        if digest(source) != video["sha256"]:
            raise ValueError("原视频内容已改变，请重新导入")
        media = await asyncio.to_thread(
            EvidenceStore,
            source,
            self.settings.media_cache,
            page_sec=self.settings.page_sec,
            overlap_sec=5,
        )
        plan = self._plan_path(video["sha256"])
        if plan.exists():
            media.restore_pages(json.loads(plan.read_text())["pages"])
        self._compact_tail(media)
        dialogue, dialogue_ref = await asyncio.to_thread(self.dialogues.load, video, media)
        first, first_calls, stories = await self._read(media, video["sha256"], 1, dialogue)
        second, second_calls, _ = await self._read(media, video["sha256"], 2, dialogue)
        merged, issues = reconcile(
            first, second, media.video_info.duration_sec, self.settings.agreement_iou
        )
        predictions = video.get("predictions", [])
        focus = [s for s in merged.segments if s.kind != "negative"]
        if predictions:
            focus.extend(
                Span(start_sec=p["start_sec"], end_sec=p["end_sec"])
                for p in predictions[0]["segments"]
            )
        merged, review_calls = await self._review(
            media, video["sha256"], merged, focus, first, second, stories, dialogue, predictions
        )
        if digest(source) != video["sha256"]:
            raise ValueError("标注过程中原视频发生变化，本轮结果不能保存")
        provenance = {
            "source_sha256": video["sha256"],
            "dialogue": dialogue_ref,
            "model": self.settings.model,
            "base_url": self.settings.base_url,
            "reasoning_effort_requested": self.settings.reasoning_effort,
            "prompt_sha256": hashlib.sha256(self._prompt().encode()).hexdigest(),
            "review_prompt_sha256": hashlib.sha256(self._prompt(review=True).encode()).hexdigest(),
            "preferred_highlight_sec": self.settings.preferred_highlight_sec,
            "agreement_iou": self.settings.agreement_iou,
            "initial_issues": issues,
            "unresolved_segments": sum(s.kind == "uncertain" for s in merged.segments),
            "highlight_ratio": sum(
                s.end_sec - s.start_sec for s in merged.segments if s.kind == "positive"
            )
            / merged.duration_sec,
            "feedback_model": predictions[0]["model_id"] if predictions else None,
            "calls": [*first_calls, *second_calls, *review_calls],
            "review_calls": review_calls,
        }
        return merged, provenance

    @staticmethod
    def _compact_tail(media):
        # The previous view already contains this tail: retain its evidence and
        # enlarge ownership rather than sending the same frames a third time.
        pages = list(media.pages)
        if (
            len(pages) > 1
            and pages[-2].read_end_sec == media.video_info.duration_sec
            and pages[-1].core_end_sec - pages[-1].core_start_sec
            < pages[-2].core_end_sec - pages[-2].core_start_sec
        ):
            previous, tail = pages[-2:]
            pages[-2:] = [previous.model_copy(update={"core_end_sec": tail.core_end_sec})]
            media.restore_pages(pages)

    async def _review(
        self, media, source_hash, annotation, focus, first, second, stories, dialogue, predictions
    ):
        references = []
        index = 0
        review_all = not annotation.segments
        while index < len(media.pages):
            page = media.pages[index]
            core = Span(start_sec=page.core_start_sec, end_sec=page.core_end_sec)
            if not review_all and not any(overlap(core, span) for span in focus):
                index += 1
                continue
            notes = {
                "待选择的片段": [s.model_dump() for s in annotation.segments if overlap(core, s)],
                "第一次观看": [s.model_dump() for s in first if overlap(core, s)],
                "第二次观看": [s.model_dump() for s in second if overlap(core, s)],
                "小模型候选": [
                    p
                    for prediction in predictions[:1]
                    for p in prediction["segments"]
                    if p["start_sec"] < core.end_sec and p["end_sec"] > core.start_sec
                ],
            }
            try:
                result, reference = await self._call(
                    media,
                    page,
                    source_hash,
                    "review",
                    stories[max(start for start in stories if start <= page.core_start_sec)],
                    dialogue,
                    notes=notes,
                )
            except PageNeedsSplit:
                self._split_page(media, page, source_hash)
                continue
            annotation = self._replace_review(annotation, core, result.segments)
            references.append(reference)
            index += 1

        # A neighboring review can discover a different boundary. Resolve only
        # actual remaining conflicts using the same selection step and evidence.
        for index in range(len(annotation.segments)):
            conflicts = overlapping_highlights(annotation.segments)
            if not conflicts:
                return annotation, references
            span = min(conflicts, key=lambda s: s.end_sec - s.start_sec)
            page = VideoPage(
                page_id=f"review-conflict-{index}",
                core_start_sec=span.start_sec,
                core_end_sec=span.end_sec,
                read_start_sec=max(0, span.start_sec - media.overlap_sec),
                read_end_sec=min(annotation.duration_sec, span.end_sec + media.overlap_sec),
            )
            result, reference = await self._call(
                media,
                page,
                source_hash,
                "review",
                stories[max(start for start in stories if start <= span.start_sec)],
                dialogue,
                notes={
                    "待选择的片段": [
                        s.model_dump() for s in annotation.segments if overlap(span, s)
                    ]
                },
            )
            annotation = self._replace_review(annotation, span, result.segments)
            references.append(reference)
        if overlapping_highlights(annotation.segments):
            raise ValueError("最终高光仍有重叠，需要重新复核")
        return annotation, references

    @classmethod
    def _replace_review(cls, annotation, core, reviewed):
        retained = [
            s
            for s in annotation.segments
            if s.kind == "positive"
            and not core.start_sec <= (s.start_sec + s.end_sec) / 2 < core.end_sec
        ]
        retained.extend(
            Label(**piece.model_dump(), kind=s.kind, reason=s.reason)
            for s in annotation.segments
            if s.kind != "positive"
            for piece in subtract(s, [core])
        )
        owned = []
        for segment in reviewed:
            if segment.kind == "positive":
                if core.start_sec <= (segment.start_sec + segment.end_sec) / 2 < core.end_sec:
                    owned.append(segment)
            elif overlap(core, segment):
                owned.append(
                    segment.model_copy(
                        update={
                            "start_sec": max(core.start_sec, segment.start_sec),
                            "end_sec": min(core.end_sec, segment.end_sec),
                        }
                    )
                )
        return cls._annotation(annotation.duration_sec, [*retained, *owned])

    async def _read(
        self, media: EvidenceStore, source_hash: str, pass_id: int, dialogue: list[dict]
    ):
        story, segments, references, stories = "", [], [], {}
        index = 0
        while index < len(media.pages):
            page = media.pages[index]
            try:
                result, reference = await self._call(
                    media, page, source_hash, pass_id, story, dialogue
                )
            except PageNeedsSplit:
                self._split_page(media, page, source_hash)
                continue
            stories[page.core_start_sec] = story
            # Keep complete candidates from the overlap too. Ownership is applied
            # to final review results, never by rejecting an entire response.
            segments.extend(s for s in result.segments if s not in segments)
            references.append(reference)
            story = result.story_so_far
            index += 1
        return segments, references, stories

    def _plan_path(self, source_hash):
        return self.settings.root / "pages" / f"{source_hash}-{self.settings.page_sec:g}.json"

    def _split_page(self, media, page, source_hash):
        children = media.split_page(page.page_id)
        write_json(self._plan_path(source_hash), {"pages": [p.model_dump() for p in media.pages]})
        print(
            f"  缩小页面：{page.core_start_sec:.1f}–{page.core_end_sec:.1f} 秒，"
            f"每页 {children[0].core_end_sec - children[0].core_start_sec:.1f} 秒",
            flush=True,
        )

    async def _call(self, media, page, source_hash, pass_id, story, dialogue, notes=None):
        for attempt in range(1, self.settings.request_attempts + 1):
            try:
                return await self._call_once(
                    media, page, source_hash, pass_id, story, dialogue, notes, attempt
                )
            except PageNeedsSplit:
                raise
            except APIStatusError as error:
                if error.status_code == 413:
                    raise PageNeedsSplit("网关拒绝过大的图片请求") from error
                if error.status_code not in {408, 409, 429} and error.status_code < 500:
                    raise
                if attempt == self.settings.request_attempts:
                    raise
                reason = f"HTTP {error.status_code}"
            except (OpenAIError, httpx.TransportError, TimeoutError, ValueError) as error:
                reason = type(error).__name__
                if attempt == self.settings.request_attempts:
                    raise
            delay = min(2**attempt, 30)
            print(
                f"  {reason}，{delay} 秒后重试 {attempt + 1}/{self.settings.request_attempts}",
                flush=True,
            )
            await asyncio.sleep(delay)

    async def _call_once(self, media, page, source_hash, pass_id, story, dialogue, notes, attempt):
        margin = (
            max(
                page.core_start_sec - page.read_start_sec,
                page.read_end_sec - page.core_end_sec,
            )
            if notes is not None
            else 0
        )
        read_start = max(0, page.read_start_sec - margin)
        read_end = min(media.video_info.duration_sec, page.read_end_sec + margin)
        times = [
            read_start + i / self.settings.frame_fps
            for i in range(math.ceil((read_end - read_start) * self.settings.frame_fps))
            if read_start + i / self.settings.frame_fps < read_end
        ]
        frames = await asyncio.to_thread(media.frames, times, width=768)
        read_start = min(read_start, frames[0]["timestamp_sec"])
        context = {
            "负责范围": [page.core_start_sec, page.core_end_sec],
            "前后文范围": [read_start, read_end],
            "原片时长": media.video_info.duration_sec,
            "原片有音轨": media.video_info.has_audio,
            "之前的剧情": story,
            "对白与字幕": [
                t for t in dialogue if read_start <= t["start_sec"] < t["end_sec"] <= read_end
            ],
        }
        if notes is not None:
            context["待核对意见"] = notes
        prompt = self._prompt(review=notes is not None)
        sizes = [Path(f["path"]).stat().st_size for f in frames]
        if sum(4 * ((size + 2) // 3) for size in sizes) + 100_000 >= MAX_REQUEST_BYTES:
            raise PageNeedsSplit("图片请求过大")
        signature = {
            "source": source_hash,
            "pass": pass_id,
            "context": context,
            "page": page.model_dump(),
            "model": self.settings.model,
            "base_url": self.settings.base_url,
            "prompt": prompt,
            "reasoning": self.settings.reasoning_effort,
            "frame_fps": self.settings.frame_fps,
            "input": "timestamped_frames_and_dialogue",
            "api": "responses_stream",
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "schema": PageLabels.model_json_schema(),
        }
        request_id = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        path = self.settings.root / "readings" / f"{request_id}.json"
        cached = path.exists()
        if cached:
            record = json.loads(path.read_text())
            response = PageLabels.model_validate(record["result"])
        else:
            parts = [{"type": "input_text", "text": json.dumps(context, ensure_ascii=False)}]
            for frame in frames:
                encoded = base64.b64encode(Path(frame["path"]).read_bytes()).decode("ascii")
                parts.extend(
                    [
                        {"type": "input_text", "text": f"原片 {frame['timestamp_sec']:.3f} 秒"},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{encoded}",
                            "detail": "auto",
                        },
                    ]
                )
            if (
                len(json.dumps(parts, ensure_ascii=False).encode()) + len(prompt.encode()) + 10_000
                >= MAX_REQUEST_BYTES
            ):
                raise PageNeedsSplit("图片请求过大")
            started = time.monotonic()
            stage = "重点复核" if notes is not None else f"独立观看 {pass_id}/2"
            print(
                f"  [{source_hash[:8]}] {stage}：{read_start:.1f}–{read_end:.1f} 秒，"
                f"{len(frames)} 帧",
                flush=True,
            )
            try:
                result = await self._response(prompt, parts)
            except (OpenAIError, httpx.TransportError, ValueError, TimeoutError) as error:
                failure = self.settings.root / "failures" / f"{request_id}-{time.time_ns()}.json"
                write_json(
                    failure,
                    {
                        "request_id": request_id,
                        "attempt": attempt,
                        "input": signature,
                        "status": "request_failed",
                        "error_class": type(error).__name__,
                        "error_code": getattr(error, "code", None),
                        "error_type": getattr(error, "type", None),
                        "http_status": getattr(error, "status_code", None),
                        "elapsed_sec": round(time.monotonic() - started, 3),
                        "video": str(media.video_info.path),
                        "frames": frames,
                    },
                )
                raise
            record = {
                "request_id": request_id,
                "attempt": attempt,
                "input": signature,
                "status": result.status,
                "error_code": result.error.code if result.error else None,
                "incomplete_details": result.incomplete_details.model_dump()
                if result.incomplete_details
                else None,
                "usage": result.usage.model_dump(
                    include={
                        "input_tokens",
                        "output_tokens",
                        "total_tokens",
                        "input_tokens_details",
                        "output_tokens_details",
                    }
                )
                if result.usage
                else None,
                "elapsed_sec": round(time.monotonic() - started, 3),
                "returned_model": result.model,
                "reasoning": result.reasoning.model_dump() if result.reasoning else None,
                "video": str(media.video_info.path),
                "frames": frames,
            }
            try:
                if result.status != "completed":
                    raise ValueError(
                        f"视频标注未完成：{result.status} {record['incomplete_details']}"
                    )
                response = PageLabels.model_validate_json(result.output_text)
                self._validate_page(response, read_start, read_end, media.video_info.duration_sec)
                if notes is not None:
                    self._validate_review(response, media.video_info.duration_sec)
            except ValueError:
                record["output_text"] = result.output_text
                failure = self.settings.root / "failures" / f"{request_id}-{time.time_ns()}.json"
                write_json(failure, record)
                raise
            record["result"] = response.model_dump()
        if cached:
            self._validate_page(response, read_start, read_end, media.video_info.duration_sec)
            if notes is not None:
                self._validate_review(response, media.video_info.duration_sec)
        if not cached:
            write_json(path, record)
        return response, str(path.relative_to(self.settings.root))

    @staticmethod
    def _validate_review(response, duration):
        Annotation(duration_sec=duration, segments=response.segments)
        if overlapping_highlights(response.segments):
            raise ValueError("同次复核返回了重叠高光，需要统一看点与边界")

    @staticmethod
    def _annotation(duration, segments):
        # A verified highlight includes its necessary context; background labels
        # from neighboring pages must not turn that context into negative targets.
        positive = [s for s in segments if s.kind == "positive"]
        clean = list(positive)
        for kind in ("uncertain", "negative"):
            excluded = list(clean)
            for segment in (s for s in segments if s.kind == kind):
                clean.extend(
                    Label(**piece.model_dump(), kind=kind, reason=segment.reason)
                    for piece in subtract(segment, excluded)
                )
        return Annotation(duration_sec=duration, segments=clean)

    @staticmethod
    def _validate_page(response, read_start, read_end, duration):
        for segment in response.segments:
            if not read_start <= segment.start_sec < segment.end_sec <= read_end:
                raise ValueError("标注超出实际观看的视频范围")
            touches_edge = (read_start > 0 and segment.start_sec <= read_start) or (
                read_end < duration and segment.end_sec >= read_end
            )
            if segment.kind == "positive" and touches_edge:
                segment.kind = "uncertain"
                segment.reason = "观看边缘的候选需要补充前后文：" + segment.reason


def run(
    store: Store,
    settings: Settings,
    limit: int,
    split: str | None,
    retry_failed: bool,
    video_ids: list[str] | None = None,
):
    return asyncio.run(_run(store, settings, limit, split, retry_failed, video_ids))


async def _run(store, settings, limit, split, retry_failed, video_ids):
    for video_id in video_ids or []:
        video = store.get(video_id)
        if split is not None and video["split"] != split:
            raise ValueError(f"{video_id} 不属于 {split} 组")
    completed = failed = gateway_failures = submitted = 0

    async def produce(video):
        labeler = Labeler(settings)
        try:
            return await labeler.label(video)
        finally:
            labeler.close()

    with FileLock(str(store.root / "producer.lock"), timeout=0):
        Labeler(settings).close()
        store.recover(retry_failed, split, video_ids)
        pending = {}
        stopped = False
        try:
            while pending or (submitted < limit and not stopped):
                while len(pending) < settings.workers and submitted < limit and not stopped:
                    video = store.claim(split, video_ids)
                    if video is None:
                        stopped = True
                        break
                    submitted += 1
                    print(
                        f"正在标注 {submitted}/{limit}：{video['video_id']} "
                        f"({video['split']}，{video['duration_sec']:.1f} 秒，"
                        f"证据 {video['sha256'][:8]})",
                        flush=True,
                    )
                    pending[asyncio.create_task(produce(video))] = video
                if not pending:
                    break
                done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for future in done:
                    video = pending.pop(future)
                    try:
                        annotation, provenance = future.result()
                        store.save(
                            video["video_id"],
                            annotation,
                            "model_reviewed",
                            provenance,
                            expected_label_id=video["label_id"],
                        )
                        completed += 1
                        gateway_failures = 0
                        positive_sec = sum(
                            s.end_sec - s.start_sec
                            for s in annotation.segments
                            if s.kind == "positive"
                        )
                        print(
                            f"已保存 {video['video_id']}：{len(annotation.segments)} 段，"
                            f"高光覆盖 {positive_sec / annotation.duration_sec:.1%}，"
                            f"未确认 {provenance['unresolved_segments']} 段",
                            flush=True,
                        )
                    except (
                        OpenAIError,
                        httpx.TransportError,
                        ValueError,
                        OSError,
                        RuntimeError,
                        subprocess.SubprocessError,
                    ) as error:
                        message = type(error).__name__
                        if isinstance(error, APIStatusError):
                            message += f" HTTP {error.status_code}"
                        elif isinstance(error, ValueError):
                            message += f": {error}"
                        store.fail(video["video_id"], message)
                        failed += 1
                        print(f"失败 {video['video_id']}: {message}", flush=True)
                        if isinstance(error, APIStatusError) and error.status_code in {
                            401,
                            403,
                            404,
                        }:
                            print("网关鉴权或模型配置有误，停止领取新视频。", flush=True)
                            stopped = True
                        gateway_failures = (
                            gateway_failures + 1
                            if isinstance(error, (OpenAIError, httpx.TransportError, TimeoutError))
                            else 0
                        )
                        if gateway_failures >= 3:
                            print("连续 3 条视频网关失败，停止领取新视频。", flush=True)
                            stopped = True
        finally:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
    return {"completed": completed, "failed": failed}

"""The sole detection entry point: native video agent and frozen clip artifacts."""

from collections.abc import Callable
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path

from .config import Settings
from .models import AnalysisSummary, DetectionResult, DetectionTask, Highlight, VideoSummary
from .providers.frame_text import FrameText
from .providers.gemini_client import GeminiClient
from .providers.local_proposals import LocalProposals
from .providers.openai_client import OpenAIClient
from .providers.retrieval import TranscriptSearch
from .providers.video_perception import VideoPerception
from .providers.video_search import QwenSearch
from .runtime.agent import VideoAgent
from .runtime.evidence import EvidenceStore
from .storage import write_json


class HighlightDetectionService:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        on_progress: Callable[[str], None] | None = None,
    ):
        self.settings = settings or Settings()
        self.on_progress = on_progress

    def detect(self, task: DetectionTask) -> DetectionResult:
        settings = self.settings
        output = (settings.job_output_dir / task.job_id).resolve()
        output.mkdir(parents=True, exist_ok=True)
        evidence = EvidenceStore(
            task.video_path, settings.media_cache_dir, page_sec=settings.page_sec
        )
        transcript = TranscriptSearch(evidence, settings, task.subtitle_path, task.language)
        local_proposals = (
            LocalProposals(evidence, settings, task.language, task.video_id)
            if settings.local_checkpoint
            else None
        )
        with ExitStack() as stack:
            video_client = stack.enter_context(
                GeminiClient(
                    settings.gemini_api_key,
                    settings.gemini_base_url,
                    settings.gemini_video_model,
                    timeout=settings.request_timeout_sec,
                    seed=settings.generation_seed,
                    thinking_level=settings.gemini_thinking_level,
                )
            )
            client = stack.enter_context(
                OpenAIClient(
                    settings.openai_api_key,
                    settings.openai_base_url,
                    settings.openai_agent_model,
                    effort=settings.openai_reasoning_effort,
                    timeout=settings.request_timeout_sec,
                )
            )
            search = (
                QwenSearch(
                    evidence,
                    settings.qwen_embedding_url,
                    settings.qwen_reranker_url,
                    transcript if transcript.available else None,
                    timeout=settings.request_timeout_sec,
                )
                if settings.qwen_embedding_url
                else None
            )
            if search:
                stack.callback(search.close)
            frame_text = FrameText(settings, task.language)
            agent = VideoAgent(
                client,
                evidence,
                output,
                perception=VideoPerception(video_client, settings.video_fps, evidence),
                frame_fps=settings.agent_frame_fps,
                search=search,
                frame_text=frame_text if frame_text.available else None,
                task=task.instruction,
                max_highlights=task.max_highlights,
                min_clip_sec=task.min_clip_sec,
                max_clip_sec=task.max_clip_sec,
                total_duration=task.total_duration_sec,
                allow_overlap=task.allow_overlap,
                video_fps=settings.video_fps,
                context_token_budget=settings.context_token_budget,
                context_byte_budget=settings.context_byte_budget,
                max_stagnant_steps=settings.max_stagnant_steps,
                max_request_attempts=settings.max_request_attempts,
                on_progress=self.on_progress,
                transcript=transcript if transcript.available else None,
                local_proposals=local_proposals,
            )
            raw = agent.run(resume=task.resume)
        clips = {}
        highlights = []
        for item in raw["highlights"]:
            identity = "hl_" + sha256(item["id"].encode()).hexdigest()[:16]
            clips[identity] = {
                "path": item["clip_path"],
                "event_id": item["id"].removeprefix("clip_"),
                "start_sec": item["start_sec"],
                "end_sec": item["end_sec"],
            }
            highlights.append(
                Highlight(
                    highlight_id=identity,
                    **{k: v for k, v in item.items() if k not in {"id", "clip_path"}},
                    clip_url=f"/api/jobs/{task.job_id}/clips/{identity}",
                )
            )
        result = DetectionResult(
            job_id=task.job_id,
            video=VideoSummary(
                video_id=task.video_id or evidence.media_id,
                title=Path(task.video_path).name,
                duration_sec=evidence.video_info.duration_sec,
            ),
            completion=raw["completion"],
            message=raw["message"],
            analysis=AnalysisSummary.model_validate(raw["analysis"]),
            highlights=highlights,
        )
        write_json(output / "clips.json", clips)
        write_json(output / "result.json", result.model_dump(mode="json"))
        return result

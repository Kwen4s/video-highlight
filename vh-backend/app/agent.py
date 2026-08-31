import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Thread

from .config import Settings
from .models import AgentDetectionResult
from .repository import JobRepository

AGENT_STAGE_ORDER = {
    "preprocessing": 0,
    "perception": 1,
    "fusion": 2,
    "reasoning": 3,
}


def detect_agent_stage(job_dir: Path) -> str:
    """Infer a public stage from durable artifacts without exposing Agent internals."""
    cache_root = job_dir / "cache"
    if not cache_root.is_dir():
        return "preprocessing"
    cache_dirs = [path for path in cache_root.iterdir() if path.is_dir()]
    if any((path / "scene_map").is_dir() for path in cache_dirs):
        return "reasoning"
    if any((path / "preprocess.json").is_file() for path in cache_dirs):
        return "fusion"
    if any(
        (path / "frames" / ".complete").is_file() and (path / "audio.wav").is_file()
        for path in cache_dirs
    ):
        return "perception"
    return "preprocessing"


class AgentInvoker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self, *, job_id: str, video_path: Path, language: str) -> AgentDetectionResult:
        if not self.settings.agent_root.is_dir():
            raise RuntimeError("vh-agent directory is not available")

        job_dir = self.settings.jobs_dir / job_id
        log_path = job_dir / "agent.log"
        command = [
            self.settings.agent_uv_executable,
            "run",
            "--directory",
            str(self.settings.agent_root),
            "--no-sync",
            "vh",
            "run",
            str(video_path),
            "--job-id",
            job_id,
            "--video-id",
            job_id,
            "--language",
            language,
        ]
        environment = os.environ.copy()
        # The backend and vh-agent have separate uv environments. Inheriting the
        # backend's VIRTUAL_ENV makes uv warn that it does not match the agent
        # project and can select the wrong environment on Windows and Linux.
        environment.pop("VIRTUAL_ENV", None)
        environment.update(
            {
                "VH_JOB_OUTPUT_DIR": str(self.settings.jobs_dir),
                "VH_MEDIA_CACHE_DIR": str(job_dir / "cache"),
                "VH_EXPORT_CLIPS": "false",
                "PYTHONUTF8": "1",
            }
        )
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=self.settings.agent_root,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=self.settings.agent_timeout_sec,
                    check=False,
                    creationflags=creation_flags,
                )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("vh-agent execution timed out") from error
        if completed.returncode != 0:
            raise RuntimeError(f"vh-agent exited with code {completed.returncode}")

        result_path = job_dir / "result.json"
        if not result_path.is_file():
            raise RuntimeError("vh-agent did not produce result.json")
        result = AgentDetectionResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        if result.job_id != job_id:
            raise RuntimeError("vh-agent returned a mismatched job id")
        return result


class AgentJobRunner:
    def __init__(
        self,
        settings: Settings,
        repository: JobRepository,
        invoker: AgentInvoker | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.invoker = invoker or AgentInvoker(settings)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vh-agent")

    def enqueue(self, job_id: str) -> None:
        self.executor.submit(self._execute, job_id)

    def _observe_progress(self, job_id: str, stop: Event) -> None:
        job_dir = self.settings.jobs_dir / job_id
        current_stage = "preprocessing"
        while not stop.wait(0.5):
            observed_stage = detect_agent_stage(job_dir)
            if AGENT_STAGE_ORDER[observed_stage] <= AGENT_STAGE_ORDER[current_stage]:
                continue
            current_stage = observed_stage
            self.repository.set_current_stage(job_id, current_stage)

    def _execute(self, job_id: str) -> None:
        job = self.repository.get(job_id)
        if job is None:
            return
        video_path = self.settings.jobs_dir / job_id / "source" / job["stored_name"]
        max_attempts = min(
            int(job.get("max_attempts") or self.settings.agent_max_attempts),
            self.settings.agent_max_attempts,
        )
        for attempt in range(1, max_attempts + 1):
            self.repository.set_attempt(job_id, attempt)
            progress_stop = Event()
            progress_thread = Thread(
                target=self._observe_progress,
                args=(job_id, progress_stop),
                name=f"vh-progress-{job_id}",
                daemon=True,
            )
            progress_thread.start()
            try:
                result = self.invoker.run(
                    job_id=job_id,
                    video_path=video_path,
                    language=job["language"],
                )
                self.repository.set_current_stage(
                    job_id,
                    detect_agent_stage(self.settings.jobs_dir / job_id),
                )
                self.repository.save_result(
                    job_id,
                    result.to_public_result().model_dump(mode="json"),
                )
                return
            except Exception as error:
                error_type = type(error).__name__
                if attempt >= max_attempts:
                    self.repository.set_status(
                        job_id,
                        "failed",
                        error_message=(
                            f"高光提取连续失败 {attempt}/{max_attempts} 次"
                            f"（{error_type}），请查看任务日志"
                        ),
                    )
                    return
                self.repository.set_attempt(
                    job_id,
                    attempt,
                    status="queued",
                    error_message=(
                        f"第 {attempt}/{max_attempts} 次执行失败（{error_type}），正在自动重试"
                    ),
                )
                if self.settings.agent_retry_delay_sec:
                    time.sleep(self.settings.agent_retry_delay_sec)
            finally:
                progress_stop.set()
                progress_thread.join(timeout=1)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)

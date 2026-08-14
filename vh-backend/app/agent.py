import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import Settings
from .models import DetectionResult
from .repository import JobRepository


class AgentInvoker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self, *, job_id: str, video_path: Path, language: str) -> DetectionResult:
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
        environment.update(
            {
                "VH_JOB_OUTPUT_DIR": str(self.settings.jobs_dir),
                "VH_MEDIA_CACHE_DIR": str(self.settings.storage_dir / "cache"),
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
        result = DetectionResult.model_validate_json(result_path.read_text(encoding="utf-8"))
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

    def _execute(self, job_id: str) -> None:
        job = self.repository.get(job_id)
        if job is None:
            return
        self.repository.set_status(job_id, "processing")
        video_path = self.settings.jobs_dir / job_id / "source" / job["stored_name"]
        try:
            result = self.invoker.run(
                job_id=job_id,
                video_path=video_path,
                language=job["language"],
            )
            self.repository.save_result(job_id, result.model_dump(mode="json"))
        except Exception as error:
            self.repository.set_status(
                job_id,
                "failed",
                error_message=f"高光提取失败（{type(error).__name__}），请查看任务日志",
            )

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)

"""Frozen multimodal moment feature extraction backed by media caches."""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ..preprocessing.media import video_fingerprint
from ..preprocessing.sensevoice import EMOTIONS, EVENTS
from .config import HighlightModelConfig
from .dataset import SilverVideo, feature_path

FEATURE_SCHEMA = 1
VISION_INSTRUCTION = (
    "Represent this short-drama moment for locating decisive narrative state changes."
)
EVENT_NAMES = tuple(value.lower() for value in EVENTS)
EMOTION_NAMES = tuple(value.lower() for value in EMOTIONS)


class FrozenMomentFeatureExtractor:
    def __init__(self, config: HighlightModelConfig) -> None:
        self.config = config
        self.device = torch.device(config.feature_device)
        self._vision_embedder: Any | None = None
        self._audio_bundle: Any | None = None

    def prepare(self, videos: list[SilverVideo]) -> dict[str, int]:
        reused = created = 0
        for index, video in enumerate(videos, start=1):
            target = feature_path(self.config.feature_cache_dir, video.video_id)
            expected = self._signature(video)
            if target.is_file():
                cached = torch.load(target, map_location="cpu", weights_only=False)
                if cached.get("signature") == expected:
                    reused += 1
                    continue
            self._extract(video, target, expected)
            created += 1
            if index % 10 == 0 or index == len(videos):
                print(f"features {index}/{len(videos)} created={created} reused={reused}")
        return {"created": created, "reused": reused}

    def close(self) -> None:
        self._vision_embedder = None
        self._audio_bundle = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _extract(self, video: SilverVideo, target: Path, signature: str) -> None:
        cache_dir = self.config.media_cache_dir / video_fingerprint(video.path)
        preprocess_path = cache_dir / "preprocess.json"
        audio_path = cache_dir / "audio.wav"
        if not preprocess_path.is_file() or not audio_path.is_file():
            raise FileNotFoundError(f"incomplete media cache for {video.video_id}: {cache_dir}")
        artifacts = json.loads(preprocess_path.read_text(encoding="utf-8"))
        length = max(1, math.ceil(video.duration_sec))
        frame_paths = _moment_frame_paths(artifacts.get("frame_samples", []), length)
        vision = self._vision_features(
            frame_paths,
            artifacts.get("transcript", []),
            video.language,
        )
        audio = self._audio_features(audio_path, length, video.language)
        audio_prior = _audio_priors(audio_path, artifacts.get("audio_events", []), length)
        availability = torch.ones((length, 2), dtype=torch.float32)
        scene_bounds = _merge_scene_bounds(artifacts.get("scenes", []), length)

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        torch.save(
            {
                "schema": FEATURE_SCHEMA,
                "signature": signature,
                "video_id": video.video_id,
                "vision": vision.half(),
                "audio": audio.half(),
                "audio_prior": audio_prior.half(),
                "availability": availability,
                "scene_bounds": scene_bounds,
            },
            temporary,
        )
        temporary.replace(target)

    def _vision_features(
        self,
        frame_paths: list[tuple[str, str, str]],
        transcript: list[dict[str, Any]],
        language: str,
    ) -> Tensor:
        embedder = self._get_vision_embedder()
        batches: list[Tensor] = []
        for start in range(0, len(frame_paths), self.config.vision_batch_size):
            clips = frame_paths[start : start + self.config.vision_batch_size]
            inputs = []
            for local_index, clip in enumerate(clips):
                timestamp = start + local_index
                speech = _nearby_text(transcript, timestamp, "asr")
                ocr = _nearby_text(transcript, timestamp, "ocr")
                text = (
                    f"Language: {language}\n"
                    f"Speech: {speech or '[no speech]'}\n"
                    f"On-screen text: {ocr or '[no caption]'}"
                )
                inputs.append(
                    {
                        "video": list(clip),
                        "text": text,
                        "instruction": VISION_INSTRUCTION,
                    }
                )
            batches.append(embedder.process(inputs, normalize=True).float().cpu())
        return torch.cat(batches)

    def _audio_features(self, audio_path: Path, length: int, language: str) -> Tensor:
        bundle = self._get_audio_bundle()
        from funasr.utils.load_utils import extract_fbank

        waveform = _read_waveform(audio_path)
        sample_rate = 16000
        window = self.config.audio_window_sec * sample_rate
        overlap = self.config.audio_overlap_sec * sample_rate
        step = window - overlap
        if step <= 0:
            raise ValueError("audio overlap must be smaller than the audio window")
        result = torch.zeros((length, 512), dtype=torch.float32)
        counts = torch.zeros(length, dtype=torch.float32)
        starts = list(range(0, max(1, len(waveform) - overlap), step))
        language_token = 24885 if language == "en" else 24884
        prompt = torch.tensor([[language_token, 0, 0, 25016]], device=self.device)
        model = bundle.model
        frontend = bundle.kwargs["frontend"]
        model.eval()
        for start in starts:
            chunk = waveform[start : min(len(waveform), start + window)]
            if len(chunk) < sample_rate // 2:
                continue
            speech, speech_lengths = extract_fbank(chunk, frontend=frontend)
            speech = speech.to(self.device)
            speech_lengths = speech_lengths.to(self.device)
            with torch.inference_mode():
                encoded, encoded_lengths = model.encode(speech, speech_lengths, prompt)
            valid_length = max(0, int(encoded_lengths[0]) - 4)
            encoded = encoded[0, 4 : 4 + valid_length].float().cpu()
            if not len(encoded):
                continue
            chunk_start_sec = start / sample_rate
            chunk_duration_sec = len(chunk) / sample_rate
            times = chunk_start_sec + (
                torch.arange(len(encoded), dtype=torch.float32) + 0.5
            ) * (chunk_duration_sec / len(encoded))
            seconds = times.floor().long().clamp(0, length - 1)
            result.index_add_(0, seconds, encoded)
            counts.index_add_(0, seconds, torch.ones_like(times))
        missing = counts == 0
        result = result / counts.clamp_min(1).unsqueeze(1)
        if missing.any():
            available = (~missing).nonzero(as_tuple=False).flatten()
            if len(available):
                nearest = torch.argmin(
                    torch.abs(
                        torch.arange(length)[:, None] - available[None, :]
                    ),
                    dim=1,
                )
                result[missing] = result[available[nearest[missing]]]
        return F.normalize(result, dim=1)

    def _get_vision_embedder(self) -> Any:
        if self._vision_embedder is not None:
            return self._vision_embedder
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        script = self.config.vision_model_path / "scripts" / "qwen3_vl_embedding.py"
        spec = importlib.util.spec_from_file_location("vh_highlight_qwen3_vl", script)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load Qwen embedding implementation: {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self._vision_embedder = module.Qwen3VLEmbedder(
            str(self.config.vision_model_path),
            torch_dtype=dtype,
            max_length=2048,
            max_pixels=256 * 32 * 32,
            num_frames=3,
            max_frames=3,
        )
        self._vision_embedder.model.to(self.device).eval()
        return self._vision_embedder

    def _get_audio_bundle(self) -> Any:
        if self._audio_bundle is not None:
            return self._audio_bundle
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise RuntimeError("FunASR is required for frozen SenseVoice features") from exc
        self._audio_bundle = AutoModel(
            model=str(self.config.audio_model_path),
            device=str(self.device),
            disable_update=True,
            trust_remote_code=True,
        )
        self._audio_bundle.model.eval()
        for parameter in self._audio_bundle.model.parameters():
            parameter.requires_grad_(False)
        return self._audio_bundle

    def _signature(self, video: SilverVideo) -> str:
        cache_dir = self.config.media_cache_dir / video_fingerprint(video.path)
        preprocess = json.loads((cache_dir / "preprocess.json").read_text(encoding="utf-8"))
        payload = {
            "schema": FEATURE_SCHEMA,
            "video": video_fingerprint(video.path),
            "preprocess": preprocess.get("signature"),
            "vision_model": str(self.config.vision_model_path.resolve()),
            "audio_model": str(self.config.audio_model_path.resolve()),
            "audio_window_sec": self.config.audio_window_sec,
            "audio_overlap_sec": self.config.audio_overlap_sec,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def _moment_frame_paths(
    samples: list[dict[str, Any]],
    length: int,
) -> list[tuple[str, str, str]]:
    available = [
        (float(item["timestamp_sec"]), str(item["path"]))
        for item in samples
        if Path(str(item["path"])).is_file()
    ]
    if not available:
        raise RuntimeError("media cache contains no readable frames")
    nearest = [
        min(available, key=lambda item: abs(item[0] - timestamp))[1]
        for timestamp in range(length)
    ]
    return [
        (
            nearest[max(0, timestamp - 1)],
            nearest[timestamp],
            nearest[min(length - 1, timestamp + 1)],
        )
        for timestamp in range(length)
    ]


def _nearby_text(
    transcript: list[dict[str, Any]],
    timestamp: float,
    source: str,
) -> str:
    return " ".join(
        str(segment["text"])
        for segment in transcript
        if segment.get("source", "asr") == source
        and float(segment["end_sec"]) >= timestamp - 1.5
        and float(segment["start_sec"]) <= timestamp + 1.5
    ).strip()


def _read_waveform(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as source:
        if source.getframerate() != 16000:
            raise RuntimeError(f"expected 16 kHz audio cache: {path}")
        channels = source.getnchannels()
        raw = source.readframes(source.getnframes())
    waveform = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        waveform = waveform.reshape(-1, channels).mean(axis=1)
    return waveform


def _audio_priors(
    audio_path: Path,
    events: list[dict[str, Any]],
    length: int,
) -> Tensor:
    waveform = _read_waveform(audio_path)
    energy = torch.zeros(length)
    for second in range(length):
        chunk = waveform[second * 16000 : (second + 1) * 16000]
        if len(chunk):
            energy[second] = float(np.sqrt(np.mean(np.square(chunk))))
    positive = energy[energy > 0]
    scale = torch.quantile(positive, 0.9) if len(positive) else torch.tensor(1.0)
    energy = torch.log1p(energy / scale.clamp_min(1e-6)).unsqueeze(1)
    event_prior = torch.zeros((length, len(EVENT_NAMES)))
    emotion_prior = torch.zeros((length, len(EMOTION_NAMES)))
    for item in events:
        start = max(0, math.floor(float(item.get("start_sec", 0))))
        end = min(length, max(start + 1, math.ceil(float(item.get("end_sec", start + 1)))))
        event = str(item.get("event", "speech")).lower()
        emotion = str(item.get("emotion", "neutral")).lower()
        if event in EVENT_NAMES:
            event_prior[start:end, EVENT_NAMES.index(event)] = 1.0
        if emotion in EMOTION_NAMES:
            emotion_prior[start:end, EMOTION_NAMES.index(emotion)] = 1.0
    return torch.cat([energy, event_prior, emotion_prior], dim=1)


def _merge_scene_bounds(
    scenes: list[dict[str, Any]],
    length: int,
    min_sec: int = 8,
    target_sec: int = 20,
    max_sec: int = 40,
) -> Tensor:
    ends = sorted(
        {
            min(length, max(1, round(float(scene["end_sec"]))))
            for scene in scenes
            if isinstance(scene, dict) and "end_sec" in scene
        }
    )
    bounds: list[tuple[int, int]] = []
    start = 0
    while start < length:
        candidates = [end for end in ends if start + min_sec <= end <= start + max_sec]
        preferred = [end for end in candidates if end >= start + target_sec]
        if preferred:
            end = preferred[0]
        elif candidates:
            end = candidates[-1]
        else:
            end = min(length, start + max_sec)
        if length - end < min_sec:
            end = length
        bounds.append((start, max(start + 1, end)))
        start = end
    return torch.tensor(bounds, dtype=torch.long)

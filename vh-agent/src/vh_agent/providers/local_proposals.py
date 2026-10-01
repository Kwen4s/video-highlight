"""Optional trained localizer: full-episode inference, uncapped proposal pool."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from ..preprocessing.media import extract_audio, extract_frames, video_fingerprint
from ..storage import write_json


class LocalProposals:
    def __init__(self, evidence, settings, language, video_id=None):
        if not language:
            raise ValueError("Local proposals require an explicit video language")
        self.evidence, self.settings, self.language = evidence, settings, language
        self.video_id = video_id

    @property
    def profile(self) -> dict:
        checkpoint = self.settings.local_checkpoint
        return {
            "checkpoint_hash": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "config_hash": hashlib.sha256(
                checkpoint.with_name("config.json").read_bytes()
            ).hexdigest(),
            "language": self.language,
            "settings": {
                key: str(value)
                for key, value in self.settings.model_dump().items()
                if key.startswith(("local_", "asr_", "sensevoice_", "ocr_"))
            },
            "implementation": hashlib.sha256(
                b"".join(
                    p.read_bytes()
                    for p in sorted((Path(__file__).parents[1] / "highlight_model").glob("*.py"))
                )
            ).hexdigest(),
        }

    def __call__(self):
        import torch

        from ..highlight_model.config import HighlightModelConfig
        from ..highlight_model.dataset import SilverVideo, feature_path
        from ..highlight_model.decoding import decode_segments
        from ..highlight_model.encoders import FrozenMomentFeatureExtractor
        from ..highlight_model.network import NarrativeTransitionLocalizer

        if not self.evidence.video_info.has_audio:
            raise ValueError(
                "This checkpoint requires audio; local proposals unavailable for silent video"
            )
        settings = self.settings
        config_path = settings.local_checkpoint.parent / "config.json"
        values = json.loads(config_path.read_text())
        for key in (
            "annotations",
            "output_dir",
            "vision_model_path",
            "audio_model_path",
            "media_cache_dir",
            "feature_cache_dir",
            "init_checkpoint",
        ):
            if values.get(key) is not None:
                values[key] = Path(values[key])
        media_profile = {
            "language": self.language,
            "preparation": {
                key: str(value)
                for key, value in settings.model_dump().items()
                if key.startswith(("asr_", "sensevoice_", "ocr_"))
            },
            "frame_fps": 1,
            "frame_width": 480,
            "implementation": hashlib.sha256(
                b"".join(
                    p.read_bytes()
                    for p in sorted((Path(__file__).parents[1] / "preprocessing").glob("*.py"))
                )
            ).hexdigest(),
        }
        cache_namespace = (
            "local_"
            + hashlib.sha256(json.dumps(media_profile, sort_keys=True).encode()).hexdigest()[:16]
        )
        values.update(
            device=settings.local_device,
            feature_device=settings.local_device,
            media_cache_dir=settings.media_cache_dir / cache_namespace,
            feature_cache_dir=settings.local_feature_cache_dir / cache_namespace,
        )
        config = HighlightModelConfig(**values)
        info = self.evidence.video_info
        video_id = self.video_id or video_fingerprint(info.path)
        cache = config.media_cache_dir / video_fingerprint(info.path)
        cache.mkdir(parents=True, exist_ok=True)
        if not (cache / "preprocess.json").is_file():
            from ..preprocessing.scene_detection import detect_scenes
            from ..preprocessing.sensevoice import extract_audio_events
            from ..preprocessing.subtitle_ocr import extract_subtitle_segments
            from ..preprocessing.transcription import FasterWhisperTranscriber

            audio = extract_audio(info.path, cache / "audio.wav")
            frames = extract_frames(info.path, cache / "frames", sample_fps=1, frame_width=480)
            transcriber = FasterWhisperTranscriber(
                settings.asr_model, settings.asr_device, settings.asr_compute_type
            )
            transcript = transcriber.transcribe(audio, self.language)
            transcript += extract_subtitle_segments(
                frames,
                language=self.language,
                device=settings.ocr_device,
                ocr_version=settings.ocr_version,
            )
            events = extract_audio_events(
                audio,
                settings.sensevoice_model,
                settings.sensevoice_vad_model,
                settings.sensevoice_device,
                info.duration_sec,
            )
            write_json(
                cache / "preprocess.json",
                {
                    "signature": {"source": video_fingerprint(info.path), **media_profile},
                    "frame_samples": [f.model_dump(mode="json") for f in frames],
                    "transcript": [t.model_dump(mode="json") for t in transcript],
                    "audio_events": [e.model_dump(mode="json") for e in events],
                    "scenes": [s.model_dump(mode="json") for s in detect_scenes(info.path)],
                },
            )
        video = SilverVideo(
            video_id, video_id, info.path, info.duration_sec, self.language, (), ()
        )
        extractor = FrozenMomentFeatureExtractor(config)
        try:
            extractor.prepare([video])
        finally:
            extractor.close()
        features = torch.load(
            feature_path(config.feature_cache_dir, video_id), map_location="cpu", weights_only=True
        )
        model = (
            NarrativeTransitionLocalizer(
                vision_dim=features["vision"].shape[1],
                audio_dim=features["audio"].shape[1],
                audio_prior_dim=features["audio_prior"].shape[1],
                model_dim=config.model_dim,
                heads=config.attention_heads,
                temporal_layers_per_level=config.temporal_layers_per_level,
                dropout=config.dropout,
                max_before_sec=config.max_before_sec,
                max_after_sec=config.max_after_sec,
                max_center_offset_sec=config.max_center_offset_sec,
                min_segment_duration_sec=config.min_segment_duration_sec,
                max_segment_duration_sec=config.max_segment_duration_sec,
            )
            .to(config.device)
            .eval()
        )
        checkpoint = torch.load(
            settings.local_checkpoint, map_location=config.device, weights_only=True
        )
        if checkpoint["schema"] != 4:
            raise ValueError("Local checkpoint must use schema 4")
        model.load_state_dict(checkpoint["model"], strict=True)
        with torch.inference_mode():
            output = model(
                *(
                    features[k].to(config.device).float()
                    for k in ("vision", "audio", "audio_prior", "availability")
                ),
                features["scene_bounds"].to(config.device),
            )
        return [
            asdict(p)
            for p in decode_segments(
                output,
                info.duration_sec,
                threshold=config.score_threshold,
                nms_iou=config.nms_iou,
                max_highlights=None,
                min_duration_sec=config.min_segment_duration_sec,
                max_duration_sec=config.max_segment_duration_sec,
            )
        ]

from vh_agent.preprocessing.media import _clean_title


def test_clean_short_drama_filename() -> None:
    stem = "(洋洋放映官)P9_第9集_#一纸医院报告，拆穿儿媳谎言【火爆新剧，免费观看全集】_480P"
    assert _clean_title(stem) == "一纸医院报告，拆穿儿媳谎言"


def test_media_cache_reuses_matching_audio_and_frames(monkeypatch, tmp_path) -> None:
    import json
    import wave

    from PIL import Image

    from vh_agent.preprocessing import media

    audio_path = tmp_path / "audio.wav"
    with wave.open(str(audio_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x00\x00" * 160)

    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    Image.new("RGB", (32, 32), "black").save(frame_dir / "frame_000001.jpg")
    (frame_dir / ".complete").write_text(
        json.dumps({"sample_fps": 1.0, "frame_width": 480}, sort_keys=True),
        encoding="utf-8",
    )

    def fail_if_called(command):
        raise AssertionError(f"unexpected media extraction: {command}")

    monkeypatch.setattr(media, "_run", fail_if_called)
    assert media.extract_audio(tmp_path / "video.mp4", audio_path) == audio_path
    frames = media.extract_frames(tmp_path / "video.mp4", frame_dir, 1.0, 480)
    assert len(frames) == 1
    assert frames[0].timestamp_sec == 0.0


def test_english_audio_uses_bilingual_chinese_ocr_model(monkeypatch) -> None:
    from vh_agent.preprocessing import subtitle_ocr

    requested: list[str] = []

    def fake_load_ocr(*, lang, **_kwargs):
        requested.append(lang)
        return object()

    monkeypatch.setattr(subtitle_ocr, "_load_ocr", fake_load_ocr)
    assert subtitle_ocr.extract_subtitle_segments([], "en", "cpu") == []
    assert requested == ["ch"]


def test_subtitle_ocr_batches_frames_without_losing_timestamps(monkeypatch, tmp_path) -> None:
    from PIL import Image

    from vh_agent.models import FrameSample
    from vh_agent.preprocessing import subtitle_ocr

    frames = []
    for index in range(9):
        path = tmp_path / f"frame_{index:02d}.jpg"
        Image.new("RGB", (32, 48), "black").save(path)
        frames.append(FrameSample(timestamp_sec=float(index), path=path))

    class FakeOCR:
        def __init__(self):
            self.batch_sizes = []

        def predict(self, *, input):
            self.batch_sizes.append(len(input))
            return [
                {
                    "res": {
                        "rec_texts": [f"subtitle-{len(self.batch_sizes)}-{index}"],
                        "rec_scores": [0.9],
                    }
                }
                for index, _image in enumerate(input)
            ]

    ocr = FakeOCR()
    monkeypatch.setattr(subtitle_ocr, "_load_ocr", lambda **_kwargs: ocr)
    segments = subtitle_ocr.extract_subtitle_segments(frames, "zh", "cpu")

    assert ocr.batch_sizes == [8, 1]
    assert len(segments) == 9
    assert [segment.start_sec for segment in segments] == [float(index) for index in range(9)]



def test_whisper_uses_deterministic_greedy_decoding(tmp_path) -> None:
    from types import SimpleNamespace

    from vh_agent.preprocessing.transcription import FasterWhisperTranscriber

    requests = []

    class FakeModel:
        def transcribe(self, audio_path, **kwargs):
            requests.append((audio_path, kwargs))
            return (
                [
                    SimpleNamespace(
                        start=1.0,
                        end=2.0,
                        text="台词",
                        avg_logprob=-0.1,
                    )
                ],
                None,
            )

    transcriber = FasterWhisperTranscriber.__new__(FasterWhisperTranscriber)
    transcriber.model = FakeModel()
    segments = transcriber.transcribe(tmp_path / "audio.wav", "zh")

    assert len(segments) == 1
    assert requests[0][1]["beam_size"] == 1
    assert requests[0][1]["best_of"] == 1
    assert requests[0][1]["condition_on_previous_text"] is False

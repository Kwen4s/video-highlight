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

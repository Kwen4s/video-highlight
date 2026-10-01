from types import SimpleNamespace

from vh_agent.config import Settings
from vh_agent.providers.retrieval import TranscriptSearch


def test_subtitle_search_is_paginated_and_source_attributed(tmp_path):
    path = tmp_path / "test.srt"
    path.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nHello world\n\n2\n00:00:03,000 --> 00:00:04,000\nHello again\n"
    )
    evidence = SimpleNamespace(
        media_id="v", output_dir=tmp_path, video_info=SimpleNamespace(duration_sec=10)
    )
    search = TranscriptSearch(evidence, Settings(), path, "en")
    first = search.search("hello", 0, None, 0, 1)
    assert first["total"] == 2 and first["next_offset"] == 1
    assert first["matches"][0]["source"] == "subtitle"
    assert first["matches"][0]["start_sec"] == 1
    assert search.search("again", 0, None, 0, 10)["matches"][0]["start_sec"] == 3

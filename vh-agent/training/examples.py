"""Export first success and failure cases per drama, with visible time markers."""

import subprocess
from concurrent.futures import ThreadPoolExecutor


def export_examples(report, folder):
    selected = []
    for drama in sorted({d["drama_id"] for d in report["details"]}):
        cohort = [d for d in report["details"] if d["drama_id"] == drama]
        for categories in ({"命中"}, {"漏选", "误选", "漏选和误选", "输出错误"}, {"边界偏差"}):
            found = next((d for d in cohort if d["category"] in categories), None)
            if found is not None:
                selected.append(found)
    directory = folder / "examples"
    directory.mkdir(exist_ok=True)
    for i, detail in enumerate(selected, 1):
        detail["example_file"] = f"examples/{i:02d}.mp4"

    def encode(detail):
        video = detail["video"]
        destination = folder / detail["example_file"]
        filters = ["scale=w='min(720,iw)':h=-2"]
        for spans, color, y, label in (
            (detail["reference"], "0x158354", 8, "REF"),
            (detail["predicted"], "0x245ccc", 40, "MODEL"),
        ):
            for s in spans:
                filters.append(
                    "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
                    f"text='{label}':fontsize=20:fontcolor=white:box=1:boxcolor={color}@0.85:"
                    f"x=8:y={y}:enable='between(t,{s['start_sec']},{s['end_sec']})'"
                )
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                str(video["start_sec"]),
                "-i",
                video["path"],
                "-t",
                str(video["end_sec"] - video["start_sec"]),
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-vf",
                ",".join(filters),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-threads",
                "2",
                "-c:a",
                "aac",
                "-b:a",
                "96k",
                "-movflags",
                "+faststart",
                str(destination),
            ],
            check=True,
        )
        print(f"example_ready={detail['example_file']} category={detail['category']}", flush=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(encode, selected))

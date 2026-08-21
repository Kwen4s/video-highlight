import base64
import hashlib
import json
from io import BytesIO

from openai import OpenAI
from PIL import Image, ImageDraw

from .config import MAX_HIGHLIGHT_SEC, SETUP_LEAD_SEC, Settings
from .models import (
    EvidenceLedger,
    FrameSample,
    GlobalRanking,
    JudgeConsensus,
    JudgeDecision,
    RankedHighlight,
    SceneCard,
    SceneNarrative,
    VideoInfo,
)

MAP_SYSTEM_PROMPT = """从当前 SceneCard 中抽取可核验的场景事实。只描述输入中可见或可听的内容，不裁决它是不是高光。

action 记录可观察行为；claims 记录人物说出的事实性主张，主张本身是可观察言语，但其内容仍未证实；state_before 和 state_after 记录人物或观众在认知、关系、目标、风险、权力或选择上的前后状态，材料不足时填 unknown；new_evidence 记录造成变化的可观察触发；relationship_change 和 emotion 只写本场可支持的结果。

event_type 使用一到两个简短场景标签，可优先使用 conflict、reversal、reveal、payoff、emotion、action、romance、cliffhanger，也可使用更准确的开放标签。salience 衡量本场对叙事推进的强度，uncertainty 衡量解释的不确定性。每项核心判断引用带时间戳的帧、字幕或声音证据。人物无法可靠识别时使用稳定的角色描述。scene_id 与输入一致。
只输出符合指定结构的 JSON。"""

JUDGE_SYSTEM_PROMPT = """判断当前 SceneCard 是否具备独立短视频高光价值。只核验 Scene Map 已提出的核心事件；更早账本仅用于定位背景，事实依据来自当前场或紧邻前场。

高光可由五种通用结构成立：
1. 叙事变化：旧状态 → 决定性证据 → 新状态。
2. 冲突兑现：约束或目标分歧 → 对抗 → 可见后果。
3. 情绪或关系兑现：必要铺垫 → 触发 → 明确反应或关系变化。
4. 动作峰值：风险或目标建立 → 峰值行为 → 可见结果。
5. 悬念钩子：关键信息出现 → 高风险问题成立且仍未解决。

人物主张的真实性与戏剧作用分开处理：缺少独立证据时仍可把言语行为作为冲突或情绪触发，但不能把主张内容写成已证实事实。

分别给出四项 0 到 1 的分数：evidence_grounding 衡量核心解释是否有直接证据；narrative_impact 衡量状态、风险、关系、情绪或悬念变化的强度；standalone_clarity 衡量脱离全片后是否能看懂；clipability 衡量必要铺垫、触发和结果能否在 24 秒内形成完整片段。分数只反映各维度，不围绕通过阈值打分。

播放窗口覆盖最短必要铺垫、当前场的决定性证据和已有反应。当前场承接紧邻前场的同一事件时 continue_previous_scene=true。counter_evidence 记录削弱核心解释或独立成片价值的材料。
只输出符合指定结构的 JSON。"""

EVIDENCE_JUDGE_PROMPT = (
    JUDGE_SYSTEM_PROMPT
    + """

本次以证据核验为主：优先检查 Scene Map 的核心事件、时间锚点和前后状态是否被输入直接支持；证据不足时降低 evidence_grounding，不补写缺失事实。"""
)

EDITOR_JUDGE_PROMPT = (
    JUDGE_SYSTEM_PROMPT
    + """

本次以成片判断为主：优先检查事件是否造成值得保留的变化、脱离全片是否可懂、24 秒内能否形成完整观看单元；不要因为台词激烈或情绪外显而自动判为高光。"""
)

ADJUDICATOR_SYSTEM_PROMPT = """你是高光标注仲裁者。比较两份独立 Judge 结果，并重新核对同一份 SceneCard 证据。选择证据更充分的解释；两份都不成立时明确否决，两份都成立但类型不同则选择最能描述核心变化的类型。保持通用判据，不引入输入之外的剧情。输出与 Judge 相同结构的 JSON。"""

LISTWISE_SYSTEM_PROMPT = """对已通过证据核验的候选进行全片级选择。保留不重复的核心事件并覆盖不同叙事功能；同一决定性证据或相互包含的完整事件视为重复。按证据充分性、叙事影响、独立可懂性和成片完整性综合排序，只选择值得最终出片的最强组合。数量预算是上限而非配额，可以少选或不选。候选的事实、类型和时间边界保持不变。只输出 JSON。"""

MAP_FEW_SHOT_MESSAGES = [
    {
        "role": "user",
        "content": """合成结构示例，P/Q/E 只是占位符，实际任务必须改写为输入中的具体事实：
SceneCard scene_example [0.00, 9.00]
[1.00s ASR] 角色甲主张命题 P，并据此维持决定 Q。
[5.00s FRAME] 可见证据 E 直接否定 P。
[7.00s ASR] 角色甲承认证据并改变决定。""",
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "scene_id": "scene_example",
                "actors": ["角色甲"],
                "action": "角色甲看到证据后改变决定",
                "claims": ["命题 P"],
                "event_type": ["reversal"],
                "state_before": "角色甲相信 P 并维持 Q",
                "new_evidence": "可见证据 E 否定 P",
                "state_after": "角色甲不再相信 P，并改变 Q",
                "relationship_change": "",
                "emotion": ["震惊"],
                "salience": 0.88,
                "uncertainty": 0.08,
                "evidence": [
                    "[1.00s ASR] 角色甲主张 P",
                    "[5.00s FRAME] 证据 E 否定 P",
                    "[7.00s ASR] 角色甲改变决定",
                ],
            },
            ensure_ascii=False,
        ),
    },
]

JUDGE_FEW_SHOT_MESSAGES = [
    {
        "role": "user",
        "content": """合成正例：Scene Map 显示旧状态 Q 在 1 秒成立，5 秒出现直接证据 E，7 秒出现明确的新状态与反应；证据均来自当前场。请按 Judge 合同裁决。""",
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "map_supported": True,
                "highlight_type": "reversal",
                "description": "证据推翻旧认知并改变决定",
                "reason": "形成旧状态、决定性证据和新状态的完整叙事变化",
                "evidence_grounding": 0.95,
                "narrative_impact": 0.9,
                "standalone_clarity": 0.88,
                "clipability": 0.92,
                "start_sec": 1.0,
                "end_sec": 8.0,
                "evidence": ["[1.00s] 旧状态 Q", "[5.00s] 证据 E", "[7.00s] 新状态"],
                "setup_evidence_times_sec": [1.0],
                "decisive_evidence_times_sec": [5.0, 7.0],
                "counter_evidence": [],
                "continue_previous_scene": False,
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "user",
        "content": """合成负例：当前场只有角色甲重复主张 P，没有独立证据、可见后果、关系变化、动作结果或未解决的高风险问题。请按 Judge 合同裁决。""",
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "map_supported": True,
                "highlight_type": "other",
                "description": "角色重复一项未经证实的主张",
                "reason": "只有主张，没有形成五种通用高光结构中的任一种",
                "evidence_grounding": 0.9,
                "narrative_impact": 0.18,
                "standalone_clarity": 0.55,
                "clipability": 0.35,
                "start_sec": None,
                "end_sec": None,
                "evidence": ["[3.00s ASR] 角色甲主张 P"],
                "setup_evidence_times_sec": [],
                "decisive_evidence_times_sec": [],
                "counter_evidence": ["没有证据或可见状态变化"],
                "continue_previous_scene": False,
            },
            ensure_ascii=False,
        ),
    },
]


def scene_map_request_fingerprint(settings: Settings, video: VideoInfo, scene: SceneCard) -> str:
    frame_files = [
        {
            "path": str(frame.path),
            "size": frame.path.stat().st_size,
            "mtime_ns": frame.path.stat().st_mtime_ns,
        }
        for frame in scene.frame_samples[:3]
    ]
    payload = {
        "provider": settings.reasoning_provider,
        "base_url": settings.reasoning_base_url,
        "model": settings.reasoning_map_model,
        "system_prompt": MAP_SYSTEM_PROMPT,
        "few_shot": MAP_FEW_SHOT_MESSAGES,
        "user_prompt": _scene_map_prompt(video, scene),
        "scene": scene.model_dump(mode="json"),
        "frame_files": frame_files,
        "max_frames": 3,
        "detail": "low",
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


class OpenAIReasoner:
    """Cloud reasoner coordinating Scene Map, evidence verification, and ranking."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if not settings.reasoning_api_key:
            raise ValueError(f"{settings.reasoning_provider.upper()}_API_KEY is empty")
        self.client = OpenAI(
            api_key=settings.reasoning_api_key,
            base_url=settings.reasoning_base_url,
            timeout=settings.request_timeout_sec,
            max_retries=settings.request_max_retries,
        )

    def map_scene(self, video: VideoInfo, scene: SceneCard) -> SceneNarrative:
        try:
            payload = self._json_completion(
                model=self.settings.reasoning_map_model,
                messages=[
                    {"role": "system", "content": MAP_SYSTEM_PROMPT},
                    *MAP_FEW_SHOT_MESSAGES,
                    {
                        "role": "user",
                        "content": _multimodal_content(
                            scene,
                            _scene_map_prompt(video, scene),
                            max_frames=3,
                            detail="low",
                        ),
                    },
                ],
                max_tokens=self._map_output_tokens(),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Scene Map failed for {scene.scene_id} "
                f"[{scene.start_sec:.2f}, {scene.end_sec:.2f}]: {exc}"
            ) from exc
        narrative = SceneNarrative.model_validate(payload)
        if narrative.scene_id != scene.scene_id:
            raise ValueError("Scene Map response scene_id does not match the submitted SceneCard")
        return narrative

    def judge(
        self,
        video: VideoInfo,
        scene: SceneCard,
        previous_scene: SceneCard | None,
        evidence_ledger: EvidenceLedger,
    ) -> JudgeConsensus:
        first = self._judge_once(
            video,
            scene,
            previous_scene,
            evidence_ledger,
            EVIDENCE_JUDGE_PROMPT,
        )
        second = self._judge_once(
            video,
            scene,
            previous_scene,
            evidence_ledger,
            EDITOR_JUDGE_PROMPT,
        )
        votes = [first, second]
        if _needs_adjudication(first, second, self.settings.final_threshold):
            votes.append(
                self._adjudicate(
                    video,
                    scene,
                    previous_scene,
                    evidence_ledger,
                    first,
                    second,
                )
            )
        decision = _consensus_decision(votes, self.settings.final_threshold)
        return JudgeConsensus(decision=decision, votes=votes, calls=len(votes))

    def _judge_once(
        self,
        video: VideoInfo,
        scene: SceneCard,
        previous_scene: SceneCard | None,
        evidence_ledger: EvidenceLedger,
        system_prompt: str,
    ) -> JudgeDecision:
        payload = self._json_completion(
            model=self.settings.reasoning_judge_model,
            messages=[
                {"role": "system", "content": system_prompt},
                *JUDGE_FEW_SHOT_MESSAGES,
                {
                    "role": "user",
                    "content": _multimodal_content(
                        scene,
                        _judge_prompt(
                            video,
                            scene,
                            previous_scene,
                            evidence_ledger,
                        ),
                        max_frames=8,
                        detail="high",
                    ),
                },
            ],
            max_tokens=self._judge_output_tokens(),
        )
        return _finalize_vote(JudgeDecision.model_validate(payload), self.settings.final_threshold)

    def _adjudicate(
        self,
        video: VideoInfo,
        scene: SceneCard,
        previous_scene: SceneCard | None,
        evidence_ledger: EvidenceLedger,
        first: JudgeDecision,
        second: JudgeDecision,
    ) -> JudgeDecision:
        prompt = _judge_prompt(video, scene, previous_scene, evidence_ledger)
        prompt += (
            "\n两份独立裁决：\n"
            + json.dumps(
                [first.model_dump(mode="json"), second.model_dump(mode="json")],
                ensure_ascii=False,
            )
            + "\n重新核对原始证据后输出最终裁决。"
        )
        payload = self._json_completion(
            model=self.settings.reasoning_judge_model,
            messages=[
                {"role": "system", "content": ADJUDICATOR_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _multimodal_content(scene, prompt, max_frames=8, detail="high"),
                },
            ],
            max_tokens=self._judge_output_tokens(),
        )
        return _finalize_vote(JudgeDecision.model_validate(payload), self.settings.final_threshold)

    def rank_highlights(
        self,
        video: VideoInfo,
        highlights: list[RankedHighlight],
        max_selected: int,
    ) -> GlobalRanking:
        candidates = [
            {
                "highlight_id": item.highlight_id,
                "time": [item.start_sec, item.end_sec],
                "judge_score": item.judge_score,
                "confidence": item.confidence,
                "highlight_type": item.highlight_type,
                "description": item.description,
                "reason": item.reason,
                "evidence": item.evidence,
            }
            for item in highlights
        ]
        prompt = (
            f"视频：{video.title}，时长 {video.duration_sec:.2f}s\n"
            f"最多选择 {max_selected} 条，不为填满预算保留弱候选。\n"
            "候选："
            + json.dumps(candidates, ensure_ascii=False)
            + "\n输出 JSON：ranked_highlight_ids 为全部候选的完整排序；"
            "selected_highlight_ids 为最终选择；rationale 为简短理由。"
        )
        payload = self._json_completion(
            model=self.settings.reasoning_judge_model,
            messages=[
                {"role": "system", "content": LISTWISE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self._judge_output_tokens(),
        )
        ranking = GlobalRanking.model_validate(payload)
        expected = [item.highlight_id for item in highlights]
        if len(ranking.ranked_highlight_ids) != len(expected) or set(
            ranking.ranked_highlight_ids
        ) != set(expected):
            raise ValueError("Listwise ranking must be a complete candidate permutation")
        if (
            len(ranking.selected_highlight_ids) > max_selected
            or len(set(ranking.selected_highlight_ids)) != len(ranking.selected_highlight_ids)
            or not set(ranking.selected_highlight_ids).issubset(expected)
        ):
            raise ValueError("Listwise selection is invalid")
        return ranking

    def _json_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, object]],
        max_tokens: int,
    ) -> dict[str, object]:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            usage = response.usage
            raise ValueError(
                "Empty model completion: "
                f"model={model}, finish_reason={response.choices[0].finish_reason}, "
                f"completion_tokens={getattr(usage, 'completion_tokens', None)}"
            )
        return _decode_json_object(content)

    def _map_output_tokens(self) -> int:
        return 3000 if self.settings.reasoning_provider == "gemini" else 1600

    def _judge_output_tokens(self) -> int:
        return 2400 if self.settings.reasoning_provider == "gemini" else 1200


def _decode_json_object(content: str) -> dict[str, object]:
    text = content.strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[len("```json\n") : -len("\n```")]
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise TypeError("Model response is not a JSON object")
    return payload


def _scene_map_prompt(video: VideoInfo, scene: SceneCard) -> str:
    scene_payload = {
        "scene_id": scene.scene_id,
        "time": [scene.start_sec, scene.end_sec],
        "transcript": scene.transcript or "[无字幕]",
        "audio_context": scene.audio_context or "[无声音事件]",
        "speakers": scene.speakers,
    }
    return f"""视频：{video.title}
输入 SceneCard：{json.dumps(scene_payload, ensure_ascii=False)}

输出：
{{
  "scene_id": "输入 ID", "actors": ["人物或 speaker 标识"],
  "action": "画面和声音里实际发生的动作",
  "claims": ["人物声称但未经本场独立证实的内容"],
  "event_type": ["一到两个简短场景标签"],
  "state_before": "开场可见的旧状态，否则 unknown",
  "new_evidence": "本场可独立观察的证据",
  "state_after": "本场证据支持的新状态与当场反应",
  "relationship_change": "关系或权力变化",
  "emotion": ["本场可直接观察的主要情绪"],
  "salience": 0到1, "uncertainty": 0到1,
  "evidence": ["F编号或带时间戳的字幕/声音证据"]
}}"""


def _judge_prompt(
    video: VideoInfo,
    scene: SceneCard,
    previous_scene: SceneCard | None,
    evidence_ledger: EvidenceLedger,
) -> str:
    return f"""视频：{video.title}
当前场景：{json.dumps(scene.model_dump(mode="json", exclude={"frame_samples"}), ensure_ascii=False)}
紧邻前场，可供铺垫：{json.dumps(previous_scene.model_dump(mode="json", exclude={"frame_samples"}) if previous_scene else {}, ensure_ascii=False)}
更早场次的未验证账本：{json.dumps(evidence_ledger.model_dump(mode="json"), ensure_ascii=False)}

关键帧左上角的 F 编号和秒数是证据标识。evidence 引用支持核心状态变化的帧、字幕或声音。

输出：
{{
  "map_supported": true,
  "highlight_type": "conflict|reversal|reveal|payoff|emotion|action|romance|cliffhanger|other",
  "description": "可观察到的核心事件", "reason": "符合哪一种通用高光结构及其证据",
  "evidence_grounding": 0到1, "narrative_impact": 0到1,
  "standalone_clarity": 0到1, "clipability": 0到1,
  "start_sec": 数字或 null, "end_sec": 数字或 null,
  "evidence": ["带时间戳的帧、台词或声音证据"],
  "setup_evidence_times_sec": [证明旧状态或必要铺垫的秒数，可在当前场或紧邻前场],
  "decisive_evidence_times_sec": [造成状态变化的证据秒数，须在当前场],
  "counter_evidence": ["与核心解释冲突、或降低独立性的证据"],
  "continue_previous_scene": false
}}"""


def _multimodal_content(
    scene: SceneCard, prompt: str, *, max_frames: int, detail: str
) -> list[dict[str, object]]:
    content: list[dict[str, object]] = []
    frame_samples = scene.frame_samples
    if len(frame_samples) > max_frames:
        indices = [
            round(index * (len(frame_samples) - 1) / (max_frames - 1))
            for index in range(max_frames)
        ]
        frame_samples = [frame_samples[index] for index in indices]
    for index, frame in enumerate(frame_samples, start=1):
        label = f"F{index:02d}  {frame.timestamp_sec:.1f}s"
        content.append({"type": "text", "text": label})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_data_url(frame, label), "detail": detail},
            }
        )
    content.append({"type": "text", "text": prompt})
    return content


def _image_data_url(frame: FrameSample, label: str) -> str:
    with Image.open(frame.path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = draw.textbbox((0, 0), label)
    draw.rectangle((0, 0, right - left + 12, bottom - top + 10), fill="black")
    draw.text((6, 5), label, fill="white")
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _score_decision(decision: JudgeDecision) -> JudgeDecision:
    score = (
        0.30 * decision.evidence_grounding
        + 0.30 * decision.narrative_impact
        + 0.20 * decision.standalone_clarity
        + 0.20 * decision.clipability
    )
    return decision.model_copy(update={"score": round(score, 4)})


def _provisional_highlight(decision: JudgeDecision, threshold: float) -> bool:
    return bool(
        decision.map_supported
        and decision.highlight_type != "other"
        and decision.score >= threshold
        and decision.evidence
        and decision.decisive_evidence_times_sec
    )


def _finalize_vote(decision: JudgeDecision, threshold: float) -> JudgeDecision:
    scored = _score_decision(decision)
    return scored.model_copy(update={"is_highlight": _provisional_highlight(scored, threshold)})


def _needs_adjudication(
    first: JudgeDecision,
    second: JudgeDecision,
    threshold: float,
) -> bool:
    first_passes = _provisional_highlight(first, threshold)
    second_passes = _provisional_highlight(second, threshold)
    if first.map_supported != second.map_supported or first_passes != second_passes:
        return True
    if first_passes and first.highlight_type != second.highlight_type:
        return True
    if abs(first.score - second.score) >= 0.15:
        return True
    if first_passes and _decision_window_iou(first, second) < 0.5:
        return True
    return False


def _decision_window_iou(first: JudgeDecision, second: JudgeDecision) -> float:
    if None in (first.start_sec, first.end_sec, second.start_sec, second.end_sec):
        return 0.0
    overlap = max(
        0.0,
        min(first.end_sec, second.end_sec) - max(first.start_sec, second.start_sec),
    )
    union = max(first.end_sec, second.end_sec) - min(first.start_sec, second.start_sec)
    return overlap / union if union > 0 else 0.0


def _consensus_decision(votes: list[JudgeDecision], threshold: float) -> JudgeDecision:
    chosen = (
        votes[-1]
        if len(votes) == 3
        else max(
            votes,
            key=lambda item: (item.evidence_grounding, item.score),
        )
    )
    chosen_passes = _provisional_highlight(chosen, threshold)
    agreeing = sum(
        _provisional_highlight(vote, threshold) == chosen_passes
        and (not chosen_passes or vote.highlight_type == chosen.highlight_type)
        for vote in votes
    )
    score_spread = max(vote.score for vote in votes) - min(vote.score for vote in votes)
    confidence = 0.75 * (agreeing / len(votes)) + 0.25 * (1.0 - score_spread)
    return chosen.model_copy(
        update={
            "is_highlight": chosen_passes,
            "confidence": round(max(0.0, min(1.0, confidence)), 4),
        }
    )


def validate_highlight_decision(
    decision: JudgeDecision,
    scene: SceneCard,
    previous_scene: SceneCard | None,
) -> JudgeDecision:
    """Keep the Judge event window if the causal core fits in 24s.

    Distant setup may locate old cognition but must not stretch the clip.
    If the event-local core itself exceeds 24s, the clip fails.
    """
    if not decision.map_supported:
        return decision.model_copy(update={"is_highlight": False})
    if not decision.is_highlight:
        return decision
    setup_floor = previous_scene.start_sec if previous_scene is not None else scene.start_sec
    setup_times = [
        time for time in decision.setup_evidence_times_sec if setup_floor <= time <= scene.end_sec
    ]
    decisive_times = [
        time
        for time in decision.decisive_evidence_times_sec
        if scene.start_sec <= time <= scene.end_sec
    ]
    if not decision.evidence or not decisive_times:
        return decision.model_copy(update={"is_highlight": False})

    start_sec = decision.start_sec if decision.start_sec is not None else scene.start_sec
    end_sec = decision.end_sec if decision.end_sec is not None else scene.end_sec
    event_setup = _event_local_setup_times(setup_times, decisive_times, start_sec, end_sec)
    fitted = _fit_playable_window(
        start_sec,
        end_sec,
        floor=setup_floor,
        ceiling=scene.end_sec,
        core_times=[*event_setup, *decisive_times],
    )
    if fitted is None:
        return decision.model_copy(update={"is_highlight": False})
    start_sec, end_sec = fitted
    if end_sec - start_sec < 0.5:
        return decision.model_copy(update={"is_highlight": False})
    return decision.model_copy(
        update={
            "start_sec": start_sec,
            "end_sec": end_sec,
            "setup_evidence_times_sec": [
                time for time in setup_times if start_sec <= time <= end_sec
            ],
            "decisive_evidence_times_sec": [
                time for time in decisive_times if start_sec <= time <= end_sec
            ],
        }
    )


def _event_local_setup_times(
    setup_times: list[float],
    decisive_times: list[float],
    window_start: float,
    window_end: float,
) -> list[float]:
    """Setup may reshape a clip only if it already sits in the Judge window or
    immediately precedes the first decisive time. Earlier scene context stays
    evidence of old cognition, not extra padding.
    """
    first_decisive = min(decisive_times) if decisive_times else window_start
    last_decisive = max(decisive_times) if decisive_times else window_end
    return [
        time
        for time in setup_times
        if window_start <= time <= window_end
        or first_decisive - SETUP_LEAD_SEC <= time <= last_decisive
    ]


def _fit_playable_window(
    start_sec: float,
    end_sec: float,
    *,
    floor: float,
    ceiling: float,
    core_times: list[float],
    max_span: float = MAX_HIGHLIGHT_SEC,
) -> tuple[float, float] | None:
    """Keep the causal core. Fail if that core itself cannot fit in max_span."""
    start_sec = max(floor, min(start_sec, ceiling))
    end_sec = max(start_sec, min(end_sec, ceiling))
    if not core_times:
        if end_sec - start_sec > max_span:
            return None
        return start_sec, end_sec
    core_start = max(floor, min(min(core_times), ceiling))
    core_end = max(core_start, min(max(core_times), ceiling))
    if core_end - core_start > max_span:
        return None
    start_sec = max(floor, min(start_sec, core_start))
    end_sec = min(ceiling, max(end_sec, core_end))
    if end_sec - start_sec <= max_span:
        return start_sec, end_sec
    extra = max_span - (core_end - core_start)
    lead = min(extra, max(0.0, core_start - start_sec))
    start_sec = core_start - lead
    extra -= lead
    end_sec = min(ceiling, core_end + extra)
    return max(floor, start_sec), end_sec


class EvidenceLedgerStore:
    """Compact, explicitly unverified observations from preceding Scene Maps."""

    def __init__(self) -> None:
        self.ledger = EvidenceLedger()

    def snapshot(self) -> EvidenceLedger:
        return self.ledger.model_copy(deep=True)

    def update(self, scene: SceneCard) -> None:
        if not scene.evidence:
            return
        stamp = f"[{scene.start_sec:.2f}-{scene.end_sec:.2f}s]"
        source = "；".join(scene.evidence[:2])
        citation = f"（证据：{source}）"

        self.ledger.characters = _unique(self.ledger.characters + scene.actors)[-30:]

        observation = scene.new_evidence or scene.action
        if observation:
            self.ledger.observations = _unique(
                self.ledger.observations + [f"{stamp} {observation}{citation}"]
            )[-30:]
        for claim in scene.claims[:4]:
            self.ledger.observations = _unique(
                self.ledger.observations + [f"{stamp} 声称：{claim}{citation}"]
            )[-30:]

        if scene.relationship_change:
            self.ledger.relationships = _unique(
                self.ledger.relationships + [f"{stamp} {scene.relationship_change}{citation}"]
            )[-20:]

        if "cliffhanger" in scene.event_type:
            question = scene.action or scene.new_evidence
            if question:
                self.ledger.open_threads = _unique(
                    self.ledger.open_threads + [f"{stamp} {question}{citation}"]
                )[-8:]

        summary = scene.action or scene.new_evidence
        if summary:
            self.ledger.recent_summaries = _unique(
                self.ledger.recent_summaries + [f"{stamp} {summary}"]
            )[-5:]


def build_evidence_ledgers(
    scenes: list[SceneCard],
) -> tuple[dict[str, EvidenceLedger], EvidenceLedger]:
    store = EvidenceLedgerStore()
    before: dict[str, EvidenceLedger] = {}
    for scene in sorted(scenes, key=lambda item: item.start_sec):
        before[scene.scene_id] = store.snapshot()
        store.update(scene)
    return before, store.snapshot()


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))

import { analysisProgress, coveragePercent, isReviewReady, workflowSnapshot } from './workflow'
import type { TaskSpecification, WorkflowCapabilities, WorkflowJob } from './workflow'

export const WORKFLOW_STAGES = [
  { title: '原片接入', subtitle: 'SOURCE INTAKE', description: '接收原片、可选字幕与提取要求。完整接入后生成任务回执，原片与字幕保留同一时间基准。', output: '完整原片 · 提取要求' },
  { title: '任务排队', subtitle: 'TASK QUEUE', description: '已接入的任务交给服务端调度。等待期间可以继续导入或查看其他任务，当前任务会保留在归档中。', output: '执行回执 · 原片证据' },
  { title: '视频分析', subtitle: 'VIDEO ANALYSIS', description: '连续检查全片，按需检索语音与画面文字，围绕事件核验候选，再制作并独立观看拟采用的片段。', output: '核验事件 · 复核成片' },
  { title: '成片交付', subtitle: 'RESULT DELIVERY', description: '完成事件处置与成片复核后才发布最终高光。全片检查达到 100% 时，仍可能需要继续核验和取舍。', output: '高光结果 · 可播放成片' },
  { title: '人工复核', subtitle: 'HUMAN REVIEW', description: '结合高光理由与原片上下文进行人工复核，调整边界并标记采用；采用的片段可进入广告编排与导出。', output: '编辑采用 · 广告导出' },
] as const

type DetailJob = WorkflowJob & {
  original_name: string
  size_bytes: number
  language: 'zh' | 'en'
  attempt: number
  revision: number
  result: (WorkflowJob['result'] & {
    video: { duration_sec: number }
    highlights: { review_status: string; clip_url: string }[]
  }) | null
}

type DetailUpload = {
  phase: string
  progress: number
  fileName?: string
  task?: TaskSpecification
  language?: 'zh' | 'en'
}

export type DetailNode = {
  title: string
  description: string
  input: string
  output: string
  value: string
  state: 'done' | 'current' | 'waiting' | 'unknown' | 'available' | 'failed'
  label: string
}

export type DetailGroup = { id: string; title: string; nodes: DetailNode[] }

function amount(value: number | null | undefined, unit = '') {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? `${value}${unit}` : '尚未回传'
}

function time(value: number) {
  const seconds = Math.floor(value)
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`
}

function node(title: string, description: string, input: string, output: string, value: string, state: DetailNode['state'] = 'unknown', label = '等待回传'): DetailNode {
  return { title, description, input, output, value, state, label }
}

export function workflowDetailGroups(stage: number, job: DetailJob | null, upload: DetailUpload): DetailGroup[] {
  const snapshot = workflowSnapshot(job)
  const analysis = analysisProgress(job)
  const coverage = coveragePercent(analysis)
  const task = job?.task || upload.task
  const complete = isReviewReady(job)
  const intake = upload.phase !== 'idle'
  const count = job?.result?.highlights.length
  const waiting: DetailNode['state'] = job ? 'unknown' : 'waiting'
  const capability = (key: keyof WorkflowCapabilities, title: string, description: string, input: string, output: string) => {
    const enabled = snapshot?.capabilities?.[key]
    return node(title, description, input, output,
      enabled === true ? '可按问题调用，执行次数未单独回传' : enabled === false ? '本任务未启用此辅助能力' : '等待服务端回传能力状态',
      enabled === true ? 'available' : waiting,
      enabled === true ? '可调用' : enabled === false ? '未启用' : '状态未回传')
  }
  const pending = (title: string, description: string, input: string, output: string, value: number | undefined) => node(
    title, description, input, output, amount(value, ' 项'),
    value === undefined ? waiting : value > 0 && job?.status === 'processing' ? 'current' : 'unknown',
    value === undefined ? '等待回传' : value > 0 ? '待处理' : '当前无待处理',
  )
  const clipLimits = task ? `${task.min_clip_sec}–${task.max_clip_sec} 秒 / 段` : '本任务要求尚未回传'
  const budget = task ? `${task.max_highlights === null ? '数量不限' : `最多 ${task.max_highlights} 段`} · ${task.total_duration_sec === null ? '总时长不限' : `总时长 ≤ ${task.total_duration_sec} 秒`}` : '输出预算尚未回传'
  const transcriptValue = snapshot?.transcript_rows_read == null ? '文字读取量尚未回传' : `已读取 ${snapshot.transcript_rows_read} 行文字`

  if (stage === 0) return [{ id: 'intake', title: '原片与任务要求', nodes: [
    node('文件传输', '原片与可选字幕随同一个任务发送。切换页面时上传会继续。', '本机视频 · 可选字幕', '完整上传请求', job?.original_name || upload.fileName || '等待选择原片', job ? 'done' : upload.phase === 'failed' || upload.phase === 'cancelled' ? 'failed' : upload.phase === 'waiting' ? 'waiting' : intake ? 'current' : 'waiting', job ? '已接收' : upload.phase === 'confirming' ? '传输完成' : upload.phase === 'failed' || upload.phase === 'cancelled' ? '中断' : upload.phase === 'waiting' ? '等待上传' : `上传 ${upload.progress}%`),
    node('接入确认', '等待服务端完整写入原片并返回任务；传输结束不等于接入回执已返回。', '完整视频字节', '任务编号 · 原片访问地址', job ? '服务端已创建任务，归档中可继续查看' : '等待服务端接入回执', job ? 'done' : upload.phase === 'confirming' ? 'current' : 'waiting', job ? '已确认' : '等待确认'),
    node('配套字幕', '提供的字幕用于定位台词；未提供时可按需调用 ASR，事件仍须结合原片确认。', '字幕文件 / 原片语音', '带时间信息的文字线索', task ? task.subtitle_provided ? '已提供外部字幕' : '未提供外部字幕 · ASR 按需调用' : '字幕信息尚未回传', task ? 'available' : waiting, task ? '要求已记录' : '等待回传'),
    node('提取目标', '导入时保存当前提取要求，后续调整新任务要求不会覆盖本任务。', '观看目标 · 输出约束', '本任务提取要求', task?.instruction || '本任务目标尚未回传', task ? 'available' : waiting, task ? '要求已记录' : '等待回传'),
  ] }]
  if (stage === 1) return [{ id: 'queue', title: '任务调度', nodes: [
    node('任务登记', '任务记录持久化后进入服务端队列，上传记录按同一编号衔接。', '已接入原片 · 任务要求', '可查询任务记录', job ? '任务已保存到服务端' : '原片接入后登记', job ? 'done' : 'waiting', job ? '已登记' : '等待接入'),
    node('执行调度', '服务端分配分析资源；等待期间可打开其他任务，排队不计入全片检查覆盖率。', '排队任务', '视频分析执行', job?.status === 'queued' ? '等待服务端分析资源' : job ? '任务已开始执行' : '等待原片接入', job?.status === 'queued' ? 'current' : job ? 'done' : 'waiting', job?.status === 'queued' ? '排队中' : job ? '已调度' : '等待'),
    node('执行回执', '首次分析与继续分析均由服务端记录执行次数。', '任务编号', '执行次数 · 状态回执', job ? `${job.attempt} 次执行 · ${job.language === 'zh' ? '中文' : '英文'}视频` : '执行信息尚未回传', job ? 'available' : 'waiting', job ? '服务端回执' : '等待回传'),
  ] }]
  if (stage === 2) return [
    { id: 'scan', title: '全片检查', nodes: [
      node('原片分段', '按原片时间线组织带上下文的观看区间；片段过大时可继续拆分。', '完整原片', '时间对齐的观看区间', `${amount(snapshot?.page_count, ' 个区间')} · ${snapshot?.duration_sec ? `原片 ${time(snapshot.duration_sec)}` : '原片时长待回传'}`, snapshot?.page_count !== undefined && snapshot.page_count !== null ? 'available' : waiting, snapshot?.page_count != null ? '已回传' : '等待回传'),
      node('连续观看', '逐段检查全片内容，覆盖率按已检查的原片范围计算。', '原片区间 · 上下文', '已检查范围', `${coverage === null ? '覆盖率待回传' : `全片检查 ${coverage}%`} · ${amount(snapshot?.completed_page_count)} / ${amount(snapshot?.page_count)} 区间`, coverage === 100 ? 'done' : job?.status === 'processing' ? 'current' : waiting, coverage === 100 ? '全片已检查' : '检查进度'),
      pending('观察整理', '观看后登记可见事件与线索，消化观察信息后再继续证据动作。', '已观看内容', '可核验的视频观察', analysis?.pending_observation_count),
      node('上下文核对', '持续整理人物关系、事件进展与已检查内容，为后续补查和取舍提供上下文。', '观察 · 已确认线索', '连贯的剧情上下文', `${amount(analysis?.model_calls, ' 次模型调用')} · ${amount(snapshot?.turn, ' 轮分析')}`, job?.status === 'processing' ? 'current' : waiting, '分析回执'),
    ] },
    { id: 'perception', title: '辅助感知', nodes: [
      { ...capability('transcript', 'ASR / 字幕检索', '语音转写或外部字幕用于定位台词；文字线索须结合对应视频确认，不是每次分析都执行全片 ASR。', '原片语音 / 配套字幕', '带时间的台词线索'), ...(task?.subtitle_provided ? { value: `已提供字幕 · ${transcriptValue}`, label: '配套字幕' } : snapshot?.capabilities?.transcript === true && (snapshot.transcript_rows_read || 0) > 0 ? { value: `${transcriptValue} · ASR 按需检索` } : {}) },
      capability('read_text', 'OCR 画面文字', '按需读取指定原帧或画面区域中的文字，保留原片时间位置。', '指定画面 · 文字区域', '字幕 / 标牌等画面文字'),
      capability('read_frames', '原帧检查', '在细节难以确认时查看原帧，结合视频上下文核对人物、物件和动作。', '原片时间点', '可检查的画面细节'),
      capability('semantic_search', '语义检索', '用场景或台词描述定位相关内容，再回到原片核验。', '场景描述 · 台词问题', '相关原片区间'),
      capability('local_proposals', '本地候选定位', '可选的本地预测提供粗定位线索，最终取舍仍基于原片与成片复核。', '视频 · 语言', '待确认的候选位置'),
    ] },
    { id: 'events', title: '事件核验', nodes: [
      node('事件登记', '围绕一个可独立采用的看点建立事件，保留因果关系与证据范围。', '视频观察 · 台词线索', '可追踪的事件', amount(snapshot?.event_count, ' 个事件'), snapshot?.event_count != null ? 'available' : waiting, '事件回执'),
      node('候选整理', '整理可采用看点，必要时补看上下文、修订事件或合并重复描述。', '事件 · 补查证据', '完整候选池', amount(snapshot?.candidate_count, ' 条候选'), snapshot?.candidate_count != null ? 'available' : waiting, '候选回执'),
      pending('事件核验', '核对事件是否发生、是否有足够铺垫和反应；未确认的事件不能直接交付。', '事件与原片证据', '确认 / 修订 / 舍弃', analysis?.pending_event_count),
      pending('线索确认', '本地候选和检索结果只提供线索，观看对应内容后才能决定保留。', '待确认线索', '已核验的候选依据', analysis?.pending_proposal_count),
    ] },
    { id: 'clips', title: '成片复核', nodes: [
      node('片段边界', '根据事件证据保留必要铺垫与反应，检查时长、输出预算和上下文重叠要求。', '确认事件 · 输出约束', '拟采用片段边界', `${clipLimits} · ${budget}`, task ? 'available' : waiting, task ? '本任务约束' : '等待回传'),
      node('成片制作', '按初选结果制作实际视频，只有复核后就绪的片段才计入就绪数量。', '拟采用片段', '可独立观看的成片', amount(complete ? count : snapshot?.ready_clip_count, ' 段已就绪'), complete ? 'done' : waiting, complete ? '已交付' : '就绪回执'),
      pending('独立观看复核', '观看实际剪出的片段，检查事件是否可见、上下文是否充分和边界是否合适。', '实际成片', '复核意见 · 修订建议', analysis?.pending_review_count),
      node('最终取舍', '从完整候选池决定保留、替换或舍弃，并确认所选成片均通过复核。', '候选池 · 成片复核', '最终高光选择', complete ? `${count} 段高光已交付` : job?.result?.completion === 'partial' ? '分析尚未完成，可继续分析' : '等待完整核验与交付', complete ? 'done' : waiting, complete ? '取舍完成' : '尚未交付'),
    ] },
  ]
  if (stage === 3) return [{ id: 'delivery', title: '交付检查', nodes: [
    pending('事件处置', '处理待核验事件和观察，确认最终结果没有遗留的必要核验项。', '待核验事件 · 视频观察', '完成事件处置', analysis?.pending_event_count),
    node('成片就绪', '实际视频完成独立复核后，才可作为最终高光公开。', '已复核成片', '最终高光视频', complete ? `${count} 段最终高光` : `${amount(snapshot?.ready_clip_count, ' 段就绪')} · 尚未交付`, complete ? 'done' : waiting, complete ? '已交付' : '等待交付'),
    node('结果校验', '公开结果包含高光边界、类型、理由和成片地址；未完成结果不发布候选高光。', '高光选择 · 复核结果', '完整 / 未完成结果', job?.result ? job.result.completion === 'complete' ? '完整结果已保存' : '未完成结果 · 保留分析进度' : '等待生成结果', complete ? 'done' : waiting, complete ? '已确认' : '等待完整结果'),
    node('媒体交付', '通过任务的公开媒体地址播放原片和已复核成片。', '原片 · 高光成片', '播放器可访问的视频', complete ? `${job?.result?.highlights.filter(item => item.clip_url).length} 段带成片地址` : '等待成片交付', complete ? 'done' : 'waiting', complete ? '媒体已交付' : '等待'),
  ] }]
  return [{ id: 'review', title: '人工编辑', nodes: [
    node('成片预览', '查看已复核高光和入选理由，需要更多上下文时返回原片。', '高光视频 · 入选理由', '人工判断', complete ? `${count} 段高光可查看` : '完整交付后进入复核', complete ? 'available' : 'waiting', complete ? '可预览' : '等待交付'),
    node('边界调整', '拖动时间轴或输入秒数调整范围，保存后按新边界预览原片。', '原片 · 高光范围', '人工修订的边界', job ? `结果版本 ${job.revision}` : '等待生成结果', complete ? 'available' : 'waiting', complete ? '可编辑' : '等待'),
    node('采用与驳回', '人工选择采用、驳回或修订；采用后的片段进入广告编排。', '高光与原片上下文', '人工复核标记', complete ? `${job?.result?.highlights.filter(item => item.review_status === 'accepted').length} 段已采用` : '等待人工复核', complete ? 'available' : 'waiting', complete ? '可复核' : '等待'),
    node('广告与导出', '为已采用高光添加片尾广告，也可直接导出当前高光。', '已采用高光 · 广告素材', '可保存的视频', complete ? '在广告编排中配置与导出' : '采用高光后开放', complete ? 'available' : 'waiting', complete ? '编辑工作区' : '等待'),
  ] }]
}

export function workflowReceipts(job: DetailJob | null) {
  const snapshot = workflowSnapshot(job)
  const progress = analysisProgress(job)
  const next = snapshot?.next_scan_range
  return [
    ['检查区间', `${amount(snapshot?.completed_page_count)} / ${amount(snapshot?.page_count)}`],
    ['接下来检查', next ? `${time(next.start_sec)}–${time(next.end_sec)} 原片区间` : coveragePercent(progress) === 100 ? '全片已检查，继续核验与取舍' : '检查区间尚未回传'],
    ['事件与候选', `${amount(snapshot?.event_count, ' 个事件')} · ${amount(snapshot?.candidate_count, ' 条候选')}`],
    ['已读取文字', amount(snapshot?.transcript_rows_read, ' 行')],
    ['成片就绪', amount(isReviewReady(job) ? job?.result?.highlights.length : snapshot?.ready_clip_count, ' 段')],
    ['模型调用', `${amount(progress?.model_calls, ' 次')} · ${amount(snapshot?.turn, ' 轮分析')}`],
  ]
}

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { DragEvent, ReactNode, SVGProps } from 'react'

const API_BASE = (import.meta.env.VITE_API_BASE_URL?.trim() || 'http://122.193.22.119:8777').replace(/\/+$/, '')
const API_ADDRESS = API_BASE.replace(/^https?:\/\//, '')
const VIDEO_EXTENSIONS = ['.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v']
const MESSAGE_CHUNK_SIZE = 4
const MESSAGE_CHUNK_DELAY_MS = 20

type View = 'workspace' | 'library' | 'ads'
type Theme = 'dark' | 'light'
type JobStatus = 'queued' | 'processing' | 'completed' | 'failed'
type ReviewStatus = 'pending' | 'accepted' | 'rejected' | 'revised'
type AdPresetId = 'cinema-cliffhanger' | 'neon-app' | 'velvet-qr'

type Highlight = {
  highlight_id: string
  start_sec: number
  end_sec: number
  score: number
  highlight_type: string
  description: string
  reason: string
  review_status: ReviewStatus
}

type DetectionResult = {
  schema_version: '1.0'
  job_id: string
  video: { video_id: string; title: string; duration_sec: number }
  highlights: Highlight[]
}

type Job = {
  job_id: string
  status: JobStatus
  original_name: string
  content_type: string
  size_bytes: number
  language: 'zh' | 'en'
  created_at: string
  updated_at: string
  revision: number
  source_url: string | null
  error_message: string | null
  result: DetectionResult | null
  messages: ChatMessage[]
}

type ChatMessage = {
  message_id: string
  role: 'user' | 'assistant'
  content: string
  created_at: string
}

type AdAsset = {
  asset_id: string
  kind: 'video' | 'image'
  original_name: string
  size_bytes: number
  duration_sec: number | null
  created_at: string
  url: string
}

type AdAssignment = {
  highlight_id: string
  kind: 'preset' | 'video'
  preset_id?: AdPresetId
  asset_id?: string
  title: string
  subtitle: string
  duration_sec: number
  qr_asset_id?: string
}

type AdPresetDefinition = {
  id: AdPresetId
  eyebrow: string
  name: string
  title: string
  subtitle: string
  duration: number
  tone: string
}

type AdExportResult = {
  canceled: boolean
  output_path?: string
  duration_sec?: number
}

const AD_PRESETS: AdPresetDefinition[] = [
  {
    id: 'cinema-cliffhanger',
    eyebrow: 'CLIFFHANGER / 01',
    name: '悬念追更',
    title: '点击下方看全集',
    subtitle: '真相就在下一集 · 立即解锁',
    duration: 4,
    tone: 'cinema',
  },
  {
    id: 'neon-app',
    eyebrow: 'APP INSTALL / 02',
    name: '霓虹下载',
    title: '下载短剧 APP',
    subtitle: '海量热剧免费看 · 新人专享',
    duration: 4,
    tone: 'neon',
  },
  {
    id: 'velvet-qr',
    eyebrow: 'SCAN / 03',
    name: '丝绒扫码',
    title: '扫码继续追剧',
    subtitle: '长按识别 · 全集立即看',
    duration: 5,
    tone: 'velvet',
  },
]

function defaultAdAssignment(highlightId: string, preset = AD_PRESETS[0]): AdAssignment {
  return {
    highlight_id: highlightId,
    kind: 'preset',
    preset_id: preset.id,
    title: preset.title,
    subtitle: preset.subtitle,
    duration_sec: preset.duration,
  }
}

function adAssignmentKey(jobId: string, highlightId: string) {
  return `${jobId}:${highlightId}`
}

type RemoteJob = Omit<Job, 'source_url' | 'messages'> & { source_url?: string | null }

function resolveSourceUrl(sourceUrl: string | null | undefined) {
  return sourceUrl ? new URL(sourceUrl, `${API_BASE}/`).toString() : null
}

function withSourceUrl(job: RemoteJob): Job {
  return { ...job, source_url: resolveSourceUrl(job.source_url), messages: [] }
}

type EditMessageStreamResult = {
  job: RemoteJob
  reply: string
  changed: boolean
}

type EditMessageStreamEvent =
  | { type: 'start' }
  | { type: 'delta'; delta: string }
  | { type: 'complete'; job: RemoteJob; changed: boolean }
  | { type: 'error'; status: number; detail: string }

type UploadState = {
  phase: 'idle' | 'uploading'
  progress: number
  fileName?: string
}

type WorkflowNode = {
  title: string
  detail: string
  topic: string
}

type WorkflowStage = {
  code: string
  title: string
  subtitle: string
  description: string
  nodes: WorkflowNode[]
}

type WorkflowState = 'done' | 'current' | 'unknown' | 'waiting' | 'failed'

type WorkflowNodeExecution = {
  state: WorkflowState
  label: string
  detail: string
}

const WORKFLOW_STAGES: WorkflowStage[] = [
  {
    code: '01',
    title: '源文件接入',
    subtitle: 'INGEST / GATEWAY',
    description: '接收桌面端上传的原始视频，在进入分析队列前完成传输进度、媒体格式和任务隔离存储检查。',
    nodes: [
      { title: '分片上传', detail: 'multipart 字节流 · 客户端进度回调', topic: 'upload.chunk.received' },
      { title: '格式校验', detail: '扩展名 · MIME · 20 GB 上限', topic: 'media.guard.passed' },
      { title: '原片落盘', detail: '任务隔离目录 · source/original', topic: 'source.committed' },
    ],
  },
  {
    code: '02',
    title: '任务编排',
    subtitle: 'FASTAPI / QUEUE',
    description: '由后端创建稳定任务编号、持久化公开状态并投递分析请求，Worker 获得执行槽后开始运行 Agent。',
    nodes: [
      { title: '任务注册', detail: 'job_id · language · revision=0', topic: 'job.created' },
      { title: '状态持久化', detail: 'SQLite · queued · 时间戳', topic: 'job.state.changed' },
      { title: 'Worker 认领', detail: '单 Worker 执行队列 · 超时保护', topic: 'analysis.requested' },
    ],
  },
  {
    code: '03',
    title: '媒体预处理',
    subtitle: 'FFMPEG / CACHE',
    description: '读取原片基础信息并建立内容缓存，同时生成后续视觉、语音和声音分析需要的帧序列、音轨与能量曲线。',
    nodes: [
      { title: '媒体探测', detail: '时长 · 编解码 · 音视频轨 · FPS', topic: 'media.probed' },
      { title: '内容指纹', detail: '视频 fingerprint · 缓存命中检查', topic: 'cache.resolved' },
      { title: '帧提取', detail: '1 FPS · 宽 480 px · 时间戳采样', topic: 'frames.extracted' },
      { title: '音频提取', detail: 'PCM WAV · 每秒能量曲线', topic: 'audio.extracted' },
    ],
  },
  {
    code: '04',
    title: '并行感知',
    subtitle: '4-WAY FAN-OUT',
    description: '场景、语音、字幕和声音事件四条支路并行执行，完成后统一回收到同一条带来源标记的多模态时间轴。',
    nodes: [
      { title: '镜头检测', detail: 'shot boundary · 场景时间段', topic: 'scenes.detected' },
      { title: 'ASR 转写', detail: 'Faster-Whisper · VAD · beam=5', topic: 'asr.transcribed' },
      { title: '字幕 OCR', detail: 'PP-OCRv6 · batch=8 · 置信度≥0.55', topic: 'ocr.recognized' },
      { title: '声音事件', detail: 'SenseVoice · 情绪 · BGM/笑/哭/掌声', topic: 'audio.events.detected' },
      { title: '时间轴对齐', detail: 'ASR + OCR 去重合并 · 来源标记', topic: 'transcript.aligned' },
    ],
  },
  {
    code: '05',
    title: '语义融合',
    subtitle: 'EMBEDDING / SALIENCY',
    description: '将画面和邻近文本编码为多模态向量，融合视觉、音频、台词与场景变化，形成高光候选窗口。',
    nodes: [
      { title: '多模态 Embedding', detail: 'Qwen3-VL · 画面 + 邻近文本', topic: 'embedding.indexed' },
      { title: '语义跃迁评分', detail: '相邻向量距离 · 帧级变化', topic: 'transition.scored' },
      { title: '显著性曲线', detail: '画面 · 音量 · 台词 · 场景 · 事件融合', topic: 'saliency.composed' },
      { title: '候选窗口', detail: '20 s 窗口 · 4 s 步长 · 分段覆盖', topic: 'candidates.generated' },
      { title: '局部去重', detail: '候选 NMS · 最多 72 个窗口', topic: 'candidates.deduplicated' },
    ],
  },
  {
    code: '06',
    title: '事件推理',
    subtitle: 'MAP / JUDGE / RANK',
    description: '把候选窗口组织成可推理的场景卡，通过证据账本、并行裁决和全局排序筛选出最终高光。',
    nodes: [
      { title: '语义场景卡', detail: '人物 · 动作 · 台词 · 声音证据', topic: 'scene.cards.built' },
      { title: 'Scene Map', detail: 'VLM 并行叙事映射 · cache', topic: 'scene.map.completed' },
      { title: 'Evidence Ledger', detail: '人物关系 · 新证据 · 未决线索', topic: 'evidence.ledger.updated' },
      { title: 'Judge 共识', detail: '并行裁决 · 类型 · 分数 · 置信度', topic: 'judge.consensus.reached' },
      { title: '边界精修', detail: 'setup / decisive evidence · 因果核心', topic: 'boundaries.refined' },
      { title: '合并与排序', detail: '重叠合并 · 全局排名 · final NMS', topic: 'highlights.ranked' },
    ],
  },
  {
    code: '07',
    title: '结果交付',
    subtitle: 'PERSIST / REVIEW',
    description: '校验高光结果契约，只持久化可公开字段，并将视频、高光列表和编辑能力交给人工复核工作台。',
    nodes: [
      { title: '契约校验', detail: 'DetectionResult schema 1.0', topic: 'result.validated' },
      { title: '结果持久化', detail: '公开字段 · revision · source_url', topic: 'result.persisted' },
      { title: '人工复核就绪', detail: '播放器 · 高光列表 · 编辑助手', topic: 'review.ready' },
    ],
  },
]

const WORKFLOW_EVENT_ROUTES = [
  ['media.probe.requested', 'ffprobe → duration / streams / codec / has_audio'],
  ['media.decode.fanout', 'frames.extract + audio.extract + energy.scan'],
  ['perception.fanout', 'scene.detect | asr | ocr | audio.event'],
  ['asr.transcribe.requested', 'language=zh · vad_filter=true · beam_size=5'],
  ['ocr.batch.requested', 'PP-OCRv6 · subtitle region · batch_size=8'],
  ['audio.event.requested', 'SenseVoice · VAD merge · emotion + event'],
  ['embedding.requested', 'frame pixels + timestamp-near transcript'],
  ['saliency.compose.requested', 'visual + audio + transcript + scene + event'],
  ['candidate.window.requested', 'window=20s · stride=4s · coverage segmentation'],
  ['scene.map.requested', 'semantic scene cards → parallel VLM mapping'],
  ['judge.consensus.requested', 'evidence ledger + previous scene + candidate'],
  ['highlight.boundary.requested', 'setup evidence + decisive evidence + saliency'],
  ['highlight.rank.requested', 'merge → budget → listwise rank → final NMS'],
  ['result.persist.requested', 'schema 1.0 → public result → review queue'],
] as const

type IconName =
  | 'spark' | 'upload' | 'library' | 'film' | 'check' | 'close' | 'minus'
  | 'square' | 'source' | 'retry' | 'folder' | 'trash' | 'x' | 'sun' | 'moon'
  | 'ad' | 'download' | 'qr' | 'plus' | 'play'

function Icon({ name, size = 18, ...props }: { name: IconName; size?: number } & SVGProps<SVGSVGElement>) {
  const paths: Record<IconName, ReactNode> = {
    spark: <path d="m12 2-1.3 5A5.1 5.1 0 0 1 7 10.7L2 12l5 1.3a5.1 5.1 0 0 1 3.7 3.7l1.3 5 1.3-5a5.1 5.1 0 0 1 3.7-3.7l5-1.3-5-1.3A5.1 5.1 0 0 1 13.3 7L12 2Z" />,
    upload: <><path d="M12 16V4m-5 5 5-5 5 5" /><path d="M5 20h14" /></>,
    library: <><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M3 9h18M8 4v5" /></>,
    film: <><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M7 4v16m10-16v16M3 9h4m10 0h4M3 15h4m10 0h4" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    close: <path d="m7 7 10 10m0-10L7 17" />,
    minus: <path d="M5 12h14" />,
    square: <rect x="7" y="7" width="10" height="10" rx="1" />,
    source: <><circle cx="12" cy="12" r="8" /><path d="m10 8 6 4-6 4V8Z" /></>,
    retry: <><path d="M20 7v5h-5" /><path d="M19 12a7 7 0 1 1-2-5" /></>,
    folder: <path d="M3 7.5A2.5 2.5 0 0 1 5.5 5H10l2 2h6.5A2.5 2.5 0 0 1 21 9.5v7a2.5 2.5 0 0 1-2.5 2.5h-13A2.5 2.5 0 0 1 3 16.5v-9Z" />,
    trash: <><path d="M4 7h16M9 7V4h6v3m3 0-1 13H7L6 7" /><path d="M10 11v5m4-5v5" /></>,
    x: <path d="m8 8 8 8m0-8-8 8" />,
    sun: <><circle cx="12" cy="12" r="3.5" /><path d="M12 2v2m0 16v2M4.93 4.93l1.42 1.42m11.3 11.3 1.42 1.42M2 12h2m16 0h2M4.93 19.07l1.42-1.42m11.3-11.3 1.42-1.42" /></>,
    moon: <path d="M20 15.1A8 8 0 0 1 8.9 4a8 8 0 1 0 11.1 11.1Z" />,
    ad: <><rect x="3" y="5" width="18" height="14" rx="2" /><path d="M7 15V9h2.3a2 2 0 0 1 0 4H7m0 0h3.2M14 15V9h1.6a2.4 2.4 0 0 1 0 6H14Z" /></>,
    download: <><path d="M12 3v12m-4-4 4 4 4-4" /><path d="M5 20h14" /></>,
    qr: <><rect x="3" y="3" width="7" height="7" rx="1" /><rect x="14" y="3" width="7" height="7" rx="1" /><rect x="3" y="14" width="7" height="7" rx="1" /><path d="M14 14h3v3h-3zm4 4h3v3h-3zm0-4h3m-7 7h3" /></>,
    plus: <path d="M12 5v14M5 12h14" />,
    play: <><circle cx="12" cy="12" r="9" /><path d="m10 8 6 4-6 4V8Z" /></>,
  }
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}>{paths[name]}</svg>
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, init)
  } catch {
    throw new Error(`无法连接后端服务（${API_ADDRESS}），请检查 API 地址、网络和后端状态`)
  }
  if (!response.ok) {
    const payload = await response.json().catch(() => null) as { detail?: string } | null
    throw new Error(payload?.detail || `请求失败（${response.status}）`)
  }
  return response.json() as Promise<T>
}

async function requestEmpty(path: string, init?: RequestInit): Promise<void> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, init)
  } catch {
    throw new Error(`无法连接后端服务（${API_ADDRESS}），请检查 API 地址、网络和后端状态`)
  }
  if (!response.ok) {
    const payload = await response.json().catch(() => null) as { detail?: string } | null
    throw new Error(payload?.detail || `请求失败（${response.status}）`)
  }
}

async function requestMessageStream(
  path: string,
  payload: Record<string, unknown>,
  onDelta: (reply: string) => void,
): Promise<EditMessageStreamResult> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
  } catch {
    throw new Error(`无法连接后端服务（${API_ADDRESS}），请检查 API 地址、网络和后端状态`)
  }
  if (!response.ok) {
    const error = await response.json().catch(() => null) as { detail?: string } | null
    if (response.status === 404 && error?.detail === 'Not Found' && path.endsWith('/stream')) {
      const fallback = await requestJson<EditMessageStreamResult>(path.slice(0, -'/stream'.length), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      let visibleReply = ''
      for (let index = 0; index < fallback.reply.length; index += MESSAGE_CHUNK_SIZE) {
        visibleReply += fallback.reply.slice(index, index + MESSAGE_CHUNK_SIZE)
        onDelta(visibleReply)
        await new Promise<void>((resolve) => window.setTimeout(resolve, MESSAGE_CHUNK_DELAY_MS))
      }
      return fallback
    }
    throw new Error(error?.detail || `请求失败（${response.status}）`)
  }
  if (!response.body) throw new Error('后端没有返回可读取的流式响应')

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let reply = ''
  let completed: Extract<EditMessageStreamEvent, { type: 'complete' }> | null = null

  const consume = (
    line: string,
  ): Extract<EditMessageStreamEvent, { type: 'complete' }> | null => {
    if (!line.trim()) return null
    let event: EditMessageStreamEvent
    try {
      event = JSON.parse(line) as EditMessageStreamEvent
    } catch {
      throw new Error('后端返回了无效的流式消息')
    }
    if (event.type === 'delta') {
      reply += event.delta
      onDelta(reply)
    } else if (event.type === 'complete') {
      return event
    } else if (event.type === 'error') {
      throw new Error(event.detail)
    }
    return null
  }

  while (true) {
    const { done, value } = await reader.read()
    buffer += decoder.decode(value, { stream: !done })
    const lines = buffer.split('\n')
    buffer = lines.pop() || ''
    for (const line of lines) completed = consume(line) || completed
    if (done) break
  }
  completed = consume(buffer) || completed
  if (!completed) throw new Error('流式响应在完成前意外结束')
  return { job: completed.job, changed: completed.changed, reply }
}

function uploadVideo(
  jobId: string,
  file: File,
  onProgress: (value: number) => void,
  register: (request: XMLHttpRequest | null) => void,
): Promise<RemoteJob> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest()
    const form = new FormData()
    form.append('job_id', jobId)
    form.append('language', 'zh')
    form.append('file', file, file.name)
    register(request)
    request.open('POST', `${API_BASE}/api/jobs`)
    request.responseType = 'json'
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(Math.round((event.loaded / event.total) * 100))
    }
    request.onload = () => {
      register(null)
      if (request.status >= 200 && request.status < 300) resolve(request.response as RemoteJob)
      else reject(new Error(request.response?.detail || `上传失败（${request.status}）`))
    }
    request.onerror = () => { register(null); reject(new Error(`无法连接后端服务（${API_ADDRESS}），请检查 API 地址、网络和后端状态`)) }
    request.onabort = () => { register(null); reject(new Error('上传已取消')) }
    request.send(form)
  })
}

function mergeRemoteJob(local: Job, remote: RemoteJob): Job {
  const currentHighlights = new Map(
    (local.result?.highlights || []).map((item) => [item.highlight_id, item]),
  )
  const result = remote.result ? {
    ...remote.result,
    highlights: remote.result.highlights.map((item) => {
      const current = currentHighlights.get(item.highlight_id)
      const agentChanged = current && (
        current.start_sec !== item.start_sec
        || current.end_sec !== item.end_sec
        || current.description !== item.description
        || current.reason !== item.reason
      )
      return {
        ...item,
        review_status: agentChanged ? item.review_status : current?.review_status || item.review_status,
      }
    }),
  } : null
  return {
    ...local,
    ...remote,
    result,
    source_url: resolveSourceUrl(remote.source_url),
    messages: local.messages,
  }
}

function createJobId() {
  return `job_${crypto.randomUUID().replace(/-/g, '').slice(0, 16)}`
}

function formatBytes(bytes: number) {
  if (!bytes) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1)
  return `${(bytes / 1024 ** index).toFixed(index > 1 ? 1 : 0)} ${units[index]}`
}

function formatTime(seconds: number) {
  const value = Math.max(0, Math.round(seconds))
  return `${String(Math.floor(value / 60)).padStart(2, '0')}:${String(value % 60).padStart(2, '0')}`
}

function formatPreciseTime(seconds: number) {
  const value = Math.max(0, seconds)
  const minutes = Math.floor(value / 60)
  const remainder = (value - minutes * 60).toFixed(2).padStart(5, '0')
  return `${String(minutes).padStart(2, '0')}:${remainder}`
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat('zh-CN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }).format(new Date(value))
}

const statusCopy: Record<JobStatus, string> = {
  queued: '等待分析', processing: '正在提取', completed: '提取完成', failed: '提取失败',
}

const reviewStatusCopy: Record<ReviewStatus, string> = {
  pending: '待复核', accepted: '已采用', rejected: '已排除', revised: '已调整',
}

const highlightTypeCopy: Record<string, string> = {
  action: '关键行动', climax: '剧情高潮', cliffhanger: '悬念', conflict: '冲突',
  emotion: '情绪', reveal: '信息揭示', reversal: '剧情反转',
}

function WindowChrome({ theme, onToggleTheme }: { theme: Theme; onToggleTheme: () => void }) {
  const nextThemeLabel = theme === 'dark' ? '切换到亮色模式' : '切换到暗色模式'
  return <div className="window-chrome">
    <div className="chrome-label"><span /> FRAME / 本地高光工作台</div>
    <div className="window-actions">
      <button className="theme-toggle" aria-label={nextThemeLabel} title={nextThemeLabel} onClick={onToggleTheme}><Icon name={theme === 'dark' ? 'sun' : 'moon'} size={16} /></button>
      <button aria-label="最小化" onClick={() => window.desktopWindow?.minimize()}><Icon name="minus" size={15} /></button>
      <button aria-label="最大化" onClick={() => window.desktopWindow?.toggleMaximize()}><Icon name="square" size={14} /></button>
      <button className="danger" aria-label="关闭" onClick={() => window.desktopWindow?.close()}><Icon name="close" size={15} /></button>
    </div>
  </div>
}

function Sidebar({ view, onView, online, jobs }: { view: View; onView: (view: View) => void; online: boolean; jobs: Job[] }) {
  const completed = jobs.filter((job) => job.status === 'completed').length
  const accepted = jobs.reduce((total, item) => total + (item.result?.highlights.filter((highlight) => highlight.review_status === 'accepted').length || 0), 0)
  return <aside className="sidebar">
    <div className="brand"><div className="brand-glyph"><i /><i /><i /></div><div><strong>FRAME</strong><span>高光工作台</span></div></div>
    <nav>
      <small>工作区</small>
      <button className={view === 'workspace' ? 'active' : ''} onClick={() => onView('workspace')}><Icon name="spark" /><span>高光提取</span><b>01</b></button>
      <button className={view === 'library' ? 'active' : ''} onClick={() => onView('library')}><Icon name="library" /><span>任务归档</span><b>02</b></button>
      <button className={view === 'ads' ? 'active' : ''} onClick={() => onView('ads')}><Icon name="ad" /><span>广告编排</span><b>03</b></button>
    </nav>
    <div className="sidebar-metric"><span>服务端任务</span><strong>{String(jobs.length).padStart(2, '0')}</strong><small>{completed} 个已完成 · {accepted} 段已采用</small></div>
    <div className={`service-state ${online ? 'online' : ''}`}><i /><div><b>{online ? '后端服务在线' : '后端服务离线'}</b><span>API · {API_ADDRESS}</span></div></div>
  </aside>
}

function UploadDropzone({ onFile, compact = false }: { onFile: (file: File) => void; compact?: boolean }) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const receive = (files: FileList | null) => { if (files?.[0]) onFile(files[0]) }
  const drop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragging(false)
    receive(event.dataTransfer.files)
  }
  if (compact) return <>
    <button className="button primary" onClick={() => inputRef.current?.click()}><Icon name="upload" size={17} />导入新视频</button>
    <input ref={inputRef} hidden type="file" accept="video/*,.mkv,.m4v" onChange={(event) => { receive(event.target.files); event.target.value = '' }} />
  </>
  return <div className={`dropzone ${dragging ? 'dragging' : ''}`} onDragOver={(event) => { event.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={drop}>
    <input ref={inputRef} hidden type="file" accept="video/*,.mkv,.m4v" onChange={(event) => { receive(event.target.files); event.target.value = '' }} />
    <div className="drop-icon"><Icon name="upload" size={27} /></div>
    <p className="overline">NEW SOURCE / 01</p>
    <h2>把原片放到这里</h2>
    <p>原片上传到后端创建分析任务；前端只通过公开 API 获取状态、结果和视频流。</p>
    <button className="button primary" onClick={() => inputRef.current?.click()}><Icon name="film" size={17} />选择视频文件</button>
    <span>MP4 · MOV · MKV · WEBM · AVI · 最大 20 GB</span>
  </div>
}

function ProgressPanel({ upload, job, onCancel }: { upload: UploadState; job: Job | null; onCancel: () => void }) {
  const uploading = upload.phase !== 'idle'
  const label = upload.phase === 'uploading' ? `正在上传 ${upload.progress}%` : job ? statusCopy[job.status] : ''
  const number = upload.phase === 'uploading' ? upload.progress : job?.status === 'completed' ? 100 : null
  const internalPipelineActive = job?.status === 'processing'
  const workflowNodeCount = WORKFLOW_STAGES.reduce((total, stage) => total + stage.nodes.length, 0)
  const automaticStageIndex = uploading ? 0 : job?.status === 'queued' ? 1 : job?.status === 'processing' || job?.status === 'failed' ? 2 : job?.status === 'completed' ? 6 : 0
  const [selectedWorkflowStageIndex, setSelectedWorkflowStageIndex] = useState(automaticStageIndex)
  useEffect(() => setSelectedWorkflowStageIndex(automaticStageIndex), [upload.phase, job?.status])
  const stageState = (stageIndex: number): WorkflowState => {
    if (uploading) return stageIndex === 0 ? 'current' : 'waiting'
    if (!job) return 'waiting'
    if (job.status === 'queued') return stageIndex === 0 ? 'done' : stageIndex === 1 ? 'current' : 'waiting'
    if (job.status === 'processing') return stageIndex < 2 ? 'done' : stageIndex === 2 ? 'current' : stageIndex < 6 ? 'unknown' : 'waiting'
    if (job.status === 'completed') return 'done'
    if (job.status === 'failed') return stageIndex < 2 ? 'done' : stageIndex === 2 ? 'failed' : 'waiting'
    return 'waiting'
  }
  const selectedWorkflowStage = WORKFLOW_STAGES[selectedWorkflowStageIndex]
  const selectedWorkflowState = stageState(selectedWorkflowStageIndex)
  const confirmedStageCount = uploading ? 0 : job?.status === 'completed' ? WORKFLOW_STAGES.length : job?.status === 'queued' ? 1 : job ? 2 : 0
  const confirmedPercent = uploading ? upload.progress : Math.round((confirmedStageCount / WORKFLOW_STAGES.length) * 100)
  const publicProgressLabel = uploading
    ? `${upload.progress}%`
    : job?.status === 'completed'
      ? '100%'
      : job?.status === 'processing' || job?.status === 'failed'
        ? `≥ ${confirmedPercent}%`
        : `${confirmedPercent}%`
  const publicProgressDetail = uploading
    ? '客户端上传进度'
    : `${confirmedStageCount} / ${WORKFLOW_STAGES.length} 个阶段已由 API 确认`
  const selectedStageLabel = selectedWorkflowState === 'done'
    ? '已确认完成'
    : selectedWorkflowState === 'failed'
      ? '执行中断'
      : selectedWorkflowState === 'current'
        ? internalPipelineActive ? 'Agent 处理中' : '进行中'
        : selectedWorkflowState === 'unknown'
          ? '状态未回传'
          : '等待调度'
  const nodeExecution = (): WorkflowNodeExecution => {
    if (selectedWorkflowState === 'done') return {
      state: 'done',
      label: 'API 已确认',
      detail: '公开任务状态已确认该阶段完成。',
    }
    if (selectedWorkflowState === 'failed') return {
      state: 'failed',
      label: '执行中断',
      detail: job?.error_message || '后端返回 failed，公开状态无法定位到具体失败节点。',
    }
    if (selectedWorkflowState === 'current' && uploading) return {
      state: 'current',
      label: '正在上传',
      detail: `客户端已发送 ${upload.progress}% 的源文件字节。`,
    }
    if (selectedWorkflowState === 'current' && job?.status === 'queued') return {
      state: 'current',
      label: '等待 Worker',
      detail: '任务已创建，等待后端 Worker 认领。',
    }
    if (selectedWorkflowState === 'current') return {
      state: 'current',
      label: '状态未回传',
      detail: '任务整体正在处理，后端暂未回传此子步骤的独立状态。',
    }
    if (selectedWorkflowState === 'unknown') return {
      state: 'unknown',
      label: '状态未回传',
      detail: '该阶段可能等待、执行中或已完成，当前公开 API 无法进一步区分。',
    }
    return {
      state: 'waiting',
      label: '等待调度',
      detail: '前序阶段完成后进入此步骤。',
    }
  }
  const eventMessages = [
    ...(uploading ? [{ kind: 'observed', topic: 'upload.stream.active', payload: `progress=${upload.progress}% · transport=multipart/form-data` }] : []),
    ...(job ? [
      { kind: 'observed', topic: 'job.created', payload: `job_id=${job.job_id} · language=${job.language} · revision=${job.revision}` },
      { kind: 'observed', topic: 'job.state.changed', payload: `status=${job.status} · updated_at=${formatDate(job.updated_at)}` },
    ] : []),
    ...(internalPipelineActive ? [{ kind: 'observed', topic: 'analysis.worker.active', payload: '后端已确认 processing · Agent 内部节点按拓扑展示' }] : []),
    ...WORKFLOW_EVENT_ROUTES.map(([topic, payload]) => ({ kind: 'route', topic, payload })),
  ]
  return <section className={`progress-panel ${job?.status || upload.phase}`}>
    <div className="progress-heading">
      <div><span className="pulse" /><div><b>{label}</b><small>{uploading ? upload.fileName : job?.original_name}</small></div></div>
      <div className="progress-heading-actions">
        {job && <span>{job.job_id}</span>}
        {upload.phase === 'uploading' && <button onClick={onCancel}><Icon name="x" size={15} />取消上传</button>}
      </div>
    </div>
    <div className={`progress-track ${job?.status === 'processing' || job?.status === 'queued' ? 'indeterminate' : ''}`}><i style={{ width: `${number ?? 38}%` }} /></div>

    <nav className="workflow-flow" aria-label="高光提取流程">
      {WORKFLOW_STAGES.map((stage, index) => {
        const state = stageState(index)
        const selected = index === selectedWorkflowStageIndex
        const stateLabel = state === 'done' ? '已完成' : state === 'current' ? '处理中' : state === 'failed' ? '已中断' : state === 'unknown' ? '状态未回传' : '等待'
        return <button
          type="button"
          className={`workflow-flow-step ${state}${selected ? ' selected' : ''}`}
          key={stage.code}
          aria-pressed={selected}
          aria-controls="workflow-stage-detail"
          onClick={() => setSelectedWorkflowStageIndex(index)}
        >
          <span className="workflow-flow-node">{stage.code}</span>
          <b>{stage.title}</b>
          <small>{stateLabel} · {stage.nodes.length} 个节点</small>
        </button>
      })}
    </nav>

    <div className="workflow-summary" aria-label="分析流程摘要">
      <span><b>{WORKFLOW_STAGES.length}</b> 个阶段</span>
      <span><b>{workflowNodeCount}</b> 个细节节点</span>
      <span><b>4</b> 路并行感知</span>
      <span><b>{WORKFLOW_EVENT_ROUTES.length}</b> 条事件路由</span>
      <em><i /><span><b>{publicProgressLabel}</b><small>{publicProgressDetail}</small></span><strong>PUBLIC STATUS · {job?.status?.toUpperCase() || 'UPLOADING'}</strong></em>
    </div>

    <div className="workflow-layout">
      <div className="workflow-graph" aria-label="高光分析编排拓扑">
        <div className="workflow-section-head workflow-graph-head"><div><span>INTERACTIVE PIPELINE</span><b>点击流程节点查看完整内容</b></div><small>当前查看 · {selectedWorkflowStage.code} {selectedWorkflowStage.title}</small></div>
        <article id="workflow-stage-detail" className={`workflow-stage-detail ${selectedWorkflowState}`} key={selectedWorkflowStage.code}>
          <header>
            <span>{selectedWorkflowStage.code}</span>
            <div><small>{selectedWorkflowStage.subtitle}</small><h2>{selectedWorkflowStage.title}</h2><p>{selectedWorkflowStage.description}</p></div>
            <aside><b>{selectedWorkflowStage.nodes.length}</b><span>执行节点</span><em>{selectedStageLabel}</em></aside>
          </header>
          <div className="workflow-stage-contents">
            {selectedWorkflowStage.nodes.map((node, nodeIndex) => {
              const execution = nodeExecution()
              return <section className={execution.state} key={node.topic}>
                <span>{selectedWorkflowStage.code}.{String(nodeIndex + 1).padStart(2, '0')}</span>
                <div>
                  <div className="workflow-node-heading"><h3>{node.title}</h3><em><i />{execution.label}</em></div>
                  <p>{node.detail}</p>
                </div>
                <small className="workflow-node-note">{execution.detail}</small>
                <code><b>EVENT</b>{node.topic}</code>
              </section>
            })}
          </div>
        </article>
      </div>

      <aside className="event-console" aria-label="事件驱动消息">
        <div className="workflow-section-head"><div><span>EVENT BUS</span><b>编排消息</b></div><small>{eventMessages.length} messages</small></div>
        <div className="event-legend"><span><i className="observed" />API 回执</span><span><i className="route" />拓扑路由</span></div>
        <ol role="log" aria-live="polite">
          {eventMessages.map((event, index) => <li className={`${event.kind}${event.kind === 'route' && internalPipelineActive ? ' active' : ''}`} key={`${event.topic}-${index}`}>
            <span>{String(index + 1).padStart(2, '0')}</span>
            <div><b>{event.topic}</b><small>{event.payload}</small></div>
            <em>{event.kind === 'observed' ? 'API' : internalPipelineActive ? 'ROUTE' : 'WAIT'}</em>
          </li>)}
        </ol>
      </aside>
    </div>
    <p className="workflow-disclosure"><span>STATUS SOURCE</span> 顶部状态来自后端公开任务 API；Agent 内部节点与事件展示的是当前编排拓扑，不读取 trace 或服务端日志，也不伪装成逐节点完成回执。</p>
  </section>
}

function Player({ job, selected, reloadToken, onSelect, onPlaybackError }: { job: Job; selected: Highlight | null; reloadToken: number; onSelect: (item: Highlight | null) => void; onPlaybackError: (message: string | null) => void }) {
  const source = job.source_url
  const highlights = job.result?.highlights || []
  const sourceDuration = job.result?.video.duration_sec || 1
  const selectedIndex = selected ? highlights.findIndex((item) => item.highlight_id === selected.highlight_id) : -1
  const videoRef = useRef<HTMLVideoElement>(null)
  const [inspectedId, setInspectedId] = useState<string | null>(null)
  const inspectedHighlight = highlights.find((item) => item.highlight_id === inspectedId)
    || selected
    || highlights[0]
    || null
  const inspectedIndex = inspectedHighlight
    ? highlights.findIndex((item) => item.highlight_id === inspectedHighlight.highlight_id)
    : -1
  useEffect(() => {
    const player = videoRef.current
    if (!player) return
    let cancelled = false
    const loadSource = async () => {
      if (!source) {
        onPlaybackError('后端没有返回视频播放地址，请确认媒体接口版本已经部署。')
        return
      }
      try {
        const response = await fetch(source, { method: 'HEAD', cache: 'no-store' })
        if (!response.ok) throw new Error(`视频源接口返回 ${response.status}`)
        if (cancelled) return
        player.src = source
        player.load()
      } catch (reason) {
        if (cancelled) return
        const detail = reason instanceof Error ? reason.message : '未知错误'
        onPlaybackError(`无法加载后端视频源：${detail}。请检查后端媒体接口和 CORS 配置。`)
      }
    }
    void loadSource()
    return () => {
      cancelled = true
      player.pause()
      player.removeAttribute('src')
      player.load()
    }
  }, [source, reloadToken, onPlaybackError])
  useEffect(() => {
    const player = videoRef.current
    if (!player) return
    const seekToSelection = () => {
      player.currentTime = selected?.start_sec || 0
      if (selected) void player.play().catch(() => undefined)
    }
    if (player.readyState >= 1) seekToSelection()
    else player.addEventListener('loadedmetadata', seekToSelection, { once: true })
    const stopAtOutPoint = () => {
      if (selected && player.currentTime >= selected.end_sec) {
        player.pause()
        player.currentTime = selected.end_sec
      }
    }
    player.addEventListener('timeupdate', stopAtOutPoint)
    return () => {
      player.removeEventListener('loadedmetadata', seekToSelection)
      player.removeEventListener('timeupdate', stopAtOutPoint)
    }
  }, [selected?.highlight_id, selected?.start_sec, selected?.end_sec])
  return <section className="player-card">
    <div className="player-top"><div><span>VIDEO MONITOR · {selected ? 'HIGHLIGHT' : 'SOURCE'}</span><b>{selected?.description || job.original_name}</b></div><span className="media-chip">{selected ? `${formatTime(selected.start_sec)} — ${formatTime(selected.end_sec)}` : formatBytes(job.size_bytes)}</span></div>
    <div className="video-frame">
      <div className="video-stage"><video ref={videoRef} key={source} controls preload="metadata" onLoadedMetadata={() => onPlaybackError(null)} onError={() => onPlaybackError('视频流已返回，但 Electron 无法解码。请确认视频编码为 Chromium 支持的 H.264/AAC、VP9/Opus 或 AV1。')} /></div>
      <div className="monitor-timeline" aria-label="完整原片高光时间轴">
        <div className="timeline-caption">
          <div><span>完整原片 · 高光轨道</span><small>点击荧光绿区间预览片段</small></div>
          {inspectedHighlight ? <div className="timeline-time-readout" aria-label={`片段 ${inspectedIndex + 1} 的入点和出点`}>
            <small>{`片段 ${String(inspectedIndex + 1).padStart(2, '0')}`}</small>
            <span><i>IN</i><b>{formatTime(inspectedHighlight.start_sec)}</b></span>
            <span><i>OUT</i><b>{formatTime(inspectedHighlight.end_sec)}</b></span>
          </div> : <b>{formatTime(sourceDuration)}</b>}
        </div>
        <div className="timeline-track" role="group" aria-label="可选择的高光片段">
          {highlights.map((item) => {
            const start = Math.max(0, Math.min(100, (item.start_sec / sourceDuration) * 100))
            const end = Math.max(start, Math.min(100, (item.end_sec / sourceDuration) * 100))
            const active = selected?.highlight_id === item.highlight_id
            const inspected = inspectedHighlight?.highlight_id === item.highlight_id
            return <button
              type="button"
              key={item.highlight_id}
              className={`timeline-highlight${active ? ' active' : ''}${inspected ? ' inspected' : ''}`}
              style={{ left: `${start}%`, width: `${end - start}%` }}
              title={`点击预览：${item.description} · 入点 ${formatTime(item.start_sec)} · 出点 ${formatTime(item.end_sec)}`}
              aria-label={`${item.description}，入点 ${formatTime(item.start_sec)}，出点 ${formatTime(item.end_sec)}，点击预览`}
              aria-pressed={active}
              onClick={() => onSelect(item)}
              onMouseEnter={() => setInspectedId(item.highlight_id)}
              onMouseLeave={() => setInspectedId(null)}
              onFocus={() => setInspectedId(item.highlight_id)}
              onBlur={() => setInspectedId(null)}
            />
          })}
        </div>
        <div className="timeline-scale"><span>00:00</span><span>悬停查看出入点 · 点击选择</span><span>{formatTime(sourceDuration)}</span></div>
      </div>
      {selected && <section className="highlight-detail-drawer" key={selected.highlight_id} aria-labelledby="highlight-detail-title">
        <div className="detail-identity">
          <div><span>SELECTED HIGHLIGHT</span><b>{String(selectedIndex + 1).padStart(2, '0')}</b></div>
          <small>{selected.highlight_id}</small>
          <button type="button" onClick={() => onSelect(null)} aria-label="关闭高光详情" title="关闭高光详情"><Icon name="x" size={14} /></button>
        </div>
        <div className="detail-narrative">
          <div className="detail-tags"><span>{selected.highlight_type}</span><b>{highlightTypeCopy[selected.highlight_type.toLowerCase()] || '事件高光'}</b><em className={selected.review_status}>{reviewStatusCopy[selected.review_status]}</em></div>
          <h3 id="highlight-detail-title">{selected.description || '未命名高光片段'}</h3>
          <div className="detail-reason"><span>WHY SELECTED</span><p>{selected.reason || 'Agent 未提供额外的入选理由。'}</p></div>
        </div>
        <div className="detail-telemetry">
          <div className="detail-score"><span>HIGHLIGHT SCORE</span><b>{Math.round(selected.score * 100)}</b><i><span style={{ width: `${Math.round(selected.score * 100)}%` }} /></i></div>
          <dl>
            <div><dt>IN</dt><dd>{formatPreciseTime(selected.start_sec)}</dd></div>
            <div><dt>OUT</dt><dd>{formatPreciseTime(selected.end_sec)}</dd></div>
            <div><dt>DURATION</dt><dd>{(selected.end_sec - selected.start_sec).toFixed(2)} s</dd></div>
          </dl>
        </div>
      </section>}
    </div>
  </section>
}

function HighlightList({ job, selectedId, configuredIds, onSelect, onReview, onAd }: { job: Job; selectedId: string | null; configuredIds: Set<string>; onSelect: (item: Highlight | null) => void; onReview: (item: Highlight, status: ReviewStatus) => void; onAd: (item: Highlight) => void }) {
  const highlights = job.result?.highlights || []
  return <aside className="result-panel">
    <div className="result-head"><div><p className="overline">AGENT OUTPUT</p><h2>高光候选</h2></div><span>{String(highlights.length).padStart(2, '0')}</span></div>
    <button className={!selectedId ? 'source-row active' : 'source-row'} onClick={() => onSelect(null)}><span><Icon name="source" size={16} /></span><div><b>完整原片</b><small>返回源视频预览</small></div></button>
    <div className="highlight-scroll">
      {highlights.map((item, index) => <article key={item.highlight_id} className={selectedId === item.highlight_id ? 'highlight-card active' : 'highlight-card'}>
        <button className="highlight-main" aria-pressed={selectedId === item.highlight_id} onClick={() => onSelect(item)}>
          <span className="clip-index">{String(index + 1).padStart(2, '0')}</span>
          <div><div className="clip-meta"><span>{item.highlight_type}</span><b>{Math.round(item.score * 100)}</b></div><h3>{item.description || '未命名高光'}</h3><p>{item.reason}</p><small>{formatTime(item.start_sec)} — {formatTime(item.end_sec)} · {Math.round(item.end_sec - item.start_sec)} 秒</small></div>
        </button>
        <div className="review-actions"><button className={item.review_status === 'accepted' ? 'accepted' : ''} onClick={() => onReview(item, item.review_status === 'accepted' ? 'pending' : 'accepted')}><Icon name="check" size={13} />{item.review_status === 'accepted' ? '已采用' : '采用'}</button><button className={item.review_status === 'rejected' ? 'rejected' : ''} onClick={() => onReview(item, item.review_status === 'rejected' ? 'pending' : 'rejected')}><Icon name="x" size={13} />{item.review_status === 'rejected' ? '已排除' : '排除'}</button><button className={configuredIds.has(item.highlight_id) ? 'ad-action configured' : 'ad-action'} disabled={item.review_status !== 'accepted'} title={item.review_status === 'accepted' ? '为这段高光添加片尾广告' : '先采用这段高光'} onClick={() => onAd(item)}><Icon name="ad" size={13} />{configuredIds.has(item.highlight_id) ? '已加广告' : '加广告'}</button></div>
      </article>)}
      {!highlights.length && <div className="empty-results"><Icon name="spark" size={22} /><b>没有生成高光片段</b><p>原片仍已安全保存在任务目录中。</p></div>}
    </div>
  </aside>
}

function QrPlaceholder() {
  return <span className="qr-placeholder" aria-label="二维码占位图">{Array.from({ length: 64 }, (_, index) => <i key={index} className={(index * 7 + Math.floor(index / 8) * 11 + index * Math.floor(index / 8)) % 5 < 2 ? 'on' : ''} />)}</span>
}

function AdCreativePreview({ assignment, assets, compact = false }: { assignment: AdAssignment; assets: AdAsset[]; compact?: boolean }) {
  if (assignment.kind === 'video') {
    const asset = assets.find((item) => item.asset_id === assignment.asset_id)
    return <div className={`ad-creative-preview custom-video${compact ? ' compact' : ''}`}>
      {asset ? <video src={asset.url} muted loop controls={!compact} autoPlay={!compact} /> : <div className="ad-missing-asset"><Icon name="film" size={25} /><span>选择一个广告视频</span></div>}
      {asset && <span className="custom-video-label"><i />CUSTOM VIDEO · {asset.duration_sec?.toFixed(1)}s</span>}
    </div>
  }
  const preset = AD_PRESETS.find((item) => item.id === assignment.preset_id) || AD_PRESETS[0]
  const qrAsset = assets.find((item) => item.asset_id === assignment.qr_asset_id)
  return <div className={`ad-creative-preview preset-${preset.tone}${compact ? ' compact' : ''}`}>
    <div className="creative-noise" />
    {preset.id === 'cinema-cliffhanger' && <>
      <span className="creative-ribbon">DRAMA EXCLUSIVE</span>
      <span className="creative-play"><Icon name="play" size={compact ? 17 : 27} /></span>
    </>}
    {preset.id === 'neon-app' && <>
      <span className="creative-gift">NEW USER GIFT</span>
      <span className="creative-phone"><i /><i /><i /></span>
    </>}
    {preset.id === 'velvet-qr' && <>
      <span className="creative-scan">SCAN TO CONTINUE</span>
      <span className="creative-qr">{qrAsset ? <img src={qrAsset.url} alt="已选择的二维码" /> : <QrPlaceholder />}</span>
    </>}
    <div className="creative-copy"><strong>{assignment.title}</strong><span>{assignment.subtitle}</span></div>
    {preset.id !== 'velvet-qr' && <span className="creative-cta">{preset.id === 'neon-app' ? '免费下载短剧 APP' : '立即观看  →'}</span>}
  </div>
}

function AdStudioDialog({ job, initialTarget, assignments, assets, onAssetsChange, onSave, onClose }: { job: Job; initialTarget: Highlight; assignments: Record<string, AdAssignment>; assets: AdAsset[]; onAssetsChange: (assets: AdAsset[]) => void; onSave: (assignment: AdAssignment) => void; onClose: () => void }) {
  const accepted = job.result?.highlights.filter((item) => item.review_status === 'accepted') || []
  const [targetId, setTargetId] = useState(initialTarget.highlight_id)
  const [draft, setDraft] = useState<AdAssignment>(() => assignments[adAssignmentKey(job.job_id, initialTarget.highlight_id)] || defaultAdAssignment(initialTarget.highlight_id))
  const [assetBusy, setAssetBusy] = useState<'video' | 'image' | null>(null)
  const [exporting, setExporting] = useState(false)
  const [studioError, setStudioError] = useState<string | null>(null)
  const [exportResult, setExportResult] = useState<AdExportResult | null>(null)
  const videoInputRef = useRef<HTMLInputElement>(null)
  const qrInputRef = useRef<HTMLInputElement>(null)
  const target = accepted.find((item) => item.highlight_id === targetId) || initialTarget
  const videoAssets = assets.filter((item) => item.kind === 'video')
  const imageAssets = assets.filter((item) => item.kind === 'image')
  const activePreset = AD_PRESETS.find((item) => item.id === draft.preset_id) || AD_PRESETS[0]

  useEffect(() => {
    const stored = assignments[adAssignmentKey(job.job_id, targetId)]
    setDraft(stored || defaultAdAssignment(targetId))
    setStudioError(null)
    setExportResult(null)
  }, [assignments, job.job_id, targetId])

  const choosePreset = (preset: AdPresetDefinition) => {
    setDraft({
      ...defaultAdAssignment(targetId, preset),
      qr_asset_id: draft.qr_asset_id,
    })
    setExportResult(null)
  }
  const importAsset = async (file: File | undefined, kind: 'video' | 'image') => {
    if (!file) return
    if (!window.adStudio) { setStudioError('广告导入仅在 Electron 桌面端可用。'); return }
    setAssetBusy(kind)
    setStudioError(null)
    try {
      const asset = await window.adStudio.importAsset<AdAsset>(file, kind)
      onAssetsChange([asset, ...assets.filter((item) => item.asset_id !== asset.asset_id)])
      if (kind === 'video') {
        setDraft({ ...draft, highlight_id: targetId, kind: 'video', asset_id: asset.asset_id })
      } else {
        const qrPreset = AD_PRESETS.find((item) => item.id === 'velvet-qr') || AD_PRESETS[2]
        setDraft({ ...defaultAdAssignment(targetId, qrPreset), qr_asset_id: asset.asset_id })
      }
    } catch (reason) {
      setStudioError(reason instanceof Error ? reason.message : '广告素材导入失败')
    } finally {
      setAssetBusy(null)
    }
  }
  const save = () => {
    const normalized = { ...draft, highlight_id: targetId }
    onSave(normalized)
    return normalized
  }
  const exportVideo = async () => {
    if (!window.adStudio) { setStudioError('视频导出仅在 Electron 桌面端可用。'); return }
    if (!job.source_url) { setStudioError('当前任务没有可读取的原片地址。'); return }
    if (draft.kind === 'video' && !draft.asset_id) { setStudioError('请先选择一个广告视频。'); return }
    setExporting(true)
    setStudioError(null)
    setExportResult(null)
    try {
      const assignment = save()
      const result = await window.adStudio.exportHighlight<AdExportResult>({
        job_id: job.job_id,
        original_name: job.original_name,
        source_url: job.source_url,
        highlight: {
          highlight_id: target.highlight_id,
          start_sec: target.start_sec,
          end_sec: target.end_sec,
          description: target.description,
        },
        assignment,
      })
      setExportResult(result)
    } catch (reason) {
      setStudioError(reason instanceof Error ? reason.message : '视频导出失败')
    } finally {
      setExporting(false)
    }
  }

  return <div className="dialog-backdrop ad-studio-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget && !exporting) onClose() }}>
    <section className="ad-studio" role="dialog" aria-modal="true" aria-labelledby="ad-studio-title">
      <header className="ad-studio-head">
        <div className="ad-studio-title"><span><Icon name="ad" size={20} /></span><div><p className="overline">SHORT DRAMA AD STUDIO</p><h2 id="ad-studio-title">高光广告编排</h2></div></div>
        <div className="ad-studio-summary"><span><b>{String(accepted.length).padStart(2, '0')}</b> 已采用片段</span><i /> <span>原片安全保留</span></div>
        <button className="ad-studio-close" aria-label="关闭广告编排" disabled={exporting} onClick={onClose}><Icon name="x" size={18} /></button>
      </header>
      <div className="ad-studio-body">
        <aside className="ad-target-column">
          <div className="ad-column-head"><span>01</span><div><b>选择高光</b><small>逐个添加并导出</small></div></div>
          <div className="ad-target-list">{accepted.map((item, index) => {
            const configured = Boolean(assignments[adAssignmentKey(job.job_id, item.highlight_id)])
            return <button key={item.highlight_id} className={item.highlight_id === targetId ? 'active' : ''} onClick={() => setTargetId(item.highlight_id)}>
              <span>{String(index + 1).padStart(2, '0')}</span><div><b>{item.description || '未命名高光'}</b><small>{formatTime(item.start_sec)} — {formatTime(item.end_sec)}</small></div>{configured && <i title="已配置广告"><Icon name="check" size={11} /></i>}
            </button>
          })}</div>
          <div className="ad-target-note"><Icon name="check" size={14} /><p><b>仅处理已采用片段</b><span>广告将追加在片段尾部，不会覆盖原始高光。</span></p></div>
        </aside>
        <section className="ad-creative-column">
          <div className="ad-column-head"><span>02</span><div><b>选择广告</b><small>3 个精修预设 + 自有视频</small></div></div>
          <div className="preset-grid">{AD_PRESETS.map((preset) => {
            const previewAssignment = defaultAdAssignment(targetId, preset)
            return <button key={preset.id} className={draft.kind === 'preset' && draft.preset_id === preset.id ? 'preset-card active' : 'preset-card'} onClick={() => choosePreset(preset)}>
              <AdCreativePreview assignment={previewAssignment} assets={assets} compact />
              <span><b>{preset.name}</b><small>{preset.eyebrow}</small></span>
              <i>{preset.duration}s</i>
            </button>
          })}</div>
          <div className="custom-preset-head"><span>我的广告视频</span><button onClick={() => videoInputRef.current?.click()} disabled={assetBusy !== null}><Icon name="plus" size={13} />{assetBusy === 'video' ? '正在导入…' : '添加视频'}</button><input ref={videoInputRef} hidden type="file" accept="video/*,.mkv,.m4v" onChange={(event) => { void importAsset(event.target.files?.[0], 'video'); event.target.value = '' }} /></div>
          <div className="custom-preset-list">
            {videoAssets.map((asset) => <button key={asset.asset_id} className={draft.kind === 'video' && draft.asset_id === asset.asset_id ? 'active' : ''} onClick={() => setDraft({ ...draft, highlight_id: targetId, kind: 'video', asset_id: asset.asset_id })}><span><Icon name="play" size={15} /></span><div><b>{asset.original_name}</b><small>{asset.duration_sec?.toFixed(1)}s · {formatBytes(asset.size_bytes)}</small></div><i>{draft.kind === 'video' && draft.asset_id === asset.asset_id ? '已选' : '使用'}</i></button>)}
            {!videoAssets.length && <div className="custom-preset-empty"><Icon name="film" size={18} /><span>可导入最长 60 秒的现成广告视频</span></div>}
          </div>
        </section>
        <section className="ad-editor-column">
          <div className="ad-column-head"><span>03</span><div><b>预览与导出</b><small>画面比例跟随原片</small></div></div>
          <div className="ad-live-monitor">
            <div className="ad-live-label"><span>AD PREVIEW</span><b>{draft.kind === 'video' ? '自有视频' : activePreset.name}</b></div>
            <AdCreativePreview assignment={draft} assets={assets} />
            <div className="ad-sequence"><span style={{ flex: Math.max(.5, target.end_sec - target.start_sec) }}><b>高光</b><small>{(target.end_sec - target.start_sec).toFixed(1)}s</small></span><span className="ad-sequence-tail" style={{ flex: draft.kind === 'video' ? Math.max(.5, assets.find((item) => item.asset_id === draft.asset_id)?.duration_sec || 1) : draft.duration_sec }}><b>广告</b><small>{draft.kind === 'video' ? `${assets.find((item) => item.asset_id === draft.asset_id)?.duration_sec?.toFixed(1) || '--'}s` : `${draft.duration_sec}s`}</small></span></div>
          </div>
          {draft.kind === 'preset' ? <div className="ad-form">
            <label><span>主文案</span><input value={draft.title} maxLength={20} onChange={(event) => setDraft({ ...draft, title: event.target.value })} /></label>
            <label><span>辅助文案</span><input value={draft.subtitle} maxLength={36} onChange={(event) => setDraft({ ...draft, subtitle: event.target.value })} /></label>
            <label className="duration-control"><span>广告时长 <b>{draft.duration_sec}s</b></span><input type="range" min="2" max="8" step="1" value={draft.duration_sec} onChange={(event) => setDraft({ ...draft, duration_sec: Number(event.target.value) })} /></label>
            <div className="qr-control"><span>二维码素材 <small>扫码模板会显示，其他模板也会保留</small></span><div><button onClick={() => qrInputRef.current?.click()} disabled={assetBusy !== null}><Icon name="qr" size={14} />{assetBusy === 'image' ? '正在导入…' : '上传二维码'}</button>{imageAssets.length > 0 && <select value={draft.qr_asset_id || ''} onChange={(event) => setDraft({ ...draft, qr_asset_id: event.target.value || undefined })}><option value="">默认占位图</option>{imageAssets.map((asset) => <option key={asset.asset_id} value={asset.asset_id}>{asset.original_name}</option>)}</select>}<input ref={qrInputRef} hidden type="file" accept="image/png,image/jpeg,image/webp" onChange={(event) => { void importAsset(event.target.files?.[0], 'image'); event.target.value = '' }} /></div></div>
          </div> : <div className="custom-video-info"><Icon name="film" size={18} /><div><b>使用完整广告视频</b><p>导出时自动适配原片画幅，并统一转码后无缝追加到高光尾部。</p></div></div>}
          {studioError && <div className="ad-studio-error"><Icon name="close" size={14} /><span>{studioError}</span></div>}
          {exportResult && !exportResult.canceled && <div className="ad-export-success"><Icon name="check" size={14} /><div><b>导出完成 · {exportResult.duration_sec?.toFixed(1)}s</b><span title={exportResult.output_path}>{exportResult.output_path}</span></div></div>}
        </section>
      </div>
      <footer className="ad-studio-footer"><div><i /> 输出格式 <b>H.264 / AAC · MP4</b><span>原高光 + 片尾广告</span></div><div><button className="save-ad-button" disabled={exporting} onClick={() => { save(); setExportResult(null) }}>保存配置</button><button className="export-ad-button" disabled={exporting || (draft.kind === 'video' && !draft.asset_id)} onClick={() => { void exportVideo() }}><Icon name="download" size={16} />{exporting ? '正在合成…' : '导出新视频'}</button></div></footer>
    </section>
  </div>
}

function HighlightAssistant({ job, selected, busy, onSend }: { job: Job; selected: Highlight | null; busy: boolean; onSend: (message: string) => Promise<void> }) {
  const focus = selected ? `「${selected.description}」` : '完整原片'
  const [input, setInput] = useState('')
  const threadRef = useRef<HTMLDivElement>(null)
  const latestMessage = job.messages[job.messages.length - 1]
  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      if (threadRef.current) threadRef.current.scrollTop = threadRef.current.scrollHeight
    })
    return () => window.cancelAnimationFrame(frame)
  }, [job.messages.length, latestMessage?.content])
  const submit = async (value: string) => {
    const message = value.trim()
    if (!message || busy) return
    setInput('')
    try {
      await onSend(message)
    } catch {
      setInput(message)
    }
  }
  return <aside className="assistant-panel">
    <div className="assistant-head">
      <div><p className="overline">AI EDIT ASSISTANT</p><h2>高光编辑助手</h2></div>
      {busy && <span className="assistant-state"><i />处理中</span>}
    </div>
    <div ref={threadRef} className="assistant-thread" role="log" aria-label="高光编辑助手对话">
      <div className="assistant-context"><span>当前上下文</span><b>{focus}</b><small>{job.result?.highlights.length || 0} 个高光候选已载入</small></div>
      <div className="message assistant-message">
        <span className="message-avatar"><Icon name="spark" size={15} /></span>
        <div><small>EDIT AGENT</small><p>我正在查看{focus}。你可以查询理由，调整时间范围，修改标题或说明，也可以删除、拆分、合并和撤销。</p></div>
      </div>
      {job.messages.map((message, index) => message.role === 'user'
        ? <div className="message user-message" key={message.message_id}><div><small>YOU</small><p>{message.content}</p></div></div>
        : <div className={`message assistant-message${busy && index === job.messages.length - 1 ? ' streaming' : ''}`} key={message.message_id}><span className="message-avatar"><Icon name="spark" size={15} /></span><div><small>EDIT AGENT</small><p>{message.content}</p></div></div>)}
      <div className="assistant-suggestions"><span>快捷指令</span><div><button type="button" disabled={!selected || busy} onClick={() => void submit('入点后移 1 秒')}>入点后移 1 秒</button><button type="button" disabled={!selected || busy} onClick={() => void submit('出点前移 1 秒')}>出点前移 1 秒</button><button type="button" disabled={busy} onClick={() => void submit('撤销')}>撤销</button></div></div>
    </div>
    <form className="assistant-composer" aria-label="AI 编辑指令输入" onSubmit={(event) => { event.preventDefault(); void submit(input) }}>
      <input value={input} onChange={(event) => setInput(event.target.value)} placeholder="询问高光，或描述你想进行的调整…" disabled={busy} />
      <button type="submit" aria-label="发送编辑指令" disabled={!input.trim() || busy}>→</button>
    </form>
  </aside>
}

function AdWorkspace({ jobs, assignments, assets, onOpen, onGoReview }: { jobs: Job[]; assignments: Record<string, AdAssignment>; assets: AdAsset[]; onOpen: (job: Job, highlight: Highlight) => void; onGoReview: () => void }) {
  const acceptedRows = jobs.flatMap((record) => (record.result?.highlights || [])
    .filter((highlight) => highlight.review_status === 'accepted')
    .map((highlight) => ({ job: record, highlight, assignment: assignments[adAssignmentKey(record.job_id, highlight.highlight_id)] })))
  const configuredCount = acceptedRows.filter((row) => row.assignment).length
  const totalDuration = acceptedRows.reduce((total, row) => total + row.highlight.end_sec - row.highlight.start_sec, 0)
  const pendingReviewCount = jobs.reduce((total, record) => total + (record.result?.highlights.filter((highlight) => highlight.review_status !== 'accepted' && highlight.review_status !== 'rejected').length || 0), 0)
  return <div className="view ad-workspace-view">
    <header className="view-header"><div><p className="overline">AD DELIVERY / 03</p><h1>广告编排</h1><p>集中处理已经确认采用的高光，为每个片段追加短剧广告并导出新视频。</p></div><button className="button ad-review-link" onClick={onGoReview}><Icon name="spark" size={16} />返回高光复核</button></header>
    <section className="ad-workspace-overview">
      <div className="ad-workspace-copy"><span>SHORT DRAMA CREATIVE KIT</span><h2>三套片尾模板，逐片交付</h2><p>模板文案、时长和二维码均可替换，也可以直接使用自己的完整广告视频。</p><div><span><i />不修改原片</span><span><i />H.264 / AAC</span><span><i />单片导出</span></div></div>
      <div className="ad-workspace-presets">{AD_PRESETS.map((preset) => <div key={preset.id}><AdCreativePreview assignment={defaultAdAssignment('preview', preset)} assets={assets} compact /><span><b>{preset.name}</b><small>{preset.duration}s</small></span></div>)}</div>
    </section>
    <section className="ad-workspace-stats"><div><small>已采用高光</small><b>{String(acceptedRows.length).padStart(2, '0')}</b></div><div><small>广告已配置</small><b>{String(configuredCount).padStart(2, '0')}</b></div><div><small>待配置</small><b>{String(Math.max(0, acceptedRows.length - configuredCount)).padStart(2, '0')}</b></div><div><small>高光总时长</small><b>{formatTime(totalDuration)}</b></div></section>
    <section className="ad-delivery-board">
      <header><div><p className="overline">DELIVERY QUEUE</p><h2>逐片交付清单</h2></div><span>{acceptedRows.length ? `${configuredCount} / ${acceptedRows.length} 已就绪` : `${pendingReviewCount} 段等待复核`}</span></header>
      {acceptedRows.length > 0 && <div className="ad-delivery-table">
        <div className="ad-delivery-head"><span>来源视频</span><span>已采用高光</span><span>广告方案</span><span>片段时长</span><span>操作</span></div>
        {acceptedRows.map(({ job: record, highlight, assignment }, index) => {
          const preset = assignment?.kind === 'preset' ? AD_PRESETS.find((item) => item.id === assignment.preset_id) : null
          const asset = assignment?.kind === 'video' ? assets.find((item) => item.asset_id === assignment.asset_id) : null
          const adLabel = preset?.name || asset?.original_name || '尚未配置'
          return <article className="ad-delivery-row" key={adAssignmentKey(record.job_id, highlight.highlight_id)}>
            <div className="ad-delivery-source"><i><Icon name="film" size={16} /></i><span><b>{record.original_name}</b><small>{record.job_id}</small></span></div>
            <div className="ad-delivery-highlight"><span>{String(index + 1).padStart(2, '0')}</span><div><b>{highlight.description || '未命名高光'}</b><small>{formatTime(highlight.start_sec)} — {formatTime(highlight.end_sec)}</small></div></div>
            <div className={assignment ? 'ad-plan-state configured' : 'ad-plan-state'}><i>{assignment ? <Icon name="check" size={11} /> : <Icon name="ad" size={12} />}</i><span><b>{assignment ? '已配置' : '待添加'}</b><small>{adLabel}</small></span></div>
            <span className="ad-delivery-duration">{(highlight.end_sec - highlight.start_sec).toFixed(1)} s</span>
            <button className={assignment ? 'configured' : ''} onClick={() => onOpen(record, highlight)}><Icon name={assignment ? 'ad' : 'plus'} size={14} />{assignment ? '编辑 / 导出' : '添加广告'}</button>
          </article>
        })}
      </div>}
      {!acceptedRows.length && <div className="ad-delivery-empty"><span><Icon name="ad" size={25} /></span><div><h3>还没有已采用的高光</h3><p>先在高光提取工作区完成复核，采用的片段会自动进入这里。</p></div><button onClick={onGoReview}>前往高光复核 →</button></div>}
    </section>
  </div>
}

function Workspace({ upload, job, selected, error, chatBusy, playerReloadToken, configuredAdIds, onFile, onSelect, onReview, onAd, onChat, onCancel, onRetry, onPlaybackError }: { upload: UploadState; job: Job | null; selected: Highlight | null; error: string | null; chatBusy: boolean; playerReloadToken: number; configuredAdIds: Set<string>; onFile: (file: File) => void; onSelect: (item: Highlight | null) => void; onReview: (item: Highlight, status: ReviewStatus) => void; onAd: (item: Highlight) => void; onChat: (message: string) => Promise<void>; onCancel: () => void; onRetry: () => void; onPlaybackError: (message: string | null) => void }) {
  const busy = upload.phase !== 'idle' || job?.status === 'queued' || job?.status === 'processing'
  const reviewing = job?.status === 'completed'
  return <div className={`view workspace-view${reviewing ? ' task-review' : ''}`}>
    {!reviewing && <header className="view-header"><div><p className="overline">VIDEO INTELLIGENCE</p><h1>高光提取</h1><p>导入视频，后端完成任务编排与高光提取，再由你做最后判断。</p></div><UploadDropzone compact onFile={onFile} /></header>}
    {error && <div className="error-banner"><span><Icon name="close" size={16} /></span><div><b>流程没有完成</b><p>{error}</p></div><button onClick={onRetry}><Icon name="retry" size={15} />重新载入</button></div>}
    {!job && upload.phase === 'idle' && <UploadDropzone onFile={onFile} />}
    {busy && <ProgressPanel upload={upload} job={job} onCancel={onCancel} />}
    {job?.status === 'failed' && <div className="failed-state"><Icon name="close" size={22} /><div><h2>Agent 未能完成分析</h2><p>{job.error_message || '后端未能完成当前任务，可以稍后重新提交。'}</p></div></div>}
    {job?.status === 'completed' && <div className="review-grid"><HighlightList job={job} selectedId={selected?.highlight_id || null} configuredIds={configuredAdIds} onSelect={onSelect} onReview={onReview} onAd={onAd} /><Player job={job} selected={selected} reloadToken={playerReloadToken} onSelect={onSelect} onPlaybackError={onPlaybackError} /><HighlightAssistant job={job} selected={selected} busy={chatBusy} onSend={onChat} /></div>}
  </div>
}

function Library({ jobs, onOpen, onFile, onDelete }: { jobs: Job[]; onOpen: (job: Job) => void; onFile: (file: File) => void; onDelete: (job: Job) => void }) {
  return <div className="view library-view">
    <header className="view-header"><div><p className="overline">SERVER TASKS</p><h1>任务归档</h1><p>这里展示后端当前保留的任务、分析结果和编辑会话。</p></div><UploadDropzone compact onFile={onFile} /></header>
    <section className="archive-summary"><div><small>全部任务</small><b>{jobs.length}</b></div><div><small>提取完成</small><b>{jobs.filter((item) => item.status === 'completed').length}</b></div><div><small>处理中</small><b>{jobs.filter((item) => ['queued', 'processing'].includes(item.status)).length}</b></div><div><small>高光片段</small><b>{jobs.reduce((sum, item) => sum + (item.result?.highlights.length || 0), 0)}</b></div></section>
    <section className="job-list">
      <div className="job-list-head"><span>视频 / 任务</span><span>大小</span><span>状态</span><span>创建时间</span><span>操作</span></div>
      {jobs.map((item) => {
        const active = item.status === 'queued' || item.status === 'processing'
        return <div className="job-row" key={item.job_id}>
          <button className="job-title" onClick={() => onOpen(item)} aria-label={`打开任务 ${item.original_name}`}><i><Icon name="film" size={17} /></i><span><b>{item.original_name}</b><small>{item.job_id}</small></span></button>
          <span>{formatBytes(item.size_bytes)}</span>
          <span><i className={`status-dot ${item.status}`} />{statusCopy[item.status]}</span>
          <span>{formatDate(item.created_at)}</span>
          <div className="job-actions"><button className="open-job" onClick={() => onOpen(item)} aria-label={`打开任务 ${item.original_name}`}>→</button><button className="delete-job" disabled={active} title={active ? '分析中的任务不能删除' : '删除任务'} onClick={() => onDelete(item)} aria-label={`删除任务 ${item.original_name}`}><Icon name="trash" size={15} /></button></div>
        </div>
      })}
      {!jobs.length && <div className="empty-archive"><Icon name="folder" size={25} /><h2>还没有导入记录</h2><p>从第一条视频开始建立高光任务库。</p></div>}
    </section>
  </div>
}

function DeleteDialog({ job, busy, error, onCancel, onConfirm }: { job: Job; busy: boolean; error: string | null; onCancel: () => void; onConfirm: () => void }) {
  return <div className="dialog-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget && !busy) onCancel() }}>
    <section className="delete-dialog" role="dialog" aria-modal="true" aria-labelledby="delete-title">
      <div className="dialog-mark"><Icon name="trash" size={21} /></div>
      <p className="overline">DELETE LOCAL TASK</p>
      <h2 id="delete-title">删除这个任务？</h2>
      <p>本地任务记录、原视频、复核状态和编辑对话都会永久删除，此操作无法撤销。</p>
      <div className="delete-target"><Icon name="film" size={17} /><div><b>{job.original_name}</b><small>{job.job_id} · {formatBytes(job.size_bytes)}</small></div></div>
      {error && <div className="dialog-error">{error}</div>}
      <div className="dialog-actions"><button disabled={busy} onClick={onCancel}>取消</button><button className="confirm-delete" disabled={busy} onClick={onConfirm}>{busy ? '正在删除…' : '永久删除'}</button></div>
    </section>
  </div>
}

export default function App() {
  const [theme, setTheme] = useState<Theme>(() => {
    try {
      return window.localStorage.getItem('frame-theme') === 'light' ? 'light' : 'dark'
    } catch {
      return 'dark'
    }
  })
  const [view, setView] = useState<View>('workspace')
  const [jobs, setJobs] = useState<Job[]>([])
  const [job, setJob] = useState<Job | null>(null)
  const [selected, setSelected] = useState<Highlight | null>(null)
  const [upload, setUpload] = useState<UploadState>({ phase: 'idle', progress: 0 })
  const [error, setError] = useState<string | null>(null)
  const [online, setOnline] = useState(false)
  const [deleteCandidate, setDeleteCandidate] = useState<Job | null>(null)
  const [deleteBusy, setDeleteBusy] = useState(false)
  const [deleteError, setDeleteError] = useState<string | null>(null)
  const [chatBusy, setChatBusy] = useState(false)
  const [playerReloadToken, setPlayerReloadToken] = useState(0)
  const [adAssignments, setAdAssignments] = useState<Record<string, AdAssignment>>(() => {
    try {
      const stored = JSON.parse(window.localStorage.getItem('frame-ad-assignments-v1') || '{}')
      return stored && typeof stored === 'object' && !Array.isArray(stored) ? stored : {}
    } catch {
      return {}
    }
  })
  const [adAssets, setAdAssets] = useState<AdAsset[]>([])
  const [adStudioTarget, setAdStudioTarget] = useState<Highlight | null>(null)
  const uploadRequest = useRef<XMLHttpRequest | null>(null)

  useEffect(() => {
    try {
      window.localStorage.setItem('frame-theme', theme)
    } catch {
      // The theme still works for the current session if storage is unavailable.
    }
  }, [theme])

  useEffect(() => {
    try {
      window.localStorage.setItem('frame-ad-assignments-v1', JSON.stringify(adAssignments))
    } catch {
      // The current editing session still works if local storage is unavailable.
    }
  }, [adAssignments])

  useEffect(() => {
    if (!window.adStudio) return
    void window.adStudio.listAssets<AdAsset>().then(setAdAssets).catch(() => undefined)
  }, [])

  const refreshJobs = useCallback(async () => {
    try {
      const remoteJobs = await requestJson<RemoteJob[]>('/api/jobs')
      const records = remoteJobs.map(withSourceUrl)
      setJobs(records)
      setJob((current) => {
        if (!current) return current
        const latest = remoteJobs.find((record) => record.job_id === current.job_id)
        return latest ? mergeRemoteJob(current, latest) : current
      })
      setOnline(true)
      return records
    } catch (reason) {
      setOnline(false)
      setError(reason instanceof Error ? reason.message : '后端任务读取失败')
      return []
    }
  }, [])

  const persistJob = useCallback(async (record: Job) => {
    const saved = record
    setJob(saved)
    setJobs((current) => {
      const exists = current.some((item) => item.job_id === saved.job_id)
      const next = exists
        ? current.map((item) => item.job_id === saved.job_id ? saved : item)
        : [saved, ...current]
      return next.sort((left, right) => right.created_at.localeCompare(left.created_at))
    })
    return saved
  }, [])

  useEffect(() => { void refreshJobs() }, [refreshJobs])

  useEffect(() => {
    if (!job || !['queued', 'processing'].includes(job.status)) return
    let cancelled = false
    const timer = window.setInterval(async () => {
      try {
        const latest = await requestJson<RemoteJob>(`/api/jobs/${job.job_id}`)
        if (cancelled) return
        const saved = await persistJob(mergeRemoteJob(job, latest))
        if (cancelled) return
        setOnline(true)
        if (!['queued', 'processing'].includes(saved.status)) {
          window.clearInterval(timer)
        }
      } catch (reason) {
        if (!cancelled) setError(reason instanceof Error ? reason.message : '任务状态查询失败')
      }
    }, 1800)
    return () => { cancelled = true; window.clearInterval(timer) }
  }, [job?.job_id, job?.status, persistJob])

  useEffect(() => {
    if (!job?.result?.highlights.length) setSelected(null)
    else if (selected) setSelected(job.result.highlights.find((item) => item.highlight_id === selected.highlight_id) || null)
  }, [job?.updated_at])

  const handleFile = async (file: File) => {
    const extension = file.name.includes('.') ? file.name.slice(file.name.lastIndexOf('.')).toLowerCase() : ''
    if (!VIDEO_EXTENSIONS.includes(extension)) { setError('请选择 MP4、MOV、MKV、WEBM、AVI 或 M4V 视频。'); return }
    if (file.size > 20 * 1024 ** 3) { setError('视频不能超过 20 GB。'); return }
    const jobId = createJobId()
    setView('workspace')
    setJob(null)
    setSelected(null)
    setError(null)
    setUpload({ phase: 'uploading', progress: 0, fileName: file.name })
    try {
      const created = await uploadVideo(jobId, file, (progress) => setUpload({ phase: 'uploading', progress, fileName: file.name }), (request) => { uploadRequest.current = request })
      await persistJob(withSourceUrl(created))
      setUpload({ phase: 'idle', progress: 0 })
      setOnline(true)
    } catch (reason) {
      setUpload({ phase: 'idle', progress: 0 })
      const message = reason instanceof Error ? reason.message : '视频导入失败'
      setError(message)
    }
  }

  const review = async (item: Highlight, reviewStatus: ReviewStatus) => {
    if (!job?.result) return
    try {
      const updated: Job = {
        ...job,
        updated_at: new Date().toISOString(),
        result: {
          ...job.result,
          highlights: job.result.highlights.map((highlight) => highlight.highlight_id === item.highlight_id ? { ...highlight, review_status: reviewStatus } : highlight),
        },
      }
      await persistJob(updated)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '复核状态保存失败')
    }
  }

  const openAdStudio = (item: Highlight) => {
    if (item.review_status !== 'accepted') return
    setSelected(item)
    setAdStudioTarget(item)
  }

  const openAdStudioFromQueue = (record: Job, item: Highlight) => {
    if (item.review_status !== 'accepted') return
    setJob(record)
    setSelected(item)
    setAdStudioTarget(item)
  }

  const saveAdAssignment = (assignment: AdAssignment) => {
    if (!job) return
    setAdAssignments((current) => ({
      ...current,
      [adAssignmentKey(job.job_id, assignment.highlight_id)]: assignment,
    }))
  }

  const sendEditMessage = async (message: string) => {
    if (!job) return
    const currentJob = job
    setChatBusy(true)
    setError(null)
    const userMessage: ChatMessage = {
      message_id: `msg_${crypto.randomUUID()}`,
      role: 'user',
      content: message,
      created_at: new Date().toISOString(),
    }
    const assistantMessage: ChatMessage = {
      message_id: `msg_${crypto.randomUUID()}`,
      role: 'assistant',
      content: '',
      created_at: new Date().toISOString(),
    }
    setJob({
      ...currentJob,
      messages: [...currentJob.messages, userMessage, assistantMessage],
    })
    try {
      const response = await requestMessageStream(
        `/api/jobs/${currentJob.job_id}/messages/stream`,
        {
          message,
          revision: currentJob.revision,
          selected_highlight_id: selected?.highlight_id || null,
        },
        (reply) => setJob((active) => active?.job_id === currentJob.job_id ? {
          ...active,
          messages: active.messages.map((item) => item.message_id === assistantMessage.message_id
            ? { ...item, content: reply }
            : item),
        } : active),
      )
      const merged = mergeRemoteJob(currentJob, response.job)
      await persistJob({
        ...merged,
        messages: [
          ...currentJob.messages,
          userMessage,
          { ...assistantMessage, content: response.reply },
        ],
      })
      setOnline(true)
    } catch (reason) {
      setJob((active) => active?.job_id === currentJob.job_id ? currentJob : active)
      setError(reason instanceof Error ? reason.message : '高光编辑失败')
      throw reason
    } finally {
      setChatBusy(false)
    }
  }

  const openJob = (record: Job) => {
    setJob(record)
    setSelected(null)
    setError(null)
    setView('workspace')
    return

    setChatBusy(true)
    void requestJson<RemoteJob>(`/api/demo-jobs/${record.job_id}/session`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        original_name: record.original_name,
        size_bytes: record.size_bytes,
        language: record.language,
        result: record.result,
      }),
    }).then(async (remote) => {
      await persistJob(mergeRemoteJob(record, remote))
      setOnline(true)
    }).catch((reason) => {
      setError(reason instanceof Error ? reason.message : '演示对话会话创建失败')
    }).finally(() => setChatBusy(false))
  }
  const requestDelete = (record: Job) => { setDeleteCandidate(record); setDeleteError(null) }
  const closeDelete = () => { if (!deleteBusy) { setDeleteCandidate(null); setDeleteError(null) } }
  const confirmDelete = async () => {
    if (!deleteCandidate) return
    setDeleteBusy(true)
    setDeleteError(null)
    try {
      await requestEmpty(`/api/jobs/${deleteCandidate.job_id}`, { method: 'DELETE' })
      setJobs((current) => current.filter((record) => record.job_id !== deleteCandidate.job_id))
      if (job?.job_id === deleteCandidate.job_id) { setJob(null); setSelected(null) }
      setDeleteCandidate(null)
    } catch (reason) {
      setDeleteError(reason instanceof Error ? reason.message : '任务删除失败')
    } finally {
      setDeleteBusy(false)
    }
  }
  const activeHighlight = useMemo(() => selected && job?.result?.highlights.find((item) => item.highlight_id === selected.highlight_id) || null, [job, selected])
  const configuredAdIds = useMemo(() => {
    if (!job?.result) return new Set<string>()
    return new Set(job.result.highlights.filter((item) => Boolean(adAssignments[adAssignmentKey(job.job_id, item.highlight_id)])).map((item) => item.highlight_id))
  }, [adAssignments, job])

  return <div className="app-shell" data-theme={theme}>
    <WindowChrome theme={theme} onToggleTheme={() => setTheme((current) => current === 'dark' ? 'light' : 'dark')} />
    <Sidebar view={view} onView={setView} online={online} jobs={jobs} />
    <main>{view === 'workspace'
      ? <Workspace upload={upload} job={job} selected={activeHighlight} error={error} chatBusy={chatBusy} playerReloadToken={playerReloadToken} configuredAdIds={configuredAdIds} onFile={handleFile} onSelect={setSelected} onReview={review} onAd={openAdStudio} onChat={sendEditMessage} onCancel={() => uploadRequest.current?.abort()} onRetry={() => { setError(null); setPlayerReloadToken((value) => value + 1); void refreshJobs() }} onPlaybackError={setError} />
      : view === 'library'
        ? <Library jobs={jobs} onOpen={openJob} onFile={handleFile} onDelete={requestDelete} />
        : <AdWorkspace jobs={jobs} assignments={adAssignments} assets={adAssets} onOpen={openAdStudioFromQueue} onGoReview={() => setView('workspace')} />}
    </main>
    {deleteCandidate && <DeleteDialog job={deleteCandidate} busy={deleteBusy} error={deleteError} onCancel={closeDelete} onConfirm={confirmDelete} />}
    {adStudioTarget && job && <AdStudioDialog job={job} initialTarget={adStudioTarget} assignments={adAssignments} assets={adAssets} onAssetsChange={setAdAssets} onSave={saveAdAssignment} onClose={() => setAdStudioTarget(null)} />}
  </div>
}

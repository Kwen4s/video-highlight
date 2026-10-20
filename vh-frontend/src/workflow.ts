export type DetectionOptions = {
  instruction: string
  min_clip_sec: number
  max_clip_sec: number
  max_highlights: number | null
  total_duration_sec: number | null
  allow_overlap: boolean
}

export const DEFAULT_DETECTION_OPTIONS: DetectionOptions = {
  instruction: '挑选有看点、能独立看懂的短剧片段，保留必要铺垫和反应，剪辑简洁流畅。',
  min_clip_sec: 3,
  max_clip_sec: 24,
  max_highlights: 12,
  total_duration_sec: null,
  allow_overlap: true,
}

export type AnalysisProgress = {
  scan_coverage: number
  pending_event_count: number
  pending_observation_count: number
  pending_proposal_count: number
  pending_review_count: number
  model_calls: number
  duration_sec?: number
  completed_page_count?: number
  page_count?: number
  event_count?: number
  candidate_count?: number
  ready_clip_count?: number
  transcript_rows_read?: number
  turn?: number
  capabilities?: WorkflowCapabilities
}

export type TaskSpecification = DetectionOptions & { subtitle_provided: boolean }

export type WorkflowCapabilities = {
  transcript?: boolean | null
  semantic_search?: boolean | null
  read_frames?: boolean | null
  read_text?: boolean | null
  local_proposals?: boolean | null
}

export type WorkflowSnapshot = {
  schema_version?: '1.0'
  updated_at?: string | null
  duration_sec?: number | null
  completed_page_count?: number | null
  page_count?: number | null
  next_scan_range?: { start_sec: number; end_sec: number } | null
  event_count?: number | null
  candidate_count?: number | null
  ready_clip_count?: number | null
  transcript_rows_read?: number | null
  model_calls?: number | null
  turn?: number | null
  capabilities?: WorkflowCapabilities
}

export type WorkflowJob = {
  status: 'queued' | 'processing' | 'completed' | 'failed'
  revision: number
  progress: AnalysisProgress | null
  result: { completion: 'complete' | 'partial'; analysis: AnalysisProgress } | null
  workflow?: WorkflowSnapshot | null
  task?: TaskSpecification | null
}

export function workflowSnapshot(job: WorkflowJob | null): WorkflowSnapshot | null {
  if (job?.workflow) return job.workflow
  // Older deployments already expose these public progress fields.
  const progress = job?.progress
  if (!progress) return null
  return {
    duration_sec: progress.duration_sec,
    completed_page_count: progress.completed_page_count,
    page_count: progress.page_count,
    event_count: progress.event_count,
    candidate_count: progress.candidate_count,
    ready_clip_count: progress.ready_clip_count,
    transcript_rows_read: progress.transcript_rows_read,
    model_calls: progress.model_calls,
    turn: progress.turn,
    capabilities: progress.capabilities,
  }
}

export type WorkflowState = 'done' | 'current' | 'waiting' | 'failed'

export function isReviewReady(job: WorkflowJob | null): boolean {
  return job?.status === 'completed' && job.result?.completion === 'complete'
}

export function canResume(job: WorkflowJob | null): boolean {
  return Boolean(job && job.revision === 0 && (
    job.status === 'failed' || (job.status === 'completed' && job.result?.completion === 'partial')
  ))
}

export function analysisProgress(job: WorkflowJob | null): AnalysisProgress | null {
  // A terminal result is authoritative; a checkpoint may precede final delivery.
  return job?.status === 'completed' ? job.result?.analysis || job.progress : job?.progress || job?.result?.analysis || null
}

export function coveragePercent(progress: AnalysisProgress | null): number | null {
  if (!progress || !Number.isFinite(progress.scan_coverage)) return null
  return Math.round(Math.max(0, Math.min(1, progress.scan_coverage)) * 100)
}

export function workflowStates(job: WorkflowJob | null, uploading: boolean): WorkflowState[] {
  if (uploading) return ['current', 'waiting', 'waiting', 'waiting', 'waiting']
  if (!job) return ['waiting', 'waiting', 'waiting', 'waiting', 'waiting']
  if (isReviewReady(job)) return ['done', 'done', 'done', 'done', 'current']
  if (job.status === 'queued') return ['done', 'current', 'waiting', 'waiting', 'waiting']
  if (job.status === 'processing') return ['done', 'done', 'current', 'waiting', 'waiting']
  return ['done', 'done', 'failed', 'waiting', 'waiting']
}

export function validateDetectionOptions(options: DetectionOptions): string | null {
  if (!options.instruction.trim()) return '请填写提取目标。'
  if (!Number.isFinite(options.min_clip_sec) || options.min_clip_sec <= 0
    || !Number.isFinite(options.max_clip_sec) || options.max_clip_sec <= 0) return '片段时长必须是大于 0 的秒数。'
  if (options.min_clip_sec > options.max_clip_sec) return '最短时长不能超过最长时长。'
  if (options.max_highlights !== null && (!Number.isInteger(options.max_highlights) || options.max_highlights <= 0)) return '片段数量必须是正整数，或留空表示不限。'
  if (options.total_duration_sec !== null && (!Number.isFinite(options.total_duration_sec) || options.total_duration_sec <= 0)) return '总时长必须大于 0，或留空表示不限。'
  return null
}

const SUBTITLE_EXTENSIONS = ['.srt', '.vtt', '.ass', '.ssa', '.sub']

function baseName(name: string) {
  return name.replace(/\.[^.]+$/, '').toLocaleLowerCase()
}

export function matchSubtitles<T extends { name: string }>(videos: T[], subtitles: T[]): (T | null)[] {
  if (subtitles.some((file) => !SUBTITLE_EXTENSIONS.some((extension) => file.name.toLowerCase().endsWith(extension)))) {
    throw new Error('字幕仅支持 SRT、VTT、ASS、SSA 或 SUB 格式。')
  }
  if (videos.length === 1 && subtitles.length === 1) return [subtitles[0]]
  const unused = new Set(subtitles)
  const matches = videos.map((video) => {
    const candidates = subtitles.filter((subtitle) => baseName(subtitle.name) === baseName(video.name))
    if (candidates.length > 1) throw new Error(`「${video.name}」匹配到多份字幕，请只保留一份。`)
    if (candidates[0]) unused.delete(candidates[0])
    return candidates[0] || null
  })
  if (unused.size) throw new Error(`字幕「${[...unused][0].name}」没有匹配的视频，请使用同名文件或移除字幕。`)
  return matches
}

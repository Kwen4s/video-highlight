const { test } = require('node:test')
const assert = require('node:assert/strict')
const workflow = import('../src/workflow.ts')

const progress = {
  scan_coverage: 1, pending_event_count: 2, pending_observation_count: 1,
  pending_proposal_count: 3, pending_review_count: 4, model_calls: 8,
}
function job(status, completion = null, revision = 0) {
  return { status, revision, progress, result: completion ? { completion, analysis: progress } : null }
}

test('100% scan coverage does not imply delivery or human review readiness', async () => {
  const { isReviewReady, workflowStates, canResume } = await workflow
  assert.equal(isReviewReady(job('processing')), false)
  assert.equal(isReviewReady(job('completed', 'partial')), false)
  assert.deepEqual(workflowStates(job('completed', 'partial'), false), ['done', 'done', 'failed', 'waiting', 'waiting'])
  assert.equal(canResume(job('completed', 'partial')), true)
  assert.equal(canResume(job('completed', 'complete')), false)
  assert.equal(isReviewReady(job('completed', 'complete')), true)
})

test('resume respects running tasks and edited results', async () => {
  const { canResume, workflowStates } = await workflow
  assert.equal(canResume(job('failed')), true)
  assert.equal(canResume(job('failed', null, 1)), false)
  assert.equal(canResume(job('completed', 'partial', 1)), false)
  assert.equal(canResume(job('queued', 'partial')), false)
  assert.equal(canResume(job('processing', 'partial')), false)
  assert.deepEqual(workflowStates(job('queued', 'partial'), false), ['done', 'current', 'waiting', 'waiting', 'waiting'])
})

test('terminal analysis wins over an older checkpoint and unavailable coverage stays unknown', async () => {
  const { analysisProgress, coveragePercent } = await workflow
  const record = job('completed', 'complete')
  record.progress = { ...progress, scan_coverage: .5 }
  assert.equal(coveragePercent(analysisProgress(record)), 100)
  assert.equal(coveragePercent(null), null)
  assert.equal(coveragePercent({ ...progress, scan_coverage: NaN }), null)
  assert.equal(coveragePercent({ ...progress, scan_coverage: 1.3 }), 100)
})

test('workflow details preserve unavailable capabilities and use the public snapshot when present', async () => {
  const { workflowSnapshot } = await workflow
  const record = job('processing')
  record.progress = { ...progress, duration_sec: 60, page_count: 2, capabilities: { transcript: true, read_text: false } }
  assert.equal(workflowSnapshot(record).page_count, 2)
  assert.equal(workflowSnapshot(record).capabilities.read_text, false)
  assert.equal(workflowSnapshot(record).capabilities.local_proposals, undefined)
  record.workflow = { page_count: 3, capabilities: { read_text: true } }
  assert.equal(workflowSnapshot(record).page_count, 3)
  assert.equal(workflowSnapshot(record).capabilities.read_text, true)
  assert.equal(workflowSnapshot(null), null)
})

test('task constraints validate before upload and accept nullable budgets and long clips', async () => {
  const { DEFAULT_DETECTION_OPTIONS: defaults, validateDetectionOptions: validate } = await workflow
  assert.equal(validate({ ...defaults, max_clip_sec: 60, max_highlights: null, total_duration_sec: null }), null)
  for (const overrides of [
    { instruction: '  ' }, { min_clip_sec: 0 }, { max_clip_sec: Infinity },
    { min_clip_sec: 30, max_clip_sec: 24 }, { max_highlights: 1.5 },
    { max_highlights: 0 }, { total_duration_sec: -1 },
  ]) assert.equal(typeof validate({ ...defaults, ...overrides }), 'string')
})

test('batch subtitles match names instead of selection order', async () => {
  const { matchSubtitles } = await workflow
  const videos = [{ name: 'EP01.mp4' }, { name: 'ep02.mov' }, { name: 'ep03.mp4' }]
  const subtitles = [{ name: 'ep02.srt' }, { name: 'ep01.VTT' }]
  assert.deepEqual(matchSubtitles(videos, subtitles), [subtitles[1], subtitles[0], null])
  assert.throws(() => matchSubtitles(videos, [{ name: 'other.srt' }]), /没有匹配/)
  assert.throws(() => matchSubtitles(videos, [{ name: 'ep01.srt' }, { name: 'ep01.ass' }]), /多份字幕/)
  assert.throws(() => matchSubtitles(videos, [{ name: 'ep01.txt' }]), /字幕仅支持/)
  assert.deepEqual(matchSubtitles([videos[0]], [{ name: '任意名称.ass' }]), [{ name: '任意名称.ass' }])
})

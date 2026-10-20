// Run with Electron after `npm run build`. All HTTP requests use local fixtures.
const { app, BrowserWindow, protocol } = require('electron')
const assert = require('node:assert/strict')
const fs = require('node:fs/promises')
const path = require('node:path')
const { execFileSync } = require('node:child_process')

const outputDir = path.resolve(__dirname, '../../.cache/frontend-ui')
require('node:fs').mkdirSync(outputDir, { recursive: true })
app.setPath('userData', path.join(outputDir, 'profile'))
app.commandLine.appendSwitch('disable-gpu')
const analysis = {
  scan_coverage: 1, pending_event_count: 0, pending_observation_count: 0,
  pending_proposal_count: 0, pending_review_count: 0, model_calls: 18, stop_reason: 'complete',
}
const highlight = {
  highlight_id: 'hl_reveal', start_sec: 18, end_sec: 31, score: .94,
  highlight_type: 'reveal', description: '关键线索揭晓', reason: '人物关系与动机在同一场景中得到确认，保留必要铺垫与反应。',
  clip_url: '/api/jobs/job_complete/clips/hl_reveal', review_status: 'pending',
}
function fixture(id, status, completion = null) {
  return {
    job_id: id, status, original_name: `${id}.mp4`, content_type: 'video/mp4', size_bytes: 12000000,
    language: 'zh', created_at: '2026-10-03T04:00:00Z', updated_at: '2026-10-03T04:02:00Z',
    revision: 0, attempt: 1, progress: status === 'failed' ? null : { ...analysis, scan_coverage: .72, pending_review_count: 2 },
    task: { instruction: '保留冲突反转与必要铺垫。', min_clip_sec: 3, max_clip_sec: 24, max_highlights: 12, total_duration_sec: null, allow_overlap: true, subtitle_provided: false },
    workflow: status === 'failed' ? null : {
      schema_version: '1.0', updated_at: '2026-10-03T04:02:00Z', duration_sec: 117,
      completed_page_count: completion ? 4 : 3, page_count: 4,
      next_scan_range: completion ? null : { start_sec: 90, end_sec: 117 },
      event_count: 6, candidate_count: 4, ready_clip_count: 2, transcript_rows_read: 18, model_calls: 18, turn: 6,
      capabilities: { transcript: true, read_text: true, read_frames: true, semantic_search: true, local_proposals: false },
    },
    source_url: `/api/jobs/${id}/source`, error_message: status === 'failed' ? '服务重启中断了分析，可从检查点继续' : null,
    result: completion ? {
      schema_version: '2.0', job_id: id, video: { video_id: id, title: id, duration_sec: 117 },
      completion, message: completion === 'partial' ? '全片已检查，仍有事件与成片等待复核。' : '高光分析完成。',
      analysis: completion === 'partial' ? { ...analysis, pending_event_count: 1, pending_review_count: 2 } : analysis,
      highlights: completion === 'complete' ? [structuredClone(highlight)] : [],
    } : null,
  }
}
let jobs = [fixture('job_complete', 'completed', 'complete'), fixture('job_partial', 'completed', 'partial'), fixture('job_failed', 'failed')]
let resumeCount = 0
let resumeMode = 'conflict'
const uploads = []
let releaseFirstUpload
const firstUploadGate = new Promise(resolve => { releaseFirstUpload = resolve })
let failUpload = true
let deferJobList = false
let releaseJobList
const staleJobListGate = new Promise(resolve => { releaseJobList = resolve })
const requests = []
let media
const headers = { 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Headers': '*', 'Access-Control-Allow-Methods': 'GET,HEAD,POST,OPTIONS' }
const json = (value, status = 200) => new Response(JSON.stringify(value), { status, headers: { ...headers, 'Content-Type': 'application/json' } })
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))
let window
async function evaluate(code) { return window.webContents.executeJavaScript(code, true) }
async function until(code, label) {
  for (let index = 0; index < 80; index++) {
    if (await evaluate(code)) return
    await sleep(50)
  }
  throw new Error(`UI timeout: ${label}`)
}
async function click(text) {
  await evaluate(`(() => { const el = [...document.querySelectorAll('button')].find(el => el.textContent.includes(${JSON.stringify(text)})); if (!el) throw new Error('Missing button'); el.click(); })()`)
  await sleep(80)
}
async function openJob(id) {
  await click('任务归档')
  await evaluate(`document.querySelector('button[aria-label="打开任务 ${id}.mp4"]').click()`)
  await sleep(120)
}
async function capture(name) {
  await sleep(150)
  await fs.writeFile(path.join(outputDir, `${name}.png`), (await window.webContents.capturePage()).toPNG())
}
async function setInput(selector, value) {
  await evaluate(`(() => { const el = document.querySelector(${JSON.stringify(selector)}); const setter = Object.getOwnPropertyDescriptor(el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype, 'value').set; setter.call(el, ${JSON.stringify(value)}); el.dispatchEvent(new Event('input', { bubbles: true })); })()`)
  await sleep(40)
}
async function attach(selector, files) {
  await evaluate(`(() => { const transfer = new DataTransfer(); for (const name of ${JSON.stringify(files)}) transfer.items.add(new File(['fixture'], name)); const input = document.querySelector(${JSON.stringify(selector)}); input.files = transfer.files; input.dispatchEvent(new Event('change', { bubbles: true })); })()`)
  await sleep(80)
}

app.whenReady().then(async () => {
  await fs.mkdir(outputDir, { recursive: true })
  const mediaPath = path.join(outputDir, 'fixture.mp4')
  execFileSync(require('ffmpeg-static'), ['-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=c=0x153020:s=360x640:r=10', '-t', '1', '-c:v', 'libx264', mediaPath])
  media = await fs.readFile(mediaPath)
  async function handle(request) {
    const url = new URL(request.url)
    requests.push(`${request.method} ${url.pathname}`)
    if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers })
    if (url.pathname === '/api/jobs' && request.method === 'GET') {
      const snapshot = structuredClone(jobs)
      if (deferJobList) { deferJobList = false; await staleJobListGate }
      return json(snapshot)
    }
    if (url.pathname === '/api/jobs' && request.method === 'POST') {
      const form = await request.formData()
      const upload = { id: form.get('job_id'), options: JSON.parse(form.get('options')), language: form.get('language'), subtitles: form.get('subtitles')?.name, name: form.get('file')?.name }
      uploads.push(upload)
      if (upload.name === 'ep01.mp4') await firstUploadGate
      await sleep(700)
      if (upload.name === 'failure.mp4' && failUpload) return json({ detail: '接入存储暂时不可用' }, 503)
      const record = fixture(upload.id, 'queued')
      record.original_name = upload.name
      record.task = { ...upload.options, subtitle_provided: Boolean(upload.subtitles) }
      jobs.push(record)
      if (upload.name === 'lost-receipt.mp4') return json({ detail: '上传回执丢失' }, 503)
      return json(record, 202)
    }
    if (url.pathname.endsWith('/resume')) {
      resumeCount++
      await sleep(250)
      if (resumeMode === 'conflict') return json({ detail: '没有可恢复的检查点' }, 409)
      const record = jobs.find((item) => url.pathname.includes(item.job_id))
      record.status = 'queued'
      return json(record)
    }
    if (url.pathname.endsWith('/range')) {
      const record = jobs.find((item) => url.pathname.includes(item.job_id))
      const input = await request.json()
      assert.equal(input.revision, record.revision)
      Object.assign(record.result.highlights[0], input, { clip_url: '', review_status: 'revised' })
      record.revision++
      return json(record)
    }
    if (url.pathname.endsWith('/messages/stream')) {
      const record = jobs.find((item) => url.pathname.includes(item.job_id))
      return new Response([
        { type: 'start' }, { type: 'delta', delta: '当前高光包含关键线索。' },
        { type: 'complete', changed: false, job: record },
      ].map((event) => JSON.stringify(event)).join('\n'), { headers: { ...headers, 'Content-Type': 'application/x-ndjson' } })
    }
    if (url.pathname.endsWith('/source') || url.pathname.includes('/clips/')) {
      return new Response(request.method === 'HEAD' ? null : media, { headers: { ...headers, 'Content-Type': 'video/mp4', 'Content-Length': String(media.length) } })
    }
    if (/^\/api\/jobs\/[^/]+$/.test(url.pathname)) {
      const record = jobs.find(item => item.job_id === url.pathname.split('/').pop())
      return record ? json(record) : json({ detail: '任务不存在' }, 404)
    }
    return json({ detail: 'Unknown fixture endpoint' }, 404)
  }
  await protocol.handle('http', handle)
  await protocol.handle('https', handle)
  await require('electron').session.defaultSession.clearStorageData()
  window = new BrowserWindow({ width: 1480, height: 940, show: false, webPreferences: { contextIsolation: true, nodeIntegration: false, backgroundThrottling: false, offscreen: true } })
  const consoleErrors = []
  window.webContents.on('console-message', (event) => {
    if (event.message?.includes('Uncaught')) consoleErrors.push(event.message)
  })
  await window.loadFile(path.resolve(__dirname, '../dist/index.html'))
  await window.webContents.insertCSS('*, *::before, *::after { animation: none !important; transition: none !important; }')
  await until('document.querySelector(".sidebar-metric strong")?.textContent === "03"', 'initial jobs')
  await capture('01-intake-dark')
  await evaluate('document.querySelector(".task-options").open = true')
  await capture('02-options-dark')
  await evaluate('document.querySelector(".theme-toggle").click()')
  await capture('03-options-light')

  await openJob('job_partial')
  assert.equal(await evaluate('!!document.querySelector(".review-grid")'), false)
  assert.match(await evaluate('document.querySelector(".progress-heading-actions").textContent'), /100%/)
  assert.equal(await evaluate('document.querySelector(".analysis-resume button").getBoundingClientRect().bottom < innerHeight'), true)
  await capture('04-partial-light')
  window.setSize(1920, 1080)
  await click('辅助感知')
  assert.equal(await evaluate('document.querySelectorAll(".workflow-detail-node").length'), 5)
  assert.match(await evaluate('document.querySelector("#workflow-group-panel").textContent'), /ASR \/ 字幕检索/)
  assert.match(await evaluate('document.querySelector("#workflow-group-panel").textContent'), /OCR 画面文字/)
  assert.match(await evaluate('[...document.querySelectorAll(".workflow-detail-node")].find(node => node.textContent.includes("OCR 画面文字")).textContent'), /可调用/)
  assert.doesNotMatch(await evaluate('[...document.querySelectorAll(".workflow-detail-node")].find(node => node.textContent.includes("OCR 画面文字")).textContent'), /已完成/)
  assert.match(await evaluate('[...document.querySelectorAll(".workflow-detail-node")].find(node => node.textContent.includes("本地候选定位")).textContent'), /未启用/)
  assert.equal(await evaluate('document.querySelectorAll(".workflow-node-io dt").length'), 10)
  await capture('16-perception-light')
  await click('事件核验')
  assert.match(await evaluate('document.querySelector("#workflow-group-panel").textContent'), /6 个事件/)
  await capture('17-events-light')
  await click('成片复核')
  assert.match(await evaluate('document.querySelector("#workflow-group-panel").textContent'), /3–24 秒/)
  assert.match(await evaluate('document.querySelector("#workflow-group-panel").textContent'), /尚未完成/)
  await capture('18-clips-light')
  await click('成片交付')
  assert.match(await evaluate('document.querySelector("#workflow-group-panel").textContent'), /未完成结果/)
  await click('人工复核')
  assert.match(await evaluate('document.querySelector("#workflow-group-panel").textContent'), /完整交付后进入复核/)
  await click('原片接入')
  assert.match(await evaluate('document.querySelector("#workflow-group-panel").textContent'), /保留冲突反转/)
  await capture('19-intake-details-light')
  await click('视频分析')
  await capture('20-scan-light')
  window.setSize(1480, 940)
  await click('继续分析')
  await until('document.querySelector(".error-banner")?.textContent.includes("没有可恢复的检查点")', 'resume conflict')
  assert.equal(resumeCount, 1)
  resumeMode = 'success'
  await evaluate('document.querySelector(".analysis-resume button").click(); document.querySelector(".analysis-resume button").click()')
  await until('document.querySelector(".progress-heading b")?.textContent === "等待分析"', 'resume response')
  assert.equal(resumeCount, 2)
  await capture('05-resumed-light')
  assert.match(await evaluate('document.querySelector("#workflow-group-panel").textContent'), /等待服务端分析资源/)

  await openJob('job_complete')
  await until('!!document.querySelector(".review-grid")', 'complete result review')
  assert.equal(await evaluate('!!document.querySelector(".task-options")'), false)
  assert.equal(await evaluate('document.querySelector("main").scrollHeight > document.querySelector("main").clientHeight'), false)
  await click('关键线索揭晓')
  await until('document.querySelector(".media-chip")?.textContent.includes("已复核成片")', 'reviewed clip label')
  await until('document.querySelector("video")?.src.includes("/clips/")', 'reviewed media URL')
  await capture('06-review-light')
  assert.equal(await evaluate('document.querySelector(".detail-narrative").getBoundingClientRect().width >= 190'), true)
  await evaluate('document.querySelector(".theme-toggle").click()')
  await capture('07-review-dark')
  await click('采用')
  await click('加广告')
  assert.equal(await evaluate('!!document.querySelector(".ad-studio")'), true)
  await capture('08-ad-studio-dark')
  await evaluate('document.querySelector(".ad-studio-close").click()')
  await setInput('.timeline-editor-controls label:nth-child(2) input', '50')
  await click('保存范围')
  await until('document.querySelector(".media-chip")?.textContent.includes("原片预览")', 'edited range source fallback')
  await until('document.querySelector("video")?.src.endsWith("/source")', 'original media after editing')
  assert.equal(jobs[0].result.highlights[0].end_sec, 50)
  await setInput('.assistant-composer input', '列出高光')
  await evaluate('document.querySelector(".assistant-composer button").click()')
  await until('document.querySelector(".assistant-thread")?.textContent.includes("当前高光包含关键线索")', 'streamed reply')

  window.setSize(1080, 700)
  await capture('09-review-minimum-size')
  const layout = await evaluate(`(() => { const grid = document.querySelector('.review-grid').getBoundingClientRect(); const main = document.querySelector('main'); return { bottom: grid.bottom, height: innerHeight, horizontalOverflow: main.scrollWidth > main.clientWidth, verticalOverflow: main.scrollHeight > main.clientHeight }; })()`)
  assert.equal(layout.horizontalOverflow, false)
  assert.equal(layout.verticalOverflow, false)
  assert(layout.bottom <= layout.height)
  window.setSize(1480, 940)

  await openJob('job_failed')
  await capture('10-failed-dark')
  // A late resume response must not replace a different task opened meanwhile.
  await evaluate('document.querySelector(".analysis-resume button").click()')
  await openJob('job_complete')
  await sleep(350)
  assert.equal(await evaluate('!!document.querySelector(".review-grid")'), true)
  await openJob('job_failed')
  await click('高光提取')
  assert.equal(await evaluate('!!document.querySelector(".dropzone")'), false)
  assert.match(await evaluate('document.querySelector(".progress-heading small").textContent'), /job_failed.mp4/)
  await evaluate('document.querySelector(".task-options").open = true')
  await setInput('.task-instruction textarea', '   ')
  await attach('.view-header input[type=file]', ['ep01.mp4'])
  await until('document.querySelector(".error-banner")?.textContent.includes("请填写提取目标")', 'blank instruction validation')
  assert.equal(uploads.length, 0)
  await setInput('.task-instruction textarea', '保留剧情反转与必要铺垫')
  await attach('.subtitle-intake input[type=file]', ['ep02.srt', 'ep01.vtt'])
  await evaluate(`window.intakeRequests = []; const originalOpen = XMLHttpRequest.prototype.open; XMLHttpRequest.prototype.open = function(method, url, ...args) { if (method === 'POST' && url.endsWith('/api/jobs')) window.intakeRequests.push(this); return originalOpen.call(this, method, url, ...args) }; undefined`)
  await attach('.view-header input[type=file]', ['ep01.mp4', 'ep02.mp4'])
  await until('document.querySelector(".progress-heading b")?.textContent === "正在上传原片"', 'immediate upload workflow')
  assert.equal(await evaluate('!!document.querySelector(".dropzone")'), false)
  assert.equal(await evaluate('document.querySelector(".sidebar-metric strong").textContent'), '05')
  await evaluate(`window.intakeRequests[0].upload.dispatchEvent(new ProgressEvent('progress', { lengthComputable: true, loaded: 524288, total: 1048576 }))`)
  assert.match(await evaluate('document.querySelector(".progress-heading-actions").textContent'), /50%/)
  assert.match(await evaluate('document.querySelector(".workflow-disclosure").textContent'), /512 KB \/ 1\.0 MB/)
  await capture('11-upload-dark')
  await click('任务归档')
  assert.equal(await evaluate('document.querySelectorAll(".job-row").length'), 5)
  assert.match(await evaluate('[...document.querySelectorAll(".job-row")].find(row => row.textContent.includes("ep01.mp4")).textContent'), /上传 50%/)
  assert.match(await evaluate('[...document.querySelectorAll(".job-row")].find(row => row.textContent.includes("ep02.mp4")).textContent'), /等待上传/)
  deferJobList = true
  await sleep(1900)
  assert.equal(deferJobList, false)
  assert.equal(await evaluate('document.querySelectorAll(".job-row").length'), 5)
  await capture('14-upload-archive-dark')
  await click('高光提取')
  assert.match(await evaluate('document.querySelector(".progress-heading small").textContent'), /ep01.mp4/)
  assert.match(await evaluate('document.querySelector(".progress-heading-actions").textContent'), /50%/)
  await evaluate(`window.intakeRequests[0].upload.dispatchEvent(new ProgressEvent('load'))`)
  await until('document.querySelector(".progress-heading b")?.textContent === "等待服务端确认"', 'waiting for backend receipt')
  await capture('15-confirming-dark')
  await click('任务归档')
  await evaluate(`document.querySelector('button[aria-label="打开任务 ep02.mp4"]').click()`)
  assert.equal(await evaluate('document.querySelector(".progress-heading b").textContent'), '等待上传原片')
  assert.match(await evaluate('document.querySelector(".progress-heading small").textContent'), /ep02.mp4/)
  await click('任务归档')
  await evaluate(`document.querySelector('button[aria-label="打开任务 ep01.mp4"]').click()`)
  releaseFirstUpload()
  await until('document.querySelector(".sidebar-metric strong")?.textContent === "05" && document.querySelector(".progress-heading b")?.textContent === "等待分析"', 'batch uploads')
  await sleep(800)
  releaseJobList()
  await sleep(150)
  assert.equal(await evaluate('document.querySelector(".sidebar-metric strong").textContent'), '05')
  assert.match(await evaluate('document.querySelector(".progress-heading small").textContent'), /ep01.mp4/)
  assert.equal(await evaluate('!!document.querySelector(".upload-queue-panel")'), false)
  assert.equal(await evaluate('!!document.querySelector(".progress-panel")'), true)
  assert.equal(uploads.length, 2)
  assert.equal(uploads[0].subtitles, 'ep01.vtt')
  assert.equal(uploads[1].subtitles, 'ep02.srt')
  assert.equal(uploads[0].options.min_clip_sec, 3)
  assert.equal(uploads[0].options.total_duration_sec, null)
  assert.equal(uploads[0].options.instruction, '保留剧情反转与必要铺垫')
  assert.equal(await evaluate('document.querySelectorAll(".subtitle-intake li").length'), 0)
  await evaluate('document.querySelector(".task-options").open = false')
  await capture('11-batch-dark')
  const running = jobs.find(record => record.job_id === uploads[0].id)
  running.status = 'processing'
  running.progress = { ...analysis, scan_coverage: 0 }
  running.workflow = { ...running.workflow, completed_page_count: 0, next_scan_range: { start_sec: 0, end_sec: 30 } }
  await until('document.querySelector(".progress-heading b")?.textContent === "正在提取高光"', 'processing state from polling')
  await evaluate('document.querySelector(".theme-toggle").click()')
  await capture('13-processing-light')
  window.setSize(2560, 1410)
  await click('辅助感知')
  window.setSize(1080, 700)
  assert.equal(await evaluate('document.querySelector("main").scrollWidth > document.querySelector("main").clientWidth'), false)
  assert.equal(await evaluate('[...document.querySelectorAll(".workflow-detail-node")].every(node => node.getBoundingClientRect().right <= innerWidth)'), true)
  window.setSize(2560, 1410)
  await capture('21-processing-perception-light')
  await evaluate(`document.querySelector('main').scrollTop += document.querySelector('.workflow-layout').getBoundingClientRect().top - 100`)
  await capture('22-perception-details-light')
  await click('全片检查')
  await capture('23-scan-details-light')
  window.setSize(1480, 940)
  await evaluate('document.querySelector("main").scrollTop = 0')
  await click('任务归档')
  await until('document.querySelectorAll(".job-row").length === 5', 'uploaded tasks in archive')
  assert.match(await evaluate('document.querySelector(".job-list").textContent'), /ep01.mp4/)
  assert.match(await evaluate('document.querySelector(".job-list").textContent'), /ep02.mp4/)
  assert.match(await evaluate('[...document.querySelectorAll(".job-row")].find(row => row.textContent.includes("ep01.mp4")).textContent'), /正在提取/)
  assert.match(await evaluate('[...document.querySelectorAll(".job-row")].find(row => row.textContent.includes("ep02.mp4")).textContent'), /等待分析/)
  await capture('12-archive-light')
  running.status = 'completed'
  running.progress = analysis
  running.result = { ...fixture(running.job_id, 'completed', 'complete').result, highlights: [] }
  await evaluate(`document.querySelector('button[aria-label="打开任务 ep01.mp4"]').click()`)
  await until('!!document.querySelector(".review-grid")', 'automatic review after completion')
  await click('任务归档')
  await attach('.view-header input[type=file]', ['failure.mp4'])
  await until('document.querySelector(".progress-heading b")?.textContent === "上传失败"', 'failed intake retained')
  assert.equal(await evaluate('!!document.querySelector(".workflow-detail-node.failed")'), true)
  await click('任务归档')
  assert.match(await evaluate('[...document.querySelectorAll(".job-row")].find(row => row.textContent.includes("failure.mp4")).textContent'), /上传失败/)
  await click('高光提取')
  assert.equal(await evaluate('document.querySelector(".progress-heading b").textContent'), '上传失败')
  failUpload = false
  await evaluate('document.querySelector(".analysis-resume button").click(); document.querySelector(".analysis-resume button").click()')
  await until('document.querySelector(".progress-heading b")?.textContent === "等待分析"', 'retry accepted')
  assert.equal(uploads.filter(item => item.name === 'failure.mp4').length, 2)
  await attach('.view-header input[type=file]', ['lost-receipt.mp4'])
  await until('document.querySelector(".progress-heading b")?.textContent === "等待分析"', 'recover accepted task after response error')
  assert.equal(await evaluate('!!document.querySelector(".error-banner")'), false)
  await click('任务归档')
  assert.equal(await evaluate('document.querySelectorAll(".job-row").length'), 7)
  await click('高光提取')
  assert.match(await evaluate('document.querySelector(".progress-heading small").textContent'), /lost-receipt.mp4/)
  assert.deepEqual(consoleErrors, [])
  const report = { passed: true, uploads: uploads.length, resumeRequests: resumeCount, minimumWindow: layout, screenshots: outputDir, requests: requests.length }
  await fs.writeFile(path.join(outputDir, 'result.json'), JSON.stringify(report, null, 2))
  await fs.unlink(path.join(outputDir, 'failure.txt')).catch(() => undefined)
  console.log(JSON.stringify(report))
}).then(() => app.quit()).catch((error) => {
  console.error(error)
  require('node:fs').writeFileSync(path.join(outputDir, 'failure.txt'), error.stack || String(error))
  app.exit(1)
})

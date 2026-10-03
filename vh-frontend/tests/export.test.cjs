const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs/promises')
const os = require('node:os')
const path = require('node:path')
const vm = require('node:vm')
const http = require('node:http')
const { execFileSync } = require('node:child_process')
const { createRequire } = require('node:module')

test('clean export preserves reviewed asset byte-for-byte instead of cutting original', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'vh-export-test-'))
  const media = path.join(root, 'reviewed.mp4')
  const output = path.join(root, 'export.mp4')
  execFileSync('ffmpeg', ['-v', 'error', '-f', 'lavfi', '-i', 'color=c=blue:s=64x64:r=10', '-t', '1', '-c:v', 'libx264', media])
  const bytes = await fs.readFile(media)
  const requests = []
  const server = http.createServer((request, response) => {
    requests.push(request.url)
    response.writeHead(200, { 'Content-Type': 'video/mp4', 'Content-Length': bytes.length })
    response.end(request.method === 'HEAD' ? undefined : bytes)
  })
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
  try {
    const filename = path.resolve(__dirname, '../electron/main.cjs')
    const nativeRequire = createRequire(filename)
    const electron = {
      app: { getPath: () => root, whenReady: () => ({ then() {} }), on() {} },
      BrowserWindow: { fromWebContents: () => null },
      dialog: { showSaveDialog: async () => ({ canceled: false, filePath: output }) },
      protocol: { registerSchemesAsPrivileged() {} },
      net: { fetch },
    }
    const modules = {
      electron,
      'ffmpeg-static': 'ffmpeg',
      'ffprobe-static': { path: 'ffprobe' },
      sharp: () => { throw new Error('Unexpected ad rendering during clean export') },
    }
    const context = {
      require: name => modules[name] ?? nativeRequire(name),
      __dirname: path.dirname(filename), process, console, URL, Buffer,
      module: { exports: {} },
    }
    vm.runInNewContext((await fs.readFile(filename, 'utf8')) + '\nmodule.exports = { exportCleanHighlight };', context)
    const result = await context.module.exports.exportCleanHighlight({ sender: {} }, {
      job_id: 'job_exporttest', original_name: 'source.mp4',
      source_url: `http://127.0.0.1:${server.address().port}/original.mp4`,
      highlight: { start_sec: 40, end_sec: 41, clip_url: '/reviewed.mp4', description: 'reviewed' },
    })
    assert.equal(result.canceled, false)
    assert.deepEqual(await fs.readFile(output), bytes)
    assert(requests.length > 0 && requests.every(url => url === '/reviewed.mp4'))
  } finally {
    await new Promise(resolve => server.close(resolve))
    await fs.rm(root, { recursive: true, force: true })
  }
})

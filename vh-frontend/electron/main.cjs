const { app, BrowserWindow, dialog, ipcMain, net, protocol } = require('electron')
const { createWriteStream } = require('node:fs')
const { copyFile, mkdir, mkdtemp, readFile, rename, rm, stat, writeFile } = require('node:fs/promises')
const { randomUUID } = require('node:crypto')
const { spawn } = require('node:child_process')
const path = require('node:path')
const { Readable } = require('node:stream')
const { pipeline } = require('node:stream/promises')
const { pathToFileURL } = require('node:url')
const sharp = require('sharp')

const isDevelopment = process.argv.includes('--dev')
const JOB_ID_PATTERN = /^job_[A-Za-z0-9_-]{8,48}$/
const AD_ASSET_ID_PATTERN = /^ad_[a-f0-9]{16}$/
const VIDEO_EXTENSIONS = new Set(['.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v'])
const IMAGE_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.webp'])
const AD_PRESET_IDS = new Set(['cinema-cliffhanger', 'neon-app', 'velvet-qr'])
const MEDIA_SCHEME = 'vh-media'
const FFMPEG_PATH = require('ffmpeg-static').replace('app.asar', 'app.asar.unpacked')
const FFPROBE_PATH = require('ffprobe-static').path.replace('app.asar', 'app.asar.unpacked')

protocol.registerSchemesAsPrivileged([{
  scheme: MEDIA_SCHEME,
  privileges: { standard: true, secure: true, supportFetchAPI: true, stream: true },
}])

function requireJobId(value) {
  if (typeof value !== 'string' || !JOB_ID_PATTERN.test(value)) throw new Error('无效的任务编号')
  return value
}

function getAdAssetsRoot() {
  return path.join(app.getPath('userData'), 'video-library', 'ad-assets')
}

function getAdAssetsIndexPath() {
  return path.join(getAdAssetsRoot(), 'assets.json')
}

function publicAdAsset(asset) {
  return {
    asset_id: asset.asset_id,
    kind: asset.kind,
    original_name: asset.original_name,
    size_bytes: asset.size_bytes,
    duration_sec: asset.duration_sec ?? null,
    created_at: asset.created_at,
    url: `${MEDIA_SCHEME}://ads/${asset.asset_id}`,
  }
}

async function readAdAssetIndex() {
  try {
    const parsed = JSON.parse(await readFile(getAdAssetsIndexPath(), 'utf8'))
    return Array.isArray(parsed) ? parsed.filter((item) => AD_ASSET_ID_PATTERN.test(item?.asset_id)) : []
  } catch (error) {
    if (error?.code === 'ENOENT') return []
    throw error
  }
}

async function writeAdAssetIndex(assets) {
  await mkdir(getAdAssetsRoot(), { recursive: true })
  const indexPath = getAdAssetsIndexPath()
  const temporaryPath = `${indexPath}.${randomUUID()}.tmp`
  await writeFile(temporaryPath, JSON.stringify(assets, null, 2), 'utf8')
  await rename(temporaryPath, indexPath)
}

function resolveAdAssetPath(asset) {
  const assetsRoot = path.resolve(getAdAssetsRoot())
  const assetPath = path.resolve(assetsRoot, asset.stored_name)
  if (path.dirname(assetPath) !== assetsRoot) throw new Error('广告素材路径无效')
  return assetPath
}

async function getAdAsset(assetId, expectedKind) {
  if (typeof assetId !== 'string' || !AD_ASSET_ID_PATTERN.test(assetId)) {
    throw new Error('广告素材编号无效')
  }
  const asset = (await readAdAssetIndex()).find((item) => item.asset_id === assetId)
  if (!asset || (expectedKind && asset.kind !== expectedKind)) throw new Error('广告素材不存在')
  return asset
}

function runMediaCommand(executable, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(executable, args, { windowsHide: true })
    let stdout = ''
    let stderr = ''
    child.stdout?.on('data', (chunk) => { stdout += chunk.toString() })
    child.stderr?.on('data', (chunk) => {
      stderr = `${stderr}${chunk.toString()}`.slice(-12000)
    })
    child.once('error', (error) => reject(new Error(`无法启动媒体处理器：${error.message}`)))
    child.once('close', (code) => {
      if (code === 0) resolve({ stdout, stderr })
      else reject(new Error(`媒体处理失败（${code ?? 'unknown'}）\n${stderr.trim()}`))
    })
  })
}

async function probeMedia(filePath) {
  const { stdout } = await runMediaCommand(FFPROBE_PATH, [
    '-v', 'error', '-show_streams', '-show_format', '-of', 'json', filePath,
  ])
  const payload = JSON.parse(stdout)
  const video = payload.streams?.find((stream) => stream.codec_type === 'video')
  if (!video) throw new Error('素材中没有可用的视频轨道')
  const rotation = Number(video.side_data_list?.find((item) => Number.isFinite(Number(item.rotation)))?.rotation || video.tags?.rotate || 0)
  const swapDimensions = Math.abs(rotation) % 180 === 90
  const rawWidth = Number(swapDimensions ? video.height : video.width)
  const rawHeight = Number(swapDimensions ? video.width : video.height)
  const width = Math.max(2, Math.floor(rawWidth / 2) * 2)
  const height = Math.max(2, Math.floor(rawHeight / 2) * 2)
  const duration = Number(video.duration || payload.format?.duration || 0)
  return {
    width,
    height,
    duration_sec: Number.isFinite(duration) ? duration : 0,
    has_audio: payload.streams.some((stream) => stream.codec_type === 'audio'),
  }
}

async function importAdAsset(input) {
  const kind = input?.kind === 'image' ? 'image' : input?.kind === 'video' ? 'video' : null
  const sourcePath = typeof input?.sourcePath === 'string' ? path.resolve(input.sourcePath) : ''
  if (!kind || !path.isAbsolute(sourcePath)) throw new Error('广告素材参数无效')
  const originalName = path.basename(typeof input?.originalName === 'string' ? input.originalName : '')
  const extension = path.extname(originalName).toLowerCase()
  const allowed = kind === 'video' ? VIDEO_EXTENSIONS : IMAGE_EXTENSIONS
  if (!allowed.has(extension)) {
    throw new Error(kind === 'video' ? '请选择 MP4、MOV、MKV、WEBM、AVI 或 M4V 视频' : '请选择 PNG、JPG 或 WEBP 图片')
  }
  const sourceStats = await stat(sourcePath)
  if (!sourceStats.isFile()) throw new Error('选择的广告素材不存在')
  if (sourceStats.size > (kind === 'video' ? 1024 ** 3 : 20 * 1024 ** 2)) {
    throw new Error(kind === 'video' ? '广告视频不能超过 1 GB' : '二维码图片不能超过 20 MB')
  }

  const assetId = `ad_${randomUUID().replace(/-/g, '').slice(0, 16)}`
  const storedName = `${assetId}${extension}`
  await mkdir(getAdAssetsRoot(), { recursive: true })
  const targetPath = path.join(getAdAssetsRoot(), storedName)
  await copyFile(sourcePath, targetPath)
  try {
    const media = kind === 'video' ? await probeMedia(targetPath) : null
    if (kind === 'video' && (!media.duration_sec || media.duration_sec > 60)) {
      throw new Error('广告视频时长需在 0–60 秒之间')
    }
    const asset = {
      asset_id: assetId,
      kind,
      original_name: originalName,
      stored_name: storedName,
      size_bytes: sourceStats.size,
      duration_sec: media?.duration_sec ?? null,
      created_at: new Date().toISOString(),
    }
    const assets = await readAdAssetIndex()
    assets.unshift(asset)
    await writeAdAssetIndex(assets.slice(0, 100))
    return publicAdAsset(asset)
  } catch (error) {
    await rm(targetPath, { force: true })
    throw error
  }
}

async function listAdAssets() {
  const assets = await readAdAssetIndex()
  const available = []
  for (const asset of assets) {
    try {
      const assetStats = await stat(resolveAdAssetPath(asset))
      if (assetStats.isFile()) available.push(asset)
    } catch {
      // Ignore stale entries; deletion and interrupted imports must not break the studio.
    }
  }
  if (available.length !== assets.length) await writeAdAssetIndex(available)
  return available.map(publicAdAsset)
}

async function deleteAdAsset(assetId) {
  const asset = await getAdAsset(assetId)
  const assets = await readAdAssetIndex()
  await rm(resolveAdAssetPath(asset), { force: true })
  await writeAdAssetIndex(assets.filter((item) => item.asset_id !== assetId))
}

function escapeXml(value) {
  return String(value).replace(/[<>&'\"]/g, (character) => ({
    '<': '&lt;', '>': '&gt;', '&': '&amp;', "'": '&apos;', '"': '&quot;',
  })[character])
}

function qrPlaceholderSvg(x, y, size) {
  const cells = 17
  const cell = size / cells
  const blocks = []
  const finder = (offsetX, offsetY) => {
    blocks.push(`<rect x="${x + offsetX * cell}" y="${y + offsetY * cell}" width="${7 * cell}" height="${7 * cell}" rx="${cell}" fill="#101010"/>`)
    blocks.push(`<rect x="${x + (offsetX + 1) * cell}" y="${y + (offsetY + 1) * cell}" width="${5 * cell}" height="${5 * cell}" rx="${cell / 2}" fill="#fff"/>`)
    blocks.push(`<rect x="${x + (offsetX + 2) * cell}" y="${y + (offsetY + 2) * cell}" width="${3 * cell}" height="${3 * cell}" fill="#101010"/>`)
  }
  finder(0, 0)
  finder(10, 0)
  finder(0, 10)
  for (let row = 0; row < cells; row += 1) {
    for (let column = 0; column < cells; column += 1) {
      const insideFinder = (row < 7 && column < 7) || (row < 7 && column > 9) || (row > 9 && column < 7)
      if (!insideFinder && ((row * 7 + column * 11 + row * column) % 5 < 2)) {
        blocks.push(`<rect x="${x + column * cell}" y="${y + row * cell}" width="${cell * .82}" height="${cell * .82}" fill="#101010"/>`)
      }
    }
  }
  return blocks.join('')
}

async function qrImageMarkup(assetId, x, y, size) {
  if (!assetId) return qrPlaceholderSvg(x, y, size)
  const asset = await getAdAsset(assetId, 'image')
  const image = await sharp(resolveAdAssetPath(asset))
    .resize(Math.round(size), Math.round(size), { fit: 'cover' })
    .png()
    .toBuffer()
  const dataUrl = `data:image/png;base64,${image.toString('base64')}`
  return `<image href="${dataUrl}" x="${x}" y="${y}" width="${size}" height="${size}" preserveAspectRatio="xMidYMid slice"/>`
}

async function renderSvgToPng(svg, width, height, targetPath) {
  await sharp(Buffer.from(svg), { density: 144 })
    .resize(width, height, { fit: 'fill' })
    .png()
    .toFile(targetPath)
}

async function createPresetCard(input, width, height, targetPath) {
  const presetId = input?.preset_id
  if (!AD_PRESET_IDS.has(presetId)) throw new Error('广告预设无效')
  const title = escapeXml(String(input?.title || '').trim().slice(0, 20) || '点击下方看全集')
  const subtitle = escapeXml(String(input?.subtitle || '').trim().slice(0, 36) || '精彩继续 · 立即解锁')
  const shortSide = Math.min(width, height)
  const titleSize = Math.round(shortSide * .085)
  const subtitleSize = Math.round(shortSide * .031)
  const pad = Math.round(shortSide * .09)
  const qrSize = Math.round(shortSide * .26)
  const centerX = width / 2
  const centerY = height / 2
  let content

  if (presetId === 'cinema-cliffhanger') {
    content = `
      <rect width="${width}" height="${height}" fill="#090806"/>
      <path d="M0 ${height * .18} L${width} 0 L${width} ${height * .24} L0 ${height * .42}Z" fill="#ff5d45" opacity=".92"/>
      <path d="M0 ${height * .79} L${width} ${height * .63} L${width} ${height} L0 ${height}Z" fill="#f2bd45" opacity=".13"/>
      <g opacity=".2" stroke="#f2bd45"><path d="M${pad} 0V${height}"/><path d="M${width - pad} 0V${height}"/></g>
      <text x="${pad}" y="${height * .12}" fill="#111" font-size="${subtitleSize}" font-weight="800" letter-spacing="${subtitleSize * .18}">DRAMA EXCLUSIVE</text>
      <circle cx="${centerX}" cy="${centerY - titleSize * 1.5}" r="${shortSide * .065}" fill="#ff5d45"/>
      <path d="M${centerX - shortSide * .018} ${centerY - titleSize * 1.54} L${centerX + shortSide * .025} ${centerY - titleSize * 1.5} L${centerX - shortSide * .018} ${centerY - titleSize * 1.19}Z" fill="#fff"/>
      <text x="${centerX}" y="${centerY}" text-anchor="middle" fill="#fff8ec" font-size="${titleSize}" font-weight="900" font-family="Microsoft YaHei, PingFang SC, sans-serif">${title}</text>
      <text x="${centerX}" y="${centerY + titleSize * .72}" text-anchor="middle" fill="#f2bd45" font-size="${subtitleSize}" font-weight="600" letter-spacing="${subtitleSize * .08}" font-family="Microsoft YaHei, PingFang SC, sans-serif">${subtitle}</text>
      <rect x="${centerX - shortSide * .23}" y="${centerY + titleSize * 1.28}" width="${shortSide * .46}" height="${shortSide * .09}" rx="${shortSide * .045}" fill="#ff5d45"/>
      <text x="${centerX}" y="${centerY + titleSize * 1.82}" text-anchor="middle" fill="#fff" font-size="${subtitleSize * .92}" font-weight="800">立即观看  →</text>`
  } else if (presetId === 'neon-app') {
    const phoneWidth = shortSide * .34
    const phoneHeight = shortSide * .58
    content = `
      <rect width="${width}" height="${height}" fill="#06100a"/>
      <circle cx="${width * .12}" cy="${height * .12}" r="${shortSide * .4}" fill="#62ff1f" opacity=".09"/>
      <circle cx="${width * .88}" cy="${height * .78}" r="${shortSide * .48}" fill="#20d9ff" opacity=".08"/>
      <path d="M0 ${height * .31} H${width}" stroke="#62ff1f" stroke-width="2" stroke-dasharray="12 18" opacity=".4"/>
      <g transform="translate(${centerX - phoneWidth / 2} ${centerY - phoneHeight * .92}) rotate(-6 ${phoneWidth / 2} ${phoneHeight / 2})">
        <rect width="${phoneWidth}" height="${phoneHeight}" rx="${phoneWidth * .1}" fill="#101713" stroke="#62ff1f" stroke-width="${Math.max(3, shortSide * .006)}"/>
        <rect x="${phoneWidth * .08}" y="${phoneWidth * .12}" width="${phoneWidth * .84}" height="${phoneHeight * .58}" rx="${phoneWidth * .04}" fill="#ff735f"/>
        <path d="M${phoneWidth * .18} ${phoneHeight * .54} L${phoneWidth * .8} ${phoneHeight * .2}" stroke="#ffe9a9" stroke-width="${shortSide * .025}" opacity=".75"/>
        <rect x="${phoneWidth * .27}" y="${phoneHeight * .77}" width="${phoneWidth * .46}" height="${phoneHeight * .045}" rx="9" fill="#62ff1f"/>
      </g>
      <rect x="${pad}" y="${height * .09}" width="${shortSide * .29}" height="${shortSide * .065}" rx="${shortSide * .032}" fill="#62ff1f"/>
      <text x="${pad + shortSide * .145}" y="${height * .09 + shortSide * .043}" text-anchor="middle" fill="#06100a" font-size="${subtitleSize * .72}" font-weight="900">NEW USER GIFT</text>
      <text x="${centerX}" y="${centerY + titleSize * .9}" text-anchor="middle" fill="#f5fff1" font-size="${titleSize}" font-weight="900" font-family="Microsoft YaHei, PingFang SC, sans-serif">${title}</text>
      <text x="${centerX}" y="${centerY + titleSize * 1.62}" text-anchor="middle" fill="#62ff1f" font-size="${subtitleSize}" font-weight="650" font-family="Microsoft YaHei, PingFang SC, sans-serif">${subtitle}</text>
      <rect x="${centerX - shortSide * .25}" y="${centerY + titleSize * 2.08}" width="${shortSide * .5}" height="${shortSide * .095}" rx="${shortSide * .018}" fill="#62ff1f"/>
      <text x="${centerX}" y="${centerY + titleSize * 2.64}" text-anchor="middle" fill="#06100a" font-size="${subtitleSize}" font-weight="900">免费下载短剧 APP</text>`
  } else {
    const qrX = centerX - qrSize / 2
    const qrY = centerY - qrSize * .88
    const qrMarkup = await qrImageMarkup(input?.qr_asset_id, qrX, qrY, qrSize)
    content = `
      <defs><linearGradient id="velvet" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#361038"/><stop offset=".55" stop-color="#7b234c"/><stop offset="1" stop-color="#f08e68"/></linearGradient></defs>
      <rect width="${width}" height="${height}" fill="url(#velvet)"/>
      <circle cx="${width * .1}" cy="${height * .12}" r="${shortSide * .32}" fill="#ffd9b8" opacity=".13"/>
      <circle cx="${width * .92}" cy="${height * .82}" r="${shortSide * .45}" fill="#210b2c" opacity=".25"/>
      <text x="${centerX}" y="${height * .12}" text-anchor="middle" fill="#ffd8b6" font-size="${subtitleSize * .76}" font-weight="800" letter-spacing="${subtitleSize * .17}">SCAN TO CONTINUE</text>
      <rect x="${qrX - shortSide * .025}" y="${qrY - shortSide * .025}" width="${qrSize + shortSide * .05}" height="${qrSize + shortSide * .05}" rx="${shortSide * .035}" fill="#fff"/>
      ${qrMarkup}
      <text x="${centerX}" y="${centerY + titleSize * 1.2}" text-anchor="middle" fill="#fff7ed" font-size="${titleSize}" font-weight="900" font-family="Microsoft YaHei, PingFang SC, sans-serif">${title}</text>
      <text x="${centerX}" y="${centerY + titleSize * 1.9}" text-anchor="middle" fill="#ffd8b6" font-size="${subtitleSize}" font-weight="650" font-family="Microsoft YaHei, PingFang SC, sans-serif">${subtitle}</text>
      <path d="M${centerX - shortSide * .22} ${centerY + titleSize * 2.4} H${centerX + shortSide * .22}" stroke="#ffd8b6" stroke-width="2"/>
      <text x="${centerX}" y="${centerY + titleSize * 2.86}" text-anchor="middle" fill="#fff" font-size="${subtitleSize * .77}" font-weight="800" letter-spacing="${subtitleSize * .12}">长按识别 · 全集立即看</text>`
  }

  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">${content}</svg>`
  await renderSvgToPng(svg, width, height, targetPath)
}

function normalizationFilter(width, height) {
  return `scale=${width}:${height}:force_original_aspect_ratio=decrease,pad=${width}:${height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,format=yuv420p,setpts=PTS-STARTPTS`
}

async function transcodeSegment({ inputPath, outputPath, width, height, startSec = 0, durationSec, hasAudio }) {
  const args = ['-hide_banner', '-loglevel', 'error', '-y']
  if (startSec > 0) args.push('-ss', String(startSec))
  args.push('-t', String(durationSec), '-i', inputPath)
  if (!hasAudio) args.push('-f', 'lavfi', '-t', String(durationSec), '-i', 'anullsrc=channel_layout=stereo:sample_rate=48000')
  const audioInput = hasAudio ? '[0:a:0]' : '[1:a:0]'
  args.push(
    '-filter_complex', `[0:v:0]${normalizationFilter(width, height)}[v];${audioInput}aresample=48000:async=1:first_pts=0,apad=pad_dur=${durationSec},atrim=duration=${durationSec},asetpts=PTS-STARTPTS[a]`,
    '-map', '[v]', '-map', '[a]', '-c:v', 'libx264', '-preset', 'medium', '-crf', '20',
    '-c:a', 'aac', '-b:a', '160k', '-ar', '48000', '-ac', '2', '-movflags', '+faststart', outputPath,
  )
  await runMediaCommand(FFMPEG_PATH, args)
}

async function renderPresetSegment({ assignment, outputPath, cardPath, width, height }) {
  const durationSec = Math.min(8, Math.max(2, Number(assignment.duration_sec) || 4))
  await createPresetCard(assignment, width, height, cardPath)
  await runMediaCommand(FFMPEG_PATH, [
    '-hide_banner', '-loglevel', 'error', '-y', '-loop', '1', '-framerate', '30', '-t', String(durationSec), '-i', cardPath,
    '-f', 'lavfi', '-t', String(durationSec), '-i', 'anullsrc=channel_layout=stereo:sample_rate=48000',
    '-vf', normalizationFilter(width, height), '-map', '0:v:0', '-map', '1:a:0',
    '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-c:a', 'aac', '-b:a', '160k',
    '-ar', '48000', '-ac', '2', '-shortest', '-movflags', '+faststart', outputPath,
  ])
  return durationSec
}

async function downloadSource(sourceUrl, targetPath) {
  let parsed
  try {
    parsed = new URL(sourceUrl)
  } catch {
    throw new Error('原片地址无效')
  }
  if (!['http:', 'https:', `${MEDIA_SCHEME}:`].includes(parsed.protocol)) throw new Error('不支持的原片地址')
  const response = await net.fetch(sourceUrl, { cache: 'no-store' })
  if (!response.ok || !response.body) throw new Error(`原片下载失败（${response.status}）`)
  await pipeline(Readable.fromWeb(response.body), createWriteStream(targetPath))
}

function safeExportName(value) {
  const base = path.basename(String(value || 'highlight'), path.extname(String(value || '')))
  return (base.replace(/[<>:"/\\|?*\u0000-\u001F]/g, '_').trim() || 'highlight').slice(0, 80)
}

async function resolveExportSource(sourceUrl, downloadedSourcePath) {
  let sourceProtocol
  try {
    sourceProtocol = new URL(sourceUrl).protocol
  } catch {
    throw new Error('原片地址无效')
  }
  let sourceInput = ['http:', 'https:'].includes(sourceProtocol) ? sourceUrl : downloadedSourcePath
  if (sourceInput === downloadedSourcePath) await downloadSource(sourceUrl, downloadedSourcePath)
  let sourceMedia
  try {
    sourceMedia = await probeMedia(sourceInput)
  } catch (probeError) {
    if (sourceInput === downloadedSourcePath) throw probeError
    await downloadSource(sourceUrl, downloadedSourcePath)
    sourceInput = downloadedSourcePath
    sourceMedia = await probeMedia(sourceInput)
  }
  return { sourceInput, sourceMedia }
}

async function exportCleanHighlight(event, input) {
  const jobId = requireJobId(input?.job_id)
  const highlight = input?.highlight
  const frozen = Boolean(highlight?.clip_url)
  const sourceUrl = frozen ? new URL(highlight.clip_url, input.source_url).toString() : input.source_url
  const startSec = frozen ? 0 : Number(highlight?.start_sec)
  const endSec = frozen ? Number(highlight.end_sec) - Number(highlight.start_sec) : Number(highlight?.end_sec)
  if (!highlight || !Number.isFinite(startSec) || !Number.isFinite(endSec) || startSec < 0 || endSec <= startSec) {
    throw new Error('高光片段时间范围无效')
  }
  const defaultName = `${safeExportName(input?.original_name)}-${safeExportName(highlight?.description)}-高光.mp4`
  const parentWindow = BrowserWindow.fromWebContents(event.sender)
  const selection = await dialog.showSaveDialog(parentWindow, {
    title: '导出高光视频',
    defaultPath: path.join(app.getPath('downloads'), defaultName),
    filters: [{ name: 'MP4 视频', extensions: ['mp4'] }],
    properties: ['showOverwriteConfirmation', 'createDirectory'],
  })
  if (selection.canceled || !selection.filePath) return { canceled: true }

  const temporaryRoot = await mkdtemp(path.join(app.getPath('temp'), 'frame-highlight-export-'))
  try {
    const downloadedSourcePath = path.join(temporaryRoot, 'source.media')
    const { sourceInput, sourceMedia } = await resolveExportSource(sourceUrl, downloadedSourcePath)
    if (frozen) {
      await downloadSource(sourceUrl, selection.filePath)
      return { canceled: false, output_path: selection.filePath, duration_sec: sourceMedia.duration_sec, job_id: jobId }
    }
    const clipDuration = Math.min(endSec, sourceMedia.duration_sec || endSec) - startSec
    if (clipDuration <= 0) throw new Error('高光片段超出原片时长')
    await transcodeSegment({
      inputPath: sourceInput,
      outputPath: selection.filePath,
      width: sourceMedia.width,
      height: sourceMedia.height,
      startSec,
      durationSec: clipDuration,
      hasAudio: sourceMedia.has_audio,
    })
    return {
      canceled: false,
      output_path: selection.filePath,
      duration_sec: Number(clipDuration.toFixed(3)),
      job_id: jobId,
    }
  } catch (error) {
    await rm(selection.filePath, { force: true }).catch(() => undefined)
    throw error
  } finally {
    await rm(temporaryRoot, { recursive: true, force: true })
  }
}

async function exportHighlight(event, input) {
  const jobId = requireJobId(input?.job_id)
  const highlight = input?.highlight
  const frozen = Boolean(highlight?.clip_url)
  const sourceUrl = frozen ? new URL(highlight.clip_url, input.source_url).toString() : input.source_url
  const startSec = frozen ? 0 : Number(highlight?.start_sec)
  const endSec = frozen ? Number(highlight.end_sec) - Number(highlight.start_sec) : Number(highlight?.end_sec)
  if (!highlight || !Number.isFinite(startSec) || !Number.isFinite(endSec) || startSec < 0 || endSec <= startSec) {
    throw new Error('高光片段时间范围无效')
  }
  const assignment = input?.assignment
  if (!assignment || !['preset', 'video'].includes(assignment.kind)) throw new Error('请选择广告素材')
  if (assignment.kind === 'preset' && !AD_PRESET_IDS.has(assignment.preset_id)) throw new Error('广告预设无效')
  const defaultName = `${safeExportName(input?.original_name)}-${safeExportName(highlight?.description)}-含广告.mp4`
  const parentWindow = BrowserWindow.fromWebContents(event.sender)
  const selection = await dialog.showSaveDialog(parentWindow, {
    title: '导出带广告的高光视频',
    defaultPath: path.join(app.getPath('downloads'), defaultName),
    filters: [{ name: 'MP4 视频', extensions: ['mp4'] }],
    properties: ['showOverwriteConfirmation', 'createDirectory'],
  })
  if (selection.canceled || !selection.filePath) return { canceled: true }

  const temporaryRoot = await mkdtemp(path.join(app.getPath('temp'), 'frame-ad-export-'))
  try {
    const downloadedSourcePath = path.join(temporaryRoot, 'source.media')
    const highlightPath = path.join(temporaryRoot, 'highlight.mp4')
    const adPath = path.join(temporaryRoot, 'advertisement.mp4')
    const cardPath = path.join(temporaryRoot, 'card.png')
    const concatPath = path.join(temporaryRoot, 'concat.txt')
    const { sourceInput, sourceMedia } = await resolveExportSource(sourceUrl, downloadedSourcePath)
    const clipDuration = Math.min(endSec, sourceMedia.duration_sec || endSec) - startSec
    if (clipDuration <= 0) throw new Error('高光片段超出原片时长')
    await transcodeSegment({
      inputPath: sourceInput,
      outputPath: highlightPath,
      width: sourceMedia.width,
      height: sourceMedia.height,
      startSec,
      durationSec: clipDuration,
      hasAudio: sourceMedia.has_audio,
    })

    let adDuration
    if (assignment.kind === 'video') {
      const asset = await getAdAsset(assignment.asset_id, 'video')
      const assetPath = resolveAdAssetPath(asset)
      const assetMedia = await probeMedia(assetPath)
      adDuration = Math.min(60, assetMedia.duration_sec)
      await transcodeSegment({
        inputPath: assetPath,
        outputPath: adPath,
        width: sourceMedia.width,
        height: sourceMedia.height,
        durationSec: adDuration,
        hasAudio: assetMedia.has_audio,
      })
    } else {
      adDuration = await renderPresetSegment({ assignment, outputPath: adPath, cardPath, width: sourceMedia.width, height: sourceMedia.height })
    }

    await writeFile(concatPath, "file 'highlight.mp4'\nfile 'advertisement.mp4'\n", 'utf8')
    await runMediaCommand(FFMPEG_PATH, [
      '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', concatPath,
      '-c', 'copy', '-movflags', '+faststart', selection.filePath,
    ])
    return {
      canceled: false,
      output_path: selection.filePath,
      duration_sec: Number((clipDuration + adDuration).toFixed(3)),
      job_id: jobId,
    }
  } catch (error) {
    await rm(selection.filePath, { force: true }).catch(() => undefined)
    throw error
  } finally {
    await rm(temporaryRoot, { recursive: true, force: true })
  }
}

function registerIpcHandlers() {
  ipcMain.on('window:minimize', (event) => BrowserWindow.fromWebContents(event.sender)?.minimize())
  ipcMain.on('window:toggle-maximize', (event) => {
    const window = BrowserWindow.fromWebContents(event.sender)
    if (!window) return
    if (window.isMaximized()) window.unmaximize()
    else window.maximize()
  })
  ipcMain.on('window:close', (event) => BrowserWindow.fromWebContents(event.sender)?.close())
  ipcMain.handle('ads:import-asset', (_event, input) => importAdAsset(input))
  ipcMain.handle('ads:list-assets', () => listAdAssets())
  ipcMain.handle('ads:delete-asset', (_event, assetId) => deleteAdAsset(assetId))
  ipcMain.handle('ads:export-highlight', (event, input) => exportHighlight(event, input))
  ipcMain.handle('highlights:export-clean', (event, input) => exportCleanHighlight(event, input))
}

function registerMediaProtocol() {
  protocol.handle(MEDIA_SCHEME, async (request) => {
    try {
      const url = new URL(request.url)
      const parts = url.pathname.split('/').filter(Boolean)
      if (url.host === 'ads' && parts.length === 1) {
        const asset = await getAdAsset(parts[0])
        return net.fetch(pathToFileURL(resolveAdAssetPath(asset)).toString(), { headers: request.headers })
      }
      return new Response('Not found', { status: 404 })
    } catch {
      return new Response('Not found', { status: 404 })
    }
  })
}

function createWindow() {
  const window = new BrowserWindow({
    width: 1480,
    height: 940,
    minWidth: 1080,
    minHeight: 700,
    backgroundColor: '#0b0d0d',
    titleBarStyle: 'hidden',
    titleBarOverlay: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  })

  if (isDevelopment) window.loadURL('http://127.0.0.1:5173')
  else window.loadFile(path.join(__dirname, '..', 'dist', 'index.html'))
}

app.whenReady().then(async () => {
  registerMediaProtocol()
  registerIpcHandlers()
  createWindow()
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

// SPDX-License-Identifier: AGPL-3.0-only
// Her node tipinin çalıştırıcısı.
import fs from 'node:fs'
import path from 'node:path'
import type { Artifact, AssetSpec, Bible, GeneratedPrompt, Job, MeshyResource, ParamValue, QAReport } from '../../shared/types'
import { assetBaseName, parseSizeMeters } from '../../shared/templates'
import { runBlender, runBlenderAnim } from '../integrations/blender'
import { kimodoGenerate, type KimodoRequest } from '../integrations/kimodo'
import { buildKimodoRequest, type KimodoSpec } from '../anim/motion'
import { audioScore, decodeWav, judgeAudio, measureAudio } from '../../shared/audioQa'
import { parseBvh } from '../../shared/bvh'
import { judgeMotion, measureMotion } from '../../shared/motionQa'
import { loadMotion, motionInfo, saveMotion, type StoredMotion } from '../anim/store'
import { execInUe } from '../integrations/ue'
import { generatePrompt, reviewAsset } from '../providers/claude'
import { comfyRun } from '../providers/comfyui'
import { soundEffect } from '../providers/elevenlabs'
import { createTask, downloadModelTask, waitTask, type MeshyTask } from '../providers/meshy'
import { getSecret, getSettings } from '../settings'
import { downloadTo, ensureDir, fileToDataUri, resourcePath, truncateWords } from '../util'

export interface ExecCtx {
  job: Job
  nodeId: string
  asset: AssetSpec
  /** Kit üyesi ise ana asset (ör. animasyon → karakter) */
  parent?: AssetSpec
  bible: Bible
  projectDir: string
  dir: string
  feedback: string[]
  signal: AbortSignal
  log: (m: string) => void
  progress: (p: number) => void
  cost: (c: { usd?: number; credits?: number; chars?: number }) => void
  warn: (m: string) => void
}

export interface ExecResult {
  outputs: Record<string, Artifact>
  qa?: { pass: boolean; feedback: string }
}

type Params = Record<string, ParamValue>
type Inputs = Record<string, Artifact | undefined>
export type Executor = (ctx: ExecCtx, params: Params, inputs: Inputs) => Promise<ExecResult>

const str = (v: ParamValue | undefined, d = '') => (v === undefined ? d : String(v))
const num = (v: ParamValue | undefined, d = 0) => (v === undefined || v === '' ? d : Number(v))
const bool = (v: ParamValue | undefined, d = false) => (v === undefined ? d : v === true || v === 'true')

function need<K extends Artifact['kind']>(inputs: Inputs, port: string, kind: K): Extract<Artifact, { kind: K }> {
  const a = inputs[port]
  if (!a) throw new Error(`"${port}" girişi bağlı değil veya boş`)
  if (a.kind !== kind) throw new Error(`"${port}" girişi ${kind} bekliyor, ${a.kind} geldi`)
  return a as Extract<Artifact, { kind: K }>
}

function promptOf(inputs: Inputs, port = 'prompt'): GeneratedPrompt | undefined {
  const a = inputs[port]
  return a?.kind === 'prompt' ? a.prompt : undefined
}

/** Asset → Proje Kitabı kategorisi → varsayılan sırasıyla üçgen bütçesi */
function triBudget(ctx: ExecCtx, explicit: number): number {
  if (explicit > 0) return explicit
  if (ctx.asset.triBudget > 0) return ctx.asset.triBudget
  const cat = ctx.bible.categories.find((c) => c.code === ctx.asset.category)
  if (cat?.triBudget) return cat.triBudget
  return ctx.bible.technical.defaultTriBudget
}

function localPath(ctx: ExecCtx, v: ParamValue | undefined): string {
  const file = str(v).trim().replace(/^"|"$/g, '').replaceAll('{id}', ctx.asset.id)
  if (!file) throw new Error('Dosya yolu boş')
  if (!fs.existsSync(file)) throw new Error(`Dosya bulunamadı: ${file}`)
  return file
}

function poseFor(param: string, prompt?: GeneratedPrompt): string | undefined {
  if (param === 'auto') return prompt && prompt.pose !== 'none' ? prompt.pose : undefined
  return param || undefined
}

async function meshyRun(ctx: ExecCtx, resource: MeshyResource, body: Record<string, unknown>, from: number, to: number): Promise<MeshyTask> {
  const id = await createTask(resource, body, ctx.signal)
  ctx.log(`Meshy ${resource} görevi: ${id}`)
  let lastNote = ''
  const task = await waitTask(
    resource,
    id,
    (p, note) => {
      ctx.progress(from + ((to - from) * p) / 100)
      if (note !== lastNote) ctx.log(`Meshy: ${note}`)
      lastNote = note
    },
    ctx.signal
  )
  ctx.cost({ credits: task.consumed_credits ?? 0 })
  return task
}

function modelSource(m: Extract<Artifact, { kind: 'model' }>): Record<string, unknown> {
  if (m.meshyTaskId) return { input_task_id: m.meshyTaskId }
  const file = m.glb ?? m.file
  const size = fs.statSync(file).size
  if (size > 95 * 1024 * 1024) throw new Error('Model 100 MB üstünde; Meshy data URI sınırı aşılıyor')
  return { model_url: fileToDataUri(file) }
}

function clampPoly(model: string, n: number): number {
  if (model === 'meshy-t2') return Math.min(15000, Math.max(100, n))
  if (model === 'meshy-t2.5') return Math.min(30000, Math.max(100, n))
  return Math.min(300000, Math.max(100, n))
}

async function modelArtifact(ctx: ExecCtx, task: MeshyTask, resource: MeshyResource, textured: boolean): Promise<Artifact> {
  const base = assetBaseName(ctx.asset)
  const d = await downloadModelTask(task, ctx.dir, base, ctx.signal)
  const file = d.glb ?? d.fbx
  if (!file) throw new Error('Meshy görevi model dosyası döndürmedi')
  return { kind: 'model', file, glb: d.glb, meshyTaskId: task.id, meshyResource: resource, thumbnail: d.thumbnail, textures: d.textures, textured }
}

/** Düğümdeki workflow → Ayarlardaki varsayılan → yerleşik SDXL şablonu */
function comfyWorkflowPath(ctx: ExecCtx, p: Params): string {
  const own = str(p.workflow).trim().replace(/^"|"$/g, '')
  if (own) return own
  return getSettings().comfyWorkflow || resourcePath('comfyui', 'default_workflow_api.json')
}

/** -1 = her çalıştırmada yeni; sabit seed kalite döngüsünde her denemede kaydırılır */
function comfySeed(ctx: ExecCtx, p: Params): number {
  const fixed = num(p.seed, -1)
  return fixed >= 0 ? fixed + ctx.feedback.length * 7919 : Math.floor(Math.random() * 2 ** 31)
}

// ---------------------------------------------------------------------------

function writeReport(ctx: ExecCtx, report: QAReport): void {
  fs.writeFileSync(path.join(ctx.dir, `qa_report_${Date.now()}.json`), JSON.stringify(report, null, 2))
  ctx.log(`Puan ${report.score}/10 → ${report.pass ? 'GEÇTİ' : 'KALDI'}: ${report.summary}`)
  for (const t of report.technical ?? []) ctx.log(`  · ${t}`)
  for (const i of report.issues) ctx.log(`  • ${i}`)
}

/** Animasyon: BVH'den ayak kayması, zemine girme, titreme + sistem klibinin hedef hız/yön/dönüşü */
function reviewAnimation(ctx: ExecCtx, subject: Extract<Artifact, { kind: 'animation' }>, minScore: number): ExecResult {
  let spec: KimodoSpec | null = null
  try {
    spec = ctx.asset.extra['KimodoSpec'] ? (JSON.parse(ctx.asset.extra['KimodoSpec']) as KimodoSpec) : null
  } catch {
    spec = null
  }
  if (!subject.bvh || !fs.existsSync(subject.bvh)) {
    ctx.log('Animasyonun BVH dosyası yok (yalnızca FBX); hareket ölçülemedi, kontrol atlandı')
    return { outputs: { subject }, qa: { pass: true, feedback: '' } }
  }
  const measured = subject.motion?.measured
  const metrics = measureMotion(parseBvh(fs.readFileSync(subject.bvh, 'utf8')))
  const qa = judgeMotion(
    metrics,
    {
      ...(spec && measured
        ? {
            speed: Math.round(((spec.speed + (spec.speedEnd ?? spec.speed)) / 2) * 100) / 100,
            measuredSpeed: measured.speed,
            direction: spec.speed > 0.2 ? spec.moveDir : undefined,
            measuredDirection: measured.direction,
            turn: spec.turn,
            measuredTurn: measured.turn
          }
        : {}),
      loopCost: subject.motion?.loop?.cost,
      stationary: spec ? spec.role === 'idle' || spec.role === 'aim' : false
    },
    minScore
  )
  const report: QAReport = {
    score: qa.score,
    pass: qa.pass,
    issues: qa.issues,
    strengths: qa.issues.length ? [] : ['Ayaklar yerde sabit, hareket akıcı'],
    prompt_fix: qa.promptFix,
    summary: qa.issues.length ? qa.issues[0] : 'Hareket teknik kontrolleri geçti',
    technical: qa.technical
  }
  writeReport(ctx, report)
  return { outputs: { subject, report: { kind: 'report', report } }, qa: { pass: qa.pass, feedback: qa.promptFix } }
}

/** Ses: her dosyada sessizlik, kırpılma, seviye, baş boşluğu, süre ve döngü dikişi */
function reviewAudio(ctx: ExecCtx, subject: Extract<Artifact, { kind: 'audio' | 'audioset' }>, minScore: number): ExecResult {
  const files = subject.kind === 'audio' ? subject.files : subject.layers.map((l) => l.file)
  const technical: string[] = []
  const issues: string[] = []
  const fixes = new Set<string>()
  const scores: number[] = []
  let skipped = 0
  for (const f of files) {
    const pcm = f.toLowerCase().endsWith('.wav') ? decodeWav(fs.readFileSync(f)) : null
    if (!pcm) {
      skipped++
      continue
    }
    const r = judgeAudio(path.basename(f), measureAudio(pcm), subject.target)
    scores.push(r.score)
    technical.push(...r.technical)
    issues.push(...r.issues)
    for (const x of r.fixes) fixes.add(x)
  }
  if (skipped) technical.push(`${skipped} dosya WAV değil (ör. MP3); ölçülmedi. Ayarlar → Proje'den PCM biçimini seçersen hepsi incelenir.`)
  // Varyasyonlu SFX'te en iyi dosya, motor katmanlarında en kötü katman belirleyicidir (hepsi kullanılır).
  // Ölçüm yapılamadıysa QA geçemez; ölçülmemiş bir dosyaya 10/10 vermek güvenli değildir.
  const { score, pass } = audioScore(scores, subject.kind, minScore)
  if (!scores.length) issues.push('Ses kalite kontrolü yapılamadı: ölçülebilir WAV dosyası yok')
  const report: QAReport = {
    score,
    pass,
    issues,
    strengths: scores.length && !issues.length ? ['Seviye, süre ve dikişler temiz'] : [],
    prompt_fix: [...fixes].join(', '),
    summary: !scores.length ? 'Ölçülebilir WAV dosyası yok' : issues.length ? issues[0] : 'Ses teknik kontrolleri geçti',
    technical
  }
  writeReport(ctx, report)
  return { outputs: { subject, report: { kind: 'report', report } }, qa: { pass, feedback: report.prompt_fix } }
}

const executors: Record<string, Executor> = {
  'asset.input': async (ctx) => ({ outputs: { asset: { kind: 'asset', asset: ctx.asset } } }),

  'model.file': async (ctx, p) => {
    const file = localPath(ctx, p.file)
    const ext = path.extname(file).toLowerCase()
    if (!['.glb', '.gltf', '.fbx', '.obj'].includes(ext)) throw new Error(`Desteklenmeyen model formatı: ${ext}`)
    ctx.log(`Yerel model: ${file}`)
    return { outputs: { model: { kind: 'model', file, glb: ext === '.glb' ? file : undefined, rigged: bool(p.rigged), textured: true } } }
  },

  'image.file': async (ctx, p) => {
    const file = localPath(ctx, p.file)
    ctx.log(`Yerel görsel: ${file}`)
    return { outputs: { image: { kind: 'image', files: [file] } } }
  },

  'prompt.generate': async (ctx, p) => {
    const { data, usd } = await generatePrompt({
      bible: ctx.bible,
      asset: ctx.asset,
      targetId: str(p.target, 'meshy_text3d'),
      extra: str(p.extra),
      feedback: ctx.feedback
    })
    ctx.cost({ usd })
    ctx.log(`Prompt (${data.prompt.length} kr): ${data.prompt}`)
    if (data.texture_prompt) ctx.log(`Doku promptu: ${data.texture_prompt}`)
    if (data.negative_prompt) ctx.log(`Negatif: ${data.negative_prompt}`)
    ctx.log(`Gerekçe: ${data.rationale}`)
    fs.writeFileSync(path.join(ctx.dir, 'prompt.json'), JSON.stringify(data, null, 2))
    return { outputs: { prompt: { kind: 'prompt', prompt: data } } }
  },

  'image.meshy': async (ctx, p, inputs) => {
    const pr = need(inputs, 'prompt', 'prompt').prompt
    const multi = bool(p.multi_view)
    const pose = poseFor(str(p.pose_mode, 'auto'), pr)
    const task = await meshyRun(
      ctx,
      'text-to-image',
      {
        ai_model: str(p.ai_model, 'nano-banana-2'),
        prompt: pr.prompt,
        ...(multi ? { generate_multi_view: true } : { aspect_ratio: str(p.aspect_ratio, '1:1') }),
        pose_mode: pose,
        remove_background: bool(p.remove_background, true)
      },
      0,
      95
    )
    const urls = task.image_urls ?? []
    if (!urls.length) throw new Error('Meshy görsel döndürmedi')
    const base = assetBaseName(ctx.asset)
    const files: string[] = []
    for (const [i, u] of urls.entries()) files.push(await downloadTo(u, path.join(ctx.dir, `${base}_ref_${i + 1}.png`), ctx.signal))
    return { outputs: { image: { kind: 'image', files, meshyTaskId: task.id, meshyResource: 'text-to-image' } } }
  },

  'image.comfyui': async (ctx, p, inputs) => {
    const pr = need(inputs, 'prompt', 'prompt').prompt
    const s = getSettings()
    const r = await comfyRun({
      url: s.comfyUrl,
      apiKey: getSecret('comfy'),
      workflowPath: comfyWorkflowPath(ctx, p),
      values: { prompt: pr.prompt, negative: pr.negative_prompt, seed: comfySeed(ctx, p) },
      outDir: ctx.dir,
      baseName: assetBaseName(ctx.asset),
      log: ctx.log,
      warn: ctx.warn,
      progress: ctx.progress,
      signal: ctx.signal
    })
    if (!r.files.image.length) throw new Error('ComfyUI görsel döndürmedi (workflow’da Save Image düğümü var mı?)')
    return { outputs: { image: { kind: 'image', files: r.files.image } } }
  },

  'comfy.workflow': async (ctx, p, inputs) => {
    const s = getSettings()
    const pr = promptOf(inputs)
    const text = pr?.prompt || str(p.text).trim() || undefined
    const img = inputs.image
    const model = inputs.model
    const audio = inputs.audio
    const seconds = pr?.duration_seconds || num(p.seconds, 0) || undefined
    const base = assetBaseName(ctx.asset)
    const r = await comfyRun({
      url: s.comfyUrl,
      apiKey: getSecret('comfy'),
      workflowPath: comfyWorkflowPath(ctx, p),
      values: { prompt: text, negative: pr?.negative_prompt || undefined, seed: comfySeed(ctx, p), seconds },
      uploads: {
        image: img?.kind === 'image' ? img.files[0] : undefined,
        model: model?.kind === 'model' ? (model.glb ?? model.file) : undefined,
        audio: audio?.kind === 'audio' ? audio.files[0] : undefined
      },
      outDir: ctx.dir,
      baseName: base,
      log: ctx.log,
      warn: ctx.warn,
      progress: (v) => ctx.progress(Math.min(90, v * 0.9)),
      signal: ctx.signal,
      timeoutMin: num(p.timeout, 30)
    })
    if (r.skipped.length) ctx.log(`Tanınmayan çıktılar atlandı: ${r.skipped.join(', ')}`)
    const outputs: Record<string, Artifact> = {}
    const f = r.files
    if (f.image.length) outputs.image = { kind: 'image', files: f.image }
    const mesh = f.model.find((x) => x.endsWith('.glb')) ?? f.model[0]
    if (mesh) outputs.model = { kind: 'model', file: mesh, glb: mesh.endsWith('.glb') ? mesh : undefined, thumbnail: f.image[0] }
    if (f.audio.length) outputs.audio = { kind: 'audio', files: f.audio, target: seconds ? { seconds, loop: pr?.loop } : undefined }
    const bvh = f.anim.find((x) => x.endsWith('.bvh'))
    let fbx = f.anim.find((x) => x.endsWith('.fbx'))
    if (bvh || fbx) {
      const clip = `AS_${base}`
      const rootMotion = ctx.asset.extra['RootMotion'] || 'root'
      let seconds = 0
      if (bvh) {
        const b = parseBvh(fs.readFileSync(bvh, 'utf8'))
        seconds = Math.round(b.frames * b.frameTime * 100) / 100
        ctx.log(`BVH: ${b.frames} kare, ${seconds} sn`)
      }
      if (!fbx && bvh) {
        if (!s.blenderPath) throw new Error('BVH’yi Unreal’a aktarmak için FBX’e çevirmek gerekiyor; Ayarlar → Yerel araçlar’dan Blender yolunu girin')
        const a = await runBlenderAnim({ blenderPath: s.blenderPath, bvh, out: path.join(ctx.dir, `${clip}.fbx`), rootMotion, log: ctx.log, signal: ctx.signal })
        fbx = a.file
        seconds = a.duration
      }
      outputs.anim = { kind: 'animation', clip, bvh: bvh ?? '', fbx: fbx!, seconds, rootMotion, loop: pr?.loop }
    }
    if (!Object.keys(outputs).length) throw new Error('ComfyUI çıktı vermedi. Workflow’da Save Image / Save Audio / Save 3D gibi bir kaydetme düğümü olmalı.')
    ctx.log(
      `Çıktılar: ${Object.entries(f)
        .filter(([, v]) => v.length)
        .map(([k, v]) => `${k} ×${v.length}`)
        .join(', ')}`
    )
    ctx.progress(100)
    return { outputs }
  },

  'model.meshy.text3d': async (ctx, p, inputs) => {
    const pr = need(inputs, 'prompt', 'prompt').prompt
    let aiModel = str(p.ai_model, 'latest')
    const smart = aiModel.startsWith('meshy-t')
    const remesh = bool(p.should_remesh, true)
    const budget = triBudget(ctx, num(p.target_polycount))
    const preview = await meshyRun(
      ctx,
      'text-to-3d',
      {
        mode: 'preview',
        prompt: pr.prompt,
        ai_model: aiModel,
        model_type: smart ? 'smart-topology' : 'standard',
        should_remesh: smart ? undefined : remesh,
        target_polycount: smart || remesh ? clampPoly(aiModel, budget) : undefined,
        pose_mode: poseFor(str(p.pose_mode, 'auto'), pr),
        target_formats: ['glb', 'fbx']
      },
      0,
      45
    )
    if (!bool(p.refine, true)) return { outputs: { model: await modelArtifact(ctx, preview, 'text-to-3d', false) } }
    if (smart) aiModel = 'latest'
    const res = str(p.texture_resolution, '2k')
    const refined = await meshyRun(
      ctx,
      'text-to-3d',
      {
        mode: 'refine',
        preview_task_id: preview.id,
        enable_pbr: bool(p.enable_pbr, true),
        texture_resolution: aiModel === 'meshy-6-lite' ? '2k' : res,
        texture_prompt: pr.texture_prompt || undefined,
        target_formats: ['glb', 'fbx']
      },
      45,
      95
    )
    return { outputs: { model: await modelArtifact(ctx, refined, 'text-to-3d', true) } }
  },

  'model.meshy.image3d': async (ctx, p, inputs) => {
    const img = need(inputs, 'image', 'image')
    const pr = promptOf(inputs)
    const aiModel = str(p.ai_model, 'latest')
    const smart = aiModel.startsWith('meshy-t')
    const remesh = bool(p.should_remesh, true)
    const texture = bool(p.should_texture, true)
    const budget = triBudget(ctx, num(p.target_polycount))
    const common = {
      ai_model: aiModel,
      model_type: smart ? 'smart-topology' : 'standard',
      should_texture: texture,
      enable_pbr: texture ? bool(p.enable_pbr, true) : undefined,
      texture_resolution: texture && aiModel !== 'meshy-6-lite' ? str(p.texture_resolution, '2k') : undefined,
      texture_prompt: texture && pr?.texture_prompt ? pr.texture_prompt : undefined,
      should_remesh: smart ? undefined : remesh,
      target_polycount: smart || remesh ? clampPoly(aiModel, budget) : undefined,
      pose_mode: poseFor(str(p.pose_mode, 'auto'), pr),
      target_formats: ['glb', 'fbx']
    }
    let resource: MeshyResource = 'image-to-3d'
    let body: Record<string, unknown>
    if (img.files.length > 1) {
      resource = 'multi-image-to-3d'
      body = { ...common, image_urls: img.files.slice(0, 4).map(fileToDataUri) }
    } else if (img.meshyResource === 'text-to-image' && img.meshyTaskId) {
      body = { ...common, input_task_id: img.meshyTaskId }
    } else {
      body = { ...common, image_url: fileToDataUri(img.files[0]) }
    }
    const task = await meshyRun(ctx, resource, body, 0, 95)
    return { outputs: { model: await modelArtifact(ctx, task, resource, texture) } }
  },

  'model.meshy.retexture': async (ctx, p, inputs) => {
    const m = need(inputs, 'model', 'model')
    const pr = need(inputs, 'prompt', 'prompt').prompt
    const aiModel = str(p.ai_model, 'latest')
    const task = await meshyRun(
      ctx,
      'retexture',
      {
        ...modelSource(m),
        text_style_prompt: (pr.texture_prompt || pr.prompt).slice(0, 800),
        ai_model: aiModel,
        enable_original_uv: bool(p.enable_original_uv, true),
        enable_pbr: bool(p.enable_pbr, true),
        texture_resolution: aiModel === 'meshy-6-lite' ? '2k' : str(p.texture_resolution, '2k'),
        target_formats: ['glb', 'fbx']
      },
      0,
      95
    )
    return { outputs: { model: await modelArtifact(ctx, task, 'retexture', true) } }
  },

  'model.meshy.remesh': async (ctx, p, inputs) => {
    const m = need(inputs, 'model', 'model')
    const budget = clampPoly('remesh', triBudget(ctx, num(p.target_polycount)))
    const task = await meshyRun(ctx, 'remesh', { ...modelSource(m), topology: str(p.topology, 'triangle'), target_polycount: budget, target_formats: ['glb', 'fbx'] }, 0, 95)
    const art = await modelArtifact(ctx, task, 'remesh', !!m.textured)
    if (art.kind === 'model' && !Object.keys(art.textures ?? {}).length) art.textures = m.textures
    return { outputs: { model: art } }
  },

  'model.meshy.rig': async (ctx, p, inputs) => {
    const m = need(inputs, 'model', 'model')
    let height = num(p.height_meters)
    if (height <= 0) height = Math.max(...parseSizeMeters(ctx.asset.size), 0) || 1.7
    const task = await meshyRun(ctx, 'rigging', { ...modelSource(m), height_meters: height }, 0, 90)
    const r = task.result ?? {}
    const base = assetBaseName(ctx.asset)
    const fbx = r.rigged_character_fbx_url ? await downloadTo(r.rigged_character_fbx_url, path.join(ctx.dir, `${base}_rigged.fbx`), ctx.signal) : undefined
    const glb = r.rigged_character_glb_url ? await downloadTo(r.rigged_character_glb_url, path.join(ctx.dir, `${base}_rigged.glb`), ctx.signal) : undefined
    if (!fbx && !glb) throw new Error('Meshy rig sonucu dosya döndürmedi')
    const animations: Record<string, string> = {}
    for (const [k, url] of Object.entries(r.basic_animations ?? {})) {
      if (url && k.endsWith('_fbx_url')) {
        const label = k.replace('_fbx_url', '')
        animations[label] = await downloadTo(url, path.join(ctx.dir, `${base}_${label}.fbx`), ctx.signal)
      }
    }
    ctx.log(`Rig tamam; animasyonlar: ${Object.keys(animations).join(', ') || 'yok'}`)
    return { outputs: { model: { ...m, kind: 'model', file: fbx ?? glb!, glb, meshyTaskId: task.id, meshyResource: 'rigging', rigged: true, animations } } }
  },

  'blender.process': async (ctx, p, inputs) => {
    const m = need(inputs, 'model', 'model')
    const res = await runBlender({
      blenderPath: getSettings().blenderPath,
      input: m.glb ?? m.file,
      outDir: ctx.dir,
      name: (m.rigged ? ctx.bible.technical.naming.skeletalMesh : ctx.bible.technical.naming.staticMesh) + assetBaseName(ctx.asset),
      format: str(p.format, 'fbx') === 'glb' ? 'glb' : 'fbx',
      targetTris: bool(p.decimate, true) ? triBudget(ctx, 0) : 0,
      targetSize: bool(p.fit_size, true) ? parseSizeMeters(ctx.asset.size) : [],
      origin: str(p.origin, 'bottom'),
      preview: bool(p.preview, true),
      skeleton: str(p.skeleton, 'none') === 'manny' ? 'manny' : 'none',
      log: ctx.log,
      signal: ctx.signal
    })
    const s = res.stats
    ctx.log(`Üçgen ${s.tris} (orijinal ${s.originalTris}) · ölçü ${s.dims.join('×')} m · UV ${s.hasUV ? 'var' : 'YOK'}`)
    return {
      outputs: {
        model: { ...m, kind: 'model', file: res.file, glb: res.file.endsWith('.glb') ? res.file : undefined, meshyTaskId: undefined, previews: res.previews, stats: s }
      }
    }
  },

  'qa.review': async (ctx, p, inputs) => {
    const subject = inputs.subject
    if (!subject) throw new Error('İncelenecek girdi yok')
    const minScore = num(p.min_score, 7)
    const technical: string[] = []
    const hardFails: string[] = []
    let images: string[] = []

    if (subject.kind === 'animation') return reviewAnimation(ctx, subject, minScore)
    if (subject.kind === 'audio' || subject.kind === 'audioset') return reviewAudio(ctx, subject, minScore)
    if (subject.kind === 'image') images = subject.files
    if (subject.kind === 'model') {
      images = subject.previews?.length ? subject.previews : subject.thumbnail ? [subject.thumbnail] : []
      const s = subject.stats
      if (s) {
        const budget = triBudget(ctx, 0)
        technical.push(`Triangles: ${s.tris} (budget ${budget})`)
        technical.push(`UVs present: ${s.hasUV}`)
        technical.push(`Bounding box (m): ${s.dims.join(' x ')} — brief size: ${ctx.asset.size || 'n/a'}`)
        technical.push(`Mesh objects: ${s.objects}, materials: ${s.materials}, rigged: ${s.rigged}`)
        if (s.tris > budget * 1.5) hardFails.push(`Üçgen sayısı bütçenin çok üstünde (${s.tris} > ${budget})`)
        if (!s.hasUV) hardFails.push('UV yok')
      }
    }

    const { data, usd } = await reviewAsset({
      bible: ctx.bible,
      asset: ctx.asset,
      subjectKind: subject.kind === 'image' ? 'reference image (will be converted to 3D or used as texture)' : '3D model renders',
      images,
      technical,
      focus: str(p.focus)
    })
    ctx.cost({ usd })
    const pass = data.score >= minScore && hardFails.length === 0
    const report: QAReport = { ...data, issues: [...hardFails, ...data.issues], pass, technical }
    fs.writeFileSync(path.join(ctx.dir, `qa_report_${Date.now()}.json`), JSON.stringify(report, null, 2))
    ctx.log(`Puan ${data.score}/10 (eşik ${minScore}) → ${pass ? 'GEÇTİ' : 'KALDI'}: ${data.summary}`)
    for (const i of report.issues) ctx.log(`  • ${i}`)
    const feedback = [data.prompt_fix, ...hardFails].filter(Boolean).join(' ')
    return { outputs: { subject, report: { kind: 'report', report } }, qa: { pass, feedback } }
  },

  'audio.elevenlabs': async (ctx, p, inputs) => {
    const pr = need(inputs, 'prompt', 'prompt').prompt
    const n = Math.min(8, Math.max(1, Math.round(num(p.variations, 1))))
    const durParam = num(p.duration)
    const loopParam = str(p.loop, 'auto')
    const files: string[] = []
    const base = ctx.bible.technical.naming.sound + assetBaseName(ctx.asset)
    for (let i = 0; i < n; i++) {
      const { data, ext } = await soundEffect(
        {
          text: pr.prompt,
          durationSeconds: durParam > 0 ? durParam : pr.duration_seconds,
          loop: loopParam === 'auto' ? pr.loop : loopParam === 'yes',
          promptInfluence: num(p.prompt_influence, 0.4),
          modelId: str(p.model_id, 'eleven_text_to_sound_v2'),
          outputFormat: getSettings().audioFormat
        },
        ctx.signal
      )
      const f = path.join(ctx.dir, `${base}_${String(i + 1).padStart(2, '0')}.${ext}`)
      fs.writeFileSync(f, data)
      files.push(f)
      ctx.cost({ chars: pr.prompt.length })
      ctx.progress(((i + 1) / n) * 100)
      ctx.log(`Ses ${i + 1}/${n}: ${path.basename(f)}`)
    }
    return { outputs: { audio: { kind: 'audio', files, target: { seconds: durParam > 0 ? durParam : pr.duration_seconds, loop: loopParam === 'auto' ? pr.loop : loopParam === 'yes' } } } }
  },

  'audio.engine': async (ctx, p, inputs) => {
    const pr = need(inputs, 'prompt', 'prompt').prompt
    const n = Math.min(8, Math.max(2, Math.round(num(p.layers, 5))))
    const idle = Math.max(300, num(p.idle_rpm, ctx.asset.dynamic?.idleRpm ?? 900))
    const max = Math.max(idle * 1.5, num(p.max_rpm, ctx.asset.dynamic?.maxRpm ?? 7000))
    // Geometrik aralık: Unreal'daki MetaSound katman aralığını yarım ton cinsinden sabit varsayar
    const ratio = Math.pow(max / idle, 1 / (n - 1))
    const bases = Array.from({ length: n }, (_, i) => Math.round(idle * Math.pow(ratio, i)))
    const timbre = pr.prompt.trim().replace(/[.\s]+$/, '')
    const prefix = ctx.bible.technical.naming.sound + assetBaseName(ctx.asset)
    const layers: { file: string; baseRpm: number; label: string }[] = []
    const load = (i: number) => (i === 0 ? 'idling smoothly, no throttle' : i === n - 1 ? 'near redline under full throttle' : 'held steady under moderate throttle')
    for (const [i, rpm] of bases.entries()) {
      const text = truncateWords(`${timbre}. Steady constant engine note at about ${rpm} RPM, ${load(i)}, no revving, no gear change, no start or stop, seamless loop.`, 450)
      const { data, ext } = await soundEffect(
        {
          text,
          durationSeconds: Math.min(30, Math.max(2, num(p.seconds, 6))),
          loop: true,
          promptInfluence: num(p.prompt_influence, 0.55),
          modelId: str(p.model_id, 'eleven_text_to_sound_v2'),
          outputFormat: getSettings().audioFormat.startsWith('pcm') ? getSettings().audioFormat : 'pcm_24000'
        },
        ctx.signal
      )
      const file = path.join(ctx.dir, `${prefix}_L${i}_${rpm}rpm.${ext}`)
      fs.writeFileSync(file, data)
      layers.push({ file, baseRpm: rpm, label: i === 0 ? 'Rölanti' : i === n - 1 ? 'Kırmızı çizgi' : `${rpm} rpm` })
      ctx.cost({ chars: text.length })
      ctx.progress(((i + 1) / n) * 100)
      ctx.log(`Katman ${i + 1}/${n}: ${rpm} rpm → ${path.basename(file)}`)
    }
    const dynamic = { type: 'engine' as const, layers: n, idleRpm: idle, maxRpm: Math.round(max), engine: timbre }
    return { outputs: { audio: { kind: 'audioset', layers, dynamic, target: { seconds: Math.min(30, Math.max(2, num(p.seconds, 6))), loop: true } } } }
  },

  'anim.kimodo': async (ctx, p) => {
    // Segmentler: parametre > kit (asset.extra) > Claude
    let texts = str(p.prompt)
      .split(/\r?\n/)
      .map((l) => l.trim())
      .filter(Boolean)
    let secs = str(p.seconds)
      .split(/[\s,;]+/)
      .map(Number)
      .filter((x) => x > 0)
    if (!texts.length && ctx.asset.extra['Kimodo']) {
      texts = ctx.asset.extra['Kimodo'].split(/\r?\n/).filter(Boolean)
      if (!secs.length && ctx.asset.extra['Süre'])
        secs = ctx.asset.extra['Süre']
          .split(/[\s,;]+/)
          .map(Number)
          .filter((x) => x > 0)
    }
    // Kalite kontrol geri bildirimi (yeniden deneme): Claude yolu bunu promptu yazarken kullanır,
    // hazır metinlere ise düzeltme eklenir
    const retry = ctx.feedback.length
    if (retry && texts.length) {
      const fix = ctx.feedback[retry - 1]
      texts = texts.map((t) => `${t.replace(/[.,\s]+$/, '')}, ${fix}.`)
      ctx.log(`Kalite kontrol düzeltmesi prompta eklendi: ${fix}`)
    }
    if (!texts.length) {
      const { data, usd } = await generatePrompt({ bible: ctx.bible, asset: ctx.asset, targetId: 'kimodo_motion', extra: str(p.extra), feedback: ctx.feedback })
      ctx.cost({ usd })
      texts = [data.prompt]
      secs = [Math.min(10, Math.max(2, data.duration_seconds || 4))]
      ctx.log(`Hareket promptu: ${data.prompt} (${secs[0]} sn) — ${data.rationale}`)
    }
    if (!secs.length) secs = [4]
    const durations = texts.map((_, i) => secs[Math.min(i, secs.length - 1)])
    // Sistem klibi: kök yolu + poz eşleme + döngü kesimi
    let spec: KimodoSpec | null = null
    try {
      spec = ctx.asset.extra['KimodoSpec'] ? (JSON.parse(ctx.asset.extra['KimodoSpec']) as KimodoSpec) : null
    } catch {
      ctx.warn('KimodoSpec okunamadı; düz üretim yapılıyor')
    }
    const seedP = num(p.seed, -1)
    const fixedSeed = spec?.seed !== undefined && spec.seed >= 0 ? spec.seed : seedP
    // Sabit seed aynı hareketi tekrar üretir: yeniden denemede kaydır
    const seed = fixedSeed >= 0 && retry ? fixedSeed + retry * 7919 : fixedSeed
    if (fixedSeed >= 0 && retry) ctx.log(`Yeniden deneme: seed ${fixedSeed} → ${seed}`)
    const req: KimodoRequest = { texts, durations, seed: seed >= 0 ? seed : undefined, steps: num(p.steps, 100) }
    if (spec) {
      const from = loadMotion(ctx.projectDir, spec.fromAssetId)?.poses?.last
      const to = loadMotion(ctx.projectDir, spec.toAssetId)?.poses?.first
      if (spec.fromAssetId && !from) ctx.warn('Başlangıç pozu kaynağı henüz üretilmemiş; poz eşleme atlandı')
      if (spec.toAssetId && !to) ctx.warn('Bitiş pozu kaynağı henüz üretilmemiş; poz eşleme atlandı')
      const g = buildKimodoRequest(spec, texts, durations, { from, to })
      Object.assign(req, { durations: g.durations, loop: g.loop, loop_min: g.loop_min, loop_max: g.loop_max, loop_keep_turn: g.loop_keep_turn, constraints: g.constraints })
      const kinds = g.constraints.map((c) => String(c.type))
      ctx.log(
        `Sistem klibi (${spec.role}${spec.loop ? ', döngü' : ''}): hedef ${spec.speed}${spec.speedEnd !== undefined && spec.speedEnd !== spec.speed ? '→' + spec.speedEnd : ''} m/sn, yön ${spec.moveDir}°, dönüş ${spec.turn}° · kısıtlar: ${kinds.join(', ') || 'yok'}`
      )
    }
    ctx.log(`Kimodo: ${texts.length} segment, ${req.durations.reduce((a, b) => a + b, 0).toFixed(1)} sn üretim`)
    ctx.progress(5)
    let res
    try {
      res = await kimodoGenerate(req, ctx.signal)
    } catch (e) {
      if (!req.constraints?.length || ctx.signal.aborted) throw e
      ctx.warn(`Kısıtlı üretim başarısız (${(e as Error).message}); kısıtsız tekrar deneniyor`)
      res = await kimodoGenerate({ ...req, constraints: [] }, ctx.signal)
    }
    ctx.log(`Kimodo üretti: ${res.frames[0]} kare @ ${res.fps} fps (${res.seconds} sn sürdü)`)
    const motion = (res.motion?.[0] ?? null) as StoredMotion | null
    if (motion) {
      const m = motion.measured
      ctx.log(
        `Ölçüm: ${m.seconds.toFixed(2)} sn, hız ${m.speed.toFixed(2)} m/sn, yön ${m.direction}°, dönüş ${m.turn}°` +
          (motion.loop ? ` · döngü ${motion.loop.start}→${motion.loop.end}. kare (fark ${motion.loop.cost})` : '') +
          ` · basışlar L ${motion.markers.L.length} / R ${motion.markers.R.length}`
      )
    }
    ctx.progress(70)
    const clip = `AS_SOMA_${assetBaseName(ctx.asset)}`
    const bvh = path.join(ctx.dir, `${clip}.bvh`)
    fs.writeFileSync(bvh, res.bvh[0])
    const rootMotion = ctx.asset.extra['RootMotion'] || str(p.root_motion, 'root')
    const a = await runBlenderAnim({ blenderPath: getSettings().blenderPath, bvh, out: path.join(ctx.dir, `${clip}.fbx`), rootMotion, log: ctx.log, signal: ctx.signal })
    ctx.log(`FBX animasyon: ${a.frames} kare, ${a.duration.toFixed(2)} sn`)
    const loop = spec ? spec.loop : ctx.asset.extra['Döngü'] === 'evet'
    if (motion) saveMotion(ctx.projectDir, ctx.asset.id, { ...motion, bvh, fbx: a.file, clip, at: Date.now() })
    return { outputs: { anim: { kind: 'animation', clip, bvh, fbx: a.file, seconds: a.duration, rootMotion, loop, motion: motionInfo(motion) } } }
  },

  'export.ue': async (ctx, p, inputs) => {
    const subject = inputs.subject
    if (!subject) throw new Error('Export edilecek girdi yok')
    const t = ctx.bible.technical
    const cat = ctx.asset.category || 'Misc'
    const root = t.ueContentRoot.replace(/\/$/, '')
    const contentPath = (str(p.content_path) || `${root}/${cat}`).replace(/\\/g, '/')
    const exportDir = ensureDir(path.join(ctx.projectDir, 'export', cat, assetBaseName(ctx.asset)))
    const base = assetBaseName(ctx.asset)
    const items: Record<string, unknown>[] = []
    const copy = (src: string, name: string) => {
      const dest = path.join(exportDir, name + path.extname(src))
      fs.copyFileSync(src, dest)
      return dest
    }

    if (subject.kind === 'model') {
      const prefix = subject.rigged ? t.naming.skeletalMesh : t.naming.staticMesh
      const name = prefix + base
      const textures: Record<string, string> = {}
      for (const [role, f] of Object.entries(subject.textures ?? {})) textures[role] = copy(f, `${t.naming.texture}${base}_${role}`)
      const item: Record<string, unknown> = { file: copy(subject.file, name), dest: contentPath, name, type: subject.rigged ? 'skeletal_mesh' : 'static_mesh', textures }
      if (subject.rigged) {
        const lods = str(p.lods, '0.5,0.25,0.1')
          .split(/[\s,;]+/)
          .map(Number)
          .filter((x) => x > 0 && x < 1)
        if (lods.length) item.lods = lods
        if (bool(p.ik_rig, true)) {
          item.ik_rig = true
          item.retarget_source_ik_rig = getSettings().mannyIkRig
        }
      }
      items.push(item)
      for (const [label, f] of Object.entries(subject.animations ?? {})) copy(f, `A_${base}_${label}`)
    } else if (subject.kind === 'audioset') {
      const layers = subject.layers.map((l, i) => ({ file: copy(l.file, `SW_${base}_L${i}`), name: `SW_${base}_L${i}`, base_rpm: l.baseRpm }))
      items.push({ type: 'dynamic_sound', name: `MS_${base}`, dest: contentPath, file: '', layers })
      ctx.log(`Dinamik ses: ${layers.length} katman → MS_${base} (girişler: RPM, Throttle)`)
    } else if (subject.kind === 'animation') {
      const fbx = copy(subject.fbx, subject.clip)
      const targets: { mesh: string; base: string }[] = []
      const par = ctx.parent
      if (par && (par.kind === 'character' || par.kind === 'creature')) {
        const pbase = assetBaseName(par)
        targets.push({ mesh: `${root}/${par.category || 'Misc'}/${t.naming.skeletalMesh}${pbase}`, base: pbase })
      }
      items.push({
        type: 'animation',
        name: subject.clip,
        file: fbx,
        dest: `${root}/Animation/SOMA`,
        soma_dest: `${root}/Animation/SOMA`,
        root_motion: subject.rootMotion === 'root',
        loop: !!subject.loop,
        markers: subject.motion?.markers ?? null,
        targets
      })
      ctx.log(targets.length ? `Animasyon → ${targets[0].base} karakterine retarget edilecek` : 'Animasyon ortak SOMA iskeletine aktarılacak (bağlı karakter yok)')
    } else if (subject.kind === 'image') {
      subject.files.forEach((f, i) => {
        const name = `${t.naming.texture}${base}${subject.files.length > 1 ? `_${i + 1}` : ''}`
        items.push({ file: copy(f, name), dest: contentPath, name, type: 'texture' })
      })
    } else if (subject.kind === 'audio') {
      subject.files.forEach((f) => {
        const name = path.basename(f, path.extname(f))
        items.push({ file: copy(f, name), dest: contentPath, name, type: 'sound' })
      })
    } else {
      throw new Error(`Export desteklenmiyor: ${subject.kind}`)
    }

    // Unreal import betiği: hem otomatik çalıştırılır hem elle çalıştırmak için saklanır
    const manifest = { master_material: t.masterMaterial, items }
    const b64 = Buffer.from(JSON.stringify(manifest), 'utf8').toString('base64')
    const script = `import base64, json\nMANIFEST = json.loads(base64.b64decode("${b64}").decode("utf-8"))\n` + fs.readFileSync(resourcePath('python', 'ue_import.py'), 'utf8')
    const scriptFile = path.join(exportDir, `ue_import_${ctx.job.id}.py`)
    fs.writeFileSync(scriptFile, script)
    ctx.log(`Export klasörü: ${exportDir}`)

    if (bool(p.auto_import, true)) {
      try {
        const r = await execInUe(script)
        const line = r.output.split('\n').find((l) => l.startsWith('FORGE_RESULT '))
        if (!r.success) ctx.warn(`Unreal import hatası: ${r.result || r.output.slice(-400)}`)
        else {
          ctx.log(`Unreal'a import edildi → ${contentPath}`)
          try {
            const res = JSON.parse(line ? line.slice(13) : '[]') as { name: string; errors: string[]; notes?: string[] }[]
            for (const e of res) {
              for (const n of e.notes ?? []) ctx.log(`  ✓ ${n}`)
              for (const er of e.errors) ctx.warn(`${e.name}: ${er}`)
            }
          } catch {
            /* sonuç satırı yok */
          }
        }
      } catch (e) {
        ctx.warn(`Unreal'a otomatik import yapılamadı (${(e as Error).message}). Elle: Unreal → File → Execute Python Script → ${scriptFile}`)
      }
    }
    return { outputs: {} }
  }
}

export function getExecutor(type: string): Executor {
  const ex = executors[type]
  if (!ex) throw new Error(`Bilinmeyen node tipi: ${type}`)
  return ex
}

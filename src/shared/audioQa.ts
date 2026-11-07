// SPDX-License-Identifier: AGPL-3.0-only
// Ses kalite kontrolü: WAV dosyasından sessizlik, kırpılma, seviye, baştaki boşluk, süre ve döngü dikişi.
// Servis gerektirmez; "Kalite Kontrol" düğümü kötü sonuçta prompt motoruna geri bildirim gönderir.

export interface Pcm {
  sampleRate: number
  channels: number
  /** Kanallar ortalanmış örnekler, -1..1 */
  samples: Float32Array
}

/** RIFF/WAVE: 16/24/32 bit PCM ve 32 bit float. Desteklenmeyen biçimde null. */
export function decodeWav(buf: Uint8Array): Pcm | null {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength)
  const tag = (o: number) => String.fromCharCode(buf[o], buf[o + 1], buf[o + 2], buf[o + 3])
  if (buf.length < 44 || tag(0) !== 'RIFF' || tag(8) !== 'WAVE') return null
  let fmt: { format: number; channels: number; rate: number; bits: number } | null = null
  let off = 12
  while (off + 8 <= buf.length) {
    const id = tag(off)
    const size = dv.getUint32(off + 4, true)
    const body = off + 8
    if (id === 'fmt ') {
      let format = dv.getUint16(body, true)
      if (format === 0xfffe && size >= 26) format = dv.getUint16(body + 24, true) // WAVE_FORMAT_EXTENSIBLE
      fmt = { format, channels: dv.getUint16(body + 2, true), rate: dv.getUint32(body + 4, true), bits: dv.getUint16(body + 14, true) }
    } else if (id === 'data' && fmt) {
      const { format, channels, rate, bits } = fmt
      const bytes = bits / 8
      const end = Math.min(buf.length, body + size)
      const frames = Math.floor((end - body) / (bytes * channels))
      const out = new Float32Array(frames)
      const read =
        format === 3 && bits === 32
          ? (o: number) => dv.getFloat32(o, true)
          : format === 1 && bits === 16
            ? (o: number) => dv.getInt16(o, true) / 32768
            : format === 1 && bits === 24
              ? (o: number) => ((buf[o] | (buf[o + 1] << 8) | (buf[o + 2] << 16)) << 8) / 2147483648
              : format === 1 && bits === 32
                ? (o: number) => dv.getInt32(o, true) / 2147483648
                : null
      if (!read || !channels) return null
      for (let f = 0; f < frames; f++) {
        let s = 0
        for (let c = 0; c < channels; c++) s += read(body + (f * channels + c) * bytes)
        out[f] = s / channels
      }
      return { sampleRate: rate, channels, samples: out }
    }
    off = body + size + (size % 2)
  }
  return null
}

export interface AudioMetrics {
  seconds: number
  peakDb: number
  rmsDb: number
  /** Tam ölçeğe dayanan örneklerin oranı */
  clipped: number
  leadingSilence: number
  trailingSilence: number
  /** Döngü: son ve ilk örnek arasındaki atlama (0–2) */
  seamJump: number
  /** Döngü: ilk ve son 100 ms arasındaki seviye farkı (dB) */
  seamLevelDb: number
}

const db = (v: number) => (v > 0 ? Math.round(20 * Math.log10(v) * 10) / 10 : -120)
const SILENCE_DB = -50

export function measureAudio(pcm: Pcm): AudioMetrics {
  const s = pcm.samples
  const n = s.length
  const rate = pcm.sampleRate || 1
  let peak = 0
  let sum = 0
  let clipped = 0
  for (let i = 0; i < n; i++) {
    const a = Math.abs(s[i])
    if (a > peak) peak = a
    if (a >= 0.999) clipped++
    sum += s[i] * s[i]
  }
  // 10 ms pencerelerle baştaki/sondaki sessizlik
  const win = Math.max(1, Math.round(rate * 0.01))
  const windowDb = (i: number) => {
    let e = 0
    const end = Math.min(n, i + win)
    for (let k = i; k < end; k++) e += s[k] * s[k]
    return db(Math.sqrt(e / Math.max(1, end - i)))
  }
  let lead = 0
  while (lead < n && windowDb(lead) < SILENCE_DB) lead += win
  let trail = n
  while (trail > lead && windowDb(Math.max(0, trail - win)) < SILENCE_DB) trail -= win
  const span = Math.max(1, Math.round(rate * 0.1))
  const rmsOf = (a: number, b: number) => {
    let e = 0
    for (let k = a; k < b; k++) e += s[k] * s[k]
    return Math.sqrt(e / Math.max(1, b - a))
  }
  const r = (v: number, d = 3) => Math.round(v * 10 ** d) / 10 ** d
  return {
    seconds: r(n / rate),
    peakDb: db(peak),
    rmsDb: db(Math.sqrt(sum / Math.max(1, n))),
    clipped: r(n ? clipped / n : 0, 5),
    leadingSilence: r(Math.min(n, lead) / rate),
    trailingSilence: r((n - Math.max(lead, trail)) / rate),
    seamJump: r(n > 1 ? Math.abs(s[n - 1] - s[0]) : 0),
    seamLevelDb: n > 2 * span ? Math.round(Math.abs(db(rmsOf(0, span)) - db(rmsOf(n - span, n))) * 10) / 10 : 0
  }
}

export interface AudioTargets {
  seconds?: number
  loop?: boolean
}

export interface AudioQa {
  score: number
  issues: string[]
  /** ElevenLabs promptuna eklenecek İngilizce düzeltme */
  fixes: string[]
  technical: string[]
}

/** Tek bir dosyanın değerlendirmesi */
export function judgeAudio(name: string, m: AudioMetrics, t: AudioTargets = {}): AudioQa {
  const issues: string[] = []
  const fixes: string[] = []
  let score = 10
  const technical = [
    `${name}: ${m.seconds} sn · tepe ${m.peakDb} dBFS · ortalama ${m.rmsDb} dBFS · kırpılma %${Math.round(m.clipped * 10000) / 100} · baş/son sessizlik ${m.leadingSilence}/${m.trailingSilence} sn${t.loop ? ` · dikiş ${m.seamJump}, ${m.seamLevelDb} dB` : ''}`
  ]
  if (m.peakDb < -45) {
    return { score: 0, issues: [`${name}: dosya neredeyse tamamen sessiz`], fixes: ['clearly audible, present sound'], technical }
  }
  if (m.clipped > 0.005) {
    score -= 3
    issues.push(`${name}: kırpılma / distorsiyon var (örneklerin %${Math.round(m.clipped * 1000) / 10}'i tam ölçekte)`)
    fixes.push('clean, undistorted sound without clipping')
  } else if (m.clipped > 0.0005) {
    score -= 1
    issues.push(`${name}: hafif kırpılma`)
  }
  if (m.rmsDb < -38) {
    score -= 2
    issues.push(`${name}: çok kısık (ortalama ${m.rmsDb} dBFS)`)
    fixes.push('louder and more present, close-miked')
  }
  if (m.leadingSilence > 0.4) {
    score -= 1
    issues.push(`${name}: başında ${m.leadingSilence} sn sessizlik var`)
    fixes.push('the sound starts immediately')
  }
  if (t.seconds && t.seconds > 0) {
    const diff = Math.abs(m.seconds - t.seconds)
    if (diff > Math.max(0.6, t.seconds * 0.3)) {
      score -= 2
      issues.push(`${name}: süre ${m.seconds} sn, istenen ${t.seconds} sn`)
    }
  }
  if (t.loop) {
    if (m.seamJump > 0.25 || m.seamLevelDb > 6) {
      score -= 2
      issues.push(`${name}: döngü dikişinde ${m.seamJump > 0.25 ? 'tık sesi' : `seviye sıçraması (${m.seamLevelDb} dB)`} olabilir`)
      fixes.push('a seamless, steady loop with constant level from start to end')
    }
    if (m.leadingSilence > 0.15 || m.trailingSilence > 0.3) {
      score -= 1
      issues.push(`${name}: döngü sesinde baş/son boşluğu var`)
      fixes.push('continuous sound with no gaps')
    }
  }
  return { score: Math.max(0, score), issues, fixes, technical }
}

/** Aggregate measured audio scores. An empty set is unmeasured, never a passing 10/10. */
export function audioScore(scores: number[], kind: 'audio' | 'audioset', minScore: number): { score: number; pass: boolean } {
  if (!scores.length) return { score: 0, pass: false }
  const score = kind === 'audioset' ? Math.min(...scores) : Math.max(...scores)
  return { score, pass: score >= minScore }
}

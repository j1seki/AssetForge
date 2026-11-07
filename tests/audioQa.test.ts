// SPDX-License-Identifier: AGPL-3.0-only
import { describe, expect, it } from 'vitest'
import { decodeWav, judgeAudio, measureAudio, type Pcm } from '../src/shared/audioQa'

const RATE = 24000

/** Mono 16 bit PCM WAV (AssetForge'un ElevenLabs PCM çıktısıyla aynı biçim) */
function wav(samples: number[], opts: { rate?: number; channels?: number; bits?: 16 | 24 | 32; float?: boolean; extensible?: boolean } = {}): Uint8Array {
  const { rate = RATE, channels = 1, bits = 16, float = false, extensible = false } = opts
  const bytes = bits / 8
  const fmtSize = extensible ? 40 : 16
  const data = samples.length * bytes
  const buf = Buffer.alloc(12 + 8 + fmtSize + 8 + data)
  buf.write('RIFF', 0)
  buf.writeUInt32LE(buf.length - 8, 4)
  buf.write('WAVE', 8)
  buf.write('fmt ', 12)
  buf.writeUInt32LE(fmtSize, 16)
  const format = float ? 3 : 1
  buf.writeUInt16LE(extensible ? 0xfffe : format, 20)
  buf.writeUInt16LE(channels, 22)
  buf.writeUInt32LE(rate, 24)
  buf.writeUInt32LE(rate * channels * bytes, 28)
  buf.writeUInt16LE(channels * bytes, 32)
  buf.writeUInt16LE(bits, 34)
  if (extensible) {
    buf.writeUInt16LE(22, 36)
    buf.writeUInt16LE(format, 36 + 8)
  }
  const d = 20 + fmtSize
  buf.write('data', d)
  buf.writeUInt32LE(data, d + 4)
  samples.forEach((v, i) => {
    const o = d + 8 + i * bytes
    const c = Math.max(-1, Math.min(1, v))
    if (float) buf.writeFloatLE(c, o)
    else if (bits === 16) buf.writeInt16LE(Math.round(c * 32767), o)
    else if (bits === 24) buf.writeIntLE(Math.round(c * 8388607), o, 3)
    else buf.writeInt32LE(Math.round(c * 2147483647), o)
  })
  return new Uint8Array(buf)
}

const tone = (seconds: number, amp = 0.5, hz = 440) => Array.from({ length: Math.round(seconds * RATE) }, (_, i) => amp * Math.sin((2 * Math.PI * hz * i) / RATE))
const silence = (seconds: number) => Array.from({ length: Math.round(seconds * RATE) }, () => 0)
const measure = (s: number[]) => measureAudio(decodeWav(wav(s)) as Pcm)

describe('WAV çözücü', () => {
  it('16/24/32 bit PCM, float, extensible ve stereo', () => {
    const s = [0, 0.5, -0.5, 0.25]
    for (const o of [{}, { bits: 24 as const }, { bits: 32 as const }, { bits: 32 as const, float: true }, { extensible: true }]) {
      const p = decodeWav(wav(s, o))!
      expect(p.sampleRate).toBe(RATE)
      expect([...p.samples].map((v) => Math.round(v * 100) / 100)).toEqual([0, 0.5, -0.5, 0.25])
    }
    const st = decodeWav(wav([0.5, -0.5, 1, 0], { channels: 2 }))!
    expect(st.channels).toBe(2)
    expect([...st.samples].map((v) => Math.round(v * 100) / 100)).toEqual([0, 0.5])
  })
  it('WAV olmayan ya da desteklenmeyen dosyada null', () => {
    expect(decodeWav(new Uint8Array([0x49, 0x44, 0x33, 4, 0, 0]))).toBeNull()
    const adpcm = wav([0, 0])
    adpcm[20] = 2
    expect(decodeWav(adpcm)).toBeNull()
  })
})

describe('ses ölçümü ve değerlendirme', () => {
  it('temiz ses geçer', () => {
    const m = measure(tone(2))
    expect(m.seconds).toBeCloseTo(2, 2)
    expect(m.peakDb).toBeCloseTo(-6, 0)
    expect(m.rmsDb).toBeCloseTo(-9, 0)
    expect(m.clipped).toBe(0)
    expect(m.leadingSilence).toBe(0)
    expect(judgeAudio('a.wav', m, { seconds: 2 })).toMatchObject({ score: 10, issues: [] })
  })
  it('sessiz dosya sıfır puan', () => {
    const r = judgeAudio('s.wav', measure(silence(2)))
    expect(r.score).toBe(0)
    expect(r.fixes.join()).toMatch(/audible/)
  })
  it('kırpılma, kısık ses ve baştaki boşluk', () => {
    expect(judgeAudio('c.wav', measure(tone(2, 1.6))).issues.join()).toMatch(/kırpılma/)
    const quiet = judgeAudio('q.wav', measure(tone(2, 0.01)))
    expect(quiet.issues.join()).toMatch(/kısık/)
    expect(quiet.fixes.join()).toMatch(/louder/)
    const late = measure([...silence(1), ...tone(1)])
    expect(late.leadingSilence).toBeCloseTo(1, 1)
    expect(judgeAudio('l.wav', late).issues.join()).toMatch(/başında/)
  })
  it('istenen süreden çok farklı', () => {
    expect(judgeAudio('d.wav', measure(tone(1)), { seconds: 4 }).issues.join()).toMatch(/istenen 4 sn/)
    expect(judgeAudio('d.wav', measure(tone(3.7)), { seconds: 4 }).issues).toEqual([])
  })
  it('döngü: temiz dikiş geçer, sönen son ve tık yakalanır', () => {
    // 440 Hz, 2 sn = tam 880 periyot → baş ve son buluşur
    expect(judgeAudio('loop.wav', measure(tone(2)), { loop: true }).issues).toEqual([])
    const fade = tone(2).map((v, i, a) => (i > a.length - RATE * 0.3 ? v * ((a.length - i) / (RATE * 0.3)) ** 3 : v))
    expect(judgeAudio('fade.wav', measure(fade), { loop: true }).issues.join()).toMatch(/seviye sıçraması|boşluğu/)
    const click = [...tone(1.99), 0.9]
    expect(measure(click).seamJump).toBeGreaterThan(0.25)
    expect(judgeAudio('click.wav', measure(click), { loop: true }).issues.join()).toMatch(/tık/)
  })
})

describe('audio QA aggregation', () => {
  it('fails closed when no WAV file could be measured', async () => {
    const { audioScore } = await import('../src/shared/audioQa')
    expect(audioScore([], 'audio', 7)).toEqual({ score: 0, pass: false })
    expect(audioScore([], 'audioset', 7)).toEqual({ score: 0, pass: false })
  })
  it('uses the best SFX variation and the weakest engine layer', async () => {
    const { audioScore } = await import('../src/shared/audioQa')
    expect(audioScore([4, 9], 'audio', 7)).toEqual({ score: 9, pass: true })
    expect(audioScore([4, 9], 'audioset', 7)).toEqual({ score: 4, pass: false })
  })
})

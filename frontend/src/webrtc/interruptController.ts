const INTERRUPT_CONFIG = {
  minSpeechMs: 300,
  vadStartMs: 150,
  speechEndSilenceMs: 800,
  cooldownMs: 500,
  volumeThreshold: 0.02,
}

export type VadEvent = 'speech-start' | 'speech-end'

export interface VadState {
  speechStartedAt: number | null
  silenceStartedAt: number | null
  validSpeech: boolean
  speechStartEmitted: boolean
}

export function createVadState(): VadState {
  return {
    speechStartedAt: null,
    silenceStartedAt: null,
    validSpeech: false,
    speechStartEmitted: false,
  }
}

export function updateVadState(
  state: VadState,
  input: { now: number; isSpeech: boolean; agentSpeaking: boolean; inCooldown: boolean },
): { state: VadState; events: VadEvent[] } {
  const next = { ...state }
  const events: VadEvent[] = []

  if (input.isSpeech) {
    if (next.speechStartedAt === null) next.speechStartedAt = input.now
    next.silenceStartedAt = null
    const speechMs = input.now - next.speechStartedAt
    if (speechMs >= INTERRUPT_CONFIG.minSpeechMs) next.validSpeech = true
    if (
      !next.speechStartEmitted &&
      next.validSpeech &&
      speechMs >= INTERRUPT_CONFIG.vadStartMs &&
      input.agentSpeaking &&
      !input.inCooldown
    ) {
      next.speechStartEmitted = true
      events.push('speech-start')
    }
    return { state: next, events }
  }

  if (next.speechStartedAt === null) return { state: next, events }
  if (next.silenceStartedAt === null) next.silenceStartedAt = input.now
  if (
    !next.validSpeech &&
    input.now - next.silenceStartedAt >= INTERRUPT_CONFIG.vadStartMs
  ) {
    return { state: createVadState(), events }
  }
  if (
    next.validSpeech &&
    input.now - next.silenceStartedAt >= INTERRUPT_CONFIG.speechEndSilenceMs
  ) {
    events.push('speech-end')
    return { state: createVadState(), events }
  }
  return { state: next, events }
}

export class InterruptController {
  private context: AudioContext | null = null
  private analyser: AnalyserNode | null = null
  private source: MediaStreamAudioSourceNode | null = null
  private frame = 0
  private vadState = createVadState()
  private cooldownUntil = 0
  private agentSpeaking = false
  private callback: (() => void) | null = null
  private speechEndCallback: (() => void) | null = null

  async start(stream: MediaStream): Promise<void> {
    this.stop()
    this.context = new AudioContext()
    this.analyser = this.context.createAnalyser()
    this.analyser.fftSize = 1024
    this.source = this.context.createMediaStreamSource(stream)
    this.source.connect(this.analyser)
    this.loop()
  }

  stop(): void {
    if (this.frame) cancelAnimationFrame(this.frame)
    this.frame = 0
    this.source?.disconnect()
    this.analyser?.disconnect()
    void this.context?.close()
    this.context = null
    this.analyser = null
    this.source = null
    this.vadState = createVadState()
  }

  onUserSpeechStart(callback: () => void): void {
    this.callback = callback
  }

  onUserSpeechEnd(callback: () => void): void {
    this.speechEndCallback = callback
  }

  setAgentSpeaking(isSpeaking: boolean): void {
    this.agentSpeaking = isSpeaking
  }

  inCooldown(): boolean {
    return performance.now() < this.cooldownUntil
  }

  startCooldown(): void {
    this.cooldownUntil = performance.now() + INTERRUPT_CONFIG.cooldownMs
  }

  private loop = (): void => {
    if (!this.analyser) return
    const data = new Float32Array(this.analyser.fftSize)
    this.analyser.getFloatTimeDomainData(data)
    let sum = 0
    for (const sample of data) sum += sample * sample
    const volume = Math.sqrt(sum / data.length)
    const now = performance.now()
    const result = updateVadState(this.vadState, {
      now,
      isSpeech: volume > INTERRUPT_CONFIG.volumeThreshold,
      agentSpeaking: this.agentSpeaking,
      inCooldown: this.inCooldown(),
    })
    this.vadState = result.state
    for (const event of result.events) {
      if (event === 'speech-start') {
        this.callback?.()
        this.startCooldown()
      } else {
        this.speechEndCallback?.()
      }
    }
    this.frame = requestAnimationFrame(this.loop)
  }
}

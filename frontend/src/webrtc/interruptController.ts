const INTERRUPT_CONFIG = {
  minSpeechMs: 300,
  vadStartMs: 150,
  cooldownMs: 500,
  volumeThreshold: 0.02,
}

export class InterruptController {
  private context: AudioContext | null = null
  private analyser: AnalyserNode | null = null
  private source: MediaStreamAudioSourceNode | null = null
  private frame = 0
  private speechStartedAt = 0
  private cooldownUntil = 0
  private agentSpeaking = false
  private callback: (() => void) | null = null

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
    this.speechStartedAt = 0
  }

  onUserSpeechStart(callback: () => void): void {
    this.callback = callback
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
    if (volume > INTERRUPT_CONFIG.volumeThreshold) {
      if (!this.speechStartedAt) this.speechStartedAt = now
      const elapsed = now - this.speechStartedAt
      if (
        this.agentSpeaking &&
        elapsed > INTERRUPT_CONFIG.minSpeechMs &&
        elapsed > INTERRUPT_CONFIG.vadStartMs &&
        !this.inCooldown()
      ) {
        this.callback?.()
        this.startCooldown()
      }
    } else {
      this.speechStartedAt = 0
    }
    this.frame = requestAnimationFrame(this.loop)
  }
}


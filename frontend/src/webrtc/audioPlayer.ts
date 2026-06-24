export class StreamingAudioPlayer {
  currentResponseId: string | null = null
  private context: AudioContext | null = null
  private nextStartTime = 0
  private sources: AudioBufferSourceNode[] = []

  setCurrentResponse(responseId: string): void {
    if (this.currentResponseId !== responseId) {
      this.clear()
      this.currentResponseId = responseId
    }
  }

  shouldPlay(responseId: string): boolean {
    return !!responseId && responseId === this.currentResponseId
  }

  enqueueAudio(responseId: string, audioBase64: string): void {
    if (!this.shouldPlay(responseId)) return
    const context = this.ensureContext()
    const pcm = this.base64ToInt16(audioBase64)
    if (!pcm.length) return
    const buffer = context.createBuffer(1, pcm.length, 24000)
    const channel = buffer.getChannelData(0)
    for (let i = 0; i < pcm.length; i += 1) {
      channel[i] = Math.max(-1, Math.min(1, pcm[i] / 32768))
    }
    const source = context.createBufferSource()
    source.buffer = buffer
    source.connect(context.destination)
    const startTime = Math.max(context.currentTime + 0.03, this.nextStartTime)
    source.start(startTime)
    this.nextStartTime = startTime + buffer.duration
    this.sources.push(source)
    source.onended = () => {
      this.sources = this.sources.filter((item) => item !== source)
    }
  }

  stop(): void {
    for (const source of this.sources) {
      try {
        source.stop()
      } catch {
        // Already stopped.
      }
    }
    this.sources = []
    if (this.context) this.nextStartTime = this.context.currentTime
  }

  clear(): void {
    this.stop()
  }

  private ensureContext(): AudioContext {
    if (!this.context) {
      this.context = new AudioContext({ sampleRate: 24000 })
      this.nextStartTime = this.context.currentTime
    }
    if (this.context.state === 'suspended') void this.context.resume()
    return this.context
  }

  private base64ToInt16(base64: string): Int16Array {
    const binary = atob(base64)
    const bytes = new Uint8Array(binary.length)
    for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i)
    return new Int16Array(bytes.buffer)
  }
}


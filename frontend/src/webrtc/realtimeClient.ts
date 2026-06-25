type MessageHandler = (message: any) => void

export class RealtimeClient {
  sessionId: string | null = null
  websocketUrl = ''
  private ws: WebSocket | null = null
  private stream: MediaStream | null = null
  private context: AudioContext | null = null
  private processor: ScriptProcessorNode | null = null
  private source: MediaStreamAudioSourceNode | null = null
  private sink: GainNode | null = null
  private handlers: MessageHandler[] = []

  onMessage(handler: MessageHandler): void {
    this.handlers.push(handler)
  }

  async createSession(payload: Record<string, any> = {}): Promise<any> {
    const response = await fetch('/api/agent/session', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    if (!response.ok) throw new Error(`创建会话失败：${response.status}`)
    const sessionPayload = await response.json()
    this.sessionId = sessionPayload.session_id
    this.websocketUrl = this.absoluteWsUrl(sessionPayload.websocket_url)
    return sessionPayload
  }

  async connect(): Promise<void> {
    if (!this.sessionId || !this.websocketUrl) await this.createSession()
    await new Promise<void>((resolve, reject) => {
      const ws = new WebSocket(this.websocketUrl)
      this.ws = ws
      ws.onopen = () => resolve()
      ws.onerror = () => reject(new Error('实时连接失败'))
      ws.onclose = () => {
        this.handlers.forEach((handler) => handler({ type: 'disconnected', message: '实时连接已断开' }))
      }
      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data)
          this.handlers.forEach((handler) => handler(message))
        } catch {
          this.handlers.forEach((handler) => handler({ type: 'error', message: '收到无效消息' }))
        }
      }
    })
  }

  async startMicrophone(): Promise<MediaStream> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) throw new Error('实时连接未建立')
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      })
    } catch (error: any) {
      throw new Error(error?.name === 'NotAllowedError' ? '麦克风权限被拒绝' : '麦克风启动失败')
    }
    this.context = new AudioContext()
    this.source = this.context.createMediaStreamSource(this.stream)
    this.processor = this.context.createScriptProcessor(4096, 1, 1)
    this.sink = this.context.createGain()
    this.sink.gain.value = 0
    this.processor.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0)
      const pcm = this.resampleTo16k(input, this.context?.sampleRate || 48000)
      this.send({ type: 'audio_chunk', audio: this.int16ToBase64(pcm) })
    }
    this.source.connect(this.processor)
    this.processor.connect(this.sink)
    this.sink.connect(this.context.destination)
    this.send({ type: 'audio_state', is_user_speaking: true })
    return this.stream
  }

  stopMicrophone(): void {
    this.send({ type: 'audio_state', is_user_speaking: false })
    this.processor?.disconnect()
    this.source?.disconnect()
    this.sink?.disconnect()
    this.stream?.getTracks().forEach((track) => track.stop())
    void this.context?.close()
    this.processor = null
    this.source = null
    this.sink = null
    this.stream = null
    this.context = null
  }

  sendUserText(text: string): void {
    this.send({ type: 'user_text', text })
  }

  interrupt(responseId: string | null, reason = 'user_speech'): void {
    this.send({ type: 'interrupt', response_id: responseId, reason })
  }

  close(): void {
    this.stopMicrophone()
    this.send({ type: 'close' })
    this.ws?.close()
    this.ws = null
  }

  private send(payload: Record<string, any>): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return
    this.ws.send(JSON.stringify({ ...payload, session_id: this.sessionId }))
  }

  private absoluteWsUrl(path: string): string {
    if (path.startsWith('ws')) return path
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    return `${protocol}//${window.location.host}${path}`
  }

  private resampleTo16k(input: Float32Array, sourceRate: number): Int16Array {
    if (sourceRate === 16000) return this.floatToInt16(input)
    const ratio = sourceRate / 16000
    const length = Math.floor(input.length / ratio)
    const output = new Float32Array(length)
    for (let i = 0; i < length; i += 1) {
      output[i] = input[Math.floor(i * ratio)] || 0
    }
    return this.floatToInt16(output)
  }

  private floatToInt16(input: Float32Array): Int16Array {
    const output = new Int16Array(input.length)
    for (let i = 0; i < input.length; i += 1) {
      const value = Math.max(-1, Math.min(1, input[i]))
      output[i] = value < 0 ? value * 0x8000 : value * 0x7fff
    }
    return output
  }

  private int16ToBase64(input: Int16Array): string {
    const bytes = new Uint8Array(input.buffer)
    let binary = ''
    for (const byte of bytes) binary += String.fromCharCode(byte)
    return btoa(binary)
  }
}

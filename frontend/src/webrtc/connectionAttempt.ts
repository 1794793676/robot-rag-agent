export interface AttemptSession {
  session_id: string
  connection_id: string
  rag_database_id: string
}

export interface ConnectionCandidate {
  open(databaseId: string): Promise<AttemptSession>
  close(): Promise<void>
}

export type AttemptResult =
  | { status: 'connected'; session: AttemptSession }
  | { status: 'stale' }
  | { status: 'failed'; error: unknown }

export class ConnectionAttemptManager<T extends ConnectionCandidate> {
  current: T | null = null
  pending: T | null = null
  private sequence = 0

  async connect(
    databaseId: string,
    candidate: T,
    isCurrent: () => boolean = () => true,
  ): Promise<AttemptResult> {
    const sequence = ++this.sequence
    const previousPending = this.pending
    this.pending = candidate
    if (previousPending && previousPending !== candidate) {
      await previousPending.close()
    }
    try {
      const session = await candidate.open(databaseId)
      if (
        sequence !== this.sequence
        || !isCurrent()
        || session.rag_database_id !== databaseId
        || !session.session_id
        || !session.connection_id
      ) {
        await candidate.close()
        if (this.pending === candidate) this.pending = null
        return { status: 'stale' }
      }
      const previous = this.current
      this.current = candidate
      if (this.pending === candidate) this.pending = null
      if (previous && previous !== candidate) await previous.close()
      if (sequence !== this.sequence || this.current !== candidate) {
        return { status: 'stale' }
      }
      return { status: 'connected', session }
    } catch (error) {
      await candidate.close()
      if (this.pending === candidate) this.pending = null
      if (sequence !== this.sequence) return { status: 'stale' }
      return { status: 'failed', error }
    }
  }

  async disconnect(): Promise<void> {
    this.sequence += 1
    const current = this.current
    const pending = this.pending
    this.current = null
    this.pending = null
    if (pending && pending !== current) await pending.close()
    if (current) await current.close()
  }
}

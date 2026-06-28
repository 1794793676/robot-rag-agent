export interface ConnectionIdentity {
  sessionId: string
  connectionId: string
  ragDatabaseId: string
}

export function hasConnectionIdentity(message: unknown): boolean {
  if (!message || typeof message !== 'object') return false
  return (
    'session_id' in message
    || 'connection_id' in message
    || 'rag_database_id' in message
  )
}

export function matchesActiveConnection(message: any, active: ConnectionIdentity | null): boolean {
  return Boolean(
    active
    && message?.session_id === active.sessionId
    && message?.connection_id === active.connectionId
    && message?.rag_database_id === active.ragDatabaseId
  )
}

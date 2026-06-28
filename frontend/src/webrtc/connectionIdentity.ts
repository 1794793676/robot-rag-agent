export interface ConnectionIdentity {
  sessionId: string
  connectionId: string
  ragDatabaseId: string
}

export interface ConnectionState {
  status: string
  inputEnabled: boolean
  pendingDatabaseId: string | null
}

type ConnectionAction =
  | { type: 'DATABASE_CHANGED'; databaseId: string; wasConnected?: boolean }
  | { type: 'CONNECTED'; databaseId: string }
  | { type: 'CONNECT_FAILED' }

export function reduceConnection(
  state: ConnectionState,
  action: ConnectionAction,
): ConnectionState {
  if (action.type === 'DATABASE_CHANGED') {
    if (action.wasConnected === false) return state
    return {
      status: 'switching_database',
      inputEnabled: false,
      pendingDatabaseId: action.databaseId,
    }
  }
  if (action.type === 'CONNECTED') {
    if (state.pendingDatabaseId && state.pendingDatabaseId !== action.databaseId) return state
    return {
      status: 'connected',
      inputEnabled: true,
      pendingDatabaseId: null,
    }
  }
  if (action.type === 'CONNECT_FAILED') {
    return { ...state, status: 'error', inputEnabled: false }
  }
  return state
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

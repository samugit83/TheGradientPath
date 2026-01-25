'use client'

import { useState, useEffect, useCallback } from 'react'

const SESSION_STORAGE_KEY = 'redamon-session-id'

function generateSessionId(): string {
  const timestamp = Date.now().toString(36)
  const randomPart = Math.random().toString(36).substring(2, 10)
  return `session_${timestamp}_${randomPart}`
}

export function useSession() {
  const [sessionId, setSessionId] = useState<string>('')
  const [mounted, setMounted] = useState(false)

  // Initialize session on mount
  useEffect(() => {
    // Use environment variable if available, otherwise generate new session ID
    const envSessionId = process.env.NEXT_PUBLIC_SESSION_ID
    const newSessionId = envSessionId || generateSessionId()
    setSessionId(newSessionId)
    sessionStorage.setItem(SESSION_STORAGE_KEY, newSessionId)
    setMounted(true)
  }, [])

  const resetSession = useCallback(() => {
    const newSessionId = generateSessionId()
    setSessionId(newSessionId)
    sessionStorage.setItem(SESSION_STORAGE_KEY, newSessionId)
    return newSessionId
  }, [])

  return {
    sessionId,
    resetSession,
    mounted,
  }
}

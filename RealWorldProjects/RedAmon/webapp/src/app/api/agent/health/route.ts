import { NextResponse } from 'next/server'

const AGENT_API_BASE_URL = process.env.AGENT_API_URL || process.env.NEXT_PUBLIC_AGENT_API_URL || 'http://localhost:8080'

export interface AgentHealthResponse {
  status: string
  version: string
  tools_loaded: number
  active_sessions: number
}

export async function GET() {
  try {
    const response = await fetch(`${AGENT_API_BASE_URL}/health`)

    if (!response.ok) {
      return NextResponse.json(
        { error: `Health check failed: ${response.status}` },
        { status: response.status }
      )
    }

    const data: AgentHealthResponse = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('Agent health check error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Health check failed' },
      { status: 500 }
    )
  }
}

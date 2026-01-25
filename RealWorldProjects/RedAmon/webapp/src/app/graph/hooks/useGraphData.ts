import { useQuery } from '@tanstack/react-query'
import { GraphData } from '../types'

async function fetchGraphData(projectId: string): Promise<GraphData> {
  const response = await fetch(`/api/graph?projectId=${projectId}`)
  if (!response.ok) {
    throw new Error('Failed to fetch graph data')
  }
  return response.json()
}

export function useGraphData(projectId: string) {
  return useQuery({
    queryKey: ['graph', projectId],
    queryFn: () => fetchGraphData(projectId),
  })
}

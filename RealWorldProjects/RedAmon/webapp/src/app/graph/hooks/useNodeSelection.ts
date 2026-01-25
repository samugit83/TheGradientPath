import { useState, useCallback } from 'react'
import { GraphNode } from '../types'

interface UseNodeSelectionReturn {
  selectedNode: GraphNode | null
  drawerOpen: boolean
  selectNode: (node: GraphNode) => void
  clearSelection: () => void
}

/**
 * Custom hook for managing node selection state
 */
export function useNodeSelection(): UseNodeSelectionReturn {
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)
  const [drawerOpen, setDrawerOpen] = useState(false)

  const selectNode = useCallback((node: GraphNode) => {
    setSelectedNode(node)
    setDrawerOpen(true)
  }, [])

  const clearSelection = useCallback(() => {
    setDrawerOpen(false)
    setSelectedNode(null)
  }, [])

  return {
    selectedNode,
    drawerOpen,
    selectNode,
    clearSelection,
  }
}

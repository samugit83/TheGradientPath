'use client'

import { GraphData, GraphNode } from '../../types'
import { GraphCanvas2D } from './GraphCanvas2D'
import { GraphCanvas3D } from './GraphCanvas3D'
import styles from './GraphCanvas.module.css'

interface GraphCanvasProps {
  data: GraphData | undefined
  isLoading: boolean
  error: Error | null
  projectId: string
  is3D: boolean
  width: number
  height: number
  showLabels: boolean
  selectedNode: GraphNode | null
  onNodeClick: (node: GraphNode) => void
  isDark?: boolean
}

export function GraphCanvas({
  data,
  isLoading,
  error,
  projectId,
  is3D,
  width,
  height,
  showLabels,
  selectedNode,
  onNodeClick,
  isDark = true,
}: GraphCanvasProps) {
  if (isLoading) {
    return <div className={styles.loading}>Loading graph data...</div>
  }

  if (error) {
    return (
      <div className={styles.error}>
        Error: {error instanceof Error ? error.message : 'Unknown error'}
      </div>
    )
  }

  if (!data || data.nodes.length === 0) {
    return (
      <div className={styles.empty}>
        No data found for project: {projectId}
      </div>
    )
  }

  // Use key to force re-mount when theme changes (ForceGraph doesn't update backgroundColor dynamically)
  const themeKey = isDark ? 'dark' : 'light'

  if (is3D) {
    return (
      <div className={styles.wrapper}>
        <GraphCanvas3D
          key={themeKey}
          data={data}
          width={width}
          height={height}
          showLabels={showLabels}
          selectedNode={selectedNode}
          onNodeClick={onNodeClick}
          isDark={isDark}
        />
      </div>
    )
  }

  return (
    <div className={styles.wrapper}>
      <GraphCanvas2D
        key={themeKey}
        data={data}
        width={width}
        height={height}
        showLabels={showLabels}
        selectedNode={selectedNode}
        onNodeClick={onNodeClick}
        isDark={isDark}
      />
    </div>
  )
}

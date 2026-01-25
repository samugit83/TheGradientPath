'use client'

import { useState, useRef, useCallback } from 'react'
import { GraphToolbar } from './components/GraphToolbar'
import { GraphCanvas } from './components/GraphCanvas'
import { NodeDrawer } from './components/NodeDrawer'
import { AIAssistantDrawer } from './components/AIAssistantDrawer'
import { PageBottomBar } from './components/PageBottomBar'
import { useGraphData, useDimensions, useNodeSelection } from './hooks'
import { useTheme, useSession } from '@/hooks'
import config from '@/config.json'
import styles from './page.module.css'

export default function GraphPage() {
  // Use environment variables as primary source, fallback to config.json
  const userId = process.env.NEXT_PUBLIC_USER_ID || config.user_id
  const projectId = process.env.NEXT_PUBLIC_PROJECT_ID || config.project_id

  const [is3D, setIs3D] = useState(true)
  const [showLabels, setShowLabels] = useState(false)
  const [isAIOpen, setIsAIOpen] = useState(false)
  const contentRef = useRef<HTMLDivElement>(null)

  const { data, isLoading, error } = useGraphData(projectId)
  const { selectedNode, drawerOpen, selectNode, clearSelection } = useNodeSelection()
  const dimensions = useDimensions(contentRef)
  const { isDark } = useTheme()
  const { sessionId, resetSession } = useSession()

  const handleToggleAI = useCallback(() => {
    setIsAIOpen((prev) => !prev)
  }, [])

  const handleCloseAI = useCallback(() => {
    setIsAIOpen(false)
  }, [])

  return (
    <div className={styles.page}>
      <GraphToolbar
        projectId={projectId}
        is3D={is3D}
        showLabels={showLabels}
        onToggle3D={setIs3D}
        onToggleLabels={setShowLabels}
        onToggleAI={handleToggleAI}
        isAIOpen={isAIOpen}
      />

      <div className={styles.body}>
        <NodeDrawer
          node={selectedNode}
          isOpen={drawerOpen}
          onClose={clearSelection}
        />

        <div ref={contentRef} className={styles.content}>
          <GraphCanvas
            data={data}
            isLoading={isLoading}
            error={error}
            projectId={projectId}
            is3D={is3D}
            width={dimensions.width}
            height={dimensions.height}
            showLabels={showLabels}
            selectedNode={selectedNode}
            onNodeClick={selectNode}
            isDark={isDark}
          />
        </div>
      </div>

      <AIAssistantDrawer
        isOpen={isAIOpen}
        onClose={handleCloseAI}
        userId={userId}
        projectId={projectId}
        sessionId={sessionId || ''}
        onResetSession={resetSession}
      />

      <PageBottomBar data={data} is3D={is3D} showLabels={showLabels} />
    </div>
  )
}

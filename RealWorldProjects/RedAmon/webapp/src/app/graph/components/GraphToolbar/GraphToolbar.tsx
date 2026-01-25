'use client'

import { Sparkles } from 'lucide-react'
import { Toggle } from '@/components/ui'
import styles from './GraphToolbar.module.css'

interface GraphToolbarProps {
  projectId: string
  is3D: boolean
  showLabels: boolean
  onToggle3D: (value: boolean) => void
  onToggleLabels: (value: boolean) => void
  onToggleAI?: () => void
  isAIOpen?: boolean
}

export function GraphToolbar({
  projectId,
  is3D,
  showLabels,
  onToggle3D,
  onToggleLabels,
  onToggleAI,
  isAIOpen = false,
}: GraphToolbarProps) {
  return (
    <div className={styles.toolbar}>
      <div className={styles.section}>
        <span className={styles.sectionLabel}>View Mode</span>
        <Toggle
          checked={is3D}
          onChange={onToggle3D}
          labelOff="2D"
          labelOn="3D"
          aria-label="Toggle 2D/3D view"
        />
      </div>

      <div className={styles.divider} />

      <div className={styles.section}>
        <span className={styles.sectionLabel}>Labels</span>
        <Toggle
          checked={showLabels}
          onChange={onToggleLabels}
          labelOff="Off"
          labelOn="On"
          aria-label="Toggle labels"
        />
      </div>

      <div className={styles.spacer} />

      <div className={styles.projectBadge}>
        <span className={styles.projectLabel}>Project:</span>
        <span className={styles.projectId}>{projectId}</span>
      </div>

      <div className={styles.divider} />

      <button
        className={`${styles.aiButton} ${isAIOpen ? styles.aiButtonActive : ''}`}
        onClick={onToggleAI}
        aria-label="Toggle RedAmon Agent"
        aria-expanded={isAIOpen}
        title="RedAmon Agent"
      >
        <Sparkles size={14} />
        <span>RedAmon Agent</span>
      </button>
    </div>
  )
}

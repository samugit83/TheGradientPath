'use client'

import { Drawer } from '@/components/ui'
import { GraphNode } from '../../types'
import { getNodeColor } from '../../utils'
import { formatPropertyValue } from '../../utils/formatters'
import styles from './NodeDrawer.module.css'

interface NodeDrawerProps {
  node: GraphNode | null
  isOpen: boolean
  onClose: () => void
}

export function NodeDrawer({ node, isOpen, onClose }: NodeDrawerProps) {
  // Sort properties with created_at and updated_at at the bottom
  const sortedProperties = node
    ? Object.entries(node.properties || {}).sort(([a], [b]) => {
        const bottomKeys = ['created_at', 'updated_at']
        const aIsBottom = bottomKeys.includes(a)
        const bIsBottom = bottomKeys.includes(b)
        if (aIsBottom && !bIsBottom) return 1
        if (!aIsBottom && bIsBottom) return -1
        if (aIsBottom && bIsBottom) return bottomKeys.indexOf(a) - bottomKeys.indexOf(b)
        return 0
      })
    : []

  return (
    <Drawer
      isOpen={isOpen}
      onClose={onClose}
      position="left"
      mode="push"
      title={node ? `${node.type}: ${node.name}` : undefined}
    >
      {node && (
        <>
          <div className={styles.section}>
            <h3 className={styles.sectionTitleBasicInfo}>Basic Info</h3>
            <div className={styles.propertyRow}>
              <span className={styles.propertyKey}>Type</span>
              <span
                className={styles.propertyBadge}
                style={{ backgroundColor: getNodeColor(node) }}
              >
                {node.type}
              </span>
            </div>
            <div className={styles.propertyRow}>
              <span className={styles.propertyKey}>ID</span>
              <span className={styles.propertyValue}>{node.id}</span>
            </div>
            <div className={styles.propertyRow}>
              <span className={styles.propertyKey}>Name</span>
              <span className={styles.propertyValue}>{node.name}</span>
            </div>
          </div>

          <div className={styles.section}>
            <h3 className={styles.sectionTitleProperties}>Properties</h3>
            {sortedProperties.map(([key, value]) => (
              <div key={key} className={styles.propertyRow}>
                <span className={styles.propertyKey}>{key}</span>
                <span className={styles.propertyValue}>
                  {formatPropertyValue(value)}
                </span>
              </div>
            ))}
            {sortedProperties.length === 0 && (
              <p className={styles.emptyProperties}>No additional properties</p>
            )}
          </div>
        </>
      )}
    </Drawer>
  )
}

'use client'

import { ReactNode } from 'react'
import styles from './Drawer.module.css'

export interface DrawerProps {
  /** Whether the drawer is open */
  isOpen: boolean
  /** Callback when drawer should close */
  onClose: () => void
  /** Position of the drawer */
  position?: 'left' | 'right'
  /** Behavior mode: 'push' shrinks adjacent content, 'overlay' slides over content */
  mode?: 'push' | 'overlay'
  /** Width of the drawer (CSS value) */
  width?: string
  /** Title shown in drawer header */
  title?: ReactNode
  /** Content of the drawer */
  children: ReactNode
  /** Additional class name */
  className?: string
}

export function Drawer({
  isOpen,
  onClose,
  position = 'left',
  mode = 'push',
  width = '300px',
  title,
  children,
  className = '',
}: DrawerProps) {
  const positionClass = position === 'left' ? styles.drawerLeft : styles.drawerRight
  const modeClass = mode === 'overlay' ? styles.drawerOverlay : styles.drawerPush

  return (
    <div
      className={`${styles.drawer} ${positionClass} ${modeClass} ${isOpen ? styles.drawerOpen : ''} ${className}`}
      style={{ '--drawer-custom-width': width } as React.CSSProperties}
    >
      {title && (
        <div className={styles.drawerHeader}>
          <h2 className={styles.drawerTitle}>{title}</h2>
          <button
            className={styles.drawerClose}
            onClick={onClose}
            aria-label="Close drawer"
          >
            ×
          </button>
        </div>
      )}
      <div className={styles.drawerContent}>{children}</div>
    </div>
  )
}

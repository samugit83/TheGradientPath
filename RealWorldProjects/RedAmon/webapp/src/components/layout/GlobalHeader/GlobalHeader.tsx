'use client'

import { Search, Bell, Settings, ChevronDown, Crosshair } from 'lucide-react'
import { ThemeToggle } from '@/components/ThemeToggle'
import styles from './GlobalHeader.module.css'

export function GlobalHeader() {
  return (
    <header className={styles.header}>
      <div className={styles.logo}>
        <Crosshair size={18} className={styles.logoSvg} />
        <span className={styles.logoText}>
          <span className={styles.logoAccent}>Red</span>Amon
        </span>
      </div>

      <div className={styles.spacer} />

      {/* Search - Mock */}
      <div className={styles.search}>
        <Search size={14} />
        <input
          type="text"
          placeholder="Search..."
          className={styles.searchInput}
          disabled
        />
      </div>

      <div className={styles.actions}>
        {/* Notifications - Mock */}
        <button className={styles.iconButton} title="Notifications">
          <Bell size={16} />
          <span className={styles.badge}>3</span>
        </button>

        {/* Settings - Mock */}
        <button className={styles.iconButton} title="Settings">
          <Settings size={16} />
        </button>

        <div className={styles.divider} />

        <ThemeToggle />

        <div className={styles.divider} />

        {/* User Menu - Mock */}
        <button className={styles.userButton}>
          <div className={styles.avatar}>
            <span>SA</span>
          </div>
          <span className={styles.userName}>Admin</span>
          <ChevronDown size={14} />
        </button>
      </div>
    </header>
  )
}

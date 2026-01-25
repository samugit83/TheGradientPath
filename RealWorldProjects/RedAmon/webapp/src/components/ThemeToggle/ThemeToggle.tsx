'use client'

import { Sun, Moon } from 'lucide-react'
import { useTheme } from '@/hooks/useTheme'
import styles from './ThemeToggle.module.css'

export function ThemeToggle() {
  const { resolvedTheme, toggleTheme, mounted } = useTheme()

  // Avoid hydration mismatch by not rendering until mounted
  if (!mounted) {
    return (
      <div className={styles.container}>
        <span className={styles.icon}>
          <Sun size={14} />
        </span>
        <button className={styles.toggle} aria-label="Toggle theme" disabled>
          <span className={styles.knob} />
        </button>
        <span className={styles.icon}>
          <Moon size={14} />
        </span>
      </div>
    )
  }

  const isDark = resolvedTheme === 'dark'

  return (
    <div className={styles.container}>
      <span className={`${styles.icon} ${!isDark ? styles.iconActive : ''}`}>
        <Sun size={14} />
      </span>
      <button
        className={styles.toggle}
        onClick={toggleTheme}
        aria-label={`Switch to ${isDark ? 'light' : 'dark'} mode`}
        role="switch"
        aria-checked={isDark}
      >
        <span className={`${styles.knob} ${isDark ? styles.knobRight : ''}`} />
      </button>
      <span className={`${styles.icon} ${isDark ? styles.iconActive : ''}`}>
        <Moon size={14} />
      </span>
    </div>
  )
}

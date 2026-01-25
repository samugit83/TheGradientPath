'use client'

import styles from './Footer.module.css'

export function Footer() {
  const currentYear = new Date().getFullYear()

  return (
    <footer className={styles.footer}>
      <div className={styles.content}>
        <span className={styles.copyright}>
          © {currentYear} RedAmon. All rights reserved.
        </span>
        <span className={styles.version}>v1.0.0</span>
      </div>
    </footer>
  )
}

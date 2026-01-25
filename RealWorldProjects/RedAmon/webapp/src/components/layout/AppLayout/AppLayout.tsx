'use client'

import { GlobalHeader } from '../GlobalHeader'
import { NavigationBar } from '../NavigationBar'
import { Footer } from '../Footer'
import styles from './AppLayout.module.css'

interface AppLayoutProps {
  children: React.ReactNode
}

export function AppLayout({ children }: AppLayoutProps) {
  return (
    <div className={styles.layout}>
      <GlobalHeader />
      <NavigationBar />
      <main className={styles.main}>{children}</main>
      <Footer />
    </div>
  )
}

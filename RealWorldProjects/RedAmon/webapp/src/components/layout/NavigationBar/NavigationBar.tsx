'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { Network, ShieldCheck, Target, ClipboardList } from 'lucide-react'
import styles from './NavigationBar.module.css'

interface NavItem {
  label: string
  href: string
  icon: React.ReactNode
  enabled: boolean
}

const navItems: NavItem[] = [
  {
    label: 'Graph Map',
    href: '/graph',
    icon: <Network size={16} />,
    enabled: true,
  },
  {
    label: 'Vulnerabilities',
    href: '/vulnerabilities',
    icon: <ShieldCheck size={16} />,
    enabled: false,
  },
  {
    label: 'MITRE ATT&CK',
    href: '/mitre',
    icon: <Target size={16} />,
    enabled: false,
  },
  {
    label: 'Actions Log',
    href: '/actions',
    icon: <ClipboardList size={16} />,
    enabled: false,
  },
]

export function NavigationBar() {
  const pathname = usePathname()

  return (
    <nav className={styles.nav}>
      <ul className={styles.navList}>
        {navItems.map((item) => {
          const isActive = pathname === item.href || pathname.startsWith(`${item.href}/`)

          if (!item.enabled) {
            return (
              <li key={item.href}>
                <span className={`${styles.navItem} ${styles.navItemDisabled}`}>
                  <span className={styles.navIcon}>{item.icon}</span>
                  <span className={styles.navLabel}>{item.label}</span>
                  <span className={styles.comingSoon}>Soon</span>
                </span>
              </li>
            )
          }

          return (
            <li key={item.href}>
              <Link
                href={item.href}
                className={`${styles.navItem} ${isActive ? styles.navItemActive : ''}`}
              >
                <span className={styles.navIcon}>{item.icon}</span>
                <span className={styles.navLabel}>{item.label}</span>
              </Link>
            </li>
          )
        })}
      </ul>
    </nav>
  )
}

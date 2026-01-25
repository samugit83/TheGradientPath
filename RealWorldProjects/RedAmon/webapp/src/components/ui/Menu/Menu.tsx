'use client'

import {
  useState,
  useRef,
  useCallback,
  useEffect,
  ReactNode,
  KeyboardEvent,
  createContext,
  useContext,
} from 'react'
import { createPortal } from 'react-dom'
import styles from './Menu.module.css'

interface MenuContextValue {
  isOpen: boolean
  close: () => void
}

const MenuContext = createContext<MenuContextValue | null>(null)

interface MenuProps {
  /** The trigger element */
  trigger: ReactNode
  /** Menu items */
  children: ReactNode
  /** Alignment relative to trigger */
  align?: 'left' | 'right'
}

export function Menu({ trigger, children, align = 'left' }: MenuProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [position, setPosition] = useState({ top: 0, left: 0 })
  const triggerRef = useRef<HTMLDivElement>(null)
  const menuRef = useRef<HTMLDivElement>(null)

  const close = useCallback(() => {
    setIsOpen(false)
  }, [])

  const toggle = useCallback(() => {
    setIsOpen((prev) => !prev)
  }, [])

  // Calculate position when opening
  useEffect(() => {
    if (isOpen && triggerRef.current) {
      const rect = triggerRef.current.getBoundingClientRect()
      const gap = 4

      let left = rect.left
      if (align === 'right') {
        left = rect.right
      }

      setPosition({
        top: rect.bottom + gap,
        left,
      })
    }
  }, [isOpen, align])

  // Handle click outside
  useEffect(() => {
    if (!isOpen) return

    const handleClickOutside = (e: MouseEvent) => {
      if (
        menuRef.current &&
        !menuRef.current.contains(e.target as Node) &&
        triggerRef.current &&
        !triggerRef.current.contains(e.target as Node)
      ) {
        close()
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [isOpen, close])

  // Handle escape key
  useEffect(() => {
    if (!isOpen) return

    const handleEscape = (e: globalThis.KeyboardEvent) => {
      if (e.key === 'Escape') {
        close()
      }
    }

    document.addEventListener('keydown', handleEscape)
    return () => document.removeEventListener('keydown', handleEscape)
  }, [isOpen, close])

  // Keyboard navigation
  const handleKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (!isOpen || !menuRef.current) return

    const items = menuRef.current.querySelectorAll<HTMLButtonElement>(
      '[role="menuitem"]:not([disabled])'
    )
    const currentIndex = Array.from(items).findIndex(
      (item) => item === document.activeElement
    )

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault()
        const nextIndex = currentIndex < items.length - 1 ? currentIndex + 1 : 0
        items[nextIndex]?.focus()
        break
      case 'ArrowUp':
        e.preventDefault()
        const prevIndex = currentIndex > 0 ? currentIndex - 1 : items.length - 1
        items[prevIndex]?.focus()
        break
      case 'Home':
        e.preventDefault()
        items[0]?.focus()
        break
      case 'End':
        e.preventDefault()
        items[items.length - 1]?.focus()
        break
    }
  }

  const alignClass = align === 'right' ? styles.menuRight : ''

  return (
    <MenuContext.Provider value={{ isOpen, close }}>
      <div
        ref={triggerRef}
        className={styles.trigger}
        onClick={toggle}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault()
            toggle()
          }
        }}
        role="button"
        tabIndex={0}
        aria-haspopup="menu"
        aria-expanded={isOpen}
      >
        {trigger}
      </div>

      {isOpen &&
        typeof document !== 'undefined' &&
        createPortal(
          <div
            ref={menuRef}
            className={`${styles.menu} ${alignClass}`}
            style={{
              top: position.top,
              left: position.left,
            }}
            role="menu"
            onKeyDown={handleKeyDown}
          >
            {children}
          </div>,
          document.body
        )}
    </MenuContext.Provider>
  )
}

interface MenuItemProps {
  /** Item content */
  children: ReactNode
  /** Click handler */
  onClick?: () => void
  /** Icon to show before label */
  icon?: ReactNode
  /** Whether this is a destructive action */
  destructive?: boolean
  /** Whether item is disabled */
  disabled?: boolean
}

export function MenuItem({
  children,
  onClick,
  icon,
  destructive = false,
  disabled = false,
}: MenuItemProps) {
  const context = useContext(MenuContext)

  const handleClick = () => {
    if (disabled) return
    onClick?.()
    context?.close()
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLButtonElement>) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      handleClick()
    }
  }

  const classes = [
    styles.menuItem,
    destructive && styles.menuItemDestructive,
    disabled && styles.menuItemDisabled,
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <button
      type="button"
      className={classes}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      role="menuitem"
      disabled={disabled}
      tabIndex={-1}
    >
      {icon && <span className={styles.menuItemIcon}>{icon}</span>}
      {children}
    </button>
  )
}

export function MenuDivider() {
  return <div className={styles.menuDivider} role="separator" />
}

interface MenuLabelProps {
  children: ReactNode
}

export function MenuLabel({ children }: MenuLabelProps) {
  return <div className={styles.menuLabel}>{children}</div>
}

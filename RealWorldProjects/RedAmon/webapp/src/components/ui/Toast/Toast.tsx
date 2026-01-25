'use client'

import {
  createContext,
  useContext,
  useState,
  useCallback,
  ReactNode,
  useEffect,
} from 'react'
import { createPortal } from 'react-dom'
import { X, CheckCircle, AlertCircle, AlertTriangle, Info } from 'lucide-react'
import styles from './Toast.module.css'

type ToastType = 'success' | 'error' | 'warning' | 'info'

interface Toast {
  id: string
  type: ToastType
  title?: string
  message: string
  duration?: number
}

interface ToastContextValue {
  toasts: Toast[]
  addToast: (toast: Omit<Toast, 'id'>) => void
  removeToast: (id: string) => void
  success: (message: string, title?: string) => void
  error: (message: string, title?: string) => void
  warning: (message: string, title?: string) => void
  info: (message: string, title?: string) => void
}

const ToastContext = createContext<ToastContextValue | null>(null)

export function useToast() {
  const context = useContext(ToastContext)
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider')
  }
  return context
}

interface ToastProviderProps {
  children: ReactNode
  /** Default duration in ms (0 = no auto-dismiss) */
  defaultDuration?: number
  /** Maximum number of toasts to show */
  maxToasts?: number
}

export function ToastProvider({
  children,
  defaultDuration = 5000,
  maxToasts = 5,
}: ToastProviderProps) {
  const [toasts, setToasts] = useState<Toast[]>([])

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id))
  }, [])

  const addToast = useCallback(
    (toast: Omit<Toast, 'id'>) => {
      const id = Math.random().toString(36).substr(2, 9)
      const newToast: Toast = {
        ...toast,
        id,
        duration: toast.duration ?? defaultDuration,
      }

      setToasts((prev) => {
        const updated = [...prev, newToast]
        // Keep only the last maxToasts
        return updated.slice(-maxToasts)
      })
    },
    [defaultDuration, maxToasts]
  )

  const success = useCallback(
    (message: string, title?: string) => {
      addToast({ type: 'success', message, title })
    },
    [addToast]
  )

  const error = useCallback(
    (message: string, title?: string) => {
      addToast({ type: 'error', message, title })
    },
    [addToast]
  )

  const warning = useCallback(
    (message: string, title?: string) => {
      addToast({ type: 'warning', message, title })
    },
    [addToast]
  )

  const info = useCallback(
    (message: string, title?: string) => {
      addToast({ type: 'info', message, title })
    },
    [addToast]
  )

  return (
    <ToastContext.Provider
      value={{ toasts, addToast, removeToast, success, error, warning, info }}
    >
      {children}
      <ToastContainer toasts={toasts} onRemove={removeToast} />
    </ToastContext.Provider>
  )
}

interface ToastContainerProps {
  toasts: Toast[]
  onRemove: (id: string) => void
}

function ToastContainer({ toasts, onRemove }: ToastContainerProps) {
  if (typeof document === 'undefined' || toasts.length === 0) {
    return null
  }

  return createPortal(
    <div className={styles.container}>
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} onRemove={onRemove} />
      ))}
    </div>,
    document.body
  )
}

interface ToastItemProps {
  toast: Toast
  onRemove: (id: string) => void
}

function ToastItem({ toast, onRemove }: ToastItemProps) {
  const [isExiting, setIsExiting] = useState(false)

  const handleClose = useCallback(() => {
    setIsExiting(true)
    setTimeout(() => onRemove(toast.id), 150)
  }, [onRemove, toast.id])

  useEffect(() => {
    if (toast.duration && toast.duration > 0) {
      const timer = setTimeout(handleClose, toast.duration)
      return () => clearTimeout(timer)
    }
  }, [toast.duration, handleClose])

  const Icon = {
    success: CheckCircle,
    error: AlertCircle,
    warning: AlertTriangle,
    info: Info,
  }[toast.type]

  const typeClass = {
    success: styles.toastSuccess,
    error: styles.toastError,
    warning: styles.toastWarning,
    info: styles.toastInfo,
  }[toast.type]

  return (
    <div
      className={`${styles.toast} ${typeClass} ${isExiting ? styles.toastExiting : ''}`}
      role="alert"
    >
      <Icon className={styles.icon} size={18} />
      <div className={styles.content}>
        {toast.title && <div className={styles.title}>{toast.title}</div>}
        <div className={styles.message}>{toast.message}</div>
      </div>
      <button
        type="button"
        className={styles.closeButton}
        onClick={handleClose}
        aria-label="Dismiss"
      >
        <X size={14} />
      </button>
    </div>
  )
}

/**
 * Tool Execution Card Component
 *
 * Displays tool execution details with streaming output.
 * Compact design with key-value arg display.
 */

'use client'

import { useState, useEffect } from 'react'
import { Wrench, ChevronDown, ChevronRight, Copy, Check, Loader2, CheckCircle2, XCircle } from 'lucide-react'
import styles from './ToolExecutionCard.module.css'
import type { ToolExecutionItem } from './AgentTimeline'

interface ToolExecutionCardProps {
  item: ToolExecutionItem
  isExpanded: boolean
  onToggleExpand: () => void
}

export function ToolExecutionCard({ item, isExpanded, onToggleExpand }: ToolExecutionCardProps) {
  const [copied, setCopied] = useState(false)
  const [duration, setDuration] = useState(0)

  // Calculate duration for running tools
  useEffect(() => {
    if (item.status === 'running') {
      const interval = setInterval(() => {
        const elapsed = Date.now() - item.timestamp.getTime()
        setDuration(Math.floor(elapsed / 1000))
      }, 1000)
      return () => clearInterval(interval)
    } else if (item.duration) {
      setDuration(Math.floor(item.duration / 1000))
    }
  }, [item.status, item.timestamp, item.duration])

  const handleCopy = async () => {
    try {
      const data = {
        tool_name: item.tool_name,
        tool_args: item.tool_args,
        output: item.output_chunks.join(''),
        status: item.status,
      }
      await navigator.clipboard.writeText(JSON.stringify(data, null, 2))
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // Silent fail
    }
  }

  const getStatusIcon = () => {
    switch (item.status) {
      case 'running':
        return <Loader2 size={14} className={`${styles.statusIcon} ${styles.spinner}`} />
      case 'success':
        return <CheckCircle2 size={14} className={`${styles.statusIcon} ${styles.successIcon}`} />
      case 'error':
        return <XCircle size={14} className={`${styles.statusIcon} ${styles.errorIcon}`} />
    }
  }

  const getStatusText = () => {
    switch (item.status) {
      case 'running':
        return `Running... (${duration}s)`
      case 'success':
        return `Completed (${duration}s)`
      case 'error':
        return 'Failed'
    }
  }

  const getStatusClass = () => {
    switch (item.status) {
      case 'running':
        return styles.statusRunning
      case 'success':
        return styles.statusSuccess
      case 'error':
        return styles.statusError
    }
  }

  // Render tool arguments as key-value pairs
  const renderToolArgs = () => {
    if (!item.tool_args || Object.keys(item.tool_args).length === 0) {
      return null
    }

    return Object.entries(item.tool_args).map(([key, value]) => {
      const valueStr = typeof value === 'string' ? value : JSON.stringify(value)
      return (
        <div key={key} className={styles.argItem}>
          <span className={styles.argKey}>{key}:</span>
          <span className={styles.argValue}>{valueStr}</span>
        </div>
      )
    })
  }

  return (
    <div className={`${styles.card} ${getStatusClass()}`}>
      <div className={styles.cardHeaderWrapper} onClick={onToggleExpand}>
        <div className={styles.cardHeaderTop}>
          <div className={styles.cardIcon}>
            <Wrench size={14} className={styles.toolIcon} />
          </div>
          <span className={styles.titleText}>{item.tool_name}</span>
          <div className={styles.cardActions}>
            <div className={styles.statusBadge}>
              {getStatusIcon()}
              <span>{getStatusText()}</span>
            </div>
            <button
              className={styles.copyButton}
              onClick={(e) => {
                e.stopPropagation()
                handleCopy()
              }}
              title="Copy JSON"
            >
              {copied ? <Check size={12} /> : <Copy size={12} />}
            </button>
            <button className={styles.expandButton}>
              {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            </button>
          </div>
        </div>
        {!isExpanded && (
          <div className={styles.argsPreview}>
            {renderToolArgs()}
          </div>
        )}
      </div>

      {isExpanded && (
        <div className={styles.cardContent}>
          {/* Tool Arguments (expanded view) */}
          {item.tool_args && Object.keys(item.tool_args).length > 0 && (
            <div className={styles.section}>
              <div className={styles.sectionLabel}>Arguments</div>
              <div className={styles.sectionContent}>
                <div className={styles.argsExpanded}>
                  {Object.entries(item.tool_args).map(([key, value]) => {
                    const valueStr = typeof value === 'string' ? value : JSON.stringify(value, null, 2)
                    return (
                      <div key={key} className={styles.argItemExpanded}>
                        <span className={styles.argKeyExpanded}>{key}:</span>
                        <pre className={styles.argValueExpanded}>{valueStr}</pre>
                      </div>
                    )
                  })}
                </div>
              </div>
            </div>
          )}

          {/* Raw Output */}
          {item.output_chunks.length > 0 && (
            <div className={styles.section}>
              <div className={styles.sectionLabel}>
                Raw Output
                {item.status === 'running' && (
                  <span className={styles.streamingLabel}>(streaming)</span>
                )}
              </div>
              <div className={styles.sectionContent}>
                <pre className={styles.codeBlock}>
                  <code>
                    {item.output_chunks.join('')}
                    {item.status === 'running' && (
                      <span className={styles.cursor}>▋</span>
                    )}
                  </code>
                </pre>
              </div>
            </div>
          )}

          {/* Analysis Summary */}
          {item.final_output && (
            <div className={styles.section}>
              <div className={styles.sectionLabel}>Analysis</div>
              <div className={styles.sectionContent}>
                <p className={styles.text}>{item.final_output}</p>
              </div>
            </div>
          )}

          {/* Actionable Findings */}
          {item.actionable_findings && item.actionable_findings.length > 0 && (
            <div className={styles.section}>
              <div className={styles.sectionLabel}>Actionable Findings</div>
              <div className={styles.sectionContent}>
                <ul className={styles.findingsList}>
                  {item.actionable_findings.map((finding, index) => (
                    <li key={index} className={styles.findingItem}>{finding}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Recommended Next Steps */}
          {item.recommended_next_steps && item.recommended_next_steps.length > 0 && (
            <div className={styles.section}>
              <div className={styles.sectionLabel}>Recommended Next Steps</div>
              <div className={styles.sectionContent}>
                <ul className={styles.stepsList}>
                  {item.recommended_next_steps.map((step, index) => (
                    <li key={index} className={styles.stepItem}>{step}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Progress Bar for Running Tools */}
          {item.status === 'running' && (
            <div className={styles.section}>
              <div className={styles.progressBar}>
                <div className={styles.progressBarFill} />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

/**
 * Agent Timeline Component
 *
 * Beautiful, interactive timeline showing agent's thinking process and tool executions.
 * Inspired by Claude Code's execution timeline UI.
 */

'use client'

import { useState } from 'react'
import styles from './AgentTimeline.module.css'
import { ThinkingCard } from './ThinkingCard'
import { ToolExecutionCard } from './ToolExecutionCard'
import type { TodoItem } from '@/lib/websocket-types'

export interface ThinkingItem {
  type: 'thinking'
  id: string
  timestamp: Date
  thought: string
  reasoning: string
  action: string
  tool_name?: string
  tool_args?: Record<string, unknown>
  phase_transition?: Record<string, unknown>
  user_question?: Record<string, unknown>
  updated_todo_list: TodoItem[]
}

export interface ToolExecutionItem {
  type: 'tool_execution'
  id: string
  timestamp: Date
  tool_name: string
  tool_args: Record<string, unknown>
  status: 'running' | 'success' | 'error'
  output_chunks: string[]
  final_output?: string
  duration?: number
  actionable_findings?: string[]
  recommended_next_steps?: string[]
}

export type TimelineItem = ThinkingItem | ToolExecutionItem

export interface AgentTimelineProps {
  items: TimelineItem[]
  isStreaming: boolean
  onItemExpand?: (itemId: string) => void
}

export function AgentTimeline({ items, isStreaming, onItemExpand }: AgentTimelineProps) {
  const [expandedItems, setExpandedItems] = useState<Set<string>>(new Set())

  const toggleExpand = (itemId: string) => {
    setExpandedItems(prev => {
      const newSet = new Set(prev)
      if (newSet.has(itemId)) {
        newSet.delete(itemId)
      } else {
        newSet.add(itemId)
      }
      return newSet
    })
    onItemExpand?.(itemId)
  }

  if (items.length === 0) {
    return null
  }

  return (
    <div className={styles.timelineItems}>
      {items.map((item, index) => {
        const isExpanded = expandedItems.has(item.id)
        const isLast = index === items.length - 1
        const isLastAndStreaming = isLast && isStreaming

        return (
          <div
            key={item.id}
            className={`${styles.timelineItemWrapper} ${isLastAndStreaming ? styles.streaming : ''}`}
          >
            {/* Timeline connector line */}
            {!isLast && <div className={styles.timelineConnector} />}

            {/* Render appropriate card based on type */}
            {item.type === 'thinking' ? (
              <ThinkingCard
                item={item}
                isExpanded={isExpanded}
                onToggleExpand={() => toggleExpand(item.id)}
              />
            ) : (
              <ToolExecutionCard
                item={item}
                isExpanded={isExpanded}
                onToggleExpand={() => toggleExpand(item.id)}
              />
            )}
          </div>
        )
      })}
    </div>
  )
}

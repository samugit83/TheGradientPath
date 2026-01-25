'use client'

import { useRef, useEffect } from 'react'
import dynamic from 'next/dynamic'
import { GraphData, GraphNode, GraphLink } from '../../types'
import { getNodeColor, getNodeSize, getGlowLevel } from '../../utils'
import { getLinkColor, getLinkWidth2D, getParticleWidth } from '../../utils/linkHelpers'
import {
  LINK_COLORS,
  LINK_SIZES,
  BASE_SIZES,
  BACKGROUND_COLORS,
  SELECTION_COLORS,
  FORCE_CONFIG,
  ANIMATION_CONFIG,
  ZOOM_CONFIG,
} from '../../config'
import { hasHighSeverityNodes } from '../../utils/nodeHelpers'
import { useAnimationFrame } from '../../hooks'

const ForceGraph2D = dynamic(() => import('react-force-graph-2d'), {
  ssr: false,
})

interface GraphCanvas2DProps {
  data: GraphData
  width: number
  height: number
  showLabels: boolean
  selectedNode: GraphNode | null
  onNodeClick: (node: GraphNode) => void
  isDark?: boolean
}

export function GraphCanvas2D({
  data,
  width,
  height,
  showLabels,
  selectedNode,
  onNodeClick,
  isDark = true,
}: GraphCanvas2DProps) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const graphRef = useRef<any>(null)
  const animationTimeRef = useRef<number>(0)

  // Set up collision detection
  useEffect(() => {
    const timer = setTimeout(() => {
      const fg = graphRef.current
      if (!fg) return

      const d3 = require('d3-force')
      fg.d3Force(
        'collide',
        d3
          .forceCollide()
          .radius(FORCE_CONFIG.collisionRadius)
          .strength(FORCE_CONFIG.collisionStrength)
          .iterations(FORCE_CONFIG.collisionIterations)
      )
      fg.d3ReheatSimulation()
    }, ANIMATION_CONFIG.initDelay)

    return () => clearTimeout(timer)
  }, [data])

  // Animation loop for pulsing glow effect
  const hasHighSeverity = hasHighSeverityNodes(data.nodes)

  useAnimationFrame(
    (time) => {
      animationTimeRef.current = time
      const fg = graphRef.current
      if (fg) {
        if (typeof fg._rerender === 'function') {
          fg._rerender()
        } else if (typeof fg.refresh === 'function') {
          fg.refresh()
        }
      }
    },
    hasHighSeverity
  )

  const selectedNodeId = selectedNode?.id

  return (
    <ForceGraph2D
      ref={graphRef}
      graphData={data}
      nodeLabel={(node) => `${(node as GraphNode).name} (${(node as GraphNode).type})`}
      nodeRelSize={BASE_SIZES.node2D}
      linkLabel={(link) => (link as GraphLink).type}
      linkColor={(link) => getLinkColor(link as GraphLink, selectedNodeId)}
      linkDirectionalArrowColor={(link) => getLinkColor(link as GraphLink, selectedNodeId)}
      linkWidth={(link) => getLinkWidth2D(link as GraphLink, selectedNodeId)}
      linkDirectionalParticles={LINK_SIZES.particleCount}
      linkDirectionalParticleWidth={(link) => getParticleWidth(link as GraphLink, selectedNodeId)}
      linkDirectionalParticleColor={() => LINK_COLORS.particle}
      linkDirectionalArrowLength={LINK_SIZES.arrowLength}
      linkDirectionalArrowRelPos={1}
      backgroundColor={isDark ? BACKGROUND_COLORS.dark.graph : BACKGROUND_COLORS.light.graph}
      width={width}
      height={height}
      d3AlphaDecay={FORCE_CONFIG.alphaDecay}
      d3VelocityDecay={FORCE_CONFIG.velocityDecay}
      cooldownTime={FORCE_CONFIG.cooldownTime}
      cooldownTicks={FORCE_CONFIG.cooldownTicks}
      onNodeClick={(node) => onNodeClick(node as GraphNode)}
      nodeCanvasObject={(node, ctx, globalScale) => {
        const graphNode = node as GraphNode & { x: number; y: number }
        const nodeSize = BASE_SIZES.node2D * getNodeSize(graphNode)
        const color = getNodeColor(graphNode)
        const isSelected = selectedNodeId === graphNode.id

        // Draw selection marker (outer ring) for selected node
        if (isSelected) {
          ctx.beginPath()
          ctx.arc(graphNode.x, graphNode.y, nodeSize + 6, 0, 2 * Math.PI)
          ctx.strokeStyle = SELECTION_COLORS.ring
          ctx.lineWidth = 3
          ctx.stroke()
        }

        // Check if this is a high/critical severity vulnerability or CVE
        const glowLevel = getGlowLevel(graphNode)

        // Draw pulsing glow effect for high/critical severity
        if (glowLevel) {
          const time = animationTimeRef.current || Date.now() / 1000
          const speed = glowLevel === 'critical' ? ANIMATION_CONFIG.criticalSpeed : ANIMATION_CONFIG.highSpeed
          const pulse = Math.sin(time * speed) * 0.5 + 0.5
          const glowRadius = nodeSize + ANIMATION_CONFIG.glow2DRadiusExtra.base + pulse * ANIMATION_CONFIG.glow2DRadiusExtra.pulse

          const gradient = ctx.createRadialGradient(
            graphNode.x,
            graphNode.y,
            nodeSize,
            graphNode.x,
            graphNode.y,
            glowRadius
          )
          gradient.addColorStop(0, color)
          gradient.addColorStop(0.5, `${color}88`)
          gradient.addColorStop(1, `${color}00`)

          ctx.beginPath()
          ctx.arc(graphNode.x, graphNode.y, glowRadius, 0, 2 * Math.PI)
          ctx.fillStyle = gradient
          ctx.fill()
        }

        // Draw main circle
        ctx.beginPath()
        ctx.arc(graphNode.x, graphNode.y, nodeSize, 0, 2 * Math.PI)
        ctx.fillStyle = color
        ctx.fill()

        // Draw label if enabled or if node is selected
        if ((showLabels && globalScale > ZOOM_CONFIG.labelVisibilityThreshold) || isSelected) {
          const label = graphNode.name
          const fontSize = Math.max(6 / globalScale, BASE_SIZES.label2D.min)
          ctx.font = `${fontSize}px Sans-Serif`
          ctx.textAlign = 'center'
          ctx.textBaseline = 'top'
          ctx.fillStyle = '#ffffff'
          ctx.fillText(label, graphNode.x, graphNode.y + nodeSize + 2)
        }
      }}
      nodePointerAreaPaint={(node, color, ctx) => {
        const graphNode = node as GraphNode & { x: number; y: number }
        ctx.beginPath()
        ctx.arc(graphNode.x, graphNode.y, 10, 0, 2 * Math.PI)
        ctx.fillStyle = color
        ctx.fill()
      }}
    />
  )
}

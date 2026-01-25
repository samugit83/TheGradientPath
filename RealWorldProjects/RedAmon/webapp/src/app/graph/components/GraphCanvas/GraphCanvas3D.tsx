'use client'

import { useRef, useEffect } from 'react'
import dynamic from 'next/dynamic'
import { GraphData, GraphNode, GraphLink } from '../../types'
import { getNodeColor, getNodeSize, getGlowLevel } from '../../utils'
import { getLinkColor, getLinkWidth3D, getParticleCount } from '../../utils/linkHelpers'
import {
  LINK_COLORS,
  LINK_SIZES,
  BASE_SIZES,
  BACKGROUND_COLORS,
  SELECTION_COLORS,
  ANIMATION_CONFIG,
  THREE_CONFIG,
} from '../../config'
import { hasHighSeverityNodes } from '../../utils/nodeHelpers'
import { useAnimationFrame } from '../../hooks'

const ForceGraph3D = dynamic(() => import('react-force-graph-3d'), {
  ssr: false,
})

interface GraphCanvas3DProps {
  data: GraphData
  width: number
  height: number
  showLabels: boolean
  selectedNode: GraphNode | null
  onNodeClick: (node: GraphNode) => void
  isDark?: boolean
}

export function GraphCanvas3D({
  data,
  width,
  height,
  showLabels,
  selectedNode,
  onNodeClick,
  isDark = true,
}: GraphCanvas3DProps) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const graph3DRef = useRef<any>(null)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const glowRingsRef = useRef<any[]>([])

  // Clear glow rings when data changes
  useEffect(() => {
    glowRingsRef.current = []
  }, [data])

  // Animation loop for pulsing glow rings
  const hasHighSeverity = hasHighSeverityNodes(data.nodes)

  useAnimationFrame(
    (time) => {
      glowRingsRef.current.forEach((ring) => {
        if (ring) {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const glowLevel = (ring as any).__glowLevel || 'high'
          const speed = glowLevel === 'critical' ? ANIMATION_CONFIG.criticalSpeed : ANIMATION_CONFIG.highSpeed
          const pulse = Math.sin(time * speed) * 0.15 + 1
          const opacity = Math.sin(time * speed) * 0.2 + 0.4

          ring.scale.set(pulse, pulse, 1)
          if (ring.material) {
            ring.material.opacity = opacity
          }
        }
      })
    },
    hasHighSeverity
  )

  const selectedNodeId = selectedNode?.id

  return (
    <ForceGraph3D
      ref={graph3DRef}
      graphData={data}
      nodeLabel={(node) => `${(node as GraphNode).name} (${(node as GraphNode).type})`}
      nodeColor={(node) => getNodeColor(node as GraphNode)}
      nodeRelSize={BASE_SIZES.node3D}
      nodeOpacity={THREE_CONFIG.nodeOpacity}
      linkLabel={(link) => (link as GraphLink).type}
      linkColor={(link) => getLinkColor(link as GraphLink, selectedNodeId)}
      linkWidth={(link) => getLinkWidth3D(link as GraphLink, selectedNodeId)}
      linkDirectionalParticles={(link) => getParticleCount(link as GraphLink, selectedNodeId)}
      linkDirectionalParticleWidth={LINK_SIZES.particleWidth}
      linkDirectionalParticleColor={() => LINK_COLORS.particle}
      linkDirectionalArrowLength={LINK_SIZES.arrowLength3D}
      linkDirectionalArrowRelPos={1}
      backgroundColor={isDark ? BACKGROUND_COLORS.dark.graph : BACKGROUND_COLORS.light.graph}
      width={width}
      height={height}
      onNodeClick={(node) => onNodeClick(node as GraphNode)}
      nodeThreeObject={(node: object) => {
        const graphNode = node as GraphNode
        const THREE = require('three')
        const SpriteText = require('three-spritetext').default

        const group = new THREE.Group()

        const sphereSize = BASE_SIZES.node3D * getNodeSize(graphNode)
        const nodeColor = getNodeColor(graphNode)
        const isSelected = selectedNodeId === graphNode.id

        // Add selection marker ring (green) for selected node
        if (isSelected) {
          const selectGeometry = new THREE.RingGeometry(
            sphereSize * THREE_CONFIG.selectionRingScale.inner,
            sphereSize * THREE_CONFIG.selectionRingScale.outer,
            THREE_CONFIG.ringSegments
          )
          const selectMaterial = new THREE.MeshBasicMaterial({
            color: SELECTION_COLORS.ring,
            transparent: true,
            opacity: THREE_CONFIG.selectionRingOpacity,
            side: THREE.DoubleSide,
          })
          const selectRing = new THREE.Mesh(selectGeometry, selectMaterial)
          selectRing.lookAt(0, 0, 1)
          group.add(selectRing)
        }

        // Add outer glow ring for high/critical severity
        const glowLevel = getGlowLevel(graphNode)
        if (glowLevel) {
          const glowGeometry = new THREE.RingGeometry(
            sphereSize * THREE_CONFIG.glowRingScale.inner,
            sphereSize * THREE_CONFIG.glowRingScale.outer,
            THREE_CONFIG.ringSegments
          )
          const glowMaterial = new THREE.MeshBasicMaterial({
            color: nodeColor,
            transparent: true,
            opacity: THREE_CONFIG.glowRingOpacity,
            side: THREE.DoubleSide,
          })
          const glowRing = new THREE.Mesh(glowGeometry, glowMaterial)

          glowRing.lookAt(0, 0, 1)

          // Store reference for animation with glow level
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          ;(glowRing as any).__glowLevel = glowLevel
          glowRingsRef.current.push(glowRing)

          group.add(glowRing)
        }

        const geometry = new THREE.SphereGeometry(
          sphereSize,
          THREE_CONFIG.sphereSegments,
          THREE_CONFIG.sphereSegments
        )
        const material = new THREE.MeshLambertMaterial({
          color: nodeColor,
          transparent: true,
          opacity: THREE_CONFIG.nodeOpacity,
        })
        const sphere = new THREE.Mesh(geometry, material)
        group.add(sphere)

        // Create label if enabled or if node is selected
        if (showLabels || isSelected) {
          const sprite = new SpriteText(graphNode.name)
          sprite.color = '#ffffff'
          sprite.textHeight = BASE_SIZES.label3D
          sprite.position.y = sphereSize + BASE_SIZES.label3D
          group.add(sprite)
        }

        return group
      }}
    />
  )
}

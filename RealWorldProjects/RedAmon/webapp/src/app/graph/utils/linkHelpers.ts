import { GraphLink, GraphNode } from '../types'
import { LINK_COLORS, LINK_SIZES } from '../config'

/**
 * Extract node ID from a link endpoint (handles both string and GraphNode types)
 */
export const getNodeId = (node: string | GraphNode): string =>
  typeof node === 'string' ? node : node.id

/**
 * Check if a link is connected to a specific node
 */
export const isLinkConnectedToNode = (link: GraphLink, nodeId: string): boolean => {
  const sourceId = getNodeId(link.source)
  const targetId = getNodeId(link.target)
  return sourceId === nodeId || targetId === nodeId
}

/**
 * Get link color based on selection state
 */
export const getLinkColor = (link: GraphLink, selectedNodeId?: string): string => {
  if (!selectedNodeId) return LINK_COLORS.default
  return isLinkConnectedToNode(link, selectedNodeId)
    ? LINK_COLORS.highlighted
    : LINK_COLORS.default
}

/**
 * Get link width based on selection state (2D)
 */
export const getLinkWidth2D = (link: GraphLink, selectedNodeId?: string): number => {
  if (!selectedNodeId) return LINK_SIZES.defaultWidth2D
  return isLinkConnectedToNode(link, selectedNodeId)
    ? LINK_SIZES.highlightedWidth2D
    : LINK_SIZES.defaultWidth2D
}

/**
 * Get link width based on selection state (3D)
 */
export const getLinkWidth3D = (link: GraphLink, selectedNodeId?: string): number => {
  if (!selectedNodeId) return LINK_SIZES.defaultWidth3D
  return isLinkConnectedToNode(link, selectedNodeId)
    ? LINK_SIZES.highlightedWidth3D
    : LINK_SIZES.defaultWidth3D
}

/**
 * Get particle width based on selection state
 */
export const getParticleWidth = (link: GraphLink, selectedNodeId?: string): number => {
  if (!selectedNodeId) return 0
  return isLinkConnectedToNode(link, selectedNodeId) ? LINK_SIZES.particleWidth : 0
}

/**
 * Get particle count based on selection state (3D)
 */
export const getParticleCount = (link: GraphLink, selectedNodeId?: string): number => {
  if (!selectedNodeId) return 0
  return isLinkConnectedToNode(link, selectedNodeId) ? LINK_SIZES.particleCount : 0
}

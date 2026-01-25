import { GraphNode, GlowLevel } from '../types'
import {
  NODE_COLORS,
  SEVERITY_COLORS_VULN,
  SEVERITY_COLORS_CVE,
  NODE_SIZES,
  SEVERITY_SIZE_MULTIPLIERS,
} from '../config'

/**
 * Get the severity level from a node's properties
 */
export const getNodeSeverity = (node: GraphNode): string => {
  return (node.properties?.severity as string)?.toLowerCase() || 'unknown'
}

/**
 * Get node color based on type and severity
 */
export const getNodeColor = (node: GraphNode): string => {
  if (node.type === 'Vulnerability') {
    const severity = getNodeSeverity(node)
    return SEVERITY_COLORS_VULN[severity] || SEVERITY_COLORS_VULN.unknown
  }
  if (node.type === 'CVE') {
    const severity = getNodeSeverity(node)
    return SEVERITY_COLORS_CVE[severity] || SEVERITY_COLORS_CVE.unknown
  }
  return NODE_COLORS[node.type] || NODE_COLORS.Default
}

/**
 * Get node size multiplier based on type and severity
 */
export const getNodeSize = (node: GraphNode): number => {
  const baseSize = NODE_SIZES[node.type] || NODE_SIZES.Default

  if (node.type === 'Vulnerability' || node.type === 'CVE') {
    const severity = getNodeSeverity(node)
    const severityMultiplier = SEVERITY_SIZE_MULTIPLIERS[severity] || 1.0
    return baseSize * severityMultiplier
  }

  return baseSize
}

/**
 * Check if a node should have glow effect (critical/high severity)
 * Returns: false, 'high', or 'critical'
 */
export const getGlowLevel = (node: GraphNode): GlowLevel => {
  if (node.type === 'Vulnerability' || node.type === 'CVE') {
    const severity = getNodeSeverity(node)
    if (severity === 'critical') return 'critical'
    if (severity === 'high') return 'high'
  }
  return false
}

/**
 * Check if a node is a high-severity vulnerability or CVE
 */
export const isHighSeverityNode = (node: GraphNode): boolean => {
  if (node.type !== 'Vulnerability' && node.type !== 'CVE') return false
  const severity = getNodeSeverity(node)
  return severity === 'critical' || severity === 'high'
}

/**
 * Check if any nodes in the data have high severity
 */
export const hasHighSeverityNodes = (nodes: GraphNode[]): boolean => {
  return nodes.some(isHighSeverityNode)
}

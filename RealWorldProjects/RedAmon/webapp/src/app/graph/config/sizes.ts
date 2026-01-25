// Node size multipliers by type (1x = default size)
export const NODE_SIZES: Record<string, number> = {
  Domain: 4,
  Subdomain: 3,
  IP: 2,
  Port: 2,
  Service: 2,
  BaseURL: 3,
  Technology: 2,
  Default: 1,
}

// Severity-based size multipliers
export const SEVERITY_SIZE_MULTIPLIERS: Record<string, number> = {
  critical: 1.0,
  high: 1.0,
  medium: 1.0,
  low: 0.7,
  info: 0.7,
  unknown: 0.7,
}

// Base sizes for rendering
export const BASE_SIZES = {
  node2D: 6,
  node3D: 5,
  label2D: { min: 2, divisor: 1 },
  label3D: 3,
} as const

// Link dimensions
export const LINK_SIZES = {
  defaultWidth2D: 1,
  highlightedWidth2D: 3,
  defaultWidth3D: 1.5,
  highlightedWidth3D: 3.5,
  arrowLength: 4,
  arrowLength3D: 3,
  particleWidth: 4,
  particleCount: 4,
} as const

// Drawer dimensions
export const DRAWER_WIDTH = 400

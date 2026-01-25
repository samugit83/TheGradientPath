// Force simulation settings
export const FORCE_CONFIG = {
  alphaDecay: 0.02,
  velocityDecay: 0.4,
  collisionRadius: 20,
  collisionStrength: 1,
  collisionIterations: 3,
  cooldownTime: Infinity,
  cooldownTicks: Infinity,
} as const

// Animation settings
export const ANIMATION_CONFIG = {
  criticalSpeed: 10,
  highSpeed: 3,
  glowPulseRange: { min: 0.85, max: 1.15 },
  glowOpacityRange: { min: 0.2, max: 0.6 },
  glow2DPulseRange: { min: 0, max: 1 },
  glow2DRadiusExtra: { base: 2, pulse: 3 },
  initDelay: 300,
} as const

// Zoom settings
export const ZOOM_CONFIG = {
  labelVisibilityThreshold: 0.4,
} as const

// 3D specific settings
export const THREE_CONFIG = {
  sphereSegments: 16,
  ringSegments: 32,
  nodeOpacity: 0.9,
  glowRingOpacity: 0.4,
  selectionRingOpacity: 0.9,
  selectionRingScale: { inner: 1.4, outer: 1.6 },
  glowRingScale: { inner: 1.15, outer: 1.4 },
} as const

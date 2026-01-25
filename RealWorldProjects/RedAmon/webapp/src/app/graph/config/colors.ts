// Node colors by type - semantic color mapping
export const NODE_COLORS: Record<string, string> = {
  // CRITICAL SECURITY (Red family) - Immediate attention needed
  Vulnerability: '#ef4444',  // Bright red - DANGER, highest priority
  CVE: '#dc2626',            // Deep red - Known vulnerabilities

  // THREAT INTELLIGENCE (Orange family) - Attack context
  MitreData: '#f97316',      // Orange - CWE/MITRE techniques
  Capec: '#eab308',          // Yellow - Attack patterns

  // DOMAIN HIERARCHY (Blue family) - Recon foundation
  Domain: '#1e3a8a',         // Deep navy - Root/foundation (most important)
  Subdomain: '#2563eb',      // Royal blue - Children of domain

  // NETWORK LAYER (Cyan/Teal family) - Infrastructure
  IP: '#0d9488',             // Teal - Network addresses
  Port: '#0e7490',           // Dark cyan - Network ports
  Service: '#06b6d4',        // Cyan - Running services

  // WEB APPLICATION LAYER (Purple family) - Web-specific assets
  BaseURL: '#6366f1',        // Indigo - Web entry points
  Endpoint: '#8b5cf6',       // Purple - Paths/routes
  Parameter: '#a855f7',      // Light purple - Inputs (attack surface)

  // CONTEXT & METADATA (Neutral family) - Supporting information
  Technology: '#22c55e',     // Green - Tech stack (good to know)
  Certificate: '#d97706',    // Amber - TLS/security context
  Header: '#78716c',         // Stone gray - HTTP metadata

  Default: '#6b7280',        // Gray - Fallback
}

// Severity-based colors for Vulnerability nodes (pure red tonality)
export const SEVERITY_COLORS_VULN: Record<string, string> = {
  critical: '#ff3333',  // Brilliant red - brightest, vivid
  high: '#b91c1c',      // Medium red (red-700)
  medium: '#b91c1c',    // Medium red (red-700)
  low: '#7f1d1d',       // Dark red (red-900)
  info: '#7f1d1d',      // Dark red (red-900)
  unknown: '#6b7280',   // Grey for unknown
}

// Severity-based colors for CVE nodes (red-purple/magenta tonality)
export const SEVERITY_COLORS_CVE: Record<string, string> = {
  critical: '#ff3377',  // Brilliant magenta-red - brightest
  high: '#be185d',      // Medium pink-red (pink-700)
  medium: '#be185d',    // Medium pink-red (pink-700)
  low: '#831843',       // Dark pink-red (pink-900)
  info: '#831843',      // Dark pink-red (pink-900)
  unknown: '#6b7280',   // Grey for unknown
}

// Link colors
export const LINK_COLORS = {
  default: '#9ca3af',
  highlighted: '#60a5fa',
  particle: '#60a5fa',
} as const

// Selection colors
export const SELECTION_COLORS = {
  ring: '#22c55e',
} as const

// Background colors by theme
export const BACKGROUND_COLORS = {
  dark: {
    graph: '#0a0a0a',
  },
  light: {
    graph: '#f3f4f6', // gray-100
  },
} as const

import { useState, useEffect, RefObject } from 'react'

interface Dimensions {
  width: number
  height: number
}

/**
 * Custom hook for tracking graph canvas dimensions
 * Uses ResizeObserver to measure the actual container element
 * @param containerRef - Reference to the container element to measure
 */
export function useDimensions(containerRef: RefObject<HTMLElement | null>): Dimensions {
  const [dimensions, setDimensions] = useState<Dimensions>({
    width: 800,
    height: 600,
  })

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const updateDimensions = () => {
      setDimensions({
        width: container.clientWidth,
        height: container.clientHeight,
      })
    }

    // Initial measurement
    updateDimensions()

    // Use ResizeObserver for accurate container size tracking
    const resizeObserver = new ResizeObserver(updateDimensions)
    resizeObserver.observe(container)

    return () => resizeObserver.disconnect()
  }, [containerRef])

  return dimensions
}

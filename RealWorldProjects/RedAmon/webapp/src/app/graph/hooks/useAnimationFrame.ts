import { useEffect, useRef } from 'react'

/**
 * Custom hook for running animation frames
 * @param callback - Function to call on each animation frame, receives current time
 * @param enabled - Whether the animation should run
 */
export function useAnimationFrame(
  callback: (time: number) => void,
  enabled: boolean
) {
  const callbackRef = useRef(callback)

  // Update callback ref on each render to avoid stale closures
  useEffect(() => {
    callbackRef.current = callback
  }, [callback])

  useEffect(() => {
    if (!enabled) return

    let animationId: number

    const animate = () => {
      const time = Date.now() / 1000
      callbackRef.current(time)
      animationId = requestAnimationFrame(animate)
    }

    animationId = requestAnimationFrame(animate)

    return () => cancelAnimationFrame(animationId)
  }, [enabled])
}

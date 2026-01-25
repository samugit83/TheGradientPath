/**
 * Format Neo4j datetime objects to readable string
 */
export function formatNeo4jDateTime(value: unknown): string | null {
  if (
    typeof value === 'object' &&
    value !== null &&
    'year' in value &&
    'month' in value &&
    'day' in value &&
    'hour' in value &&
    'minute' in value &&
    'second' in value
  ) {
    const dt = value as {
      year: { low: number }
      month: { low: number }
      day: { low: number }
      hour: { low: number }
      minute: { low: number }
      second: { low: number }
    }
    const year = dt.year.low
    const month = String(dt.month.low).padStart(2, '0')
    const day = String(dt.day.low).padStart(2, '0')
    const hour = String(dt.hour.low).padStart(2, '0')
    const minute = String(dt.minute.low).padStart(2, '0')
    const second = String(dt.second.low).padStart(2, '0')
    return `${year}-${month}-${day} ${hour}:${minute}:${second}`
  }
  return null
}

/**
 * Format a property value for display in the drawer
 */
export function formatPropertyValue(value: unknown): React.ReactNode {
  const formattedDate = formatNeo4jDateTime(value)

  if (formattedDate) {
    return formattedDate
  }

  if (
    value === null ||
    value === undefined ||
    value === 'none' ||
    value === 'None' ||
    value === 'null' ||
    value === 'NULL'
  ) {
    return '---'
  }

  if (Array.isArray(value)) {
    if (value.length === 0) {
      return '---'
    }
    return value.join(', ')
  }

  if (typeof value === 'object') {
    return JSON.stringify(value, null, 2)
  }

  if (value === '') {
    return '---'
  }

  return String(value)
}

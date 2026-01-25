import neo4j, { Driver, Session } from 'neo4j-driver'

declare global {
  var neo4jDriver: Driver | undefined
}

const uri = process.env.NEO4J_URI || 'bolt://localhost:7687'
const user = process.env.NEO4J_USER || 'neo4j'
const password = process.env.NEO4J_PASSWORD || 'password'

function createDriver(): Driver {
  return neo4j.driver(uri, neo4j.auth.basic(user, password), {
    maxConnectionPoolSize: 50,
    connectionAcquisitionTimeout: 30000,
    connectionTimeout: 30000,
  })
}

export function getDriver(): Driver {
  if (process.env.NODE_ENV === 'production') {
    return createDriver()
  }

  if (!global.neo4jDriver) {
    global.neo4jDriver = createDriver()
  }
  return global.neo4jDriver
}

export function getSession(): Session {
  return getDriver().session()
}

export async function closeDriver(): Promise<void> {
  if (global.neo4jDriver) {
    await global.neo4jDriver.close()
    global.neo4jDriver = undefined
  }
}

export async function verifyConnection(): Promise<boolean> {
  const driver = getDriver()
  try {
    await driver.verifyConnectivity()
    return true
  } catch (error) {
    console.error('Neo4j connection failed:', error)
    return false
  }
}

export { neo4j }

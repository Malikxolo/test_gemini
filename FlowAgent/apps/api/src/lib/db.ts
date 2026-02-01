import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import * as schema from '@flowagent/database/src/schema';

export function createDbClient(connectionString: string) {
  const client = postgres(connectionString, {
    prepare: false,
    max: 10,
    idle_timeout: 20,
    connect_timeout: 10,
  });

  const db = drizzle(client, { schema });

  return { client, db };
}

export type DbClient = ReturnType<typeof createDbClient>;

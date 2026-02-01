import { Context, Next } from 'hono';
import { getCookie } from 'hono/cookie';
import { createAuth } from '../lib/auth';
import { createDbClient } from '../lib/db';

export async function auth(c: Context, next: Next) {
  const { client, db } = createDbClient(c.env.DATABASE_URL);
  const lucia = createAuth(client);

  const sessionId = getCookie(c, 'session');

  if (!sessionId) {
    return c.json({ error: 'Unauthorized' }, 401);
  }

  const { session, user } = await lucia.validateSession(sessionId);

  if (!session) {
    return c.json({ error: 'Session expired' }, 401);
  }

  // Attach to context
  c.set('user', user);
  c.set('session', session);
  c.set('db', db);

  await next();
}

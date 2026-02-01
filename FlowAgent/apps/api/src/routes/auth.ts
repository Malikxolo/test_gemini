import { Hono } from 'hono';
import { zValidator } from '@hono/zod-validator';
import { z } from 'zod';
import { setCookie } from 'hono/cookie';
import { createDbClient } from '../lib/db';
import { createAuth, hashPassword, verifyPassword } from '../lib/auth';
import { users } from '@flowagent/database/src/schema';
import { eq } from 'drizzle-orm';

const app = new Hono() as AppType;
import type { AppType } from "../types/hono";

const registerSchema = z.object({
  email: z.string().email(),
  username: z.string().min(3).max(30),
  password: z.string().min(8),
  displayName: z.string().optional(),
});

const loginSchema = z.object({
  email: z.string().email(),
  password: z.string(),
});

// Register
app.post('/register', zValidator('json', registerSchema), async (c) => {
  const data = c.req.valid('json');
  const { client, db } = createDbClient(c.env.DATABASE_URL);
  const lucia = createAuth(client);

  // Check if user exists
  const existing = await db.query.users.findFirst({
    where: eq(users.email, data.email),
  });

  if (existing) {
    return c.json({ error: 'Email already registered' }, 400);
  }

  // Hash password
  const passwordHash = await hashPassword(data.password);

  // Create user
  const [user] = await db.insert(users).values({
    email: data.email,
    username: data.username,
    displayName: data.displayName,
    passwordHash,
  }).returning();

  // Create session
  const session = await lucia.createSession(user.id, {});
  const sessionCookie = lucia.createSessionCookie(session.id);

  setCookie(c, sessionCookie.name, sessionCookie.value, sessionCookie.attributes);

  return c.json({
    user: {
      id: user.id,
      email: user.email,
      username: user.username,
      displayName: user.displayName,
    },
  }, 201);
});

// Login
app.post('/login', zValidator('json', loginSchema), async (c) => {
  const data = c.req.valid('json');
  const { client, db } = createDbClient(c.env.DATABASE_URL);
  const lucia = createAuth(client);

  // Find user
  const user = await db.query.users.findFirst({
    where: eq(users.email, data.email),
  });

  if (!user || !user.passwordHash) {
    return c.json({ error: 'Invalid credentials' }, 401);
  }

  // Verify password
  const valid = await verifyPassword(user.passwordHash, data.password);

  if (!valid) {
    return c.json({ error: 'Invalid credentials' }, 401);
  }

  // Update last login
  await db.update(users).set({
    lastLoginAt: new Date(),
  }).where(eq(users.id, user.id));

  // Create session
  const session = await lucia.createSession(user.id, {});
  const sessionCookie = lucia.createSessionCookie(session.id);

  setCookie(c, sessionCookie.name, sessionCookie.value, sessionCookie.attributes);

  return c.json({
    user: {
      id: user.id,
      email: user.email,
      username: user.username,
      displayName: user.displayName,
      subscriptionTier: user.subscriptionTier,
    },
  });
});

// Logout
app.post('/logout', async (c) => {
  const { client, db } = createDbClient(c.env.DATABASE_URL);
  const lucia = createAuth(client);
  const sessionId = c.get('session')?.id;

  if (sessionId) {
    await lucia.invalidateSession(sessionId);
  }

  const sessionCookie = lucia.createBlankSessionCookie();
  setCookie(c, sessionCookie.name, sessionCookie.value, sessionCookie.attributes);

  return c.json({ success: true });
});

// Get current user
app.get('/me', async (c) => {
  const user = c.get('user');

  if (!user) {
    return c.json({ error: 'Not authenticated' }, 401);
  }

  return c.json({ user });
});

export { app as authRoutes };

import { Hono } from 'hono';
import { users, usageStats } from '@flowagent/database/src/schema';
import { eq, and } from 'drizzle-orm';
import type { AppType } from '../types/hono';

const app = new Hono() as AppType;

// Get current user profile
app.get('/me', async (c) => {
  const user = c.get('user');
  const db = c.get('db');

  const profile = await db.query.users.findFirst({
    where: eq(users.id, user.id),
    columns: {
      id: true,
      email: true,
      username: true,
      displayName: true,
      avatarUrl: true,
      subscriptionTier: true,
      subscriptionStatus: true,
      createdAt: true,
      lastLoginAt: true,
    },
  });

  return c.json(profile);
});

// Get usage stats
app.get('/me/usage', async (c) => {
  const user = c.get('user');
  const db = c.get('db');
  const year = parseInt(c.req.query('year') || new Date().getFullYear().toString());
  const month = parseInt(c.req.query('month') || (new Date().getMonth() + 1).toString());

  const stats = await db.query.usageStats.findFirst({
    where: and(
      eq(usageStats.userId, user.id),
      eq(usageStats.year, year),
      eq(usageStats.month, month)
    ),
  });

  return c.json(stats || {
    totalExecutions: 0,
    successfulRuns: 0,
    failedRuns: 0,
    inputTokens: 0,
    outputTokens: 0,
    estimatedCost: 0,
    apiCalls: 0,
  });
});

// Update profile
app.patch('/me', async (c) => {
  const user = c.get('user');
  const db = c.get('db');
  const { displayName, avatarUrl } = await c.req.json();

  const [updated] = await db.update(users)
    .set({
      displayName,
      avatarUrl,
      updatedAt: new Date(),
    })
    .where(eq(users.id, user.id))
    .returning({
      id: users.id,
      email: users.email,
      username: users.username,
      displayName: users.displayName,
      avatarUrl: users.avatarUrl,
    });

  return c.json(updated);
});

export { app as userRoutes };

import { Hono } from 'hono';
import { executions, executionSteps } from '@flowagent/database/src/schema';
import { eq, desc, and } from 'drizzle-orm';
import type { AppType } from '../types/hono';

const app = new Hono() as AppType;

// List executions
app.get('/', async (c) => {
  const user = c.get('user');
  const db = c.get('db');
  const agentId = c.req.query('agentId');
  const limit = Math.min(parseInt(c.req.query('limit') || '20'), 100);
  const offset = parseInt(c.req.query('offset') || '0');

  const where = agentId
    ? and(eq(executions.userId, user.id), eq(executions.agentId, agentId))
    : eq(executions.userId, user.id);

  const userExecutions = await db.query.executions.findMany({
    where,
    orderBy: [desc(executions.createdAt)],
    limit,
    offset,
    with: {
      agent: {
        columns: {
          id: true,
          name: true,
          icon: true,
        },
      },
    },
  });

  return c.json({
    executions: userExecutions,
  });
});

// Get single execution
app.get('/:id', async (c) => {
  const user = c.get('user');
  const db = c.get('db');
  const id = c.req.param('id');

  const execution = await db.query.executions.findFirst({
    where: and(
      eq(executions.id, id),
      eq(executions.userId, user.id)
    ),
    with: {
      agent: true,
      steps: {
        orderBy: [desc(executionSteps.stepNumber)],
      },
    },
  });

  if (!execution) {
    return c.json({ error: 'Execution not found' }, 404);
  }

  return c.json(execution);
});

// Cancel execution
app.post('/:id/cancel', async (c) => {
  const user = c.get('user');
  const db = c.get('db');
  const id = c.req.param('id');

  const execution = await db.query.executions.findFirst({
    where: and(
      eq(executions.id, id),
      eq(executions.userId, user.id)
    ),
  });

  if (!execution) {
    return c.json({ error: 'Execution not found' }, 404);
  }

  if (execution.status === 'completed' || execution.status === 'failed' || execution.status === 'cancelled') {
    return c.json({ error: 'Execution already finished' }, 400);
  }

  // Update status
  const [updated] = await db.update(executions)
    .set({
      status: 'cancelled',
      completedAt: new Date(),
    })
    .where(eq(executions.id, id))
    .returning();

  return c.json(updated);
});

export { app as executionRoutes };

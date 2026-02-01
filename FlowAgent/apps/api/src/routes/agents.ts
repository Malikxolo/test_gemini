import { Hono } from 'hono';
import { zValidator } from '@hono/zod-validator';
import { z } from 'zod';
import { agents, executions } from '@flowagent/database/src/schema';
import { eq, desc, and, sql } from 'drizzle-orm';
import { createQueue, queueAgentExecution } from '../lib/queue';
import { getUserApiKeys } from './byok';
import type { AppType } from '../types/hono';

const app = new Hono() as AppType;

const createAgentSchema = z.object({
  name: z.string().min(1).max(100),
  description: z.string().max(500).optional(),
  icon: z.string().optional(),
  mode: z.enum(['auto', 'air', 'custom', 'pro']).default('auto'),
  model: z.string().default('gpt-4o-mini'),
  temperature: z.number().min(0).max(2).default(0.7),
  maxTokens: z.number().int().min(1).max(8192).default(4096),
  systemPrompt: z.string().optional(),
  tools: z.array(z.string()).default([]),
  workflow: z.any().optional(),
  memoryEnabled: z.boolean().default(true),
  memoryWindow: z.number().int().default(10),
});

const updateAgentSchema = createAgentSchema.partial();

const executeSchema = z.object({
  input: z.object({
    message: z.string().min(1).max(10000),
  }),
  stream: z.boolean().optional().default(false),
});

// List agents
app.get('/', async (c) => {
  const user = c.get('user');
  const db = c.get('db');
  const limit = Math.min(parseInt(c.req.query('limit') || '20'), 100);
  const offset = parseInt(c.req.query('offset') || '0');

  const userAgents = await db.query.agents.findMany({
    where: eq(agents.userId, user.id),
    orderBy: [desc(agents.updatedAt)],
    limit,
    offset,
  });

  return c.json({
    agents: userAgents,
    total: userAgents.length,
  });
});

// Get single agent
app.get('/:id', async (c) => {
  const user = c.get('user');
  const db = c.get('db');
  const id = c.req.param('id');

  const agent = await db.query.agents.findFirst({
    where: and(
      eq(agents.id, id),
      eq(agents.userId, user.id)
    ),
  });

  if (!agent) {
    return c.json({ error: 'Agent not found' }, 404);
  }

  return c.json(agent);
});

// Create agent
app.post('/', zValidator('json', createAgentSchema), async (c) => {
  const user = c.get('user');
  const db = c.get('db');
  const data = c.req.valid('json');

  // Check agent limit based on subscription tier
  const [{ count }] = await db
    .select({ count: sql<number>`count(*)` })
    .from(agents)
    .where(eq(agents.userId, user.id));

  const maxAgents = user.subscriptionTier === 'free' ? 10 :
                    user.subscriptionTier === 'pro' ? 100 : 1000;

  if (count >= maxAgents) {
    return c.json({
      error: 'Agent limit reached',
      limit: maxAgents,
      current: count,
    }, 403);
  }

  // Create agent
  const [agent] = await db.insert(agents).values({
    userId: user.id,
    name: data.name,
    description: data.description,
    icon: data.icon,
    mode: data.mode,
    model: data.model,
    temperature: data.temperature.toString(),
    maxTokens: data.maxTokens,
    systemPrompt: data.systemPrompt,
    tools: data.tools,
    workflow: data.workflow,
    memoryEnabled: data.memoryEnabled,
    memoryWindow: data.memoryWindow,
  }).returning();

  return c.json(agent, 201);
});

// Update agent
app.patch('/:id', zValidator('json', updateAgentSchema), async (c) => {
  const user = c.get('user');
  const db = c.get('db');
  const id = c.req.param('id');
  const data = c.req.valid('json');

  // Check ownership
  const existing = await db.query.agents.findFirst({
    where: and(
      eq(agents.id, id),
      eq(agents.userId, user.id)
    ),
  });

  if (!existing) {
    return c.json({ error: 'Agent not found' }, 404);
  }

  // Update agent
  const [agent] = await db.update(agents)
    .set({
      name: data.name,
      description: data.description,
      icon: data.icon,
      mode: data.mode,
      model: data.model,
      temperature: data.temperature?.toString(),
      maxTokens: data.maxTokens,
      systemPrompt: data.systemPrompt,
      tools: data.tools,
      workflow: data.workflow,
      memoryEnabled: data.memoryEnabled,
      memoryWindow: data.memoryWindow,
      updatedAt: new Date(),
    })
    .where(eq(agents.id, id))
    .returning();

  return c.json(agent);
});

// Delete agent
app.delete('/:id', async (c) => {
  const user = c.get('user');
  const db = c.get('db');
  const id = c.req.param('id');

  // Check ownership
  const existing = await db.query.agents.findFirst({
    where: and(
      eq(agents.id, id),
      eq(agents.userId, user.id)
    ),
  });

  if (!existing) {
    return c.json({ error: 'Agent not found' }, 404);
  }

  // Delete agent
  await db.delete(agents).where(eq(agents.id, id));

  return c.json({ success: true });
});

// Execute agent
app.post('/:id/execute', zValidator('json', executeSchema), async (c) => {
  const user = c.get('user');
  const db = c.get('db');
  const id = c.req.param('id');
  const data = c.req.valid('json');
  const { input, stream } = data;

  // Get agent
  const agent = await db.query.agents.findFirst({
    where: and(
      eq(agents.id, id),
      eq(agents.userId, user.id)
    ),
  });

  if (!agent) {
    return c.json({ error: 'Agent not found' }, 404);
  }

  // Check execution limits
  const today = new Date();
  today.setHours(0, 0, 0, 0);

  const [{ count: executionCount }] = await db
    .select({ count: sql<number>`count(*)` })
    .from(executions)
    .where(
      and(
        eq(executions.userId, user.id),
        sql<Date>`${executions.createdAt} >= ${today}`
      )
    );

  const maxExecutions = user.subscriptionTier === 'free' ? 100 :
                        user.subscriptionTier === 'pro' ? 1000 : 10000;

  if (executionCount >= maxExecutions) {
    return c.json({
      error: 'Daily execution limit reached',
      limit: maxExecutions,
    }, 403);
  }

  // Create execution record
  const [execution] = await db.insert(executions).values({
    userId: user.id,
    agentId: id,
    mode: agent.mode || 'auto',
    input: input as any,
    status: 'pending',
  }).returning();

  // Get user's BYOK API keys (if any)
  const userApiKeys = await getUserApiKeys(db, user.id);

  // Queue execution to agent engine
  const qstash = createQueue(c.env.QSTASH_TOKEN);

  await queueAgentExecution(qstash, {
    type: 'agent.execute',
    executionId: execution.id,
    agentId: id,
    userId: user.id,
    agentConfig: agent,
    input: input as any,
    stream: stream || false,
    // @ts-ignore - userApiKeys is passed in the payload
    userApiKeys,
  }, c.env.LAMBDA_WEBHOOK_URL);

  return c.json({
    executionId: execution.id,
    status: 'queued',
  }, 202);
});

export { app as agentRoutes };
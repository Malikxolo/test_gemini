import { Hono } from 'hono';
import { Receiver } from '@upstash/qstash';

const app = new Hono() as AppType;
import type { AppType } from "../types/hono";

// QStash webhook handler
app.post('/qstash/execute', async (c) => {
  const body = await c.req.text();
  const signature = c.req.header('upstash-signature') || '';

  // Verify QStash signature
  const receiver = new Receiver({
    currentSigningKey: c.env.QSTASH_CURRENT_SIGNING_KEY,
    nextSigningKey: c.env.QSTASH_NEXT_SIGNING_KEY,
  });

  const isValid = await receiver.verify({
    signature,
    body,
  });

  if (!isValid) {
    return c.json({ error: 'Invalid signature' }, 401);
  }

  const job = JSON.parse(body);

  // Forward to Lambda
  const response = await fetch(c.env.LAMBDA_ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(job),
  });

  if (!response.ok) {
    const error = await response.text();
    console.error('Lambda execution failed:', error);
    return c.json({ error: 'Agent execution failed' }, 500);
  }

  return c.json({ success: true });
});

export { app as webhookRoutes };

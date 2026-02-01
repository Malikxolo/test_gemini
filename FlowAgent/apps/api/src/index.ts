import { Hono } from 'hono';
import { cors } from 'hono/cors';
import { logger } from 'hono/logger';
import { prettyJSON } from 'hono/pretty-json';

import { auth } from './middleware/auth';
import { rateLimit } from './middleware/rate-limit';
import { errorHandler } from './middleware/error';

import { agentRoutes } from './routes/agents';
import { executionRoutes } from './routes/executions';
import { authRoutes } from './routes/auth';
import { userRoutes } from './routes/users';
import { webhookRoutes } from './routes/webhooks';
import { byokRoutes } from './routes/byok';
import { paymentRoutes } from './routes/payments';

type Bindings = {
  DATABASE_URL: string;
  UPSTASH_REDIS_REST_URL: string;
  UPSTASH_REDIS_REST_TOKEN: string;
  QSTASH_TOKEN: string;
  QSTASH_CURRENT_SIGNING_KEY: string;
  QSTASH_NEXT_SIGNING_KEY: string;
  LAMBDA_ENDPOINT: string;
  LAMBDA_WEBHOOK_URL: string;
  OPENAI_API_KEY: string;
  ANTHROPIC_API_KEY: string;
  ENCRYPTION_KEY: string;
  FRONTEND_URL: string;
  RAZORPAY_KEY_ID: string;
  RAZORPAY_KEY_SECRET: string;
  RAZORPAY_WEBHOOK_SECRET: string;
};

const app = new Hono<{ Bindings: Bindings }>();

// Global middleware
app.use('*', logger());
app.use('*', cors({
  origin: ['https://flowagent.io', 'http://localhost:3000'],
  credentials: true,
}));
app.use('*', prettyJSON());

// Health check (no auth)
app.get('/health', (c) => c.json({
  status: 'ok',
  timestamp: Date.now(),
  version: '1.0.0'
}));

// Public routes
app.route('/api/auth', authRoutes);
app.route('/webhooks', webhookRoutes);

// Protected API routes
app.use('/api/*', auth);
app.use('/api/*', rateLimit);

app.route('/api/agents', agentRoutes);
app.route('/api/executions', executionRoutes);
app.route('/api/users', userRoutes);
app.route('/api/byok', byokRoutes);
app.route('/api/payments', paymentRoutes);

// Error handling
app.onError(errorHandler);

export default app;

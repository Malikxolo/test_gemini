import { Hono } from 'hono';
import { User as LuciaUser, Session } from 'lucia';
import { drizzle } from 'drizzle-orm/postgres-js';
import * as schema from '@flowagent/database/src/schema';

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

// Extended User type with subscription tier
export interface User extends LuciaUser {
  subscriptionTier: 'free' | 'pro' | 'enterprise';
}

// Type for the database client
type DatabaseClient = ReturnType<typeof drizzle<typeof schema>>;

type Variables = {
  user: User;
  session: Session;
  db: DatabaseClient;
};

export type AppType = Hono<{
  Bindings: Bindings;
  Variables: Variables;
}>;
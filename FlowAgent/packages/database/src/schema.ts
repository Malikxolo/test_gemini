import { pgTable, text, uuid, timestamp, boolean, integer, decimal, jsonb, index, uniqueIndex, pgEnum } from 'drizzle-orm/pg-core';
import { relations } from 'drizzle-orm';

// ============================================
// ENUMS
// ============================================

export const subscriptionTierEnum = pgEnum('subscription_tier', ['free', 'pro', 'enterprise']);
export const subscriptionStatusEnum = pgEnum('subscription_status', ['active', 'cancelled', 'past_due']);
export const agentModeEnum = pgEnum('agent_mode', ['auto', 'air', 'custom', 'pro']);
export const agentStatusEnum = pgEnum('agent_status', ['draft', 'active', 'archived']);
export const executionStatusEnum = pgEnum('execution_status', ['pending', 'running', 'completed', 'failed', 'cancelled']);
export const templateStatusEnum = pgEnum('template_status', ['draft', 'pending_review', 'approved', 'rejected']);
export const purchaseStatusEnum = pgEnum('purchase_status', ['pending', 'completed', 'failed', 'refunded']);

// ============================================
// USERS & AUTHENTICATION
// ============================================

export const users = pgTable('users', {
  id: uuid('id').primaryKey().defaultRandom(),
  email: text('email').notNull().unique(),
  username: text('username').notNull().unique(),
  displayName: text('display_name'),
  avatarUrl: text('avatar_url'),

  // Auth (Lucia-style sessions)
  passwordHash: text('password_hash'),

  // Subscription
  subscriptionTier: subscriptionTierEnum('subscription_tier').default('free'),
  subscriptionStatus: subscriptionStatusEnum('subscription_status').default('active'),

  // Stripe (optional)
  stripeCustomerId: text('stripe_customer_id'),
  stripeSubscriptionId: text('stripe_subscription_id'),

  // BYOK (Bring Your Own Key) - User's own API keys
  openaiApiKey: text('openai_api_key'), // Encrypted
  anthropicApiKey: text('anthropic_api_key'), // Encrypted
  serperApiKey: text('serper_api_key'), // Encrypted

  // Access Level ($10 tier)
  hasArchitectureAccess: boolean('has_architecture_access').default(false),
  hasDownloadAccess: boolean('has_download_access').default(false),
  accessPurchasedAt: timestamp('access_purchased_at'),
  accessExpiresAt: timestamp('access_expires_at'), // null = lifetime

  // Timestamps
  createdAt: timestamp('created_at').defaultNow().notNull(),
  updatedAt: timestamp('updated_at').defaultNow().notNull(),
  lastLoginAt: timestamp('last_login_at'),
}, (table) => ({
  emailIdx: uniqueIndex('users_email_idx').on(table.email),
  usernameIdx: uniqueIndex('users_username_idx').on(table.username),
}));

// ============================================
// SESSIONS (Lucia Auth)
// ============================================

export const userSessions = pgTable('user_sessions', {
  id: text('id').primaryKey(),
  userId: uuid('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),
  expiresAt: timestamp('expires_at').notNull(),
  createdAt: timestamp('created_at').defaultNow().notNull(),
}, (table) => ({
  userIdx: index('sessions_user_idx').on(table.userId),
  expiresIdx: index('sessions_expires_idx').on(table.expiresAt),
}));

// ============================================
// API KEYS
// ============================================

export const apiKeys = pgTable('api_keys', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),

  name: text('name').notNull(),
  keyHash: text('key_hash').notNull().unique(),
  keyPrefix: text('key_prefix').notNull(),

  permissions: text('permissions').array().default(['agents:read', 'agents:execute']),
  rateLimit: integer('rate_limit').default(100),

  lastUsedAt: timestamp('last_used_at'),
  expiresAt: timestamp('expires_at'),
  isActive: boolean('is_active').default(true),

  createdAt: timestamp('created_at').defaultNow().notNull(),
}, (table) => ({
  userIdx: index('api_keys_user_idx').on(table.userId),
  hashIdx: uniqueIndex('api_keys_hash_idx').on(table.keyHash),
}));

// ============================================
// AGENTS
// ============================================

export const agents = pgTable('agents', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),

  name: text('name').notNull(),
  description: text('description'),
  icon: text('icon'),

  // Configuration
  mode: agentModeEnum('mode').default('auto'),
  status: agentStatusEnum('status').default('draft'),

  // LLM Settings
  model: text('model').default('gpt-4o-mini'),
  temperature: decimal('temperature', { precision: 3, scale: 2 }).default('0.7'),
  maxTokens: integer('max_tokens').default(4096),
  systemPrompt: text('system_prompt'),

  // Workflow (LangGraph definition)
  workflow: jsonb('workflow'),

  // Tools
  tools: text('tools').array(),

  // Memory
  memoryEnabled: boolean('memory_enabled').default(true),
  memoryWindow: integer('memory_window').default(10),

  // Sharing
  isPublic: boolean('is_public').default(false),
  isTemplate: boolean('is_template').default(false),

  // Stats
  executionCount: integer('execution_count').default(0),
  avgExecutionTime: integer('avg_execution_time'),
  successRate: decimal('success_rate', { precision: 5, scale: 2 }),

  createdAt: timestamp('created_at').defaultNow().notNull(),
  updatedAt: timestamp('updated_at').defaultNow().notNull(),
}, (table) => ({
  userIdx: index('agents_user_idx').on(table.userId),
  publicIdx: index('agents_public_idx').on(table.isPublic),
  templateIdx: index('agents_template_idx').on(table.isTemplate),
}));

// ============================================
// EXECUTIONS
// ============================================

export const executions = pgTable('executions', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),
  agentId: uuid('agent_id').notNull().references(() => agents.id, { onDelete: 'cascade' }),

  status: executionStatusEnum('status').default('pending'),
  mode: text('mode').notNull(),

  input: jsonb('input').notNull(),
  output: jsonb('output'),
  error: text('error'),

  startedAt: timestamp('started_at'),
  completedAt: timestamp('completed_at'),
  durationMs: integer('duration_ms'),

  // Cost tracking
  inputTokens: integer('input_tokens').default(0),
  outputTokens: integer('output_tokens').default(0),
  estimatedCost: decimal('estimated_cost', { precision: 10, scale: 6 }).default('0'),

  createdAt: timestamp('created_at').defaultNow().notNull(),
}, (table) => ({
  userIdx: index('executions_user_idx').on(table.userId),
  agentIdx: index('executions_agent_idx').on(table.agentId),
  statusIdx: index('executions_status_idx').on(table.status),
  createdIdx: index('executions_created_idx').on(table.createdAt),
}));

// ============================================
// EXECUTION STEPS
// ============================================

export const executionSteps = pgTable('execution_steps', {
  id: uuid('id').primaryKey().defaultRandom(),
  executionId: uuid('execution_id').notNull().references(() => executions.id, { onDelete: 'cascade' }),

  stepNumber: integer('step_number').notNull(),
  type: text('type').notNull(),
  name: text('name').notNull(),

  input: jsonb('input'),
  output: jsonb('output'),
  error: text('error'),

  startedAt: timestamp('started_at').defaultNow(),
  completedAt: timestamp('completed_at'),
  durationMs: integer('duration_ms'),

  inputTokens: integer('input_tokens').default(0),
  outputTokens: integer('output_tokens').default(0),

  createdAt: timestamp('created_at').defaultNow().notNull(),
}, (table) => ({
  executionIdx: index('steps_execution_idx').on(table.executionId, table.stepNumber),
}));

// ============================================
// CONVERSATIONS
// ============================================

export const conversations = pgTable('conversations', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),
  agentId: uuid('agent_id').notNull().references(() => agents.id, { onDelete: 'cascade' }),

  title: text('title'),

  createdAt: timestamp('created_at').defaultNow().notNull(),
  updatedAt: timestamp('updated_at').defaultNow().notNull(),
}, (table) => ({
  userIdx: index('conversations_user_idx').on(table.userId),
  agentIdx: index('conversations_agent_idx').on(table.agentId),
}));

// ============================================
// MESSAGES
// ============================================

export const messages = pgTable('messages', {
  id: uuid('id').primaryKey().defaultRandom(),
  conversationId: uuid('conversation_id').notNull().references(() => conversations.id, { onDelete: 'cascade' }),

  role: text('role').notNull(),
  content: text('content').notNull(),

  tokens: integer('tokens'),
  model: text('model'),

  toolCalls: jsonb('tool_calls'),
  toolCallId: text('tool_call_id'),

  createdAt: timestamp('created_at').defaultNow().notNull(),
}, (table) => ({
  conversationIdx: index('messages_conversation_idx').on(table.conversationId, table.createdAt),
}));

// ============================================
// TEMPLATES (Marketplace)
// ============================================

export const templates = pgTable('templates', {
  id: uuid('id').primaryKey().defaultRandom(),
  creatorId: uuid('creator_id').notNull().references(() => users.id, { onDelete: 'cascade' }),

  name: text('name').notNull(),
  description: text('description').notNull(),
  shortDescription: text('short_description'),
  category: text('category').default('custom'),
  tags: text('tags').array(),

  icon: text('icon'),
  previewImages: text('preview_images').array(),
  previewVideo: text('preview_video'),

  agentConfig: jsonb('agent_config').notNull(),
  workflow: jsonb('workflow').notNull(),
  exampleInputs: jsonb('example_inputs'),

  price: decimal('price', { precision: 10, scale: 2 }).default('0'),
  currency: text('currency').default('USD'),

  status: templateStatusEnum('status').default('draft'),
  isFeatured: boolean('is_featured').default(false),

  viewCount: integer('view_count').default(0),
  purchaseCount: integer('purchase_count').default(0),
  rating: decimal('rating', { precision: 2, scale: 1 }),
  reviewCount: integer('review_count').default(0),

  stripeProductId: text('stripe_product_id'),
  stripePriceId: text('stripe_price_id'),

  createdAt: timestamp('created_at').defaultNow().notNull(),
  updatedAt: timestamp('updated_at').defaultNow().notNull(),
  publishedAt: timestamp('published_at'),
}, (table) => ({
  creatorIdx: index('templates_creator_idx').on(table.creatorId),
  categoryIdx: index('templates_category_idx').on(table.category),
  statusIdx: index('templates_status_idx').on(table.status),
  featuredIdx: index('templates_featured_idx').on(table.isFeatured),
}));

// ============================================
// ACCESS PAYMENTS ($10 Tier)
// ============================================

export const accessPayments = pgTable('access_payments', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),

  amount: decimal('amount', { precision: 10, scale: 2 }).notNull().default('10.00'),
  currency: text('currency').default('USD'),

  // Payment details (Razorpay)
  razorpayOrderId: text('razorpay_order_id').unique(),
  razorpayPaymentId: text('razorpay_payment_id').unique(),
  paymentMethod: text('payment_method'), // card, upi, netbanking, wallet, etc

  // Legacy Stripe fields (optional)
  stripePaymentIntentId: text('stripe_payment_intent_id').unique(),
  stripeSessionId: text('stripe_session_id'),

  status: purchaseStatusEnum('status').default('pending'),

  // Access grants
  grantsArchitectureAccess: boolean('grants_architecture_access').default(true),
  grantsDownloadAccess: boolean('grants_download_access').default(true),
  grantsDuration: text('grants_duration').default('lifetime'), // 'lifetime', '1year', '1month'

  createdAt: timestamp('created_at').defaultNow().notNull(),
  completedAt: timestamp('completed_at'),
}, (table) => ({
  userIdx: index('access_payments_user_idx').on(table.userId),
  statusIdx: index('access_payments_status_idx').on(table.status),
}));

// ============================================
// PURCHASES
// ============================================

export const purchases = pgTable('purchases', {
  id: uuid('id').primaryKey().defaultRandom(),
  buyerId: uuid('buyer_id').notNull().references(() => users.id, { onDelete: 'cascade' }),
  templateId: uuid('template_id').notNull().references(() => templates.id, { onDelete: 'cascade' }),

  price: decimal('price', { precision: 10, scale: 2 }).notNull(),
  platformFee: decimal('platform_fee', { precision: 10, scale: 2 }).notNull(),
  creatorPayout: decimal('creator_payout', { precision: 10, scale: 2 }).notNull(),
  currency: text('currency').default('USD'),

  stripePaymentIntentId: text('stripe_payment_intent_id'),
  stripeTransferId: text('stripe_transfer_id'),

  status: purchaseStatusEnum('status').default('pending'),

  refundedAt: timestamp('refunded_at'),
  refundAmount: decimal('refund_amount', { precision: 10, scale: 2 }),

  createdAt: timestamp('created_at').defaultNow().notNull(),
}, (table) => ({
  buyerIdx: index('purchases_buyer_idx').on(table.buyerId),
  templateIdx: index('purchases_template_idx').on(table.templateId),
  uniquePurchase: uniqueIndex('purchases_unique_idx').on(table.buyerId, table.templateId),
}));

// ============================================
// USAGE STATS (Aggregated)
// ============================================

export const usageStats = pgTable('usage_stats', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),

  year: integer('year').notNull(),
  month: integer('month').notNull(),

  totalExecutions: integer('total_executions').default(0),
  successfulRuns: integer('successful_runs').default(0),
  failedRuns: integer('failed_runs').default(0),

  inputTokens: integer('input_tokens').default(0),
  outputTokens: integer('output_tokens').default(0),
  estimatedCost: decimal('estimated_cost', { precision: 10, scale: 4 }).default('0'),

  apiCalls: integer('api_calls').default(0),
}, (table) => ({
  userIdx: index('usage_stats_user_idx').on(table.userId, table.year, table.month),
  uniqueStats: uniqueIndex('usage_stats_unique_idx').on(table.userId, table.year, table.month),
}));

// ============================================
// AUDIT LOGS
// ============================================

export const auditLogs = pgTable('audit_logs', {
  id: uuid('id').primaryKey().defaultRandom(),

  userId: uuid('user_id').references(() => users.id, { onDelete: 'set null' }),
  apiKeyId: uuid('api_key_id'),

  action: text('action').notNull(),
  resource: text('resource').notNull(),
  resourceId: uuid('resource_id'),

  ipAddress: text('ip_address'),
  userAgent: text('user_agent'),

  beforeState: jsonb('before_state'),
  afterState: jsonb('after_state'),

  success: boolean('success').notNull(),
  error: text('error'),

  createdAt: timestamp('created_at').defaultNow().notNull(),
}, (table) => ({
  userIdx: index('audit_logs_user_idx').on(table.userId),
  actionIdx: index('audit_logs_action_idx').on(table.action),
  createdIdx: index('audit_logs_created_idx').on(table.createdAt),
}));

// ============================================
// RELATIONS
// ============================================

export const usersRelations = relations(users, ({ many }) => ({
  sessions: many(userSessions),
  apiKeys: many(apiKeys),
  agents: many(agents),
  executions: many(executions),
  conversations: many(conversations),
  templates: many(templates),
  purchases: many(purchases),
  usageStats: many(usageStats),
  auditLogs: many(auditLogs),
}));

export const agentsRelations = relations(agents, ({ one, many }) => ({
  user: one(users, {
    fields: [agents.userId],
    references: [users.id],
  }),
  executions: many(executions),
  conversations: many(conversations),
}));

export const executionsRelations = relations(executions, ({ one, many }) => ({
  user: one(users, {
    fields: [executions.userId],
    references: [users.id],
  }),
  agent: one(agents, {
    fields: [executions.agentId],
    references: [agents.id],
  }),
  steps: many(executionSteps),
}));

export const conversationsRelations = relations(conversations, ({ one, many }) => ({
  user: one(users, {
    fields: [conversations.userId],
    references: [users.id],
  }),
  agent: one(agents, {
    fields: [conversations.agentId],
    references: [agents.id],
  }),
  messages: many(messages),
}));

export const templatesRelations = relations(templates, ({ one, many }) => ({
  creator: one(users, {
    fields: [templates.creatorId],
    references: [users.id],
  }),
  purchases: many(purchases),
}));

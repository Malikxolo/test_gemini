# FlowAgent Deployment Guide

## ✅ Production-Ready Status: COMPLETE

This guide covers deploying FlowAgent to production with all recent fixes and improvements.

## Prerequisites

### Required Accounts
- [Supabase](https://supabase.com) - Database, Auth & Storage
- [Upstash](https://upstash.com) - Redis & QStash
- [Cloudflare](https://cloudflare.com) - Workers API
- [Vercel](https://vercel.com) - Frontend hosting
- [OpenAI](https://platform.openai.com) - LLM API
- (Optional) [Anthropic](https://console.anthropic.com) - Claude API
- (Optional) [Razorpay](https://razorpay.com) - Payments

### Required Tools
- Node.js 20+
- pnpm 8+
- Wrangler CLI (`npm install -g wrangler`)

## ⚠️ CRITICAL: Database Migration (DO THIS FIRST!)

**Before deploying, you MUST run the database schema migration:**

1. Go to [Supabase Dashboard](https://supabase.com/dashboard)
2. Select your project
3. Go to SQL Editor → New Query
4. Copy contents from `supabase/schema_v2.sql`
5. Click "Run"
6. Verify these tables are created:
   - `conversations` - Chat history
   - `messages` - Individual messages
   - `api_keys` - Encrypted API key storage
   - `projects` - Project management
   - `agent_personas` - Pre-built and custom agents
   - `usage_tracking` - Token and cost tracking

**Also verify:**
- Storage bucket `chat-attachments` exists
- RLS policies are enabled on all tables
- 5 pre-built agent personas are inserted

---

## Step 1: Database Setup (Supabase)

### 1.1 Create Supabase Project

1. Go to [Supabase Dashboard](https://supabase.com/dashboard)
2. Click "New Project"
3. Name it "flowagent"
4. Select region closest to your users
5. Copy the connection string

### 1.2 Configure Database

```bash
# Set environment variable
export DATABASE_URL="postgresql://user:password@ep-xxx.region.aws.neon.tech/neondb?sslmode=require"

# Generate and run migrations
cd packages/database
pnpm drizzle-kit generate:pg
pnpm drizzle-kit migrate
```

### 1.3 Enable Connection Pooling

In Neon Console:
1. Go to "Settings" → "Connection Pooling"
2. Enable pooling
3. Copy the pooled connection string
4. Use this for production `DATABASE_URL`

## Step 2: Cache & Queue Setup (Upstash)

### 2.1 Create Redis Database

1. Go to [Upstash Console](https://console.upstash.com/redis)
2. Click "Create Database"
3. Name it "flowagent-cache"
4. Select region (same as your main users)
5. Choose "Regional" (free tier available)
6. Copy REST URL and Token

```bash
export UPSTASH_REDIS_REST_URL="https://xxx.upstash.io"
export UPSTASH_REDIS_REST_TOKEN="xxx"
```

### 2.2 Create QStash

1. Go to [QStash Console](https://console.upstash.com/qstash)
2. Copy your QStash token

```bash
export QSTASH_TOKEN="xxx"
```

## Step 3: Cloudflare Setup

### 3.1 Create Cloudflare Account

1. Sign up at [Cloudflare](https://dash.cloudflare.com)
2. Add your domain (or use workers.dev subdomain for testing)

### 3.2 Get API Token

1. Go to "My Profile" → "API Tokens"
2. Click "Create Token"
3. Use "Edit Cloudflare Workers" template
4. Copy the token

```bash
export CLOUDFLARE_API_TOKEN="xxx"
```

### 3.3 Create R2 Bucket (Optional)

1. Go to "R2" in Cloudflare dashboard
2. Click "Create Bucket"
3. Name it "flowagent-storage"
4. Copy the bucket details

## Step 4: Deploy API (Cloudflare Workers)

### 4.1 Configure Wrangler

```bash
cd apps/api

# Login to Cloudflare
wrangler login

# Set secrets
wrangler secret put DATABASE_URL
wrangler secret put UPSTASH_REDIS_REST_URL
wrangler secret put UPSTASH_REDIS_REST_TOKEN
wrangler secret put QSTASH_TOKEN
wrangler secret put OPENAI_API_KEY
wrangler secret put ANTHROPIC_API_KEY
```

### 4.2 Deploy

```bash
# Deploy to production
wrangler deploy --env production

# Your API will be available at:
# https://flowagent-api-production.<your-subdomain>.workers.dev
```

### 4.3 Configure Custom Domain (Optional)

1. In Cloudflare dashboard, go to Workers
2. Select your worker
3. Click "Triggers" → "Add Custom Domain"
4. Enter domain like `api.flowagent.io`
5. DNS records will be added automatically

## Step 5: Deploy Frontend (Vercel)

### 5.1 Install Vercel CLI

```bash
npm install -g vercel
```

### 5.2 Configure Environment

Create `.env.production` in `apps/web`:

```bash
NEXT_PUBLIC_API_URL=https://api.flowagent.io
```

### 5.3 Deploy

```bash
cd apps/web

# Login to Vercel
vercel login

# Deploy
vercel --prod

# Set environment variables in Vercel dashboard
# Project Settings → Environment Variables
```

## Step 6: Deploy Agent Engine (AWS Lambda)

### 6.1 Configure AWS Credentials

```bash
# Configure AWS CLI
aws configure

# Enter:
# - AWS Access Key ID
# - AWS Secret Access Key
# - Default region (us-east-1)
# - Default output format (json)
```

### 6.2 Deploy with Serverless Framework

```bash
cd apps/agent-engine

# Install dependencies
pip install -r requirements.txt

# Deploy
serverless deploy --stage production

# Note the endpoint URL
# Example: https://xxx.execute-api.us-east-1.amazonaws.com/production
```

### 6.3 Set Environment Variables

```bash
# In serverless.yml or via AWS Console
export DATABASE_URL="..."
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

## Step 7: Configure CI/CD (GitHub Actions)

### 7.1 Add GitHub Secrets

Go to your GitHub repository → Settings → Secrets and variables → Actions

Add the following secrets:

```
# Vercel
VERCEL_TOKEN
VERCEL_ORG_ID
VERCEL_PROJECT_ID

# Cloudflare
CLOUDFLARE_API_TOKEN

# AWS
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY

# Database & Services
DATABASE_URL
UPSTASH_REDIS_REST_URL
UPSTASH_REDIS_REST_TOKEN
QSTASH_TOKEN
OPENAI_API_KEY
ANTHROPIC_API_KEY
```

### 7.2 Enable GitHub Actions

The workflow in `.github/workflows/deploy.yml` will automatically:
- Run tests on every PR
- Deploy to production on every push to `main`

## Step 8: Monitoring & Observability

### 8.1 Enable Vercel Analytics

1. Go to Vercel dashboard
2. Select your project
3. Click "Analytics" → "Enable"

### 8.2 Enable Cloudflare Analytics

1. Go to Cloudflare dashboard
2. Select your domain
3. Analytics are enabled by default

### 8.3 Set Up Sentry (Optional)

```bash
# Install Sentry
pnpm add @sentry/nextjs @sentry/node

# Configure in apps/web/sentry.config.js
```

## Production Checklist

Before going live, verify:

### Security
- [ ] All secrets stored securely (not in code)
- [ ] HTTPS enabled on all endpoints
- [ ] CORS configured correctly
- [ ] Rate limiting enabled
- [ ] Input validation working
- [ ] Authentication tested
- [ ] API keys hashed (never stored in plain text)

### Performance
- [ ] Database indexes created
- [ ] Caching configured
- [ ] CDN enabled
- [ ] Image optimization working
- [ ] API response times < 100ms (p95)

### Reliability
- [ ] Database backups enabled (Neon auto-backup)
- [ ] Error tracking configured
- [ ] Health checks implemented
- [ ] Monitoring alerts set up
- [ ] Zero-downtime deployment tested

### Cost Management
- [ ] Free tier limits monitored
- [ ] Usage tracking enabled
- [ ] Cost alerts configured
- [ ] Budget limits set

## Scaling

### From 0 → 1,000 Users
- Stay on free tiers
- Monitor usage
- No changes needed

### From 1,000 → 10,000 Users
- Upgrade Neon to "Scale" plan ($19/mo)
- Upgrade Upstash to Pro ($10/mo)
- Upgrade Vercel to Pro ($20/mo)
- **Total**: ~$50/mo + LLM costs

### From 10,000 → 100,000 Users
- Increase Neon resources (4 vCPU)
- Increase Upstash capacity
- Enable Cloudflare Workers paid tier
- **Total**: ~$200-300/mo + LLM costs

### From 100,000 → 1M+ Users
- Enterprise Neon plan
- Dedicated Upstash instance
- Cloudflare Enterprise (optional)
- **Total**: ~$1,000-2,000/mo + LLM costs

## Troubleshooting

### API Not Responding
1. Check Cloudflare Workers logs
2. Verify environment variables are set
3. Check database connection
4. Verify Redis connection

### Agent Execution Failing
1. Check Lambda logs in AWS CloudWatch
2. Verify Python dependencies installed
3. Check DATABASE_URL is accessible from Lambda
4. Verify LLM API keys are valid

### Database Connection Issues
1. Check Neon status page
2. Verify connection string is correct
3. Check connection pooling is enabled
4. Verify IP allowlist (if enabled)

### High Costs
1. Review Neon query analytics
2. Check Redis cache hit rate
3. Analyze LLM usage (input/output tokens)
4. Enable request caching
5. Optimize database queries

## Support

- Architecture Documentation: [ARCHITECTURE.md](./ARCHITECTURE.md)
- GitHub Issues: Create an issue for bugs
- Community: Join our Discord (link)

---

**Last Updated**: January 31, 2026
**Status**: Production-Ready

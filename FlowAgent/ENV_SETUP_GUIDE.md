# FlowAgent Environment Variables Setup Guide

Complete list of environment variables needed to run FlowAgent MVP.

---

## 🔴 **CRITICAL** - Required for MVP to Run

These are absolutely necessary. The system will not work without them.

### 1. Database
```bash
DATABASE_URL=postgresql://user:password@host:5432/flowagent
```
**Where to get it**: [Neon Console](https://console.neon.tech)
- Create a new project
- Copy the connection string
- Format: `postgresql://user:password@ep-xxx.region.aws.neon.tech/neondb?sslmode=require`

**Used by**:
- ✅ API (all database operations)
- ✅ Lambda (execution status updates)
- ✅ Database migrations

**Cost**: FREE (10GB free tier)

---

### 2. Upstash Redis (Rate Limiting & Caching)
```bash
UPSTASH_REDIS_REST_URL=https://xxx.upstash.io
UPSTASH_REDIS_REST_TOKEN=xxx
```
**Where to get it**: [Upstash Console](https://console.upstash.com)
- Create a new Redis database
- Click "REST API" tab
- Copy both URL and Token

**Used by**:
- ✅ API rate limiting middleware
- ✅ Request caching

**Cost**: FREE (10,000 requests/day)

---

### 3. Upstash QStash (Job Queue)
```bash
QSTASH_TOKEN=xxx
QSTASH_CURRENT_SIGNING_KEY=xxx
QSTASH_NEXT_SIGNING_KEY=xxx
```
**Where to get it**: [Upstash QStash](https://console.upstash.com/qstash)
- Go to QStash tab
- Copy the token
- Copy both signing keys (for webhook verification)

**Used by**:
- ✅ Agent execution queueing
- ✅ Webhook signature verification

**Cost**: FREE (500 messages/day)

---

### 4. OpenAI API
```bash
OPENAI_API_KEY=sk-xxx
```
**Where to get it**: [OpenAI Platform](https://platform.openai.com/api-keys)
- Create account / login
- Click "Create new secret key"
- Copy the key (starts with `sk-`)

**Used by**:
- ✅ GPT-4o-mini model (default)
- ✅ GPT-4o model (optional)
- ✅ Agent execution

**Cost**: Pay-as-you-go
- GPT-4o-mini: $0.15/1M input tokens, $0.60/1M output tokens
- Estimated: $0.01-0.10 per agent execution

---

### 5. Lambda Endpoints
```bash
LAMBDA_ENDPOINT=https://xxx.execute-api.us-east-1.amazonaws.com/production
LAMBDA_WEBHOOK_URL=https://your-api.workers.dev/webhooks/qstash/execute
```

**How to get them**:
1. Deploy Lambda first (see below)
2. AWS will give you `LAMBDA_ENDPOINT`
3. Deploy Cloudflare Worker to get `LAMBDA_WEBHOOK_URL`

**Used by**:
- ✅ Webhook forwarding to Lambda
- ✅ Agent execution triggers

**Cost**:
- AWS Lambda: FREE (1M requests/month)
- Cloudflare Workers: FREE (100k requests/day)

---

### 6. Frontend API URL
```bash
NEXT_PUBLIC_API_URL=http://localhost:8787
```

**Values**:
- Development: `http://localhost:8787`
- Production: `https://your-api.workers.dev`

**Used by**:
- ✅ Frontend API client
- ✅ All frontend-to-backend requests

**Cost**: FREE

---

## 🟡 **OPTIONAL** - Enhances Functionality

These add features but aren't required for basic MVP.

### 7. Anthropic API (Claude Models)
```bash
ANTHROPIC_API_KEY=sk-ant-xxx
```
**Where to get it**: [Anthropic Console](https://console.anthropic.com)
- Create account
- Go to API Keys
- Generate new key

**Used by**:
- ⚠️ Claude-3 models (if selected)
- ⚠️ Alternative to OpenAI

**Cost**: Pay-as-you-go
- Claude-3-Haiku: $0.25/1M input, $1.25/1M output
- Claude-3-Sonnet: $3/1M input, $15/1M output

**Skip if**: You only plan to use OpenAI models

---

### 8. Serper API (Web Search Tool)
```bash
SERPER_API_KEY=xxx
```
**Where to get it**: [Serper.dev](https://serper.dev)
- Sign up for free
- Copy API key from dashboard

**Used by**:
- ⚠️ Web search tool in agents
- ⚠️ Real-time information lookups

**Cost**: FREE (2,500 searches/month)

**Skip if**: Your agents don't need web search

---

## ⚪ **NOT NEEDED** - Future Features

These are in `.env.example` but **not used by current MVP**.

### 9. Cloudflare (API Deployment)
```bash
CLOUDFLARE_ACCOUNT_ID=xxx
CLOUDFLARE_API_TOKEN=xxx
```
**When needed**: Only for `wrangler deploy` command
**Skip for now**: Can deploy via Cloudflare dashboard

---

### 10. Stripe (Payments)
```bash
STRIPE_SECRET_KEY=sk_test_xxx
STRIPE_WEBHOOK_SECRET=whsec_xxx
```
**When needed**: Phase 2 - when adding paid subscriptions
**Skip for now**: Not implemented in MVP

---

### 11. AWS Credentials (Lambda Deployment)
```bash
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=xxx
AWS_REGION=us-east-1
```
**When needed**: Only for deploying Lambda
**Skip for now**: Set these when you deploy Lambda, not in `.env`

---

## 📋 Quick Setup Checklist

### For Local Development:
```bash
# 1. Copy template
cp .env.example .env

# 2. Fill in these REQUIRED variables:
DATABASE_URL=                    # From Neon
UPSTASH_REDIS_REST_URL=          # From Upstash
UPSTASH_REDIS_REST_TOKEN=        # From Upstash
QSTASH_TOKEN=                    # From Upstash QStash
QSTASH_CURRENT_SIGNING_KEY=      # From Upstash QStash
QSTASH_NEXT_SIGNING_KEY=         # From Upstash QStash
OPENAI_API_KEY=                  # From OpenAI
LAMBDA_ENDPOINT=                 # From AWS (after Lambda deploy)
LAMBDA_WEBHOOK_URL=              # From Cloudflare (after Worker deploy)

# 3. Optional but recommended:
SERPER_API_KEY=                  # From Serper.dev (for web search)

# 4. For frontend (.env.local in apps/web):
NEXT_PUBLIC_API_URL=http://localhost:8787  # For dev
```

---

## 🚀 Setup Order (Recommended)

### Step 1: Get Free Accounts (10 minutes)
1. ✅ [Neon](https://console.neon.tech) - Get `DATABASE_URL`
2. ✅ [Upstash](https://console.upstash.com) - Get Redis + QStash keys
3. ✅ [OpenAI](https://platform.openai.com) - Get `OPENAI_API_KEY`
4. ⚠️ [Serper](https://serper.dev) - Get `SERPER_API_KEY` (optional)

### Step 2: Set Up Local Environment (2 minutes)
```bash
cp .env.example .env
# Edit .env with the keys from Step 1
# Leave LAMBDA_ENDPOINT and LAMBDA_WEBHOOK_URL empty for now
```

### Step 3: Deploy Lambda (15 minutes)
```bash
cd apps/agent-engine
# Set up AWS credentials (one-time)
aws configure

# Create serverless.yml and deploy
serverless deploy --stage production
# Copy the endpoint URL to LAMBDA_ENDPOINT in .env
```

### Step 4: Deploy API (10 minutes)
```bash
cd apps/api
wrangler login
wrangler deploy --env production
# Copy the worker URL to LAMBDA_WEBHOOK_URL in .env
# Also copy to NEXT_PUBLIC_API_URL for production frontend
```

### Step 5: Deploy Frontend (5 minutes)
```bash
cd apps/web
# Set NEXT_PUBLIC_API_URL to your Cloudflare Worker URL
vercel --prod
```

---

## 💾 Environment Variable Storage

### Local Development
```bash
# Root .env (for database migrations)
/FlowAgent/.env

# Frontend .env
/FlowAgent/apps/web/.env.local

# Lambda .env (for local testing)
/FlowAgent/apps/agent-engine/.env
```

### Production Deployment

**Cloudflare Workers** (API):
```bash
wrangler secret put DATABASE_URL
wrangler secret put UPSTASH_REDIS_REST_URL
wrangler secret put UPSTASH_REDIS_REST_TOKEN
wrangler secret put QSTASH_TOKEN
wrangler secret put QSTASH_CURRENT_SIGNING_KEY
wrangler secret put QSTASH_NEXT_SIGNING_KEY
wrangler secret put LAMBDA_ENDPOINT
wrangler secret put LAMBDA_WEBHOOK_URL
wrangler secret put OPENAI_API_KEY
# Optional:
wrangler secret put SERPER_API_KEY
wrangler secret put ANTHROPIC_API_KEY
```

**AWS Lambda** (Agent Engine):
```bash
# Set via AWS Console or serverless.yml
DATABASE_URL
OPENAI_API_KEY
ANTHROPIC_API_KEY  # optional
SERPER_API_KEY     # optional
```

**Vercel** (Frontend):
```bash
# Set via Vercel dashboard or CLI
NEXT_PUBLIC_API_URL=https://your-api.workers.dev
```

---

## 🔒 Security Best Practices

### ✅ DO:
- Store secrets in environment variables, never in code
- Use different keys for development vs production
- Rotate API keys regularly
- Keep `.env` files out of git (already in `.gitignore`)
- Use Cloudflare/Vercel secret management in production

### ❌ DON'T:
- Commit `.env` files to git
- Share API keys in chat/email
- Use production keys in development
- Hardcode secrets in code
- Expose keys in client-side code (except `NEXT_PUBLIC_*`)

---

## 📊 Cost Summary

### Free Tier (Sufficient for MVP):
| Service | Free Tier | Cost After |
|---------|-----------|------------|
| Neon (Database) | 10 GB | $19/month |
| Upstash Redis | 10k req/day | $0.20/100k |
| Upstash QStash | 500 msg/day | $1/10k |
| AWS Lambda | 1M req/month | $0.20/1M |
| Cloudflare Workers | 100k req/day | $5/10M |
| Vercel | 100 GB-hrs | $20/month |
| **OpenAI** | Pay-as-you-go | ~$0.01-0.10/execution |
| Serper | 2,500 searches | $50/month |

**Estimated MVP Cost**: **$0-20/month** (mostly OpenAI usage)

---

## 🐛 Troubleshooting

### "DATABASE_URL not set"
```bash
# Make sure .env exists in project root
ls -la .env

# Check it's loaded
echo $DATABASE_URL
```

### "Invalid API key"
```bash
# Check for extra spaces or quotes
# Keys should be plain text, no quotes in .env file
# ❌ OPENAI_API_KEY="sk-xxx"
# ✅ OPENAI_API_KEY=sk-xxx
```

### "Lambda endpoint not reachable"
```bash
# Make sure Lambda is deployed first
# Then add endpoint to .env
# Then redeploy Cloudflare Worker
```

### "CORS error from frontend"
```bash
# Make sure NEXT_PUBLIC_API_URL matches your API URL
# Check API has correct CORS settings (already configured)
```

---

## 📞 Getting Help

**Missing service account?**
- All services offer free trials/tiers
- No credit card needed for initial setup
- Can upgrade later when needed

**Environment variable not working?**
1. Check spelling (case-sensitive)
2. Check for extra spaces
3. Restart development server
4. Check variable is used in code (see ENV_SETUP_GUIDE.md)

---

## ✅ Verification

After setting up environment variables:

```bash
# 1. Check .env file exists
cat .env | grep -v "^#" | grep "="

# 2. Test database connection
pnpm db:migrate

# 3. Start development
pnpm dev

# 4. Check API health
curl http://localhost:8787/health
```

If all commands succeed, your environment is ready! 🎉

---

**Last Updated**: January 31, 2026
**Required Variables**: 9 critical
**Optional Variables**: 2 recommended
**Not Needed**: 5 future features

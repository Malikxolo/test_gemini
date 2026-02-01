# ⚡ Quick Environment Setup

**Copy this to your `.env` file and fill in the values.**

---

## 🔴 REQUIRED (Must Have)

```bash
# 1. Database - Get from https://console.neon.tech
DATABASE_URL=postgresql://user:password@ep-xxx.neon.tech/neondb?sslmode=require

# 2. Redis Cache - Get from https://console.upstash.com (Redis tab)
UPSTASH_REDIS_REST_URL=https://xxx.upstash.io
UPSTASH_REDIS_REST_TOKEN=xxx

# 3. Job Queue - Get from https://console.upstash.com (QStash tab)
QSTASH_TOKEN=xxx
QSTASH_CURRENT_SIGNING_KEY=xxx
QSTASH_NEXT_SIGNING_KEY=xxx

# 4. OpenAI - Get from https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-xxx

# 5. Lambda & API URLs (fill after deployment)
LAMBDA_ENDPOINT=https://xxx.execute-api.us-east-1.amazonaws.com/production
LAMBDA_WEBHOOK_URL=https://your-api.workers.dev/webhooks/qstash/execute

# 6. Frontend API URL
NEXT_PUBLIC_API_URL=http://localhost:8787
```

---

## 🟡 OPTIONAL (Recommended)

```bash
# Web Search Tool - Get from https://serper.dev
SERPER_API_KEY=xxx

# Claude Models - Get from https://console.anthropic.com (optional)
ANTHROPIC_API_KEY=sk-ant-xxx

# Razorpay (for $10/₹800 payments) - Get from https://dashboard.razorpay.com
RAZORPAY_KEY_ID=rzp_test_xxx
RAZORPAY_KEY_SECRET=xxx
RAZORPAY_WEBHOOK_SECRET=xxx

# BYOK Encryption - Generate with: openssl rand -hex 32
ENCRYPTION_KEY=your-64-char-hex-string
```

---

## ⏭️ Next Steps

```bash
# 1. Copy template
cp .env.example .env

# 2. Edit .env and paste your API keys

# 3. Run migrations
pnpm db:migrate

# 4. Start development
pnpm dev
```

---

## 📝 Where to Get Each Key

| Variable | Service | URL | Free Tier |
|----------|---------|-----|-----------|
| DATABASE_URL | Neon | console.neon.tech | ✅ 10GB |
| UPSTASH_REDIS_* | Upstash | console.upstash.com | ✅ 10k/day |
| QSTASH_* | Upstash | console.upstash.com/qstash | ✅ 500/day |
| OPENAI_API_KEY | OpenAI | platform.openai.com/api-keys | Pay-as-go |
| SERPER_API_KEY | Serper | serper.dev | ✅ 2,500/mo |

**Total Setup Time**: ~10 minutes
**Total Cost**: $0-20/month (mostly OpenAI usage)

---

## ✅ Quick Test

After setting up:

```bash
# Test database connection
pnpm db:migrate

# Start server
pnpm dev

# Visit
http://localhost:8787/health
```

If you see `{"status":"ok"}` - you're ready! 🚀

---

**Full Details**: See `ENV_SETUP_GUIDE.md`

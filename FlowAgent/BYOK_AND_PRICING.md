# FlowAgent BYOK & $10 Pricing Model

**Bring Your Own Key (BYOK)** + **$10 Architecture Access**

---

## 💡 Pricing Model Overview

FlowAgent offers a unique hybrid pricing model:

### Option 1: Free Tier (Platform API Keys)
- ✅ Free to use
- ✅ Uses FlowAgent's API keys
- ✅ Pay only for AI usage (billed per execution)
- ⚠️ Subject to rate limits
- ⚠️ Shared infrastructure

### Option 2: BYOK (Bring Your Own Key)
- ✅ Use your own OpenAI/Anthropic API keys
- ✅ Direct billing to your account
- ✅ No markup on AI costs
- ✅ Higher rate limits
- ✅ More control over usage

### Option 3: Architecture Access (One-time payment)
- ✅ Full source code access
- ✅ Complete architecture documentation
- ✅ Download entire codebase
- ✅ **Lifetime access**
- ✅ Self-host capability
- ✅ White-label ready

**Pricing by Region:**
- 🇺🇸 USA: $10.00
- 🇮🇳 India: ₹800.00
- 🇪🇺 Europe: €9.50
- 🇬🇧 UK: £8.50

---

## 🔑 BYOK (Bring Your Own Key)

### What is BYOK?

Instead of using FlowAgent's shared API keys, you can bring your own API keys from:
- **OpenAI** - For GPT models
- **Anthropic** - For Claude models
- **Serper** - For web search functionality

### Benefits

1. **Direct Billing**: Pay OpenAI/Anthropic directly at their rates
2. **No Markup**: $0.00 platform fee on AI usage
3. **Full Control**: Monitor your own usage and costs
4. **Privacy**: Your API calls go directly to OpenAI/Anthropic
5. **Higher Limits**: Use your own rate limits (not shared)

### How It Works

```
1. Get API keys from OpenAI/Anthropic
2. Add them to FlowAgent (encrypted storage)
3. Execute agents using YOUR keys
4. Get billed directly by OpenAI/Anthropic
```

### Security

- ✅ Keys encrypted with AES-256-GCM
- ✅ Stored securely in database
- ✅ Never logged or exposed
- ✅ Decrypted only during execution
- ✅ Can be deleted anytime

---

## 💵 Architecture Access Tier

### What You Get

For a **one-time payment** (pricing varies by region), you receive:

#### 1. **Full Source Code**
- Complete FlowAgent codebase
- All frontend, backend, and agent engine code
- Database schemas and migrations
- Test suite (119 tests)

#### 2. **Architecture Documentation**
- Complete system architecture (ARCHITECTURE.md)
- Deployment guides
- API documentation
- Database design docs

#### 3. **Deployment Rights**
- Self-host on your infrastructure
- White-label capabilities
- Commercial usage allowed
- No recurring fees

#### 4. **Lifetime Updates**
- Access to future updates
- Bug fixes
- Security patches
- New features

### Use Cases

**Perfect for:**
- 🏢 **Enterprises** - Want full control and customization
- 🚀 **Startups** - Need to white-label and resell
- 👨‍💻 **Developers** - Want to learn and modify
- 🔒 **Privacy-focused** - Need on-premise deployment
- 💰 **Cost-sensitive** - Want to eliminate SaaS fees

### What's Included

```
✅ Complete source code (8,000+ lines)
✅ Database migrations
✅ Docker configurations
✅ Serverless deployment configs
✅ CI/CD pipeline examples
✅ Test suite (unit + integration + E2E)
✅ Environment setup guides
✅ Troubleshooting documentation
```

### What's NOT Included

```
❌ Hosted infrastructure (you provide)
❌ API keys (OpenAI, etc - you provide)
❌ Direct support (community support only)
❌ Custom development
❌ Training/consulting
```

---

## 📊 Pricing Comparison

| Feature | Free Tier | BYOK | $10 Access |
|---------|-----------|------|------------|
| **Cost** | Free | Free | $10 one-time |
| **AI Usage** | Platform pays, passes cost | You pay direct | You host, you pay |
| **API Keys** | Platform's keys | Your keys | Your keys |
| **Rate Limits** | Shared (100/day) | Your limits | Unlimited |
| **Source Code** | ❌ No | ❌ No | ✅ Yes |
| **Self-Host** | ❌ No | ❌ No | ✅ Yes |
| **White-Label** | ❌ No | ❌ No | ✅ Yes |
| **Updates** | Auto | Auto | Manual |
| **Support** | Community | Community | Community |

---

## 🔐 BYOK API Endpoints

### Set API Keys

```bash
PUT /api/byok
Authorization: Bearer <token>
Content-Type: application/json

{
  "openaiApiKey": "sk-...",
  "anthropicApiKey": "sk-ant-...",
  "serperApiKey": "..."
}
```

### Get API Keys (Masked)

```bash
GET /api/byok
Authorization: Bearer <token>

Response:
{
  "openaiApiKey": "sk-....xxxx",
  "anthropicApiKey": "sk-ant-....xxxx",
  "serperApiKey": "****....xxxx"
}
```

### Delete API Key

```bash
DELETE /api/byok/:keyType
Authorization: Bearer <token>

# keyType: openai | anthropic | serper
```

---

## 💳 Payment API Endpoints

### Create Checkout Session

```bash
POST /api/payments/checkout
Authorization: Bearer <token>
Content-Type: application/json

{
  "plan": "access-tier",
  "currency": "INR"  # or USD, EUR, GBP
}

Response:
{
  "paymentId": "...",
  "razorpayOrderId": "order_...",
  "amount": 800,
  "currency": "INR",
  "razorpayKeyId": "rzp_test_...",
  "userEmail": "user@example.com",
  "userName": "username"
}
```

### Verify Payment (After Razorpay Checkout)

```bash
POST /api/payments/verify
Authorization: Bearer <token>
Content-Type: application/json

{
  "razorpayOrderId": "order_...",
  "razorpayPaymentId": "pay_...",
  "razorpaySignature": "signature_..."
}

Response:
{
  "success": true,
  "message": "Payment verified and access granted successfully"
}
```

### Check Access Status

```bash
GET /api/payments/access-status
Authorization: Bearer <token>

Response:
{
  "hasAccess": true,
  "hasArchitectureAccess": true,
  "hasDownloadAccess": true,
  "purchasedAt": "2026-01-31T12:00:00Z",
  "expiresAt": null  // null = lifetime
}
```

### Payment History

```bash
GET /api/payments/history
Authorization: Bearer <token>

Response:
{
  "payments": [
    {
      "id": "...",
      "amount": "10.00",
      "status": "completed",
      "createdAt": "2026-01-31T12:00:00Z"
    }
  ]
}
```

---

## 🛠️ Implementation Details

### Database Schema

```sql
-- User table additions
ALTER TABLE users ADD COLUMN openai_api_key TEXT;
ALTER TABLE users ADD COLUMN anthropic_api_key TEXT;
ALTER TABLE users ADD COLUMN serper_api_key TEXT;
ALTER TABLE users ADD COLUMN has_architecture_access BOOLEAN DEFAULT FALSE;
ALTER TABLE users ADD COLUMN has_download_access BOOLEAN DEFAULT FALSE;
ALTER TABLE users ADD COLUMN access_purchased_at TIMESTAMP;
ALTER TABLE users ADD COLUMN access_expires_at TIMESTAMP;

-- Access payments table
CREATE TABLE access_payments (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  amount DECIMAL(10,2) NOT NULL DEFAULT 10.00,
  currency TEXT DEFAULT 'USD',
  stripe_payment_intent_id TEXT UNIQUE,
  status TEXT DEFAULT 'pending',
  grants_architecture_access BOOLEAN DEFAULT TRUE,
  grants_download_access BOOLEAN DEFAULT TRUE,
  grants_duration TEXT DEFAULT 'lifetime',
  created_at TIMESTAMP DEFAULT NOW(),
  completed_at TIMESTAMP
);
```

### Encryption

API keys are encrypted using AES-256-GCM:

```typescript
// Encryption
const encrypted = encrypt(apiKey);
// Format: iv:authTag:encryptedData

// Decryption
const apiKey = decrypt(encrypted);
```

**Encryption Key**: Set `ENCRYPTION_KEY` in environment (32-byte hex)

---

## 💰 Revenue Model

### For FlowAgent Platform

1. **SaaS Subscriptions**
   - Free tier users (ad-supported or freemium)
   - Pro tier ($29/month) - higher limits
   - Enterprise tier ($299/month) - dedicated

2. **$10 Architecture Access**
   - One-time payment
   - High volume, low friction
   - Target: Developers & Enterprises

3. **Marketplace Commission**
   - 20% on template sales
   - Passive recurring revenue

### For Users

1. **Free Tier**
   - Pay per use (AI costs + markup)
   - Good for testing

2. **BYOK**
   - Direct OpenAI billing
   - No platform fees
   - Best for regular users

3. **$10 Self-Host**
   - One-time payment
   - Infinite usage
   - Full control
   - Best for heavy users

---

## 🚀 Getting Started

### 1. Use Free Tier

```bash
# Just sign up and start using
# Platform handles everything
curl -X POST /api/agents/:id/execute \
  -H "Authorization: Bearer <token>" \
  -d '{"input": {"message": "Hello"}}'
```

### 2. Switch to BYOK

```bash
# Add your API keys
curl -X PUT /api/byok \
  -H "Authorization: Bearer <token>" \
  -d '{"openaiApiKey": "sk-..."}'

# Executions now use YOUR keys
curl -X POST /api/agents/:id/execute \
  -H "Authorization: Bearer <token>" \
  -d '{"input": {"message": "Hello"}}'
```

### 3. Purchase $10 Access

```bash
# 1. Create checkout
curl -X POST /api/payments/checkout \
  -H "Authorization: Bearer <token>"

# 2. Complete payment (Stripe)
# 3. Download source code
curl -X GET /api/download/source \
  -H "Authorization: Bearer <token>" \
  -o flowagent-source.zip
```

---

## 📦 Download Package Contents

After purchasing $10 access, you'll receive:

```
flowagent-v1.0.0.zip
├── README.md
├── ARCHITECTURE.md
├── DEPLOYMENT.md
├── LICENSE.md (Commercial use allowed)
├── apps/
│   ├── api/              # Cloudflare Workers API
│   ├── web/              # Next.js Frontend
│   └── agent-engine/     # AWS Lambda Agent
├── packages/
│   └── database/         # Drizzle ORM schemas
├── tests/
│   └── e2e/              # End-to-end tests
├── docs/
│   ├── api/              # API documentation
│   ├── deployment/       # Deployment guides
│   └── architecture/     # System design
├── docker-compose.yml
├── .env.example
└── package.json
```

**Total Size**: ~50MB (compressed)

---

## 🔒 License

### $10 Access License

```
FlowAgent Commercial License
Copyright (c) 2026 FlowAgent

Permission is granted to use, modify, and distribute this software
for commercial purposes, including white-labeling and resale, subject
to the following conditions:

1. Attribution to FlowAgent is optional
2. Modifications are allowed
3. Resale is permitted
4. No warranty provided
5. Support not included

This license is perpetual and non-revocable.
```

---

## ❓ FAQ

### Q: Can I try BYOK before paying?
**A**: Yes! BYOK is free. Just add your API keys.

### Q: Is the $10 payment recurring?
**A**: No. One-time payment, lifetime access.

### Q: Can I get a refund?
**A**: Yes, within 30 days if not satisfied.

### Q: Do I need coding skills for $10 access?
**A**: Yes. This is for developers who want to self-host.

### Q: Can I resell FlowAgent after buying?
**A**: Yes. Commercial use and resale are allowed.

### Q: Will I get updates?
**A**: Yes. Future updates included lifetime.

### Q: Is support included?
**A**: Community support only. No direct support.

### Q: Can I use BYOK AND pay $10?
**A**: Yes! They're independent features.

---

## 📞 Contact

- **Purchase Issues**: support@flowagent.io
- **Technical Questions**: docs@flowagent.io
- **Enterprise**: enterprise@flowagent.io

---

**Last Updated**: January 31, 2026
**Version**: 1.0.0
**Status**: Live

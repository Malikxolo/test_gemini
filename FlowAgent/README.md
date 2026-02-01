# FlowAgent - A+ Grade AI Agent Platform

Build, deploy, and manage AI agents at scale with **zero infrastructure cost at launch**.

This is a production-ready implementation of the FlowAgent A+ Architecture - designed to scale from 0 to 10M+ users with $0 at rest and linear cost scaling.

## 🚀 Features

- **Zero Cost at Launch**: Serverless-first architecture ($0 with 0 users)
- **Infinite Scalability**: 0 to 10M+ users without architectural changes
- **99.99% Availability**: Multi-region, fault-tolerant design
- **Enterprise Security**: Multi-layer security with edge protection
- **Smart Cost Optimization**: 70% LLM cost reduction via intelligent routing
- **Global Edge Network**: Sub-50ms latency in 300+ locations

## 📐 Architecture

This implementation follows the A+ Grade Architecture (v2.0) with:

- **Frontend**: Next.js 14 (App Router) + Tailwind CSS + shadcn/ui
- **API**: Cloudflare Workers + Hono + tRPC
- **Agent Engine**: AWS Lambda + Python + LangGraph
- **Database**: Neon Postgres (Serverless)
- **Cache**: Upstash Redis (Serverless)
- **Storage**: Cloudflare R2 (S3-compatible, zero egress)
- **Auth**: Lucia Auth (Session-based)
- **Queue**: Upstash QStash
- **Monitoring**: Vercel Analytics + Sentry

## 🏗️ Project Structure

```
FlowAgent/
├── apps/
│   ├── web/              # Next.js 14 frontend
│   ├── api/              # Cloudflare Workers API (Hono)
│   └── agent-engine/     # AWS Lambda agent execution (Python)
├── packages/
│   └── database/         # Shared database schema (Drizzle ORM)
├── ARCHITECTURE.md       # Full architecture documentation
└── README.md            # This file
```

## 🚀 Quick Start

### Prerequisites

- Node.js 20+
- pnpm 8+
- Python 3.11+ (for agent engine)
- Neon Postgres account
- Upstash account (Redis + QStash)
- Cloudflare account
- AWS account (for Lambda)

### 1. Clone and Install

```bash
git clone <repo-url>
cd FlowAgent
pnpm install
```

### 2. Set Up Environment Variables

```bash
cp .env.example .env
```

Fill in your environment variables:

```bash
# Database (Neon Postgres)
DATABASE_URL=postgresql://user:password@ep-xxx.region.aws.neon.tech/neondb?sslmode=require

# Cache & Queue (Upstash)
UPSTASH_REDIS_REST_URL=https://xxx.upstash.io
UPSTASH_REDIS_REST_TOKEN=xxx
QSTASH_TOKEN=xxx

# LLM APIs
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-ant-xxx

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8787
```

### 3. Set Up Database

```bash
# Generate migrations
pnpm db:generate

# Run migrations
pnpm db:migrate

# (Optional) Open Drizzle Studio to view database
pnpm db:studio
```

### 4. Start Development

```bash
# Start all services
pnpm dev

# This will start:
# - Next.js frontend (http://localhost:3000)
# - Cloudflare Workers API (http://localhost:8787)
```

### 5. Deploy

#### Deploy Frontend (Vercel)

```bash
cd apps/web
vercel deploy
```

#### Deploy API (Cloudflare Workers)

```bash
cd apps/api
wrangler deploy
```

#### Deploy Agent Engine (AWS Lambda)

```bash
cd apps/agent-engine
pip install -r requirements.txt
serverless deploy --stage production
```

## 📊 Cost Breakdown

| Users     | Monthly Cost | Components                                    |
|-----------|------------- |-----------------------------------------------|
| 0         | $0           | Everything on free tiers                      |
| 100       | $5-15        | Mostly free tier + minimal LLM costs          |
| 1,000     | $15-25       | Still on free tiers + LLM costs               |
| 10,000    | $120-180     | Vercel Pro + Paid tiers + LLM costs           |
| 100,000   | $850-1,200   | Scaled infrastructure + significant LLM costs |
| 1,000,000 | $6,200-9,500 | Enterprise scale                              |

See [ARCHITECTURE.md](./ARCHITECTURE.md) for detailed cost analysis.

## 🔐 Security

This implementation includes:

- ✅ Multi-layer edge security (Cloudflare WAF + DDoS)
- ✅ Input validation (Zod schemas)
- ✅ Rate limiting (token bucket algorithm)
- ✅ Session-based auth (Lucia + Argon2id)
- ✅ API key hashing (SHA-256)
- ✅ SQL injection prevention (parameterized queries)
- ✅ XSS protection (CSP headers)
- ✅ CSRF protection (SameSite cookies)
- ✅ Sandboxed code execution
- ✅ Path traversal protection
- ✅ SSRF prevention
- ✅ Audit logging

## 📈 Performance Targets

- API response time: < 100ms (p95)
- Agent execution: < 2s (simple tasks)
- Time to first byte: < 50ms (global)
- Cache hit rate: > 90%
- Uptime: 99.99%

## 🛠️ Development

### Available Scripts

```bash
# Development
pnpm dev              # Start all services
pnpm dev:web          # Start Next.js only
pnpm dev:api          # Start Cloudflare Workers only

# Build
pnpm build            # Build all packages
pnpm typecheck        # Type check
pnpm lint             # Lint code

# Database
pnpm db:generate      # Generate migrations
pnpm db:migrate       # Run migrations
pnpm db:studio        # Open Drizzle Studio

# Clean
pnpm clean            # Clean all build artifacts
```

### Tech Stack Details

#### Frontend (apps/web)
- Next.js 14 with App Router
- React 18 with Server Components
- Tailwind CSS + shadcn/ui components
- TanStack Query for data fetching
- Zustand for global state

#### API (apps/api)
- Cloudflare Workers (Edge runtime)
- Hono web framework
- Lucia Auth for sessions
- Zod for validation
- Upstash Redis for caching & rate limiting

#### Agent Engine (apps/agent-engine)
- AWS Lambda (Python 3.11)
- LangGraph for agent workflows
- LangChain for LLM orchestration
- Smart model routing for cost optimization

#### Database (packages/database)
- Drizzle ORM (type-safe)
- Neon Postgres (serverless)
- Automatic connection pooling
- Partitioned tables for high-volume data

## 📚 Documentation

- [Architecture Documentation](./ARCHITECTURE.md) - Complete A+ architecture guide
- [API Documentation](./apps/api/README.md) - API endpoints and usage
- [Database Schema](./packages/database/README.md) - Schema and migrations
- [Deployment Guide](./docs/deployment.md) - Production deployment

## 🤝 Contributing

This is a reference implementation of the A+ Architecture. Feel free to:

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 License

MIT License - see LICENSE file for details

## 🌟 Credits

Built following the A+ Grade Scalable Architecture principles:

- Zero cost at launch
- Linear cost scaling
- Infinite horizontal scalability
- 99.99% availability
- Enterprise security
- Global edge distribution

---

**Status**: Production-Ready MVP
**Version**: 1.0.0
**Architecture**: v2.0 (A+ Grade)

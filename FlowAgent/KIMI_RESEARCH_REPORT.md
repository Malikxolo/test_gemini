# FlowAgent - Complete Technical Architecture
## Open Source Multi-Agent Platform

**Version:** 1.0  
**Date:** January 31, 2026  
**Status:** Technical Specification

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Monorepo Structure](#3-monorepo-structure)
4. [Database Schema](#4-database-schema)
5. [API Specification](#5-api-specification)
6. [Agent Orchestration](#6-agent-orchestration)
7. [Marketplace Architecture](#7-marketplace-architecture)
8. [Security Model](#8-security-model)
9. [Cost Optimization](#9-cost-optimization)
10. [Scalability Strategy](#10-scalability-strategy)
11. [Self-Hosting Guide](#11-self-hosting-guide)
12. [Implementation Roadmap](#12-implementation-roadmap)

---

## 1. Executive Summary

### 1.1 Project Overview

FlowAgent is an open-source, self-hostable multi-agent platform that combines the visual simplicity of workflow builders with the power of modern AI agent orchestration. Unlike closed platforms like MiniMax Agent, FlowAgent provides full transparency, control, and flexibility.

### 1.2 Key Differentiators

| Feature | MiniMax | FlowAgent |
|---------|---------|-----------|
| **License** | Closed | MIT (Open Source) |
| **Self-Hosting** | No | Yes (Docker/K8s) |
| **Pricing** | Opaque | Flat ($0/$49/$149) |
| **Model Support** | Proprietary only | Multi-provider |
| **Marketplace** | No | Yes (70/30 split) |
| **Community** | Limited | Open ecosystem |

### 1.3 Core Capabilities

1. **Multi-Agent Orchestration**
   - Auto Mode: Fully autonomous task completion
   - Air Mode: Collaborative brainstorming
   - Custom Mode: User-defined agent workflows
   - Pro Mode: Advanced configuration

2. **Web App Generation**
   - React + Node.js backend generation
   - One-click deployment
   - Custom domain support

3. **Research & Analysis**
   - Web search integration
   - Browser automation
   - API data retrieval
   - Document analysis

4. **Code Generation & Execution**
   - Multi-language support (Python, JS, TS, etc.)
   - Sandboxed execution environment
   - GitHub integration

5. **Document Creation**
   - PPT generation
   - PDF creation
   - Word documents
   - Data visualization

6. **Desktop Application**
   - Cross-platform (Windows, macOS, Linux)
   - Local-first architecture
   - Offline capabilities

### 1.4 Technical Highlights

- **Monorepo**: Turborepo with pnpm workspaces
- **Frontend**: Next.js 14, React, Tailwind, shadcn/ui
- **Backend**: Node.js (tRPC), Python (agent engine)
- **Database**: PostgreSQL with Prisma ORM
- **Agent Engine**: LangGraph + LiteLLM + CrewAI patterns
- **Sandbox**: Docker + gVisor for secure code execution
- **Payments**: Stripe Connect for marketplace
- **Auth**: Clerk (cloud) / Keycloak (self-hosted)
- **Deployment**: Vercel, Railway, Docker, Kubernetes

### 1.5 Pricing Tiers

| Tier | Price | Features |
|------|-------|----------|
| **Free** | $0 | Unlimited agents, 1k executions/mo, community support |
| **Pro** | $49/mo | Unlimited executions, priority support, advanced features |
| **Enterprise** | $149/mo | SSO, audit logs, SLA, dedicated support |

---

## 2. System Architecture

### 2.1 High-Level Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        WEB[Web App<br/>Next.js 14]
        DESKTOP[Desktop App<br/>Tauri/Electron]
        API_CLIENT[API Clients]
    end

    subgraph "API Gateway"
        TRPC[tRPC Router]
        REST[REST API]
        WS[WebSocket Server]
    end

    subgraph "Core Services"
        ORCH[Orchestration Service<br/>Node.js]
        AGENT[Agent Engine<br/>Python/LangGraph]
        EXEC[Execution Service<br/>Docker/gVisor]
    end

    subgraph "Data Layer"
        PG[(PostgreSQL)]
        REDIS[(Redis)]
        S3[(Object Storage)]
        VECTOR[(Vector DB)]
    end

    subgraph "External Services"
        STRIPE[Stripe Connect]
        CLERK[Clerk Auth]
        LLM[LLM Providers]
        SEARCH[Search APIs]
    end

    WEB --> TRPC
    DESKTOP --> TRPC
    API_CLIENT --> REST
    
    TRPC --> ORCH
    REST --> ORCH
    WS --> ORCH
    
    ORCH --> AGENT
    ORCH --> EXEC
    
    AGENT --> PG
    AGENT --> REDIS
    AGENT --> VECTOR
    
    EXEC --> S3
    
    ORCH --> STRIPE
    ORCH --> CLERK
    AGENT --> LLM
    AGENT --> SEARCH
```

### 2.2 Service Architecture

```mermaid
graph LR
    subgraph "Frontend"
        UI[UI Components<br/>shadcn/ui]
        STATE[State Management<br/>Zustand]
        FLOW[Flow Builder<br/>React Flow]
    end

    subgraph "Backend - Node.js"
        API[API Layer<br/>tRPC]
        AUTH[Auth Service<br/>Clerk/Keycloak]
        BILLING[Billing Service<br/>Stripe]
        MARKET[Marketplace Service]
    end

    subgraph "Backend - Python"
        ENGINE[Agent Engine<br/>LangGraph]
        TOOLS[Tool Registry]
        MEMORY[Memory Manager]
        SANDBOX[Sandbox Manager]
    end

    subgraph "Infrastructure"
        DOCKER[Docker Runtime]
        K8S[Kubernetes]
        QUEUE[Task Queue<br/>Celery]
    end

    UI --> API
    FLOW --> API
    STATE --> API
    
    API --> AUTH
    API --> BILLING
    API --> MARKET
    API --> ENGINE
    
    ENGINE --> TOOLS
    ENGINE --> MEMORY
    ENGINE --> SANDBOX
    
    SANDBOX --> DOCKER
    ENGINE --> QUEUE
    QUEUE --> K8S
```

### 2.3 Data Flow Architecture

```mermaid
sequenceDiagram
    participant User
    participant WebApp
    participant APIGateway
    participant Orchestrator
    participant AgentEngine
    participant Sandbox
    participant Database
    participant LLM

    User->>WebApp: Create agent workflow
    WebApp->>APIGateway: POST /api/agents
    APIGateway->>Orchestrator: CreateAgentRequest
    Orchestrator->>Database: Save agent config
    Database-->>Orchestrator: Agent created
    Orchestrator-->>APIGateway: AgentResponse
    APIGateway-->>WebApp: 201 Created

    User->>WebApp: Execute agent
    WebApp->>APIGateway: POST /api/agents/{id}/execute
    APIGateway->>Orchestrator: ExecuteRequest
    Orchestrator->>Database: Get agent config
    Database-->>Orchestrator: Agent config
    
    Orchestrator->>AgentEngine: Initialize workflow
    AgentEngine->>LLM: Decompose task
    LLM-->>AgentEngine: Subtasks
    
    loop For each subtask
        AgentEngine->>AgentEngine: Select tools
        AgentEngine->>Sandbox: Execute code (if needed)
        Sandbox-->>AgentEngine: Results
        AgentEngine->>LLM: Process results
        LLM-->>AgentEngine: Next action
    end
    
    AgentEngine->>Database: Save execution state
    AgentEngine-->>Orchestrator: Execution complete
    Orchestrator-->>APIGateway: ExecutionResponse
    APIGateway-->>WebApp: Stream results
    WebApp-->>User: Display results
```

---

## 3. Monorepo Structure

### 3.1 Repository Layout

```
flowagent/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml
│   │   ├── release.yml
│   │   └── docker-publish.yml
│   └── ISSUE_TEMPLATE/
├── apps/
│   ├── web/                          # Next.js 14 web application
│   │   ├── app/                      # App router
│   │   ├── components/               # React components
│   │   ├── lib/                      # Utilities
│   │   ├── hooks/                    # Custom hooks
│   │   ├── stores/                   # Zustand stores
│   │   ├── types/                    # TypeScript types
│   │   ├── public/                   # Static assets
│   │   ├── next.config.js
│   │   ├── tailwind.config.ts
│   │   └── package.json
│   │
│   ├── desktop/                      # Tauri desktop app
│   │   ├── src/
│   │   ├── src-tauri/                # Rust backend
│   │   ├── package.json
│   │   └── tauri.conf.json
│   │
│   ├── api/                          # Node.js API server
│   │   ├── src/
│   │   │   ├── routers/              # tRPC routers
│   │   │   ├── services/             # Business logic
│   │   │   ├── middleware/           # Express middleware
│   │   │   ├── utils/                # Utilities
│   │   │   └── server.ts             # Entry point
│   │   ├── package.json
│   │   └── tsconfig.json
│   │
│   └── agent-engine/                 # Python agent engine
│       ├── src/
│       │   ├── agents/               # Agent implementations
│       │   ├── workflows/            # Workflow definitions
│       │   ├── tools/                # Tool implementations
│       │   ├── memory/               # Memory management
│       │   ├── sandbox/              # Sandbox management
│       │   ├── models/               # LLM integrations
│       │   └── main.py               # Entry point
│       ├── requirements.txt
│       ├── Dockerfile
│       └── pyproject.toml
│
├── packages/
│   ├── ui/                           # Shared UI components
│   │   ├── src/
│   │   │   ├── components/           # shadcn/ui components
│   │   │   ├── primitives/           # Base components
│   │   │   └── index.ts
│   │   ├── package.json
│   │   └── tailwind.config.ts
│   │
│   ├── config/                       # Shared configurations
│   │   ├── eslint/
│   │   ├── typescript/
│   │   └── tailwind/
│   │
│   ├── types/                        # Shared TypeScript types
│   │   ├── src/
│   │   │   ├── api.ts
│   │   │   ├── agents.ts
│   │   │   ├── marketplace.ts
│   │   │   └── index.ts
│   │   └── package.json
│   │
│   ├── database/                     # Prisma schema & client
│   │   ├── prisma/
│   │   │   ├── schema.prisma
│   │   │   └── migrations/
│   │   ├── src/
│   │   │   └── client.ts
│   │   └── package.json
│   │
│   ├── sdk/                          # FlowAgent SDK
│   │   ├── src/
│   │   │   ├── client.ts
│   │   │   ├── agents.ts
│   │   │   └── index.ts
│   │   └── package.json
│   │
│   └── utils/                        # Shared utilities
│       ├── src/
│       │   ├── validation.ts
│       │   ├── formatting.ts
│       │   └── index.ts
│       └── package.json
│
├── infrastructure/
│   ├── docker/
│   │   ├── docker-compose.yml
│   │   ├── Dockerfile.web
│   │   ├── Dockerfile.api
│   │   ├── Dockerfile.agent
│   │   └── Dockerfile.desktop
│   │
│   ├── kubernetes/
│   │   ├── base/
│   │   │   ├── namespace.yml
│   │   │   ├── configmap.yml
│   │   │   ├── secret.yml
│   │   │   ├── postgres.yml
│   │   │   ├── redis.yml
│   │   │   ├── api.yml
│   │   │   ├── agent-engine.yml
│   │   │   └── web.yml
│   │   └── overlays/
│   │       ├── development/
│   │       ├── staging/
│   │       └── production/
│   │
│   └── terraform/
│       ├── main.tf
│       ├── variables.tf
│       └── outputs.tf
│
├── docs/
│   ├── README.md
│   ├── architecture/
│   ├── api/
│   ├── deployment/
│   └── development/
│
├── scripts/
│   ├── setup.sh
│   ├── dev.sh
│   ├── test.sh
│   └── deploy.sh
│
├── .env.example
├── .gitignore
├── turbo.json
├── pnpm-workspace.yaml
├── pnpm-lock.yaml
├── package.json
├── LICENSE (MIT)
└── README.md
```

### 3.2 Package.json Configuration

**Root package.json:**

```json
{
  "name": "flowagent",
  "version": "1.0.0",
  "private": true,
  "description": "Open source multi-agent platform",
  "scripts": {
    "build": "turbo run build",
    "dev": "turbo run dev --parallel",
    "lint": "turbo run lint",
    "test": "turbo run test",
    "typecheck": "turbo run typecheck",
    "db:generate": "turbo run db:generate",
    "db:migrate": "turbo run db:migrate",
    "db:studio": "turbo run db:studio",
    "clean": "turbo run clean && rm -rf node_modules",
    "format": "prettier --write \"**/*.{ts,tsx,md}\"",
    "changeset": "changeset",
    "version-packages": "changeset version",
    "release": "turbo run build --filter=docs^... && changeset publish"
  },
  "devDependencies": {
    "@changesets/cli": "^2.27.1",
    "prettier": "^3.2.5",
    "turbo": "^1.12.4"
  },
  "packageManager": "pnpm@8.15.1",
  "engines": {
    "node": ">=18.0.0"
  }
}
```

**turbo.json:**

```json
{
  "$schema": "https://turbo.build/schema.json",
  "globalDependencies": ["**/.env.*local"],
  "pipeline": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": [".next/**", "!.next/cache/**", "dist/**"]
    },
    "dev": {
      "cache": false,
      "persistent": true
    },
    "lint": {
      "dependsOn": ["^build"]
    },
    "test": {
      "dependsOn": ["^build"]
    },
    "typecheck": {
      "dependsOn": ["^build"]
    },
    "db:generate": {
      "cache": false
    },
    "db:migrate": {
      "cache": false
    },
    "db:studio": {
      "cache": false
    },
    "clean": {
      "cache": false
    }
  }
}
```

**pnpm-workspace.yaml:**

```yaml
packages:
  - "apps/*"
  - "packages/*"
```

---

## 4. Database Schema

### 4.1 Complete Prisma Schema

```prisma
// packages/database/prisma/schema.prisma

generator client {
  provider = "prisma-client-js"
  output   = "../src/client"
}

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

// ============================================
// USER & AUTHENTICATION
// ============================================

model User {
  id            String    @id @default(cuid())
  email         String    @unique
  username      String    @unique
  displayName   String?
  avatarUrl     String?
  
  // Clerk integration
  clerkId       String    @unique
  
  // Subscription & billing
  subscription  Subscription?
  stripeCustomerId String?
  
  // User settings
  settings      UserSettings?
  
  // API access
  apiKeys       ApiKey[]
  
  // Content ownership
  agents        Agent[]
  templates     Template[]
  purchases     Purchase[]
  sales         Sale[]
  
  // Usage tracking
  executions    Execution[]
  usageStats    UsageStats?
  
  // Timestamps
  createdAt     DateTime  @default(now())
  updatedAt     DateTime  @updatedAt
  lastLoginAt   DateTime?
  
  // Relations
  reviews       Review[]
  conversations Conversation[]
  
  @@index([email])
  @@index([clerkId])
  @@map("users")
}

model UserSettings {
  id                    String   @id @default(cuid())
  userId                String   @unique
  user                  User     @relation(fields: [userId], references: [id], onDelete: Cascade)
  
  // Preferences
  defaultModel          String   @default("gpt-4o-mini")
  theme                 String   @default("system")
  language              String   @default("en")
  
  // Notifications
  emailNotifications    Boolean  @default(true)
  marketingEmails       Boolean  @default(false)
  
  // Privacy
  publicProfile         Boolean  @default(true)
  showActivity          Boolean  @default(true)
  
  // Advanced
  autoSave              Boolean  @default(true)
  confirmDestructive    Boolean  @default(true)
  
  createdAt             DateTime @default(now())
  updatedAt             DateTime @updatedAt
  
  @@map("user_settings")
}

// ============================================
// SUBSCRIPTION & BILLING
// ============================================

enum SubscriptionTier {
  FREE
  PRO
  ENTERPRISE
}

enum SubscriptionStatus {
  ACTIVE
  CANCELED
  PAST_DUE
  UNPAID
  TRIALING
}

model Subscription {
  id                String             @id @default(cuid())
  userId            String             @unique
  user              User               @relation(fields: [userId], references: [id], onDelete: Cascade)
  
  tier              SubscriptionTier   @default(FREE)
  status            SubscriptionStatus @default(ACTIVE)
  
  // Stripe integration
  stripeSubscriptionId String?
  stripePriceId        String?
  
  // Billing cycle
  currentPeriodStart DateTime
  currentPeriodEnd   DateTime
  cancelAtPeriodEnd  Boolean           @default(false)
  
  // Usage limits
  maxAgents          Int               @default(10)
  maxExecutions      Int               @default(1000)
  maxTemplates       Int               @default(5)
  maxApiKeys         Int               @default(3)
  
  // Features
  features           Json              @default("{}")
  
  createdAt          DateTime          @default(now())
  updatedAt          DateTime          @updatedAt
  
  @@map("subscriptions")
}

model UsageStats {
  id                String   @id @default(cuid())
  userId            String   @unique
  user              User     @relation(fields: [userId], references: [id], onDelete: Cascade)
  
  // Monthly usage
  month             Int
  year              Int
  
  // Execution counts
  totalExecutions   Int      @default(0)
  successfulRuns    Int      @default(0)
  failedRuns        Int      @default(0)
  
  // Token usage (approximate cost tracking)
  inputTokens       Int      @default(0)
  outputTokens      Int      @default(0)
  estimatedCost     Decimal  @default(0) @db.Decimal(10, 4)
  
  // API usage
  apiCalls          Int      @default(0)
  
  // Unique constraint for monthly stats
  @@unique([userId, year, month])
  @@map("usage_stats")
}

// ============================================
// API KEYS & RATE LIMITING
// ============================================

model ApiKey {
  id            String    @id @default(cuid())
  userId        String
  user          User      @relation(fields: [userId], references: [id], onDelete: Cascade)
  
  name          String
  keyHash       String    @unique // Hashed key for lookup
  keyPrefix     String    // First 8 chars for display
  
  // Permissions
  permissions   String[]  @default(["agents:read", "agents:execute"])
  
  // Rate limiting
  rateLimit     Int       @default(100) // requests per minute
  
  // Usage
  lastUsedAt    DateTime?
  usageCount    Int       @default(0)
  
  // Status
  isActive      Boolean   @default(true)
  expiresAt     DateTime?
  
  createdAt     DateTime  @default(now())
  
  @@index([userId])
  @@index([keyHash])
  @@map("api_keys")
}

// ============================================
// AGENTS & TEMPLATES
// ============================================

enum AgentMode {
  AUTO      // Fully autonomous
  AIR       // Collaborative brainstorming
  CUSTOM    // User-defined workflow
  PRO       // Advanced configuration
}

enum AgentStatus {
  DRAFT
  ACTIVE
  ARCHIVED
}

model Agent {
  id            String       @id @default(cuid())
  userId        String
  user          User         @relation(fields: [userId], references: [id], onDelete: Cascade)
  
  // Basic info
  name          String
  description   String?
  icon          String?      // Emoji or icon URL
  
  // Configuration
  mode          AgentMode    @default(AUTO)
  status        AgentStatus  @default(DRAFT)
  
  // LLM settings
  model         String       @default("gpt-4o-mini")
  temperature   Float        @default(0.7)
  maxTokens     Int?         @default(4096)
  systemPrompt  String?      @db.Text
  
  // Workflow definition (JSON)
  workflow      Json?        // LangGraph workflow definition
  
  // Tools
  tools         Tool[]
  
  // Memory settings
  memoryEnabled Boolean      @default(true)
  memoryWindow  Int          @default(10) // Number of messages to remember
  
  // Sharing
  isPublic      Boolean      @default(false)
  isTemplate    Boolean      @default(false)
  templateId    String?      // If created from template
  
  // Marketplace
  template      Template?    @relation("AgentTemplate")
  
  // Stats
  executionCount Int         @default(0)
  avgExecutionTime Float?
  successRate   Float?
  
  // Relations
  executions    Execution[]
  conversations Conversation[]
  
  createdAt     DateTime     @default(now())
  updatedAt     DateTime     @updatedAt
  
  @@index([userId])
  @@index([isPublic])
  @@index([isTemplate])
  @@map("agents")
}

model Tool {
  id            String    @id @default(cuid())
  
  // Tool identification
  name          String    @unique
  displayName   String
  description   String
  icon          String?
  
  // Tool configuration
  type          ToolType
  config        Json      // Tool-specific configuration
  
  // Schema for inputs
  inputSchema   Json      // JSON Schema
  
  // Relations
  agents        Agent[]
  
  // Metadata
  isBuiltin     Boolean   @default(false)
  isPremium     Boolean   @default(false)
  
  createdAt     DateTime  @default(now())
  updatedAt     DateTime  @updatedAt
  
  @@map("tools")
}

enum ToolType {
  WEB_SEARCH
  BROWSER
  CODE_EXECUTION
  API_CALL
  FILE_OPERATION
  DATABASE
  CUSTOM
}

// ============================================
// MARKETPLACE
// ============================================

enum TemplateCategory {
  PRODUCTIVITY
  DEVELOPMENT
  RESEARCH
  MARKETING
  SALES
  SUPPORT
  FINANCE
  HR
  LEGAL
  CUSTOM
}

enum TemplateStatus {
  DRAFT
  PENDING_REVIEW
  APPROVED
  REJECTED
  REMOVED
}

model Template {
  id                String           @id @default(cuid())
  creatorId         String
  creator           User             @relation(fields: [creatorId], references: [id], onDelete: Cascade)
  
  // Template info
  name              String
  description       String           @db.Text
  shortDescription  String?          // For cards
  category          TemplateCategory @default(CUSTOM)
  tags              String[]
  
  // Media
  icon              String?          // Emoji or URL
  previewImages     String[]         // URLs
  previewVideo      String?          // URL
  
  // Content
  agentConfig       Json             // Complete agent configuration
  workflow          Json             // Workflow definition
  exampleInputs     Json?            // Example inputs for preview
  
  // Pricing
  price             Decimal          @default(0) @db.Decimal(10, 2)
  currency          String           @default("USD")
  
  // Status
  status            TemplateStatus   @default(DRAFT)
  isFeatured        Boolean          @default(false)
  
  // Stats
  viewCount         Int              @default(0)
  purchaseCount     Int              @default(0)
  rating            Float?           // Average rating
  reviewCount       Int              @default(0)
  
  // Stripe Connect
  stripeProductId   String?
  stripePriceId     String?
  
  // Relations
  agent             Agent?           @relation("AgentTemplate", fields: [agentId], references: [id])
  agentId           String?          @unique
  reviews           Review[]
  purchases         Purchase[]
  
  createdAt         DateTime         @default(now())
  updatedAt         DateTime         @updatedAt
  publishedAt       DateTime?
  
  @@index([creatorId])
  @@index([category])
  @@index([status])
  @@index([isFeatured])
  @@index([price])
  @@map("templates")
}

model Review {
  id            String    @id @default(cuid())
  
  templateId    String
  template      Template  @relation(fields: [templateId], references: [id], onDelete: Cascade)
  
  userId        String
  user          User      @relation(fields: [userId], references: [id], onDelete: Cascade)
  
  rating        Int       // 1-5
  title         String?
  content       String    @db.Text
  
  // Verified purchase
  isVerified    Boolean   @default(false)
  purchaseId    String?
  
  createdAt     DateTime  @default(now())
  updatedAt     DateTime  @updatedAt
  
  @@unique([templateId, userId])
  @@map("reviews")
}

model Purchase {
  id                String    @id @default(cuid())
  
  buyerId           String
  buyer             User      @relation(fields: [buyerId], references: [id], onDelete: Cascade)
  
  templateId        String
  template          Template  @relation(fields: [templateId], references: [id], onDelete: Cascade)
  
  // Pricing
  price             Decimal   @db.Decimal(10, 2)
  platformFee       Decimal   @db.Decimal(10, 2) // 30%
  creatorPayout     Decimal   @db.Decimal(10, 2) // 70%
  currency          String    @default("USD")
  
  // Stripe
  stripePaymentIntentId String?
  stripeTransferId      String?
  
  // Status
  status            PurchaseStatus @default(PENDING)
  
  // Refund
  refundedAt        DateTime?
  refundAmount      Decimal?   @db.Decimal(10, 2)
  
  createdAt         DateTime   @default(now())
  
  @@unique([buyerId, templateId])
  @@map("purchases")
}

enum PurchaseStatus {
  PENDING
  COMPLETED
  FAILED
  REFUNDED
}

model Sale {
  id                String    @id @default(cuid())
  sellerId          String
  seller            User      @relation(fields: [sellerId], references: [id], onDelete: Cascade)
  
  // Aggregated stats for payout calculations
  periodStart       DateTime
  periodEnd         DateTime
  
  totalSales        Int       @default(0)
  totalRevenue      Decimal   @default(0) @db.Decimal(10, 2)
  platformFees      Decimal   @default(0) @db.Decimal(10, 2)
  netPayout         Decimal   @default(0) @db.Decimal(10, 2)
  
  // Payout
  payoutStatus      PayoutStatus @default(PENDING)
  payoutDate        DateTime?
  stripeTransferId  String?
  
  createdAt         DateTime  @default(now())
  updatedAt         DateTime  @updatedAt
  
  @@map("sales")
}

enum PayoutStatus {
  PENDING
  PROCESSING
  COMPLETED
  FAILED
}

// ============================================
// EXECUTION & CONVERSATIONS
// ============================================

enum ExecutionStatus {
  PENDING
  RUNNING
  PAUSED
  COMPLETED
  FAILED
  CANCELLED
  TIMEOUT
}

model Execution {
  id            String           @id @default(cuid())
  
  agentId       String
  agent         Agent            @relation(fields: [agentId], references: [id], onDelete: Cascade)
  
  userId        String
  user          User             @relation(fields: [userId], references: [id], onDelete: Cascade)
  
  // Execution details
  status        ExecutionStatus  @default(PENDING)
  mode          AgentMode
  
  // Input/Output
  input         Json             // Initial input
  output        Json?            // Final output
  error         String?          // Error message if failed
  
  // Performance
  startedAt     DateTime?
  completedAt   DateTime?
  durationMs    Int?             // Execution time
  
  // Cost tracking
  inputTokens   Int              @default(0)
  outputTokens  Int              @default(0)
  estimatedCost Decimal          @default(0) @db.Decimal(10, 6)
  
  // Steps/Trace
  steps         ExecutionStep[]
  
  // Conversation
  conversationId String?
  conversation   Conversation?   @relation(fields: [conversationId], references: [id])
  
  createdAt     DateTime         @default(now())
  updatedAt     DateTime         @updatedAt
  
  @@index([agentId])
  @@index([userId])
  @@index([status])
  @@index([createdAt])
  @@map("executions")
}

model ExecutionStep {
  id            String    @id @default(cuid())
  
  executionId   String
  execution     Execution @relation(fields: [executionId], references: [id], onDelete: Cascade)
  
  stepNumber    Int
  
  // Step details
  type          String    // "llm", "tool", "condition", etc.
  name          String    // Human-readable name
  
  // Input/Output
  input         Json?     // Step input
  output        Json?     // Step output
  error         String?   // Error if failed
  
  // Performance
  startedAt     DateTime  @default(now())
  completedAt   DateTime?
  durationMs    Int?
  
  // Cost
  inputTokens   Int       @default(0)
  outputTokens  Int       @default(0)
  
  createdAt     DateTime  @default(now())
  
  @@unique([executionId, stepNumber])
  @@map("execution_steps")
}

model Conversation {
  id            String    @id @default(cuid())
  
  userId        String
  user          User      @relation(fields: [userId], references: [id], onDelete: Cascade)
  
  agentId       String
  agent         Agent     @relation(fields: [agentId], references: [id], onDelete: Cascade)
  
  title         String?   // Auto-generated or user-defined
  
  // Messages
  messages      Message[]
  
  // Executions in this conversation
  executions    Execution[]
  
  createdAt     DateTime  @default(now())
  updatedAt     DateTime  @updatedAt
  
  @@index([userId])
  @@index([agentId])
  @@map("conversations")
}

model Message {
  id              String        @id @default(cuid())
  
  conversationId  String
  conversation    Conversation  @relation(fields: [conversationId], references: [id], onDelete: Cascade)
  
  role            MessageRole
  content         String        @db.Text
  
  // Metadata
  tokens          Int?          // Token count
  model           String?       // Model used
  
  // Tool calls
  toolCalls       Json?         // If assistant with tool calls
  toolCallId      String?       // If tool response
  
  // Attachments
  attachments     Attachment[]
  
  createdAt       DateTime      @default(now())
  
  @@index([conversationId])
  @@map("messages")
}

enum MessageRole {
  SYSTEM
  USER
  ASSISTANT
  TOOL
}

model Attachment {
  id          String    @id @default(cuid())
  
  messageId   String
  message     Message   @relation(fields: [messageId], references: [id], onDelete: Cascade)
  
  name        String
  type        String    // MIME type
  size        Int       // Bytes
  url         String    // Storage URL
  
  createdAt   DateTime  @default(now())
  
  @@map("attachments")
}

// ============================================
// SYSTEM & CONFIGURATION
// ============================================

model SystemConfig {
  id            String    @id @default(cuid())
  key           String    @unique
  value         Json
  
  description   String?
  
  createdAt     DateTime  @default(now())
  updatedAt     DateTime  @updatedAt
  
  @@map("system_config")
}

model RateLimitLog {
  id            String    @id @default(cuid())
  
  identifier    String    // API key or user ID
  endpoint      String
  
  requests      Int       @default(1)
  windowStart   DateTime  @default(now())
  
  blocked       Boolean   @default(false)
  blockedUntil  DateTime?
  
  @@index([identifier])
  @@index([windowStart])
  @@map("rate_limit_logs")
}

// ============================================
// AUDIT LOG
// ============================================

model AuditLog {
  id            String    @id @default(cuid())
  
  userId        String?
  apiKeyId      String?
  
  action        String    // e.g., "agent.create", "agent.execute"
  resource      String    // e.g., "agent", "template"
  resourceId    String?
  
  // Request details
  ipAddress     String?
  userAgent     String?
  
  // Change details
  before        Json?     // Previous state
  after         Json?     // New state
  
  // Result
  success       Boolean
  error         String?
  
  createdAt     DateTime  @default(now())
  
  @@index([userId])
  @@index([action])
  @@index([createdAt])
  @@map("audit_logs")
}
```

### 4.2 Database Indexes

Key indexes for performance:

```sql
-- User lookups
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_clerk_id ON users(clerkId);

-- Agent queries
CREATE INDEX idx_agents_user_id ON agents(userId);
CREATE INDEX idx_agents_public ON agents(isPublic) WHERE isPublic = true;
CREATE INDEX idx_agents_template ON agents(isTemplate) WHERE isTemplate = true;

-- Execution queries
CREATE INDEX idx_executions_user_id ON executions(userId);
CREATE INDEX idx_executions_agent_id ON executions(agentId);
CREATE INDEX idx_executions_status ON executions(status);
CREATE INDEX idx_executions_created_at ON executions(createdAt);

-- Marketplace queries
CREATE INDEX idx_templates_category ON templates(category);
CREATE INDEX idx_templates_status ON templates(status);
CREATE INDEX idx_templates_featured ON templates(isFeatured) WHERE isFeatured = true;
CREATE INDEX idx_templates_price ON templates(price);

-- Full-text search
CREATE INDEX idx_templates_search ON templates USING gin(to_tsvector('english', name || ' ' || coalesce(description, '')));
```

---

## 5. API Specification

### 5.1 tRPC Router Structure

```typescript
// apps/api/src/routers/index.ts

import { router } from '../trpc';
import { agentRouter } from './agent';
import { templateRouter } from './template';
import { marketplaceRouter } from './marketplace';
import { userRouter } from './user';
import { billingRouter } from './billing';
import { executionRouter } from './execution';
import { apiKeyRouter } from './api-key';

export const appRouter = router({
  agent: agentRouter,
  template: templateRouter,
  marketplace: marketplaceRouter,
  user: userRouter,
  billing: billingRouter,
  execution: executionRouter,
  apiKey: apiKeyRouter,
});

export type AppRouter = typeof appRouter;
```

### 5.2 Agent Router

```typescript
// apps/api/src/routers/agent.ts

import { z } from 'zod';
import { router, protectedProcedure, publicProcedure } from '../trpc';
import { AgentMode, AgentStatus } from '@flowagent/database';

const createAgentSchema = z.object({
  name: z.string().min(1).max(100),
  description: z.string().max(500).optional(),
  mode: z.nativeEnum(AgentMode).default(AgentMode.AUTO),
  model: z.string().default('gpt-4o-mini'),
  temperature: z.number().min(0).max(2).default(0.7),
  systemPrompt: z.string().optional(),
  tools: z.array(z.string()).default([]),
  workflow: z.record(z.any()).optional(),
  isPublic: z.boolean().default(false),
});

const updateAgentSchema = createAgentSchema.partial().extend({
  id: z.string(),
});

export const agentRouter = router({
  // List agents
  list: protectedProcedure
    .input(z.object({
      cursor: z.string().optional(),
      limit: z.number().min(1).max(100).default(20),
      mode: z.nativeEnum(AgentMode).optional(),
      status: z.nativeEnum(AgentStatus).optional(),
      search: z.string().optional(),
    }).optional())
    .query(async ({ ctx, input }) => {
      const { prisma, user } = ctx;
      
      const agents = await prisma.agent.findMany({
        where: {
          userId: user.id,
          ...(input?.mode && { mode: input.mode }),
          ...(input?.status && { status: input.status }),
          ...(input?.search && {
            OR: [
              { name: { contains: input.search, mode: 'insensitive' } },
              { description: { contains: input.search, mode: 'insensitive' } },
            ],
          }),
        },
        take: input?.limit ?? 20,
        skip: input?.cursor ? 1 : 0,
        cursor: input?.cursor ? { id: input.cursor } : undefined,
        orderBy: { updatedAt: 'desc' },
        include: {
          tools: true,
          _count: {
            select: { executions: true },
          },
        },
      });
      
      const nextCursor = agents.length === (input?.limit ?? 20) 
        ? agents[agents.length - 1].id 
        : undefined;
      
      return { agents, nextCursor };
    }),

  // Get single agent
  get: protectedProcedure
    .input(z.object({ id: z.string() }))
    .query(async ({ ctx, input }) => {
      const { prisma, user } = ctx;
      
      const agent = await prisma.agent.findFirst({
        where: {
          id: input.id,
          OR: [
            { userId: user.id },
            { isPublic: true },
          ],
        },
        include: {
          tools: true,
          template: true,
        },
      });
      
      if (!agent) {
        throw new Error('Agent not found');
      }
      
      return agent;
    }),

  // Create agent
  create: protectedProcedure
    .input(createAgentSchema)
    .mutation(async ({ ctx, input }) => {
      const { prisma, user } = ctx;
      
      // Check agent limit
      const subscription = await prisma.subscription.findUnique({
        where: { userId: user.id },
      });
      
      const agentCount = await prisma.agent.count({
        where: { userId: user.id },
      });
      
      if (agentCount >= (subscription?.maxAgents ?? 10)) {
        throw new Error('Agent limit reached. Upgrade to create more agents.');
      }
      
      const agent = await prisma.agent.create({
        data: {
          ...input,
          userId: user.id,
          status: AgentStatus.DRAFT,
          tools: {
            connect: input.tools.map(id => ({ id })),
          },
        },
        include: {
          tools: true,
        },
      });
      
      // Log audit
      await prisma.auditLog.create({
        data: {
          userId: user.id,
          action: 'agent.create',
          resource: 'agent',
          resourceId: agent.id,
          after: agent,
          success: true,
        },
      });
      
      return agent;
    }),

  // Update agent
  update: protectedProcedure
    .input(updateAgentSchema)
    .mutation(async ({ ctx, input }) => {
      const { prisma, user } = ctx;
      const { id, tools, ...data } = input;
      
      const existing = await prisma.agent.findFirst({
        where: { id, userId: user.id },
      });
      
      if (!existing) {
        throw new Error('Agent not found');
      }
      
      const agent = await prisma.agent.update({
        where: { id },
        data: {
          ...data,
          ...(tools && {
            tools: {
              set: [],
              connect: tools.map(id => ({ id })),
            },
          }),
        },
        include: {
          tools: true,
        },
      });
      
      await prisma.auditLog.create({
        data: {
          userId: user.id,
          action: 'agent.update',
          resource: 'agent',
          resourceId: agent.id,
          before: existing,
          after: agent,
          success: true,
        },
      });
      
      return agent;
    }),

  // Delete agent
  delete: protectedProcedure
    .input(z.object({ id: z.string() }))
    .mutation(async ({ ctx, input }) => {
      const { prisma, user } = ctx;
      
      const existing = await prisma.agent.findFirst({
        where: { id, userId: user.id },
      });
      
      if (!existing) {
        throw new Error('Agent not found');
      }
      
      await prisma.agent.delete({
        where: { id: input.id },
      });
      
      await prisma.auditLog.create({
        data: {
          userId: user.id,
          action: 'agent.delete',
          resource: 'agent',
          resourceId: input.id,
          before: existing,
          success: true,
        },
      });
      
      return { success: true };
    }),

  // Execute agent
  execute: protectedProcedure
    .input(z.object({
      id: z.string(),
      input: z.record(z.any()),
      stream: z.boolean().default(false),
    }))
    .mutation(async ({ ctx, input }) => {
      const { prisma, user, agentEngine } = ctx;
      
      // Check execution limits
      const subscription = await prisma.subscription.findUnique({
        where: { userId: user.id },
      });
      
      const executionCount = await prisma.execution.count({
        where: {
          userId: user.id,
          createdAt: {
            gte: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
          },
        },
      });
      
      if (executionCount >= (subscription?.maxExecutions ?? 1000)) {
        throw new Error('Execution limit reached. Upgrade for more executions.');
      }
      
      const agent = await prisma.agent.findFirst({
        where: { id: input.id, userId: user.id },
        include: { tools: true },
      });
      
      if (!agent) {
        throw new Error('Agent not found');
      }
      
      // Create execution record
      const execution = await prisma.execution.create({
        data: {
          agentId: agent.id,
          userId: user.id,
          mode: agent.mode,
          input: input.input,
          status: 'PENDING',
        },
      });
      
      // Queue execution
      await agentEngine.execute({
        executionId: execution.id,
        agent,
        input: input.input,
        stream: input.stream,
      });
      
      return execution;
    }),

  // Get public agents
  listPublic: publicProcedure
    .input(z.object({
      cursor: z.string().optional(),
      limit: z.number().min(1).max(50).default(20),
      mode: z.nativeEnum(AgentMode).optional(),
    }).optional())
    .query(async ({ ctx, input }) => {
      const { prisma } = ctx;
      
      const agents = await prisma.agent.findMany({
        where: {
          isPublic: true,
          status: AgentStatus.ACTIVE,
          ...(input?.mode && { mode: input.mode }),
        },
        take: input?.limit ?? 20,
        skip: input?.cursor ? 1 : 0,
        cursor: input?.cursor ? { id: input.cursor } : undefined,
        orderBy: { executionCount: 'desc' },
        include: {
          user: {
            select: {
              id: true,
              username: true,
              displayName: true,
              avatarUrl: true,
            },
          },
          _count: {
            select: { executions: true },
          },
        },
      });
      
      const nextCursor = agents.length === (input?.limit ?? 20)
        ? agents[agents.length - 1].id
        : undefined;
      
      return { agents, nextCursor };
    }),
});
```

### 5.3 Execution Router

```typescript
// apps/api/src/routers/execution.ts

import { z } from 'zod';
import { router, protectedProcedure } from '../trpc';
import { ExecutionStatus } from '@flowagent/database';
import { observable } from '@trpc/server/observable';

export const executionRouter = router({
  // List executions
  list: protectedProcedure
    .input(z.object({
      cursor: z.string().optional(),
      limit: z.number().min(1).max(100).default(20),
      agentId: z.string().optional(),
      status: z.nativeEnum(ExecutionStatus).optional(),
    }).optional())
    .query(async ({ ctx, input }) => {
      const { prisma, user } = ctx;
      
      const executions = await prisma.execution.findMany({
        where: {
          userId: user.id,
          ...(input?.agentId && { agentId: input.agentId }),
          ...(input?.status && { status: input.status }),
        },
        take: input?.limit ?? 20,
        skip: input?.cursor ? 1 : 0,
        cursor: input?.cursor ? { id: input.cursor } : undefined,
        orderBy: { createdAt: 'desc' },
        include: {
          agent: {
            select: {
              id: true,
              name: true,
              icon: true,
            },
          },
        },
      });
      
      const nextCursor = executions.length === (input?.limit ?? 20)
        ? executions[executions.length - 1].id
        : undefined;
      
      return { executions, nextCursor };
    }),

  // Get execution details
  get: protectedProcedure
    .input(z.object({ id: z.string() }))
    .query(async ({ ctx, input }) => {
      const { prisma, user } = ctx;
      
      const execution = await prisma.execution.findFirst({
        where: {
          id: input.id,
          userId: user.id,
        },
        include: {
          agent: true,
          steps: {
            orderBy: { stepNumber: 'asc' },
          },
        },
      });
      
      if (!execution) {
        throw new Error('Execution not found');
      }
      
      return execution;
    }),

  // Stream execution updates
  onUpdate: protectedProcedure
    .input(z.object({ executionId: z.string() }))
    .subscription(({ ctx, input }) => {
      const { executionEvents } = ctx;
      
      return observable<{ status: ExecutionStatus; output?: any; error?: string }>((emit) => {
        const unsubscribe = executionEvents.subscribe(
          input.executionId,
          (data) => {
            emit.next(data);
            
            if (['COMPLETED', 'FAILED', 'CANCELLED'].includes(data.status)) {
              emit.complete();
            }
          }
        );
        
        return () => {
          unsubscribe();
        };
      });
    }),

  // Cancel execution
  cancel: protectedProcedure
    .input(z.object({ id: z.string() }))
    .mutation(async ({ ctx, input }) => {
      const { prisma, user, agentEngine } = ctx;
      
      const execution = await prisma.execution.findFirst({
        where: {
          id: input.id,
          userId: user.id,
          status: { in: ['PENDING', 'RUNNING', 'PAUSED'] },
        },
      });
      
      if (!execution) {
        throw new Error('Execution not found or cannot be cancelled');
      }
      
      await agentEngine.cancel(execution.id);
      
      await prisma.execution.update({
        where: { id: input.id },
        data: { status: 'CANCELLED' },
      });
      
      return { success: true };
    }),

  // Retry failed execution
  retry: protectedProcedure
    .input(z.object({ id: z.string() }))
    .mutation(async ({ ctx, input }) => {
      const { prisma, user, agentEngine } = ctx;
      
      const execution = await prisma.execution.findFirst({
        where: {
          id: input.id,
          userId: user.id,
          status: 'FAILED',
        },
        include: { agent: { include: { tools: true } } },
      });
      
      if (!execution) {
        throw new Error('Execution not found or cannot be retried');
      }
      
      // Create new execution
      const newExecution = await prisma.execution.create({
        data: {
          agentId: execution.agentId,
          userId: user.id,
          mode: execution.mode,
          input: execution.input,
          status: 'PENDING',
        },
      });
      
      await agentEngine.execute({
        executionId: newExecution.id,
        agent: execution.agent,
        input: execution.input,
      });
      
      return newExecution;
    }),
});
```

### 5.4 Marketplace Router

```typescript
// apps/api/src/routers/marketplace.ts

import { z } from 'zod';
import { router, protectedProcedure, publicProcedure } from '../trpc';
import { TemplateCategory, TemplateStatus } from '@flowagent/database';

export const marketplaceRouter = router({
  // List templates
  list: publicProcedure
    .input(z.object({
      cursor: z.string().optional(),
      limit: z.number().min(1).max(50).default(20),
      category: z.nativeEnum(TemplateCategory).optional(),
      minPrice: z.number().optional(),
      maxPrice: z.number().optional(),
      sortBy: z.enum(['popular', 'newest', 'rating', 'price_asc', 'price_desc']).default('popular'),
      search: z.string().optional(),
    }).optional())
    .query(async ({ ctx, input }) => {
      const { prisma } = ctx;
      
      const orderBy = {
        popular: { purchaseCount: 'desc' as const },
        newest: { createdAt: 'desc' as const },
        rating: { rating: 'desc' as const },
        price_asc: { price: 'asc' as const },
        price_desc: { price: 'desc' as const },
      }[input?.sortBy ?? 'popular'];
      
      const templates = await prisma.template.findMany({
        where: {
          status: TemplateStatus.APPROVED,
          ...(input?.category && { category: input.category }),
          ...(input?.minPrice !== undefined && { price: { gte: input.minPrice } }),
          ...(input?.maxPrice !== undefined && { price: { lte: input.maxPrice } }),
          ...(input?.search && {
            OR: [
              { name: { contains: input.search, mode: 'insensitive' } },
              { description: { contains: input.search, mode: 'insensitive' } },
              { tags: { has: input.search } },
            ],
          }),
        },
        take: input?.limit ?? 20,
        skip: input?.cursor ? 1 : 0,
        cursor: input?.cursor ? { id: input.cursor } : undefined,
        orderBy,
        include: {
          creator: {
            select: {
              id: true,
              username: true,
              displayName: true,
              avatarUrl: true,
            },
          },
          _count: {
            select: { reviews: true },
          },
        },
      });
      
      const nextCursor = templates.length === (input?.limit ?? 20)
        ? templates[templates.length - 1].id
        : undefined;
      
      return { templates, nextCursor };
    }),

  // Get template details
  get: publicProcedure
    .input(z.object({ id: z.string() }))
    .query(async ({ ctx, input }) => {
      const { prisma, user } = ctx;
      
      const template = await prisma.template.findFirst({
        where: {
          id: input.id,
          status: TemplateStatus.APPROVED,
        },
        include: {
          creator: {
            select: {
              id: true,
              username: true,
              displayName: true,
              avatarUrl: true,
            },
          },
          reviews: {
            take: 10,
            orderBy: { createdAt: 'desc' },
            include: {
              user: {
                select: {
                  id: true,
                  username: true,
                  displayName: true,
                  avatarUrl: true,
                },
              },
            },
          },
          _count: {
            select: { reviews: true, purchases: true },
          },
        },
      });
      
      if (!template) {
        throw new Error('Template not found');
      }
      
      // Check if user has purchased
      let hasPurchased = false;
      if (user) {
        const purchase = await prisma.purchase.findUnique({
          where: {
            buyerId_templateId: {
              buyerId: user.id,
              templateId: template.id,
            },
          },
        });
        hasPurchased = !!purchase;
      }
      
      // Increment view count
      await prisma.template.update({
        where: { id: input.id },
        data: { viewCount: { increment: 1 } },
      });
      
      return { ...template, hasPurchased };
    }),

  // Purchase template
  purchase: protectedProcedure
    .input(z.object({ templateId: z.string() }))
    .mutation(async ({ ctx, input }) => {
      const { prisma, user, stripe } = ctx;
      
      const template = await prisma.template.findFirst({
        where: {
          id: input.templateId,
          status: TemplateStatus.APPROVED,
        },
        include: {
          creator: true,
        },
      });
      
      if (!template) {
        throw new Error('Template not found');
      }
      
      if (template.creatorId === user.id) {
        throw new Error('Cannot purchase your own template');
      }
      
      // Check if already purchased
      const existingPurchase = await prisma.purchase.findUnique({
        where: {
          buyerId_templateId: {
            buyerId: user.id,
            templateId: template.id,
          },
        },
      });
      
      if (existingPurchase) {
        throw new Error('Already purchased this template');
      }
      
      // Free template
      if (template.price === 0) {
        const purchase = await prisma.purchase.create({
          data: {
            buyerId: user.id,
            templateId: template.id,
            price: 0,
            platformFee: 0,
            creatorPayout: 0,
            status: 'COMPLETED',
          },
        });
        
        // Create agent from template
        const agent = await prisma.agent.create({
          data: {
            userId: user.id,
            name: template.name,
            description: template.description,
            mode: template.agentConfig.mode,
            model: template.agentConfig.model,
            temperature: template.agentConfig.temperature,
            systemPrompt: template.agentConfig.systemPrompt,
            workflow: template.workflow,
            templateId: template.id,
          },
        });
        
        return { purchase, agent };
      }
      
      // Paid template - create Stripe checkout
      const platformFee = template.price * 0.3; // 30%
      const creatorPayout = template.price * 0.7; // 70%
      
      const session = await stripe.checkout.sessions.create({
        mode: 'payment',
        customer: user.stripeCustomerId || undefined,
        line_items: [
          {
            price_data: {
              currency: template.currency.toLowerCase(),
              product_data: {
                name: template.name,
                description: template.shortDescription || undefined,
              },
              unit_amount: Math.round(template.price * 100), // Convert to cents
            },
            quantity: 1,
          },
        ],
        payment_intent_data: {
          transfer_data: {
            destination: template.creator.stripeConnectAccountId,
            amount: Math.round(creatorPayout * 100),
          },
          metadata: {
            templateId: template.id,
            buyerId: user.id,
          },
        },
        success_url: `${process.env.APP_URL}/marketplace/success?session_id={CHECKOUT_SESSION_ID}`,
        cancel_url: `${process.env.APP_URL}/marketplace/template/${template.id}`,
      });
      
      // Create pending purchase
      const purchase = await prisma.purchase.create({
        data: {
          buyerId: user.id,
          templateId: template.id,
          price: template.price,
          platformFee,
          creatorPayout,
          currency: template.currency,
          stripePaymentIntentId: session.payment_intent as string,
          status: 'PENDING',
        },
      });
      
      return { purchase, checkoutUrl: session.url };
    }),

  // Create review
  createReview: protectedProcedure
    .input(z.object({
      templateId: z.string(),
      rating: z.number().min(1).max(5),
      title: z.string().max(100).optional(),
      content: z.string().max(2000),
    }))
    .mutation(async ({ ctx, input }) => {
      const { prisma, user } = ctx;
      
      // Verify purchase
      const purchase = await prisma.purchase.findUnique({
        where: {
          buyerId_templateId: {
            buyerId: user.id,
            templateId: input.templateId,
          },
        },
      });
      
      const review = await prisma.review.create({
        data: {
          templateId: input.templateId,
          userId: user.id,
          rating: input.rating,
          title: input.title,
          content: input.content,
          isVerified: !!purchase,
          purchaseId: purchase?.id,
        },
      });
      
      // Update template rating
      const reviews = await prisma.review.findMany({
        where: { templateId: input.templateId },
        select: { rating: true },
      });
      
      const avgRating = reviews.reduce((sum, r) => sum + r.rating, 0) / reviews.length;
      
      await prisma.template.update({
        where: { id: input.templateId },
        data: {
          rating: avgRating,
          reviewCount: reviews.length,
        },
      });
      
      return review;
    }),
});
```

### 5.5 Additional Routers

```typescript
// apps/api/src/routers/billing.ts

export const billingRouter = router({
  getSubscription: protectedProcedure.query(async ({ ctx }) => {
    const { prisma, user } = ctx;
    
    const subscription = await prisma.subscription.findUnique({
      where: { userId: user.id },
    });
    
    return subscription;
  }),

  createCheckout: protectedProcedure
    .input(z.object({ tier: z.enum(['PRO', 'ENTERPRISE']) }))
    .mutation(async ({ ctx, input }) => {
      // Stripe checkout session creation
    }),

  getUsage: protectedProcedure.query(async ({ ctx }) => {
    // Get current month usage stats
  }),

  getInvoices: protectedProcedure.query(async ({ ctx }) => {
    // List Stripe invoices
  }),
});

// apps/api/src/routers/api-key.ts

export const apiKeyRouter = router({
  list: protectedProcedure.query(async ({ ctx }) => {
    // List API keys
  }),

  create: protectedProcedure
    .input(z.object({ name: z.string(), permissions: z.array(z.string()) }))
    .mutation(async ({ ctx, input }) => {
      // Create new API key
    }),

  revoke: protectedProcedure
    .input(z.object({ id: z.string() }))
    .mutation(async ({ ctx, input }) => {
      // Revoke API key
    }),
});
```

### 5.6 REST API Endpoints

For external integrations, provide REST endpoints:

```typescript
// apps/api/src/rest/routes.ts

import { Router } from 'express';
import { authenticateApiKey } from '../middleware/auth';

const router = Router();

// Agent execution via API key
router.post('/v1/agents/:id/execute', authenticateApiKey, async (req, res) => {
  // Execute agent
});

// Get execution status
router.get('/v1/executions/:id', authenticateApiKey, async (req, res) => {
  // Get execution
});

// Webhook for Stripe
router.post('/webhooks/stripe', async (req, res) => {
  // Handle Stripe events
});

export default router;
```

---

## 6. Agent Orchestration

### 6.1 Architecture Overview

```mermaid
graph TB
    subgraph "User Request"
        PROMPT[User Prompt]
    end

    subgraph "Orchestration Layer"
        DECOMP[Task Decomposer]
        PLANNER[Execution Planner]
        COORD[Agent Coordinator]
    end

    subgraph "Agent Pool"
        A1[Research Agent]
        A2[Code Agent]
        A3[Analysis Agent]
        A4[Writing Agent]
    end

    subgraph "Tool Registry"
        T1[Web Search]
        T2[Browser]
        T3[Code Exec]
        T4[API Call]
        T5[File Ops]
    end

    subgraph "Memory Store"
        M1[Short-term Memory]
        M2[Long-term Memory]
        M3[Knowledge Base]
    end

    PROMPT --> DECOMP
    DECOMP --> PLANNER
    PLANNER --> COORD
    
    COORD --> A1
    COORD --> A2
    COORD --> A3
    COORD --> A4
    
    A1 --> T1
    A1 --> T2
    A2 --> T3
    A3 --> T4
    A4 --> T5
    
    A1 --> M1
    A2 --> M2
    A3 --> M3
    
    A1 --> COORD
    A2 --> COORD
    A3 --> COORD
    A4 --> COORD
    
    COORD --> RESULT[Final Result]
```

### 6.2 Agent Modes Implementation

```python
# apps/agent-engine/src/modes/auto.py

from typing import List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from ..models.llm_router import LLMRouter
from ..tools.registry import ToolRegistry
from ..memory.manager import MemoryManager

class AutoModeExecutor:
    """
    Fully autonomous agent mode.
    Agent decomposes tasks, plans execution, and completes autonomously.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.llm = LLMRouter(config['model'], config.get('temperature', 0.7))
        self.tools = ToolRegistry()
        self.memory = MemoryManager()
        self.max_iterations = config.get('max_iterations', 10)
        
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task in auto mode."""
        
        # Step 1: Task decomposition
        subtasks = await self._decompose_task(input_data['prompt'])
        
        # Step 2: Build execution graph
        graph = self._build_execution_graph(subtasks)
        
        # Step 3: Execute with checkpointing
        result = await graph.ainvoke({
            'input': input_data,
            'subtasks': subtasks,
            'current_task': 0,
            'results': [],
            'memory': await self.memory.load(input_data.get('conversation_id')),
        })
        
        # Step 4: Save memory
        await self.memory.save(
            input_data.get('conversation_id'),
            result['memory']
        )
        
        return {
            'output': result['final_output'],
            'steps': result['results'],
            'tokens_used': result.get('tokens', {}),
        }
    
    async def _decompose_task(self, prompt: str) -> List[Dict[str, Any]]:
        """Decompose high-level task into subtasks."""
        
        decomposition_prompt = f"""
        Analyze the following task and break it down into specific, actionable subtasks.
        
        Task: {prompt}
        
        Provide your response as a JSON array of subtasks, where each subtask has:
        - id: unique identifier
        - description: detailed description
        - type: one of [research, analysis, code, writing, verification]
        - dependencies: list of subtask IDs that must complete first
        - estimated_cost: low/medium/high
        
        Example:
        [
          {{
            "id": "1",
            "description": "Search for recent data on X",
            "type": "research",
            "dependencies": [],
            "estimated_cost": "low"
          }}
        ]
        """
        
        response = await self.llm.ainvoke(decomposition_prompt)
        return self._parse_subtasks(response.content)
    
    def _build_execution_graph(self, subtasks: List[Dict[str, Any]]) -> StateGraph:
        """Build LangGraph execution graph from subtasks."""
        
        # Create state graph
        workflow = StateGraph(dict)
        
        # Add nodes for each subtask
        for subtask in subtasks:
            workflow.add_node(
                subtask['id'],
                lambda state, st=subtask: self._execute_subtask(state, st)
            )
        
        # Add edges based on dependencies
        for subtask in subtasks:
            if subtask['dependencies']:
                for dep in subtask['dependencies']:
                    workflow.add_edge(dep, subtask['id'])
            else:
                workflow.set_entry_point(subtask['id'])
        
        # Add final aggregation node
        workflow.add_node('aggregate', self._aggregate_results)
        
        # Connect all terminal nodes to aggregate
        for subtask in subtasks:
            if not any(st['id'] in st.get('dependencies', []) for st in subtasks):
                workflow.add_edge(subtask['id'], 'aggregate')
        
        workflow.add_edge('aggregate', END)
        
        return workflow.compile()
    
    async def _execute_subtask(self, state: Dict[str, Any], subtask: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single subtask."""
        
        # Select appropriate tools
        tools = self.tools.select_for_task(subtask['type'])
        
        # Build prompt with context
        context = self._build_context(state, subtask)
        
        # Execute with tool calling
        response = await self.llm.ainvoke(
            context,
            tools=tools,
        )
        
        # Handle tool calls
        if response.tool_calls:
            tool_results = await self._execute_tool_calls(response.tool_calls)
            response = await self.llm.ainvoke(
                context + [response] + tool_results
            )
        
        # Update state
        state['results'].append({
            'subtask_id': subtask['id'],
            'type': subtask['type'],
            'output': response.content,
            'tokens': response.usage_metadata if hasattr(response, 'usage_metadata') else {},
        })
        
        return state
    
    async def _aggregate_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate all subtask results into final output."""
        
        aggregation_prompt = f"""
        Synthesize the following subtask results into a coherent final response.
        
        Original task: {state['input']['prompt']}
        
        Subtask results:
        {state['results']}
        
        Provide a comprehensive, well-structured response that addresses the original task.
        """
        
        final_response = await self.llm.ainvoke(aggregation_prompt)
        
        state['final_output'] = final_response.content
        state['tokens'] = self._calculate_total_tokens(state['results'])
        
        return state
```

### 6.3 Tool Registry

```python
# apps/agent-engine/src/tools/registry.py

from typing import List, Dict, Any, Callable
from enum import Enum
import importlib

class ToolType(Enum):
    WEB_SEARCH = "web_search"
    BROWSER = "browser"
    CODE_EXECUTION = "code_execution"
    API_CALL = "api_call"
    FILE_OPERATION = "file_operation"
    DATABASE = "database"
    CUSTOM = "custom"

class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        type: ToolType,
        handler: Callable,
        input_schema: Dict[str, Any],
        cost_tier: str = "low",
        timeout: int = 30,
    ):
        self.name = name
        self.description = description
        self.type = type
        self.handler = handler
        self.input_schema = input_schema
        self.cost_tier = cost_tier
        self.timeout = timeout
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute tool with given parameters."""
        return await self.handler(**kwargs)

class ToolRegistry:
    """Central registry for all available tools."""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._load_builtin_tools()
    
    def _load_builtin_tools(self):
        """Load all built-in tools."""
        
        # Web Search
        self.register(Tool(
            name="web_search",
            description="Search the web for information",
            type=ToolType.WEB_SEARCH,
            handler=self._web_search,
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "num_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        ))
        
        # Browser
        self.register(Tool(
            name="browser",
            description="Browse a specific webpage",
            type=ToolType.BROWSER,
            handler=self._browser,
            input_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "action": {"type": "string", "enum": ["scrape", "screenshot", "click", "type"]},
                    "selector": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["url", "action"],
            },
        ))
        
        # Code Execution
        self.register(Tool(
            name="code_execution",
            description="Execute code in sandboxed environment",
            type=ToolType.CODE_EXECUTION,
            handler=self._code_execution,
            input_schema={
                "type": "object",
                "properties": {
                    "language": {"type": "string", "enum": ["python", "javascript", "typescript", "bash"]},
                    "code": {"type": "string"},
                    "timeout": {"type": "integer", "default": 30},
                },
                "required": ["language", "code"],
            },
            cost_tier="high",
        ))
        
        # API Call
        self.register(Tool(
            name="api_call",
            description="Make HTTP API requests",
            type=ToolType.API_CALL,
            handler=self._api_call,
            input_schema={
                "type": "object",
                "properties": {
                    "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"]},
                    "url": {"type": "string"},
                    "headers": {"type": "object"},
                    "body": {"type": "object"},
                },
                "required": ["method", "url"],
            },
        ))
        
        # File Operations
        self.register(Tool(
            name="file_operation",
            description="Read, write, or manipulate files",
            type=ToolType.FILE_OPERATION,
            handler=self._file_operation,
            input_schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["read", "write", "delete", "list", "move"]},
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["action", "path"],
            },
        ))
    
    def register(self, tool: Tool):
        """Register a new tool."""
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Tool:
        """Get tool by name."""
        return self._tools.get(name)
    
    def list_all(self) -> List[Tool]:
        """List all registered tools."""
        return list(self._tools.values())
    
    def select_for_task(self, task_type: str) -> List[Tool]:
        """Select appropriate tools for a task type."""
        
        tool_mapping = {
            "research": ["web_search", "browser"],
            "analysis": ["web_search", "api_call", "file_operation"],
            "code": ["code_execution", "file_operation"],
            "writing": ["file_operation", "web_search"],
            "verification": ["web_search", "api_call"],
        }
        
        tool_names = tool_mapping.get(task_type, [])
        return [self._tools[name] for name in tool_names if name in self._tools]
    
    # Tool implementations
    async def _web_search(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """Execute web search using configured provider."""
        from .providers import SearchProvider
        
        provider = SearchProvider()
        results = await provider.search(query, num_results)
        
        return {
            "results": results,
            "query": query,
            "total_results": len(results),
        }
    
    async def _browser(self, url: str, action: str, **kwargs) -> Dict[str, Any]:
        """Execute browser action."""
        from .providers import BrowserProvider
        
        provider = BrowserProvider()
        result = await provider.execute(url, action, **kwargs)
        
        return result
    
    async def _code_execution(self, language: str, code: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute code in sandboxed environment."""
        from ..sandbox.manager import SandboxManager
        
        sandbox = SandboxManager()
        result = await sandbox.execute(language, code, timeout)
        
        return result
    
    async def _api_call(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP API request."""
        import httpx
        
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=method,
                url=url,
                headers=kwargs.get('headers'),
                json=kwargs.get('body'),
            )
            
            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text,
            }
    
    async def _file_operation(self, action: str, path: str, **kwargs) -> Dict[str, Any]:
        """Execute file operation."""
        import aiofiles
        import os
        
        if action == "read":
            async with aiofiles.open(path, 'r') as f:
                content = await f.read()
            return {"content": content, "path": path}
        
        elif action == "write":
            async with aiofiles.open(path, 'w') as f:
                await f.write(kwargs.get('content', ''))
            return {"success": True, "path": path}
        
        elif action == "delete":
            os.remove(path)
            return {"success": True, "path": path}
        
        elif action == "list":
            files = os.listdir(path)
            return {"files": files, "path": path}
        
        elif action == "move":
            os.rename(path, kwargs.get('destination'))
            return {"success": True, "from": path, "to": kwargs.get('destination')}
```

### 6.4 Memory Management

```python
# apps/agent-engine/src/memory/manager.py

from typing import Dict, Any, Optional, List
import json
from datetime import datetime, timedelta

class MemoryManager:
    """
    Multi-tier memory management for agents.
    - Short-term: Current conversation context
    - Long-term: User preferences and history
    - Knowledge: Vector embeddings for semantic search
    """
    
    def __init__(self, redis_client, vector_store, db_client):
        self.redis = redis_client
        self.vector_store = vector_store
        self.db = db_client
    
    async def load(self, conversation_id: Optional[str]) -> Dict[str, Any]:
        """Load memory for a conversation."""
        
        if not conversation_id:
            return self._empty_memory()
        
        # Try to load from Redis (short-term)
        memory_key = f"memory:{conversation_id}"
        cached = await self.redis.get(memory_key)
        
        if cached:
            return json.loads(cached)
        
        # Load from database (long-term)
        conversation = await self.db.conversation.find_unique(
            where={"id": conversation_id},
            include={"messages": {"order_by": {"created_at": "desc"}, "take": 50}},
        )
        
        if not conversation:
            return self._empty_memory()
        
        # Build memory structure
        memory = {
            "conversation_id": conversation_id,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.created_at.isoformat(),
                }
                for msg in reversed(conversation.messages)
            ],
            "summary": await self._generate_summary(conversation.messages),
            "entities": await self._extract_entities(conversation.messages),
        }
        
        # Cache in Redis
        await self.redis.setex(
            memory_key,
            timedelta(hours=1),
            json.dumps(memory)
        )
        
        return memory
    
    async def save(self, conversation_id: str, memory: Dict[str, Any]):
        """Save memory for a conversation."""
        
        # Update Redis cache
        memory_key = f"memory:{conversation_id}"
        await self.redis.setex(
            memory_key,
            timedelta(hours=1),
            json.dumps(memory)
        )
        
        # Update vector store with new information
        await self._update_knowledge_base(conversation_id, memory)
    
    async def search_knowledge(self, query: str, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search knowledge base for relevant information."""
        
        # Generate embedding for query
        embedding = await self._generate_embedding(query)
        
        # Search vector store
        results = await self.vector_store.search(
            embedding=embedding,
            filter={"user_id": user_id},
            limit=limit,
        )
        
        return results
    
    async def add_to_knowledge(self, user_id: str, content: str, metadata: Dict[str, Any]):
        """Add information to user's knowledge base."""
        
        embedding = await self._generate_embedding(content)
        
        await self.vector_store.add(
            embedding=embedding,
            content=content,
            metadata={
                "user_id": user_id,
                **metadata,
            }
        )
    
    def _empty_memory(self) -> Dict[str, Any]:
        """Create empty memory structure."""
        return {
            "conversation_id": None,
            "messages": [],
            "summary": None,
            "entities": {},
        }
    
    async def _generate_summary(self, messages: List[Any]) -> str:
        """Generate summary of conversation."""
        # Use LLM to generate summary
        pass
    
    async def _extract_entities(self, messages: List[Any]) -> Dict[str, Any]:
        """Extract key entities from conversation."""
        # Use NER to extract entities
        pass
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        # Use embedding model
        pass
    
    async def _update_knowledge_base(self, conversation_id: str, memory: Dict[str, Any]):
        """Update knowledge base with new information."""
        # Extract and store new facts
        pass
```

---

## 7. Marketplace Architecture

### 7.1 Payment Flow

```mermaid
sequenceDiagram
    participant Buyer
    participant FlowAgent
    participant Stripe
    participant Creator

    Buyer->>FlowAgent: Purchase Template ($100)
    FlowAgent->>Stripe: Create Checkout Session
    Note over Stripe: Platform fee: $30 (30%)<br/>Creator payout: $70 (70%)
    Stripe-->>FlowAgent: Session URL
    FlowAgent-->>Buyer: Redirect to Checkout
    
    Buyer->>Stripe: Complete Payment
    Stripe->>Stripe: Process Payment
    Stripe->>Creator: Transfer $70 to Connect Account
    Stripe-->>FlowAgent: Payment Success Webhook
    
    FlowAgent->>FlowAgent: Create Purchase Record
    FlowAgent->>FlowAgent: Create Agent from Template
    FlowAgent-->>Buyer: Purchase Complete + Agent Access
    
    alt Refund Request
        Buyer->>FlowAgent: Request Refund
        FlowAgent->>Stripe: Create Refund
        Stripe->>Buyer: Refund $100
        Stripe->>Creator: Reverse $70 Transfer
        Stripe-->>FlowAgent: Refund Webhook
        FlowAgent->>FlowAgent: Update Purchase Status
    end
```

### 7.2 Stripe Connect Integration

```typescript
// apps/api/src/services/stripe-connect.ts

import Stripe from 'stripe';

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!, {
  apiVersion: '2023-10-16',
});

export class StripeConnectService {
  /**
   * Create a Stripe Connect account for a creator
   */
  async createConnectAccount(userId: string, email: string) {
    const account = await stripe.accounts.create({
      type: 'standard',
      email,
      metadata: {
        userId,
      },
      capabilities: {
        transfers: { requested: true },
      },
    });

    // Generate onboarding link
    const accountLink = await stripe.accountLinks.create({
      account: account.id,
      refresh_url: `${process.env.APP_URL}/marketplace/onboarding?refresh=true`,
      return_url: `${process.env.APP_URL}/marketplace/onboarding?success=true`,
      type: 'account_onboarding',
    });

    return {
      accountId: account.id,
      onboardingUrl: accountLink.url,
    };
  }

  /**
   * Create a product and price for a template
   */
  async createTemplateProduct(template: {
    id: string;
    name: string;
    description: string;
    price: number;
    creatorStripeAccountId: string;
  }) {
    // Create product
    const product = await stripe.products.create({
      name: template.name,
      description: template.description,
      metadata: {
        templateId: template.id,
      },
    });

    // Create price
    const price = await stripe.prices.create({
      product: product.id,
      unit_amount: Math.round(template.price * 100), // Convert to cents
      currency: 'usd',
      transfer_data: {
        destination: template.creatorStripeAccountId,
      },
      metadata: {
        templateId: template.id,
      },
    });

    return {
      productId: product.id,
      priceId: price.id,
    };
  }

  /**
   * Create checkout session for template purchase
   */
  async createCheckoutSession(params: {
    template: {
      id: string;
      name: string;
      description?: string;
      price: number;
      creatorStripeAccountId: string;
    };
    buyer: {
      id: string;
      email: string;
      stripeCustomerId?: string;
    };
    successUrl: string;
    cancelUrl: string;
  }) {
    const { template, buyer, successUrl, cancelUrl } = params;

    // Calculate fees
    const totalAmount = Math.round(template.price * 100);
    const platformFee = Math.round(template.price * 0.3 * 100); // 30%
    const creatorAmount = totalAmount - platformFee; // 70%

    // Get or create customer
    let customerId = buyer.stripeCustomerId;
    if (!customerId) {
      const customer = await stripe.customers.create({
        email: buyer.email,
        metadata: { userId: buyer.id },
      });
      customerId = customer.id;
    }

    // Create checkout session
    const session = await stripe.checkout.sessions.create({
      customer: customerId,
      mode: 'payment',
      line_items: [
        {
          price_data: {
            currency: 'usd',
            product_data: {
              name: template.name,
              description: template.description,
            },
            unit_amount: totalAmount,
          },
          quantity: 1,
        },
      ],
      payment_intent_data: {
        application_fee_amount: platformFee,
        transfer_data: {
          destination: template.creatorStripeAccountId,
          amount: creatorAmount,
        },
        metadata: {
          templateId: template.id,
          buyerId: buyer.id,
          type: 'template_purchase',
        },
      },
      success_url: successUrl,
      cancel_url: cancelUrl,
      metadata: {
        templateId: template.id,
        buyerId: buyer.id,
        type: 'template_purchase',
      },
    });

    return session;
  }

  /**
   * Handle webhook events
   */
  async handleWebhookEvent(event: Stripe.Event) {
    switch (event.type) {
      case 'checkout.session.completed': {
        const session = event.data.object as Stripe.Checkout.Session;
        await this.handleCheckoutCompleted(session);
        break;
      }

      case 'charge.refunded': {
        const charge = event.data.object as Stripe.Charge;
        await this.handleRefund(charge);
        break;
      }

      case 'account.updated': {
        const account = event.data.object as Stripe.Account;
        await this.handleAccountUpdate(account);
        break;
      }
    }
  }

  private async handleCheckoutCompleted(session: Stripe.Checkout.Session) {
    const { templateId, buyerId } = session.metadata!;

    // Update purchase status
    await prisma.purchase.updateMany({
      where: {
        buyerId,
        templateId,
        status: 'PENDING',
      },
      data: {
        status: 'COMPLETED',
        stripePaymentIntentId: session.payment_intent as string,
      },
    });

    // Create agent from template
    const template = await prisma.template.findUnique({
      where: { id: templateId },
    });

    if (template) {
      await prisma.agent.create({
        data: {
          userId: buyerId,
          name: template.name,
          description: template.description,
          mode: template.agentConfig.mode,
          model: template.agentConfig.model,
          temperature: template.agentConfig.temperature,
          systemPrompt: template.agentConfig.systemPrompt,
          workflow: template.workflow,
          templateId: template.id,
        },
      });
    }

    // Increment purchase count
    await prisma.template.update({
      where: { id: templateId },
      data: { purchaseCount: { increment: 1 } },
    });
  }

  private async handleRefund(charge: Stripe.Charge) {
    const { templateId, buyerId } = charge.metadata;

    // Update purchase status
    await prisma.purchase.updateMany({
      where: {
        buyerId,
        templateId,
      },
      data: {
        status: 'REFUNDED',
        refundedAt: new Date(),
        refundAmount: charge.amount_refunded / 100,
      },
    });
  }

  private async handleAccountUpdate(account: Stripe.Account) {
    // Update creator's onboarding status
    const user = await prisma.user.findFirst({
      where: { stripeConnectAccountId: account.id },
    });

    if (user) {
      await prisma.user.update({
        where: { id: user.id },
        data: {
          stripeConnectOnboarded: account.details_submitted,
          stripeConnectChargesEnabled: account.charges_enabled,
          stripeConnectPayoutsEnabled: account.payouts_enabled,
        },
      });
    }
  }
}
```

### 7.3 Creator Onboarding Flow

```typescript
// apps/api/src/services/creator-onboarding.ts

export class CreatorOnboardingService {
  /**
   * Start creator onboarding process
   */
  async startOnboarding(userId: string) {
    const user = await prisma.user.findUnique({
      where: { id: userId },
    });

    if (!user) {
      throw new Error('User not found');
    }

    // Check if already has Connect account
    if (user.stripeConnectAccountId) {
      // Generate new onboarding link
      const accountLink = await stripe.accountLinks.create({
        account: user.stripeConnectAccountId,
        refresh_url: `${process.env.APP_URL}/marketplace/onboarding?refresh=true`,
        return_url: `${process.env.APP_URL}/marketplace/onboarding?success=true`,
        type: 'account_onboarding',
      });

      return { onboardingUrl: accountLink.url };
    }

    // Create new Connect account
    const { accountId, onboardingUrl } = await stripeConnectService.createConnectAccount(
      userId,
      user.email
    );

    // Save account ID
    await prisma.user.update({
      where: { id: userId },
      data: { stripeConnectAccountId: accountId },
    });

    return { onboardingUrl };
  }

  /**
   * Check if creator can publish templates
   */
  async canPublish(userId: string): Promise<boolean> {
    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: {
        stripeConnectOnboarded: true,
        stripeConnectChargesEnabled: true,
        stripeConnectPayoutsEnabled: true,
      },
    });

    if (!user) return false;

    return (
      user.stripeConnectOnboarded &&
      user.stripeConnectChargesEnabled &&
      user.stripeConnectPayoutsEnabled
    );
  }

  /**
   * Handle template submission
   */
  async submitTemplate(templateId: string, userId: string) {
    // Verify creator can publish
    const canPublish = await this.canPublish(userId);

    if (!canPublish) {
      throw new Error('Please complete Stripe onboarding before publishing templates');
    }

    // Update template status
    await prisma.template.update({
      where: { id: templateId },
      data: { status: 'PENDING_REVIEW' },
    });

    // Create Stripe product
    const template = await prisma.template.findUnique({
      where: { id: templateId },
      include: { creator: true },
    });

    if (template && template.price > 0) {
      const { productId, priceId } = await stripeConnectService.createTemplateProduct({
        id: template.id,
        name: template.name,
        description: template.description,
        price: template.price,
        creatorStripeAccountId: template.creator.stripeConnectAccountId!,
      });

      await prisma.template.update({
        where: { id: templateId },
        data: {
          stripeProductId: productId,
          stripePriceId: priceId,
        },
      });
    }

    // Notify admin for review
    await notificationService.notifyAdmin({
      type: 'TEMPLATE_SUBMITTED',
      templateId,
      creatorId: userId,
    });
  }
}
```

---

## 8. Security Model

### 8.1 API Key Security

```typescript
// apps/api/src/middleware/api-key-auth.ts

import { Request, Response, NextFunction } from 'express';
import crypto from 'crypto';

export async function authenticateApiKey(
  req: Request,
  res: Response,
  next: NextFunction
) {
  const apiKey = req.headers['x-api-key'] as string;

  if (!apiKey) {
    return res.status(401).json({ error: 'API key required' });
  }

  // Hash the provided key for lookup
  const keyHash = crypto.createHash('sha256').update(apiKey).digest('hex');

  // Find API key in database
  const keyRecord = await prisma.apiKey.findUnique({
    where: { keyHash },
    include: { user: true },
  });

  if (!keyRecord || !keyRecord.isActive) {
    return res.status(401).json({ error: 'Invalid API key' });
  }

  // Check expiration
  if (keyRecord.expiresAt && keyRecord.expiresAt < new Date()) {
    return res.status(401).json({ error: 'API key expired' });
  }

  // Check rate limit
  const rateLimitKey = `ratelimit:${keyRecord.id}:${Math.floor(Date.now() / 60000)}`;
  const currentRequests = await redis.incr(rateLimitKey);

  if (currentRequests === 1) {
    await redis.expire(rateLimitKey, 60);
  }

  if (currentRequests > keyRecord.rateLimit) {
    return res.status(429).json({
      error: 'Rate limit exceeded',
      limit: keyRecord.rateLimit,
      window: '1 minute',
    });
  }

  // Update usage
  await prisma.apiKey.update({
    where: { id: keyRecord.id },
    data: {
      lastUsedAt: new Date(),
      usageCount: { increment: 1 },
    },
  });

  // Attach user to request
  req.user = keyRecord.user;
  req.apiKey = keyRecord;

  next();
}
```

### 8.2 Code Sandbox Security

```python
# apps/agent-engine/src/sandbox/manager.py

import docker
import tempfile
import os
import shutil
from typing import Dict, Any
import asyncio

class SandboxManager:
    """
    Secure code execution environment using Docker + gVisor.
    """
    
    def __init__(self):
        self.docker = docker.from_env()
        self.network = 'sandbox-network'
        self._ensure_network()
    
    def _ensure_network(self):
        """Create isolated network for sandboxes."""
        try:
            self.docker.networks.get(self.network)
        except docker.errors.NotFound:
            self.docker.networks.create(
                self.network,
                driver='bridge',
                internal=True,  # No external access
            )
    
    async def execute(
        self,
        language: str,
        code: str,
        timeout: int = 30,
        memory_limit: str = '512m',
        cpu_limit: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Execute code in sandboxed environment.
        """
        
        # Create temporary directory for code
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Write code to file
            filename = self._get_filename(language)
            code_path = os.path.join(temp_dir, filename)
            
            with open(code_path, 'w') as f:
                f.write(code)
            
            # Select Docker image
            image = self._get_image(language)
            
            # Run container with strict limits
            container = self.docker.containers.run(
                image,
                command=self._get_command(language, filename),
                volumes={
                    temp_dir: {'bind': '/code', 'mode': 'ro'},
                },
                network=self.network,
                mem_limit=memory_limit,
                cpu_quota=int(cpu_limit * 100000),
                cpu_period=100000,
                pids_limit=50,
                security_opt=['no-new-privileges:true'],
                cap_drop=['ALL'],
                read_only=True,
                detach=True,
            )
            
            # Wait for completion with timeout
            try:
                result = container.wait(timeout=timeout)
                logs = container.logs().decode('utf-8')
                
                return {
                    'success': result['StatusCode'] == 0,
                    'output': logs,
                    'exit_code': result['StatusCode'],
                    'execution_time': timeout,  # Actual time tracking
                }
                
            except docker.errors.ReadTimeout:
                container.kill()
                return {
                    'success': False,
                    'output': 'Execution timed out',
                    'exit_code': -1,
                    'error': 'TIMEOUT',
                }
            
            finally:
                container.remove(force=True)
        
        finally:
            shutil.rmtree(temp_dir)
    
    def _get_filename(self, language: str) -> str:
        """Get appropriate filename for language."""
        extensions = {
            'python': 'script.py',
            'javascript': 'script.js',
            'typescript': 'script.ts',
            'bash': 'script.sh',
        }
        return extensions.get(language, 'script.txt')
    
    def _get_image(self, language: str) -> str:
        """Get Docker image for language."""
        images = {
            'python': 'python:3.11-slim',
            'javascript': 'node:20-slim',
            'typescript': 'node:20-slim',
            'bash': 'alpine:latest',
        }
        return images.get(language, 'alpine:latest')
    
    def _get_command(self, language: str, filename: str) -> str:
        """Get execution command for language."""
        commands = {
            'python': f'python /code/{filename}',
            'javascript': f'node /code/{filename}',
            'typescript': f'npx ts-node /code/{filename}',
            'bash': f'sh /code/{filename}',
        }
        return commands.get(language, f'cat /code/{filename}')
```

### 8.3 Data Encryption

```typescript
// apps/api/src/utils/encryption.ts

import crypto from 'crypto';

const ALGORITHM = 'aes-256-gcm';
const KEY_LENGTH = 32;
const IV_LENGTH = 16;
const AUTH_TAG_LENGTH = 16;

export class EncryptionService {
  private key: Buffer;

  constructor(masterKey: string) {
    // Derive key from master key
    this.key = crypto.scryptSync(masterKey, 'salt', KEY_LENGTH);
  }

  encrypt(text: string): string {
    const iv = crypto.randomBytes(IV_LENGTH);
    const cipher = crypto.createCipheriv(ALGORITHM, this.key, iv);
    
    let encrypted = cipher.update(text, 'utf8', 'hex');
    encrypted += cipher.final('hex');
    
    const authTag = cipher.getAuthTag();
    
    // Return iv:authTag:encrypted
    return `${iv.toString('hex')}:${authTag.toString('hex')}:${encrypted}`;
  }

  decrypt(encryptedData: string): string {
    const [ivHex, authTagHex, encrypted] = encryptedData.split(':');
    
    const iv = Buffer.from(ivHex, 'hex');
    const authTag = Buffer.from(authTagHex, 'hex');
    
    const decipher = crypto.createDecipheriv(ALGORITHM, this.key, iv);
    decipher.setAuthTag(authTag);
    
    let decrypted = decipher.update(encrypted, 'hex', 'utf8');
    decrypted += decipher.final('utf8');
    
    return decrypted;
  }
}

// Usage for sensitive data
export function encryptApiKey(key: string): string {
  return encryptionService.encrypt(key);
}

export function decryptApiKey(encrypted: string): string {
  return encryptionService.decrypt(encrypted);
}
```

---

## 9. Cost Optimization

### 9.1 Intelligent Model Routing

```python
# apps/agent-engine/src/models/llm_router.py

from typing import Dict, Any, Optional
import os

class LLMRouter:
    """
    Routes requests to appropriate LLM based on task complexity and cost.
    """
    
    MODELS = {
        'gpt-4o': {
            'provider': 'openai',
            'input_cost': 0.005,  # per 1k tokens
            'output_cost': 0.015,
            'capabilities': ['complex_reasoning', 'code', 'creative'],
        },
        'gpt-4o-mini': {
            'provider': 'openai',
            'input_cost': 0.00015,
            'output_cost': 0.0006,
            'capabilities': ['general', 'simple_tasks'],
        },
        'claude-3-opus': {
            'provider': 'anthropic',
            'input_cost': 0.015,
            'output_cost': 0.075,
            'capabilities': ['complex_reasoning', 'analysis', 'long_context'],
        },
        'claude-3-sonnet': {
            'provider': 'anthropic',
            'input_cost': 0.003,
            'output_cost': 0.015,
            'capabilities': ['general', 'code', 'analysis'],
        },
        'llama-3-70b': {
            'provider': 'local',
            'input_cost': 0.0,
            'output_cost': 0.0,
            'capabilities': ['general', 'simple_tasks'],
        },
    }
    
    def __init__(self, default_model: str = 'gpt-4o-mini', budget_tier: str = 'standard'):
        self.default_model = default_model
        self.budget_tier = budget_tier
        self.cache = {}  # Simple in-memory cache
    
    async def route(
        self,
        prompt: str,
        task_type: Optional[str] = None,
        complexity: Optional[str] = None,
        preferred_model: Optional[str] = None,
    ) -> str:
        """
        Select best model for the task.
        """
        
        # Use preferred model if specified and available
        if preferred_model and preferred_model in self.MODELS:
            return preferred_model
        
        # Determine complexity if not provided
        if not complexity:
            complexity = await self._estimate_complexity(prompt, task_type)
        
        # Route based on complexity and budget
        routing_map = {
            'low': {
                'economy': 'llama-3-70b',
                'standard': 'gpt-4o-mini',
                'premium': 'claude-3-sonnet',
            },
            'medium': {
                'economy': 'gpt-4o-mini',
                'standard': 'claude-3-sonnet',
                'premium': 'gpt-4o',
            },
            'high': {
                'economy': 'claude-3-sonnet',
                'standard': 'gpt-4o',
                'premium': 'claude-3-opus',
            },
        }
        
        return routing_map.get(complexity, {}).get(
            self.budget_tier,
            self.default_model
        )
    
    async def _estimate_complexity(self, prompt: str, task_type: Optional[str]) -> str:
        """Estimate task complexity from prompt."""
        
        # Simple heuristics
        complexity_indicators = {
            'high': [
                'analyze', 'compare', 'evaluate', 'synthesize',
                'architect', 'design', 'optimize', 'debug',
            ],
            'medium': [
                'explain', 'summarize', 'convert', 'transform',
                'implement', 'create', 'generate',
            ],
            'low': [
                'what is', 'how to', 'list', 'find',
                'get', 'fetch', 'retrieve',
            ],
        }
        
        prompt_lower = prompt.lower()
        
        for level, indicators in complexity_indicators.items():
            if any(ind in prompt_lower for ind in indicators):
                return level
        
        # Default based on length
        if len(prompt) > 1000:
            return 'medium'
        
        return 'low'
    
    async def invoke(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Invoke LLM with caching and cost tracking.
        """
        
        # Check cache
        cache_key = self._get_cache_key(prompt, model)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Select model
        selected_model = model or await self.route(prompt)
        model_config = self.MODELS[selected_model]
        
        # Invoke appropriate provider
        provider = model_config['provider']
        
        if provider == 'openai':
            response = await self._invoke_openai(selected_model, prompt, **kwargs)
        elif provider == 'anthropic':
            response = await self._invoke_anthropic(selected_model, prompt, **kwargs)
        elif provider == 'local':
            response = await self._invoke_local(selected_model, prompt, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        # Track cost
        if hasattr(response, 'usage_metadata'):
            input_tokens = response.usage_metadata.get('input_tokens', 0)
            output_tokens = response.usage_metadata.get('output_tokens', 0)
            
            cost = (
                input_tokens / 1000 * model_config['input_cost'] +
                output_tokens / 1000 * model_config['output_cost']
            )
            
            # Log cost for tracking
            await self._log_cost(selected_model, cost, input_tokens, output_tokens)
        
        # Cache result
        self.cache[cache_key] = response
        
        return response
    
    def _get_cache_key(self, prompt: str, model: Optional[str]) -> str:
        """Generate cache key for prompt."""
        import hashlib
        key = f"{model or 'default'}:{prompt}"
        return hashlib.md5(key.encode()).hexdigest()
    
    async def _log_cost(self, model: str, cost: float, input_tokens: int, output_tokens: int):
        """Log cost for analytics."""
        # Implementation for cost tracking
        pass
```

### 9.2 Response Caching

```typescript
// apps/api/src/services/cache.ts

import Redis from 'ioredis';
import crypto from 'crypto';

const redis = new Redis(process.env.REDIS_URL);

export class ResponseCache {
  private defaultTTL: number = 3600; // 1 hour

  /**
   * Generate cache key from request
   */
  generateKey(prefix: string, data: any): string {
    const hash = crypto
      .createHash('md5')
      .update(JSON.stringify(data))
      .digest('hex');
    
    return `cache:${prefix}:${hash}`;
  }

  /**
   * Get cached response
   */
  async get<T>(key: string): Promise<T | null> {
    const cached = await redis.get(key);
    
    if (cached) {
      return JSON.parse(cached);
    }
    
    return null;
  }

  /**
   * Set cached response
   */
  async set(key: string, value: any, ttl?: number): Promise<void> {
    await redis.setex(
      key,
      ttl || this.defaultTTL,
      JSON.stringify(value)
    );
  }

  /**
   * Cache LLM response
   */
  async cacheLLMResponse(
    prompt: string,
    model: string,
    response: any,
    ttl: number = 3600
  ): Promise<void> {
    const key = this.generateKey('llm', { prompt, model });
    await this.set(key, response, ttl);
  }

  /**
   * Get cached LLM response
   */
  async getCachedLLMResponse(
    prompt: string,
    model: string
  ): Promise<any | null> {
    const key = this.generateKey('llm', { prompt, model });
    return this.get(key);
  }

  /**
   * Invalidate cache by pattern
   */
  async invalidate(pattern: string): Promise<void> {
    const keys = await redis.keys(`cache:${pattern}:*`);
    
    if (keys.length > 0) {
      await redis.del(...keys);
    }
  }
}

export const cache = new ResponseCache();
```

---

## 10. Scalability Strategy

### 10.1 Horizontal Scaling

```yaml
# infrastructure/kubernetes/base/hpa.yml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: flowagent-api
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: flowagent-api
  minReplicas: 2
  maxReplicas: 100
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Percent
          value: 100
          periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60
```

### 10.2 Database Scaling

```typescript
// packages/database/src/client.ts

import { PrismaClient } from '@prisma/client';

// Connection pooling for serverless
const prismaClientSingleton = () => {
  return new PrismaClient({
    log: process.env.NODE_ENV === 'development' 
      ? ['query', 'error', 'warn'] 
      : ['error'],
    datasources: {
      db: {
        url: process.env.DATABASE_URL,
      },
    },
  });
};

declare global {
  var prisma: undefined | ReturnType<typeof prismaClientSingleton>;
}

export const prisma = globalThis.prisma ?? prismaClientSingleton();

if (process.env.NODE_ENV !== 'production') {
  globalThis.prisma = prisma;
}

// Read replica support for scaling
export function getReadClient() {
  if (process.env.DATABASE_READ_URL) {
    return new PrismaClient({
      datasources: {
        db: {
          url: process.env.DATABASE_READ_URL,
        },
      },
    });
  }
  return prisma;
}
```

### 10.3 Scaling Phases

**Phase 1: 0-100 Users**
- Single API instance
- Shared PostgreSQL instance
- Redis on same server
- Static files from CDN

**Phase 2: 100-10K Users**
- 2-5 API instances with load balancer
- Managed PostgreSQL (RDS/Cloud SQL)
- Managed Redis (ElastiCache/MemoryStore)
- Separate agent engine workers

**Phase 3: 10K-100K Users**
- Kubernetes with HPA
- PostgreSQL read replicas
- Redis cluster
- Queue-based execution (Celery + RabbitMQ)
- Multi-region deployment

**Phase 4: 100K-1M Users**
- Microservices architecture
- Database sharding by user_id
- Global CDN
- Edge computing for low latency
- Dedicated infrastructure per tenant (Enterprise)

---

## 11. Self-Hosting Guide

### 11.1 Docker Compose Setup

```yaml
# infrastructure/docker/docker-compose.yml

version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: flowagent
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: flowagent
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U flowagent"]
      interval: 5s
      timeout: 5s
      retries: 5

  # Redis Cache
  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  # API Server
  api:
    build:
      context: ../..
      dockerfile: infrastructure/docker/Dockerfile.api
    environment:
      - NODE_ENV=production
      - DATABASE_URL=postgresql://flowagent:${POSTGRES_PASSWORD}@postgres:5432/flowagent
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379
      - JWT_SECRET=${JWT_SECRET}
      - ENCRYPTION_KEY=${ENCRYPTION_KEY}
    ports:
      - "3001:3001"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - uploads:/app/uploads
    restart: unless-stopped

  # Agent Engine
  agent-engine:
    build:
      context: ../..
      dockerfile: infrastructure/docker/Dockerfile.agent
    environment:
      - DATABASE_URL=postgresql://flowagent:${POSTGRES_PASSWORD}@postgres:5432/flowagent
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    depends_on:
      - postgres
      - redis
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - sandbox_data:/tmp/sandbox
    privileged: true  # Required for Docker-in-Docker
    restart: unless-stopped

  # Web App
  web:
    build:
      context: ../..
      dockerfile: infrastructure/docker/Dockerfile.web
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:3001
    ports:
      - "3000:3000"
    depends_on:
      - api
    restart: unless-stopped

  # Nginx Reverse Proxy
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - web
      - api
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  uploads:
  sandbox_data:
```

### 11.2 Environment Variables

```bash
# .env.example

# Database
POSTGRES_PASSWORD=your_secure_password_here
DATABASE_URL=postgresql://flowagent:${POSTGRES_PASSWORD}@localhost:5432/flowagent

# Redis
REDIS_PASSWORD=your_redis_password
REDIS_URL=redis://:${REDIS_PASSWORD}@localhost:6379

# Security
JWT_SECRET=your_jwt_secret_min_32_chars
ENCRYPTION_KEY=your_encryption_key_32_chars

# LLM Providers (at least one required)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# Optional: Local models
OLLAMA_BASE_URL=http://localhost:11434

# Stripe (optional, for marketplace)
STRIPE_SECRET_KEY=sk_...
STRIPE_WEBHOOK_SECRET=whsec_...

# Email (optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=...
SMTP_PASS=...

# Storage (optional, defaults to local)
S3_BUCKET=...
S3_REGION=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

### 11.3 Installation Script

```bash
#!/bin/bash
# scripts/setup.sh

set -e

echo "🚀 FlowAgent Self-Hosted Setup"
echo "================================"

# Check dependencies
command -v docker >/dev/null 2>&1 || { echo "❌ Docker required but not installed."; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "❌ Docker Compose required but not installed."; exit 1; }

# Create directories
mkdir -p data/postgres data/redis data/uploads data/ssl

# Generate secrets if not provided
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    
    POSTGRES_PASSWORD=$(openssl rand -base64 32)
    REDIS_PASSWORD=$(openssl rand -base64 32)
    JWT_SECRET=$(openssl rand -base64 32)
    ENCRYPTION_KEY=$(openssl rand -base64 32)
    
    cat > .env << EOF
POSTGRES_PASSWORD=$POSTGRES_PASSWORD
REDIS_PASSWORD=$REDIS_PASSWORD
JWT_SECRET=$JWT_SECRET
ENCRYPTION_KEY=$ENCRYPTION_KEY
DATABASE_URL=postgresql://flowagent:$POSTGRES_PASSWORD@postgres:5432/flowagent
REDIS_URL=redis://:$REDIS_PASSWORD@redis:6379
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
EOF
    
    echo "✅ .env file created. Please add your API keys."
else
    echo "✅ .env file already exists."
fi

# Pull images
echo "📦 Pulling Docker images..."
docker-compose pull

# Start services
echo "🚀 Starting services..."
docker-compose up -d

# Wait for database
echo "⏳ Waiting for database..."
sleep 10

# Run migrations
echo "🔄 Running database migrations..."
docker-compose exec api npx prisma migrate deploy

# Create admin user
echo "👤 Creating admin user..."
docker-compose exec api npm run create-admin

echo ""
echo "✅ FlowAgent is ready!"
echo ""
echo "🌐 Web Interface: http://localhost"
echo "🔌 API Endpoint: http://localhost/api"
echo ""
echo "Next steps:"
echo "1. Add your API keys to .env"
echo "2. Restart: docker-compose restart"
echo "3. Visit http://localhost to complete setup"
```

---

## 12. Implementation Roadmap

### Week 1: Foundation

**Days 1-2: Project Setup**
- [ ] Initialize monorepo with Turborepo
- [ ] Set up package structure
- [ ] Configure CI/CD pipeline
- [ ] Set up development environment

**Days 3-4: Database & Auth**
- [ ] Design and implement Prisma schema
- [ ] Set up Clerk authentication
- [ ] Create user management endpoints
- [ ] Implement API key system

**Days 5-7: Basic API**
- [ ] Set up tRPC router structure
- [ ] Implement agent CRUD operations
- [ ] Create execution endpoints
- [ ] Add basic rate limiting

### Week 2: Agent Engine

**Days 8-10: Core Engine**
- [ ] Set up Python agent engine
- [ ] Implement LangGraph integration
- [ ] Create tool registry
- [ ] Build basic Auto mode

**Days 11-12: Tools**
- [ ] Implement web search tool
- [ ] Create browser automation tool
- [ ] Add code execution sandbox
- [ ] Build API call tool

**Days 13-14: Memory**
- [ ] Implement conversation memory
- [ ] Add vector store integration
- [ ] Create knowledge base system
- [ ] Build memory management UI

### Week 3: Frontend

**Days 15-17: UI Foundation**
- [ ] Set up Next.js 14 with App Router
- [ ] Configure Tailwind + shadcn/ui
- [ ] Create layout and navigation
- [ ] Build authentication pages

**Days 18-19: Agent Builder**
- [ ] Implement visual workflow builder
- [ ] Create agent configuration forms
- [ ] Add tool selection UI
- [ ] Build prompt editor

**Days 20-21: Execution UI**
- [ ] Create execution dashboard
- [ ] Implement real-time streaming
- [ ] Add execution history
- [ ] Build conversation view

### Week 4: Advanced Features

**Days 22-24: Additional Modes**
- [ ] Implement Air mode (collaborative)
- [ ] Build Custom mode (workflow editor)
- [ ] Create Pro mode (advanced settings)
- [ ] Add mode switching UI

**Days 25-26: Web App Generation**
- [ ] Build React code generator
- [ ] Create deployment pipeline
- [ ] Add custom domain support
- [ ] Implement preview mode

**Days 27-28: Document Creation**
- [ ] Add PPT generation
- [ ] Implement PDF creation
- [ ] Create document templates
- [ ] Build export functionality

### Week 5: Marketplace

**Days 29-31: Core Marketplace**
- [ ] Set up Stripe Connect
- [ ] Create template submission flow
- [ ] Build template browsing UI
- [ ] Implement purchase flow

**Days 32-33: Creator Features**
- [ ] Create creator dashboard
- [ ] Add template analytics
- [ ] Implement payout system
- [ ] Build review system

**Days 34-35: Admin Tools**
- [ ] Create admin panel
- [ ] Add template moderation
- [ ] Implement user management
- [ ] Build system analytics

### Week 6: Desktop App

**Days 36-38: Tauri Setup**
- [ ] Initialize Tauri project
- [ ] Configure desktop-specific features
- [ ] Implement local-first architecture
- [ ] Add offline support

**Days 39-40: Desktop Features**
- [ ] Create system tray integration
- [ ] Add global shortcuts
- [ ] Implement file system access
- [ ] Build auto-updater

**Days 41-42: Testing**
- [ ] Test on Windows
- [ ] Test on macOS
- [ ] Test on Linux
- [ ] Fix platform-specific issues

### Week 7: Optimization

**Days 43-45: Performance**
- [ ] Implement response caching
- [ ] Add intelligent model routing
- [ ] Optimize database queries
- [ ] Add CDN for static assets

**Days 46-47: Security**
- [ ] Security audit
- [ ] Implement additional safeguards
- [ ] Add penetration testing
- [ ] Create security documentation

**Days 48-49: Monitoring**
- [ ] Set up error tracking (Sentry)
- [ ] Add performance monitoring
- [ ] Create cost tracking dashboard
- [ ] Build alerting system

### Week 8: Launch Preparation

**Days 50-52: Documentation**
- [ ] Write API documentation
- [ ] Create user guides
- [ ] Add deployment guides
- [ ] Build example templates

**Days 53-54: Testing**
- [ ] End-to-end testing
- [ ] Load testing
- [ ] Security testing
- [ ] User acceptance testing

**Days 55-56: Launch**
- [ ] Deploy to production
- [ ] Create launch assets
- [ ] Write announcement blog post
- [ ] Post on Hacker News, Product Hunt

### Weeks 9-10: Post-Launch

**Days 57-63: Iteration**
- [ ] Monitor metrics
- [ ] Fix critical bugs
- [ ] Gather user feedback
- [ ] Implement quick wins

**Days 64-70: Scale Preparation**
- [ ] Optimize for scale
- [ ] Add horizontal scaling support
- [ ] Implement caching layers
- [ ] Create runbooks

---

## Appendices

### A. API Rate Limits

| Tier | Requests/Min | Executions/Month | Concurrent |
|------|-------------|------------------|------------|
| Free | 60 | 1,000 | 1 |
| Pro | 600 | Unlimited | 5 |
| Enterprise | 6,000 | Unlimited | 20 |

### B. Cost Estimates

**Development (10 weeks):**
- 2 Senior Engineers: $30,000
- 1 Designer: $8,000
- Infrastructure: $2,000
- **Total: $40,000**

**Monthly Operating Costs (1K users):**
- Database (RDS): $200
- Redis (ElastiCache): $100
- Compute (ECS): $500
- Storage (S3): $50
- LLM API costs: $1,000
- **Total: ~$1,850/month**

### C. Third-Party Services

**Required:**
- PostgreSQL (self-hosted or managed)
- Redis (self-hosted or managed)
- LLM Provider (OpenAI, Anthropic, or local)

**Optional:**
- Clerk (auth) - can use Keycloak for self-hosted
- Stripe (payments) - only for marketplace
- Sentry (error tracking)
- PostHog (analytics)

### D. Success Metrics

**Technical:**
- API response time < 200ms (p95)
- Agent execution success rate > 95%
- System uptime > 99.9%
- Error rate < 0.1%

**Business:**
- 1,000 GitHub stars (Month 3)
- 500 active users (Month 6)
- $10K MRR (Month 12)
- 50 marketplace templates (Month 6)

---

**Document Version:** 1.0  
**Last Updated:** January 31, 2026  
**Status:** Ready for Development

This document provides a complete technical specification for building FlowAgent. Developers should be able to use this as the primary reference for implementation.

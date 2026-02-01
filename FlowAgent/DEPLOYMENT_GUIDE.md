# FlowAgent - Production Deployment Guide

## 🎯 Overview

This guide provides step-by-step instructions for deploying FlowAgent to production with enterprise-grade security and reliability.

---

## 📋 PRE-DEPLOYMENT CHECKLIST

### ✅ Phase 1: Prerequisites (Before You Start)

#### 1.1 Required Accounts
- [ ] **Supabase Account** (Database, Auth, Storage)
  - Sign up at: https://supabase.com
  - Create a new project
  - Note down: Project URL, Anon Key, Service Role Key

- [ ] **Vercel Account** (Frontend Hosting)
  - Sign up at: https://vercel.com
  - Connect your GitHub/GitLab/Bitbucket account

- [ ] **Domain Name** (Optional but recommended)
  - Purchase from Namecheap, Cloudflare, or your preferred registrar
  - Recommended: Your own domain for production

#### 1.2 Local Development Setup
- [ ] Node.js 18+ installed
- [ ] pnpm installed (`npm install -g pnpm`)
- [ ] Git repository initialized and pushed to GitHub

---

## 🔐 Phase 2: Security Setup (CRITICAL)

### 2.1 Generate Encryption Key

**This is the MOST IMPORTANT step for security!**

```bash
# Generate a secure 32+ character encryption key
node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"

# Example output: a1b2c3d4e5f6... (64 hex characters = 32 bytes)
```

**⚠️ CRITICAL WARNINGS:**
- Save this key in a secure password manager (1Password, Bitwarden, etc.)
- NEVER commit this key to git
- Use different keys for staging and production
- If you lose this key, all encrypted API keys become unrecoverable

### 2.2 Environment Variables Setup

Create `apps/web/.env.local`:

```bash
# ============================================
# SUPABASE CONFIGURATION (Required)
# ============================================
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIs...
SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIs...

# ============================================
# APP CONFIGURATION
# ============================================
NEXT_PUBLIC_APP_URL=https://your-domain.com

# ============================================
# SECURITY: API KEY ENCRYPTION (CRITICAL)
# ============================================
# Generate with: node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
ENCRYPTION_KEY=your-32-char-secure-key-here!!
```

**Validation Checklist:**
- [ ] `ENCRYPTION_KEY` is exactly 32+ characters
- [ ] `ENCRYPTION_KEY` is NOT marked as `NEXT_PUBLIC_` (server-side only)
- [ ] All Supabase credentials are from the same project
- [ ] `.env.local` is in `.gitignore`

---

## 🗄️ Phase 3: Database Setup

### 3.1 Run Database Migration

**⚠️ CRITICAL: Must be done before first deploy!**

1. Go to [Supabase Dashboard](https://supabase.com/dashboard)
2. Select your project
3. Navigate to **SQL Editor** → **New Query**
4. Copy the entire contents of `supabase/schema_v2.sql`
5. Click **Run**
6. Verify no errors occurred

### 3.2 Verify Database Tables

Run this query in Supabase SQL Editor to verify:

```sql
-- Check all tables were created
SELECT table_name 
FROM information.tables 
WHERE table_schema = 'public' 
AND table_name IN ('conversations', 'messages', 'api_keys', 'projects', 'agent_personas', 'usage_tracking');

-- Should return 6 rows
```

**Checklist:**
- [ ] `conversations` table exists
- [ ] `messages` table exists
- [ ] `api_keys` table exists
- [ ] `projects` table exists
- [ ] `agent_personas` table exists
- [ ] `usage_tracking` table exists
- [ ] RLS policies are enabled
- [ ] 5 pre-built agent personas are inserted

### 3.3 Verify Storage Bucket

1. Go to **Storage** in Supabase Dashboard
2. Verify bucket `chat-attachments` exists
3. Check that it's set to **Public: true**
4. Verify storage policies are created

---

## 🔧 Phase 4: Supabase Configuration

### 4.1 Authentication Settings

1. Go to **Authentication** → **Providers**
2. Enable **Email** provider
3. Configure settings:
   - **Confirm email:** Enabled (recommended for production)
   - **Secure email change:** Enabled
   - **Secure password change:** Enabled

### 4.2 URL Configuration

1. Go to **Authentication** → **URL Configuration**
2. Set **Site URL:** `https://your-domain.com`
3. Add **Redirect URLs:**
   - `https://your-domain.com/auth/callback`
   - `https://your-domain.com/login`
   - `http://localhost:3000` (for local development)

### 4.3 Email Templates (Optional)

Customize email templates for:
- [ ] Confirm signup
- [ ] Invite user
- [ ] Magic link
- [ ] Change email address
- [ ] Reset password

---

## 🚀 Phase 5: Deployment

### 5.1 Deploy to Vercel

#### Option A: Git Integration (Recommended)

1. Push your code to GitHub
2. Go to [Vercel Dashboard](https://vercel.com/dashboard)
3. Click **Add New...** → **Project**
4. Import your GitHub repository
5. Configure:
   - **Framework Preset:** Next.js
   - **Root Directory:** `apps/web`
   - **Build Command:** `npm run build`
   - **Output Directory:** `.next`
6. Add **Environment Variables**:
   - Copy all variables from `.env.local`
   - Add them one by one in Vercel dashboard
7. Click **Deploy**

#### Option B: Vercel CLI

```bash
# Install Vercel CLI
npm i -g vercel

# Login
vercel login

# Deploy
vercel --prod
```

### 5.2 Configure Custom Domain (Optional)

1. In Vercel Dashboard, go to your project
2. Click **Settings** → **Domains**
3. Add your custom domain
4. Follow DNS configuration instructions
5. Wait for SSL certificate provisioning (usually automatic)

### 5.3 Verify Deployment

**Checklist:**
- [ ] Build completed successfully
- [ ] No build errors in logs
- [ ] Site loads at `https://your-domain.com`
- [ ] SSL certificate is valid (green lock in browser)

---

## 🧪 Phase 6: Post-Deployment Testing

### 6.1 Smoke Tests

**Authentication:**
- [ ] Can register new account
- [ ] Can login with email/password
- [ ] Can logout
- [ ] Session persists across page reloads
- [ ] Protected routes redirect to login when not authenticated

**API Keys:**
- [ ] Navigate to `/byok`
- [ ] Add OpenAI API key (format: `sk-...`)
- [ ] Verify key is saved (shows in list)
- [ ] Set key as default
- [ ] Delete key works

**Chat Functionality:**
- [ ] Start new conversation from dashboard
- [ ] Send message
- [ ] Receive streaming response
- [ ] Agent names appear (Project Manager, etc.)
- [ ] Budget indicator updates
- [ ] Conversation history persists

**File Upload:**
- [ ] Upload file in chat (under 50MB)
- [ ] File appears in message
- [ ] Can download/view file

### 6.2 Security Tests

**API Key Security:**
- [ ] API keys don't appear in browser DevTools Network tab
- [ ] API keys don't appear in browser console
- [ ] Page source doesn't contain API keys
- [ ] LocalStorage doesn't contain API keys

**Authentication Security:**
- [ ] Can't access `/api/chat` without authentication
- [ ] Can't access `/api/keys` without authentication
- [ ] RLS policies prevent cross-user data access

### 6.3 Performance Tests

- [ ] Page loads in under 3 seconds
- [ ] Chat streaming is smooth
- [ ] No memory leaks during long conversations
- [ ] File uploads work reliably

---

## 📊 Phase 7: Monitoring & Maintenance

### 7.1 Set Up Monitoring (Recommended)

**Error Tracking:**
- [ ] Sign up for Sentry (https://sentry.io)
- [ ] Add Sentry DSN to environment variables
- [ ] Install Sentry SDK in Next.js

**Analytics:**
- [ ] Enable Vercel Analytics
- [ ] Or set up Google Analytics
- [ ] Track key metrics: signups, active users, API usage

**Database Monitoring:**
- [ ] Enable Supabase database reports
- [ ] Set up alerts for high connection counts
- [ ] Monitor storage usage

### 7.2 Backup Strategy

**Database:**
- [ ] Enable Supabase daily backups (included in Pro plan)
- [ ] Test restore procedure

**Environment Variables:**
- [ ] Store in secure vault (1Password, etc.)
- [ ] Document all variables
- [ ] Keep staging and production configs separate

### 7.3 Update Strategy

**Regular Updates:**
- [ ] Schedule monthly dependency updates
- [ ] Test updates in staging first
- [ ] Monitor for breaking changes

**Security Updates:**
- [ ] Subscribe to security advisories
- [ ] Apply critical patches within 24 hours
- [ ] Review encryption key rotation annually

---

## 🚨 Phase 8: Troubleshooting

### Common Issues & Solutions

#### Issue: "ENCRYPTION_KEY environment variable is required"
**Solution:**
- Verify `ENCRYPTION_KEY` is set in Vercel environment variables
- Ensure it's NOT marked as "Preview" only
- Redeploy after adding the variable

#### Issue: "Failed to decrypt API key"
**Solution:**
- This happens when you change ENCRYPTION_KEY
- Users need to re-add their API keys
- Always backup your encryption key!

#### Issue: "No API key configured" error in chat
**Solution:**
- User must add API key at `/byok` first
- Verify key is marked as "default"
- Check that key format is valid (starts with sk- for OpenAI)

#### Issue: Database connection errors
**Solution:**
- Check Supabase project is active
- Verify connection string is correct
- Check if IP is allowlisted (if using IP restrictions)

#### Issue: File upload fails
**Solution:**
- Verify `chat-attachments` bucket exists
- Check storage policies are correct
- Ensure file is under 50MB

---

## 📈 Phase 9: Scaling (Future)

### When You Need to Scale:

**Database:**
- Upgrade Supabase plan for more connections
- Enable connection pooling
- Consider read replicas for analytics

**Performance:**
- Enable Vercel Edge Network
- Use Next.js Image optimization
- Implement caching strategies

**Security:**
- Implement rate limiting per user
- Add DDoS protection (Cloudflare)
- Set up WAF rules

---

## ✅ FINAL DEPLOYMENT CHECKLIST

### Before Deploy:
- [ ] Database migration run successfully
- [ ] Encryption key generated and saved securely
- [ ] All environment variables configured
- [ ] Supabase auth URLs configured
- [ ] Build passes locally (`npm run build`)
- [ ] Type check passes (`npm run typecheck`)

### During Deploy:
- [ ] Vercel project created
- [ ] Environment variables added to Vercel
- [ ] Build completes successfully
- [ ] Custom domain configured (if using)
- [ ] SSL certificate active

### After Deploy:
- [ ] Smoke tests pass
- [ ] Security tests pass
- [ ] API keys can be added
- [ ] Chat functionality works
- [ ] File upload works
- [ ] Monitoring active
- [ ] Team notified of deployment

---

## 🎉 SUCCESS!

If you've completed all checklists above, your FlowAgent instance is:
- ✅ **Fully functional**
- ✅ **Secure** (AES-256-GCM encryption)
- ✅ **Production-ready**
- ✅ **Monitored**
- ✅ **Backed up**

**Your users can now:**
- Register and login securely
- Add their own API keys (encrypted)
- Chat with AI agents
- Upload files
- Track usage and budget

---

## 📞 Support

If you encounter issues:
1. Check Vercel deployment logs
2. Check Supabase database logs
3. Review browser console for errors
4. Check Sentry for error reports (if configured)

**Emergency Contacts:**
- Supabase Support: support@supabase.io
- Vercel Support: vercel.com/support
- FlowAgent Issues: GitHub Issues

---

**Last Updated:** February 2025
**Version:** 2.0 (Secure Edition)
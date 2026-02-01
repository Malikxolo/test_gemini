# FlowAgent - Complete "Do It For Me" Deployment Guide

This guide will walk you through deployment with copy-paste commands. Just follow each step in order.

## ⚡ STEP-BY-STEP DEPLOYMENT

---

## **STEP 1: Prerequisites (2 minutes)**

### 1.1 Install Required Tools

**macOS:**
```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Node.js
brew install node

# Install pnpm
npm install -g pnpm

# Verify installations
node --version  # Should show v18+
pnpm --version  # Should show 8+
```

**Windows:**
```powershell
# Install Node.js from https://nodejs.org (LTS version)
# Then install pnpm
npm install -g pnpm
```

**Linux:**
```bash
# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install pnpm
npm install -g pnpm
```

### 1.2 Create Required Accounts

1. **Supabase** (Database + Auth): https://supabase.com
   - Sign up with GitHub
   - Click "New Project"
   - Name: `flowagent-prod`
   - Password: (create secure password)
   - Region: Choose closest to your users
   - Click "Create new project"

2. **Vercel** (Hosting): https://vercel.com
   - Sign up with GitHub
   - Click "Continue with GitHub"

3. **GitHub** (Code repository): https://github.com
   - Create new repository called `flowagent`
   - Make it private
   - Don't initialize with README

✅ **Checkpoint:** You have Node.js, pnpm, and accounts created.

---

## **STEP 2: Prepare Your Code (3 minutes)**

### 2.1 Navigate to Project

```bash
# Make sure you're in the project root
cd /Users/_iayushsharma_/Documents/final voice agent/aakashbhai/FlowAgent

# Verify you're in the right place
ls -la
# Should see: apps/, packages/, supabase/, package.json
```

### 2.2 Initialize Git Repository

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - FlowAgent production ready"

# Add remote (replace with your actual GitHub repo URL)
git remote add origin https://github.com/YOUR_USERNAME/flowagent.git

# Push to GitHub
git push -u origin main
```

✅ **Checkpoint:** Code is on GitHub.

---

## **STEP 3: Database Setup (5 minutes)**

### 3.1 Get Supabase Credentials

1. Go to https://supabase.com/dashboard
2. Click on your `flowagent-prod` project
3. In the left sidebar, click **Settings** (gear icon at bottom)
4. Click **API** in the settings menu
5. Copy these values:
   - `URL` → Save as: `SUPABASE_URL`
   - `anon public` → Save as: `SUPABASE_ANON_KEY`
   - `service_role secret` → Save as: `SUPABASE_SERVICE_ROLE_KEY`

### 3.2 Run Database Migration

1. In Supabase Dashboard, click **SQL Editor** (in left sidebar)
2. Click **New Query**
3. Copy the ENTIRE contents of `supabase/schema_v2.sql`
4. Paste into the SQL Editor
5. Click **Run**
6. Wait for "Success. No rows returned" message

### 3.3 Verify Database Setup

Run this query in SQL Editor to verify:

```sql
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public';
```

You should see:
- `agent_personas`
- `api_keys`
- `conversations`
- `messages`
- `projects`
- `usage_tracking`

✅ **Checkpoint:** Database is ready with all tables.

---

## **STEP 4: Generate Encryption Key (1 minute)**

This is CRITICAL for security!

```bash
# Generate secure encryption key
node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
```

**Output will look like:**
```
a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456
```

**⚠️ CRITICAL:**
1. Copy this key
2. Save it in your password manager (1Password, Bitwarden, etc.)
3. Label it: "FlowAgent Production Encryption Key"
4. **NEVER lose this key** - you cannot recover encrypted API keys without it

✅ **Checkpoint:** You have your `ENCRYPTION_KEY` saved securely.

---

## **STEP 5: Environment Variables (3 minutes)**

### 5.1 Create Environment File

```bash
# Create the environment file
cat > apps/web/.env.local << 'EOF'
# ============================================
# SUPABASE CONFIGURATION (Required)
# ============================================
NEXT_PUBLIC_SUPABASE_URL=PASTE_YOUR_URL_HERE
NEXT_PUBLIC_SUPABASE_ANON_KEY=PASTE_YOUR_ANON_KEY_HERE
SUPABASE_SERVICE_ROLE_KEY=PASTE_YOUR_SERVICE_ROLE_KEY_HERE

# ============================================
# APP CONFIGURATION
# ============================================
NEXT_PUBLIC_APP_URL=http://localhost:3000

# ============================================
# SECURITY: API KEY ENCRYPTION (CRITICAL)
# ============================================
ENCRYPTION_KEY=PASTE_YOUR_64_CHAR_KEY_HERE
EOF
```

### 5.2 Replace Placeholders

Edit `apps/web/.env.local` and replace:
- `PASTE_YOUR_URL_HERE` → Your Supabase URL (e.g., `https://abcdefgh123456.supabase.co`)
- `PASTE_YOUR_ANON_KEY_HERE` → Your Supabase anon key
- `PASTE_YOUR_SERVICE_ROLE_KEY_HERE` → Your Supabase service role key
- `PASTE_YOUR_64_CHAR_KEY_HERE` → Your 64-character encryption key from Step 4

### 5.3 Verify File

```bash
# Check file exists
cat apps/web/.env.local

# Make sure it's NOT in git
cat .gitignore | grep ".env"
# Should see: .env.local
```

✅ **Checkpoint:** Environment variables configured.

---

## **STEP 6: Test Locally (5 minutes)**

### 6.1 Install Dependencies

```bash
# Install all dependencies
pnpm install
```

### 6.2 Build Project

```bash
# Build the project
npm run build
```

**Expected output:**
```
Tasks:    2 successful, 2 total
```

If you see errors, check:
- All environment variables are set correctly
- Database migration was successful
- Node.js version is 18+

### 6.3 Test Locally (Optional)

```bash
# Start development server
cd apps/web && pnpm dev

# Open browser to http://localhost:3000
# You should see the FlowAgent homepage
```

**Test:**
1. Click "Get Started" or "Sign Up"
2. Create an account
3. Login
4. Navigate to Settings → API Keys
5. Add a test API key (use a fake key like `sk-test123`)
6. Verify it saves successfully

Press `Ctrl+C` to stop the dev server when done.

✅ **Checkpoint:** Project builds successfully and runs locally.

---

## **STEP 7: Deploy to Vercel (10 minutes)**

### 7.1 Install Vercel CLI

```bash
# Install Vercel CLI globally
npm install -g vercel

# Login to Vercel
vercel login
# This will open a browser window - click "Authorize"
```

### 7.2 Deploy Project

```bash
# Navigate to web app
cd apps/web

# Deploy to Vercel
vercel --prod
```

**During deployment, you'll be asked:**
- `Set up and deploy "~/Documents/.../FlowAgent/apps/web"?` → Type `Y`
- `Which scope do you want to deploy to?` → Select your account
- `Link to existing project?` → Type `N`
- `What's your project name?` → Type `flowagent` (or your preferred name)

**Wait for deployment to complete...**

You should see:
```
🔍  Inspect: https://vercel.com/.../flowagent/...
✅  Production: https://flowagent-xxx.vercel.app
```

✅ **Checkpoint:** Project is deployed to Vercel.

---

## **STEP 8: Configure Environment Variables on Vercel (5 minutes)**

### 8.1 Add Environment Variables

1. Go to https://vercel.com/dashboard
2. Click on your `flowagent` project
3. Click **Settings** tab (top navigation)
4. Click **Environment Variables** in left sidebar
5. Add each variable one by one:

**Variable 1:**
- Name: `NEXT_PUBLIC_SUPABASE_URL`
- Value: (paste your Supabase URL)
- Environment: Production ✓, Preview ✓, Development ✓

**Variable 2:**
- Name: `NEXT_PUBLIC_SUPABASE_ANON_KEY`
- Value: (paste your Supabase anon key)
- Environment: Production ✓, Preview ✓, Development ✓

**Variable 3:**
- Name: `SUPABASE_SERVICE_ROLE_KEY`
- Value: (paste your Supabase service role key)
- Environment: Production ✓, Preview ✓, Development ✓

**Variable 4:**
- Name: `ENCRYPTION_KEY`
- Value: (paste your 64-character encryption key)
- Environment: Production ✓, Preview ✓, Development ✓

**Variable 5:**
- Name: `NEXT_PUBLIC_APP_URL`
- Value: `https://your-domain.vercel.app` (replace with your actual Vercel URL)
- Environment: Production ✓, Preview ✓, Development ✓

6. Click **Save** after adding each variable

### 8.2 Redeploy

After adding all environment variables:

```bash
# Redeploy to apply environment variables
cd apps/web
vercel --prod
```

Or use the Vercel Dashboard:
1. Go to Deployments tab
2. Click the three dots (...) on latest deployment
3. Click "Redeploy"

✅ **Checkpoint:** Environment variables are set on Vercel.

---

## **STEP 9: Configure Supabase Auth URLs (3 minutes)**

### 9.1 Set Site URL

1. Go to https://supabase.com/dashboard
2. Select your `flowagent-prod` project
3. Click **Authentication** in left sidebar
4. Click **URL Configuration**
5. Set **Site URL:** `https://your-domain.vercel.app` (your actual Vercel URL)

### 9.2 Add Redirect URLs

Add these URLs (one per line):
```
https://your-domain.vercel.app
https://your-domain.vercel.app/auth/callback
http://localhost:3000
```

6. Click **Save**

✅ **Checkpoint:** Authentication is configured.

---

## **STEP 10: Production Testing (10 minutes)**

### 10.1 Basic Smoke Test

Open your deployed URL: `https://your-domain.vercel.app`

**Test 1: Homepage Loads**
- [ ] Homepage loads without errors
- [ ] No console errors (Press F12 → Console)

**Test 2: User Registration**
- [ ] Click "Sign Up"
- [ ] Fill in email, password, username
- [ ] Submit form
- [ ] Check email for confirmation (if enabled)
- [ ] Confirm email

**Test 3: User Login**
- [ ] Click "Login"
- [ ] Enter credentials
- [ ] Successfully logged in
- [ ] See dashboard

### 10.2 API Key Test

1. Go to Settings → API Keys (or `/byok`)
2. Add OpenAI API Key:
   - Provider: OpenAI
   - Key Name: Default
   - API Key: (your real OpenAI key starting with `sk-`)
   - Click Save
3. Verify key appears in the list
4. Mark it as default

### 10.3 Chat Test

1. Go to Dashboard
2. Type a message: "Hello, what can you do?"
3. Press Enter or click Send
4. **Expected:** You should see a streaming response
5. Check that:
   - [ ] Message sends successfully
   - [ ] Response streams in
   - [ ] Agent name appears (e.g., "Project Manager")
   - [ ] Budget indicator updates
   - [ ] Message is saved in history

### 10.4 File Upload Test

1. In chat, click the paperclip icon
2. Select a small file (under 50MB)
3. Verify file uploads
4. Send message with file

### 10.5 Security Test

Open browser DevTools (F12):

```javascript
// Check localStorage - should be empty or not contain API keys
localStorage.getItem('apiKey')  // Should return null

// Check sessionStorage
sessionStorage.getItem('apiKey')  // Should return null

// Check cookies
document.cookie  // Should not contain API keys
```

✅ **Checkpoint:** All tests pass!

---

## **STEP 11: Setup Monitoring (Optional, 5 minutes)**

### 11.1 Enable Vercel Analytics

1. In Vercel Dashboard → Your Project
2. Click **Analytics** tab
3. Click "Enable Analytics"
4. Redeploy if prompted

### 11.2 Add Error Tracking (Recommended)

**Option A: Sentry (Free tier)**
1. Sign up at https://sentry.io
2. Create new project → Next.js
3. Follow their setup wizard
4. Add DSN to environment variables

**Option B: LogRocket**
1. Sign up at https://logrocket.com
2. Install their SDK
3. Configure in your app

✅ **Checkpoint:** Monitoring is active.

---

## **STEP 12: Final Verification**

### Deployment Checklist

- [ ] Database migration completed
- [ ] Environment variables set on Vercel
- [ ] Build successful
- [ ] Site loads at production URL
- [ ] User registration works
- [ ] User login works
- [ ] API keys can be added
- [ ] Chat streaming works
- [ ] Budget tracking works
- [ ] File upload works
- [ ] Supabase auth URLs configured
- [ ] SSL certificate active (green lock in browser)
- [ ] Monitoring enabled (optional)

### Security Checklist

- [ ] API keys not visible in browser DevTools
- [ ] `ENCRYPTION_KEY` not exposed to client
- [ ] HTTPS enabled
- [ ] RLS policies active
- [ ] No secrets in git repository

---

## 🎉 SUCCESS!

Your FlowAgent instance is now:
- ✅ **Live at:** https://your-domain.vercel.app
- ✅ **Secure** with AES-256-GCM encryption
- ✅ **Functional** with chat, file upload, budget tracking
- ✅ **Production-ready**

---

## 🆘 TROUBLESHOOTING

### Issue: "Database error"
```bash
# Solution: Re-run migration
# Go to Supabase SQL Editor and run schema_v2.sql again
```

### Issue: "ENCRYPTION_KEY required"
```bash
# Solution: Add ENCRYPTION_KEY to Vercel environment variables
# Make sure it's 32+ characters
# Redeploy after adding
```

### Issue: "Build failed"
```bash
# Solution: Check logs
cd apps/web
vercel logs --production

# Common fixes:
# 1. Make sure all env vars are set
# 2. Run npm install again
# 3. Check for TypeScript errors: npm run typecheck
```

### Issue: "No API key configured" in chat
```bash
# Solution:
# 1. Go to /byok
# 2. Add your OpenAI API key (must start with sk-)
# 3. Mark it as default
# 4. Refresh and try again
```

### Issue: "Authentication failed"
```bash
# Solution:
# 1. Check Supabase auth URL configuration
# 2. Verify NEXT_PUBLIC_SUPABASE_URL is correct
# 3. Check that Site URL matches your Vercel URL
```

---

## 📞 NEXT STEPS

### Immediate:
1. Share your app URL with team
2. Add real API keys
3. Test with real users

### This Week:
1. Set up custom domain (optional)
2. Configure email templates
3. Set up monitoring alerts

### This Month:
1. Review usage and costs
2. Optimize performance
3. Add team members to project

---

## 📚 RESOURCES

- **Full Guide:** `DEPLOYMENT_GUIDE.md`
- **Security Checklist:** `SECURITY_CHECKLIST.md`
- **Quick Reference:** `QUICKSTART.md`

---

**Your FlowAgent is ready! 🚀**

If you encounter any issues, check the troubleshooting section above or review the detailed guides.

**Deploy time:** ~45 minutes
**Difficulty:** Easy (just copy-paste commands)
**Result:** Production-ready secure AI platform
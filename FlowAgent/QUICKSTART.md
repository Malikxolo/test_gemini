# Quick Start - 5 Minute Deployment

## 🚀 Deploy FlowAgent in 5 Steps

### Step 1: Prerequisites (1 minute)
```bash
# Ensure you have:
- Node.js 18+
- Git
- A Supabase account (free tier works)
- A Vercel account (free tier works)
```

### Step 2: Database Setup (1 minute)

1. Go to [Supabase Dashboard](https://supabase.com/dashboard)
2. Create new project
3. Go to **SQL Editor** → **New Query**
4. Copy and paste the entire `supabase/schema_v2.sql` file
5. Click **Run**

✅ **Done!** Your database is ready.

### Step 3: Get Credentials (1 minute)

In Supabase Dashboard:
- **Settings** → **API** → Copy:
  - `URL` → `NEXT_PUBLIC_SUPABASE_URL`
  - `anon public` → `NEXT_PUBLIC_SUPABASE_ANON_KEY`
  - `service_role secret` → `SUPABASE_SERVICE_ROLE_KEY`

### Step 4: Generate Encryption Key (30 seconds)

```bash
node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
```

Copy the output - this is your `ENCRYPTION_KEY`

**⚠️ SAVE THIS SECURELY!** Lose it = lose all encrypted API keys

### Step 5: Deploy to Vercel (2 minutes)

#### Option A: One-Click Deploy
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new)

1. Import your GitHub repo
2. Set **Root Directory:** `apps/web`
3. Add Environment Variables:
   ```
   NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
   NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
   SUPABASE_SERVICE_ROLE_KEY=your-service-key
   NEXT_PUBLIC_APP_URL=https://your-domain.vercel.app
   ENCRYPTION_KEY=your-32-char-key-from-step-4
   ```
4. Click **Deploy**

#### Option B: CLI
```bash
# Install Vercel CLI
npm i -g vercel

# Login
vercel login

# Deploy
cd apps/web
vercel --prod
```

---

## ✅ Verification (30 seconds)

1. Open your deployed URL
2. Click **Sign Up** → Create account
3. Go to **Settings** → **API Keys**
4. Add your OpenAI key (starts with `sk-`)
5. Go to **Dashboard** → Start a chat
6. Send a message

**If you get a response → SUCCESS! 🎉**

---

## 🆘 Troubleshooting

### "Database error"
- Did you run the SQL migration in Step 2?
- Check Supabase project is active

### "ENCRYPTION_KEY required"
- Add `ENCRYPTION_KEY` to Vercel environment variables
- Must be 32+ characters

### "No API key configured"
- Go to `/byok` and add your OpenAI/Anthropic key
- Key must be valid format (starts with sk-)

### "Build failed"
- Check `apps/web/.env.local` exists
- Run `npm run build` locally to see errors

---

## 📚 Next Steps

- **Full Guide:** See `DEPLOYMENT_GUIDE.md` for detailed instructions
- **Security:** Review security architecture in main README
- **Customization:** Edit `apps/web/src/lib/llm/swarm.ts` to customize agents

---

**Need Help?** 
- Check full deployment guide
- Review troubleshooting section
- Check Vercel/Supabase docs

**Deploy Time:** ~5 minutes
**Difficulty:** Easy
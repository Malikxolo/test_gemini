# Supabase Setup Guide

## 1. Create Supabase Project

1. Go to [https://supabase.com](https://supabase.com)
2. Sign up or log in
3. Click "New Project"
4. Fill in the details:
   - Name: FlowAgent
   - Database Password: (generate a secure one)
   - Region: Choose closest to your users

## 2. Get Your API Keys

After the project is created, go to **Project Settings** → **API**:

Copy these values:
- `URL` → Set as `NEXT_PUBLIC_SUPABASE_URL`
- `anon public` → Set as `NEXT_PUBLIC_SUPABASE_ANON_KEY`
- `service_role secret` → Set as `SUPABASE_SERVICE_ROLE_KEY` (keep this secret!)

## 3. Set Environment Variables

Create `.env.local` file in `apps/web/`:

```env
NEXT_PUBLIC_SUPABASE_URL=your_project_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
```

## 4. Run Database Schema

1. Go to **SQL Editor** in Supabase Dashboard
2. Click **New Query**
3. Copy the contents of `supabase/schema.sql`
4. Paste and click **Run**

This will create:
- `profiles` table with RLS policies
- `agents` table with RLS policies
- Triggers for auto-creating profiles
- Updated_at triggers

## 5. Enable Email Auth (Optional)

Go to **Authentication** → **Providers** → **Email**:
- Enable "Confirm email" if you want email verification
- Or disable it for easier testing

## 6. Test the Authentication

Start the development server:

```bash
cd apps/web
pnpm dev
```

Visit:
- `http://localhost:3000/signup` - Create an account
- `http://localhost:3000/login` - Log in
- `http://localhost:3000/dashboard` - Access protected route

## Features Enabled

✅ Email/Password Authentication
✅ Session Management (auto-refresh)
✅ Row Level Security (RLS)
✅ Automatic Profile Creation
✅ Protected Routes via Middleware
✅ User Metadata (username, display_name)

## Database Schema

### Profiles Table
- `id` - References auth.users
- `username` - Unique username
- `display_name` - Display name
- `avatar_url` - Profile image URL
- `subscription_tier` - User's plan (free/pro)
- `created_at`, `updated_at` - Timestamps

### Agents Table
- `id` - Unique agent ID
- `user_id` - Owner reference
- `name`, `description` - Agent info
- `system_prompt` - AI instructions
- `model`, `temperature`, `max_tokens` - Model settings
- `tools` - Enabled tools array
- `is_public` - Visibility flag
- `created_at`, `updated_at` - Timestamps

## Security

- All tables use Row Level Security (RLS)
- Users can only access their own data
- Public agents are visible to all
- Service role key should only be used server-side

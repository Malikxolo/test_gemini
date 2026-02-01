# FlowAgent Deployment Documentation

## 📚 Available Guides

### 🚀 Quick Start (5 minutes)
**File:** `QUICKSTART.md`

For experienced developers who want to deploy fast.
- 5-step deployment process
- Minimal configuration
- Get running in 5 minutes

**Use this if:** You're familiar with Next.js, Supabase, and Vercel

---

### 📖 Complete Deployment Guide
**File:** `DEPLOYMENT_GUIDE.md`

Comprehensive step-by-step guide with detailed checklists.
- 9 deployment phases
- Security setup
- Testing procedures
- Troubleshooting
- Monitoring setup

**Use this if:** You want detailed instructions and best practices

---

### 🔒 Security Checklist
**File:** `SECURITY_CHECKLIST.md`

Security-focused deployment verification.
- Security architecture review
- Penetration testing procedures
- Incident response plans
- Security monitoring setup

**Use this if:** Security is your top priority

---

## 🎯 Which Guide Should I Use?

| Scenario | Recommended Guide |
|----------|------------------|
| Quick prototype/MVP | QUICKSTART.md |
| Production deployment | DEPLOYMENT_GUIDE.md |
| Security audit | SECURITY_CHECKLIST.md |
| Team onboarding | DEPLOYMENT_GUIDE.md |
| Emergency deployment | QUICKSTART.md |

---

## 📋 Quick Reference

### Essential Files
- `supabase/schema_v2.sql` - Database migration
- `apps/web/.env.example` - Environment variables template
- `DEPLOYMENT.md` - Original deployment notes

### Critical Environment Variables
```bash
NEXT_PUBLIC_SUPABASE_URL=
NEXT_PUBLIC_SUPABASE_ANON_KEY=
SUPABASE_SERVICE_ROLE_KEY=
ENCRYPTION_KEY=  # CRITICAL: 32+ chars, keep secret!
```

### Deployment Order
1. Run database migration
2. Set environment variables
3. Deploy to Vercel
4. Configure Supabase auth URLs
5. Test functionality
6. Set up monitoring

---

## 🆘 Need Help?

1. Check the troubleshooting section in your chosen guide
2. Review error logs in Vercel/Supabase dashboards
3. Verify all environment variables are set
4. Ensure database migration completed successfully

---

**Start with:** [QUICKSTART.md](QUICKSTART.md) for fastest deployment

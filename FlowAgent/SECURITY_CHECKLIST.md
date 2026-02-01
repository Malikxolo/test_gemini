# Production Security Checklist

## 🔒 Critical Security Requirements

### Pre-Deployment Security Audit

#### Encryption Setup
- [ ] `ENCRYPTION_KEY` is exactly 32+ characters
- [ ] `ENCRYPTION_KEY` is NOT exposed to client (no `NEXT_PUBLIC_` prefix)
- [ ] `ENCRYPTION_KEY` is stored in secure vault (1Password, etc.)
- [ ] Different keys for staging vs production
- [ ] Encryption key backup exists

#### Database Security
- [ ] RLS policies enabled on all tables
- [ ] `api_keys` table has encrypted_key column
- [ ] `chat-attachments` bucket is public (required for file access)
- [ ] Storage policies restrict access to authenticated users
- [ ] Database backups enabled

#### Authentication
- [ ] Supabase Auth configured
- [ ] Email confirmation enabled (production)
- [ ] Secure password requirements
- [ ] Session management working
- [ ] Protected routes redirect to login

#### API Security
- [ ] `/api/chat` requires authentication
- [ ] `/api/keys` requires authentication
- [ ] API keys never returned in GET responses
- [ ] Rate limiting configured
- [ ] CORS properly configured

#### Environment Variables
- [ ] `.env.local` in `.gitignore`
- [ ] No secrets in code repository
- [ ] All env vars documented
- [ ] Production values different from staging
- [ ] Service role key never exposed to client

---

## 🛡️ Security Architecture Verification

### Client-Side Security
```
✅ API keys NEVER stored in:
   - localStorage
   - sessionStorage  
   - Cookies
   - React state
   - Browser memory

✅ API keys NEVER visible in:
   - Browser DevTools Network tab
   - Browser console
   - Page source
   - React DevTools
```

### Server-Side Security
```
✅ API keys ONLY decrypted in:
   - /api/chat route
   - /api/keys route
   - Server-side functions

✅ Encryption uses:
   - AES-256-GCM algorithm
   - Random IV per encryption
   - Authentication tag
   - Scrypt key derivation
```

### Database Security
```
✅ RLS Policies enforce:
   - Users can only access own data
   - API keys encrypted at rest
   - File access restricted to owners
   - No direct table access
```

---

## 🔍 Security Testing Procedures

### Test 1: API Key Exposure
```javascript
// In browser console, run:
localStorage.getItem('apiKey') // Should return null
sessionStorage.getItem('apiKey') // Should return null
document.cookie // Should not contain api keys
```

### Test 2: Network Inspection
1. Open DevTools → Network tab
2. Send a chat message
3. Check request payload - should NOT contain API key
4. Check response - should NOT contain API key

### Test 3: Database Access
```sql
-- Try to access another user's data (should fail)
SELECT * FROM api_keys WHERE user_id != 'your-user-id';
-- Should return 0 rows due to RLS
```

### Test 4: Authentication Bypass
```bash
# Try to access API without auth
curl https://your-domain.com/api/chat \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"message":"test"}'
# Should return 401 Unauthorized
```

---

## 🚨 Security Incident Response

### If Encryption Key is Compromised

1. **Immediate Actions:**
   - Rotate encryption key immediately
   - Notify all users to re-add API keys
   - Review access logs

2. **Key Rotation Process:**
   ```bash
   # Generate new key
   node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
   
   # Update ENCRYPTION_KEY in Vercel
   # Redeploy application
   # All users must re-add API keys
   ```

### If Database is Breached

1. **Immediate Actions:**
   - Reset all user passwords
   - Invalidate all sessions
   - Audit access logs
   - Notify affected users

2. **Recovery:**
   - API keys are encrypted (attacker can't use them)
   - Users need to re-add keys
   - Review and strengthen RLS policies

---

## 📊 Security Monitoring

### Set Up Alerts For:
- [ ] Unusual API usage patterns
- [ ] Failed authentication attempts
- [ ] Database connection spikes
- [ ] File upload anomalies
- [ ] Rate limit violations

### Regular Security Tasks:
- [ ] Weekly: Review access logs
- [ ] Monthly: Update dependencies
- [ ] Quarterly: Security audit
- [ ] Annually: Penetration testing

---

## ✅ Final Security Sign-Off

**Before production deployment, verify:**

- [ ] All checklist items completed
- [ ] Security tests pass
- [ ] Encryption working correctly
- [ ] No secrets in repository
- [ ] Monitoring active
- [ ] Incident response plan documented
- [ ] Team trained on security procedures

**Security Score Target: 9.5+/10**

---

## 📞 Security Contacts

- **Critical Issues:** Immediate fix required
- **Security Questions:** Review documentation
- **Vulnerability Reports:** Handle confidentially

---

**Last Updated:** February 2025
**Security Version:** 2.0
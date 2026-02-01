# FlowAgent Frontend Testing Guide

## Quick Start - Testing Steps

### 1. Start the Backend API

First, make sure your backend is running:

```bash
cd /Users/_iayushsharma_/Documents/final\ voice\ agent/aakashbhai/FlowAgent/apps/api
pnpm dev
```

The API should be running on `http://localhost:8787`

### 2. Start the Frontend

In a new terminal:

```bash
cd /Users/_iayushsharma_/Documents/final\ voice\ agent/aakashbhai/FlowAgent/apps/web
pnpm dev
```

The frontend will be available at `http://localhost:3000`

### 3. Test User Flow

#### Step 1: Create an Account
1. Go to `http://localhost:3000/signup`
2. Create a new account with:
   - Email: test@example.com
   - Username: testuser
   - Password: TestPass123!

#### Step 2: Login
1. Go to `http://localhost:3000/login`
2. Login with your credentials

#### Step 3: Create Your First Agent
1. Navigate to "My Agents" in the sidebar
2. Click "Create Agent"
3. Fill in:
   - Name: "Test Assistant"
   - Description: "A helpful test agent"
   - System Prompt: "You are a helpful assistant."
   - Model: GPT-3.5 Turbo
   - Temperature: 0.7
4. Click "Create Agent"

#### Step 4: Test the Agent
1. Click "Run" on your agent card
2. Type a message: "Hello!"
3. The agent should respond (if you have BYOK keys set up)

#### Step 5: Set Up BYOK (Bring Your Own Key)
1. Go to Settings → API Keys
2. Add your OpenAI API key (starts with `sk-`)
3. Save the key

#### Step 6: Check Billing
1. Navigate to "Usage & Billing"
2. View your usage stats
3. Check architecture access status

#### Step 7: Test Purchase Flow (Test Mode)
1. Go to "Purchase Access"
2. Click on any pricing card
3. Use Razorpay test credentials:
   - Card: 5267 3181 8797 5449
   - Expiry: Any future date
   - CVV: Any 3 digits
   - OTP: 123456

### 4. Test All Pages

Navigate through each page to ensure they load:

- [ ] `/dashboard` - Dashboard
- [ ] `/agents` - Agent list
- [ ] `/agents/new` - Create agent
- [ ] `/agents/[id]` - Edit agent
- [ ] `/agents/[id]/execute` - Chat interface
- [ ] `/billing` - Usage & billing
- [ ] `/settings` - General settings
- [ ] `/settings/api-keys` - API key management
- [ ] `/purchase` - Pricing page
- [ ] `/download` - Download page (after purchase)

### 5. Test Responsive Design

Open DevTools (F12) and test different screen sizes:
- Desktop: 1920x1080
- Tablet: 768x1024
- Mobile: 375x667

### 6. Test Error Handling

Try these scenarios:
- Submit forms with empty fields
- Enter invalid API keys
- Navigate to non-existent agent IDs
- Test without internet connection

## Environment Variables

Create a `.env.local` file in `apps/web/`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8787
```

## Common Issues

### Issue 1: "Cannot connect to API"
**Solution:** Make sure the backend is running on port 8787

### Issue 2: "CORS errors"
**Solution:** The backend should allow requests from localhost:3000

### Issue 3: "BYOK not working"
**Solution:** 
1. Add valid API keys in Settings → API Keys
2. Make sure keys start with correct prefix:
   - OpenAI: `sk-`
   - Anthropic: `sk-ant-`

### Issue 4: "Payment not working"
**Solution:** Use Razorpay test mode credentials (see Step 7 above)

## Testing Checklist

### Authentication
- [ ] Sign up works
- [ ] Login works
- [ ] Logout works
- [ ] Protected routes redirect to login

### Agent Management
- [ ] Create agent
- [ ] Edit agent
- [ ] Delete agent
- [ ] View agent list
- [ ] Execute agent (chat)

### BYOK
- [ ] Add API keys
- [ ] Update API keys
- [ ] Delete API keys
- [ ] Keys are masked
- [ ] Validation works

### Billing
- [ ] View usage stats
- [ ] View payment history
- [ ] Purchase access
- [ ] Download source (after purchase)

### UI/UX
- [ ] Sidebar navigation works
- [ ] All pages load without errors
- [ ] Responsive on mobile
- [ ] Loading states show
- [ ] Error messages display

## Automated Testing

Run the test suite:

```bash
cd /Users/_iayushsharma_/Documents/final\ voice\ agent/aakashbhai/FlowAgent/apps/web
pnpm test
```

## Build for Production

Test the production build:

```bash
cd /Users/_iayushsharma_/Documents/final\ voice\ agent/aakashbhai/FlowAgent/apps/web
pnpm build
pnpm start
```

## Debug Mode

Enable React Developer Tools:
1. Install React DevTools browser extension
2. Open DevTools → Components tab
3. Inspect component hierarchy

Enable Network Logging:
1. Open DevTools → Network tab
2. Filter by "Fetch/XHR"
3. Monitor API requests

## Support

If you encounter issues:
1. Check browser console for errors
2. Verify backend is running
3. Check environment variables
4. Clear browser cache
5. Restart both frontend and backend

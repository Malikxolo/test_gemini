# Quick Reference for Kimi 2.5

**TL;DR: Everything you need to know in 5 minutes**

---

## 🎯 Mission

Build 5 frontend pages for FlowAgent:
1. API Keys Settings
2. Purchase/Pricing
3. Purchase Success
4. Download Page
5. Enhanced Settings

---

## 🔌 API Endpoints (Copy-Paste Ready)

### BYOK
```typescript
GET    /api/byok              // Get masked keys
PUT    /api/byok              // Update keys
DELETE /api/byok/:keyType     // Delete key
```

### Payments
```typescript
POST /api/payments/checkout        // Create order
POST /api/payments/verify          // Verify payment
GET  /api/payments/access-status   // Check access
GET  /api/payments/history         // Payment history
```

---

## 💳 Razorpay Integration (Copy-Paste)

```typescript
// 1. Add Script (layout.tsx)
<Script src="https://checkout.razorpay.com/v1/checkout.js" />

// 2. Create Order
const order = await fetch('/api/payments/checkout', {
  method: 'POST',
  headers: { 'Authorization': `Bearer ${token}` },
  body: JSON.stringify({ plan: 'access-tier', currency: 'INR' })
}).then(r => r.json());

// 3. Open Checkout
const rzp = new window.Razorpay({
  key: order.razorpayKeyId,
  amount: order.amount * 100,
  currency: order.currency,
  order_id: order.razorpayOrderId,
  handler: (response) => {
    // Verify payment
    fetch('/api/payments/verify', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${token}` },
      body: JSON.stringify({
        razorpayOrderId: response.razorpay_order_id,
        razorpayPaymentId: response.razorpay_payment_id,
        razorpaySignature: response.razorpay_signature
      })
    });
  }
});
rzp.open();
```

---

## 🎨 Existing Components (Use These!)

```typescript
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
```

---

## 💰 Pricing

| Currency | Amount |
|----------|--------|
| INR | ₹800 |
| USD | $10 |
| EUR | €9.50 |
| GBP | £8.50 |

---

## 📱 Pages to Create

### 1. `/settings/api-keys`
- Show 3 cards: OpenAI, Anthropic, Serper
- Each card: masked key, delete button, update button
- Modal for adding/updating keys

### 2. `/purchase`
- 4 pricing cards (INR, USD, EUR, GBP)
- Click → Razorpay checkout
- Mobile: 2 columns, Desktop: 4 columns

### 3. `/purchase/success`
- Success message
- Download button
- "Go to Dashboard" button

### 4. `/download`
- Check access first
- Show download button
- Links to docs

### 5. `/settings` (enhance)
- Add "Billing" tab
- Show access status
- Show payment history

---

## 🎨 Design System

```typescript
// Colors
Primary: #3399cc
Success: #22c55e
Error: #ef4444

// Typography
H1: text-3xl font-bold
H2: text-2xl font-semibold
Body: text-base text-gray-700
```

---

## ✅ Must-Haves

- ✅ Loading states on all buttons
- ✅ Error handling with toast
- ✅ Mobile responsive
- ✅ TypeScript typed
- ✅ Masked API keys (never show full)
- ✅ Access check before download

---

## 🚫 Don'ts

- ❌ Don't log API keys
- ❌ Don't show full keys
- ❌ Don't skip loading states
- ❌ Don't forget mobile layout
- ❌ Don't hardcode backend URL

---

## 📦 File Structure

```
src/
├── app/
│   ├── settings/api-keys/page.tsx     [NEW]
│   ├── purchase/page.tsx               [NEW]
│   ├── purchase/success/page.tsx       [NEW]
│   ├── download/page.tsx               [NEW]
│   └── settings/page.tsx               [UPDATE]
└── components/
    ├── payment/
    │   ├── RazorpayCheckout.tsx        [NEW]
    │   └── PricingCard.tsx             [NEW]
    └── settings/
        └── APIKeyCard.tsx              [NEW]
```

---

## 🧪 Test Credentials

**Razorpay Test Card:**
- Card: 4111 1111 1111 1111
- CVV: 123
- Expiry: Any future date

**Test UPI:**
- UPI ID: success@razorpay

---

**That's it! Full details in `KIMI_FRONTEND_CONTEXT.md`**

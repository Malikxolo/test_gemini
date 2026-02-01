# Razorpay Integration Guide

**Payment Gateway for India & International**

---

## 🇮🇳 Why Razorpay?

Razorpay is **the best payment gateway for India** with full international support:

### ✅ Advantages for Indian Users
- **Available in India** (Stripe is not)
- **UPI Support** - GPay, PhonePe, Paytm
- **Indian Cards** - Rupay, Visa, Mastercard
- **Net Banking** - All major banks
- **Wallets** - Paytm, MobiKwik, Freecharge
- **Local Language Support**
- **INR Pricing** - ₹800 instead of $10

### ✅ International Support
- **100+ Countries** supported
- **International Cards** - Visa, Mastercard, Amex
- **Multi-currency** - USD, EUR, GBP, INR, etc.
- **Global compliance** - PCI DSS certified

### 💰 Pricing
- **Transaction Fee**: 2% (India) | 3% (International)
- **No setup fees**
- **No annual fees**
- **Instant settlements** available

---

## 🚀 Quick Start

### 1. Create Razorpay Account

```bash
1. Go to https://dashboard.razorpay.com/signup
2. Sign up with email (free account)
3. Complete KYC (for live mode)
4. Get API keys from Settings → API Keys
```

### 2. Get API Credentials

**Test Mode** (for development):
```
Key ID: rzp_test_xxxxxxxxxxxxx
Key Secret: xxxxxxxxxxxxxxxxxxxxx
```

**Live Mode** (after KYC):
```
Key ID: rzp_live_xxxxxxxxxxxxx
Key Secret: xxxxxxxxxxxxxxxxxxxxx
```

### 3. Add to Environment

```bash
# .env
RAZORPAY_KEY_ID=rzp_test_xxxxxxxxxxxxx
RAZORPAY_KEY_SECRET=your_key_secret
RAZORPAY_WEBHOOK_SECRET=your_webhook_secret
```

---

## 💳 Payment Flow

### Standard Payment Flow (Recommended)

```
1. User clicks "Purchase $10 Access"
   ↓
2. Frontend calls POST /api/payments/checkout
   ↓
3. Backend creates Razorpay order
   ↓
4. Frontend receives order ID + Razorpay key
   ↓
5. Frontend opens Razorpay checkout modal
   ↓
6. User completes payment (UPI/Card/etc)
   ↓
7. Razorpay sends webhook to backend
   ↓
8. Backend verifies payment & grants access
   ↓
9. User gets lifetime source code access
```

### Frontend Integration

```typescript
// 1. Create order
const response = await fetch('/api/payments/checkout', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    plan: 'access-tier',
    currency: 'INR', // or USD, EUR, GBP
  }),
});

const { razorpayOrderId, amount, currency, razorpayKeyId } = await response.json();

// 2. Open Razorpay checkout
const options = {
  key: razorpayKeyId,
  amount: amount * 100, // Convert to paise/cents
  currency: currency,
  name: 'FlowAgent',
  description: 'Architecture Access - Lifetime',
  order_id: razorpayOrderId,
  handler: async function (response) {
    // 3. Verify payment on backend
    await fetch('/api/payments/verify', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        razorpayOrderId: response.razorpay_order_id,
        razorpayPaymentId: response.razorpay_payment_id,
        razorpaySignature: response.razorpay_signature,
      }),
    });

    // 4. Show success message
    alert('Payment successful! You now have lifetime access.');
  },
  prefill: {
    email: user.email,
    contact: user.phone,
  },
  theme: {
    color: '#3399cc',
  },
};

const rzp = new Razorpay(options);
rzp.open();
```

---

## 🔐 Backend Implementation

### API Endpoints

#### 1. Create Checkout
```bash
POST /api/payments/checkout
Authorization: Bearer <token>
Content-Type: application/json

{
  "plan": "access-tier",
  "currency": "INR"  # or USD, EUR, GBP
}

Response:
{
  "paymentId": "uuid",
  "razorpayOrderId": "order_xxx",
  "amount": 800,
  "currency": "INR",
  "razorpayKeyId": "rzp_test_xxx",
  "userEmail": "user@example.com",
  "userName": "username"
}
```

#### 2. Verify Payment
```bash
POST /api/payments/verify
Authorization: Bearer <token>
Content-Type: application/json

{
  "razorpayOrderId": "order_xxx",
  "razorpayPaymentId": "pay_xxx",
  "razorpaySignature": "signature_xxx"
}

Response:
{
  "success": true,
  "message": "Payment verified and access granted successfully"
}
```

#### 3. Webhook Handler
```bash
POST /webhooks/razorpay
x-razorpay-signature: <webhook_signature>
Content-Type: application/json

# Razorpay automatically sends payment events
# Backend verifies signature and grants access
```

---

## 💰 Pricing by Currency

FlowAgent supports multiple currencies:

| Currency | Amount | Symbol |
|----------|--------|--------|
| **USD** | $10.00 | $ |
| **INR** | ₹800.00 | ₹ |
| **EUR** | €9.50 | € |
| **GBP** | £8.50 | £ |

**Note**: Prices are approximate and may vary based on exchange rates.

---

## 🔒 Security

### Webhook Signature Verification

```typescript
// Backend automatically verifies webhook signatures
const expectedSignature = crypto
  .createHmac('sha256', RAZORPAY_WEBHOOK_SECRET)
  .update(webhookBody)
  .digest('hex');

if (expectedSignature !== razorpaySignature) {
  throw new Error('Invalid signature');
}
```

### Payment Signature Verification

```typescript
// Verify payment after user completes checkout
const generatedSignature = crypto
  .createHmac('sha256', RAZORPAY_KEY_SECRET)
  .update(`${orderId}|${paymentId}`)
  .digest('hex');

if (generatedSignature !== razorpaySignature) {
  throw new Error('Invalid payment signature');
}
```

---

## 📊 Payment Methods Supported

### India 🇮🇳
- ✅ **UPI** - Google Pay, PhonePe, Paytm, BHIM
- ✅ **Cards** - Credit, Debit, Rupay
- ✅ **Net Banking** - All major banks
- ✅ **Wallets** - Paytm, MobiKwik, Freecharge, Ola Money
- ✅ **EMI** - Cardless EMI, Card EMI
- ✅ **Pay Later** - LazyPay, Simpl, etc.

### International 🌍
- ✅ **Credit Cards** - Visa, Mastercard, Amex
- ✅ **Debit Cards** - Visa, Mastercard
- ✅ **Apple Pay** (coming soon)
- ✅ **Google Pay** (international)

---

## 🧪 Testing

### Test Cards (India)

**Success**:
```
Card: 4111 1111 1111 1111
CVV: 123
Expiry: Any future date
```

**Failure**:
```
Card: 4000 0000 0000 0002
CVV: 123
Expiry: Any future date
```

### Test UPI

```
UPI ID: success@razorpay
PIN: Any 4-6 digits
```

### Test Net Banking

```
Bank: Any bank
Status: Select "Success"
```

---

## 🔧 Database Schema

```sql
-- Add Razorpay fields to access_payments table
ALTER TABLE access_payments
  ADD COLUMN razorpay_order_id TEXT UNIQUE,
  ADD COLUMN razorpay_payment_id TEXT UNIQUE;

CREATE INDEX idx_razorpay_order ON access_payments(razorpay_order_id);
```

---

## 📱 Mobile Integration

Razorpay works seamlessly on mobile:

- ✅ **Responsive checkout** - Mobile optimized
- ✅ **UPI deep links** - Direct app opening
- ✅ **Mobile wallets** - One-tap payment
- ✅ **Native SDKs** - iOS & Android

---

## 🌐 International Payments

### Supported Countries (100+)

**Asia**: India, Singapore, Malaysia, UAE, Saudi Arabia, etc.
**Europe**: UK, Germany, France, Spain, Netherlands, etc.
**Americas**: USA, Canada, Brazil, Mexico, etc.
**Others**: Australia, New Zealand, South Africa, etc.

### Currency Support

- USD (US Dollar)
- EUR (Euro)
- GBP (British Pound)
- INR (Indian Rupee)
- AED (UAE Dirham)
- SGD (Singapore Dollar)
- MYR (Malaysian Ringgit)
- And 100+ more currencies

---

## 💡 Best Practices

### 1. **Always Verify Signatures**
```typescript
// Never trust payment data without verification
verifySignature(webhookBody, signature, secret);
```

### 2. **Handle Webhooks Idempotently**
```typescript
// Check if payment already processed
if (payment.status === 'completed') {
  return { received: true };
}
```

### 3. **Log All Transactions**
```typescript
// Keep audit trail
console.log('Payment received:', {
  orderId,
  paymentId,
  amount,
  status,
});
```

### 4. **Handle Failures Gracefully**
```typescript
// Show user-friendly error messages
if (payment.status === 'failed') {
  return 'Payment failed. Please try again.';
}
```

---

## 🐛 Troubleshooting

### Issue: "Invalid API Key"
**Solution**: Check that `RAZORPAY_KEY_ID` and `RAZORPAY_KEY_SECRET` are correct

### Issue: "Webhook signature mismatch"
**Solution**: Ensure `RAZORPAY_WEBHOOK_SECRET` matches Razorpay dashboard

### Issue: "Payment not getting verified"
**Solution**: Check webhook URL is publicly accessible and returns 200 OK

### Issue: "Amount mismatch"
**Solution**: Razorpay expects amount in smallest currency unit (paise for INR, cents for USD)

---

## 📞 Razorpay Support

- **Dashboard**: https://dashboard.razorpay.com
- **Docs**: https://razorpay.com/docs/
- **Support**: support@razorpay.com
- **Status**: https://status.razorpay.com
- **Community**: https://discuss.razorpay.com

---

## ✅ Migration from Stripe

If you were using Stripe before:

1. ✅ Database schema compatible (both use similar fields)
2. ✅ Webhook flow nearly identical
3. ✅ Payment verification similar
4. ✅ Frontend integration straightforward

**Just switch the payment gateway and you're done!**

---

## 🎯 Go Live Checklist

### Before Launch:

- [ ] Complete Razorpay KYC
- [ ] Switch to live API keys
- [ ] Set up webhook URL
- [ ] Test live payment (₹1 test)
- [ ] Enable auto-capture
- [ ] Set up email notifications
- [ ] Configure refund policy
- [ ] Add customer support email

### Post Launch:

- [ ] Monitor webhook failures
- [ ] Track payment success rate
- [ ] Review transaction logs daily
- [ ] Set up Razorpay analytics
- [ ] Enable fraud detection

---

## 📈 Analytics & Reporting

Razorpay Dashboard provides:

- 📊 **Real-time analytics** - Revenue, success rate, failures
- 📧 **Email reports** - Daily/weekly summaries
- 🔍 **Payment search** - Find any transaction
- 📱 **Mobile app** - Monitor on the go
- 📥 **Export data** - CSV/Excel downloads

---

## 🌟 Razorpay vs Stripe

| Feature | Razorpay | Stripe |
|---------|----------|--------|
| **India** | ✅ Yes | ❌ No |
| **UPI** | ✅ Yes | ❌ No |
| **Setup** | 🟢 Easy | 🟢 Easy |
| **Fees (India)** | 2% | N/A |
| **Fees (Intl)** | 3% | 2.9% |
| **KYC** | Required | Required |
| **Settlement** | 1-2 days | 2-7 days |

**Winner**: **Razorpay** (for India-based businesses)

---

## 🚀 Quick Commands

### Setup Razorpay
```bash
# 1. Set environment variables
export RAZORPAY_KEY_ID=rzp_test_xxx
export RAZORPAY_KEY_SECRET=xxx
export RAZORPAY_WEBHOOK_SECRET=xxx

# 2. Test API connection
curl -u $RAZORPAY_KEY_ID:$RAZORPAY_KEY_SECRET \
  https://api.razorpay.com/v1/payments

# 3. Create test order
curl -X POST https://api.razorpay.com/v1/orders \
  -u $RAZORPAY_KEY_ID:$RAZORPAY_KEY_SECRET \
  -d amount=80000 \
  -d currency=INR \
  -d receipt=test_001
```

---

**Last Updated**: January 31, 2026
**Status**: Production Ready ✅
**Support**: Available for India & International

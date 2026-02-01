# API Request/Response Examples

**Exact examples of all API calls with real responses**

---

## 🔑 BYOK API Examples

### 1. Get API Keys (Masked)

**Request:**
```http
GET http://localhost:8787/api/byok
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Response (200 OK):**
```json
{
  "openaiApiKey": "sk-....a1b2",
  "anthropicApiKey": "sk-ant-....x9y8",
  "serperApiKey": "****....m5n6"
}
```

**Response (when no keys set):**
```json
{
  "openaiApiKey": null,
  "anthropicApiKey": null,
  "serperApiKey": null
}
```

---

### 2. Update API Keys

**Request:**
```http
PUT http://localhost:8787/api/byok
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json

{
  "openaiApiKey": "sk-proj-abc123def456ghi789jkl012mno345pqr678stu901vwx234yz",
  "anthropicApiKey": "sk-ant-api03-xyz789abc012def345ghi678jkl901mno234pqr567stu890vwx123yz"
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "API keys updated successfully"
}
```

**Error Response (400 Bad Request):**
```json
{
  "error": "Invalid API key format"
}
```

---

### 3. Delete API Key

**Request:**
```http
DELETE http://localhost:8787/api/byok/openai
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "openai API key deleted successfully"
}
```

**Error Response (400 Bad Request):**
```json
{
  "error": "Invalid key type"
}
```

---

## 💳 Payment API Examples

### 1. Create Checkout Session

**Request (India - INR):**
```http
POST http://localhost:8787/api/payments/checkout
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json

{
  "plan": "access-tier",
  "currency": "INR"
}
```

**Response (201 Created):**
```json
{
  "paymentId": "550e8400-e29b-41d4-a716-446655440000",
  "razorpayOrderId": "order_NhZbPmJqKbXPQx",
  "amount": 800,
  "currency": "INR",
  "razorpayKeyId": "rzp_test_1DP5mmOlF5G5ag",
  "userEmail": "user@example.com",
  "userName": "johndoe"
}
```

**Request (USA - USD):**
```http
POST http://localhost:8787/api/payments/checkout
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json

{
  "plan": "access-tier",
  "currency": "USD"
}
```

**Response (201 Created):**
```json
{
  "paymentId": "660f9511-f3ac-52e5-b827-557766551111",
  "razorpayOrderId": "order_OiAcQnKlLcYRSy",
  "amount": 10,
  "currency": "USD",
  "razorpayKeyId": "rzp_test_1DP5mmOlF5G5ag",
  "userEmail": "user@example.com",
  "userName": "johndoe"
}
```

**Error Response (400 Bad Request - Already has access):**
```json
{
  "error": "You already have access to all features"
}
```

---

### 2. Verify Payment

**Request:**
```http
POST http://localhost:8787/api/payments/verify
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json

{
  "razorpayOrderId": "order_NhZbPmJqKbXPQx",
  "razorpayPaymentId": "pay_NhZcDmJqKbXRTy",
  "razorpaySignature": "9ef12de6c5b8dc49e9a5e5eaea5c6e8f12345678abcdef"
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "Payment verified and access granted successfully"
}
```

**Error Response (400 Bad Request):**
```json
{
  "error": "Invalid payment signature"
}
```

**Error Response (404 Not Found):**
```json
{
  "error": "Payment not found"
}
```

---

### 3. Check Access Status

**Request:**
```http
GET http://localhost:8787/api/payments/access-status
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Response (Has Access):**
```json
{
  "hasAccess": true,
  "hasArchitectureAccess": true,
  "hasDownloadAccess": true,
  "purchasedAt": "2026-01-31T12:34:56.789Z",
  "expiresAt": null
}
```

**Response (No Access):**
```json
{
  "hasAccess": false,
  "hasArchitectureAccess": false,
  "hasDownloadAccess": false,
  "purchasedAt": null,
  "expiresAt": null
}
```

---

### 4. Get Payment History

**Request:**
```http
GET http://localhost:8787/api/payments/history
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Response (200 OK):**
```json
{
  "payments": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "userId": "770f9511-f3ac-52e5-b827-557766551111",
      "amount": "800.00",
      "currency": "INR",
      "razorpayOrderId": "order_NhZbPmJqKbXPQx",
      "razorpayPaymentId": "pay_NhZcDmJqKbXRTy",
      "paymentMethod": "upi",
      "status": "completed",
      "grantsArchitectureAccess": true,
      "grantsDownloadAccess": true,
      "grantsDuration": "lifetime",
      "createdAt": "2026-01-31T12:30:00.000Z",
      "completedAt": "2026-01-31T12:34:56.789Z"
    },
    {
      "id": "660f9511-f3ac-52e5-b827-557766551112",
      "userId": "770f9511-f3ac-52e5-b827-557766551111",
      "amount": "10.00",
      "currency": "USD",
      "razorpayOrderId": "order_OiAcQnKlLcYRSy",
      "razorpayPaymentId": null,
      "paymentMethod": null,
      "status": "pending",
      "grantsArchitectureAccess": true,
      "grantsDownloadAccess": true,
      "grantsDuration": "lifetime",
      "createdAt": "2026-01-30T08:15:00.000Z",
      "completedAt": null
    }
  ]
}
```

**Response (Empty History):**
```json
{
  "payments": []
}
```

---

## 👤 Auth API Examples (Reference)

### Get Current User

**Request:**
```http
GET http://localhost:8787/api/auth/me
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Response (200 OK):**
```json
{
  "user": {
    "id": "770f9511-f3ac-52e5-b827-557766551111",
    "email": "user@example.com",
    "username": "johndoe",
    "displayName": "John Doe",
    "subscriptionTier": "free",
    "hasArchitectureAccess": true,
    "hasDownloadAccess": true,
    "createdAt": "2026-01-15T10:00:00.000Z",
    "lastLoginAt": "2026-01-31T09:00:00.000Z"
  }
}
```

**Error Response (401 Unauthorized):**
```json
{
  "error": "Not authenticated"
}
```

---

## 🎯 Complete Flow Example

### User Journey: Purchase to Download

**Step 1: Check if already has access**
```http
GET /api/payments/access-status
→ Response: { "hasAccess": false }
```

**Step 2: Create checkout**
```http
POST /api/payments/checkout
Body: { "plan": "access-tier", "currency": "INR" }
→ Response: { "razorpayOrderId": "order_xxx", "amount": 800, ... }
```

**Step 3: Open Razorpay (Frontend)**
```javascript
// Razorpay SDK handles payment
const rzp = new Razorpay({ /* options */ });
rzp.open();
// User completes payment in Razorpay modal
```

**Step 4: Razorpay callback (Frontend)**
```javascript
handler: (response) => {
  // Got: razorpay_order_id, razorpay_payment_id, razorpay_signature
}
```

**Step 5: Verify payment**
```http
POST /api/payments/verify
Body: {
  "razorpayOrderId": "order_xxx",
  "razorpayPaymentId": "pay_yyy",
  "razorpaySignature": "signature_zzz"
}
→ Response: { "success": true, "message": "..." }
```

**Step 6: Check access again**
```http
GET /api/payments/access-status
→ Response: { "hasAccess": true, "hasDownloadAccess": true }
```

**Step 7: Download source code**
```http
GET /api/download/source
→ Downloads flowagent-v1.0.0.zip
```

---

## 🧪 Testing Examples

### Test with cURL

**Get API Keys:**
```bash
curl -X GET http://localhost:8787/api/byok \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Create Checkout:**
```bash
curl -X POST http://localhost:8787/api/payments/checkout \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"plan":"access-tier","currency":"INR"}'
```

**Check Access:**
```bash
curl -X GET http://localhost:8787/api/payments/access-status \
  -H "Authorization: Bearer YOUR_TOKEN"
```

---

## 💡 Frontend Usage Examples

### Fetch API Keys
```typescript
const fetchAPIKeys = async () => {
  const token = getToken();
  const response = await fetch('http://localhost:8787/api/byok', {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  });

  if (!response.ok) {
    throw new Error('Failed to fetch API keys');
  }

  const data = await response.json();
  // data = { openaiApiKey: "sk-....xxxx", ... }
  return data;
};
```

### Create Payment
```typescript
const createPayment = async (currency: 'INR' | 'USD' | 'EUR' | 'GBP') => {
  const token = getToken();
  const response = await fetch('http://localhost:8787/api/payments/checkout', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      plan: 'access-tier',
      currency: currency
    })
  });

  if (!response.ok) {
    throw new Error('Failed to create payment');
  }

  const data = await response.json();
  // data = { razorpayOrderId: "order_xxx", amount: 800, ... }
  return data;
};
```

### Verify Payment
```typescript
const verifyPayment = async (
  orderId: string,
  paymentId: string,
  signature: string
) => {
  const token = getToken();
  const response = await fetch('http://localhost:8787/api/payments/verify', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      razorpayOrderId: orderId,
      razorpayPaymentId: paymentId,
      razorpaySignature: signature
    })
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Payment verification failed');
  }

  const data = await response.json();
  // data = { success: true, message: "..." }
  return data;
};
```

---

## 🔍 Error Codes Reference

| Status | Meaning | Example |
|--------|---------|---------|
| 200 | Success | Request completed successfully |
| 201 | Created | Payment order created |
| 400 | Bad Request | Invalid data, already has access |
| 401 | Unauthorized | Missing or invalid token |
| 404 | Not Found | Payment not found |
| 500 | Server Error | Internal server error |

---

**All APIs are live and ready to use! Just copy-paste these examples.** 🚀

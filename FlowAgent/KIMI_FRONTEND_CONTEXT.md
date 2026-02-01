# FlowAgent Frontend Development Context for Kimi 2.5

**Complete specification for building the FlowAgent frontend UI**

---

## 📋 Project Overview

**Project Name**: FlowAgent
**Tech Stack**: Next.js 14 (App Router), TypeScript, Tailwind CSS, Radix UI
**Purpose**: AI Agent platform with BYOK (Bring Your Own Key) + $10 architecture access

---

## 🎯 What Needs to Be Built

You need to create **5 main pages** for the FlowAgent frontend:

### 1. **Settings Page - BYOK (Bring Your Own Key)**
   - Path: `/settings/api-keys` or `/settings/byok`
   - Purpose: Users can add/manage their own API keys

### 2. **Payment/Checkout Page**
   - Path: `/purchase` or `/pricing`
   - Purpose: Users can purchase $10 (₹800) architecture access

### 3. **Payment Success Page**
   - Path: `/purchase/success`
   - Purpose: Show success message and download button

### 4. **Download Page**
   - Path: `/download`
   - Purpose: Allow users to download source code after payment

### 5. **Settings/Account Page Enhancement**
   - Path: `/settings`
   - Purpose: Show access status, payment history, account details

---

## 🏗️ Project Structure

```
apps/web/
├── src/
│   ├── app/
│   │   ├── (auth)/
│   │   │   ├── login/page.tsx          ✅ EXISTS
│   │   │   └── signup/page.tsx         ✅ EXISTS
│   │   ├── dashboard/page.tsx          ✅ EXISTS
│   │   ├── settings/
│   │   │   ├── page.tsx                ⚠️ NEEDS UPDATE
│   │   │   └── api-keys/page.tsx       ❌ CREATE NEW
│   │   ├── purchase/
│   │   │   ├── page.tsx                ❌ CREATE NEW
│   │   │   └── success/page.tsx        ❌ CREATE NEW
│   │   └── download/page.tsx           ❌ CREATE NEW
│   ├── components/
│   │   ├── ui/                         ✅ EXISTS (Radix UI)
│   │   │   ├── button.tsx
│   │   │   ├── input.tsx
│   │   │   ├── label.tsx
│   │   │   ├── card.tsx
│   │   │   └── ... (other Radix components)
│   │   ├── payment/                    ❌ CREATE NEW
│   │   │   ├── RazorpayCheckout.tsx
│   │   │   └── PricingCard.tsx
│   │   ├── settings/                   ❌ CREATE NEW
│   │   │   ├── APIKeyForm.tsx
│   │   │   └── AccessStatusCard.tsx
│   │   └── layout/
│   │       └── DashboardLayout.tsx     ⚠️ MAY NEED UPDATE
│   └── lib/
│       └── api.ts                      ✅ EXISTS
└── public/
    └── razorpay-logo.svg               ❌ ADD IF NEEDED
```

---

## 🔌 API Integration (Backend is 100% Ready)

### Base URL
```typescript
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8787';
```

### Available Endpoints

#### 1. **BYOK API** (`/api/byok`)

**Get API Keys (Masked)**
```typescript
GET /api/byok
Headers: { Authorization: `Bearer ${token}` }

Response:
{
  "openaiApiKey": "sk-....xxxx",      // Masked
  "anthropicApiKey": "sk-ant-....xxxx", // Masked
  "serperApiKey": "****....xxxx"       // Masked
}
```

**Update API Keys**
```typescript
PUT /api/byok
Headers: { Authorization: `Bearer ${token}` }
Body: {
  "openaiApiKey": "sk-proj-...",       // Optional
  "anthropicApiKey": "sk-ant-...",     // Optional
  "serperApiKey": "..."                // Optional
}

Response:
{
  "success": true,
  "message": "API keys updated successfully"
}
```

**Delete API Key**
```typescript
DELETE /api/byok/:keyType
Headers: { Authorization: `Bearer ${token}` }
// keyType: openai | anthropic | serper

Response:
{
  "success": true,
  "message": "openai API key deleted successfully"
}
```

#### 2. **Payment API** (`/api/payments`)

**Create Checkout Session**
```typescript
POST /api/payments/checkout
Headers: { Authorization: `Bearer ${token}` }
Body: {
  "plan": "access-tier",
  "currency": "INR"  // or USD, EUR, GBP
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

**Verify Payment**
```typescript
POST /api/payments/verify
Headers: { Authorization: `Bearer ${token}` }
Body: {
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

**Get Access Status**
```typescript
GET /api/payments/access-status
Headers: { Authorization: `Bearer ${token}` }

Response:
{
  "hasAccess": true,
  "hasArchitectureAccess": true,
  "hasDownloadAccess": true,
  "purchasedAt": "2026-01-31T12:00:00Z",
  "expiresAt": null  // null = lifetime
}
```

**Get Payment History**
```typescript
GET /api/payments/history
Headers: { Authorization: `Bearer ${token}` }

Response:
{
  "payments": [
    {
      "id": "uuid",
      "amount": "800.00",
      "currency": "INR",
      "status": "completed",
      "razorpayOrderId": "order_xxx",
      "createdAt": "2026-01-31T12:00:00Z",
      "completedAt": "2026-01-31T12:05:00Z"
    }
  ]
}
```

#### 3. **Auth API** (Already exists - reference)

```typescript
GET /api/auth/me
Headers: { Authorization: `Bearer ${token}` }

Response:
{
  "user": {
    "id": "uuid",
    "email": "user@example.com",
    "username": "username",
    "displayName": "Display Name",
    "subscriptionTier": "free",
    "hasArchitectureAccess": false,
    "hasDownloadAccess": false
  }
}
```

---

## 💳 Razorpay Integration

### Setup (Frontend)

**1. Add Razorpay SDK**
```html
<!-- Add to app/layout.tsx or specific pages -->
<Script src="https://checkout.razorpay.com/v1/checkout.js" />
```

**2. Payment Flow**
```typescript
// Step 1: Create order
const createOrder = async (currency: 'INR' | 'USD' | 'EUR' | 'GBP') => {
  const response = await fetch('/api/payments/checkout', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      plan: 'access-tier',
      currency: currency,
    }),
  });

  return await response.json();
};

// Step 2: Open Razorpay checkout
const openCheckout = (orderData) => {
  const options = {
    key: orderData.razorpayKeyId,
    amount: orderData.amount * 100, // Convert to paise/cents
    currency: orderData.currency,
    name: 'FlowAgent',
    description: 'Architecture Access - Lifetime',
    order_id: orderData.razorpayOrderId,
    handler: async function (response) {
      // Step 3: Verify payment
      await verifyPayment(response);
    },
    prefill: {
      email: orderData.userEmail,
      name: orderData.userName,
    },
    theme: {
      color: '#3399cc',
    },
    modal: {
      ondismiss: function() {
        // User closed the checkout
        console.log('Checkout dismissed');
      }
    }
  };

  const rzp = new window.Razorpay(options);
  rzp.open();
};

// Step 3: Verify payment
const verifyPayment = async (response) => {
  const verification = await fetch('/api/payments/verify', {
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

  if (verification.ok) {
    // Redirect to success page
    router.push('/purchase/success');
  } else {
    // Show error
    alert('Payment verification failed');
  }
};
```

---

## 🎨 Design System & UI Components

### Colors (Tailwind Classes)
```typescript
// Primary: Blue
primary: 'bg-blue-600 hover:bg-blue-700'
primaryText: 'text-blue-600'

// Success: Green
success: 'bg-green-600 text-white'
successText: 'text-green-600'

// Error: Red
error: 'bg-red-600 text-white'
errorText: 'text-red-600'

// Warning: Yellow
warning: 'bg-yellow-500 text-black'
warningText: 'text-yellow-600'

// Backgrounds
bgPrimary: 'bg-white dark:bg-gray-900'
bgSecondary: 'bg-gray-50 dark:bg-gray-800'
```

### Typography
```typescript
// Headings
h1: 'text-3xl font-bold tracking-tight text-gray-900'
h2: 'text-2xl font-semibold text-gray-900'
h3: 'text-xl font-medium text-gray-900'

// Body
body: 'text-base text-gray-700'
bodySmall: 'text-sm text-gray-600'
caption: 'text-xs text-gray-500'
```

### Existing Components (Radix UI)
```typescript
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
```

---

## 📱 Page-by-Page Specifications

### PAGE 1: API Keys Settings (`/settings/api-keys`)

**Purpose**: Allow users to add/manage their BYOK API keys

**Layout**:
```
┌─────────────────────────────────────────┐
│  ← Back to Settings                     │
│                                         │
│  API Keys                               │
│  Manage your AI provider API keys      │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │ 🔑 OpenAI API Key                 │ │
│  │                                   │ │
│  │ Current: sk-....xxxx  [Delete]   │ │
│  │                                   │ │
│  │ [Update OpenAI Key]               │ │
│  └───────────────────────────────────┘ │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │ 🤖 Anthropic API Key              │ │
│  │                                   │ │
│  │ Not configured                    │ │
│  │                                   │ │
│  │ [Add Anthropic Key]               │ │
│  └───────────────────────────────────┘ │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │ 🔍 Serper API Key (Web Search)    │ │
│  │                                   │ │
│  │ Current: ****....yyyy  [Delete]   │ │
│  │                                   │ │
│  │ [Update Serper Key]               │ │
│  └───────────────────────────────────┘ │
│                                         │
│  ℹ️ Benefits of BYOK:                  │
│  • Pay OpenAI/Anthropic directly       │
│  • No platform markup                  │
│  • Full control over usage             │
│  • Higher rate limits                  │
└─────────────────────────────────────────┘
```

**Features**:
- Show masked keys for existing keys
- Delete button for each key
- Update/Add button that opens modal
- Modal with:
  - Input field for new key
  - Validation (starts with sk- for OpenAI, sk-ant- for Anthropic)
  - Save & Cancel buttons
  - Security note: "Keys are encrypted and stored securely"

**State Management**:
```typescript
const [keys, setKeys] = useState({
  openai: null,
  anthropic: null,
  serper: null,
});
const [editingKey, setEditingKey] = useState<'openai' | 'anthropic' | 'serper' | null>(null);
const [newKeyValue, setNewKeyValue] = useState('');
const [loading, setLoading] = useState(false);
```

**Functions Needed**:
```typescript
const fetchKeys = async () => { /* GET /api/byok */ };
const updateKey = async (type, value) => { /* PUT /api/byok */ };
const deleteKey = async (type) => { /* DELETE /api/byok/:type */ };
```

---

### PAGE 2: Purchase/Pricing (`/purchase`)

**Purpose**: Display pricing and handle payment

**Layout**:
```
┌─────────────────────────────────────────────────┐
│  FlowAgent Architecture Access                  │
│                                                  │
│  Get Lifetime Access to:                        │
│  ✅ Full source code                            │
│  ✅ Architecture documentation                  │
│  ✅ Self-hosting rights                         │
│  ✅ White-label capabilities                    │
│  ✅ Lifetime updates                            │
│                                                  │
│  ┌────────────────┐  ┌────────────────┐        │
│  │   🇮🇳 India    │  │   🇺🇸 USA      │        │
│  │                │  │                │        │
│  │   ₹800         │  │   $10          │        │
│  │   One-time     │  │   One-time     │        │
│  │                │  │                │        │
│  │  [Purchase]    │  │  [Purchase]    │        │
│  └────────────────┘  └────────────────┘        │
│                                                  │
│  ┌────────────────┐  ┌────────────────┐        │
│  │   🇪🇺 Europe   │  │   🇬🇧 UK       │        │
│  │                │  │                │        │
│  │   €9.50        │  │   £8.50        │        │
│  │   One-time     │  │   One-time     │        │
│  │                │  │                │        │
│  │  [Purchase]    │  │  [Purchase]    │        │
│  └────────────────┘  └────────────────┘        │
│                                                  │
│  💳 Payment Methods Available:                  │
│  India: UPI, Cards, Net Banking, Wallets       │
│  International: Visa, Mastercard, Amex         │
│                                                  │
│  🔒 Secure payment powered by Razorpay         │
└─────────────────────────────────────────────────┘
```

**Features**:
- 4 pricing cards (INR, USD, EUR, GBP)
- Highlight recommended option (INR for India)
- Click to purchase opens Razorpay checkout
- Show loading state during checkout creation
- Mobile responsive (2 columns on mobile)

**State**:
```typescript
const [selectedCurrency, setSelectedCurrency] = useState<'INR' | 'USD' | 'EUR' | 'GBP'>('INR');
const [loading, setLoading] = useState(false);
const [hasAccess, setHasAccess] = useState(false);
```

**Functions**:
```typescript
const checkAccess = async () => { /* GET /api/payments/access-status */ };
const handlePurchase = async (currency) => {
  // 1. Create order
  const order = await createOrder(currency);
  // 2. Open Razorpay
  openCheckout(order);
};
```

---

### PAGE 3: Purchase Success (`/purchase/success`)

**Purpose**: Show success message after payment

**Layout**:
```
┌─────────────────────────────────────────┐
│                                         │
│            ✅                           │
│      Payment Successful!                │
│                                         │
│  You now have lifetime access to        │
│  FlowAgent architecture!                │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │ What's Next:                      │ │
│  │                                   │ │
│  │ 1. Download source code           │ │
│  │ 2. Review architecture docs       │ │
│  │ 3. Deploy on your infrastructure  │ │
│  │ 4. Customize as needed            │ │
│  └───────────────────────────────────┘ │
│                                         │
│  [Download Source Code]                 │
│  [View Documentation]                   │
│                                         │
│  📧 A receipt has been sent to your     │
│      email address                      │
│                                         │
│  [Go to Dashboard]                      │
└─────────────────────────────────────────┘
```

**Features**:
- Success animation (checkmark)
- Clear next steps
- Download button
- Link to documentation
- Email confirmation note

---

### PAGE 4: Download (`/download`)

**Purpose**: Allow users to download source code

**Layout**:
```
┌─────────────────────────────────────────┐
│  Download FlowAgent Source Code         │
│                                         │
│  ✅ Access Verified                     │
│  Purchased on: Jan 31, 2026            │
│  License: Commercial Use Allowed        │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │ 📦 FlowAgent v1.0.0               │ │
│  │                                   │ │
│  │ Size: ~50 MB (compressed)         │ │
│  │ Format: .zip                      │ │
│  │                                   │ │
│  │ Includes:                         │ │
│  │ • Full source code                │ │
│  │ • Database schemas                │ │
│  │ • Docker configs                  │ │
│  │ • Documentation                   │ │
│  │ • Deployment guides               │ │
│  │                                   │ │
│  │ [Download ZIP]                    │ │
│  └───────────────────────────────────┘ │
│                                         │
│  📚 Documentation                       │
│  • Architecture Guide                   │
│  • Deployment Guide                     │
│  • API Reference                        │
│  • Database Schema                      │
│                                         │
│  💬 Need Help?                          │
│  Join our Discord community             │
│  [Join Discord]                         │
└─────────────────────────────────────────┘
```

**Features**:
- Verify access before showing download
- Show purchase date
- Download button triggers download
- Links to documentation
- Community support link

**Access Check**:
```typescript
useEffect(() => {
  const checkAccess = async () => {
    const status = await fetch('/api/payments/access-status', {
      headers: { Authorization: `Bearer ${token}` }
    });
    const data = await status.json();

    if (!data.hasDownloadAccess) {
      router.push('/purchase');
    }
  };
  checkAccess();
}, []);
```

---

### PAGE 5: Settings Enhancement (`/settings`)

**Purpose**: Show account info, access status, payment history

**Layout**:
```
┌─────────────────────────────────────────┐
│  Account Settings                       │
│                                         │
│  Tabs: [General] [API Keys] [Billing]  │
│                                         │
│  === General Tab ===                    │
│  Email: user@example.com                │
│  Username: username                     │
│  Display Name: User Name                │
│  [Edit Profile]                         │
│                                         │
│  === API Keys Tab ===                   │
│  (Link to /settings/api-keys)           │
│                                         │
│  === Billing Tab ===                    │
│  ┌───────────────────────────────────┐ │
│  │ 🎯 Architecture Access            │ │
│  │                                   │ │
│  │ Status: ✅ Active                 │ │
│  │ Purchased: Jan 31, 2026           │ │
│  │ Expires: Never (Lifetime)         │ │
│  │                                   │ │
│  │ [Download Source Code]            │ │
│  └───────────────────────────────────┘ │
│                                         │
│  Payment History                        │
│  ┌───────────────────────────────────┐ │
│  │ Jan 31, 2026  ₹800  Completed     │ │
│  │ Architecture Access - Lifetime    │ │
│  │ [View Receipt]                    │ │
│  └───────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

**Features**:
- Tabs for different settings sections
- Access status card
- Payment history list
- Download button if has access

---

## 🎨 Component Examples

### Example 1: API Key Card Component

```typescript
// components/settings/APIKeyCard.tsx
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

interface APIKeyCardProps {
  name: string;
  icon: string;
  currentKey: string | null;
  onUpdate: () => void;
  onDelete: () => void;
  description?: string;
}

export function APIKeyCard({
  name,
  icon,
  currentKey,
  onUpdate,
  onDelete,
  description,
}: APIKeyCardProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <span>{icon}</span>
          {name}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {currentKey ? (
            <>
              <div className="flex items-center justify-between">
                <code className="text-sm bg-gray-100 px-2 py-1 rounded">
                  {currentKey}
                </code>
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={onDelete}
                >
                  Delete
                </Button>
              </div>
              <Button onClick={onUpdate} className="w-full">
                Update Key
              </Button>
            </>
          ) : (
            <>
              <Badge variant="secondary">Not configured</Badge>
              <Button onClick={onUpdate} className="w-full">
                Add Key
              </Button>
            </>
          )}
          {description && (
            <p className="text-sm text-gray-600">{description}</p>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
```

### Example 2: Pricing Card Component

```typescript
// components/payment/PricingCard.tsx
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

interface PricingCardProps {
  country: string;
  flag: string;
  amount: number;
  currency: 'INR' | 'USD' | 'EUR' | 'GBP';
  recommended?: boolean;
  onPurchase: (currency: 'INR' | 'USD' | 'EUR' | 'GBP') => void;
  loading?: boolean;
}

export function PricingCard({
  country,
  flag,
  amount,
  currency,
  recommended = false,
  onPurchase,
  loading = false,
}: PricingCardProps) {
  const formatAmount = (amt: number, curr: string) => {
    const symbols = { INR: '₹', USD: '$', EUR: '€', GBP: '£' };
    return `${symbols[curr]}${amt}`;
  };

  return (
    <Card className={recommended ? 'border-blue-500 border-2' : ''}>
      {recommended && (
        <div className="bg-blue-500 text-white text-center py-1 text-sm font-medium">
          Recommended
        </div>
      )}
      <CardHeader>
        <CardTitle className="flex items-center justify-center gap-2">
          <span className="text-3xl">{flag}</span>
          {country}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="text-center">
          <div className="text-4xl font-bold">
            {formatAmount(amount, currency)}
          </div>
          <div className="text-sm text-gray-600">One-time payment</div>
        </div>
        <Button
          onClick={() => onPurchase(currency)}
          disabled={loading}
          className="w-full"
        >
          {loading ? 'Processing...' : 'Purchase'}
        </Button>
      </CardContent>
    </Card>
  );
}
```

### Example 3: Razorpay Checkout Component

```typescript
// components/payment/RazorpayCheckout.tsx
'use client';

import { useEffect } from 'react';
import Script from 'next/script';

interface RazorpayCheckoutProps {
  orderId: string;
  amount: number;
  currency: string;
  keyId: string;
  userEmail: string;
  userName: string;
  onSuccess: (response: any) => void;
  onFailure?: (error: any) => void;
}

export function RazorpayCheckout({
  orderId,
  amount,
  currency,
  keyId,
  userEmail,
  userName,
  onSuccess,
  onFailure,
}: RazorpayCheckoutProps) {
  const openCheckout = () => {
    const options = {
      key: keyId,
      amount: amount * 100,
      currency: currency,
      name: 'FlowAgent',
      description: 'Architecture Access - Lifetime',
      order_id: orderId,
      handler: function (response: any) {
        onSuccess(response);
      },
      prefill: {
        email: userEmail,
        name: userName,
      },
      theme: {
        color: '#3399cc',
      },
      modal: {
        ondismiss: function() {
          if (onFailure) {
            onFailure({ message: 'Payment cancelled by user' });
          }
        }
      }
    };

    // @ts-ignore
    const rzp = new window.Razorpay(options);
    rzp.open();
  };

  useEffect(() => {
    if (orderId && keyId) {
      openCheckout();
    }
  }, [orderId, keyId]);

  return (
    <>
      <Script
        src="https://checkout.razorpay.com/v1/checkout.js"
        strategy="lazyOnload"
      />
    </>
  );
}
```

---

## 🔐 Authentication & State Management

### Get User Token

```typescript
// Assuming you have auth context or cookies
import { getCookie } from 'cookies-next';

const getToken = () => {
  return getCookie('auth_token') || localStorage.getItem('auth_token');
};

// Or use context
import { useAuth } from '@/contexts/AuthContext';

const { token, user } = useAuth();
```

### API Client Usage

```typescript
import { api } from '@/lib/api';

// Example usage
const fetchKeys = async () => {
  try {
    const data = await api.byok.get();
    setKeys(data);
  } catch (error) {
    console.error('Failed to fetch keys:', error);
  }
};

const updateKey = async (type: string, key: string) => {
  try {
    await api.byok.update({ [`${type}ApiKey`]: key });
    toast.success('Key updated successfully');
  } catch (error) {
    toast.error('Failed to update key');
  }
};
```

---

## 📱 Responsive Design Requirements

### Breakpoints (Tailwind)
```typescript
// Mobile first approach
sm: '640px'   // Small devices
md: '768px'   // Medium devices
lg: '1024px'  // Large devices
xl: '1280px'  // Extra large devices
```

### Grid Layouts
```typescript
// Pricing cards
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">

// API key cards
<div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
```

---

## 🎯 User Experience Requirements

### Loading States
- Show skeleton loaders while fetching data
- Disable buttons during API calls
- Show spinner on buttons: `{loading ? 'Loading...' : 'Submit'}`

### Error Handling
```typescript
// Use toast notifications
import { toast } from 'sonner';

toast.success('Key updated successfully');
toast.error('Failed to update key. Please try again.');
toast.info('Your payment is being processed');
```

### Form Validation
```typescript
// OpenAI key validation
const isValidOpenAIKey = (key: string) => {
  return key.startsWith('sk-') && key.length > 20;
};

// Anthropic key validation
const isValidAnthropicKey = (key: string) => {
  return key.startsWith('sk-ant-') && key.length > 20;
};
```

---

## 🚀 Performance Optimization

### Code Splitting
```typescript
// Lazy load Razorpay component
const RazorpayCheckout = dynamic(
  () => import('@/components/payment/RazorpayCheckout'),
  { ssr: false }
);
```

### Image Optimization
```typescript
import Image from 'next/image';

<Image
  src="/razorpay-logo.svg"
  alt="Razorpay"
  width={120}
  height={40}
/>
```

---

## ✅ Testing Checklist

Before considering complete, test:

- [ ] **API Keys Page**
  - [ ] Load existing keys (masked)
  - [ ] Add new key
  - [ ] Update existing key
  - [ ] Delete key
  - [ ] Error handling (invalid key format)

- [ ] **Purchase Page**
  - [ ] Display all pricing options
  - [ ] Click purchase opens Razorpay
  - [ ] Complete test payment
  - [ ] Redirect to success page

- [ ] **Success Page**
  - [ ] Show after successful payment
  - [ ] Download button works
  - [ ] Links to documentation work

- [ ] **Download Page**
  - [ ] Only accessible with valid access
  - [ ] Download button triggers download
  - [ ] Shows correct purchase date

- [ ] **Settings Page**
  - [ ] Show access status
  - [ ] Display payment history
  - [ ] Navigate to API keys page

- [ ] **Mobile Responsive**
  - [ ] All pages work on mobile
  - [ ] Razorpay checkout works on mobile
  - [ ] Forms are usable on mobile

---

## 🎨 Design Inspiration & References

### Similar UIs to Reference
- **Vercel Dashboard** - Clean, modern, card-based
- **Stripe Dashboard** - Professional, data-heavy
- **Linear** - Minimalist, keyboard-friendly
- **Retool** - Developer-focused

### Color Palette
```typescript
Primary Blue: #3399cc
Success Green: #22c55e
Error Red: #ef4444
Warning Yellow: #eab308
Gray Scale: #f9fafb → #111827
```

---

## 📚 Additional Resources

### Documentation to Link
- Architecture: `/docs/architecture.md`
- Deployment: `/docs/deployment.md`
- API Reference: `/docs/api-reference.md`

### Support Channels
- Discord: https://discord.gg/flowagent
- Email: support@flowagent.io
- Docs: https://docs.flowagent.io

---

## 🔍 Important Notes for Kimi

1. **All backend APIs are 100% ready** - Just call them, they work
2. **Use existing UI components** - Don't recreate what exists
3. **Razorpay is production-ready** - Integration code provided above
4. **TypeScript is mandatory** - Type everything properly
5. **Mobile-first design** - Start with mobile, scale up
6. **Error handling is critical** - Always show user-friendly errors
7. **Security matters** - Never log API keys, always mask them
8. **Loading states everywhere** - User should always know what's happening
9. **Accessibility** - Use proper ARIA labels, keyboard navigation
10. **Performance** - Lazy load heavy components, optimize images

---

## 📦 Summary

**What Kimi needs to build:**
1. ✅ `/settings/api-keys` - BYOK management page
2. ✅ `/purchase` - Pricing & checkout page
3. ✅ `/purchase/success` - Success confirmation
4. ✅ `/download` - Source code download
5. ✅ `/settings` (enhance) - Add billing tab

**Total estimated effort:**
- API Keys Page: 2-3 hours
- Purchase Page: 3-4 hours (Razorpay integration)
- Success Page: 1 hour
- Download Page: 1-2 hours
- Settings Enhancement: 1 hour

**Total: ~8-12 hours of development**

---

**All backend endpoints are live and ready to use. Just build the UI and connect to the APIs provided above. Good luck! 🚀**

---

**Last Updated**: January 31, 2026
**Version**: 1.0
**Status**: Complete specification ready for implementation

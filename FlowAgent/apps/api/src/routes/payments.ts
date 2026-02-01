import { Hono } from 'hono';
import { zValidator } from '@hono/zod-validator';
import { z } from 'zod';
import { users, accessPayments } from '@flowagent/database/src/schema';
import { eq } from 'drizzle-orm';
import crypto from 'node:crypto';

const app = new Hono() as AppType;
import type { AppType } from "../types/hono";

const createPaymentSchema = z.object({
  plan: z.enum(['access-tier']).default('access-tier'),
  currency: z.enum(['USD', 'INR', 'EUR', 'GBP']).optional().default('USD'),
});

// Currency conversion rates (approximate)
const CURRENCY_RATES = {
  USD: 10.00,
  INR: 800.00,  // ₹800
  EUR: 9.50,
  GBP: 8.50,
};

// Create Razorpay order for $10 access tier
app.post('/checkout', zValidator('json', createPaymentSchema), async (c) => {
  const user = c.get('user');
  const db = c.get('db');
  const data = c.req.valid('json');

  // Check if user already has access
  const userData = await db.query.users.findFirst({
    where: eq(users.id, user.id),
    columns: {
      hasArchitectureAccess: true,
      hasDownloadAccess: true,
      email: true,
      username: true,
    },
  });

  if (userData?.hasArchitectureAccess && userData?.hasDownloadAccess) {
    return c.json({
      error: 'You already have access to all features',
    }, 400);
  }

  const currency = data.currency || 'USD';
  const amount = CURRENCY_RATES[currency];

  // Create payment record
  const [payment] = await db.insert(accessPayments).values({
    userId: user.id,
    amount: amount.toString(),
    currency: currency,
    status: 'pending',
    grantsArchitectureAccess: true,
    grantsDownloadAccess: true,
    grantsDuration: 'lifetime',
  }).returning();

  // Create Razorpay order
  const razorpayKeyId = c.env.RAZORPAY_KEY_ID;
  const razorpayKeySecret = c.env.RAZORPAY_KEY_SECRET;

  if (!razorpayKeyId || !razorpayKeySecret) {
    // Return payment link for manual processing
    return c.json({
      paymentId: payment.id,
      amount: amount,
      currency: currency,
      manualPayment: true,
      message: 'Please contact support to complete payment',
    }, 201);
  }

  try {
    // Create Razorpay order
    const orderAmount = Math.round(amount * 100); // Convert to smallest currency unit
    const orderPayload = {
      amount: orderAmount,
      currency: currency,
      receipt: payment.id,
      notes: {
        paymentId: payment.id,
        userId: user.id,
        plan: 'architecture-access',
      },
    };

    const auth = Buffer.from(`${razorpayKeyId}:${razorpayKeySecret}`).toString('base64');

    const response = await fetch('https://api.razorpay.com/v1/orders', {
      method: 'POST',
      headers: {
        'Authorization': `Basic ${auth}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(orderPayload),
    });

    if (!response.ok) {
      throw new Error('Failed to create Razorpay order');
    }

    const order = await response.json() as { id: string };

    // Update payment with Razorpay order ID
    await db.update(accessPayments)
      .set({
        razorpayOrderId: order.id,
      })
      .where(eq(accessPayments.id, payment.id));

    return c.json({
      paymentId: payment.id,
      razorpayOrderId: order.id,
      amount: amount,
      currency: currency,
      razorpayKeyId: razorpayKeyId,
      userEmail: userData?.email || '',
      userName: userData?.username || '',
    }, 201);

  } catch (err: any) {
    console.error('Razorpay order creation failed:', err);
    return c.json({
      error: 'Failed to create payment order',
      message: err.message,
    }, 500);
  }
});

// Webhook handler for Razorpay payment success
app.post('/webhook/razorpay', async (c) => {
  const db = c.get('db');
  const body = await c.req.text();
  const razorpaySignature = c.req.header('x-razorpay-signature');
  const razorpayWebhookSecret = c.env.RAZORPAY_WEBHOOK_SECRET;

  try {
    // Verify Razorpay webhook signature
    if (razorpayWebhookSecret && razorpaySignature) {
      const expectedSignature = crypto
        .createHmac('sha256', razorpayWebhookSecret)
        .update(body)
        .digest('hex');

      if (expectedSignature !== razorpaySignature) {
        return c.json({ error: 'Invalid signature' }, 401);
      }
    }

    const event = JSON.parse(body);

    // Handle payment.captured event
    if (event.event === 'payment.captured') {
      const payment = event.payload.payment.entity;
      const razorpayOrderId = payment.order_id;
      const razorpayPaymentId = payment.id;

      // Find payment by Razorpay order ID
      const accessPayment = await db.query.accessPayments.findFirst({
        where: eq(accessPayments.razorpayOrderId, razorpayOrderId),
      });

      if (accessPayment) {
        // Update payment status
        await db.update(accessPayments)
          .set({
            status: 'completed',
            razorpayPaymentId: razorpayPaymentId,
            paymentMethod: payment.method,
            completedAt: new Date(),
          })
          .where(eq(accessPayments.id, accessPayment.id));

        // Grant access to user
        await db.update(users)
          .set({
            hasArchitectureAccess: true,
            hasDownloadAccess: true,
            accessPurchasedAt: new Date(),
            accessExpiresAt: null, // Lifetime access
            updatedAt: new Date(),
          })
          .where(eq(users.id, accessPayment.userId));
      }
    }

    // Handle order.paid event (backup)
    if (event.event === 'order.paid') {
      const order = event.payload.order.entity;
      const razorpayOrderId = order.id;

      const accessPayment = await db.query.accessPayments.findFirst({
        where: eq(accessPayments.razorpayOrderId, razorpayOrderId),
      });

      if (accessPayment && accessPayment.status === 'pending') {
        await db.update(accessPayments)
          .set({
            status: 'completed',
            completedAt: new Date(),
          })
          .where(eq(accessPayments.id, accessPayment.id));

        await db.update(users)
          .set({
            hasArchitectureAccess: true,
            hasDownloadAccess: true,
            accessPurchasedAt: new Date(),
            accessExpiresAt: null,
            updatedAt: new Date(),
          })
          .where(eq(users.id, accessPayment.userId));
      }
    }

    return c.json({ received: true });
  } catch (err: any) {
    console.error('Webhook error:', err);
    return c.json({ error: 'Webhook error', message: err.message }, 400);
  }
});

// Get user's payment history
app.get('/history', async (c) => {
  const user = c.get('user');
  const db = c.get('db');

  const payments = await db.query.accessPayments.findMany({
    where: eq(accessPayments.userId, user.id),
    orderBy: (accessPayments, { desc }) => [desc(accessPayments.createdAt)],
  });

  return c.json({
    payments,
  });
});

// Get current access status
app.get('/access-status', async (c) => {
  const user = c.get('user');
  const db = c.get('db');

  const userData = await db.query.users.findFirst({
    where: eq(users.id, user.id),
    columns: {
      hasArchitectureAccess: true,
      hasDownloadAccess: true,
      accessPurchasedAt: true,
      accessExpiresAt: true,
    },
  });

  return c.json({
    hasAccess: userData?.hasArchitectureAccess && userData?.hasDownloadAccess,
    hasArchitectureAccess: userData?.hasArchitectureAccess || false,
    hasDownloadAccess: userData?.hasDownloadAccess || false,
    purchasedAt: userData?.accessPurchasedAt,
    expiresAt: userData?.accessExpiresAt,
  });
});

// Verify Razorpay payment manually
app.post('/verify', zValidator('json', z.object({
  razorpayOrderId: z.string(),
  razorpayPaymentId: z.string(),
  razorpaySignature: z.string(),
})), async (c) => {
  const user = c.get('user');
  const db = c.get('db');
  const data = c.req.valid('json');

  const razorpayKeySecret = c.env.RAZORPAY_KEY_SECRET;

  if (!razorpayKeySecret) {
    return c.json({ error: 'Payment gateway not configured' }, 500);
  }

  // Verify payment signature
  const generatedSignature = crypto
    .createHmac('sha256', razorpayKeySecret)
    .update(`${data.razorpayOrderId}|${data.razorpayPaymentId}`)
    .digest('hex');

  if (generatedSignature !== data.razorpaySignature) {
    return c.json({ error: 'Invalid payment signature' }, 400);
  }

  // Find payment by order ID
  const payment = await db.query.accessPayments.findFirst({
    where: eq(accessPayments.razorpayOrderId, data.razorpayOrderId),
  });

  if (!payment || payment.userId !== user.id) {
    return c.json({ error: 'Payment not found' }, 404);
  }

  // Update payment
  await db.update(accessPayments)
    .set({
      status: 'completed',
      razorpayPaymentId: data.razorpayPaymentId,
      completedAt: new Date(),
    })
    .where(eq(accessPayments.id, payment.id));

  // Grant access
  await db.update(users)
    .set({
      hasArchitectureAccess: true,
      hasDownloadAccess: true,
      accessPurchasedAt: new Date(),
      accessExpiresAt: null,
      updatedAt: new Date(),
    })
    .where(eq(users.id, user.id));

  return c.json({
    success: true,
    message: 'Payment verified and access granted successfully',
  });
});

// Manual payment confirmation (for testing/admin)
app.post('/confirm/:paymentId', async (c) => {
  const user = c.get('user');
  const db = c.get('db');
  const paymentId = c.req.param('paymentId');

  const payment = await db.query.accessPayments.findFirst({
    where: eq(accessPayments.id, paymentId),
  });

  if (!payment || payment.userId !== user.id) {
    return c.json({ error: 'Payment not found' }, 404);
  }

  // Update payment
  await db.update(accessPayments)
    .set({
      status: 'completed',
      completedAt: new Date(),
    })
    .where(eq(accessPayments.id, paymentId));

  // Grant access
  await db.update(users)
    .set({
      hasArchitectureAccess: true,
      hasDownloadAccess: true,
      accessPurchasedAt: new Date(),
      accessExpiresAt: null,
      updatedAt: new Date(),
    })
    .where(eq(users.id, user.id));

  return c.json({
    success: true,
    message: 'Access granted successfully',
  });
});

export { app as paymentRoutes };

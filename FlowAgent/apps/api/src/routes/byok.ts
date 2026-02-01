import { Hono } from 'hono';
import { zValidator } from '@hono/zod-validator';
import { z } from 'zod';
import { users } from '@flowagent/database/src/schema';
import { eq } from 'drizzle-orm';
import crypto from 'node:crypto';
import type { AppType } from '../types/hono';

const app = new Hono() as AppType;

// Encryption helpers (simple AES-256-GCM)
const ENCRYPTION_KEY = process.env.ENCRYPTION_KEY || crypto.randomBytes(32).toString('hex');

function encrypt(text: string): string {
  const iv = crypto.randomBytes(16);
  const key = Buffer.from(ENCRYPTION_KEY.slice(0, 64), 'hex');
  const cipher = crypto.createCipheriv('aes-256-gcm', key, iv);

  let encrypted = cipher.update(text, 'utf8', 'hex');
  encrypted += cipher.final('hex');
  const authTag = cipher.getAuthTag();

  return `${iv.toString('hex')}:${authTag.toString('hex')}:${encrypted}`;
}

function decrypt(encryptedData: string): string {
  const [ivHex, authTagHex, encrypted] = encryptedData.split(':');
  const key = Buffer.from(ENCRYPTION_KEY.slice(0, 64), 'hex');
  const iv = Buffer.from(ivHex, 'hex');
  const authTag = Buffer.from(authTagHex, 'hex');

  const decipher = crypto.createDecipheriv('aes-256-gcm', key, iv);
  decipher.setAuthTag(authTag);

  let decrypted = decipher.update(encrypted, 'hex', 'utf8');
  decrypted += decipher.final('utf8');

  return decrypted;
}

const byokSchema = z.object({
  openaiApiKey: z.string().optional(),
  anthropicApiKey: z.string().optional(),
  serperApiKey: z.string().optional(),
});

// Get user's BYOK keys (masked)
app.get('/', async (c) => {
  const user = c.get('user');
  const db = c.get('db');

  const userData = await db.query.users.findFirst({
    where: eq(users.id, user.id),
    columns: {
      openaiApiKey: true,
      anthropicApiKey: true,
      serperApiKey: true,
    },
  });

  // Return masked keys
  return c.json({
    openaiApiKey: userData?.openaiApiKey ? 'sk-....' + userData.openaiApiKey.slice(-4) : null,
    anthropicApiKey: userData?.anthropicApiKey ? 'sk-ant-....' + userData.anthropicApiKey.slice(-4) : null,
    serperApiKey: userData?.serperApiKey ? '****....' + userData.serperApiKey.slice(-4) : null,
  });
});

// Update BYOK keys
app.put('/', zValidator('json', byokSchema), async (c) => {
  const user = c.get('user');
  const db = c.get('db');
  const data = c.req.valid('json');

  const updates: any = {};

  // Encrypt and store keys
  if (data.openaiApiKey) {
    updates.openaiApiKey = encrypt(data.openaiApiKey);
  }
  if (data.anthropicApiKey) {
    updates.anthropicApiKey = encrypt(data.anthropicApiKey);
  }
  if (data.serperApiKey) {
    updates.serperApiKey = encrypt(data.serperApiKey);
  }

  if (Object.keys(updates).length > 0) {
    await db.update(users)
      .set({
        ...updates,
        updatedAt: new Date(),
      })
      .where(eq(users.id, user.id));
  }

  return c.json({
    success: true,
    message: 'API keys updated successfully',
  });
});

// Delete specific BYOK key
app.delete('/:keyType', async (c) => {
  const user = c.get('user');
  const db = c.get('db');
  const keyType = c.req.param('keyType');

  const validTypes = ['openai', 'anthropic', 'serper'];
  if (!validTypes.includes(keyType)) {
    return c.json({ error: 'Invalid key type' }, 400);
  }

  const columnName = `${keyType}ApiKey`;

  await db.update(users)
    .set({
      [columnName]: null,
      updatedAt: new Date(),
    })
    .where(eq(users.id, user.id));

  return c.json({
    success: true,
    message: `${keyType} API key deleted successfully`,
  });
});

// Helper function to get decrypted keys (for internal use)
export async function getUserApiKeys(db: any, userId: string) {
  const userData = await db.query.users.findFirst({
    where: eq(users.id, userId),
    columns: {
      openaiApiKey: true,
      anthropicApiKey: true,
      serperApiKey: true,
    },
  });

  return {
    openaiApiKey: userData?.openaiApiKey ? decrypt(userData.openaiApiKey) : null,
    anthropicApiKey: userData?.anthropicApiKey ? decrypt(userData.anthropicApiKey) : null,
    serperApiKey: userData?.serperApiKey ? decrypt(userData.serperApiKey) : null,
  };
}

export { app as byokRoutes };

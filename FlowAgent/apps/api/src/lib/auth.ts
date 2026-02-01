import { Lucia } from 'lucia';
import { PostgresJsAdapter } from '@lucia-auth/adapter-postgresql';
import type postgres from 'postgres';

export function createAuth(postgresClient: postgres.Sql) {
  const adapter = new PostgresJsAdapter(postgresClient, {
    user: 'users',
    session: 'user_sessions',
  });

  return new Lucia(adapter, {
    sessionCookie: {
      name: 'session',
      expires: false,
      attributes: {
        secure: true,
        sameSite: 'strict',
        path: '/',
      },
    },
    getUserAttributes: (attributes: any) => ({
      id: attributes.id,
      email: attributes.email,
      username: attributes.username,
      subscriptionTier: attributes.subscription_tier,
    }),
  });
}

// Password hashing
export async function hashPassword(password: string): Promise<string> {
  const { hash } = await import('@node-rs/argon2');
  return await hash(password, {
    memoryCost: 19456,
    timeCost: 2,
    outputLen: 32,
    parallelism: 1,
  });
}

export async function verifyPassword(hash: string, password: string): Promise<boolean> {
  const { verify } = await import('@node-rs/argon2');
  return await verify(hash, password);
}

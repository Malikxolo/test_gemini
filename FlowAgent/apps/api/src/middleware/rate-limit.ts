import { Context, Next } from 'hono';
import { Ratelimit } from '@upstash/ratelimit';
import { Redis } from '@upstash/redis/cloudflare';
import type { User } from '../types/hono';

const RATE_LIMITS: Record<string, { requests: number; window: `${number} m` }> = {
  free: { requests: 60, window: '1 m' },
  pro: { requests: 600, window: '1 m' },
  enterprise: { requests: 6000, window: '1 m' },
};

export async function rateLimit(c: Context, next: Next) {
  const user = c.get('user') as User;
  const tier = user?.subscriptionTier || 'free';

  const redis = new Redis({
    url: c.env.UPSTASH_REDIS_REST_URL,
    token: c.env.UPSTASH_REDIS_REST_TOKEN,
  });

  const limitConfig = RATE_LIMITS[tier];
  const ratelimit = new Ratelimit({
    redis,
    limiter: Ratelimit.slidingWindow(
      limitConfig.requests,
      limitConfig.window
    ),
  });

  const { success, limit, remaining, reset } = await ratelimit.limit(user.id);

  // Add rate limit headers
  c.header('X-RateLimit-Limit', limit.toString());
  c.header('X-RateLimit-Remaining', remaining.toString());
  c.header('X-RateLimit-Reset', reset.toString());

  if (!success) {
    return c.json({
      error: 'Rate limit exceeded',
      limit,
      remaining: 0,
      reset,
    }, 429);
  }

  await next();
}

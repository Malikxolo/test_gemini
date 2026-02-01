import { Context } from 'hono';
import { HTTPException } from 'hono/http-exception';

export function errorHandler(err: Error, c: Context) {
  console.error({
    event: 'http.error',
    error: err.message,
    stack: err.stack,
    path: c.req.path,
    user_id: c.get('user')?.id,
  });

  if (err instanceof HTTPException) {
    return c.json({
      error: err.message,
      status: err.status,
    }, err.status);
  }

  return c.json({
    error: 'Internal server error',
    message: err.message,
  }, 500);
}

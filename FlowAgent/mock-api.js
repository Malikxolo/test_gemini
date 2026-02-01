// Simple mock API server for testing
const http = require('http');

const PORT = 8787;

// CORS headers
const corsHeaders = {
  'Access-Control-Allow-Origin': 'http://localhost:3000',
  'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
  'Access-Control-Allow-Credentials': 'true',
};

const server = http.createServer((req, res) => {
  // Handle CORS preflight
  if (req.method === 'OPTIONS') {
    res.writeHead(204, corsHeaders);
    res.end();
    return;
  }

  // Set CORS headers for all responses
  Object.entries(corsHeaders).forEach(([key, value]) => {
    res.setHeader(key, value);
  });

  const url = req.url;
  const method = req.method;

  console.log(`${method} ${url}`);

  // Health check
  if (url === '/health' && method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'ok', timestamp: Date.now() }));
    return;
  }

  // Signup
  if (url === '/api/auth/register' && method === 'POST') {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', () => {
      try {
        const data = JSON.parse(body);
        console.log('Signup attempt:', data.email);
        
        // Mock successful signup
        res.writeHead(201, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          user: {
            id: 'user_' + Date.now(),
            email: data.email,
            username: data.username,
            displayName: data.displayName || null,
          },
          token: 'mock_token_' + Date.now(),
        }));
      } catch (e) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Invalid request' }));
      }
    });
    return;
  }

  // Login
  if (url === '/api/auth/login' && method === 'POST') {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', () => {
      try {
        const data = JSON.parse(body);
        console.log('Login attempt:', data.email);
        
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          user: {
            id: 'user_123',
            email: data.email,
            username: 'testuser',
            displayName: 'Test User',
          },
          token: 'mock_token_' + Date.now(),
        }));
      } catch (e) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Invalid request' }));
      }
    });
    return;
  }

  // Get current user
  if (url === '/api/auth/me' && method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      user: {
        id: 'user_123',
        email: 'test@example.com',
        username: 'testuser',
        displayName: 'Test User',
      },
    }));
    return;
  }

  // Agents list
  if (url === '/api/agents' && method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify([
      {
        id: 'agent_1',
        name: 'Test Assistant',
        description: 'A helpful test agent',
        model: 'gpt-3.5-turbo',
        temperature: 0.7,
        maxTokens: 2000,
        tools: [],
        isPublic: false,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      },
    ]));
    return;
  }

  // Create agent
  if (url === '/api/agents' && method === 'POST') {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', () => {
      try {
        const data = JSON.parse(body);
        res.writeHead(201, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          id: 'agent_' + Date.now(),
          ...data,
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        }));
      } catch (e) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Invalid request' }));
      }
    });
    return;
  }

  // BYOK - Get API keys
  if (url === '/api/byok' && method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      openaiApiKey: null,
      anthropicApiKey: null,
      serperApiKey: null,
    }));
    return;
  }

  // Payments - Access status
  if (url === '/api/payments/access-status' && method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      hasAccess: false,
      hasArchitectureAccess: false,
      hasDownloadAccess: false,
      purchasedAt: null,
      expiresAt: null,
    }));
    return;
  }

  // User usage
  if (url === '/api/users/me/usage' && method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      totalExecutions: 0,
      totalTokensUsed: 0,
      totalCost: 0,
      monthlyExecutions: 0,
      monthlyTokensUsed: 0,
      monthlyCost: 0,
    }));
    return;
  }

  // Default 404
  res.writeHead(404, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ error: 'Not found' }));
});

server.listen(PORT, () => {
  console.log(`✅ Mock API server running on http://localhost:${PORT}`);
  console.log('📋 Available endpoints:');
  console.log('  GET  /health');
  console.log('  POST /api/auth/register');
  console.log('  POST /api/auth/login');
  console.log('  GET  /api/auth/me');
  console.log('  GET  /api/agents');
  console.log('  POST /api/agents');
  console.log('  GET  /api/byok');
  console.log('  GET  /api/payments/access-status');
  console.log('  GET  /api/users/me/usage');
});

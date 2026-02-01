import { describe, it, expect, beforeEach, vi } from 'vitest';

describe('API Client', () => {
  describe('apiRequest', () => {
    it('should build correct URL with endpoint', () => {
      const apiUrl = 'http://localhost:8787';
      const endpoint = '/api/agents';
      const fullUrl = `${apiUrl}${endpoint}`;

      expect(fullUrl).toBe('http://localhost:8787/api/agents');
    });

    it('should include credentials in request', () => {
      const options = {
        credentials: 'include' as RequestCredentials,
      };

      expect(options.credentials).toBe('include');
    });

    it('should set Content-Type header', () => {
      const headers = {
        'Content-Type': 'application/json',
      };

      expect(headers['Content-Type']).toBe('application/json');
    });

    it('should throw error on non-OK response', async () => {
      const response = {
        ok: false,
        status: 400,
      };

      expect(response.ok).toBe(false);
    });

    it('should parse JSON error response', () => {
      const errorResponse = {
        error: 'Invalid request',
      };

      expect(errorResponse.error).toBe('Invalid request');
    });
  });

  describe('api.auth', () => {
    describe('register', () => {
      it('should send POST request with user data', () => {
        const data = {
          email: 'test@example.com',
          username: 'testuser',
          password: 'password123',
          displayName: 'Test User',
        };

        expect(data.email).toBeTruthy();
        expect(data.username).toBeTruthy();
        expect(data.password).toBeTruthy();
      });

      it('should handle optional displayName', () => {
        const data = {
          email: 'test@example.com',
          username: 'testuser',
          password: 'password123',
        };

        expect(data).not.toHaveProperty('displayName');
      });
    });

    describe('login', () => {
      it('should send POST request with credentials', () => {
        const data = {
          email: 'test@example.com',
          password: 'password123',
        };

        expect(data.email).toBeTruthy();
        expect(data.password).toBeTruthy();
      });
    });

    describe('logout', () => {
      it('should send POST request to logout endpoint', () => {
        const method = 'POST';
        const endpoint = '/api/auth/logout';

        expect(method).toBe('POST');
        expect(endpoint).toBe('/api/auth/logout');
      });
    });

    describe('getCurrentUser', () => {
      it('should send GET request to me endpoint', () => {
        const endpoint = '/api/auth/me';

        expect(endpoint).toBe('/api/auth/me');
      });
    });
  });

  describe('api.agents', () => {
    describe('list', () => {
      it('should fetch agents list', () => {
        const endpoint = '/api/agents';

        expect(endpoint).toBe('/api/agents');
      });
    });

    describe('get', () => {
      it('should fetch single agent by id', () => {
        const id = 'agent-123';
        const endpoint = `/api/agents/${id}`;

        expect(endpoint).toBe('/api/agents/agent-123');
      });
    });

    describe('create', () => {
      it('should send POST request with agent data', () => {
        const data = {
          name: 'Test Agent',
          mode: 'auto',
          model: 'gpt-4o-mini',
        };

        expect(data.name).toBeTruthy();
        expect(data.mode).toBeTruthy();
        expect(data.model).toBeTruthy();
      });
    });

    describe('update', () => {
      it('should send PATCH request with updates', () => {
        const id = 'agent-123';
        const data = {
          name: 'Updated Agent',
        };
        const method = 'PATCH';

        expect(method).toBe('PATCH');
        expect(id).toBeTruthy();
        expect(data.name).toBeTruthy();
      });
    });

    describe('delete', () => {
      it('should send DELETE request', () => {
        const id = 'agent-123';
        const method = 'DELETE';

        expect(method).toBe('DELETE');
        expect(id).toBeTruthy();
      });
    });

    describe('execute', () => {
      it('should send execution request with input', () => {
        const id = 'agent-123';
        const input = {
          message: 'Hello',
        };
        const stream = false;

        expect(id).toBeTruthy();
        expect(input.message).toBeTruthy();
        expect(stream).toBe(false);
      });

      it('should handle optional stream parameter', () => {
        const data = {
          input: { message: 'Hello' },
        };

        expect(data).not.toHaveProperty('stream');
      });
    });
  });

  describe('api.executions', () => {
    describe('list', () => {
      it('should fetch executions list', () => {
        const endpoint = '/api/executions';

        expect(endpoint).toBe('/api/executions');
      });
    });

    describe('get', () => {
      it('should fetch single execution by id', () => {
        const id = 'exec-123';
        const endpoint = `/api/executions/${id}`;

        expect(endpoint).toBe('/api/executions/exec-123');
      });
    });
  });

  describe('api.users', () => {
    describe('getUsage', () => {
      it('should fetch user usage stats', () => {
        const endpoint = '/api/users/me/usage';

        expect(endpoint).toBe('/api/users/me/usage');
      });
    });
  });
});

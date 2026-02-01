const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8787';

export async function apiRequest<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const response = await fetch(`${API_URL}${endpoint}`, {
    ...options,
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: 'Request failed' }));
    throw new Error(error.error || `HTTP ${response.status}`);
  }

  return response.json() as Promise<T>;
}

// Type definitions for API responses
export interface User {
  id: string;
  email: string;
  username: string;
  displayName: string | null;
  avatarUrl: string | null;
  createdAt: string;
  subscriptionTier?: string;
}

export interface Agent {
  id: string;
  name: string;
  description: string | null;
  systemPrompt: string;
  model: string;
  temperature: number;
  maxTokens: number;
  tools: string[];
  isPublic: boolean;
  createdAt: string;
  updatedAt: string;
  mode?: string;
}

export interface Execution {
  id: string;
  agentId: string;
  input: string;
  output: string | null;
  status: 'pending' | 'running' | 'completed' | 'failed';
  tokensUsed: number | null;
  cost: number | null;
  startedAt: string;
  completedAt: string | null;
}

export interface UsageStats {
  totalExecutions: number;
  totalTokensUsed: number;
  totalCost: number;
  monthlyExecutions: number;
  monthlyTokensUsed: number;
  monthlyCost: number;
}

export interface APIKeys {
  openaiApiKey: string | null;
  anthropicApiKey: string | null;
  serperApiKey: string | null;
}

export interface CheckoutData {
  paymentId: string;
  razorpayOrderId: string;
  amount: number;
  currency: string;
  razorpayKeyId: string;
  userEmail: string;
  userName: string;
}

export interface AccessStatus {
  hasAccess: boolean;
  hasArchitectureAccess: boolean;
  hasDownloadAccess: boolean;
  purchasedAt: string;
  expiresAt: string | null;
}

export interface Payment {
  id: string;
  amount: string;
  currency: string;
  status: string;
  razorpayOrderId: string;
  createdAt: string;
  completedAt: string;
}

export interface PaymentHistory {
  payments: Payment[];
}

export const api = {
  auth: {
    register: (data: { email: string; username: string; password: string; displayName?: string }) =>
      apiRequest<User>('/api/auth/register', {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    login: (data: { email: string; password: string }) =>
      apiRequest<User>('/api/auth/login', {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    logout: () =>
      apiRequest<void>('/api/auth/logout', {
        method: 'POST',
      }),
    getCurrentUser: () => apiRequest<User>('/api/auth/me'),
  },
  agents: {
    list: () => apiRequest<Agent[]>('/api/agents'),
    get: (id: string) => apiRequest<Agent>(`/api/agents/${id}`),
    create: (data: any) =>
      apiRequest<Agent>('/api/agents', {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    update: (id: string, data: any) =>
      apiRequest<Agent>(`/api/agents/${id}`, {
        method: 'PATCH',
        body: JSON.stringify(data),
      }),
    delete: (id: string) =>
      apiRequest<void>(`/api/agents/${id}`, {
        method: 'DELETE',
      }),
    execute: (id: string, input: any, stream?: boolean) =>
      apiRequest<Execution>(`/api/agents/${id}/execute`, {
        method: 'POST',
        body: JSON.stringify({ input, stream }),
      }),
  },
  executions: {
    list: () => apiRequest<Execution[]>('/api/executions'),
    get: (id: string) => apiRequest<Execution>(`/api/executions/${id}`),
  },
  users: {
    getUsage: () => apiRequest<UsageStats>('/api/users/me/usage'),
  },
  byok: {
    get: () => apiRequest<APIKeys>('/api/byok'),
    update: (data: { openaiApiKey?: string; anthropicApiKey?: string; serperApiKey?: string }) =>
      apiRequest<APIKeys>('/api/byok', {
        method: 'PUT',
        body: JSON.stringify(data),
      }),
    delete: (keyType: 'openai' | 'anthropic' | 'serper') =>
      apiRequest<void>(`/api/byok/${keyType}`, {
        method: 'DELETE',
      }),
  },
  payments: {
    checkout: (data: { plan: string; currency: 'INR' | 'USD' | 'EUR' | 'GBP' }) =>
      apiRequest<CheckoutData>('/api/payments/checkout', {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    verify: (data: { razorpayOrderId: string; razorpayPaymentId: string; razorpaySignature: string }) =>
      apiRequest<{ success: boolean }>('/api/payments/verify', {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    getAccessStatus: () => apiRequest<AccessStatus>('/api/payments/access-status'),
    getHistory: () => apiRequest<PaymentHistory>('/api/payments/history'),
  },
};

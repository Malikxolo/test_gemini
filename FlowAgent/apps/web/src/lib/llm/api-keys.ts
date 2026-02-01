import { createClientComponentClient } from '@/lib/supabase';
import type { Provider } from './provider';

const supabase = createClientComponentClient();

export interface ApiKey {
  id: string;
  provider: Provider;
  key_name: string;
  is_active: boolean;
  is_default: boolean;
  created_at: string;
}

/**
 * Get all API keys for the current user
 * Note: Returns metadata only, NOT the actual key values
 */
export async function getApiKeys(): Promise<ApiKey[]> {
  const response = await fetch('/api/keys');
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to fetch API keys');
  }
  
  const data = await response.json();
  return data.keys || [];
}

/**
 * Add a new API key
 * The key is encrypted server-side, never stored in plain text
 */
export async function addApiKey(
  provider: Provider,
  key: string,
  keyName: string,
  isDefault: boolean = false
): Promise<void> {
  const response = await fetch('/api/keys', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      provider,
      key,
      keyName,
      isDefault,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to save API key');
  }
}

/**
 * Delete an API key
 */
export async function deleteApiKey(id: string): Promise<void> {
  const response = await fetch(`/api/keys?id=${id}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to delete API key');
  }
}

/**
 * Set an API key as the default for its provider
 */
export async function setDefaultApiKey(id: string, provider: Provider): Promise<void> {
  const response = await fetch('/api/keys', {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      id,
      provider,
      action: 'set-default',
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to update API key');
  }
}

/**
 * Check if a provider has a default key configured
 * Note: This only checks existence, doesn't return the key
 */
export async function hasDefaultApiKey(provider: Provider): Promise<boolean> {
  try {
    const keys = await getApiKeys();
    return keys.some((k) => k.provider === provider && k.is_default && k.is_active);
  } catch {
    return false;
  }
}

/**
 * Get available providers
 */
export function getAvailableProviders(): Provider[] {
  return ['openai', 'anthropic', 'google', 'openrouter'];
}

/**
 * Get models available for a provider
 */
export function getModelsForProvider(provider: Provider): string[] {
  switch (provider) {
    case 'openai':
      return ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo'];
    case 'anthropic':
      return ['claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku'];
    case 'google':
      return ['gemini-pro', 'gemini-pro-vision'];
    case 'openrouter':
      return ['openrouter'];
    default:
      return [];
  }
}
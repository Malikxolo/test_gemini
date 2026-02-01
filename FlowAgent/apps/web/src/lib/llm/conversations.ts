import { createClientComponentClient } from '@/lib/supabase';
import type { Provider } from './provider';

const supabase = createClientComponentClient();

export interface Conversation {
  id: string;
  title: string;
  project_id?: string;
  agent_persona_id?: string;
  model: string;
  provider: Provider;
  total_tokens: number;
  total_cost: number;
  budget_used: number;
  budget_limit: number;
  created_at: string;
  updated_at: string;
}

export interface Message {
  id: string;
  conversation_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  agent_name?: string;
  tokens_used: number;
  cost: number;
  model?: string;
  provider?: string;
  attachments?: Attachment[];
  created_at: string;
}

export interface Attachment {
  name: string;
  type: string;
  size: number;
  url: string;
}

export interface StreamingCallbacks {
  onToken: (token: string) => void;
  onComplete: (fullResponse: string, usage: { input: number; output: number; total: number; cost: number }) => void;
  onError: (error: Error) => void;
}

/**
 * Stream a chat message through the secure server-side API
 * API keys are never exposed to the client
 */
export async function streamChatMessage(
  message: string,
  conversationId: string,
  provider: Provider,
  model: string,
  history: Array<{ role: 'user' | 'assistant'; content: string }>,
  callbacks: StreamingCallbacks,
  systemPrompt?: string
): Promise<void> {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      conversationId,
      provider,
      model,
      history,
      systemPrompt,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to start chat');
  }

  if (!response.body) {
    throw new Error('No response body');
  }

  // Handle SSE stream
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));

            if (data.error) {
              callbacks.onError(new Error(data.error));
              return;
            }

            if (data.done) {
              callbacks.onComplete(data.fullResponse, data.usage);
              return;
            }

            if (data.token) {
              callbacks.onToken(data.token);
            }
          } catch (e) {
            // Ignore parse errors for non-JSON lines
          }
        }
      }
    }
  } catch (error) {
    callbacks.onError(error as Error);
  } finally {
    reader.releaseLock();
  }
}

export async function createConversation(
  title: string,
  model: string,
  provider: Provider,
  projectId?: string,
  agentPersonaId?: string
): Promise<Conversation> {
  const { data, error } = await (supabase as any)
    .from('conversations')
    .insert({
      title,
      model,
      provider,
      project_id: projectId,
      agent_persona_id: agentPersonaId,
      budget_limit: 5.00
    })
    .select()
    .single();

  if (error) throw error;
  return data;
}

export async function getConversations(): Promise<Conversation[]> {
  const { data, error } = await (supabase as any)
    .from('conversations')
    .select('*')
    .order('updated_at', { ascending: false });

  if (error) throw error;
  return data || [];
}

export async function getConversation(id: string): Promise<Conversation | null> {
  const { data, error } = await (supabase as any)
    .from('conversations')
    .select('*')
    .eq('id', id)
    .single();

  if (error) return null;
  return data;
}

export async function addMessage(
  conversationId: string,
  role: 'user' | 'assistant' | 'system',
  content: string,
  metadata?: {
    agent_name?: string;
    tokens_used?: number;
    cost?: number;
    model?: string;
    provider?: string;
    attachments?: Attachment[];
  }
): Promise<Message> {
  const { data, error } = await (supabase as any)
    .from('messages')
    .insert({
      conversation_id: conversationId,
      role,
      content,
      agent_name: metadata?.agent_name,
      tokens_used: metadata?.tokens_used || 0,
      cost: metadata?.cost || 0,
      model: metadata?.model,
      provider: metadata?.provider,
      attachments: metadata?.attachments || []
    })
    .select()
    .single();

  if (error) throw error;

  // Update conversation totals
  await updateConversationStats(conversationId, metadata?.tokens_used || 0, metadata?.cost || 0);

  return data;
}

export async function getMessages(conversationId: string): Promise<Message[]> {
  const { data, error } = await (supabase as any)
    .from('messages')
    .select('*')
    .eq('conversation_id', conversationId)
    .order('created_at', { ascending: true });

  if (error) throw error;
  return data || [];
}

async function updateConversationStats(
  conversationId: string, 
  tokens: number, 
  cost: number
): Promise<void> {
  const { data: conv } = await (supabase as any)
    .from('conversations')
    .select('total_tokens, total_cost, budget_used')
    .eq('id', conversationId)
    .single();

  if (conv) {
    await (supabase as any)
      .from('conversations')
      .update({
        total_tokens: (conv.total_tokens || 0) + tokens,
        total_cost: parseFloat(((conv.total_cost || 0) + cost).toFixed(4)),
        budget_used: parseFloat(((conv.budget_used || 0) + cost).toFixed(4)),
        updated_at: new Date().toISOString()
      })
      .eq('id', conversationId);
  }
}

export async function deleteConversation(id: string): Promise<void> {
  const { error } = await (supabase as any)
    .from('conversations')
    .delete()
    .eq('id', id);

  if (error) throw error;
}

export async function checkBudget(conversationId: string): Promise<{ 
  withinBudget: boolean; 
  used: number; 
  limit: number; 
  remaining: number 
}> {
  const { data, error } = await (supabase as any)
    .from('conversations')
    .select('budget_used, budget_limit')
    .eq('id', conversationId)
    .single();

  if (error || !data) {
    return { withinBudget: false, used: 0, limit: 5, remaining: 0 };
  }

  const remaining = (data.budget_limit || 5) - (data.budget_used || 0);
  return {
    withinBudget: (data.budget_used || 0) < (data.budget_limit || 5),
    used: data.budget_used || 0,
    limit: data.budget_limit || 5,
    remaining: Math.max(0, remaining)
  };
}
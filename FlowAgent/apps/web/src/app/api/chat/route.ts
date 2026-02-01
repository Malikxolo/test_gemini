import { NextRequest, NextResponse } from 'next/server';
import OpenAI from 'openai';
import Anthropic from '@anthropic-ai/sdk';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { createClient } from '@/lib/supabase/server';
import { decryptKey } from '@/lib/server/encryption';
import type { Provider } from '@/lib/llm/provider';

// Cost per 1K tokens for each model (in USD)
const MODEL_COSTS: Record<string, { input: number; output: number }> = {
  'gpt-4': { input: 0.03, output: 0.06 },
  'gpt-4-turbo': { input: 0.01, output: 0.03 },
  'gpt-3.5-turbo': { input: 0.0005, output: 0.0015 },
  'claude-3-opus': { input: 0.015, output: 0.075 },
  'claude-3-sonnet': { input: 0.003, output: 0.015 },
  'claude-3-haiku': { input: 0.00025, output: 0.00125 },
  'gemini-pro': { input: 0.0005, output: 0.0015 },
  'openrouter': { input: 0.01, output: 0.02 },
};

interface ChatRequest {
  message: string;
  conversationId: string;
  provider: Provider;
  model: string;
  systemPrompt?: string;
  history: Array<{ role: 'user' | 'assistant'; content: string }>;
}

export async function POST(req: NextRequest) {
  try {
    // Authenticate user
    const supabase = createClient();
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    
    if (authError || !user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    // Parse request
    const body: ChatRequest = await req.json();
    const { message, conversationId, provider, model, systemPrompt, history } = body;

    if (!message || !provider || !model) {
      return NextResponse.json({ error: 'Missing required fields' }, { status: 400 });
    }

    // Get user's API key from database
    const { data: keyData, error: keyError } = await (supabase as any)
      .from('api_keys')
      .select('encrypted_key')
      .eq('user_id', user.id)
      .eq('provider', provider)
      .eq('is_default', true)
      .eq('is_active', true)
      .single();

    if (keyError || !keyData) {
      return NextResponse.json(
        { error: `No API key configured for ${provider}. Please add one in settings.` },
        { status: 400 }
      );
    }

    // Decrypt API key (server-side only)
    let apiKey: string;
    try {
      apiKey = decryptKey(keyData.encrypted_key);
    } catch (error) {
      console.error('Failed to decrypt API key:', error);
      return NextResponse.json(
        { error: 'Failed to decrypt API key. Please re-add your API key in settings.' },
        { status: 500 }
      );
    }

    // Initialize provider client
    const client = initializeClient(provider, apiKey);

    // Create streaming response
    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      async start(controller) {
        try {
          let fullResponse = '';
          let inputTokens = 0;
          let outputTokens = 0;

          // Stream based on provider
          switch (provider) {
            case 'openai':
            case 'openrouter':
              await streamOpenAI(
                client as OpenAI,
                model,
                message,
                history,
                systemPrompt,
                (token) => {
                  fullResponse += token;
                  controller.enqueue(encoder.encode(`data: ${JSON.stringify({ token })}

`));
                },
                (usage) => {
                  inputTokens = usage.input;
                  outputTokens = usage.output;
                }
              );
              break;

            case 'anthropic':
              await streamAnthropic(
                client as Anthropic,
                model,
                message,
                history,
                systemPrompt,
                (token) => {
                  fullResponse += token;
                  controller.enqueue(encoder.encode(`data: ${JSON.stringify({ token })}

`));
                },
                (usage) => {
                  inputTokens = usage.input;
                  outputTokens = usage.output;
                }
              );
              break;

            case 'google':
              await streamGoogle(
                client as GoogleGenerativeAI,
                model,
                message,
                history,
                systemPrompt,
                (token) => {
                  fullResponse += token;
                  controller.enqueue(encoder.encode(`data: ${JSON.stringify({ token })}

`));
                },
                (usage) => {
                  inputTokens = usage.input;
                  outputTokens = usage.output;
                }
              );
              break;

            default:
              throw new Error(`Unsupported provider: ${provider}`);
          }

          // Calculate cost
          const costs = MODEL_COSTS[model] || MODEL_COSTS['openrouter'];
          const inputCost = (inputTokens / 1000) * costs.input;
          const outputCost = (outputTokens / 1000) * costs.output;
          const totalCost = parseFloat((inputCost + outputCost).toFixed(4));

          // Send completion event with usage stats
          controller.enqueue(
            encoder.encode(
              `data: ${JSON.stringify({
                done: true,
                fullResponse,
                usage: {
                  input: inputTokens,
                  output: outputTokens,
                  total: inputTokens + outputTokens,
                  cost: totalCost,
                },
              })}

`
            )
          );

          controller.close();
        } catch (error) {
          console.error('Streaming error:', error);
          controller.enqueue(
            encoder.encode(
              `data: ${JSON.stringify({ error: 'Streaming failed' })}

`
            )
          );
          controller.close();
        }
      },
    });

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        Connection: 'keep-alive',
      },
    });
  } catch (error) {
    console.error('Chat API error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

function initializeClient(provider: Provider, apiKey: string): any {
  switch (provider) {
    case 'openai':
      return new OpenAI({ apiKey });
    case 'anthropic':
      return new Anthropic({ apiKey });
    case 'google':
      return new GoogleGenerativeAI(apiKey);
    case 'openrouter':
      return new OpenAI({
        apiKey,
        baseURL: 'https://openrouter.ai/api/v1',
      });
    default:
      throw new Error(`Unsupported provider: ${provider}`);
  }
}

async function streamOpenAI(
  client: OpenAI,
  model: string,
  message: string,
  history: Array<{ role: 'user' | 'assistant'; content: string }>,
  systemPrompt: string | undefined,
  onToken: (token: string) => void,
  onComplete: (usage: { input: number; output: number }) => void
) {
  const messages = systemPrompt
    ? [{ role: 'system', content: systemPrompt }, ...history, { role: 'user', content: message }]
    : [...history, { role: 'user', content: message }];

  const stream = await client.chat.completions.create({
    model,
    messages: messages as any,
    temperature: 0.7,
    max_tokens: 2000,
    stream: true,
  });

  let fullResponse = '';
  let inputTokens = 0;
  let outputTokens = 0;

  for await (const chunk of stream) {
    const content = chunk.choices[0]?.delta?.content || '';
    if (content) {
      fullResponse += content;
      onToken(content);
    }
    if (chunk.usage) {
      inputTokens = chunk.usage.prompt_tokens;
      outputTokens = chunk.usage.completion_tokens;
    }
  }

  // Estimate tokens if not provided
  if (inputTokens === 0) {
    inputTokens = Math.ceil(messages.map((m) => m.content).join('').length / 4);
    outputTokens = Math.ceil(fullResponse.length / 4);
  }

  onComplete({ input: inputTokens, output: outputTokens });
}

async function streamAnthropic(
  client: Anthropic,
  model: string,
  message: string,
  history: Array<{ role: 'user' | 'assistant'; content: string }>,
  systemPrompt: string | undefined,
  onToken: (token: string) => void,
  onComplete: (usage: { input: number; output: number }) => void
) {
  const messages = [...history, { role: 'user', content: message }];

  const stream = await client.messages.create({
    model,
    max_tokens: 2000,
    temperature: 0.7,
    system: systemPrompt,
    messages: messages.map((m) => ({
      role: m.role,
      content: m.content,
    })) as any,
    stream: true,
  });

  let fullResponse = '';
  let inputTokens = 0;
  let outputTokens = 0;

  for await (const chunk of stream) {
    if (chunk.type === 'content_block_delta') {
      const content = (chunk.delta as any).text || '';
      fullResponse += content;
      onToken(content);
    }
    if (chunk.type === 'message_start') {
      inputTokens = (chunk.message as any).usage?.input_tokens || 0;
    }
    if (chunk.type === 'message_delta') {
      outputTokens = (chunk.usage as any)?.output_tokens || 0;
    }
  }

  if (outputTokens === 0) {
    outputTokens = Math.ceil(fullResponse.length / 4);
  }

  onComplete({ input: inputTokens, output: outputTokens });
}

async function streamGoogle(
  client: GoogleGenerativeAI,
  model: string,
  message: string,
  history: Array<{ role: 'user' | 'assistant'; content: string }>,
  systemPrompt: string | undefined,
  onToken: (token: string) => void,
  onComplete: (usage: { input: number; output: number }) => void
) {
  const genModel = client.getGenerativeModel({
    model,
    systemInstruction: systemPrompt,
  });

  const chat = genModel.startChat({
    history: history.map((m) => ({
      role: m.role === 'user' ? 'user' : 'model',
      parts: [{ text: m.content }],
    })),
  });

  const result = await chat.sendMessageStream(message);

  let fullResponse = '';

  for await (const chunk of result.stream) {
    const content = chunk.text();
    fullResponse += content;
    onToken(content);
  }

  const inputTokens = Math.ceil(
    [...history.map((m) => m.content), message].join('').length / 4
  );
  const outputTokens = Math.ceil(fullResponse.length / 4);

  onComplete({ input: inputTokens, output: outputTokens });
}
import OpenAI from 'openai';
import Anthropic from '@anthropic-ai/sdk';
import { GoogleGenerativeAI } from '@google/generative-ai';

export type Provider = 'openai' | 'anthropic' | 'google' | 'openrouter';

export interface ModelConfig {
  provider: Provider;
  model: string;
  temperature?: number;
  maxTokens?: number;
}

export interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  name?: string;
}

export interface StreamingCallbacks {
  onToken: (token: string, agentName?: string) => void;
  onComplete: (fullResponse: string, usage: TokenUsage) => void;
  onError: (error: Error) => void;
}

export interface TokenUsage {
  input: number;
  output: number;
  total: number;
  cost: number;
}

// Cost per 1K tokens for each model (in USD)
const MODEL_COSTS: Record<string, { input: number; output: number }> = {
  // OpenAI
  'gpt-4': { input: 0.03, output: 0.06 },
  'gpt-4-turbo': { input: 0.01, output: 0.03 },
  'gpt-3.5-turbo': { input: 0.0005, output: 0.0015 },
  // Anthropic
  'claude-3-opus': { input: 0.015, output: 0.075 },
  'claude-3-sonnet': { input: 0.003, output: 0.015 },
  'claude-3-haiku': { input: 0.00025, output: 0.00125 },
  // Google
  'gemini-pro': { input: 0.0005, output: 0.0015 },
  // OpenRouter (approximate, varies by model)
  'openrouter': { input: 0.01, output: 0.02 },
};

export class LLMProvider {
  private clients: Map<Provider, any> = new Map();
  private apiKeys: Map<Provider, string> = new Map();

  setApiKey(provider: Provider, key: string) {
    this.apiKeys.set(provider, key);
    this.initializeClient(provider, key);
  }

  private initializeClient(provider: Provider, key: string) {
    switch (provider) {
      case 'openai':
        this.clients.set(provider, new OpenAI({ apiKey: key }));
        break;
      case 'anthropic':
        this.clients.set(provider, new Anthropic({ apiKey: key }));
        break;
      case 'google':
        this.clients.set(provider, new GoogleGenerativeAI(key));
        break;
      case 'openrouter':
        this.clients.set(provider, new OpenAI({ 
          apiKey: key,
          baseURL: 'https://openrouter.ai/api/v1'
        }));
        break;
    }
  }

  async streamChat(
    config: ModelConfig,
    messages: Message[],
    callbacks: StreamingCallbacks,
    systemPrompt?: string
  ): Promise<void> {
    const client = this.clients.get(config.provider);
    if (!client) {
      throw new Error(`No API key set for provider: ${config.provider}`);
    }

    try {
      switch (config.provider) {
        case 'openai':
        case 'openrouter':
          await this.streamOpenAI(client, config, messages, callbacks, systemPrompt);
          break;
        case 'anthropic':
          await this.streamAnthropic(client, config, messages, callbacks, systemPrompt);
          break;
        case 'google':
          await this.streamGoogle(client, config, messages, callbacks, systemPrompt);
          break;
      }
    } catch (error) {
      callbacks.onError(error as Error);
    }
  }

  private async streamOpenAI(
    client: OpenAI,
    config: ModelConfig,
    messages: Message[],
    callbacks: StreamingCallbacks,
    systemPrompt?: string
  ): Promise<void> {
    const formattedMessages = systemPrompt 
      ? [{ role: 'system', content: systemPrompt }, ...messages]
      : messages;

    const stream = await client.chat.completions.create({
      model: config.model,
      messages: formattedMessages as any,
      temperature: config.temperature ?? 0.7,
      max_tokens: config.maxTokens ?? 2000,
      stream: true,
    });

    let fullResponse = '';
    let inputTokens = 0;
    let outputTokens = 0;

    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content || '';
      if (content) {
        fullResponse += content;
        callbacks.onToken(content);
      }
      
      if (chunk.usage) {
        inputTokens = chunk.usage.prompt_tokens;
        outputTokens = chunk.usage.completion_tokens;
      }
    }

    // Estimate tokens if not provided
    if (inputTokens === 0) {
      inputTokens = this.estimateTokens(formattedMessages.map(m => m.content).join(''));
      outputTokens = this.estimateTokens(fullResponse);
    }

    const usage = this.calculateUsage(config.model, inputTokens, outputTokens);
    callbacks.onComplete(fullResponse, usage);
  }

  private async streamAnthropic(
    client: Anthropic,
    config: ModelConfig,
    messages: Message[],
    callbacks: StreamingCallbacks,
    systemPrompt?: string
  ): Promise<void> {
    const system = systemPrompt || messages.find(m => m.role === 'system')?.content;
    const chatMessages = messages.filter(m => m.role !== 'system');

    const stream = await client.messages.create({
      model: config.model,
      max_tokens: config.maxTokens ?? 2000,
      temperature: config.temperature ?? 0.7,
      system: system,
      messages: chatMessages.map(m => ({
        role: m.role as 'user' | 'assistant',
        content: m.content
      })),
      stream: true,
    });

    let fullResponse = '';
    let inputTokens = 0;
    let outputTokens = 0;

    for await (const chunk of stream) {
      if (chunk.type === 'content_block_delta') {
        const content = (chunk.delta as any).text || '';
        fullResponse += content;
        callbacks.onToken(content);
      }
      if (chunk.type === 'message_start') {
        inputTokens = (chunk.message as any).usage?.input_tokens || 0;
      }
      if (chunk.type === 'message_delta') {
        outputTokens = (chunk.usage as any)?.output_tokens || 0;
      }
    }

    if (outputTokens === 0) {
      outputTokens = this.estimateTokens(fullResponse);
    }

    const usage = this.calculateUsage(config.model, inputTokens, outputTokens);
    callbacks.onComplete(fullResponse, usage);
  }

  private async streamGoogle(
    client: GoogleGenerativeAI,
    config: ModelConfig,
    messages: Message[],
    callbacks: StreamingCallbacks,
    systemPrompt?: string
  ): Promise<void> {
    const model = client.getGenerativeModel({ 
      model: config.model,
      systemInstruction: systemPrompt
    });

    const chat = model.startChat({
      history: messages.slice(0, -1).map(m => ({
        role: m.role === 'user' ? 'user' : 'model',
        parts: [{ text: m.content }]
      }))
    });

    const lastMessage = messages[messages.length - 1];
    const result = await chat.sendMessageStream(lastMessage.content);

    let fullResponse = '';

    for await (const chunk of result.stream) {
      const content = chunk.text();
      fullResponse += content;
      callbacks.onToken(content);
    }

    const inputTokens = this.estimateTokens(
      messages.map(m => m.content).join('')
    );
    const outputTokens = this.estimateTokens(fullResponse);
    const usage = this.calculateUsage(config.model, inputTokens, outputTokens);
    
    callbacks.onComplete(fullResponse, usage);
  }

  private estimateTokens(text: string): number {
    // Rough estimate: ~4 characters per token
    return Math.ceil(text.length / 4);
  }

  private calculateUsage(model: string, input: number, output: number): TokenUsage {
    const costs = MODEL_COSTS[model] || MODEL_COSTS['openrouter'];
    const inputCost = (input / 1000) * costs.input;
    const outputCost = (output / 1000) * costs.output;
    
    return {
      input,
      output,
      total: input + output,
      cost: parseFloat((inputCost + outputCost).toFixed(4))
    };
  }

  validateApiKey(provider: Provider, key: string): Promise<boolean> {
    this.setApiKey(provider, key);
    const client = this.clients.get(provider);
    
    if (!client) return Promise.resolve(false);

    // Quick validation call
    switch (provider) {
      case 'openai':
      case 'openrouter':
        return client.models.list().then(() => true).catch(() => false);
      case 'anthropic':
        return client.messages.create({
          model: 'claude-3-haiku',
          max_tokens: 1,
          messages: [{ role: 'user', content: 'Hi' }]
        }).then(() => true).catch(() => false);
      case 'google':
        return Promise.resolve(true); // Google doesn't have a simple validation
      default:
        return Promise.resolve(false);
    }
  }
}

// Singleton instance
export const llmProvider = new LLMProvider();

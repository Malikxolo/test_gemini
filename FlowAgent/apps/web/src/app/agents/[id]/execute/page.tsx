'use client';

import { useState, useRef, useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { MiniMaxLayout } from '@/components/layout/MiniMaxLayout';
import { api, Agent as ApiAgent, User as ApiUser } from '@/lib/api';
import { toast } from 'sonner';
import {
  Send,
  Loader2,
  Bot,
  User,
  Sparkles,
  Settings,
  ChevronLeft,
  Copy,
  Check,
  MoreHorizontal,
  Trash2,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
}

export default function AgentExecutionPage() {
  const params = useParams();
  const router = useRouter();
  const agentId = params.id as string;
  
  const [agent, setAgent] = useState<ApiAgent | null>(null);
  const [user, setUser] = useState<ApiUser | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [isExecuting, setIsExecuting] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [showMenu, setShowMenu] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    loadAgent();
  }, [agentId]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const loadAgent = async () => {
    try {
      setLoading(true);
      const [agentData, userData] = await Promise.all([
        api.agents.get(agentId),
        api.auth.getCurrentUser(),
      ]);
      setAgent(agentData);
      setUser(userData);
    } catch (error) {
      toast.error('Failed to load agent');
      router.push('/agents');
    } finally {
      setLoading(false);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isExecuting) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsExecuting(true);

    const assistantMessage: Message = {
      id: (Date.now() + 1).toString(),
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      isStreaming: true,
    };

    setMessages(prev => [...prev, assistantMessage]);

    try {
      const response = await api.agents.execute(agentId, { message: userMessage.content }, false);
      
      setMessages(prev =>
        prev.map(msg =>
          msg.id === assistantMessage.id
            ? { ...msg, content: response.output || 'No response', isStreaming: false }
            : msg
        )
      );
    } catch (error: any) {
      setMessages(prev =>
        prev.map(msg =>
          msg.id === assistantMessage.id
            ? { ...msg, content: 'Error: ' + (error.message || 'Failed to execute'), isStreaming: false }
            : msg
        )
      );
    } finally {
      setIsExecuting(false);
      inputRef.current?.focus();
    }
  };

  const copyToClipboard = async (content: string, id: string) => {
    await navigator.clipboard.writeText(content);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const clearChat = () => {
    if (confirm('Are you sure you want to clear the chat?')) {
      setMessages([]);
    }
  };

  const getModelColor = (model: string) => {
    if (model?.includes('gpt-4')) return 'from-purple-500 to-purple-600';
    if (model?.includes('gpt-3.5')) return 'from-blue-500 to-blue-600';
    if (model?.includes('claude')) return 'from-orange-500 to-orange-600';
    return 'from-gray-500 to-gray-600';
  };

  if (loading) {
    return (
      <MiniMaxLayout>
        <div className="flex items-center justify-center h-full">
          <Loader2 className="h-8 w-8 animate-spin text-white" />
        </div>
      </MiniMaxLayout>
    );
  }

  if (!agent) {
    return null;
  }

  return (
    <MiniMaxLayout>
      <div className="flex flex-col h-full bg-[#0d0d0d]">
        {/* Header */}
        <div className="border-b border-white/5 bg-[#0d0d0d] px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <button
                onClick={() => router.push('/agents')}
                className="flex items-center gap-1 text-white/60 hover:text-white transition-colors"
              >
                <ChevronLeft className="w-4 h-4" />
                Back
              </button>
              
              <div className="h-6 w-px bg-white/10" />
              
              <div className="flex items-center gap-3">
                <div className={`w-10 h-10 rounded-xl bg-gradient-to-br ${getModelColor(agent.model)} flex items-center justify-center`}>
                  <Bot className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h1 className="font-semibold text-white">{agent.name}</h1>
                  <div className="flex items-center gap-2">
                    <span className="px-2 py-0.5 bg-white/10 text-white/70 text-xs rounded">{agent.model}</span>
                    {agent.isPublic && <span className="px-2 py-0.5 bg-green-500/20 text-green-400 text-xs rounded">Public</span>}
                  </div>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={() => router.push(`/agents/${agentId}`)}
                className="flex items-center gap-2 px-3 py-1.5 text-white/60 hover:text-white hover:bg-white/5 rounded-lg transition-colors"
              >
                <Settings className="w-4 h-4" />
                Settings
              </button>
              
              <div className="relative">
                <button 
                  onClick={() => setShowMenu(!showMenu)}
                  className="p-1.5 text-white/60 hover:text-white hover:bg-white/5 rounded-lg transition-colors"
                >
                  <MoreHorizontal className="w-4 h-4" />
                </button>
                
                {showMenu && (
                  <div className="absolute right-0 mt-2 w-48 bg-[#141414] border border-white/10 rounded-xl shadow-lg z-50">
                    <button
                      onClick={() => {
                        clearChat();
                        setShowMenu(false);
                      }}
                      className="w-full flex items-center gap-2 px-4 py-2 text-red-400 hover:bg-white/5 transition-colors first:rounded-t-xl last:rounded-b-xl"
                    >
                      <Trash2 className="w-4 h-4" />
                      Clear Chat
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-6">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="w-16 h-16 rounded-full bg-gradient-to-br from-blue-400 to-purple-500 flex items-center justify-center mb-4">
                <Sparkles className="w-8 h-8 text-white" />
              </div>
              <h2 className="text-xl font-semibold text-white mb-2">Start a conversation</h2>
              <p className="text-white/50 max-w-md mb-6">
                {agent.description || 'This agent is ready to help you. Type a message below to get started.'}
              </p>
              <div className="flex flex-wrap gap-2 justify-center">
                {['Hello!', 'What can you do?', 'Help me with...', 'Explain this to me'].map((suggestion) => (
                  <button
                    key={suggestion}
                    onClick={() => setInput(suggestion)}
                    className="px-4 py-2 bg-white/5 text-white/70 rounded-lg border border-white/10 hover:bg-white/10 hover:text-white transition-colors"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((message) => (
              <div
                key={message.id}
                className={`flex gap-4 ${message.role === 'user' ? 'flex-row-reverse' : ''}`}
              >
                <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                  message.role === 'user' ? 'bg-white/10' : 'bg-gradient-to-br from-blue-400 to-purple-500'
                }`}>
                  {message.role === 'user' ? (
                    <User className="w-4 h-4 text-white/70" />
                  ) : (
                    <Bot className="w-4 h-4 text-white" />
                  )}
                </div>
                
                <div className={`flex-1 max-w-3xl ${message.role === 'user' ? 'text-right' : ''}`}>
                  <div className={`inline-block rounded-2xl px-4 py-3 ${
                    message.role === 'user' 
                      ? 'bg-white/10 text-white' 
                      : 'bg-[#141414] border border-white/5 text-white'
                  }`}>
                    {message.isStreaming ? (
                      <div className="flex items-center gap-2">
                        <Loader2 className="w-4 h-4 animate-spin text-white/50" />
                        <span className="text-white/50">Thinking... (Using BYOK)</span>
                      </div>
                    ) : (
                      <div className="prose prose-sm max-w-none prose-invert">
                        <ReactMarkdown>{message.content}</ReactMarkdown>
                      </div>
                    )}
                    
                    <div className="flex items-center justify-end gap-2 mt-2">
                      <span className="text-xs text-white/30">
                        {message.timestamp.toLocaleTimeString()}
                      </span>
                      {!message.isStreaming && (
                        <button
                          onClick={() => copyToClipboard(message.content, message.id)}
                          className="p-1 hover:bg-white/5 rounded transition-colors"
                        >
                          {copiedId === message.id ? (
                            <Check className="w-3 h-3 text-green-400" />
                          ) : (
                            <Copy className="w-3 h-3 text-white/40" />
                          )}
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="border-t border-white/5 bg-[#0d0d0d] px-6 py-4">
          <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
            <div className="relative">
              <input
                ref={inputRef}
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Type your message..."
                disabled={isExecuting}
                className="w-full bg-[#141414] border border-white/10 rounded-2xl px-5 py-4 pr-14 text-white placeholder-white/30 focus:outline-none focus:border-white/20"
              />
              <button
                type="submit"
                disabled={!input.trim() || isExecuting}
                className="absolute right-2 top-1/2 -translate-y-1/2 p-2 bg-white text-black rounded-xl disabled:opacity-30 transition-colors"
              >
                {isExecuting ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
              </button>
            </div>
            
            <p className="text-xs text-white/30 mt-2 text-center">
              This agent uses your BYOK API keys. You pay directly to OpenAI/Anthropic.
            </p>
          </form>
        </div>
      </div>
    </MiniMaxLayout>
  );
}

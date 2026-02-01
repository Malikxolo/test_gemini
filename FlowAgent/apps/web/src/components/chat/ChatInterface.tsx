'use client';

import { useState, useRef, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { MiniMaxLayout } from '@/components/layout/MiniMaxLayout';
import { useSupabase } from '@/components/providers/supabase-provider';
import { 
  createConversation, 
  addMessage, 
  getMessages,
  checkBudget,
  streamChatMessage,
  type Message 
} from '@/lib/llm/conversations';
import { AgentSwarm, PREBUILT_AGENTS } from '@/lib/llm/swarm';
import { hasDefaultApiKey } from '@/lib/llm/api-keys';
import { uploadFiles, type UploadedFile } from '@/lib/llm/file-upload';
import { 
  Send, 
  Loader2, 
  Paperclip, 
  Bot, 
  User,
  AlertCircle,
  ChevronDown,
  Settings,
  FileText
} from 'lucide-react';
import { toast } from 'sonner';

interface ChatPageProps {
  conversationId?: string;
  projectId?: string;
  agentId?: string;
}

export default function ChatPage({ conversationId, projectId, agentId }: ChatPageProps) {
  const router = useRouter();
  const { user, isLoading: authLoading } = useSupabase();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentConvId, setCurrentConvId] = useState<string | null>(conversationId || null);
  const [budget, setBudget] = useState({ used: 0, limit: 5, remaining: 5 });
  const [selectedModel, setSelectedModel] = useState('gpt-4');
  const [selectedProvider, setSelectedProvider] = useState('openai');
  const [attachments, setAttachments] = useState<File[]>([]);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [showModelSelector, setShowModelSelector] = useState(false);
  const [activeAgent, setActiveAgent] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Check for API key on mount
  useEffect(() => {
    checkApiKey();
  }, [selectedProvider]);

  const checkApiKey = async () => {
    const hasKey = await hasDefaultApiKey(selectedProvider as any);
    if (!hasKey) {
      console.log(`No API key configured for ${selectedProvider}`);
    }
  };

  // Load existing conversation
  useEffect(() => {
    if (currentConvId) {
      loadConversation();
    }
  }, [currentConvId]);

  // Check for pending message from dashboard
  useEffect(() => {
    const pendingMessage = localStorage.getItem('pendingMessage');
    if (pendingMessage && !currentConvId) {
      localStorage.removeItem('pendingMessage');
      setInput(pendingMessage);
      // Auto-send after a short delay to ensure everything is loaded
      setTimeout(() => {
        handleSend();
      }, 500);
    }
  }, [currentConvId]);

  // Scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const loadConversation = async () => {
    if (!currentConvId) return;
    try {
      const msgs = await getMessages(currentConvId);
      setMessages(msgs);
      const budgetStatus = await checkBudget(currentConvId);
      setBudget({
        used: budgetStatus.used,
        limit: budgetStatus.limit,
        remaining: budgetStatus.remaining
      });
    } catch (error) {
      toast.error('Failed to load conversation');
    }
  };

  const handleSend = async () => {
    if (!input.trim() || isStreaming) return;

    // Check budget
    if (currentConvId) {
      const budgetStatus = await checkBudget(currentConvId);
      if (!budgetStatus.withinBudget) {
        toast.error('Budget exceeded! ($5 limit per conversation)');
        return;
      }
    }

    // Check API key
    const hasKey = await hasDefaultApiKey(selectedProvider as any);
    if (!hasKey) {
      toast.error(`No API key configured for ${selectedProvider}. Please add one in settings.`);
      return;
    }

    setIsStreaming(true);
    const userMessage = input.trim();
    setInput('');

    try {
      // Create conversation if new
      let convId = currentConvId;
      if (!convId) {
        const conv = await createConversation(
          userMessage.slice(0, 50) + '...',
          selectedModel,
          selectedProvider as any,
          projectId
        );
        convId = conv.id;
        setCurrentConvId(convId);
        
        // Update URL without reload
        router.push(`/chat/${convId}`, { scroll: false });
      }

      // Add user message
      await addMessage(convId, 'user', userMessage);
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        conversation_id: convId,
        role: 'user',
        content: userMessage,
        created_at: new Date().toISOString(),
        tokens_used: 0,
        cost: 0
      }]);

      // Initialize swarm
      const swarm = new AgentSwarm(convId, {
        provider: selectedProvider as any,
        model: selectedModel
      }, budget.remaining);

      // Process with swarm
      let streamingContent = '';
      let currentAgent: string | null = null;

      // Convert messages to chat history format
      const chatHistory = messages
        .filter(m => m.role === 'user' || m.role === 'assistant')
        .map(m => ({ role: m.role as 'user' | 'assistant', content: m.content }));

      const result = await swarm.processMessage(userMessage, chatHistory, (update: { agent: string; status: 'thinking' | 'working' | 'complete'; partial?: string; fullResponse?: string }) => {
        if (update.status === 'thinking' || update.status === 'working') {
          currentAgent = update.agent;
          setActiveAgent(update.agent);
          
          if (update.partial) {
            streamingContent = update.partial;
            setMessages(prev => {
              const lastMsg = prev[prev.length - 1];
              if (lastMsg?.role === 'assistant' && lastMsg.agent_name === update.agent) {
                return [...prev.slice(0, -1), {
                  ...lastMsg,
                  content: streamingContent
                }];
              }
              return [...prev, {
                id: Date.now().toString(),
                conversation_id: convId!,
                role: 'assistant',
                content: streamingContent,
                agent_name: update.agent,
                created_at: new Date().toISOString(),
                tokens_used: 0,
                cost: 0
              }];
            });
          }
        } else if (update.status === 'complete') {
          streamingContent = update.fullResponse || '';
          setMessages(prev => {
            const lastMsg = prev[prev.length - 1];
            if (lastMsg?.role === 'assistant') {
              return [...prev.slice(0, -1), {
                ...lastMsg,
                content: streamingContent
              }];
            }
            return prev;
          });
          setActiveAgent(null);
        }
      });

      // Save the final response to the database
      await addMessage(convId, 'assistant', result.response, {
        agent_name: currentAgent || 'Project Manager',
        tokens_used: result.tokens,
        cost: result.cost,
        model: selectedModel,
        provider: selectedProvider
      });

      // Reload to get final stats
      await loadConversation();

    } catch (error: any) {
      toast.error(error.message || 'Failed to send message');
    } finally {
      setIsStreaming(false);
      setActiveAgent(null);
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    const totalSize = files.reduce((acc, f) => acc + f.size, 0);
    
    if (totalSize > 50 * 1024 * 1024) {
      toast.error('Total file size must be under 50MB');
      return;
    }
    
    setAttachments(files);
    
    // Upload files immediately if we have a conversation
    if (currentConvId) {
      setIsUploading(true);
      try {
        const uploaded = await uploadFiles(files, currentConvId);
        setUploadedFiles(prev => [...prev, ...uploaded]);
        toast.success(`${uploaded.length} file(s) uploaded successfully`);
      } catch (error) {
        toast.error(error instanceof Error ? error.message : 'Failed to upload files');
        setAttachments([]);
      } finally {
        setIsUploading(false);
      }
    }
  };

  const removeAttachment = (index: number) => {
    setAttachments(prev => prev.filter((_, i) => i !== index));
    setUploadedFiles(prev => prev.filter((_, i) => i !== index));
  };

  if (authLoading) {
    return (
      <MiniMaxLayout>
        <div className="flex h-full items-center justify-center">
          <Loader2 className="w-8 h-8 animate-spin text-white" />
        </div>
      </MiniMaxLayout>
    );
  }

  return (
    <MiniMaxLayout>
      <div className="flex flex-col h-full bg-[#0d0d0d]">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-3 border-b border-white/5">
          <div className="flex items-center gap-3">
            <span className="text-sm text-white/60">Chat</span>
            {activeAgent && (
              <span className="px-2 py-1 bg-blue-500/20 text-blue-400 text-xs rounded-full animate-pulse">
                {activeAgent} is thinking...
              </span>
            )}
          </div>
          
          <div className="flex items-center gap-4">
            {/* Budget Indicator */}
            <div className="flex items-center gap-2 px-3 py-1.5 bg-white/5 rounded-full">
              <span className="text-xs text-white/60">Budget:</span>
              <span className={`text-xs font-medium ${budget.remaining < 1 ? 'text-red-400' : 'text-green-400'}`}>
                ${budget.used.toFixed(2)} / ${budget.limit}
              </span>
            </div>
            
            {/* Model Selector */}
            <div className="relative">
              <button
                onClick={() => setShowModelSelector(!showModelSelector)}
                className="flex items-center gap-2 px-3 py-1.5 bg-white/5 hover:bg-white/10 rounded-lg text-sm text-white/80 transition-colors"
              >
                <Settings className="w-4 h-4" />
                <span>{selectedModel}</span>
                <ChevronDown className="w-3 h-3" />
              </button>
              
              {showModelSelector && (
                <div className="absolute right-0 top-full mt-2 w-64 bg-[#1a1a1a] border border-white/10 rounded-xl p-2 z-50">
                  <div className="text-xs text-white/40 px-2 py-1">Provider</div>
                  {['openai', 'anthropic', 'google', 'openrouter'].map(provider => (
                    <button
                      key={provider}
                      onClick={() => {
                        setSelectedProvider(provider);
                        setSelectedModel(
                          provider === 'openai' ? 'gpt-4' :
                          provider === 'anthropic' ? 'claude-3-opus' :
                          provider === 'google' ? 'gemini-pro' : 'openrouter'
                        );
                      }}
                      className={`w-full text-left px-3 py-2 rounded-lg text-sm capitalize ${
                        selectedProvider === provider ? 'bg-white/10 text-white' : 'text-white/60 hover:bg-white/5'
                      }`}
                    >
                      {provider}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-6">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center mb-4">
                <Bot className="w-8 h-8 text-white" />
              </div>
              <h2 className="text-2xl font-semibold text-white mb-2">Start a conversation</h2>
              <p className="text-white/50 max-w-md mb-6">
                Ask anything! The Project Manager will coordinate with specialized agents to help you.
              </p>
              <div className="flex flex-wrap gap-2 justify-center">
                {[
                  'Build a landing page',
                  'Analyze this data',
                  'Set up cloud infrastructure',
                  'Review my code'
                ].map(suggestion => (
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
            messages.map((message, index) => (
              <div
                key={message.id || index}
                className={`flex gap-4 ${message.role === 'user' ? 'flex-row-reverse' : ''}`}
              >
                <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                  message.role === 'user' 
                    ? 'bg-white/10' 
                    : 'bg-gradient-to-br from-blue-400 to-purple-500'
                }`}>
                  {message.role === 'user' ? (
                    <User className="w-4 h-4 text-white/70" />
                  ) : (
                    <Bot className="w-4 h-4 text-white" />
                  )}
                </div>
                
                <div className={`flex-1 max-w-3xl ${message.role === 'user' ? 'text-right' : ''}`}>
                  <div className={`inline-block rounded-2xl px-4 py-3 text-left ${
                    message.role === 'user' 
                      ? 'bg-white/10 text-white' 
                      : 'bg-[#141414] border border-white/5 text-white'
                  }`}>
                    {message.agent_name && message.role === 'assistant' && (
                      <div className="text-xs text-blue-400 mb-1 font-medium">
                        {message.agent_name}
                      </div>
                    )}
                    <div className="prose prose-sm max-w-none prose-invert">
                      {message.content}
                    </div>
                  </div>
                  
                  {message.cost > 0 && (
                    <div className="text-xs text-white/30 mt-1">
                      ${message.cost.toFixed(4)} • {message.tokens_used} tokens
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
          
          {isStreaming && !activeAgent && (
            <div className="flex gap-4">
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-400 to-purple-500 flex items-center justify-center">
                <Bot className="w-4 h-4 text-white" />
              </div>
              <div className="bg-[#141414] border border-white/5 rounded-2xl px-4 py-3">
                <Loader2 className="w-5 h-5 animate-spin text-white/50" />
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="border-t border-white/5 px-6 py-4">
          {budget.remaining < 1 && (
            <div className="mb-3 flex items-center gap-2 px-4 py-2 bg-red-500/10 border border-red-500/20 rounded-xl">
              <AlertCircle className="w-4 h-4 text-red-400" />
              <span className="text-sm text-red-400">
                Low budget remaining: ${budget.remaining.toFixed(2)}
              </span>
            </div>
          )}
          
          {(attachments.length > 0 || uploadedFiles.length > 0) && (
            <div className="mb-3 flex gap-2 flex-wrap">
              {attachments.map((file, index) => (
                <div key={`file-${index}`} className="flex items-center gap-2 px-3 py-1.5 bg-white/5 rounded-lg text-sm">
                  <FileText className="w-4 h-4 text-white/50" />
                  <span className="text-white/70 truncate max-w-[150px]">{file.name}</span>
                  {isUploading && <Loader2 className="w-3 h-3 animate-spin text-white/50" />}
                  <button 
                    onClick={() => removeAttachment(index)}
                    className="text-white/40 hover:text-white"
                    disabled={isUploading}
                  >
                    ×
                  </button>
                </div>
              ))}
              {uploadedFiles.map((file, index) => (
                <div key={`uploaded-${index}`} className="flex items-center gap-2 px-3 py-1.5 bg-green-500/10 border border-green-500/20 rounded-lg text-sm">
                  <FileText className="w-4 h-4 text-green-400" />
                  <span className="text-green-400 truncate max-w-[150px]">{file.name}</span>
                  <button 
                    onClick={() => removeAttachment(index)}
                    className="text-green-400/60 hover:text-green-400"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          )}
          
          <div className="flex items-end gap-2 bg-[#1a1a1a] rounded-2xl border border-white/10 p-2">
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isStreaming}
              className="p-2 hover:bg-white/5 rounded-xl transition-colors disabled:opacity-50"
            >
              <Paperclip className="w-5 h-5 text-white/50" />
            </button>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              onChange={handleFileSelect}
              className="hidden"
              accept=".txt,.pdf,.doc,.docx,.png,.jpg,.jpeg,.json,.js,.ts,.tsx,.py,.html,.css"
            />
            
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder="Type your message..."
              disabled={isStreaming}
              rows={1}
              className="flex-1 bg-transparent text-white placeholder-white/30 resize-none outline-none py-3 px-2 max-h-[200px]"
              style={{ minHeight: '44px' }}
            />
            
            <button
              onClick={handleSend}
              disabled={!input.trim() || isStreaming}
              className="p-2 bg-white text-black rounded-xl disabled:opacity-30 transition-colors"
            >
              {isStreaming ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
            </button>
          </div>
          
          <div className="mt-2 text-center text-xs text-white/30">
            Press Enter to send, Shift+Enter for new line • 50MB max file size
          </div>
        </div>
      </div>
    </MiniMaxLayout>
  );
}

'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { MiniMaxLayout } from '@/components/layout/MiniMaxLayout';
import { api, User } from '@/lib/api';
import { toast } from 'sonner';
import { ArrowLeft, Sparkles, Loader2, Bot, Wrench, Eye, Check } from 'lucide-react';

const models = [
  { id: 'gpt-4', name: 'GPT-4', description: 'Most capable model', provider: 'OpenAI', color: 'from-purple-500 to-purple-600' },
  { id: 'gpt-4-turbo', name: 'GPT-4 Turbo', description: 'Fast and capable', provider: 'OpenAI', color: 'from-purple-400 to-purple-500' },
  { id: 'gpt-3.5-turbo', name: 'GPT-3.5 Turbo', description: 'Fast and cost-effective', provider: 'OpenAI', color: 'from-blue-500 to-blue-600' },
  { id: 'claude-3-opus', name: 'Claude 3 Opus', description: 'Anthropic\'s best model', provider: 'Anthropic', color: 'from-orange-500 to-orange-600' },
  { id: 'claude-3-sonnet', name: 'Claude 3 Sonnet', description: 'Balanced performance', provider: 'Anthropic', color: 'from-orange-400 to-orange-500' },
];

const tools = [
  { id: 'web_search', name: 'Web Search', description: 'Search the internet for current information', icon: '🔍' },
  { id: 'code_execution', name: 'Code Execution', description: 'Execute Python code', icon: '💻' },
  { id: 'file_read', name: 'File Reader', description: 'Read and analyze files', icon: '📄' },
  { id: 'calculator', name: 'Calculator', description: 'Perform mathematical calculations', icon: '🧮' },
];

export default function CreateAgentPage() {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    systemPrompt: '',
    model: 'gpt-4',
    temperature: 0.7,
    maxTokens: 2000,
    tools: [] as string[],
    isPublic: false,
  });

  useEffect(() => {
    checkAuth();
  }, []);

  const checkAuth = async () => {
    try {
      const userData = await api.auth.getCurrentUser();
      setUser(userData);
    } catch (err: any) {
      if (err.message.includes('401') || err.message.includes('Not authenticated')) {
        router.push('/login');
      }
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!formData.name.trim()) {
      toast.error('Agent name is required');
      return;
    }
    
    if (!formData.systemPrompt.trim()) {
      toast.error('System prompt is required');
      return;
    }

    try {
      setLoading(true);
      await api.agents.create({
        name: formData.name,
        description: formData.description,
        systemPrompt: formData.systemPrompt,
        model: formData.model,
        temperature: formData.temperature,
        maxTokens: formData.maxTokens,
        tools: formData.tools,
        isPublic: formData.isPublic,
      });
      
      toast.success('Agent created successfully!');
      router.push('/agents');
    } catch (error: any) {
      toast.error(error.message || 'Failed to create agent');
    } finally {
      setLoading(false);
    }
  };

  const toggleTool = (toolId: string) => {
    setFormData(prev => ({
      ...prev,
      tools: prev.tools.includes(toolId)
        ? prev.tools.filter(t => t !== toolId)
        : [...prev.tools, toolId],
    }));
  };

  return (
    <MiniMaxLayout>
      <div className="h-full overflow-y-auto">
        <div className="max-w-5xl mx-auto px-6 py-8">
          {/* Header */}
          <div className="mb-8">
            <button
              onClick={() => router.push('/agents')}
              className="flex items-center gap-2 text-white/60 hover:text-white mb-4 transition-colors"
            >
              <ArrowLeft className="w-4 h-4" />
              Back to Agents
            </button>
            
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold text-white mb-2">Create New Agent</h1>
                <p className="text-white/50">Configure your AI agent with custom settings</p>
              </div>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Basic Info */}
            <div className="bg-[#141414] rounded-xl p-6 border border-white/5">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 bg-blue-500/20 rounded-xl flex items-center justify-center">
                  <Bot className="w-5 h-5 text-blue-400" />
                </div>
                <div>
                  <h2 className="text-xl font-semibold text-white">Basic Information</h2>
                  <p className="text-sm text-white/50">Give your agent a name and description</p>
                </div>
              </div>
              
              <div className="space-y-6">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-white">Agent Name *</label>
                  <input
                    type="text"
                    placeholder="e.g., Customer Support Bot"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    className="w-full bg-[#0d0d0d] border border-white/10 rounded-xl px-4 py-3 text-white placeholder-white/30 focus:outline-none focus:border-white/20"
                  />
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium text-white">Description</label>
                  <textarea
                    placeholder="What does this agent do?"
                    value={formData.description}
                    onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                    rows={3}
                    className="w-full bg-[#0d0d0d] border border-white/10 rounded-xl px-4 py-3 text-white placeholder-white/30 focus:outline-none focus:border-white/20 resize-none"
                  />
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium text-white">System Prompt *</label>
                  <textarea
                    placeholder="Instructions for how the agent should behave..."
                    value={formData.systemPrompt}
                    onChange={(e) => setFormData({ ...formData, systemPrompt: e.target.value })}
                    rows={6}
                    className="w-full bg-[#0d0d0d] border border-white/10 rounded-xl px-4 py-3 text-white placeholder-white/30 focus:outline-none focus:border-white/20 resize-none"
                  />
                  <p className="text-sm text-white/40">
                    The system prompt defines your agent&apos;s personality and capabilities
                  </p>
                </div>
              </div>
            </div>

            {/* Model Settings */}
            <div className="bg-[#141414] rounded-xl p-6 border border-white/5">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 bg-purple-500/20 rounded-xl flex items-center justify-center">
                  <Sparkles className="w-5 h-5 text-purple-400" />
                </div>
                <div>
                  <h2 className="text-xl font-semibold text-white">Model Settings</h2>
                  <p className="text-sm text-white/50">Choose the AI model and configure its behavior</p>
                </div>
              </div>
              
              <div className="space-y-6">
                <div className="space-y-4">
                  <label className="text-sm font-medium text-white">Select Model</label>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {models.map((model) => (
                      <div
                        key={model.id}
                        onClick={() => setFormData({ ...formData, model: model.id })}
                        className={`p-4 rounded-xl border-2 cursor-pointer transition-all ${
                          formData.model === model.id
                            ? 'border-white/30 bg-white/5'
                            : 'border-white/5 hover:border-white/10'
                        }`}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-semibold text-white">{model.name}</span>
                          <span className="px-2 py-0.5 bg-white/10 text-white/70 text-xs rounded">{model.provider}</span>
                        </div>
                        <p className="text-sm text-white/50">{model.description}</p>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-medium text-white">Temperature: {formData.temperature}</label>
                    <span className="text-sm text-white/50">
                      {formData.temperature < 0.3 ? 'Focused' : formData.temperature > 0.7 ? 'Creative' : 'Balanced'}
                    </span>
                  </div>
                  <input
                    type="range"
                    value={formData.temperature}
                    onChange={(e) => setFormData({ ...formData, temperature: parseFloat(e.target.value) })}
                    min={0}
                    max={2}
                    step={0.1}
                    className="w-full"
                  />
                  <p className="text-sm text-white/40">
                    Lower values make the agent more focused and deterministic
                  </p>
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium text-white">Max Tokens</label>
                  <input
                    type="number"
                    value={formData.maxTokens}
                    onChange={(e) => setFormData({ ...formData, maxTokens: parseInt(e.target.value) })}
                    min={100}
                    max={8000}
                    className="w-full bg-[#0d0d0d] border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-white/20"
                  />
                </div>
              </div>
            </div>

            {/* Tools */}
            <div className="bg-[#141414] rounded-xl p-6 border border-white/5">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 bg-orange-500/20 rounded-xl flex items-center justify-center">
                  <Wrench className="w-5 h-5 text-orange-400" />
                </div>
                <div>
                  <h2 className="text-xl font-semibold text-white">Tools & Capabilities</h2>
                  <p className="text-sm text-white/50">Select tools your agent can use</p>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {tools.map((tool) => (
                  <div
                    key={tool.id}
                    onClick={() => toggleTool(tool.id)}
                    className={`p-4 rounded-xl border-2 cursor-pointer transition-all ${
                      formData.tools.includes(tool.id)
                        ? 'border-white/30 bg-white/5'
                        : 'border-white/5 hover:border-white/10'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <div className={`w-5 h-5 rounded border-2 flex items-center justify-center ${
                        formData.tools.includes(tool.id)
                          ? 'bg-white border-white'
                          : 'border-white/30'
                      }`}>
                        {formData.tools.includes(tool.id) && (
                          <Check className="w-3 h-3 text-black" />
                        )}
                      </div>
                      <div>
                        <p className="font-medium text-white">{tool.name}</p>
                        <p className="text-sm text-white/50">{tool.description}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Visibility */}
            <div className="bg-[#141414] rounded-xl p-6 border border-white/5">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 bg-green-500/20 rounded-xl flex items-center justify-center">
                  <Eye className="w-5 h-5 text-green-400" />
                </div>
                <div>
                  <h2 className="text-xl font-semibold text-white">Visibility</h2>
                  <p className="text-sm text-white/50">Control who can access this agent</p>
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium text-white">Make this agent public</p>
                  <p className="text-sm text-white/50">Others can view and use this agent</p>
                </div>
                <button
                  type="button"
                  onClick={() => setFormData({ ...formData, isPublic: !formData.isPublic })}
                  className={`w-14 h-7 rounded-full transition-colors relative ${
                    formData.isPublic ? 'bg-white' : 'bg-white/20'
                  }`}
                >
                  <div className={`w-5 h-5 rounded-full bg-black absolute top-1 transition-all ${
                    formData.isPublic ? 'right-1' : 'left-1'
                  }`} />
                </button>
              </div>
            </div>

            {/* Actions */}
            <div className="flex gap-4">
              <button
                type="button"
                onClick={() => router.push('/agents')}
                className="px-6 py-3 bg-white/5 text-white rounded-xl font-medium hover:bg-white/10 transition-colors"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={loading}
                className="flex-1 px-6 py-3 bg-white text-black rounded-xl font-medium hover:bg-white/90 transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Creating Agent...
                  </>
                ) : (
                  'Create Agent'
                )}
              </button>
            </div>
          </form>
        </div>
      </div>
    </MiniMaxLayout>
  );
}

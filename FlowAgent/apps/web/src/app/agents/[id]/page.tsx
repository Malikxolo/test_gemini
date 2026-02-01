'use client';

import { useState, useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { MiniMaxLayout } from '@/components/layout/MiniMaxLayout';
import { api, Agent as ApiAgent, User as ApiUser } from '@/lib/api';
import { toast } from 'sonner';
import {
  ArrowLeft,
  Bot,
  Save,
  Loader2,
  Play,
  Trash2,
  Sparkles,
  Settings,
  Wrench,
  Eye,
  Check,
  ChevronRight,
} from 'lucide-react';

const models = [
  { id: 'gpt-4', name: 'GPT-4', provider: 'OpenAI', color: 'from-purple-500 to-purple-600' },
  { id: 'gpt-4-turbo', name: 'GPT-4 Turbo', provider: 'OpenAI', color: 'from-purple-400 to-purple-500' },
  { id: 'gpt-3.5-turbo', name: 'GPT-3.5 Turbo', provider: 'OpenAI', color: 'from-blue-500 to-blue-600' },
  { id: 'claude-3-opus', name: 'Claude 3 Opus', provider: 'Anthropic', color: 'from-orange-500 to-orange-600' },
  { id: 'claude-3-sonnet', name: 'Claude 3 Sonnet', provider: 'Anthropic', color: 'from-orange-400 to-orange-500' },
];

const tools = [
  { id: 'web_search', name: 'Web Search', description: 'Search the internet', icon: '🔍' },
  { id: 'code_execution', name: 'Code Execution', description: 'Execute Python code', icon: '💻' },
  { id: 'file_read', name: 'File Reader', description: 'Read files', icon: '📄' },
  { id: 'calculator', name: 'Calculator', description: 'Math calculations', icon: '🧮' },
];

export default function AgentDetailPage() {
  const params = useParams();
  const router = useRouter();
  const agentId = params.id as string;
  
  const [agent, setAgent] = useState<ApiAgent | null>(null);
  const [user, setUser] = useState<ApiUser | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [activeTab, setActiveTab] = useState('general');
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    systemPrompt: '',
    model: '',
    temperature: 0.7,
    maxTokens: 2000,
    tools: [] as string[],
    isPublic: false,
  });

  useEffect(() => {
    loadAgent();
  }, [agentId]);

  const loadAgent = async () => {
    try {
      setLoading(true);
      const [agentData, userData] = await Promise.all([
        api.agents.get(agentId),
        api.auth.getCurrentUser(),
      ]);
      setAgent(agentData);
      setUser(userData);
      setFormData({
        name: agentData.name,
        description: agentData.description || '',
        systemPrompt: agentData.systemPrompt,
        model: agentData.model,
        temperature: agentData.temperature,
        maxTokens: agentData.maxTokens,
        tools: agentData.tools || [],
        isPublic: agentData.isPublic,
      });
    } catch (error) {
      toast.error('Failed to load agent');
      router.push('/agents');
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    try {
      setSaving(true);
      await api.agents.update(agentId, {
        name: formData.name,
        description: formData.description,
        systemPrompt: formData.systemPrompt,
        model: formData.model,
        temperature: formData.temperature,
        maxTokens: formData.maxTokens,
        tools: formData.tools,
        isPublic: formData.isPublic,
      });
      toast.success('Agent updated successfully');
    } catch (error) {
      toast.error('Failed to update agent');
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!confirm('Are you sure you want to delete this agent? This action cannot be undone.')) return;
    
    try {
      await api.agents.delete(agentId);
      toast.success('Agent deleted successfully');
      router.push('/agents');
    } catch (error) {
      toast.error('Failed to delete agent');
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

  if (!agent) return null;

  return (
    <MiniMaxLayout>
      <div className="h-full overflow-y-auto">
        <div className="max-w-5xl mx-auto px-6 py-8">
          {/* Header */}
          <div className="mb-8">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <button
                  onClick={() => router.push('/agents')}
                  className="flex items-center gap-2 text-white/60 hover:text-white transition-colors"
                >
                  <ArrowLeft className="w-4 h-4" />
                  Back
                </button>
                
                <div className="h-6 w-px bg-white/10" />
                
                <div className="flex items-center gap-3">
                  <div className={`w-10 h-10 rounded-xl bg-gradient-to-br ${getModelColor(agent.model)} flex items-center justify-center`}>
                    <Bot className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <h1 className="text-2xl font-bold text-white">{agent.name}</h1>
                    <div className="flex items-center gap-2">
                      <span className="px-2 py-0.5 bg-white/10 text-white/70 text-xs rounded">{agent.model}</span>
                      {agent.isPublic && <span className="px-2 py-0.5 bg-green-500/20 text-green-400 text-xs rounded">Public</span>}
                    </div>
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={() => router.push(`/agents/${agentId}/execute`)}
                  className="flex items-center gap-2 px-4 py-2 bg-white/5 text-white rounded-xl hover:bg-white/10 transition-colors"
                >
                  <Play className="w-4 h-4" />
                  Test Agent
                </button>
                <button
                  onClick={handleDelete}
                  className="flex items-center gap-2 px-4 py-2 bg-red-500/20 text-red-400 rounded-xl hover:bg-red-500/30 transition-colors"
                >
                  <Trash2 className="w-4 h-4" />
                  Delete
                </button>
              </div>
            </div>
          </div>

          {/* Tabs */}
          <div className="flex gap-1 p-1 bg-[#141414] rounded-xl border border-white/5 mb-6 w-fit">
            {[
              { id: 'general', label: 'General', icon: Settings },
              { id: 'model', label: 'Model', icon: Sparkles },
              { id: 'tools', label: 'Tools', icon: Wrench },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all ${
                  activeTab === tab.id
                    ? 'bg-white/10 text-white'
                    : 'text-white/50 hover:text-white/70'
                }`}
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          <div className="space-y-6">
            {activeTab === 'general' && (
              <div className="bg-[#141414] rounded-xl p-6 border border-white/5">
                <h2 className="text-xl font-semibold text-white mb-6">Basic Information</h2>
                
                <div className="space-y-6">
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-white">Agent Name</label>
                    <input
                      type="text"
                      value={formData.name}
                      onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                      className="w-full bg-[#0d0d0d] border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-white/20"
                    />
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium text-white">Description</label>
                    <textarea
                      value={formData.description}
                      onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                      rows={3}
                      className="w-full bg-[#0d0d0d] border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-white/20 resize-none"
                    />
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium text-white">System Prompt</label>
                    <textarea
                      value={formData.systemPrompt}
                      onChange={(e) => setFormData({ ...formData, systemPrompt: e.target.value })}
                      rows={6}
                      className="w-full bg-[#0d0d0d] border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-white/20 resize-none"
                    />
                  </div>

                  <div className="flex items-center justify-between pt-4">
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
              </div>
            )}

            {activeTab === 'model' && (
              <div className="bg-[#141414] rounded-xl p-6 border border-white/5">
                <h2 className="text-xl font-semibold text-white mb-6">Model Settings</h2>
                
                <div className="space-y-6">
                  <div className="space-y-4">
                    <label className="text-sm font-medium text-white">Model</label>
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
                          <div className="flex items-center justify-between">
                            <span className="font-semibold text-white">{model.name}</span>
                            <span className="px-2 py-0.5 bg-white/10 text-white/70 text-xs rounded">{model.provider}</span>
                          </div>
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
            )}

            {activeTab === 'tools' && (
              <div className="bg-[#141414] rounded-xl p-6 border border-white/5">
                <h2 className="text-xl font-semibold text-white mb-6">Tools & Capabilities</h2>
                
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
            )}
          </div>

          {/* Save Button */}
          <div className="flex justify-end pt-6">
            <button
              onClick={handleSave}
              disabled={saving}
              className="flex items-center gap-2 px-6 py-3 bg-white text-black rounded-xl font-medium hover:bg-white/90 transition-colors disabled:opacity-50"
            >
              {saving ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Saving...
                </>
              ) : (
                <>
                  <Save className="w-4 h-4" />
                  Save Changes
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </MiniMaxLayout>
  );
}

'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { MiniMaxLayout } from '@/components/layout/MiniMaxLayout';
import { api, Agent as ApiAgent, User as ApiUser } from '@/lib/api';
import { toast } from 'sonner';
import {
  Plus,
  Play,
  Settings,
  MoreVertical,
  Bot,
  Clock,
  Zap,
  Loader2,
  Trash2,
  Sparkles,
  ChevronRight,
} from 'lucide-react';

interface AgentWithStats extends ApiAgent {
  lastUsed?: string;
  totalRuns?: number;
}

export default function AgentsPage() {
  const router = useRouter();
  const [agents, setAgents] = useState<AgentWithStats[]>([]);
  const [user, setUser] = useState<ApiUser | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      const [userData, agentsData] = await Promise.all([
        api.auth.getCurrentUser(),
        api.agents.list(),
      ]);
      
      setUser(userData);
      setAgents(agentsData.map(agent => ({
        ...agent,
        lastUsed: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString(),
        totalRuns: Math.floor(Math.random() * 100),
      })));
    } catch (err: any) {
      if (err.message.includes('401') || err.message.includes('Not authenticated')) {
        router.push('/login');
      } else {
        toast.error(err.message || 'Failed to load agents');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Are you sure you want to delete this agent?')) return;
    
    try {
      await api.agents.delete(id);
      toast.success('Agent deleted successfully');
      loadData();
    } catch (error) {
      toast.error('Failed to delete agent');
    }
  };

  const getModelColor = (model: string) => {
    if (model.includes('gpt-4')) return 'from-purple-500 to-purple-600';
    if (model.includes('gpt-3.5')) return 'from-blue-500 to-blue-600';
    if (model.includes('claude')) return 'from-orange-500 to-orange-600';
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

  return (
    <MiniMaxLayout>
      <div className="h-full overflow-y-auto">
        <div className="max-w-7xl mx-auto px-6 py-8">
          {/* Header */}
          <div className="mb-8">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold text-white mb-2">My Agents</h1>
                <p className="text-white/50">Create and manage your AI agents</p>
              </div>
              <button 
                onClick={() => router.push('/agents/new')}
                className="flex items-center gap-2 px-4 py-2.5 bg-white text-black rounded-xl font-medium hover:bg-white/90 transition-colors"
              >
                <Plus className="w-4 h-4" />
                Create Agent
              </button>
            </div>
          </div>

          {/* Stats Overview */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            <div className="bg-[#141414] rounded-xl p-6 border border-white/5">
              <div className="flex items-center gap-4">
                <div className="p-3 bg-blue-500/20 rounded-xl">
                  <Bot className="w-6 h-6 text-blue-400" />
                </div>
                <div>
                  <p className="text-sm text-white/50">Total Agents</p>
                  <p className="text-2xl font-bold text-white">{agents.length}</p>
                </div>
              </div>
            </div>

            <div className="bg-[#141414] rounded-xl p-6 border border-white/5">
              <div className="flex items-center gap-4">
                <div className="p-3 bg-green-500/20 rounded-xl">
                  <Zap className="w-6 h-6 text-green-400" />
                </div>
                <div>
                  <p className="text-sm text-white/50">Total Executions</p>
                  <p className="text-2xl font-bold text-white">
                    {agents.reduce((acc, agent) => acc + (agent.totalRuns || 0), 0)}
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-[#141414] rounded-xl p-6 border border-white/5">
              <div className="flex items-center gap-4">
                <div className="p-3 bg-purple-500/20 rounded-xl">
                  <Clock className="w-6 h-6 text-purple-400" />
                </div>
                <div>
                  <p className="text-sm text-white/50">Last Activity</p>
                  <p className="text-2xl font-bold text-white">Just now</p>
                </div>
              </div>
            </div>
          </div>

          {/* Agents Grid */}
          {agents.length === 0 ? (
            <div className="bg-[#141414] rounded-xl p-16 border border-white/5 text-center">
              <div className="w-16 h-16 bg-white/5 rounded-full flex items-center justify-center mx-auto mb-4">
                <Bot className="w-8 h-8 text-white/30" />
              </div>
              <h3 className="text-lg font-medium text-white mb-2">No agents yet</h3>
              <p className="text-white/50 mb-6">Create your first AI agent to get started</p>
              <button 
                onClick={() => router.push('/agents/new')}
                className="px-6 py-2.5 bg-white text-black rounded-xl font-medium hover:bg-white/90 transition-colors"
              >
                Create Agent
              </button>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {agents.map((agent) => (
                <div 
                  key={agent.id} 
                  className="group bg-[#141414] hover:bg-[#1a1a1a] rounded-xl p-6 border border-white/5 hover:border-white/10 transition-all"
                >
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className={`w-10 h-10 rounded-xl bg-gradient-to-br ${getModelColor(agent.model)} flex items-center justify-center`}>
                        <Sparkles className="w-5 h-5 text-white" />
                      </div>
                      <div>
                        <h3 className="font-semibold text-white line-clamp-1">{agent.name}</h3>
                        <p className="text-xs text-white/40">{agent.model}</p>
                      </div>
                    </div>
                    <div className="relative">
                      <button className="p-1.5 hover:bg-white/5 rounded-lg transition-colors">
                        <MoreVertical className="w-4 h-4 text-white/50" />
                      </button>
                    </div>
                  </div>

                  <p className="text-sm text-white/50 line-clamp-2 mb-4">
                    {agent.description || 'No description'}
                  </p>

                  <div className="flex items-center gap-4 text-sm text-white/40 mb-4">
                    <div className="flex items-center gap-1">
                      <Play className="w-3.5 h-3.5" />
                      <span>{agent.totalRuns} runs</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Clock className="w-3.5 h-3.5" />
                      <span>{new Date(agent.lastUsed || agent.updatedAt).toLocaleDateString()}</span>
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <button
                      onClick={() => router.push(`/agents/${agent.id}/execute`)}
                      className="flex-1 flex items-center justify-center gap-1.5 px-4 py-2 bg-white text-black rounded-lg text-sm font-medium hover:bg-white/90 transition-colors"
                    >
                      <Play className="w-3.5 h-3.5" />
                      Run
                    </button>
                    <button
                      onClick={() => router.push(`/agents/${agent.id}`)}
                      className="flex-1 flex items-center justify-center gap-1.5 px-4 py-2 bg-white/5 text-white rounded-lg text-sm font-medium hover:bg-white/10 transition-colors"
                    >
                      <Settings className="w-3.5 h-3.5" />
                      Edit
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </MiniMaxLayout>
  );
}

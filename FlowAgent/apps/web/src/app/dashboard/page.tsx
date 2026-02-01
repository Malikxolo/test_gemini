'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { MiniMaxLayout } from '@/components/layout/MiniMaxLayout';
import { useSupabase } from '@/components/providers/supabase-provider';
import { api } from '@/lib/api';
import { 
  Paperclip, 
  SlidersHorizontal, 
  FolderOpen, 
  ChevronDown, 
  ArrowUp,
  FileText,
  Radio,
  Layers,
  Gift,
  ChevronsUpDown,
  Monitor,
  Smile,
  Folder,
  PieChart,
  FileCode,
  TrendingUp,
  BookOpen,
  Users,
  Plus,
  Loader2,
} from 'lucide-react';

const categories = [
  { name: 'File Organization', icon: FileText, color: 'text-yellow-400' },
  { name: 'Media Publish', icon: Radio, color: 'text-red-400' },
  { name: 'Batch Processing', icon: Layers, color: 'text-green-400' },
  { name: 'Life Assistant', icon: Gift, color: 'text-purple-400' },
  { name: 'More', icon: ChevronsUpDown, color: 'text-white/60' },
];

const defaultExperts = [
  {
    id: 1,
    name: 'Landing Page Builder',
    description: 'Professional high-end Landing Page generation tool, creating visually stunning pages...',
    author: 'FlowAgent',
    views: 6456,
    icon: Monitor,
    color: 'from-blue-400 to-cyan-400',
  },
  {
    id: 2,
    name: 'GIF Sticker Maker',
    description: 'Cute cartoon sticker generator. Use this cookbook when users need to...',
    author: 'FlowAgent',
    views: 1456,
    icon: Smile,
    color: 'from-yellow-400 to-orange-400',
  },
  {
    id: 3,
    name: 'Tidy Folder',
    description: 'Professional folder organization assistant that helps users safely...',
    author: 'FlowAgent',
    views: 2142,
    icon: Folder,
    color: 'from-blue-500 to-indigo-500',
  },
  {
    id: 4,
    name: 'Visual Lab',
    description: 'Professional visual content generation tool, using AI image...',
    author: 'FlowAgent',
    views: 2707,
    icon: PieChart,
    color: 'from-purple-400 to-pink-400',
  },
];

const additionalExperts = [
  { name: 'Doc Processor', description: 'Professional document processing tool, supporting PDF and DOCX...', author: 'FlowAgent', views: 1339, icon: FileCode, color: 'from-blue-600 to-blue-800' },
  { name: 'Icon Maker', description: 'AI icon generator that creates professional-grade icons based on...', author: 'FlowAgent', views: 1490, icon: Smile, color: 'from-pink-400 to-rose-500' },
  { name: 'Topic Tracker', description: 'Based on user-input topics, searches latest sources, discover...', author: 'FlowAgent', views: 383, icon: TrendingUp, color: 'from-green-400 to-emerald-600' },
  { name: 'Video Story Generator', description: 'Automatically generates complete video stories from images or text...', author: 'FlowAgent', views: 1754, icon: BookOpen, color: 'from-orange-400 to-red-500' },
  { name: 'AI Trading Consortium', description: 'An AI-powered hedge fund Expert Agent that combines multi-expert...', author: 'FlowAgent', views: 652, icon: Users, color: 'from-indigo-400 to-purple-600' },
  { name: 'PRD Assistant', description: 'Product requirements analysis and PRD generation assistant. From...', author: 'FlowAgent', views: 638, icon: FileText, color: 'from-cyan-400 to-blue-500' },
  { name: 'Crypto Trading Agent', description: 'A professional-grade crypto trading decision agent for BTC/ETH/SOL...', author: 'FlowAgent', views: 478, icon: TrendingUp, color: 'from-yellow-500 to-orange-500' },
  { name: 'Hedge Fund Team', description: 'A team of 18 top investment experts AI hedge fund team. When users need stock investment analysis...', author: 'FlowAgent', views: 1312, icon: Users, color: 'from-red-500 to-pink-600' },
];

export default function DashboardPage() {
  const router = useRouter();
  const { session, isLoading: isPending } = useSupabase();
  const [agents, setAgents] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [inputText, setInputText] = useState('');
  const [showMore, setShowMore] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState('All');

  useEffect(() => {
    if (!isPending) {
      if (!session) {
        router.push('/login');
      } else {
        loadAgents();
      }
    }
  }, [session, isPending]);

  const loadAgents = async () => {
    try {
      const agentsData = await api.agents.list();
      setAgents(agentsData);
    } catch (err) {
      console.error('Failed to load agents:', err);
    } finally {
      setLoading(false);
    }
  };

  const getGreeting = () => {
    const hour = new Date().getHours();
    const greeting = hour < 12 ? 'Good morning' : hour < 18 ? 'Good afternoon' : 'Good evening';
    // Supabase stores user metadata in user_metadata
    const displayName = session?.user?.user_metadata?.display_name || 
                       session?.user?.user_metadata?.username || 
                       session?.user?.email || 
                       'User';
    return `${greeting}, ${displayName}`;
  };

  const handleCreateAgent = () => {
    router.push('/agents/new');
  };

  const handleExecuteAgent = (agentId: string) => {
    router.push(`/agents/${agentId}/execute`);
  };

  if (isPending || loading) {
    return (
      <div className="flex h-screen bg-[#0d0d0d] items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-white" />
      </div>
    );
  }

  return (
    <MiniMaxLayout>
      <div className="min-h-full flex flex-col">
        {/* Hero Section */}
        <div className="flex-1 flex flex-col items-center justify-center px-6 py-12 relative">
          <div className="absolute inset-0 hero-gradient opacity-50 pointer-events-none" />
          <div className="absolute inset-0 dot-pattern opacity-20 pointer-events-none" />

          <div className="w-full max-w-3xl relative z-10">
            <h1 className="text-3xl md:text-4xl font-semibold text-center mb-8 text-white">
              {getGreeting()}, cowork with me!
            </h1>

            {/* Input Box */}
            <div className="bg-[#1a1a1a] rounded-2xl border border-white/10 overflow-hidden">
              <div className="p-4">
                <textarea
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder="Job Search: Search LinkedIn for Product Designer positions based in New York with salary $100k+, and compile into a spreadsheet."
                  className="w-full bg-transparent text-white placeholder-white/30 resize-none outline-none min-h-[100px] text-base"
                  rows={3}
                />
              </div>

              <div className="flex items-center justify-between px-4 py-3 border-t border-white/5">
                <div className="flex items-center gap-2">
                  <button 
                    onClick={() => document.getElementById('dashboard-file-input')?.click()}
                    className="p-2 hover:bg-white/5 rounded-lg transition-colors"
                  >
                    <Paperclip className="w-4 h-4 text-white/50" />
                  </button>
                  <input 
                    id="dashboard-file-input" 
                    type="file" 
                    className="hidden" 
                    onChange={(e) => {
                      // Handle file upload
                      if (e.target.files?.length) {
                        localStorage.setItem('pendingAttachments', JSON.stringify(Array.from(e.target.files).map(f => f.name)));
                      }
                    }}
                  />
                  <button className="p-2 hover:bg-white/5 rounded-lg transition-colors">
                    <SlidersHorizontal className="w-4 h-4 text-white/50" />
                  </button>
                  <button className="flex items-center gap-2 px-3 py-1.5 bg-white/5 hover:bg-white/10 rounded-lg transition-colors text-sm text-white/70">
                    <FolderOpen className="w-4 h-4" />
                    <span>projects</span>
                    <ChevronDown className="w-3 h-3" />
                  </button>
                </div>

                <div className="flex items-center gap-2">
                  <button className="flex items-center gap-1 px-3 py-1.5 text-sm text-white/50 hover:text-white/70 transition-colors">
                    <span>Auto</span>
                    <ChevronDown className="w-3 h-3" />
                  </button>
                  <button 
                    onClick={() => {
                      if (inputText.trim()) {
                        localStorage.setItem('pendingMessage', inputText);
                        router.push('/chat');
                      }
                    }}
                    disabled={!inputText.trim()}
                    className="p-2 bg-white text-black hover:bg-white/90 rounded-lg transition-colors disabled:opacity-30"
                  >
                    <ArrowUp className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>

            {/* Category Pills */}
            <div className="flex items-center justify-center gap-2 mt-6 flex-wrap">
              {categories.map((category) => (
                <button
                  key={category.name}
                  onClick={() => setSelectedCategory(category.name)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-full border transition-all ${
                    selectedCategory === category.name
                      ? 'bg-white/10 border-white/20 text-white'
                      : 'bg-transparent border-white/10 text-white/60 hover:border-white/20 hover:text-white/80'
                  }`}
                >
                  <category.icon className={`w-4 h-4 ${category.color}`} />
                  <span className="text-sm">{category.name}</span>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* User Agents Section */}
        <div className="px-6 pb-4">
          <div className="max-w-6xl mx-auto">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <span className="px-3 py-1.5 bg-white/10 rounded-lg text-sm font-medium">
                  My Agents
                </span>
                <span className="text-white/40 text-sm">
                  {agents.length} agent{agents.length !== 1 ? 's' : ''}
                </span>
              </div>
              <button 
                onClick={handleCreateAgent}
                className="flex items-center gap-2 px-4 py-2 bg-white text-black rounded-lg text-sm font-medium hover:bg-white/90 transition-colors"
              >
                <Plus className="w-4 h-4" />
                New Agent
              </button>
            </div>

            {agents.length === 0 ? (
              <div className="bg-[#141414] rounded-xl p-8 border border-white/5 text-center">
                <p className="text-white/50 mb-4">You haven&apos;t created any agents yet</p>
                <button 
                  onClick={handleCreateAgent}
                  className="px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg text-white transition-colors"
                >
                  Create your first agent
                </button>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {agents.map((agent) => (
                  <div
                    key={agent.id}
                    className="group bg-[#141414] hover:bg-[#1a1a1a] rounded-xl p-4 border border-white/5 hover:border-white/10 transition-all cursor-pointer"
                  >
                    <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-400 to-purple-500 flex items-center justify-center mb-4">
                      <Monitor className="w-5 h-5 text-white" />
                    </div>

                    <h3 className="font-medium text-white mb-2 group-hover:text-blue-400 transition-colors">
                      {agent.name}
                    </h3>
                    <p className="text-sm text-white/50 line-clamp-2 mb-4">
                      {agent.description || 'No description'}
                    </p>

                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 text-xs text-white/40">
                        <span>{agent.model}</span>
                      </div>
                      <button
                        onClick={() => handleExecuteAgent(agent.id)}
                        className="px-3 py-1.5 bg-white/10 hover:bg-white/20 rounded-lg text-xs text-white transition-colors"
                      >
                        Run
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Experts Section */}
        <div className="px-6 pb-8">
          <div className="max-w-6xl mx-auto">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <span className="px-3 py-1.5 bg-white/10 rounded-lg text-sm font-medium">
                  Experts by FlowAgent
                </span>
              </div>
              <button 
                onClick={() => setShowMore(!showMore)}
                className="flex items-center gap-1 text-sm text-white/50 hover:text-white/70 transition-colors"
              >
                <span>{showMore ? 'Show Less' : 'Show More'}</span>
                {showMore ? (
                  <ChevronsUpDown className="w-4 h-4 rotate-180" />
                ) : (
                  <ChevronsUpDown className="w-4 h-4" />
                )}
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {defaultExperts.map((expert) => (
                <div
                  key={expert.id}
                  className="group bg-[#141414] hover:bg-[#1a1a1a] rounded-xl p-4 border border-white/5 hover:border-white/10 transition-all cursor-pointer"
                >
                  <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${expert.color} flex items-center justify-center mb-4`}>
                    <expert.icon className="w-5 h-5 text-white" />
                  </div>

                  <h3 className="font-medium text-white mb-2 group-hover:text-blue-400 transition-colors">
                    {expert.name}
                  </h3>
                  <p className="text-sm text-white/50 line-clamp-2 mb-4">
                    {expert.description}
                  </p>

                  <div className="flex items-center justify-between text-xs text-white/40">
                    <span>By {expert.author}</span>
                    <span>{expert.views.toLocaleString()} Views</span>
                  </div>
                </div>
              ))}
            </div>

            {showMore && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mt-4">
                {additionalExperts.map((expert, index) => (
                  <div
                    key={index}
                    className="group bg-[#141414] hover:bg-[#1a1a1a] rounded-xl p-4 border border-white/5 hover:border-white/10 transition-all cursor-pointer"
                  >
                    <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${expert.color} flex items-center justify-center mb-4`}>
                      <expert.icon className="w-5 h-5 text-white" />
                    </div>
                    <h3 className="font-medium text-white mb-2 group-hover:text-blue-400 transition-colors">
                      {expert.name}
                    </h3>
                    <p className="text-sm text-white/50 line-clamp-2 mb-4">
                      {expert.description}
                    </p>
                    <div className="flex items-center justify-between text-xs text-white/40">
                      <span>By {expert.author}</span>
                      <span>{expert.views.toLocaleString()} Views</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </MiniMaxLayout>
  );
}

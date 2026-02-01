'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { MiniMaxLayout } from '@/components/layout/MiniMaxLayout';
import { Plus, ChevronDown, Monitor, Smile, Folder, PieChart, FileCode, TrendingUp, BookOpen, Users, FileText } from 'lucide-react';
import { useSupabase } from '@/components/providers/supabase-provider';

const allExperts = [
  { name: 'Landing Page Builder', description: 'Professional high-end Landing Page generation tool, creating visually stunning pages...', author: 'FlowAgent', views: 6456, icon: Monitor, color: 'from-blue-400 to-cyan-400' },
  { name: 'Visual Lab', description: 'Professional visual content generation tool, using AI image...', author: 'FlowAgent', views: 2707, icon: PieChart, color: 'from-purple-400 to-pink-400' },
  { name: 'Tidy Folder', description: 'Professional folder organization assistant that helps users safely...', author: 'FlowAgent', views: 2142, icon: Folder, color: 'from-blue-500 to-indigo-500' },
  { name: 'Video Story Generator', description: 'Automatically generates complete video stories from images or text...', author: 'FlowAgent', views: 1754, icon: BookOpen, color: 'from-orange-400 to-red-500' },
  { name: 'Icon Maker', description: 'AI icon generator that creates professional-grade icons based on...', author: 'FlowAgent', views: 1490, icon: Smile, color: 'from-pink-400 to-rose-500' },
  { name: 'GIF Sticker Maker', description: 'Cute cartoon sticker generator. Use this cookbook when users need to...', author: 'FlowAgent', views: 1456, icon: Smile, color: 'from-yellow-400 to-orange-400' },
  { name: 'Doc Processor', description: 'Professional document processing tool, supporting PDF and DOCX...', author: 'FlowAgent', views: 1339, icon: FileCode, color: 'from-blue-600 to-blue-800' },
  { name: 'Hedge Fund Team', description: 'A team of 18 top investment experts AI hedge fund team. When users need stock investment analysis...', author: 'FlowAgent', views: 1312, icon: Users, color: 'from-red-500 to-pink-600' },
  { name: 'AI Trading Consortium', description: 'An AI-powered hedge fund Expert Agent that combines multi-expert...', author: 'FlowAgent', views: 652, icon: Users, color: 'from-indigo-400 to-purple-600' },
  { name: 'PRD Assistant', description: 'Product requirements analysis and PRD generation assistant. From...', author: 'FlowAgent', views: 638, icon: FileText, color: 'from-cyan-400 to-blue-500' },
  { name: 'Crypto Trading Agent', description: 'A professional-grade crypto trading decision agent for BTC/ETH/SOL...', author: 'FlowAgent', views: 478, icon: TrendingUp, color: 'from-yellow-500 to-orange-500' },
  { name: 'Topic Tracker', description: 'Based on user-input topics, searches latest sources, discover...', author: 'FlowAgent', views: 383, icon: TrendingUp, color: 'from-green-400 to-emerald-600' },
];

export default function ExpertsPage() {
  const { session, isLoading: isPending } = useSupabase();
  const [activeTab, setActiveTab] = useState('community');
  const router = useRouter();

  useEffect(() => {
    if (!isPending && !session) {
      router.push('/login');
    }
  }, [session, isPending, router]);

  const handleCreate = () => {
    router.push('/agents/new');
  };

  if (isPending) {
    return (
      <div className="flex h-screen bg-[#0d0d0d] items-center justify-center">
        <div className="text-white/60">Loading...</div>
      </div>
    );
  }

  return (
    <MiniMaxLayout>
      <div className="h-full overflow-y-auto">
        <div className="max-w-6xl mx-auto px-6 py-8">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-4xl font-bold mb-2">Experts</h1>
            <p className="text-white/50">Browse and create experts to get more done.</p>
          </div>

          {/* Create Button */}
          <div className="flex justify-end mb-6">
            <button 
              onClick={handleCreate}
              className="flex items-center gap-2 px-4 py-2 bg-white text-black rounded-lg font-medium hover:bg-white/90 transition-colors"
            >
              <Plus className="w-4 h-4" />
              Create
            </button>
          </div>

          {/* Tabs */}
          <div className="flex items-center justify-between border-b border-white/10 mb-6">
            <div className="flex gap-6">
              <button
                onClick={() => setActiveTab('community')}
                className={`pb-3 text-sm font-medium border-b-2 transition-colors ${
                  activeTab === 'community'
                    ? 'border-white text-white'
                    : 'border-transparent text-white/50 hover:text-white/70'
                }`}
              >
                Expert community
              </button>
              <button
                onClick={() => setActiveTab('my')}
                className={`pb-3 text-sm font-medium border-b-2 transition-colors ${
                  activeTab === 'my'
                    ? 'border-white text-white'
                    : 'border-transparent text-white/50 hover:text-white/70'
                }`}
              >
                My experts
              </button>
            </div>

            <button className="flex items-center gap-2 px-3 py-1.5 text-sm text-white/50 hover:text-white/70 transition-colors">
              <span>Popular</span>
              <ChevronDown className="w-4 h-4" />
            </button>
          </div>

          {/* Experts Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {allExperts.map((expert, index) => (
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
        </div>
      </div>
    </MiniMaxLayout>
  );
}

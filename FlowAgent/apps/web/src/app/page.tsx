'use client';

import Link from 'next/link';
import { ArrowRight, Zap, Shield, TrendingUp, Sparkles, Check } from 'lucide-react';

export default function HomePage() {
  return (
    <div className="min-h-screen bg-[#0d0d0d] flex flex-col">
      {/* Hero Section */}
      <section className="flex-1 flex items-center justify-center relative overflow-hidden">
        {/* Background Gradient */}
        <div className="absolute inset-0 hero-gradient opacity-30 pointer-events-none" />
        <div className="absolute inset-0 dot-pattern opacity-10 pointer-events-none" />

        <div className="container mx-auto px-6 py-20 relative z-10">
          <div className="max-w-4xl mx-auto text-center">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-white/5 rounded-full border border-white/10 mb-8">
              <Sparkles className="w-4 h-4 text-blue-400" />
              <span className="text-sm text-white/70">Now with MiniMax-style UI</span>
            </div>

            <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 leading-tight">
              Build AI Agents
              <br />
              <span className="bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
                at Scale
              </span>
            </h1>
            
            <p className="text-xl text-white/50 max-w-2xl mx-auto mb-10">
              Create, deploy, and manage intelligent AI agents with zero infrastructure cost.
              From 0 to millions of users in minutes.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link href="/signup">
                <button className="flex items-center justify-center gap-2 px-8 py-4 bg-white text-black rounded-xl font-medium hover:bg-white/90 transition-colors">
                  Get Started Free
                  <ArrowRight className="w-4 h-4" />
                </button>
              </Link>
              <Link href="/login">
                <button className="px-8 py-4 bg-white/5 text-white rounded-xl font-medium hover:bg-white/10 transition-colors border border-white/10">
                  Sign In
                </button>
              </Link>
            </div>
          </div>

          {/* Features */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-5xl mx-auto mt-20">
            <div className="bg-[#141414] rounded-2xl p-6 border border-white/5 hover:border-white/10 transition-all">
              <div className="w-12 h-12 bg-blue-500/20 rounded-xl flex items-center justify-center mb-4">
                <Zap className="w-6 h-6 text-blue-400" />
              </div>
              <h3 className="text-xl font-bold text-white mb-2">Zero Cost at Launch</h3>
              <p className="text-white/50">
                Serverless-first architecture means $0 with 0 users. Scale linearly with usage.
              </p>
            </div>

            <div className="bg-[#141414] rounded-2xl p-6 border border-white/5 hover:border-white/10 transition-all">
              <div className="w-12 h-12 bg-green-500/20 rounded-xl flex items-center justify-center mb-4">
                <Shield className="w-6 h-6 text-green-400" />
              </div>
              <h3 className="text-xl font-bold text-white mb-2">Enterprise Security</h3>
              <p className="text-white/50">
                Multi-layer security with edge protection, rate limiting, and sandboxed execution.
              </p>
            </div>

            <div className="bg-[#141414] rounded-2xl p-6 border border-white/5 hover:border-white/10 transition-all">
              <div className="w-12 h-12 bg-purple-500/20 rounded-xl flex items-center justify-center mb-4">
                <TrendingUp className="w-6 h-6 text-purple-400" />
              </div>
              <h3 className="text-xl font-bold text-white mb-2">Infinite Scale</h3>
              <p className="text-white/50">
                From 0 to 10M+ users without architectural changes. Global edge deployment.
              </p>
            </div>
          </div>

          {/* Features List */}
          <div className="max-w-3xl mx-auto mt-20">
            <div className="bg-[#141414] rounded-2xl p-8 border border-white/5">
              <h3 className="text-2xl font-bold text-white mb-6 text-center">Everything you need</h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {[
                  'Multi-tenant architecture',
                  'BYOK (Bring Your Own Key)',
                  'Usage tracking & billing',
                  'White-label capabilities',
                  'OpenAI, Anthropic, Mistral support',
                  'Python agent executor',
                  'Rate limiting & security',
                  'Self-hosted option',
                ].map((feature, index) => (
                  <div key={index} className="flex items-center gap-3">
                    <div className="w-5 h-5 rounded-full bg-green-500/20 flex items-center justify-center flex-shrink-0">
                      <Check className="w-3 h-3 text-green-400" />
                    </div>
                    <span className="text-white/70">{feature}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/5 py-8">
        <div className="container mx-auto px-6">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <p className="text-white/40 text-sm">
              © 2025 FlowAgent. All rights reserved.
            </p>
            
            <div className="flex items-center gap-6">
              <a href="#" className="text-white/40 hover:text-white text-sm transition-colors">Privacy</a>
              <a href="#" className="text-white/40 hover:text-white text-sm transition-colors">Terms</a>
              <a href="#" className="text-white/40 hover:text-white text-sm transition-colors">GitHub</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

'use client';

import { ReactNode, useState } from 'react';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { cn } from '@/lib/utils';
import { useSupabase } from '@/components/providers/supabase-provider';
import { TaskHistory } from '@/components/chat/TaskHistory';
import { 
  Plus, 
  Search, 
  Sparkles, 
  Clock, 
  Settings,
  LogOut,
  ChevronRight,
  Zap,
  MessageSquare,
  FileText,
  ChevronDown,
  X,
  Key,
  Loader2,
} from 'lucide-react';

interface SidebarProps {
  children: ReactNode;
}

export function MiniMaxLayout({ children }: SidebarProps) {
  const pathname = usePathname();
  const router = useRouter();
  const { user, isLoading, signOut } = useSupabase();
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const [creditsOpen, setCreditsOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [loggingOut, setLoggingOut] = useState(false);

  const handleLogout = async () => {
    setLoggingOut(true);
    await signOut();
    router.push('/login');
  };

  if (isLoading) {
    return (
      <div className="flex h-screen bg-[#0d0d0d] items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-white" />
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-[#0d0d0d] text-white overflow-hidden">
      {/* Sidebar */}
      <aside className="w-[280px] flex-shrink-0 flex flex-col bg-[#0d0d0d] border-r border-white/5">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-white/5">
          <Link href="/" className="text-sm font-medium text-white/80 hover:text-white transition-colors">
            Home
          </Link>
          <button className="p-1.5 hover:bg-white/5 rounded-md transition-colors">
            <Plus className="w-4 h-4 text-white/60" />
          </button>
        </div>

        {/* Main Logo/Icon */}
        <div className="px-4 py-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-white" />
            </div>
            <div className="flex-1" />
            <button className="p-2 hover:bg-white/5 rounded-lg transition-colors">
              <div className="w-5 h-5 border border-white/20 rounded flex items-center justify-center">
                <div className="w-2 h-2 bg-white/40 rounded-sm" />
              </div>
            </button>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-3 space-y-1 overflow-y-auto">
          {/* New Task Button */}
          <Link href="/chat">
            <button className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg bg-white/5 hover:bg-white/10 transition-colors text-left">
              <Plus className="w-4 h-4 text-white/70" />
              <span className="text-sm font-medium">New Task</span>
            </button>
          </Link>

          {/* Search Button */}
          <button 
            onClick={() => setSearchOpen(true)}
            className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-white/5 transition-colors text-left text-white/70"
          >
            <Search className="w-4 h-4" />
            <span className="text-sm">Search</span>
          </button>

          {/* Section Label */}
          <div className="pt-4 pb-2 px-3">
            <span className="text-xs text-white/40 font-medium">Experts</span>
          </div>

          {/* Explore Experts */}
          <Link 
            href="/experts" 
            className={cn(
              "flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors",
              pathname === '/experts' ? 'bg-white/10 text-white' : 'text-white/70 hover:bg-white/5'
            )}
          >
            <Sparkles className="w-4 h-4" />
            <span className="text-sm">Explore Experts</span>
            <span className="ml-auto text-[10px] px-1.5 py-0.5 bg-white/10 rounded text-white/60">New</span>
          </Link>

          {/* Task History Section */}
          <TaskHistory />
        </nav>

        {/* User Profile */}
        <div className="p-3 border-t border-white/5 relative">
          <button 
            onClick={() => setUserMenuOpen(!userMenuOpen)}
            className="w-full flex items-center gap-3 p-2 hover:bg-white/5 rounded-lg transition-colors"
          >
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-400 to-purple-500 flex items-center justify-center text-sm font-medium">
              {user?.user_metadata?.display_name?.charAt(0) || user?.email?.charAt(0) || 'U'}
            </div>
            <div className="flex-1 text-left">
              <div className="text-sm font-medium truncate">{user?.user_metadata?.display_name || user?.email || 'User'}</div>
              <div className="text-xs text-white/50">Free</div>
            </div>
            <ChevronDown className={cn("w-4 h-4 text-white/40 transition-transform", userMenuOpen && "rotate-180")} />
          </button>

          {/* User Menu Dropdown */}
          {userMenuOpen && (
            <div className="absolute bottom-full left-3 right-3 mb-2 bg-[#1a1a1a] rounded-lg border border-white/5 overflow-hidden shadow-lg">
              <div className="p-2 space-y-1">
                <Link
                  href="/byok"
                  onClick={() => setUserMenuOpen(false)}
                  className={cn(
                    "w-full flex items-center gap-3 px-3 py-2 rounded-lg transition-colors",
                    pathname === '/byok' ? 'bg-white/10 text-white' : 'text-white/70 hover:bg-white/5'
                  )}
                >
                  <Key className="w-4 h-4" />
                  <span className="text-sm">BYOK Settings</span>
                </Link>

                <Link
                  href="/billing"
                  onClick={() => setUserMenuOpen(false)}
                  className={cn(
                    "w-full flex items-center gap-3 px-3 py-2 rounded-lg transition-colors",
                    pathname === '/billing' ? 'bg-white/10 text-white' : 'text-white/70 hover:bg-white/5'
                  )}
                >
                  <Zap className="w-4 h-4 text-green-400" />
                  <span className="text-sm">Usage & Billing</span>
                </Link>

                <Link
                  href="/settings"
                  onClick={() => setUserMenuOpen(false)}
                  className={cn(
                    "w-full flex items-center gap-3 px-3 py-2 rounded-lg transition-colors",
                    pathname === '/settings' ? 'bg-white/10 text-white' : 'text-white/70 hover:bg-white/5'
                  )}
                >
                  <Settings className="w-4 h-4" />
                  <span className="text-sm">Settings</span>
                </Link>

                <button 
                  onClick={() => setUserMenuOpen(false)}
                  className="w-full flex items-center justify-between px-3 py-2 text-white/70 hover:bg-white/5 rounded-lg transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <MessageSquare className="w-4 h-4" />
                    <span className="text-sm">Contact us</span>
                  </div>
                  <ChevronRight className="w-4 h-4 text-white/40" />
                </button>

                <button 
                  onClick={() => setUserMenuOpen(false)}
                  className="w-full flex items-center justify-between px-3 py-2 text-white/70 hover:bg-white/5 rounded-lg transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <FileText className="w-4 h-4" />
                    <span className="text-sm">Learn More</span>
                  </div>
                  <ChevronRight className="w-4 h-4 text-white/40" />
                </button>

                <button 
                  onClick={handleLogout}
                  disabled={loggingOut}
                  className="w-full flex items-center gap-3 px-3 py-2 text-white/70 hover:bg-white/5 rounded-lg transition-colors disabled:opacity-50"
                >
                  {loggingOut ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span className="text-sm">Logging out...</span>
                    </>
                  ) : (
                    <>
                      <LogOut className="w-4 h-4" />
                      <span className="text-sm">Logout</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          )}
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Top Header */}
        <header className="flex items-center justify-between px-6 py-3 border-b border-white/5">
          <div />
          <div className="flex items-center gap-4">
            <button className="flex items-center gap-2 px-3 py-1.5 bg-white/5 rounded-full text-sm">
              <span className="w-5 h-5 rounded-full bg-gradient-to-br from-blue-400 to-purple-500 flex items-center justify-center text-xs font-bold">M</span>
              <span className="text-white/60">0</span>
            </button>
          </div>
        </header>

        {/* Page Content */}
        <div className="flex-1 overflow-y-auto">
          {children}
        </div>
      </main>

      {/* Search Modal */}
      {searchOpen && (
        <div 
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm"
          onClick={() => setSearchOpen(false)}
        >
          <div 
            className="w-full max-w-2xl bg-[#1a1a1a] rounded-2xl border border-white/10 overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-4 border-b border-white/5">
              <div className="flex items-center gap-3">
                <input
                  type="text"
                  placeholder="Search"
                  autoFocus
                  className="flex-1 bg-transparent text-lg outline-none placeholder-white/30"
                />
                <button 
                  onClick={() => setSearchOpen(false)}
                  className="p-2 hover:bg-white/5 rounded-lg transition-colors"
                >
                  <X className="w-5 h-5 text-white/50" />
                </button>
              </div>
            </div>

            <div className="p-4">
              <button className="w-full flex items-center gap-3 px-3 py-3 rounded-lg hover:bg-white/5 transition-colors text-left"
              >
                <Plus className="w-5 h-5" />
                <span>New Task</span>
              </button>

              <div className="mt-8 text-center text-white/40">
                No Recent Searches
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { getConversations, type Conversation } from '@/lib/llm/conversations';
import { MessageSquare, Clock, Trash2 } from 'lucide-react';
import { toast } from 'sonner';

export function TaskHistory() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [loading, setLoading] = useState(true);
  const pathname = usePathname();

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    try {
      setLoading(true);
      const convs = await getConversations();
      setConversations(convs);
    } catch (error) {
      console.error('Failed to load history:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (!confirm('Delete this conversation?')) return;
    
    try {
      const { deleteConversation } = await import('@/lib/llm/conversations');
      await deleteConversation(id);
      toast.success('Conversation deleted');
      loadHistory();
    } catch (error) {
      toast.error('Failed to delete');
    }
  };

  if (loading) {
    return (
      <div className="px-3 py-4">
        <div className="flex items-center gap-2 mb-2">
          <Clock className="w-3 h-3 text-white/40" />
          <span className="text-xs text-white/40">Task History</span>
        </div>
        <div className="space-y-2">
          {[1, 2, 3].map(i => (
            <div key={i} className="h-8 bg-white/5 rounded-lg animate-pulse" />
          ))}
        </div>
      </div>
    );
  }

  if (conversations.length === 0) {
    return (
      <div className="px-3 py-4">
        <div className="flex items-center gap-2 mb-2">
          <Clock className="w-3 h-3 text-white/40" />
          <span className="text-xs text-white/40">Task History</span>
        </div>
        <div className="px-3 py-2 text-sm text-white/40">
          No Task History
        </div>
      </div>
    );
  }

  return (
    <div className="px-3 py-4">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Clock className="w-3 h-3 text-white/40" />
          <span className="text-xs text-white/40">Task History</span>
        </div>
        <span className="text-xs text-white/30">{conversations.length}</span>
      </div>
      
      <div className="space-y-1">
        {conversations.slice(0, 10).map(conv => (
          <Link
            key={conv.id}
            href={`/chat/${conv.id}`}
            className={`group flex items-center gap-2 px-3 py-2 rounded-lg transition-colors ${
              pathname === `/chat/${conv.id}` 
                ? 'bg-white/10 text-white' 
                : 'text-white/60 hover:bg-white/5 hover:text-white'
            }`}
          >
            <MessageSquare className="w-3.5 h-3.5 flex-shrink-0" />
            <span className="text-sm truncate flex-1">{conv.title}</span>
            
            <button
              onClick={(e) => handleDelete(conv.id, e)}
              className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-500/20 rounded transition-all"
            >
              <Trash2 className="w-3 h-3 text-red-400" />
            </button>
          </Link>
        ))}
        
        {conversations.length > 10 && (
          <div className="px-3 py-1 text-xs text-white/30 text-center">
            +{conversations.length - 10} more
          </div>
        )}
      </div>
    </div>
  );
}

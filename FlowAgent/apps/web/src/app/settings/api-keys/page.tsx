'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { api } from '@/lib/api';
import { APIKeyCard } from '@/components/settings/APIKeyCard';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { toast } from 'sonner';
import { ArrowLeft, Shield, Loader2 } from 'lucide-react';

interface APIKeys {
  openaiApiKey: string | null;
  anthropicApiKey: string | null;
  serperApiKey: string | null;
}

export default function APIKeysPage() {
  const router = useRouter();
  const [keys, setKeys] = useState<APIKeys>({
    openaiApiKey: null,
    anthropicApiKey: null,
    serperApiKey: null,
  });
  const [editingKey, setEditingKey] = useState<'openai' | 'anthropic' | 'serper' | null>(null);
  const [newKeyValue, setNewKeyValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [fetchLoading, setFetchLoading] = useState(true);

  useEffect(() => {
    fetchKeys();
  }, []);

  const fetchKeys = async () => {
    try {
      setFetchLoading(true);
      const data = await api.byok.get();
      setKeys({
        openaiApiKey: data.openaiApiKey || null,
        anthropicApiKey: data.anthropicApiKey || null,
        serperApiKey: data.serperApiKey || null,
      });
    } catch (error) {
      toast.error('Failed to load API keys');
    } finally {
      setFetchLoading(false);
    }
  };

  const validateKey = (type: string, key: string): boolean => {
    if (type === 'openai') {
      return key.startsWith('sk-') && key.length > 20;
    }
    if (type === 'anthropic') {
      return key.startsWith('sk-ant-') && key.length > 20;
    }
    return key.length > 10;
  };

  const handleUpdate = async () => {
    if (!editingKey || !newKeyValue) return;

    if (!validateKey(editingKey, newKeyValue)) {
      toast.error(`Invalid ${editingKey} API key format`);
      return;
    }

    try {
      setLoading(true);
      const updateData: Record<string, string> = {};
      updateData[`${editingKey}ApiKey`] = newKeyValue;
      
      await api.byok.update(updateData);
      toast.success('API key updated successfully');
      setEditingKey(null);
      setNewKeyValue('');
      await fetchKeys();
    } catch (error) {
      toast.error('Failed to update API key');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (type: 'openai' | 'anthropic' | 'serper') => {
    try {
      setLoading(true);
      await api.byok.delete(type);
      toast.success(`${type} API key deleted successfully`);
      await fetchKeys();
    } catch (error) {
      toast.error(`Failed to delete ${type} API key`);
    } finally {
      setLoading(false);
    }
  };

  const openEditModal = (type: 'openai' | 'anthropic' | 'serper') => {
    setEditingKey(type);
    setNewKeyValue('');
  };

  if (fetchLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-8">
          <Button
            variant="ghost"
            onClick={() => router.push('/settings')}
            className="mb-4"
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Settings
          </Button>
          
          <h1 className="text-3xl font-bold text-gray-900">API Keys</h1>
          <p className="mt-2 text-gray-600">Manage your AI provider API keys</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <APIKeyCard
            name="OpenAI API Key"
            icon="🔑"
            currentKey={keys.openaiApiKey}
            onUpdate={() => openEditModal('openai')}
            onDelete={() => handleDelete('openai')}
            description="Required for GPT-4, GPT-3.5, and other OpenAI models"
            loading={loading}
          />

          <APIKeyCard
            name="Anthropic API Key"
            icon="🤖"
            currentKey={keys.anthropicApiKey}
            onUpdate={() => openEditModal('anthropic')}
            onDelete={() => handleDelete('anthropic')}
            description="Required for Claude models"
            loading={loading}
          />

          <APIKeyCard
            name="Serper API Key"
            icon="🔍"
            currentKey={keys.serperApiKey}
            onUpdate={() => openEditModal('serper')}
            onDelete={() => handleDelete('serper')}
            description="Required for web search functionality"
            loading={loading}
          />
        </div>

        <div className="mt-8 bg-blue-50 border border-blue-200 rounded-lg p-6">
          <div className="flex items-start gap-3">
            <Shield className="h-5 w-5 text-blue-600 mt-0.5" />
            <div>
              <h3 className="font-semibold text-blue-900">Benefits of BYOK</h3>
              <ul className="mt-2 text-sm text-blue-800 space-y-1">
                <li>• Pay OpenAI/Anthropic directly - no platform markup</li>
                <li>• Full control over your API usage and rate limits</li>
                <li>• Your keys are encrypted and stored securely</li>
                <li>• Use your own billing and monitoring</li>
              </ul>
            </div>
          </div>
        </div>

        <Dialog open={!!editingKey} onOpenChange={() => setEditingKey(null)}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>
                {keys[`${editingKey}ApiKey` as keyof APIKeys] ? 'Update' : 'Add'} {editingKey?.charAt(0).toUpperCase()}{editingKey?.slice(1)} API Key
              </DialogTitle>
            </DialogHeader>
            
            <div className="space-y-4 py-4">
              <div>
                <Label htmlFor="api-key">API Key</Label>
                <Input
                  id="api-key"
                  type="password"
                  placeholder={editingKey === 'openai' ? 'sk-...' : editingKey === 'anthropic' ? 'sk-ant-...' : 'Enter API key'}
                  value={newKeyValue}
                  onChange={(e) => setNewKeyValue(e.target.value)}
                />
              </div>
              
              <p className="text-sm text-gray-600">
                Your API key will be encrypted and stored securely. We never share your keys with third parties.
              </p>
              
              <div className="flex gap-3">
                <Button
                  variant="outline"
                  onClick={() => setEditingKey(null)}
                  disabled={loading}
                >
                  Cancel
                </Button>
                <Button onClick={handleUpdate} disabled={loading || !newKeyValue}>
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Saving...
                    </>
                  ) : (
                    'Save Key'
                  )}
                </Button>
              </div>
            </div>
          </DialogContent>
        </Dialog>
      </div>
    </div>
  );
}

'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { MiniMaxLayout } from '@/components/layout/MiniMaxLayout';
import { Key, Eye, EyeOff, Save, Check, AlertCircle, Trash2, Loader2, Star } from 'lucide-react';
import { toast } from 'sonner';
import { useSupabase } from '@/components/providers/supabase-provider';
import { addApiKey, getApiKeys, deleteApiKey, setDefaultApiKey, type ApiKey } from '@/lib/llm/api-keys';
import type { Provider } from '@/lib/llm/provider';

interface ProviderConfig {
  key: Provider;
  name: string;
  description: string;
  placeholder: string;
  validate: (key: string) => boolean;
}

const providers: ProviderConfig[] = [
  {
    key: 'openai',
    name: 'OpenAI',
    description: 'GPT-4, GPT-3.5 models',
    placeholder: 'sk-...',
    validate: (key: string) => key.startsWith('sk-') && key.length > 20,
  },
  {
    key: 'anthropic',
    name: 'Anthropic',
    description: 'Claude models',
    placeholder: 'sk-ant-...',
    validate: (key: string) => key.startsWith('sk-ant-') && key.length > 20,
  },
  {
    key: 'google',
    name: 'Google',
    description: 'Gemini models',
    placeholder: 'Enter API key',
    validate: (key: string) => key.length > 10,
  },
  {
    key: 'openrouter',
    name: 'OpenRouter',
    description: 'Access multiple models',
    placeholder: 'sk-or-...',
    validate: (key: string) => key.startsWith('sk-or-') && key.length > 20,
  },
];

export default function BYOKPage() {
  const { session, isLoading: isPending } = useSupabase();
  const router = useRouter();
  
  const [keyInputs, setKeyInputs] = useState<Record<Provider, { key: string; keyName: string }>>({
    openai: { key: '', keyName: 'Default' },
    anthropic: { key: '', keyName: 'Default' },
    google: { key: '', keyName: 'Default' },
    openrouter: { key: '', keyName: 'Default' },
  });
  
  const [showKeys, setShowKeys] = useState<Record<Provider, boolean>>({
    openai: false,
    anthropic: false,
    google: false,
    openrouter: false,
  });
  
  const [savedKeys, setSavedKeys] = useState<Record<Provider, ApiKey[]>>({
    openai: [],
    anthropic: [],
    google: [],
    openrouter: [],
  });
  
  const [saving, setSaving] = useState<Record<Provider, boolean>>({
    openai: false,
    anthropic: false,
    google: false,
    openrouter: false,
  });
  
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!isPending && !session) {
      router.push('/login');
    }
  }, [session, isPending, router]);

  useEffect(() => {
    loadSavedKeys();
  }, []);

  const loadSavedKeys = async () => {
    try {
      const apiKeys = await getApiKeys();
      const grouped: Record<Provider, ApiKey[]> = {
        openai: [],
        anthropic: [],
        google: [],
        openrouter: [],
      };
      apiKeys.forEach((key) => {
        if (grouped[key.provider]) {
          grouped[key.provider].push(key);
        }
      });
      setSavedKeys(grouped);
    } catch (error) {
      toast.error('Failed to load saved API keys');
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async (provider: Provider) => {
    const keyData = keyInputs[provider];
    if (!keyData.key.trim()) {
      toast.error(`Please enter an API key for ${provider}`);
      return;
    }

    setSaving((prev) => ({ ...prev, [provider]: true }));
    try {
      await addApiKey(provider, keyData.key, keyData.keyName, true);
      toast.success(`${provider} API key saved successfully`);
      setKeyInputs((prev) => ({
        ...prev,
        [provider]: { key: '', keyName: 'Default' },
      }));
      await loadSavedKeys();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : `Failed to save ${provider} API key`);
    } finally {
      setSaving((prev) => ({ ...prev, [provider]: false }));
    }
  };

  const handleDelete = async (id: string, provider: Provider) => {
    try {
      await deleteApiKey(id);
      toast.success('API key deleted');
      await loadSavedKeys();
    } catch (error) {
      toast.error('Failed to delete API key');
    }
  };

  const handleSetDefault = async (id: string, provider: Provider) => {
    try {
      await setDefaultApiKey(id, provider);
      toast.success('Default API key updated');
      await loadSavedKeys();
    } catch (error) {
      toast.error('Failed to update default key');
    }
  };

  const toggleVisibility = (provider: Provider) => {
    setShowKeys((prev) => ({
      ...prev,
      [provider]: !prev[provider],
    }));
  };

  if (isPending || loading) {
    return (
      <MiniMaxLayout>
        <div className="flex h-full items-center justify-center">
          <Loader2 className="h-8 w-8 animate-spin text-white" />
        </div>
      </MiniMaxLayout>
    );
  }

  return (
    <MiniMaxLayout>
      <div className="h-full overflow-y-auto">
        <div className="max-w-3xl mx-auto px-6 py-8">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold mb-2">BYOK Settings</h1>
            <p className="text-white/50">
              Bring Your Own Key (BYOK) allows you to use your own API keys for AI providers.
              You pay directly to the providers, avoiding platform markups.
            </p>
          </div>

          {/* Info Card */}
          <div className="bg-blue-500/10 border border-blue-500/20 rounded-xl p-4 mb-8">
            <div className="flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
              <div className="text-sm text-blue-200">
                <p className="font-medium mb-1">Why use BYOK?</p>
                <ul className="space-y-1 text-blue-200/70">
                  <li>• Pay providers directly - no platform markup</li>
                  <li>• Full control over your API usage and costs</li>
                  <li>• Use your own rate limits (not shared)</li>
                  <li>• Your keys are encrypted and stored securely</li>
                </ul>
              </div>
            </div>
          </div>

          {/* API Keys Form */}
          <div className="space-y-8">
            {providers.map((provider) => (
              <div
                key={provider.key}
                className="bg-[#141414] rounded-xl p-6 border border-white/5"
              >
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 bg-white/5 rounded-lg flex items-center justify-center">
                    <Key className="w-5 h-5 text-white/70" />
                  </div>
                  <div>
                    <h3 className="font-medium">{provider.name}</h3>
                    <p className="text-sm text-white/50">{provider.description}</p>
                  </div>
                </div>

                {/* Saved Keys */}
                {savedKeys[provider.key].length > 0 && (
                  <div className="mb-4 space-y-2">
                    <p className="text-sm text-white/50">Saved keys:</p>
                    {savedKeys[provider.key].map((savedKey) => (
                      <div
                        key={savedKey.id}
                        className="flex items-center justify-between p-3 bg-[#0d0d0d] rounded-lg border border-white/5"
                      >
                        <div className="flex items-center gap-2">
                          <span className="text-sm text-white">{savedKey.key_name}</span>
                          {savedKey.is_default && (
                            <span className="px-2 py-0.5 bg-green-500/20 text-green-400 text-xs rounded">
                              Default
                            </span>
                          )}
                        </div>
                        <div className="flex items-center gap-2">
                          {!savedKey.is_default && (
                            <button
                              onClick={() => handleSetDefault(savedKey.id, provider.key)}
                              className="p-1.5 hover:bg-white/5 rounded-lg transition-colors"
                              title="Set as default"
                            >
                              <Star className="w-4 h-4 text-white/50" />
                            </button>
                          )}
                          <button
                            onClick={() => handleDelete(savedKey.id, provider.key)}
                            className="p-1.5 hover:bg-red-500/20 rounded-lg transition-colors"
                            title="Delete key"
                          >
                            <Trash2 className="w-4 h-4 text-red-400" />
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* Add New Key */}
                <div className="space-y-3">
                  <input
                    type="text"
                    value={keyInputs[provider.key].keyName}
                    onChange={(e) =>
                      setKeyInputs((prev) => ({
                        ...prev,
                        [provider.key]: { ...prev[provider.key], keyName: e.target.value },
                      }))
                    }
                    placeholder="Key name (e.g., Production, Personal)"
                    className="w-full bg-[#0d0d0d] border border-white/10 rounded-lg px-4 py-2 text-white placeholder-white/30 focus:outline-none focus:border-white/20 text-sm"
                  />
                  <div className="relative">
                    <input
                      type={showKeys[provider.key] ? 'text' : 'password'}
                      value={keyInputs[provider.key].key}
                      onChange={(e) =>
                        setKeyInputs((prev) => ({
                          ...prev,
                          [provider.key]: { ...prev[provider.key], key: e.target.value },
                        }))
                      }
                      placeholder={provider.placeholder}
                      className="w-full bg-[#0d0d0d] border border-white/10 rounded-lg px-4 py-3 pr-24 text-white placeholder-white/30 focus:outline-none focus:border-white/20"
                    />
                    <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1">
                      <button
                        onClick={() => toggleVisibility(provider.key)}
                        className="p-1.5 hover:bg-white/5 rounded-lg transition-colors"
                      >
                        {showKeys[provider.key] ? (
                          <EyeOff className="w-4 h-4 text-white/50" />
                        ) : (
                          <Eye className="w-4 h-4 text-white/50" />
                        )}
                      </button>
                      <button
                        onClick={() => handleSave(provider.key)}
                        disabled={saving[provider.key] || !keyInputs[provider.key].key}
                        className="px-3 py-1.5 bg-white text-black rounded-lg text-sm font-medium hover:bg-white/90 transition-colors disabled:opacity-50 flex items-center gap-1"
                      >
                        {saving[provider.key] ? (
                          <Loader2 className="w-3 h-3 animate-spin" />
                        ) : (
                          <Save className="w-3 h-3" />
                        )}
                        Save
                      </button>
                    </div>
                  </div>

                  {keyInputs[provider.key].key && (
                    <div className="flex items-center gap-2 text-sm">
                      {provider.validate(keyInputs[provider.key].key) ? (
                        <>
                          <Check className="w-4 h-4 text-green-400" />
                          <span className="text-green-400">Valid key format</span>
                        </>
                      ) : (
                        <>
                          <AlertCircle className="w-4 h-4 text-yellow-400" />
                          <span className="text-yellow-400">Invalid key format</span>
                        </>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </MiniMaxLayout>
  );
}
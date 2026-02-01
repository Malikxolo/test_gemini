'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { api } from '@/lib/api';
import { toast } from 'sonner';
import { Loader2, CheckCircle, Package, FileText, ExternalLink } from 'lucide-react';

interface AccessStatus {
  hasAccess: boolean;
  hasDownloadAccess: boolean;
  purchasedAt: string;
  expiresAt: string | null;
}

export default function DownloadPage() {
  const router = useRouter();
  const [accessStatus, setAccessStatus] = useState<AccessStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [downloading, setDownloading] = useState(false);

  useEffect(() => {
    checkAccess();
  }, []);

  const checkAccess = async () => {
    try {
      const status = await api.payments.getAccessStatus();
      setAccessStatus(status);
      
      if (!status.hasDownloadAccess) {
        toast.error('You need to purchase access first');
        router.push('/purchase');
      }
    } catch (error) {
      toast.error('Failed to verify access');
      router.push('/purchase');
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async () => {
    try {
      setDownloading(true);
      // In a real app, this would trigger the actual download
      // For now, we'll simulate it
      await new Promise((resolve) => setTimeout(resolve, 1500));
      
      // Create a dummy download link (replace with actual download URL)
      const link = document.createElement('a');
      link.href = '/api/download/source-code';
      link.download = 'flowagent-source-v1.0.0.zip';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      toast.success('Download started!');
    } catch (error) {
      toast.error('Failed to start download');
    } finally {
      setDownloading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-[#0d0d0d] flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-white" />
      </div>
    );
  }

  if (!accessStatus?.hasDownloadAccess) {
    return null; // Will redirect
  }

  return (
    <div className="min-h-screen bg-[#0d0d0d] py-8 px-4">
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Download FlowAgent Source Code</h1>
          <p className="text-white/50">Access your purchased architecture files</p>
        </div>

        <div className="bg-green-500/10 border border-green-500/20 rounded-xl p-4 mb-6">
          <div className="flex items-center gap-2">
            <CheckCircle className="h-5 w-5 text-green-400" />
            <span className="font-semibold text-green-400">Access Verified</span>
          </div>
          <div className="mt-2 text-sm text-green-400/70">
            <p>Purchased on: {new Date(accessStatus.purchasedAt).toLocaleDateString()}</p>
            <p>License: Commercial Use Allowed</p>
          </div>
        </div>

        <div className="bg-[#141414] rounded-xl p-6 border border-white/5 mb-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 bg-blue-500/20 rounded-xl flex items-center justify-center">
              <Package className="w-5 h-5 text-blue-400" />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-white">FlowAgent v1.0.0</h2>
            </div>
          </div>
          
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div className="bg-[#0d0d0d] rounded-xl p-4 border border-white/5">
                <span className="text-white/50">Size:</span>
                <span className="ml-2 font-medium text-white">~50 MB (compressed)</span>
              </div>
              <div className="bg-[#0d0d0d] rounded-xl p-4 border border-white/5">
                <span className="text-white/50">Format:</span>
                <span className="ml-2 font-medium text-white">.zip</span>
              </div>
            </div>

            <div>
              <h4 className="font-medium text-white mb-2">Includes:</h4>
              <ul className="text-sm text-white/50 space-y-1">
                <li>• Full source code (Next.js + Cloudflare Workers + Python)</li>
                <li>• Database schemas and migrations</li>
                <li>• Docker and docker-compose configurations</li>
                <li>• Architecture documentation</li>
                <li>• Deployment guides</li>
                <li>• Environment configuration templates</li>
              </ul>
            </div>

            <button
              onClick={handleDownload}
              disabled={downloading}
              className="w-full flex items-center justify-center gap-2 px-6 py-4 bg-white text-black rounded-xl font-medium hover:bg-white/90 transition-colors disabled:opacity-50"
            >
              {downloading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Preparing Download...
                </>
              ) : (
                <>
                  <span>📦</span>
                  Download ZIP
                </>
              )}
            </button>
          </div>
        </div>

        <div className="bg-[#141414] rounded-xl p-6 border border-white/5 mb-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 bg-purple-500/20 rounded-xl flex items-center justify-center">
              <FileText className="w-5 h-5 text-purple-400" />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-white">Documentation</h2>
            </div>
          </div>
          
          <div className="space-y-2">
            {[
              { name: 'Architecture Guide', href: '/docs/architecture.md' },
              { name: 'Deployment Guide', href: '/docs/deployment.md' },
              { name: 'API Reference', href: '/docs/api-reference.md' },
              { name: 'Database Schema', href: '/docs/database-schema.md' },
            ].map((doc) => (
              <a
                key={doc.name}
                href={doc.href}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center justify-between p-3 rounded-xl bg-[#0d0d0d] border border-white/5 hover:bg-white/5 transition-colors"
              >
                <span className="text-white">{doc.name}</span>
                <ExternalLink className="h-4 w-4 text-white/40" />
              </a>
            ))}
          </div>
        </div>

        <div className="bg-blue-500/10 border border-blue-500/20 rounded-xl p-6">
          <h3 className="font-semibold text-blue-400 mb-2">Need Help?</h3>
          <p className="text-sm text-blue-400/70 mb-4">
            Join our Discord community for support, discussions, and updates.
          </p>
          <button
            onClick={() => window.open('https://discord.gg/flowagent', '_blank')}
            className="flex items-center gap-2 px-4 py-2 bg-white/5 text-white rounded-xl hover:bg-white/10 transition-colors"
          >
            Join Discord
            <ExternalLink className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}

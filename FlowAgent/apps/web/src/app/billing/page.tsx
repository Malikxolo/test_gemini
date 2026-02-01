'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { MiniMaxLayout } from '@/components/layout/MiniMaxLayout';
import { api, User as ApiUser, UsageStats, AccessStatus, PaymentHistory } from '@/lib/api';
import { toast } from 'sonner';
import {
  Zap,
  CreditCard,
  TrendingUp,
  Clock,
  CheckCircle,
  AlertCircle,
  Download,
  Sparkles,
  ChevronRight,
  Loader2,
  Bot,
  Key,
  DollarSign,
  ArrowRight,
} from 'lucide-react';

export default function BillingPage() {
  const router = useRouter();
  const [user, setUser] = useState<ApiUser | null>(null);
  const [usage, setUsage] = useState<UsageStats | null>(null);
  const [accessStatus, setAccessStatus] = useState<AccessStatus | null>(null);
  const [payments, setPayments] = useState<PaymentHistory | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      const [userData, usageData, accessData, paymentsData] = await Promise.all([
        api.auth.getCurrentUser(),
        api.users.getUsage(),
        api.payments.getAccessStatus(),
        api.payments.getHistory(),
      ]);
      
      setUser(userData);
      setUsage(usageData);
      setAccessStatus(accessData);
      setPayments(paymentsData);
    } catch (error) {
      toast.error('Failed to load billing data');
    } finally {
      setLoading(false);
    }
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(amount);
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
                <h1 className="text-3xl font-bold text-white mb-2">Usage & Billing</h1>
                <p className="text-white/50">Monitor your usage, manage billing, and track payments</p>
              </div>
              <button 
                onClick={() => router.push('/purchase')}
                className="flex items-center gap-2 px-4 py-2.5 bg-white/5 text-white rounded-xl hover:bg-white/10 transition-colors"
              >
                <CreditCard className="w-4 h-4" />
                Purchase Access
              </button>
            </div>
          </div>

          {/* Stats Overview */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <div className="bg-[#141414] rounded-xl p-6 border border-white/5">
              <div className="flex items-center gap-4">
                <div className="p-3 bg-blue-500/20 rounded-xl">
                  <Zap className="w-6 h-6 text-blue-400" />
                </div>
                <div>
                  <p className="text-sm text-white/50">Total Executions</p>
                  <p className="text-2xl font-bold text-white">{usage?.totalExecutions || 0}</p>
                </div>
              </div>
            </div>

            <div className="bg-[#141414] rounded-xl p-6 border border-white/5">
              <div className="flex items-center gap-4">
                <div className="p-3 bg-purple-500/20 rounded-xl">
                  <TrendingUp className="w-6 h-6 text-purple-400" />
                </div>
                <div>
                  <p className="text-sm text-white/50">Tokens Used</p>
                  <p className="text-2xl font-bold text-white">
                    {(usage?.totalTokensUsed || 0).toLocaleString()}
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-[#141414] rounded-xl p-6 border border-white/5">
              <div className="flex items-center gap-4">
                <div className="p-3 bg-green-500/20 rounded-xl">
                  <DollarSign className="w-6 h-6 text-green-400" />
                </div>
                <div>
                  <p className="text-sm text-white/50">Total Cost</p>
                  <p className="text-2xl font-bold text-white">
                    {formatCurrency(usage?.totalCost || 0)}
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-[#141414] rounded-xl p-6 border border-white/5">
              <div className="flex items-center gap-4">
                <div className="p-3 bg-orange-500/20 rounded-xl">
                  <Clock className="w-6 h-6 text-orange-400" />
                </div>
                <div>
                  <p className="text-sm text-white/50">Monthly Executions</p>
                  <p className="text-2xl font-bold text-white">{usage?.monthlyExecutions || 0}</p>
                </div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Main Content */}
            <div className="lg:col-span-2 space-y-6">
              {/* Access Status */}
              <div className="bg-[#141414] rounded-xl p-6 border border-white/5">
                <div className="flex items-center justify-between mb-6">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-blue-500/20 rounded-xl flex items-center justify-center">
                      <Sparkles className="w-5 h-5 text-blue-400" />
                    </div>
                    <div>
                      <h2 className="text-xl font-semibold text-white">Architecture Access</h2>
                      <p className="text-sm text-white/50">Your source code access status</p>
                    </div>
                  </div>
                  
                  {accessStatus?.hasAccess ? (
                    <span className="flex items-center gap-1.5 px-3 py-1.5 bg-green-500/20 text-green-400 rounded-lg text-sm">
                      <CheckCircle className="w-4 h-4" />
                      Active
                    </span>
                  ) : (
                    <span className="flex items-center gap-1.5 px-3 py-1.5 bg-white/10 text-white/60 rounded-lg text-sm">
                      <AlertCircle className="w-4 h-4" />
                      Not Purchased
                    </span>
                  )}
                </div>

                {accessStatus?.hasAccess ? (
                  <div className="space-y-6">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-[#0d0d0d] rounded-xl p-4 border border-white/5">
                        <p className="text-sm text-white/50 mb-1">Purchased On</p>
                        <p className="font-medium text-white">
                          {accessStatus.purchasedAt 
                            ? new Date(accessStatus.purchasedAt).toLocaleDateString()
                            : 'N/A'}
                        </p>
                      </div>
                      <div className="bg-[#0d0d0d] rounded-xl p-4 border border-white/5">
                        <p className="text-sm text-white/50 mb-1">Access Type</p>
                        <p className="font-medium text-white">Lifetime</p>
                      </div>
                    </div>

                    <button 
                      onClick={() => router.push('/download')}
                      className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-white text-black rounded-xl font-medium hover:bg-white/90 transition-colors"
                    >
                      <Download className="w-4 h-4" />
                      Download Source Code
                    </button>
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <div className="w-16 h-16 bg-white/5 rounded-full flex items-center justify-center mx-auto mb-4">
                      <Sparkles className="w-8 h-8 text-white/30" />
                    </div>
                    <h3 className="text-lg font-medium text-white mb-2">Get Full Access</h3>
                    <p className="text-white/50 mb-6 max-w-md mx-auto">
                      Purchase lifetime access to the complete FlowAgent source code, 
                      architecture documentation, and self-hosting rights.
                    </p>
                    <button 
                      onClick={() => router.push('/purchase')}
                      className="px-6 py-3 bg-white text-black rounded-xl font-medium hover:bg-white/90 transition-colors"
                    >
                      Purchase Access
                      <ArrowRight className="w-4 h-4 inline ml-2" />
                    </button>
                  </div>
                )}
              </div>

              {/* Payment History */}
              <div className="bg-[#141414] rounded-xl p-6 border border-white/5">
                <h2 className="text-xl font-semibold text-white mb-6">Payment History</h2>
                
                {payments?.payments && payments.payments.length > 0 ? (
                  <div className="space-y-3">
                    {payments.payments.map((payment) => (
                      <div
                        key={payment.id}
                        className="flex items-center justify-between p-4 bg-[#0d0d0d] rounded-xl border border-white/5"
                      >
                        <div className="flex items-center gap-4">
                          <div className="p-2 bg-green-500/20 rounded-xl">
                            <CheckCircle className="w-5 h-5 text-green-400" />
                          </div>
                          <div>
                            <p className="font-medium text-white">Architecture Access</p>
                            <p className="text-sm text-white/50">
                              {new Date(payment.createdAt).toLocaleDateString()}
                            </p>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className="font-semibold text-white">
                            {payment.currency === 'INR' ? '₹' : 
                             payment.currency === 'USD' ? '$' :
                             payment.currency === 'EUR' ? '€' : '£'}
                            {payment.amount}
                          </p>
                          <span className="px-2 py-0.5 bg-green-500/20 text-green-400 text-xs rounded">
                            {payment.status}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-white/50 text-center py-8">No payments yet</p>
                )}
              </div>
            </div>

            {/* Sidebar */}
            <div className="space-y-6">
              {/* BYOK Status */}
              <div className="bg-[#141414] rounded-xl p-6 border border-white/5">
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-10 h-10 bg-purple-500/20 rounded-xl flex items-center justify-center">
                    <Key className="w-5 h-5 text-purple-400" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold text-white">BYOK Status</h2>
                    <p className="text-sm text-white/50">Bring Your Own Key configuration</p>
                  </div>
                </div>
                
                <div className="space-y-3">
                  <div className="flex items-center justify-between p-3 bg-[#0d0d0d] rounded-xl border border-white/5">
                    <div className="flex items-center gap-3">
                      <Bot className="w-5 h-5 text-white/50" />
                      <span className="text-white">OpenAI</span>
                    </div>
                    <span className="px-2 py-0.5 bg-white/10 text-white/60 text-xs rounded">Not Set</span>
                  </div>

                  <div className="flex items-center justify-between p-3 bg-[#0d0d0d] rounded-xl border border-white/5">
                    <div className="flex items-center gap-3">
                      <Bot className="w-5 h-5 text-white/50" />
                      <span className="text-white">Anthropic</span>
                    </div>
                    <span className="px-2 py-0.5 bg-white/10 text-white/60 text-xs rounded">Not Set</span>
                  </div>
                </div>

                <button 
                  onClick={() => router.push('/byok')}
                  className="w-full mt-4 flex items-center justify-center gap-2 px-4 py-2.5 bg-white/5 text-white rounded-xl hover:bg-white/10 transition-colors"
                >
                  <Key className="w-4 h-4" />
                  Configure API Keys
                </button>
              </div>

              {/* Quick Links */}
              <div className="bg-[#141414] rounded-xl p-6 border border-white/5">
                <h2 className="text-lg font-semibold text-white mb-4">Quick Links</h2>
                
                <div className="space-y-2">
                  {[
                    { label: 'API Keys', href: '/byok' },
                    { label: 'Purchase Access', href: '/purchase' },
                    { label: 'Download', href: '/download' },
                  ].map((link) => (
                    <Link 
                      key={link.href}
                      href={link.href}
                      className="flex items-center justify-between p-3 rounded-xl hover:bg-white/5 transition-colors text-white"
                    >
                      <span className="text-sm">{link.label}</span>
                      <ChevronRight className="w-4 h-4 text-white/40" />
                    </Link>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </MiniMaxLayout>
  );
}

'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { MiniMaxLayout } from '@/components/layout/MiniMaxLayout';
import { api, User, AccessStatus } from '@/lib/api';
import { toast } from 'sonner';
import {
  User as UserIcon,
  Key,
  CreditCard,
  Receipt,
  Loader2,
  Download,
  ExternalLink,
  CheckCircle,
  AlertCircle,
} from 'lucide-react';

interface Payment {
  id: string;
  amount: string;
  currency: string;
  status: string;
  razorpayOrderId: string;
  createdAt: string;
  completedAt: string;
}

export default function SettingsPage() {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [accessStatus, setAccessStatus] = useState<AccessStatus | null>(null);
  const [payments, setPayments] = useState<Payment[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('general');

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      const [userData, accessData, paymentsData] = await Promise.all([
        api.auth.getCurrentUser(),
        api.payments.getAccessStatus(),
        api.payments.getHistory(),
      ]);
      
      setUser(userData);
      setAccessStatus(accessData);
      setPayments(paymentsData.payments || []);
    } catch (error) {
      toast.error('Failed to load settings');
    } finally {
      setLoading(false);
    }
  };

  const getStatusBadge = (status: string) => {
    const styles: Record<string, string> = {
      completed: 'bg-green-500/20 text-green-400',
      pending: 'bg-yellow-500/20 text-yellow-400',
      failed: 'bg-red-500/20 text-red-400',
      refunded: 'bg-white/10 text-white/60',
    };
    return <span className={`px-2 py-0.5 rounded text-xs ${styles[status] || styles.pending}`}>{status}</span>;
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
        <div className="max-w-4xl mx-auto px-6 py-8">
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-white mb-2">Account Settings</h1>
            <p className="text-white/50">Manage your account and access</p>
          </div>

          {/* Tabs */}
          <div className="flex gap-1 p-1 bg-[#141414] rounded-xl border border-white/5 mb-6 w-fit">
            {[
              { id: 'general', label: 'General', icon: UserIcon },
              { id: 'api-keys', label: 'API Keys', icon: Key },
              { id: 'billing', label: 'Billing', icon: CreditCard },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all ${
                  activeTab === tab.id
                    ? 'bg-white/10 text-white'
                    : 'text-white/50 hover:text-white/70'
                }`}
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          <div className="space-y-6">
            {activeTab === 'general' && (
              <div className="bg-[#141414] rounded-xl p-6 border border-white/5">
                <h2 className="text-xl font-semibold text-white mb-6">Profile Information</h2>
                
                <div className="grid grid-cols-2 gap-6">
                  <div className="bg-[#0d0d0d] rounded-xl p-4 border border-white/5">
                    <label className="text-sm text-white/50 block mb-1">Email</label>
                    <p className="font-medium text-white">{user?.email}</p>
                  </div>
                  
                  <div className="bg-[#0d0d0d] rounded-xl p-4 border border-white/5">
                    <label className="text-sm text-white/50 block mb-1">Username</label>
                    <p className="font-medium text-white">{user?.username}</p>
                  </div>
                  
                  <div className="bg-[#0d0d0d] rounded-xl p-4 border border-white/5">
                    <label className="text-sm text-white/50 block mb-1">Display Name</label>
                    <p className="font-medium text-white">{user?.displayName || 'Not set'}</p>
                  </div>
                  
                  <div className="bg-[#0d0d0d] rounded-xl p-4 border border-white/5">
                    <label className="text-sm text-white/50 block mb-1">Subscription</label>
                    <span className="px-2 py-0.5 bg-white/10 text-white text-xs rounded capitalize">
                      {user?.subscriptionTier}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'api-keys' && (
              <div className="space-y-6">
                <div className="bg-[#141414] rounded-xl p-6 border border-white/5">
                  <div className="flex items-center gap-3 mb-6">
                    <div className="w-10 h-10 bg-blue-500/20 rounded-xl flex items-center justify-center">
                      {accessStatus?.hasAccess ? (
                        <CheckCircle className="w-5 h-5 text-green-400" />
                      ) : (
                        <AlertCircle className="w-5 h-5 text-orange-400" />
                      )}
                    </div>
                    <div>
                      <h2 className="text-xl font-semibold text-white">Access Status</h2>
                      <p className="text-sm text-white/50">{accessStatus?.hasAccess ? 'You have full access' : 'Purchase access to unlock all features'}</p>
                    </div>
                  </div>

                  {accessStatus?.hasAccess ? (
                    <div className="space-y-4">
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
                          <p className="text-sm text-white/50 mb-1">Expires At</p>
                          <p className="font-medium text-white">
                            {accessStatus.expiresAt 
                              ? new Date(accessStatus.expiresAt).toLocaleDateString()
                              : 'Never'}
                          </p>
                        </div>
                      </div>

                      {accessStatus.hasDownloadAccess && (
                        <button
                          onClick={() => router.push('/download')}
                          className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-white text-black rounded-xl font-medium hover:bg-white/90 transition-colors"
                        >
                          <Download className="w-4 h-4" />
                          Download Source Code
                        </button>
                      )}
                    </div>
                  ) : (
                    <button
                      onClick={() => router.push('/purchase')}
                      className="w-full px-6 py-3 bg-white text-black rounded-xl font-medium hover:bg-white/90 transition-colors"
                    >
                      Purchase Access
                    </button>
                  )}
                </div>
                
                <button 
                  onClick={() => router.push('/byok')}
                  className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-white/5 text-white rounded-xl hover:bg-white/10 transition-colors"
                >
                  <Key className="w-4 h-4" />
                  Manage API Keys
                  <ExternalLink className="w-4 h-4" />
                </button>
              </div>
            )}

            {activeTab === 'billing' && (
              <div className="space-y-6">
                <div className="bg-[#141414] rounded-xl p-6 border border-white/5">
                  <div className="flex items-center gap-3 mb-6">
                    <div className="w-10 h-10 bg-blue-500/20 rounded-xl flex items-center justify-center">
                      <Receipt className="w-5 h-5 text-blue-400" />
                    </div>
                    <div>
                      <h2 className="text-xl font-semibold text-white">Payment History</h2>
                      <p className="text-sm text-white/50">Your recent transactions</p>
                    </div>
                  </div>

                  {payments.length === 0 ? (
                    <p className="text-white/50 text-center py-8">No payments found.</p>
                  ) : (
                    <div className="space-y-3">
                      {payments.map((payment) => (
                        <div
                          key={payment.id}
                          className="flex items-center justify-between p-4 bg-[#0d0d0d] rounded-xl border border-white/5"
                        >
                          <div>
                            <div className="flex items-center gap-2">
                              <span className="font-medium text-white">
                                {new Date(payment.createdAt).toLocaleDateString()}
                              </span>
                              {getStatusBadge(payment.status)}
                            </div>
                            <div className="text-sm text-white/50">
                              Architecture Access - Lifetime
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="font-semibold text-white">
                              {payment.currency === 'INR' ? '₹' : 
                               payment.currency === 'USD' ? '$' :
                               payment.currency === 'EUR' ? '€' : '£'}
                              {payment.amount}
                            </div>
                            <button
                              onClick={() => toast.info('Receipt download coming soon')}
                              className="text-sm text-white/50 hover:text-white transition-colors"
                            >
                              View Receipt
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </MiniMaxLayout>
  );
}

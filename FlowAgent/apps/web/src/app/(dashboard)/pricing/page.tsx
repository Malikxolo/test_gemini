'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { MiniMaxLayout } from '@/components/layout/MiniMaxLayout';
import { X, Check, Users, Sparkles } from 'lucide-react';
import { useSupabase } from '@/components/providers/supabase-provider';

const plans = [
  {
    name: 'Basic',
    badge: 'Early Bird',
    originalPrice: 39,
    price: 19,
    period: 'month',
    credits: '5,000',
    validity: '1 month',
    features: [
      'About 30 Pro mode tasks available',
      'Peak Hour Priority',
      'Remove Watermark and Custom Domain',
    ],
  },
  {
    name: 'Pro',
    badge: 'Early Bird',
    originalPrice: 119,
    price: 69,
    period: 'month',
    credits: '20,000',
    validity: '1 month',
    features: [
      'About 120 Pro mode tasks available',
      'Peak Hour Priority',
      'Remove Watermark and Custom Domain',
    ],
    popular: true,
  },
];

const addOns = [
  {
    name: 'Starter Pack',
    price: 39.00,
    credits: '5,000',
    validity: '1 year',
  },
  {
    name: 'Booster Pack',
    price: 78.00,
    credits: '10,000',
    validity: '1 year',
  },
  {
    name: 'Mega Pack',
    price: 156.00,
    credits: '20,000',
    validity: '1 year',
  },
];

export default function PricingPage() {
  const { session, isLoading: isPending } = useSupabase();
  const [activeTab, setActiveTab] = useState('plan');
  const [isOpen, setIsOpen] = useState(true);
  const router = useRouter();

  useEffect(() => {
    if (!isPending && !session) {
      router.push('/login');
    }
  }, [session, isPending, router]);

  if (isPending) {
    return (
      <div className="flex h-screen bg-[#0d0d0d] items-center justify-center">
        <div className="text-white/60">Loading...</div>
      </div>
    );
  }

  if (!isOpen) {
    router.push('/dashboard');
    return null;
  }

  return (
    <MiniMaxLayout>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
        <div className="w-full max-w-4xl bg-[#0d0d0d] rounded-2xl border border-white/10 overflow-hidden">
          {/* Header */}
          <div className="relative px-6 py-6 text-center">
            <button 
              onClick={() => setIsOpen(false)}
              className="absolute right-4 top-4 p-2 hover:bg-white/5 rounded-lg transition-colors"
            >
              <X className="w-5 h-5 text-white/50" />
            </button>

            <h2 className="text-2xl font-semibold mb-6">
              Choose Your Plan to Maximize Your Agent&apos;s Potential
            </h2>

            {/* Toggle */}
            <div className="inline-flex bg-white/5 rounded-full p-1">
              <button
                onClick={() => setActiveTab('plan')}
                className={`px-8 py-2 rounded-full text-sm font-medium transition-all ${
                  activeTab === 'plan'
                    ? 'bg-white/10 text-white'
                    : 'text-white/50 hover:text-white/70'
                }`}
              >
                Plan
              </button>
              <button
                onClick={() => setActiveTab('addon')}
                className={`px-8 py-2 rounded-full text-sm font-medium transition-all ${
                  activeTab === 'addon'
                    ? 'bg-white/10 text-white'
                    : 'text-white/50 hover:text-white/70'
                }`}
              >
                Add-on
              </button>
            </div>
          </div>

          {/* Content */}
          <div className="px-6 pb-6">
            {activeTab === 'plan' ? (
              <>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {plans.map((plan) => (
                    <div
                      key={plan.name}
                      className={`relative bg-[#141414] rounded-2xl p-6 border ${
                        plan.popular ? 'border-white/30' : 'border-white/10'
                      }`}
                    >
                      {plan.popular && (
                        <div className="absolute -top-px right-6 px-3 py-1 bg-white text-black text-xs font-medium rounded-b-lg">
                          Best Deal
                        </div>
                      )}

                      <div className="flex items-center gap-2 mb-4">
                        <span className="text-lg font-medium">{plan.name}</span>
                        <span className="px-2 py-0.5 bg-orange-500/20 text-orange-400 text-xs rounded">
                          {plan.badge}
                        </span>
                      </div>

                      <div className="mb-4">
                        <div className="text-white/40 line-through text-sm">
                          ${plan.originalPrice}.00
                        </div>
                        <div className="flex items-baseline gap-1">
                          <span className="text-4xl font-bold">${plan.price}.00</span>
                          <span className="text-white/50">/{plan.period}</span>
                        </div>
                      </div>

                      <button className="w-full py-3 bg-white text-black rounded-xl font-medium hover:bg-white/90 transition-colors mb-6">
                        Subscribe
                      </button>

                      <div className="flex items-center gap-2 mb-4">
                        <div className="w-6 h-6 rounded-full bg-white/10 flex items-center justify-center">
                          <span className="text-xs font-bold">M</span>
                        </div>
                        <span className="text-xl font-semibold">{plan.credits}</span>
                      </div>

                      <div className="text-sm text-white/50 mb-6">
                        Valid for {plan.validity}
                      </div>

                      <ul className="space-y-3">
                        {plan.features.map((feature, index) => (
                          <li key={index} className="flex items-start gap-3">
                            <Check className="w-5 h-5 text-white/70 flex-shrink-0 mt-0.5" />
                            <span className="text-sm text-white/70">{feature}</span>
                          </li>
                        ))}
                      </ul>

                      <div className="mt-6 pt-4 border-t border-white/5">
                        <div className="flex items-center gap-1 text-sm text-white/40">
                          <span>Credits will roll over</span>
                          <div className="w-4 h-4 rounded-full border border-white/20 flex items-center justify-center text-xs">?</div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Team Section */}
                <div className="mt-6 bg-[#141414] rounded-2xl p-6 border border-white/10">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div className="w-12 h-12 bg-white/5 rounded-xl flex items-center justify-center">
                        <Users className="w-6 h-6 text-white/70" />
                      </div>
                      <div>
                        <div className="font-medium">Team</div>
                        <div className="text-sm text-white/50">Maximize every talent on your team</div>
                      </div>
                    </div>
                    <button className="px-6 py-2.5 bg-white text-black rounded-xl font-medium hover:bg-white/90 transition-colors">
                      Get Team
                    </button>
                  </div>
                </div>
              </>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {addOns.map((addon) => (
                  <div
                    key={addon.name}
                    className="bg-[#141414] rounded-2xl p-6 border border-white/10"
                  >
                    <div className="text-lg font-medium mb-4">{addon.name}</div>

                    <div className="text-4xl font-bold mb-4">${addon.price}</div>

                    <button className="w-full py-3 bg-white text-black rounded-xl font-medium hover:bg-white/90 transition-colors mb-6">
                      Get Credits
                    </button>

                    <div className="flex items-center gap-2 mb-2">
                      <div className="w-6 h-6 rounded-full bg-white/10 flex items-center justify-center">
                        <span className="text-xs font-bold">M</span>
                      </div>
                      <span className="text-xl font-semibold">{addon.credits}</span>
                    </div>

                    <div className="text-sm text-white/50">
                      Valid for {addon.validity}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Footer */}
            <div className="mt-6 text-center text-sm text-white/40">
              Manage your subscription in{' '}
              <button className="text-white/60 hover:text-white underline">
                Settings
              </button>
            </div>
          </div>
        </div>
      </div>
    </MiniMaxLayout>
  );
}

'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { api } from '@/lib/api';
import { toast } from 'sonner';
import { Check, Loader2, Lock, Sparkles } from 'lucide-react';

interface CheckoutData {
  paymentId: string;
  razorpayOrderId: string;
  amount: number;
  currency: string;
  razorpayKeyId: string;
  userEmail: string;
  userName: string;
}

const pricingOptions = [
  { country: 'India', flag: '🇮🇳', amount: 800, currency: 'INR' as const, recommended: true },
  { country: 'USA', flag: '🇺🇸', amount: 10, currency: 'USD' as const },
  { country: 'Europe', flag: '🇪🇺', amount: 9.5, currency: 'EUR' as const },
  { country: 'UK', flag: '🇬🇧', amount: 8.5, currency: 'GBP' as const },
];

export default function PurchasePage() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [checkoutData, setCheckoutData] = useState<CheckoutData | null>(null);
  const [hasAccess, setHasAccess] = useState(false);

  useEffect(() => {
    checkAccess();
  }, []);

  const checkAccess = async () => {
    try {
      const status = await api.payments.getAccessStatus();
      setHasAccess(status.hasAccess);
      if (status.hasAccess) {
        router.push('/download');
      }
    } catch (error) {
      console.error('Failed to check access:', error);
    }
  };

  const handlePurchase = async (currency: 'INR' | 'USD' | 'EUR' | 'GBP') => {
    try {
      setLoading(true);
      const order = await api.payments.checkout({
        plan: 'access-tier',
        currency,
      });
      setCheckoutData(order);
    } catch (error) {
      toast.error('Failed to create checkout. Please try again.');
      setLoading(false);
    }
  };

  const handlePaymentSuccess = async (response: any) => {
    try {
      await api.payments.verify({
        razorpayOrderId: response.razorpay_order_id,
        razorpayPaymentId: response.razorpay_payment_id,
        razorpaySignature: response.razorpay_signature,
      });
      
      toast.success('Payment successful!');
      router.push('/purchase/success');
    } catch (error) {
      toast.error('Payment verification failed. Please contact support.');
      setLoading(false);
    }
  };

  const handlePaymentFailure = (error: any) => {
    toast.error(error.message || 'Payment failed');
    setLoading(false);
    setCheckoutData(null);
  };

  if (hasAccess) {
    return (
      <div className="min-h-screen bg-[#0d0d0d] flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-white" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0d0d0d] py-12 px-4">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-12">
          <div className="w-16 h-16 bg-gradient-to-br from-blue-400 to-purple-500 rounded-2xl flex items-center justify-center mx-auto mb-6">
            <Sparkles className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-4xl font-bold text-white mb-4">
            FlowAgent Architecture Access
          </h1>
          <p className="text-xl text-white/50 max-w-2xl mx-auto">
            Get lifetime access to the complete FlowAgent source code, architecture documentation, and self-hosting rights.
          </p>
        </div>

        {/* Features */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-12 max-w-3xl mx-auto">
          {[
            'Full source code',
            'Architecture documentation',
            'Self-hosting rights',
            'White-label capabilities',
            'Lifetime updates',
            'Commercial use allowed',
          ].map((feature, index) => (
            <div key={index} className="flex items-center gap-3">
              <div className="w-5 h-5 rounded-full bg-green-500/20 flex items-center justify-center">
                <Check className="w-3 h-3 text-green-400" />
              </div>
              <span className="text-white/70">{feature}</span>
            </div>
          ))}
        </div>

        {/* Pricing Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-12">
          {pricingOptions.map((option) => (
            <div
              key={option.currency}
              className={`relative bg-[#141414] rounded-2xl p-6 border transition-all hover:border-white/10 ${
                option.recommended ? 'border-white/20' : 'border-white/5'
              }`}
            >
              {option.recommended && (
                <div className="absolute -top-px right-6 px-3 py-1 bg-white text-black text-xs font-medium rounded-b-lg">
                  Recommended
                </div>
              )}

              <div className="flex items-center gap-2 mb-4">
                <span className="text-2xl">{option.flag}</span>
                <span className="font-medium text-white">{option.country}</span>
              </div>

              <div className="mb-6">
                <span className="text-4xl font-bold text-white">{option.currency === 'INR' ? '₹' : option.currency === 'USD' ? '$' : option.currency === 'EUR' ? '€' : '£'}</span>
                <span className="text-4xl font-bold text-white">{option.amount}</span>
                <span className="text-white/50"></span>
              </div>

              <button
                onClick={() => handlePurchase(option.currency)}
                disabled={loading && !checkoutData}
                className="w-full py-3 bg-white text-black rounded-xl font-medium hover:bg-white/90 transition-colors disabled:opacity-50"
              >
                {loading && !checkoutData ? (
                  <Loader2 className="w-4 h-4 animate-spin mx-auto" />
                ) : (
                  'Purchase'
                )}
              </button>
            </div>
          ))}
        </div>

        {/* Secure Payment */}
        <div className="bg-[#141414] rounded-2xl p-6 border border-white/5 max-w-2xl mx-auto">
          <div className="flex items-center gap-2 mb-4">
            <Lock className="h-5 w-5 text-white/50" />
            <h3 className="font-semibold text-white">Secure Payment</h3>
          </div>
          
          <div className="text-sm text-white/50 space-y-2">
            <p>
              <strong className="text-white">India:</strong> UPI, Cards, Net Banking, Wallets (Paytm, PhonePe, etc.)
            </p>
            <p>
              <strong className="text-white">International:</strong> Visa, Mastercard, American Express
            </p>
            <p className="flex items-center gap-2 pt-2">
              <span>🔒</span>
              Secure payment powered by Razorpay
            </p>
          </div>
        </div>
      </div>

      {/* Razorpay Checkout would be rendered here */}
      {checkoutData && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-[#141414] rounded-2xl p-8 max-w-md w-full text-center border border-white/10">
            <Loader2 className="w-8 h-8 animate-spin text-white mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-white mb-2">Loading Payment...</h3>
            <p className="text-white/50">Please wait while we redirect you to the payment gateway.</p>
          </div>
        </div>
      )}
    </div>
  );
}

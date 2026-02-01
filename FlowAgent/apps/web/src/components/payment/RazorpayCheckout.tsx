'use client';

import { useEffect } from 'react';
import Script from 'next/script';

interface RazorpayCheckoutProps {
  orderId: string;
  amount: number;
  currency: string;
  keyId: string;
  userEmail: string;
  userName: string;
  onSuccess: (response: any) => void;
  onFailure?: (error: any) => void;
}

export function RazorpayCheckout({
  orderId,
  amount,
  currency,
  keyId,
  userEmail,
  userName,
  onSuccess,
  onFailure,
}: RazorpayCheckoutProps) {
  useEffect(() => {
    if (orderId && keyId && typeof window !== 'undefined') {
      // Check if Razorpay is loaded
      if ((window as any).Razorpay) {
        openCheckout();
      }
    }
  }, [orderId, keyId]);

  const openCheckout = () => {
    const options = {
      key: keyId,
      amount: amount * 100, // Convert to paise/cents
      currency: currency,
      name: 'FlowAgent',
      description: 'Architecture Access - Lifetime',
      order_id: orderId,
      handler: function (response: any) {
        onSuccess(response);
      },
      prefill: {
        email: userEmail,
        name: userName,
      },
      theme: {
        color: '#3399cc',
      },
      modal: {
        ondismiss: function() {
          if (onFailure) {
            onFailure({ message: 'Payment cancelled by user' });
          }
        }
      }
    };

    const rzp = new (window as any).Razorpay(options);
    rzp.open();
  };

  return (
    <Script
      src="https://checkout.razorpay.com/v1/checkout.js"
      strategy="lazyOnload"
    />
  );
}

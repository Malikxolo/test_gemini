import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Loader2 } from 'lucide-react';

interface PricingCardProps {
  country: string;
  flag: string;
  amount: number;
  currency: 'INR' | 'USD' | 'EUR' | 'GBP';
  recommended?: boolean;
  onPurchase: (currency: 'INR' | 'USD' | 'EUR' | 'GBP') => void;
  loading?: boolean;
}

export function PricingCard({
  country,
  flag,
  amount,
  currency,
  recommended = false,
  onPurchase,
  loading = false,
}: PricingCardProps) {
  const formatAmount = (amt: number, curr: string) => {
    const symbols: Record<string, string> = { INR: '₹', USD: '$', EUR: '€', GBP: '£' };
    return `${symbols[curr]}${amt}`;
  };

  return (
    <Card className={recommended ? 'border-blue-500 border-2' : ''}>
      {recommended && (
        <div className="bg-blue-500 text-white text-center py-1 text-sm font-medium">
          Recommended
        </div>
      )}
      <CardHeader>
        <CardTitle className="flex items-center justify-center gap-2">
          <span className="text-3xl">{flag}</span>
          {country}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="text-center">
          <div className="text-4xl font-bold">
            {formatAmount(amount, currency)}
          </div>
          <div className="text-sm text-gray-600">One-time payment</div>
        </div>
        <Button
          onClick={() => onPurchase(currency)}
          disabled={loading}
          className="w-full"
        >
          {loading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Processing...
            </>
          ) : (
            'Purchase'
          )}
        </Button>
      </CardContent>
    </Card>
  );
}

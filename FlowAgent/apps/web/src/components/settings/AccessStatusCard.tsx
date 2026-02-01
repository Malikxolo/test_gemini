import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { CheckCircle, Download, ExternalLink } from 'lucide-react';

interface AccessStatusCardProps {
  hasAccess: boolean;
  purchasedAt?: string;
  expiresAt?: string | null;
  onDownload?: () => void;
}

export function AccessStatusCard({
  hasAccess,
  purchasedAt,
  expiresAt,
  onDownload,
}: AccessStatusCardProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <span>🎯</span>
          Architecture Access
        </CardTitle>
      </CardHeader>
      <CardContent>
        {hasAccess ? (
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <CheckCircle className="h-5 w-5 text-green-500" />
              <Badge variant="default" className="bg-green-500">Active</Badge>
            </div>
            
            {purchasedAt && (
              <div className="text-sm text-gray-600">
                Purchased: {new Date(purchasedAt).toLocaleDateString()}
              </div>
            )}
            
            <div className="text-sm text-gray-600">
              Expires: {expiresAt ? new Date(expiresAt).toLocaleDateString() : 'Never (Lifetime)'}
            </div>
            
            {onDownload && (
              <Button onClick={onDownload} className="w-full">
                <Download className="mr-2 h-4 w-4" />
                Download Source Code
              </Button>
            )}
          </div>
        ) : (
          <div className="space-y-4">
            <Badge variant="secondary">No Access</Badge>
            <p className="text-sm text-gray-600">
              Purchase architecture access to download the source code and documentation.
            </p>
            <Button asChild className="w-full">
              <a href="/purchase">
                Purchase Access
                <ExternalLink className="ml-2 h-4 w-4" />
              </a>
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

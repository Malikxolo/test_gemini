import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Loader2, Trash2 } from 'lucide-react';

interface APIKeyCardProps {
  name: string;
  icon: string;
  currentKey: string | null;
  onUpdate: () => void;
  onDelete: () => void;
  description?: string;
  loading?: boolean;
}

export function APIKeyCard({
  name,
  icon,
  currentKey,
  onUpdate,
  onDelete,
  description,
  loading = false,
}: APIKeyCardProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <span>{icon}</span>
          {name}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {currentKey ? (
            <>
              <div className="flex items-center justify-between">
                <code className="text-sm bg-gray-100 px-2 py-1 rounded">
                  {currentKey}
                </code>
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={onDelete}
                  disabled={loading}
                >
                  {loading ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Trash2 className="h-4 w-4" />
                  )}
                </Button>
              </div>
              <Button onClick={onUpdate} className="w-full" disabled={loading}>
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Updating...
                  </>
                ) : (
                  'Update Key'
                )}
              </Button>
            </>
          ) : (
            <>
              <Badge variant="secondary">Not configured</Badge>
              <Button onClick={onUpdate} className="w-full" disabled={loading}>
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Adding...
                  </>
                ) : (
                  'Add Key'
                )}
              </Button>
            </>
          )}
          {description && (
            <p className="text-sm text-gray-600">{description}</p>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

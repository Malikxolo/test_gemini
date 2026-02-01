'use client'

import { useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { AlertCircle } from 'lucide-react'

export default function ErrorBoundary({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  useEffect(() => {
    // Log error to monitoring service
    console.error('Application error:', error)
  }, [error])

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#0d0d0d]">
      <div className="max-w-md w-full mx-4 p-8 bg-[#141414] rounded-2xl border border-white/10 text-center">
        <div className="w-16 h-16 mx-auto mb-6 bg-red-500/10 rounded-full flex items-center justify-center">
          <AlertCircle className="w-8 h-8 text-red-400" />
        </div>
        
        <h2 className="text-xl font-semibold text-white mb-2">
          Something went wrong
        </h2>
        
        <p className="text-white/50 mb-6">
          {error.message || 'An unexpected error occurred. Please try again.'}
        </p>
        
        <div className="flex gap-3 justify-center">
          <Button
            onClick={reset}
            variant="default"
            className="bg-white text-black hover:bg-white/90"
          >
            Try again
          </Button>
          
          <Button
            onClick={() => window.location.href = '/'}
            variant="outline"
            className="border-white/20 text-white hover:bg-white/5"
          >
            Go home
          </Button>
        </div>
        
        {error.digest && (
          <p className="mt-4 text-xs text-white/30 font-mono">
            Error ID: {error.digest}
          </p>
        )}
      </div>
    </div>
  )
}
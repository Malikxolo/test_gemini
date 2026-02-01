import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Home, AlertCircle } from 'lucide-react'

export default function NotFound() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-[#0d0d0d]">
      <div className="max-w-md w-full mx-4 p-8 bg-[#141414] rounded-2xl border border-white/10 text-center">
        <div className="w-16 h-16 mx-auto mb-6 bg-white/5 rounded-full flex items-center justify-center">
          <AlertCircle className="w-8 h-8 text-white/50" />
        </div>
        
        <h1 className="text-4xl font-bold text-white mb-2">404</h1>
        <h2 className="text-xl font-semibold text-white mb-2">Page not found</h2>
        
        <p className="text-white/50 mb-6">
          The page you&apos;re looking for doesn&apos;t exist or has been moved.
        </p>
        
        <Link href="/">
          <Button className="bg-white text-black hover:bg-white/90">
            <Home className="w-4 h-4 mr-2" />
            Go home
          </Button>
        </Link>
      </div>
    </div>
  )
}
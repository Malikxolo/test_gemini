'use client'

import { Loader2 } from 'lucide-react'

export default function Loading() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-[#0d0d0d]">
      <div className="text-center">
        <Loader2 className="w-10 h-10 animate-spin text-white/50 mx-auto mb-4" />
        <p className="text-white/50 text-sm">Loading FlowAgent...</p>
      </div>
    </div>
  )
}
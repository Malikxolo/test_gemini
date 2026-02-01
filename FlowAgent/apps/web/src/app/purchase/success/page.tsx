'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { CheckCircle, Download, FileText, ArrowRight, Mail, Sparkles } from 'lucide-react';

export default function PurchaseSuccessPage() {
  const router = useRouter();

  useEffect(() => {
    // Optional: Check access status
    const checkAccess = async () => {
      // If needed, verify user has access before showing this page
    };
    checkAccess();
  }, []);

  const handleDownload = () => {
    router.push('/download');
  };

  return (
    <div className="min-h-screen bg-[#0d0d0d] flex items-center justify-center py-12 px-4">
      <div className="max-w-2xl w-full">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-20 h-20 bg-green-500/20 rounded-full mb-6">
            <CheckCircle className="h-10 w-10 text-green-400" />
          </div>
          
          <h1 className="text-3xl font-bold text-white mb-2">
            Payment Successful!
          </h1>
          
          <p className="text-lg text-white/50">
            You now have lifetime access to FlowAgent architecture!
          </p>
        </div>

        <div className="bg-[#141414] rounded-2xl p-6 border border-white/5 mb-6">
          <h2 className="font-semibold text-lg text-white mb-4">What&apos;s Next:</h2>
          
          <ol className="space-y-3 text-white/70">
            <li className="flex items-start gap-2">
              <span className="font-semibold text-blue-400">1.</span>
              Download the source code
            </li>
            <li className="flex items-start gap-2">
              <span className="font-semibold text-blue-400">2.</span>
              Review the architecture documentation
            </li>
            <li className="flex items-start gap-2">
              <span className="font-semibold text-blue-400">3.</span>
              Deploy on your infrastructure
            </li>
            <li className="flex items-start gap-2">
              <span className="font-semibold text-blue-400">4.</span>
              Customize as needed
            </li>
          </ol>
        </div>

        <div className="space-y-4">
          <button 
            onClick={handleDownload}
            className="w-full flex items-center justify-center gap-2 px-6 py-4 bg-white text-black rounded-xl font-medium hover:bg-white/90 transition-colors"
          >
            <Download className="w-5 h-5" />
            Download Source Code
          </button>

          <button
            onClick={() => window.open('/docs/architecture.md', '_blank')}
            className="w-full flex items-center justify-center gap-2 px-6 py-4 bg-white/5 text-white rounded-xl font-medium hover:bg-white/10 transition-colors"
          >
            <FileText className="w-5 h-5" />
            View Documentation
          </button>
        </div>

        <div className="mt-8 text-center">
          <div className="flex items-center justify-center gap-2 text-white/50 mb-4">
            <Mail className="h-4 w-4" />
            <span className="text-sm">
              A receipt has been sent to your email address
            </span>
          </div>

          <button
            onClick={() => router.push('/dashboard')}
            className="text-blue-400 hover:text-blue-300 transition-colors"
          >
            Go to Dashboard
            <ArrowRight className="w-4 h-4 inline ml-2" />
          </button>
        </div>
      </div>
    </div>
  );
}

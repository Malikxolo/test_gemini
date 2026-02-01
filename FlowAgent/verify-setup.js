// Verification script for FlowAgent setup
const fs = require('fs');
const path = require('path');

console.log('🔍 Verifying FlowAgent Setup...\n');

let allPassed = true;

// Check 1: .env.local exists
const envPath = path.join(__dirname, 'apps/web/.env.local');
if (fs.existsSync(envPath)) {
  console.log('✅ .env.local file exists');
  
  const envContent = fs.readFileSync(envPath, 'utf8');
  
  // Check required variables
  const required = [
    'NEXT_PUBLIC_SUPABASE_URL',
    'NEXT_PUBLIC_SUPABASE_ANON_KEY',
    'SUPABASE_SERVICE_ROLE_KEY',
    'ENCRYPTION_KEY',
    'NEXT_PUBLIC_APP_URL'
  ];
  
  const missing = [];
  required.forEach(key => {
    if (!envContent.includes(key + '=')) {
      missing.push(key);
    }
  });
  
  if (missing.length === 0) {
    console.log('✅ All required environment variables present');
  } else {
    console.log('❌ Missing variables:', missing.join(', '));
    allPassed = false;
  }
  
  // Check ENCRYPTION_KEY length
  const encryptionMatch = envContent.match(/ENCRYPTION_KEY=(.+)/);
  if (encryptionMatch) {
    const key = encryptionMatch[1].trim();
    if (key.length >= 32) {
      console.log('✅ ENCRYPTION_KEY is valid (32+ characters)');
    } else {
      console.log('❌ ENCRYPTION_KEY too short (needs 32+ characters)');
      allPassed = false;
    }
  }
  
  // Check Supabase URL format
  if (envContent.includes('supabase.co')) {
    console.log('✅ Supabase URL configured');
  } else {
    console.log('❌ Supabase URL not configured');
    allPassed = false;
  }
  
} else {
  console.log('❌ .env.local file NOT found');
  allPassed = false;
}

// Check 2: Build output exists
const buildPath = path.join(__dirname, 'apps/web/.next');
if (fs.existsSync(buildPath)) {
  console.log('✅ Build output exists (.next directory)');
} else {
  console.log('⚠️  Build output not found (run: npm run build)');
}

// Check 3: API routes exist
const apiRoutesPath = path.join(__dirname, 'apps/web/src/app/api');
if (fs.existsSync(apiRoutesPath)) {
  console.log('✅ API routes directory exists');
} else {
  console.log('❌ API routes directory NOT found');
  allPassed = false;
}

// Check 4: Security modules exist
const encryptionPath = path.join(__dirname, 'apps/web/src/lib/server/encryption.ts');
if (fs.existsSync(encryptionPath)) {
  console.log('✅ Server-side encryption module exists');
} else {
  console.log('❌ Encryption module NOT found');
  allPassed = false;
}

console.log('\n' + '='.repeat(50));
if (allPassed) {
  console.log('✅ ALL CHECKS PASSED - Ready for deployment!');
  console.log('\nNext steps:');
  console.log('1. Deploy to Vercel: cd apps/web && vercel --prod');
  console.log('2. Add environment variables to Vercel Dashboard');
  console.log('3. Configure Supabase Auth URLs');
  console.log('4. Test your deployment!');
} else {
  console.log('❌ SOME CHECKS FAILED - Please review above');
}
console.log('='.repeat(50));

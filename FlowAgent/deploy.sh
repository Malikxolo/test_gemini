#!/bin/bash

echo "🚀 FlowAgent Deployment Script"
echo "================================"
echo ""
echo "This script will guide you through deployment."
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js is not installed${NC}"
    echo "Install from: https://nodejs.org"
    exit 1
fi

if ! command -v pnpm &> /dev/null; then
    echo -e "${YELLOW}⚠️  pnpm not found. Installing...${NC}"
    npm install -g pnpm
fi

if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Git is not installed${NC}"
    echo "Install from: https://git-scm.com"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites met${NC}"
echo ""

# Generate encryption key
echo "🔐 Generating encryption key..."
ENCRYPTION_KEY=$(node -e "console.log(require('crypto').randomBytes(32).toString('hex'))")
echo ""
echo -e "${YELLOW}⚠️  SAVE THIS KEY SECURELY:${NC}"
echo "================================"
echo "$ENCRYPTION_KEY"
echo "================================"
echo ""
echo "Copy this key to your password manager NOW!"
echo "Press Enter once you've saved it..."
read

# Check if .env.local exists
if [ ! -f "apps/web/.env.local" ]; then
    echo "📝 Creating .env.local template..."
    cat > apps/web/.env.local <> EOF
# Fill in these values from Supabase Dashboard:
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key-here
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key-here
NEXT_PUBLIC_APP_URL=http://localhost:3000
ENCRYPTION_KEY=$ENCRYPTION_KEY
EOF
    echo -e "${YELLOW}⚠️  Edit apps/web/.env.local and add your Supabase credentials${NC}"
    echo "Get them from: https://supabase.com/dashboard → Your Project → Settings → API"
else
    echo -e "${GREEN}✅ .env.local already exists${NC}"
fi

echo ""
echo "📋 NEXT STEPS:"
echo "=============="
echo ""
echo "1. Edit apps/web/.env.local with your Supabase credentials"
echo ""
echo "2. Run database migration:"
echo "   - Go to https://supabase.com/dashboard"
echo "   - SQL Editor → New Query"
echo "   - Copy contents of supabase/schema_v2.sql"
echo "   - Run the SQL"
echo ""
echo "3. Push code to GitHub:"
echo "   git add ."
echo "   git commit -m 'Ready for deployment'"
echo "   git push origin main"
echo ""
echo "4. Deploy to Vercel:"
echo "   cd apps/web"
echo "   vercel --prod"
echo ""
echo "5. Add environment variables in Vercel Dashboard"
echo ""
echo "Complete guide: DO_IT_FOR_ME.md"
echo ""

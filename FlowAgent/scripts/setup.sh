#!/bin/bash

# FlowAgent Setup Script
# This script helps you set up FlowAgent for local development

set -e

echo "🚀 FlowAgent Setup Script"
echo "=========================="
echo ""

# Check for required tools
echo "📋 Checking prerequisites..."

command -v node >/dev/null 2>&1 || { echo "❌ Node.js is required but not installed. Please install Node.js 20+"; exit 1; }
command -v pnpm >/dev/null 2>&1 || { echo "❌ pnpm is required but not installed. Run: npm install -g pnpm"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "⚠️  Python 3.11+ recommended for agent engine"; }

echo "✅ All prerequisites found!"
echo ""

# Check Node version
NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt "20" ]; then
    echo "⚠️  Node.js 20+ recommended (you have v$NODE_VERSION)"
fi

# Install dependencies
echo "📦 Installing dependencies..."
pnpm install

echo "✅ Dependencies installed!"
echo ""

# Check for .env file
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "✅ .env file created!"
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env and add your API keys:"
    echo "   - DATABASE_URL (Neon Postgres)"
    echo "   - UPSTASH_REDIS_REST_URL"
    echo "   - UPSTASH_REDIS_REST_TOKEN"
    echo "   - OPENAI_API_KEY"
    echo ""
    echo "Press Enter when done..."
    read
else
    echo "✅ .env file already exists"
fi

# Check if DATABASE_URL is set
if grep -q "DATABASE_URL=postgresql://user:password@host" .env; then
    echo ""
    echo "⚠️  WARNING: DATABASE_URL still has placeholder value!"
    echo "   Please update it in .env before continuing."
    echo ""
    echo "Continue anyway? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Run database setup
echo ""
echo "🗄️  Setting up database..."
echo ""
echo "This will:"
echo "  1. Generate database migrations"
echo "  2. Run migrations on your database"
echo ""
echo "Make sure your DATABASE_URL is configured!"
echo "Continue? (y/N)"
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "Generating migrations..."
    cd packages/database
    pnpm drizzle-kit generate:pg

    echo "Running migrations..."
    pnpm drizzle-kit migrate

    cd ../..
    echo "✅ Database setup complete!"
else
    echo "⏭️  Skipping database setup"
fi

# Setup complete
echo ""
echo "✨ Setup Complete!"
echo ""
echo "Next steps:"
echo "  1. Make sure all environment variables are set in .env"
echo "  2. Run 'pnpm dev' to start development servers"
echo "  3. Open http://localhost:3000 in your browser"
echo ""
echo "Development servers:"
echo "  - Frontend (Next.js):  http://localhost:3000"
echo "  - API (Workers):       http://localhost:8787"
echo ""
echo "For deployment, see DEPLOYMENT.md"
echo ""
echo "Happy coding! 🎉"

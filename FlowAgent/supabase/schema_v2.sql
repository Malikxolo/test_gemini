-- Create the updated_at trigger function first
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = timezone('utc'::text, now());
  RETURN NEW;
END;
$$ language 'plpgsql';

-- API Keys table (encrypted storage) - No dependencies
CREATE TABLE IF NOT EXISTS public.api_keys (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  provider TEXT NOT NULL CHECK (provider IN ('openai', 'anthropic', 'google', 'openrouter')),
  key_name TEXT NOT NULL,
  encrypted_key TEXT NOT NULL,
  is_active BOOLEAN DEFAULT true,
  is_default BOOLEAN DEFAULT false,
  last_used_at TIMESTAMP WITH TIME ZONE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()),
  UNIQUE(user_id, provider, key_name)
);

-- Projects table (project management) - No dependencies
CREATE TABLE IF NOT EXISTS public.projects (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  name TEXT NOT NULL,
  description TEXT,
  context TEXT,
  files JSONB DEFAULT '[]',
  default_model TEXT DEFAULT 'gpt-4',
  default_provider TEXT DEFAULT 'openai',
  total_conversations INTEGER DEFAULT 0,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now())
);

-- Agent Personas table (pre-built and custom agents) - No dependencies
CREATE TABLE IF NOT EXISTS public.agent_personas (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  role TEXT NOT NULL,
  description TEXT NOT NULL,
  system_prompt TEXT NOT NULL,
  icon TEXT DEFAULT 'bot',
  color TEXT DEFAULT 'blue',
  is_prebuilt BOOLEAN DEFAULT false,
  is_active BOOLEAN DEFAULT true,
  model TEXT DEFAULT 'gpt-4',
  provider TEXT DEFAULT 'openai',
  temperature DECIMAL(3,2) DEFAULT 0.7,
  max_tokens INTEGER DEFAULT 2000,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now())
);

-- Conversations table (chat history) - Depends on projects and agent_personas
CREATE TABLE IF NOT EXISTS public.conversations (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  title TEXT NOT NULL,
  project_id UUID REFERENCES public.projects(id) ON DELETE SET NULL,
  agent_persona_id UUID REFERENCES public.agent_personas(id) ON DELETE SET NULL,
  model TEXT DEFAULT 'gpt-4',
  provider TEXT DEFAULT 'openai',
  total_tokens INTEGER DEFAULT 0,
  total_cost DECIMAL(10,4) DEFAULT 0,
  budget_used DECIMAL(10,4) DEFAULT 0,
  budget_limit DECIMAL(10,4) DEFAULT 5.00,
  status TEXT DEFAULT 'active',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now())
);

-- Messages table (individual chat messages) - Depends on conversations
CREATE TABLE IF NOT EXISTS public.messages (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  conversation_id UUID REFERENCES public.conversations(id) ON DELETE CASCADE NOT NULL,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
  content TEXT NOT NULL,
  agent_name TEXT,
  tokens_used INTEGER DEFAULT 0,
  cost DECIMAL(10,4) DEFAULT 0,
  model TEXT,
  provider TEXT,
  attachments JSONB DEFAULT '[]',
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now())
);

-- Usage tracking table
CREATE TABLE IF NOT EXISTS public.usage_tracking (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  conversation_id UUID REFERENCES public.conversations(id) ON DELETE CASCADE,
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  tokens_input INTEGER DEFAULT 0,
  tokens_output INTEGER DEFAULT 0,
  tokens_total INTEGER DEFAULT 0,
  cost_input DECIMAL(10,4) DEFAULT 0,
  cost_output DECIMAL(10,4) DEFAULT 0,
  cost_total DECIMAL(10,4) DEFAULT 0,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now())
);

-- Enable RLS on all tables
ALTER TABLE public.conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.api_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.agent_personas ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.usage_tracking ENABLE ROW LEVEL SECURITY;

-- RLS Policies for conversations
CREATE POLICY "Users can view own conversations" 
  ON public.conversations FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can create own conversations" 
  ON public.conversations FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own conversations" 
  ON public.conversations FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own conversations" 
  ON public.conversations FOR DELETE USING (auth.uid() = user_id);

-- RLS Policies for messages
CREATE POLICY "Users can view messages in their conversations" 
  ON public.messages FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can create messages in their conversations" 
  ON public.messages FOR INSERT WITH CHECK (auth.uid() = user_id);

-- RLS Policies for api_keys
CREATE POLICY "Users can view own API keys" 
  ON public.api_keys FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can manage own API keys" 
  ON public.api_keys FOR ALL USING (auth.uid() = user_id);

-- RLS Policies for projects
CREATE POLICY "Users can view own projects" 
  ON public.projects FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can manage own projects" 
  ON public.projects FOR ALL USING (auth.uid() = user_id);

-- RLS Policies for agent_personas
CREATE POLICY "Users can view prebuilt and own personas" 
  ON public.agent_personas FOR SELECT 
  USING (is_prebuilt = true OR auth.uid() = user_id);

CREATE POLICY "Users can manage own personas" 
  ON public.agent_personas FOR ALL 
  USING (auth.uid() = user_id);

-- RLS Policies for usage_tracking
CREATE POLICY "Users can view own usage" 
  ON public.usage_tracking FOR SELECT USING (auth.uid() = user_id);

-- Insert pre-built agent personas
INSERT INTO public.agent_personas (name, role, description, system_prompt, icon, color, is_prebuilt, model, provider) VALUES
(
  'Project Manager',
  'pm',
  'Orchestrates complex projects by breaking down tasks and coordinating specialized agents',
  'You are a senior project manager with 15+ years of experience leading complex technical projects. Your role is to:
1. Analyze user requests and break them into subtasks
2. Coordinate with specialized agents (SDE, Designer, DevOps, Data Analyst)
3. Synthesize their outputs into cohesive, actionable responses
4. Ensure quality and completeness of deliverables
5. Ask clarifying questions when requirements are unclear
You are the primary interface to the user and manage the entire swarm.',
  'users',
  'purple',
  true,
  'gpt-4',
  'openai'
),
(
  'Senior SDE',
  'sde',
  'Expert software engineer specializing in coding, architecture, and technical implementation',
  'You are a senior software engineer with 10+ years of experience. Your expertise includes:
- Writing clean, efficient, production-ready code
- System architecture and design patterns
- Code review and optimization
- Debugging complex issues
- Best practices and industry standards
Provide detailed, well-documented code solutions with explanations.',
  'code',
  'blue',
  true,
  'gpt-4',
  'openai'
),
(
  'UI/UX Designer',
  'designer',
  'Creative designer focused on user experience, visual design, and frontend implementation',
  'You are a creative UI/UX designer with expertise in:
- User-centered design principles
- Visual design and aesthetics
- Frontend implementation (HTML, CSS, React, etc.)
- Accessibility standards (WCAG)
- Design systems and component libraries
- Prototyping and wireframing
Create beautiful, intuitive, and accessible designs.',
  'palette',
  'pink',
  true,
  'gpt-4',
  'openai'
),
(
  'DevOps Engineer',
  'devops',
  'Infrastructure expert handling deployment, automation, and cloud architecture',
  'You are a DevOps engineer specializing in:
- Cloud infrastructure (AWS, Azure, GCP)
- CI/CD pipelines and automation
- Container orchestration (Docker, Kubernetes)
- Infrastructure as Code (Terraform, CloudFormation)
- Monitoring and logging
- Security and compliance
Provide robust, scalable infrastructure solutions.',
  'server',
  'green',
  true,
  'gpt-4',
  'openai'
),
(
  'Data Analyst',
  'analyst',
  'Data scientist specializing in analysis, visualization, and insights extraction',
  'You are a data scientist with expertise in:
- Data analysis and statistical methods
- Data visualization (charts, dashboards)
- Python, SQL, and data processing
- Machine learning basics
- Business intelligence and insights
- Data cleaning and preparation
Transform raw data into actionable insights with clear visualizations.',
  'bar-chart',
  'orange',
  true,
  'gpt-4',
  'openai'
);

-- Create triggers for updated_at
CREATE TRIGGER update_conversations_updated_at 
  BEFORE UPDATE ON public.conversations 
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_messages_updated_at 
  BEFORE UPDATE ON public.messages 
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_api_keys_updated_at 
  BEFORE UPDATE ON public.api_keys 
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_projects_updated_at 
  BEFORE UPDATE ON public.projects 
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agent_personas_updated_at 
  BEFORE UPDATE ON public.agent_personas 
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create storage bucket for chat attachments
INSERT INTO storage.buckets (id, name, public) 
VALUES ('chat-attachments', 'chat-attachments', true)
ON CONFLICT (id) DO NOTHING;

-- Storage policies for chat attachments
CREATE POLICY "Users can upload attachments to their conversations"
  ON storage.objects FOR INSERT 
  WITH CHECK (
    bucket_id = 'chat-attachments' 
    AND auth.uid() IS NOT NULL
  );

CREATE POLICY "Users can view their conversation attachments"
  ON storage.objects FOR SELECT 
  USING (
    bucket_id = 'chat-attachments' 
    AND auth.uid() IS NOT NULL
  );

CREATE POLICY "Users can delete their attachments"
  ON storage.objects FOR DELETE 
  USING (
    bucket_id = 'chat-attachments' 
    AND auth.uid() IS NOT NULL
  );

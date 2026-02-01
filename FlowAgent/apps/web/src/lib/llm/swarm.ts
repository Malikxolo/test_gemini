import { streamChatMessage } from './conversations';
import type { Provider } from './provider';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface AgentPersona {
  id: string;
  name: string;
  role: string;
  systemPrompt: string;
  model: string;
  provider: string;
}

export interface ModelConfig {
  provider: Provider;
  model: string;
  temperature?: number;
  maxTokens?: number;
}

export interface SwarmTask {
  agent: AgentPersona;
  task: string;
  context: string;
}

export interface SwarmResponse {
  agentName: string;
  agentRole: string;
  response: string;
  tokens: number;
  cost: number;
}

export interface SwarmUpdate {
  agent: string;
  status: 'thinking' | 'working' | 'complete';
  partial?: string;
  fullResponse?: string;
}

// The 5 pre-built agents
export const PREBUILT_AGENTS: AgentPersona[] = [
  {
    id: 'pm',
    name: 'Project Manager',
    role: 'pm',
    systemPrompt: `You are a senior project manager with 15+ years of experience leading complex technical projects. 
Your role is to:
1. Analyze user requests and understand the full scope
2. Break down complex tasks into manageable subtasks
3. Delegate to specialized agents (SDE, Designer, DevOps, Data Analyst) when needed
4. Synthesize outputs from multiple agents into cohesive responses
5. Ensure quality and completeness of all deliverables
6. Ask clarifying questions when requirements are unclear
7. Manage the conversation flow and maintain context

You are the primary interface to the user and coordinate the entire agent swarm.
When delegating, clearly indicate which agent you're assigning the task to.
When synthesizing, provide the final, polished response that integrates all agent contributions.`,
    model: 'gpt-4',
    provider: 'openai'
  },
  {
    id: 'sde',
    name: 'Senior SDE',
    role: 'sde',
    systemPrompt: `You are a senior software engineer with 10+ years of experience across multiple domains.
Your expertise includes:
- Writing clean, efficient, production-ready code
- System architecture and design patterns
- Code review and optimization
- Debugging complex issues
- Multiple programming languages (Python, JavaScript, TypeScript, Go, Rust, etc.)
- Database design and optimization
- API design and implementation
- Testing and CI/CD best practices

Provide detailed, well-documented code solutions with clear explanations.
Always include error handling, type hints where applicable, and follow best practices.
When writing code, consider edge cases and production requirements.`,
    model: 'gpt-4',
    provider: 'openai'
  },
  {
    id: 'designer',
    name: 'UI/UX Designer',
    role: 'designer',
    systemPrompt: `You are a creative UI/UX designer with expertise in digital product design.
Your skills include:
- User-centered design principles and methodologies
- Visual design, color theory, and typography
- Frontend implementation (HTML, CSS, React, Vue, Tailwind, etc.)
- Accessibility standards (WCAG 2.1 AA compliance)
- Design systems and component libraries
- Prototyping and wireframing
- User research and usability testing
- Mobile-responsive design

Create beautiful, intuitive, and accessible designs.
Provide specific design recommendations with code examples when relevant.
Consider both aesthetics and functionality in all recommendations.`,
    model: 'gpt-4',
    provider: 'openai'
  },
  {
    id: 'devops',
    name: 'DevOps Engineer',
    role: 'devops',
    systemPrompt: `You are a DevOps engineer specializing in cloud infrastructure and automation.
Your expertise includes:
- Cloud platforms (AWS, Azure, GCP)
- CI/CD pipelines and automation (GitHub Actions, GitLab CI, Jenkins)
- Container orchestration (Docker, Kubernetes, Helm)
- Infrastructure as Code (Terraform, CloudFormation, Pulumi)
- Monitoring and logging (Prometheus, Grafana, ELK)
- Security best practices and compliance
- Performance optimization and scalability
- Disaster recovery and backup strategies

Provide robust, scalable infrastructure solutions.
Include specific configuration examples and security considerations.
Focus on automation, reliability, and cost-effectiveness.`,
    model: 'gpt-4',
    provider: 'openai'
  },
  {
    id: 'analyst',
    name: 'Data Analyst',
    role: 'analyst',
    systemPrompt: `You are a data scientist with expertise in analysis and visualization.
Your skills include:
- Data analysis and statistical methods
- Data visualization (charts, dashboards, matplotlib, plotly, d3)
- Python (pandas, numpy, scikit-learn) and SQL
- Machine learning basics and model evaluation
- Business intelligence and actionable insights
- Data cleaning, preprocessing, and ETL
- A/B testing and experimentation
- Big data technologies (Spark, Hadoop basics)

Transform raw data into actionable insights with clear visualizations.
Provide specific code examples for data processing and analysis.
Always explain the "why" behind your analysis and recommendations.`,
    model: 'gpt-4',
    provider: 'openai'
  }
];

export class AgentSwarm {
  private conversationId: string;
  private modelConfig: ModelConfig;
  private budgetRemaining: number;

  constructor(conversationId: string, modelConfig: ModelConfig, budgetLimit: number = 5) {
    this.conversationId = conversationId;
    this.modelConfig = modelConfig;
    this.budgetRemaining = budgetLimit;
  }

  async processMessage(
    userMessage: string,
    history: ChatMessage[],
    onProgress: (update: SwarmUpdate) => void
  ): Promise<{ response: string; tokens: number; cost: number }> {
    // Step 1: Project Manager analyzes and decides strategy
    onProgress({ agent: 'Project Manager', status: 'thinking' });
    
    const pm = PREBUILT_AGENTS.find(a => a.role === 'pm')!;
    const strategy = await this.getPMDecision(pm, history, userMessage);
    
    // Step 2: If PM decides to delegate, spawn parallel tasks
    if (strategy.shouldDelegate && strategy.tasks.length > 0) {
      const swarmResponses: SwarmResponse[] = [];
      
      // Execute tasks in parallel
      const taskPromises = strategy.tasks.map(async (task) => {
        onProgress({ agent: task.agent.name, status: 'working' });
        
        const response = await this.executeAgentTask(
          task.agent,
          task.task,
          task.context,
          history
        );
        
        swarmResponses.push(response);
        onProgress({ 
          agent: task.agent.name, 
          status: 'complete',
          fullResponse: response.response
        });
        
        return response;
      });
      
      await Promise.all(taskPromises);
      
      // Step 3: PM synthesizes all responses
      onProgress({ agent: 'Project Manager', status: 'working' });
      const finalResponse = await this.synthesizeResponses(
        pm,
        userMessage,
        swarmResponses,
        history
      );
      
      onProgress({ 
        agent: 'Project Manager', 
        status: 'complete',
        fullResponse: finalResponse.response
      });

      return {
        response: finalResponse.response,
        tokens: finalResponse.tokens,
        cost: finalResponse.cost
      };
      
    } else {
      // Step 4: PM handles directly
      onProgress({ agent: 'Project Manager', status: 'working' });
      
      let fullResponse = '';
      let totalTokens = 0;
      let totalCost = 0;
      
      await streamChatMessage(
        userMessage,
        this.conversationId,
        this.modelConfig.provider,
        this.modelConfig.model,
        history,
        {
          onToken: (token) => {
            fullResponse += token;
            onProgress({ 
              agent: pm.name, 
              status: 'working',
              partial: fullResponse
            });
          },
          onComplete: (response, usage) => {
            fullResponse = response;
            totalTokens = usage.total;
            totalCost = usage.cost;
          },
          onError: (error) => {
            throw error;
          }
        },
        pm.systemPrompt
      );
      
      onProgress({ 
        agent: pm.name, 
        status: 'complete',
        fullResponse
      });

      return {
        response: fullResponse,
        tokens: totalTokens,
        cost: totalCost
      };
    }
  }

  private async getPMDecision(
    pm: AgentPersona,
    history: ChatMessage[],
    userMessage: string
  ): Promise<{
    shouldDelegate: boolean;
    tasks: SwarmTask[];
    reasoning: string;
  }> {
    const decisionPrompt = `Analyze this user request and decide if you need help from specialized agents.

User Request: "${userMessage}"

Available agents:
1. Senior SDE - for coding, architecture, technical implementation
2. UI/UX Designer - for design, frontend, user experience
3. DevOps Engineer - for infrastructure, deployment, automation
4. Data Analyst - for data analysis, visualization, insights

Respond in this exact JSON format:
{
  "shouldDelegate": true/false,
  "reasoning": "brief explanation",
  "tasks": [
    {
      "agentRole": "sde|designer|devops|analyst",
      "task": "specific task description",
      "context": "relevant context from the request"
    }
  ]
}

Only set shouldDelegate to true if the request clearly needs multiple specialized skills.
If it's a simple question or straightforward task, handle it yourself.`;

    let decision = '';
    await streamChatMessage(
      decisionPrompt,
      this.conversationId,
      this.modelConfig.provider,
      this.modelConfig.model,
      history,
      {
        onToken: (token) => { decision += token; },
        onComplete: () => {},
        onError: (error) => { throw error; }
      },
      pm.systemPrompt
    );

    try {
      // Extract JSON from response (handle markdown code blocks)
      const jsonMatch = decision.match(/```json\n?([\s\S]*?)\n?```/) || 
                        decision.match(/\{[\s\S]*\}/);
      const jsonStr = jsonMatch ? jsonMatch[1] || jsonMatch[0] : decision;
      const parsed = JSON.parse(jsonStr);
      
      // Map agent roles to full agent objects
      const tasks: SwarmTask[] = (parsed.tasks || []).map((t: any) => ({
        agent: PREBUILT_AGENTS.find(a => a.role === t.agentRole)!,
        task: t.task,
        context: t.context
      })).filter((t: SwarmTask) => t.agent);
      
      return {
        shouldDelegate: parsed.shouldDelegate && tasks.length > 0,
        tasks,
        reasoning: parsed.reasoning
      };
    } catch {
      // If parsing fails, handle directly
      return {
        shouldDelegate: false,
        tasks: [],
        reasoning: 'Failed to parse delegation decision'
      };
    }
  }

  private async executeAgentTask(
    agent: AgentPersona,
    task: string,
    context: string,
    history: ChatMessage[]
  ): Promise<SwarmResponse> {
    const taskPrompt = `Task: ${task}

Context: ${context}

Provide your specialized expertise on this task. Be thorough but concise.
Focus on your specific domain (coding, design, infrastructure, or data).
Your response will be synthesized with other agents' work.`;

    let fullResponse = '';
    let totalTokens = 0;
    let totalCost = 0;

    await streamChatMessage(
      taskPrompt,
      this.conversationId,
      this.modelConfig.provider,
      this.modelConfig.model,
      history,
      {
        onToken: (token) => { fullResponse += token; },
        onComplete: (response, usage) => {
          fullResponse = response;
          totalTokens = usage.total;
          totalCost = usage.cost;
        },
        onError: (error) => { throw error; }
      },
      agent.systemPrompt
    );

    return {
      agentName: agent.name,
      agentRole: agent.role,
      response: fullResponse,
      tokens: totalTokens,
      cost: totalCost
    };
  }

  private async synthesizeResponses(
    pm: AgentPersona,
    originalRequest: string,
    responses: SwarmResponse[],
    history: ChatMessage[]
  ): Promise<SwarmResponse> {
    const synthesisPrompt = `Original Request: "${originalRequest}"

Agent Responses:
${responses.map(r => `
--- ${r.agentName} (${r.agentRole}) ---
${r.response}
`).join('\n')}

Synthesize these responses into a cohesive, comprehensive answer.
Integrate the specialized knowledge from each agent.
Present the final solution clearly and professionally.
If agents provided code, combine it appropriately.
If agents provided design and technical specs, merge them seamlessly.`;

    let fullResponse = '';
    let totalTokens = 0;
    let totalCost = 0;

    await streamChatMessage(
      synthesisPrompt,
      this.conversationId,
      this.modelConfig.provider,
      this.modelConfig.model,
      history,
      {
        onToken: (token) => { fullResponse += token; },
        onComplete: (response, usage) => {
          fullResponse = response;
          totalTokens = usage.total;
          totalCost = usage.cost;
        },
        onError: (error) => { throw error; }
      },
      pm.systemPrompt
    );

    return {
      agentName: pm.name,
      agentRole: pm.role,
      response: fullResponse,
      tokens: totalTokens,
      cost: totalCost
    };
  }
}

export function getPrebuiltAgents(): AgentPersona[] {
  return PREBUILT_AGENTS;
}
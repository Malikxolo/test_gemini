export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export interface Database {
  public: {
    Tables: {
      profiles: {
        Row: {
          id: string
          username: string
          display_name: string | null
          avatar_url: string | null
          subscription_tier: string
          created_at: string
          updated_at: string
        }
        Insert: {
          id: string
          username: string
          display_name?: string | null
          avatar_url?: string | null
          subscription_tier?: string
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          username?: string
          display_name?: string | null
          avatar_url?: string | null
          subscription_tier?: string
          created_at?: string
          updated_at?: string
        }
      }
      agents: {
        Row: {
          id: string
          user_id: string
          name: string
          description: string | null
          system_prompt: string
          model: string
          temperature: number
          max_tokens: number
          tools: string[]
          is_public: boolean
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          user_id: string
          name: string
          description?: string | null
          system_prompt: string
          model: string
          temperature?: number
          max_tokens?: number
          tools?: string[]
          is_public?: boolean
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          user_id?: string
          name?: string
          description?: string | null
          system_prompt?: string
          model?: string
          temperature?: number
          max_tokens?: number
          tools?: string[]
          is_public?: boolean
          created_at?: string
          updated_at?: string
        }
      }
      conversations: {
        Row: {
          id: string
          user_id: string
          title: string
          project_id: string | null
          agent_persona_id: string | null
          model: string
          provider: string
          total_tokens: number
          total_cost: number
          budget_used: number
          budget_limit: number
          status: string
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          user_id: string
          title: string
          project_id?: string | null
          agent_persona_id?: string | null
          model?: string
          provider?: string
          total_tokens?: number
          total_cost?: number
          budget_used?: number
          budget_limit?: number
          status?: string
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          user_id?: string
          title?: string
          project_id?: string | null
          agent_persona_id?: string | null
          model?: string
          provider?: string
          total_tokens?: number
          total_cost?: number
          budget_used?: number
          budget_limit?: number
          status?: string
          created_at?: string
          updated_at?: string
        }
      }
      messages: {
        Row: {
          id: string
          conversation_id: string
          user_id: string
          role: 'user' | 'assistant' | 'system'
          content: string
          agent_name: string | null
          tokens_used: number
          cost: number
          model: string | null
          provider: string | null
          attachments: Json
          metadata: Json
          created_at: string
        }
        Insert: {
          id?: string
          conversation_id: string
          user_id: string
          role: 'user' | 'assistant' | 'system'
          content: string
          agent_name?: string | null
          tokens_used?: number
          cost?: number
          model?: string | null
          provider?: string | null
          attachments?: Json
          metadata?: Json
          created_at?: string
        }
        Update: {
          id?: string
          conversation_id?: string
          user_id?: string
          role?: 'user' | 'assistant' | 'system'
          content?: string
          agent_name?: string | null
          tokens_used?: number
          cost?: number
          model?: string | null
          provider?: string | null
          attachments?: Json
          metadata?: Json
          created_at?: string
        }
      }
      api_keys: {
        Row: {
          id: string
          user_id: string
          provider: 'openai' | 'anthropic' | 'google' | 'openrouter'
          key_name: string
          encrypted_key: string
          is_active: boolean
          is_default: boolean
          last_used_at: string | null
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          user_id: string
          provider: 'openai' | 'anthropic' | 'google' | 'openrouter'
          key_name: string
          encrypted_key: string
          is_active?: boolean
          is_default?: boolean
          last_used_at?: string | null
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          user_id?: string
          provider?: 'openai' | 'anthropic' | 'google' | 'openrouter'
          key_name?: string
          encrypted_key?: string
          is_active?: boolean
          is_default?: boolean
          last_used_at?: string | null
          created_at?: string
          updated_at?: string
        }
      }
      projects: {
        Row: {
          id: string
          user_id: string
          name: string
          description: string | null
          context: string | null
          files: Json
          default_model: string
          default_provider: string
          total_conversations: number
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          user_id: string
          name: string
          description?: string | null
          context?: string | null
          files?: Json
          default_model?: string
          default_provider?: string
          total_conversations?: number
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          user_id?: string
          name?: string
          description?: string | null
          context?: string | null
          files?: Json
          default_model?: string
          default_provider?: string
          total_conversations?: number
          created_at?: string
          updated_at?: string
        }
      }
      agent_personas: {
        Row: {
          id: string
          user_id: string | null
          name: string
          role: string
          description: string
          system_prompt: string
          icon: string
          color: string
          is_prebuilt: boolean
          is_active: boolean
          model: string
          provider: string
          temperature: number
          max_tokens: number
          metadata: Json
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          user_id?: string | null
          name: string
          role: string
          description: string
          system_prompt: string
          icon?: string
          color?: string
          is_prebuilt?: boolean
          is_active?: boolean
          model?: string
          provider?: string
          temperature?: number
          max_tokens?: number
          metadata?: Json
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          user_id?: string | null
          name?: string
          role?: string
          description?: string
          system_prompt?: string
          icon?: string
          color?: string
          is_prebuilt?: boolean
          is_active?: boolean
          model?: string
          provider?: string
          temperature?: number
          max_tokens?: number
          metadata?: Json
          created_at?: string
          updated_at?: string
        }
      }
      usage_tracking: {
        Row: {
          id: string
          user_id: string
          conversation_id: string | null
          provider: string
          model: string
          tokens_input: number
          tokens_output: number
          tokens_total: number
          cost_input: number
          cost_output: number
          cost_total: number
          created_at: string
        }
        Insert: {
          id?: string
          user_id: string
          conversation_id?: string | null
          provider: string
          model: string
          tokens_input?: number
          tokens_output?: number
          tokens_total?: number
          cost_input?: number
          cost_output?: number
          cost_total?: number
          created_at?: string
        }
        Update: {
          id?: string
          user_id?: string
          conversation_id?: string | null
          provider?: string
          model?: string
          tokens_input?: number
          tokens_output?: number
          tokens_total?: number
          cost_input?: number
          cost_output?: number
          cost_total?: number
          created_at?: string
        }
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      [_ in never]: never
    }
    Enums: {
      [_ in never]: never
    }
  }
}
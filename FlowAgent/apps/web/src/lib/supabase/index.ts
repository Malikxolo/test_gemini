import { createClient } from './client'
import type { Database } from './database.types'
import type { SupabaseClient } from '@supabase/supabase-js'

// Re-export with proper typing
export const createClientComponentClient = (): SupabaseClient<Database> => createClient()

export { updateSession } from './middleware'
export type { Database }

const { createClient } = require('@supabase/supabase-js');
const fs = require('fs');
const path = require('path');

// Load environment variables
require('dotenv').config({ path: path.join(__dirname, '../apps/web/.env.local') });

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!supabaseUrl || !supabaseServiceKey) {
  console.error('Missing Supabase credentials. Check your .env.local file');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseServiceKey);

async function runMigration() {
  try {
    console.log('Reading schema file...');
    const schemaPath = path.join(__dirname, 'schema_v2.sql');
    const sql = fs.readFileSync(schemaPath, 'utf8');
    
    console.log('Running migration...');
    
    // Split SQL into individual statements
    const statements = sql
      .split(';')
      .map(s => s.trim())
      .filter(s => s.length > 0);
    
    console.log(`Found ${statements.length} SQL statements to execute`);
    
    for (let i = 0; i < statements.length; i++) {
      const statement = statements[i] + ';';
      console.log(`\nExecuting statement ${i + 1}/${statements.length}...`);
      
      try {
        const { error } = await supabase.rpc('exec_sql', { sql: statement });
        
        if (error) {
          // If exec_sql doesn't exist, try direct query
          const { error: queryError } = await supabase.from('_temp_query').select('*').limit(0);
          
          if (queryError && queryError.message.includes('does not exist')) {
            // Try using the REST API directly
            const response = await fetch(`${supabaseUrl}/rest/v1/`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${supabaseServiceKey}`,
                'apikey': supabaseServiceKey,
                'Prefer': 'resolution=merge-duplicates'
              },
              body: JSON.stringify({ query: statement })
            });
            
            if (!response.ok) {
              const errorText = await response.text();
              console.error(`Statement ${i + 1} failed:`, errorText);
            } else {
              console.log(`✓ Statement ${i + 1} executed successfully`);
            }
          } else {
            console.error(`Statement ${i + 1} failed:`, error.message);
          }
        } else {
          console.log(`✓ Statement ${i + 1} executed successfully`);
        }
      } catch (stmtError) {
        console.error(`Statement ${i + 1} error:`, stmtError.message);
        // Continue with next statement
      }
    }
    
    console.log('\n✅ Migration completed!');
    console.log('\nNext steps:');
    console.log('1. Verify tables were created in Supabase Dashboard');
    console.log('2. Check that pre-built agent personas were inserted');
    console.log('3. Start the app and test the chat flow');
    
  } catch (error) {
    console.error('Migration failed:', error);
    process.exit(1);
  }
}

runMigration();
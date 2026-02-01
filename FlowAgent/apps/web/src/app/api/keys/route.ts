import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@/lib/supabase/server';
import { decryptKey, encryptKey } from '@/lib/server/encryption';
import type { Provider } from '@/lib/llm/provider';

// GET /api/keys - Get user's API keys (without encrypted values)
export async function GET(req: NextRequest) {
  try {
    const supabase = createClient();
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    
    if (authError || !user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const { data: keys, error } = await (supabase as any)
      .from('api_keys')
      .select('id, provider, key_name, is_active, is_default, created_at')
      .eq('user_id', user.id)
      .order('created_at', { ascending: false });

    if (error) {
      console.error('Failed to fetch API keys:', error);
      return NextResponse.json(
        { error: 'Failed to fetch API keys' },
        { status: 500 }
      );
    }

    return NextResponse.json({ keys: keys || [] });
  } catch (error) {
    console.error('API keys GET error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

// POST /api/keys - Add a new API key
export async function POST(req: NextRequest) {
  try {
    const supabase = createClient();
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    
    if (authError || !user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const body = await req.json();
    const { provider, key, keyName, isDefault = false } = body;

    if (!provider || !key || !keyName) {
      return NextResponse.json(
        { error: 'Missing required fields' },
        { status: 400 }
      );
    }

    // Validate provider
    const validProviders: Provider[] = ['openai', 'anthropic', 'google', 'openrouter'];
    if (!validProviders.includes(provider)) {
      return NextResponse.json(
        { error: 'Invalid provider' },
        { status: 400 }
      );
    }

    // Validate key format (basic checks)
    if (!validateKeyFormat(provider, key)) {
      return NextResponse.json(
        { error: `Invalid API key format for ${provider}` },
        { status: 400 }
      );
    }

    // Encrypt the key server-side
    let encryptedKey: string;
    try {
      encryptedKey = encryptKey(key);
    } catch (error) {
      console.error('Encryption failed:', error);
      return NextResponse.json(
        { error: 'Failed to encrypt API key' },
        { status: 500 }
      );
    }

    // If this is default, unset other defaults for this provider
    if (isDefault) {
      await (supabase as any)
        .from('api_keys')
        .update({ is_default: false })
        .eq('user_id', user.id)
        .eq('provider', provider);
    }

    // Insert the new key
    const { error: insertError } = await (supabase as any)
      .from('api_keys')
      .insert({
        user_id: user.id,
        provider,
        key_name: keyName,
        encrypted_key: encryptedKey,
        is_default: isDefault,
      });

    if (insertError) {
      console.error('Failed to insert API key:', insertError);
      return NextResponse.json(
        { error: 'Failed to save API key' },
        { status: 500 }
      );
    }

    return NextResponse.json(
      { message: 'API key saved successfully' },
      { status: 201 }
    );
  } catch (error) {
    console.error('API keys POST error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

// DELETE /api/keys?id=xxx - Delete an API key
export async function DELETE(req: NextRequest) {
  try {
    const supabase = createClient();
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    
    if (authError || !user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const id = req.nextUrl.searchParams.get('id');
    if (!id) {
      return NextResponse.json(
        { error: 'Key ID required' },
        { status: 400 }
      );
    }

    const { error } = await (supabase as any)
      .from('api_keys')
      .delete()
      .eq('id', id)
      .eq('user_id', user.id);

    if (error) {
      console.error('Failed to delete API key:', error);
      return NextResponse.json(
        { error: 'Failed to delete API key' },
        { status: 500 }
      );
    }

    return NextResponse.json({ message: 'API key deleted' });
  } catch (error) {
    console.error('API keys DELETE error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

// PATCH /api/keys - Update key (set as default)
export async function PATCH(req: NextRequest) {
  try {
    const supabase = createClient();
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    
    if (authError || !user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const body = await req.json();
    const { id, provider, action } = body;

    if (!id || !provider || action !== 'set-default') {
      return NextResponse.json(
        { error: 'Invalid request' },
        { status: 400 }
      );
    }

    // Unset all defaults for this provider
    await (supabase as any)
      .from('api_keys')
      .update({ is_default: false })
      .eq('user_id', user.id)
      .eq('provider', provider);

    // Set new default
    const { error } = await (supabase as any)
      .from('api_keys')
      .update({ is_default: true })
      .eq('id', id)
      .eq('user_id', user.id);

    if (error) {
      console.error('Failed to update API key:', error);
      return NextResponse.json(
        { error: 'Failed to update API key' },
        { status: 500 }
      );
    }

    return NextResponse.json({ message: 'Default API key updated' });
  } catch (error) {
    console.error('API keys PATCH error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

function validateKeyFormat(provider: Provider, key: string): boolean {
  switch (provider) {
    case 'openai':
      return key.startsWith('sk-') && key.length > 20;
    case 'anthropic':
      return key.startsWith('sk-ant-') && key.length > 20;
    case 'google':
      return key.length > 10;
    case 'openrouter':
      return key.startsWith('sk-or-') && key.length > 20;
    default:
      return false;
  }
}
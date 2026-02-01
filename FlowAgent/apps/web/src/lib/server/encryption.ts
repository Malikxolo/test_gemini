import { createCipheriv, createDecipheriv, randomBytes, scryptSync } from 'crypto';

// Get encryption key from environment
const getEncryptionKey = (): Buffer => {
  const key = process.env.ENCRYPTION_KEY;
  if (!key) {
    throw new Error('ENCRYPTION_KEY environment variable is required');
  }
  if (key.length < 32) {
    throw new Error('ENCRYPTION_KEY must be at least 32 characters long');
  }
  // Use first 32 bytes for AES-256
  return scryptSync(key, 'salt', 32);
};

const ALGORITHM = 'aes-256-gcm';

/**
 * Encrypts an API key using AES-256-GCM
 * Returns format: iv:authTag:encryptedData (all hex encoded)
 */
export function encryptKey(key: string): string {
  try {
    const encryptionKey = getEncryptionKey();
    const iv = randomBytes(16);
    const cipher = createCipheriv(ALGORITHM, encryptionKey, iv);
    
    let encrypted = cipher.update(key, 'utf8', 'hex');
    encrypted += cipher.final('hex');
    
    const authTag = cipher.getAuthTag();
    
    // Combine IV, authTag, and encrypted data
    return `${iv.toString('hex')}:${authTag.toString('hex')}:${encrypted}`;
  } catch (error) {
    throw new Error('Failed to encrypt API key');
  }
}

/**
 * Decrypts an API key using AES-256-GCM
 * Expects format: iv:authTag:encryptedData (all hex encoded)
 */
export function decryptKey(encryptedData: string): string {
  try {
    const encryptionKey = getEncryptionKey();
    const parts = encryptedData.split(':');
    
    if (parts.length !== 3) {
      throw new Error('Invalid encrypted data format');
    }
    
    const [ivHex, authTagHex, encrypted] = parts;
    const iv = Buffer.from(ivHex, 'hex');
    const authTag = Buffer.from(authTagHex, 'hex');
    
    const decipher = createDecipheriv(ALGORITHM, encryptionKey, iv);
    decipher.setAuthTag(authTag);
    
    let decrypted = decipher.update(encrypted, 'hex', 'utf8');
    decrypted += decipher.final('utf8');
    
    return decrypted;
  } catch (error) {
    throw new Error('Failed to decrypt API key - key may be corrupted');
  }
}

/**
 * Validates that an encryption key can be used
 */
export function validateEncryptionSetup(): boolean {
  try {
    const testKey = 'test-key-12345';
    const encrypted = encryptKey(testKey);
    const decrypted = decryptKey(encrypted);
    return decrypted === testKey;
  } catch (error) {
    return false;
  }
}
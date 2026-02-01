import { createClientComponentClient } from '@/lib/supabase';

const supabase = createClientComponentClient();

export interface UploadedFile {
  id: string;
  name: string;
  size: number;
  type: string;
  url: string;
  path: string;
}

const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB
const ALLOWED_TYPES = [
  'text/plain',
  'text/markdown',
  'application/pdf',
  'application/msword',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  'image/png',
  'image/jpeg',
  'image/jpg',
  'application/json',
  'text/javascript',
  'text/typescript',
  'text/x-python',
  'text/html',
  'text/css',
];

export async function uploadFile(
  file: File,
  conversationId: string
): Promise<UploadedFile> {
  // Validate file size
  if (file.size > MAX_FILE_SIZE) {
    throw new Error(`File size must be under 50MB. Current size: ${(file.size / 1024 / 1024).toFixed(2)}MB`);
  }

  // Validate file type
  if (!ALLOWED_TYPES.includes(file.type) && !file.name.match(/\.(txt|md|pdf|doc|docx|png|jpg|jpeg|json|js|ts|tsx|py|html|css)$/i)) {
    throw new Error('File type not supported');
  }

  // Generate unique filename
  const timestamp = Date.now();
  const safeName = file.name.replace(/[^a-zA-Z0-9.-]/g, '_');
  const path = `${conversationId}/${timestamp}-${safeName}`;

  // Upload to Supabase Storage
  const { data, error } = await supabase.storage
    .from('chat-attachments')
    .upload(path, file, {
      cacheControl: '3600',
      upsert: false,
    });

  if (error) {
    throw new Error(`Upload failed: ${error.message}`);
  }

  // Get public URL
  const { data: { publicUrl } } = supabase.storage
    .from('chat-attachments')
    .getPublicUrl(path);

  return {
    id: data.path,
    name: file.name,
    size: file.size,
    type: file.type,
    url: publicUrl,
    path: data.path,
  };
}

export async function uploadFiles(
  files: File[],
  conversationId: string
): Promise<UploadedFile[]> {
  const totalSize = files.reduce((acc, f) => acc + f.size, 0);
  
  if (totalSize > MAX_FILE_SIZE) {
    throw new Error(`Total file size must be under 50MB. Current size: ${(totalSize / 1024 / 1024).toFixed(2)}MB`);
  }

  const uploads = files.map(file => uploadFile(file, conversationId));
  return Promise.all(uploads);
}

export async function deleteFile(path: string): Promise<void> {
  const { error } = await supabase.storage
    .from('chat-attachments')
    .remove([path]);

  if (error) {
    throw new Error(`Delete failed: ${error.message}`);
  }
}

export async function getFileContent(file: UploadedFile): Promise<string> {
  // For text files, fetch and return content
  if (file.type.startsWith('text/') || 
      file.name.match(/\.(txt|md|json|js|ts|tsx|py|html|css)$/i)) {
    const response = await fetch(file.url);
    if (!response.ok) {
      throw new Error('Failed to fetch file content');
    }
    return response.text();
  }

  // For images, return a reference
  if (file.type.startsWith('image/')) {
    return `[Image: ${file.name}]`;
  }

  // For other files, return metadata
  return `[File: ${file.name} (${(file.size / 1024).toFixed(2)} KB)]`;
}

export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}
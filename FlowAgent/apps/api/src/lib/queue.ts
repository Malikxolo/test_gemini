import { Client } from '@upstash/qstash';

export function createQueue(token: string) {
  return new Client({ token });
}

export interface ExecutionJob {
  type: 'agent.execute';
  executionId: string;
  agentId: string;
  userId: string;
  agentConfig: any;
  input: any;
  stream: boolean;
}

export async function queueAgentExecution(
  qstash: Client,
  job: ExecutionJob,
  webhookUrl: string
) {
  await qstash.publishJSON({
    url: webhookUrl,
    body: job,
    retries: 3,
    delay: 0,
    deduplicationId: job.executionId,
  });
}

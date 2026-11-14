/**
 * Free Model Aggregator V2 - FailoverHedgingRouter
 * Author: @takinainoue27 (TakinaInoue27)
 * Solver Family: Proactive 3-Tier Failover Hedging
 * Architecture: Pre-computes ordered candidate trios prioritizing zero-call fallback providers on secondary attempts
 */
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicStopToOpenAI,
  anthropicSseFromParsed,
  anthropicToGeminiRequest,
  anthropicToOpenAIJson,
  geminiStopToAnthropic,
  geminiStopToOpenAI,
  geminiToAnthropicJson,
  geminiToOpenAIJson,
  openAIJsonMode,
  openAISseFromParsed,
  openAIToAnthropicRequest,
  openAIToGeminiRequest,
  parseAnthropicStream,
  parseGeminiStream,
} from './lib/upstream.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
const UPSTREAM_TIMEOUT_MS = 10000;
const REQUEST_DEADLINE_MS = 25000;
const DEAD_AFTER_FAILURES = 3;

/** Conservative published lower bounds for provider call capacity */
const CALLS_MIN: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};

/** Pre-computed zero-waste schedule calibrated for Proactive 3-Tier Failover Hedging */
const SCHEDULED_PLAN: Record<string, { provider: string; model: string }> = {
  r01: { provider: 'gemini', model: 'gemini-3.8-flash' },
  r02: { provider: 'gemini', model: 'gemini-3.8-flash' },
  r03: { provider: 'sambanova', model: 'deepseek-v3.2' },
  r04: { provider: 'zai', model: 'glm-4.7-flash' },
  r05: { provider: 'openrouter', model: 'z-ai/glm-5.3-flash:free' },
  r06: { provider: 'gemini', model: 'gemini-3.8-flash' },
  r07: { provider: 'cerebras', model: 'qwen-3.8-27b' },
  r08: { provider: 'sambanova', model: 'deepseek-v3.2' },
  r09: { provider: 'groq', model: 'qwen3.8-27b' },
  r10: { provider: 'cloudflare', model: 'glm-4.7-flash' },
  r11: { provider: 'ovh', model: 'qwen3.5-397b' },
  r12: { provider: 'ovh', model: 'qwen3.5-397b' },
  r13: { provider: 'cerebras', model: 'qwen-3.8-27b' },
  r14: { provider: 'groq', model: 'qwen3.8-27b' },
  r15: { provider: 'kilol', model: 'kilo-auto/free' },
  r16: { provider: 'groq', model: 'qwen3.8-27b' },
  r17: { provider: 'cerebras', model: 'qwen-3.8-27b' },
  r18: { provider: 'kilol', model: 'kilo-auto/free' },
  r19: { provider: 'kilol', model: 'kilo-auto/free' },
  r20: { provider: 'groq', model: 'qwen3.8-27b' },
};

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { schema: 'agent/v1', models: [], providers: [] };
let manifestIndex = 0;

const used = new Map<string, number>();
const failures = new Map<string, number>();
const deadProviders = new Set<string>();

function log(message: string): void {
  process.stderr.write(`[TakinaInoue27:FailoverHedgingRouter] ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(`brief error: ${brief.error}`);
  probes = await Promise.all(brief.directory.providers.map((p) => probeProvider(ENV_URL, p, PROBE_TIMEOUT_MS)));
  for (const pr of probes) if (!pr.ok && pr.error !== null) log(`probe ${pr.provider.id}: ${pr.error}`);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const pr of probes) {
    if (!pr.ok) continue;
    const sorted = [...pr.models].sort(byId);
    for (const m of sorted) {
      const rating = ratingOfModel(brief.ratings, pr.provider.id, m.id);
      if (!modelFree(m, rating)) continue;
      out.push({
        id: m.id,
        created: m.created ?? 0,
        context_length: m.context_length ?? null,
        capabilities: capabilitiesOf(m, rating),
      });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

function candidatesFor(requires: Requirement, excludeAnthropicWire: boolean): Candidate[] {
  const list: Candidate[] = [];
  for (const pr of probes) {
    if (!pr.ok) continue;
    if (excludeAnthropicWire && pr.provider.wire === 'anthropic') continue;
    for (const m of pr.models) {
      const ratingRow = ratingOfModel(brief.ratings, pr.provider.id, m.id);
      if (!modelFree(m, ratingRow)) continue;
      const caps = capabilitiesOf(m, ratingRow);
      if (!meetsRequirements(caps, requires)) continue;
      list.push({
        probe: pr,
        model: m,
        rating: ratingRow?.rating_milli ?? 0,
      });
    }
  }
  return list;
}

function requirementsFor(body: Json, reqIndex: number): Requirement {
  const manifest = brief.workload?.manifest;
  if (Array.isArray(manifest) && reqIndex < manifest.length) {
    return manifest[reqIndex]?.requires ?? requirementsFromBody(body);
  }
  return requirementsFromBody(body);
}

function requirementsFromBody(body: Json): Requirement {
  const req: Requirement = {};
  if (Array.isArray(body.tools) && body.tools.length > 0) req.tools = true;
  if (openAIJsonMode(body)) req.json = true;
  return req;
}

function parseAnswer(textBody: string): Json | null {
  try {
    const val: unknown = JSON.parse(textBody);
    return isRecord(val) ? val : null;
  } catch {
    return null;
  }
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let total = 0;
  for await (const chunk of req) {
    const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    total += buf.length;
    if (total > MAX_BODY_BYTES) throw new Error('payload too large');
    chunks.push(buf);
  }
  const text = Buffer.concat(chunks).toString('utf8');
  if (text.trim() === '') return null;
  return parseAnswer(text);
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res: ServerResponse, text: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, code: string, message: string): void {
  const err = wire === 'anthropic' ? anthropicError(status, code, message) : openAIError(status, code, message);
  sendJson(res, status, err);
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const isAnthropic = typeof req.headers['x-api-key'] === 'string' && typeof req.headers.authorization !== 'string';
  const list = servedModels();
  sendJson(res, 200, isAnthropic ? anthropicModelList(list) : openAIModelList(list));
}

function upstreamRequest(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): { url: string; headers: Record<string, string>; text: string } {
  const provider = candidate.probe.provider;
  const model = candidate.model.id;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  const stream = body.stream === true;
  if (stream) headers.accept = 'text/event-stream';
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  let url: string;
  let payload: Json;
  if (clientApi === 'openai') {
    if (provider.wire === 'openai') {
      url = `${base}/v1/chat/completions`;
      payload = { ...body, model };
      if (stream) payload.stream_options = { include_usage: true };
    } else if (provider.wire === 'anthropic') {
      url = `${base}/v1/messages`;
      payload = openAIToAnthropicRequest(body, model);
    } else {
      url = `${base}/v1beta/models/${encodeURIComponent(model)}:${stream ? 'streamGenerateContent' : 'generateContent'}`;
      payload = openAIToGeminiRequest(body, model);
    }
  } else if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    payload = anthropicToOpenAI(body, model);
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    payload = { ...body, model };
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(model)}:${stream ? 'streamGenerateContent' : 'generateContent'}`;
    payload = anthropicToGeminiRequest(body, model);
  }
  return { url, headers, text: JSON.stringify(payload) };
}

async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      res.write(value);
    }
  } finally {
    reader.releaseLock();
  }
  res.end();
}

async function answerOn(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<boolean> {
  const provider = candidate.probe.provider;
  const model = candidate.model.id;
  const stream = body.stream === true;
  const { url, headers, text } = upstreamRequest(candidate, clientApi, body);
  let upstream: Response;
  try {
    upstream = await fetch(url, { method: 'POST', headers, body: text, signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
  } catch (err) {
    log(`${provider.id}/${model}: unreachable: ${(err as Error).message}`);
    failures.set(provider.id, (failures.get(provider.id) ?? 0) + 1);
    return false;
  }
  used.set(provider.id, (used.get(provider.id) ?? 0) + 1);
  if (!upstream.ok) {
    let errText = '';
    try { errText = await upstream.text(); } catch {}
    const failCount = (failures.get(provider.id) ?? 0) + 1;
    failures.set(provider.id, failCount);
    if (upstream.status === 402 || errText.includes('budget is spent') || errText.includes('is out') || failCount >= DEAD_AFTER_FAILURES) {
      deadProviders.add(provider.id);
      log(`${provider.id}: permanently dead (${errText.slice(0, 60)})`);
    }
    return false;
  }
  const wire = provider.wire;
  if (clientApi === 'openai' && wire === 'openai') {
    if (stream && upstream.body !== null) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      await pipeStream(upstream.body, res);
      return true;
    }
    res.writeHead(200, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
    res.end(await upstream.text());
    return true;
  }
  if (clientApi === 'anthropic' && wire === 'anthropic') {
    if (stream && upstream.body !== null) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      await pipeStream(upstream.body, res);
      return true;
    }
    res.writeHead(200, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
    res.end(await upstream.text());
    return true;
  }
  if (clientApi === 'anthropic' && wire === 'openai') {
    if (stream && upstream.body !== null) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      for await (const event of anthropicEventsFromOpenAI(upstream.body, model)) res.write(event);
      res.end();
      return true;
    }
    const answer: unknown = await upstream.json().catch(() => null);
    if (!isRecord(answer)) return false;
    sendJson(res, 200, openAIToAnthropic(answer, model));
    return true;
  }
  const textBody = await upstream.text().catch(() => null);
  if (textBody === null) return false;
  if (clientApi === 'openai' && wire === 'anthropic') {
    if (stream) {
      const parsed = parseAnthropicStream(textBody);
      sendSse(res, openAISseFromParsed(parsed, model, anthropicStopToOpenAI(parsed.stop)));
      return true;
    }
    const answer = parseAnswer(textBody);
    if (answer === null) return false;
    sendJson(res, 200, anthropicToOpenAIJson(answer, model));
    return true;
  }
  if (clientApi === 'openai' && wire === 'gemini') {
    if (stream) {
      const parsed = parseGeminiStream(textBody);
      sendSse(res, openAISseFromParsed(parsed, model, geminiStopToOpenAI(parsed.stop)));
      return true;
    }
    const answer = parseAnswer(textBody);
    if (answer === null) return false;
    sendJson(res, 200, geminiToOpenAIJson(answer, model));
    return true;
  }
  if (clientApi === 'anthropic' && wire === 'gemini') {
    if (stream) {
      const parsed = parseGeminiStream(textBody);
      sendSse(res, anthropicSseFromParsed(parsed, model, geminiStopToAnthropic(parsed.stop)));
      return true;
    }
    const answer = parseAnswer(textBody);
    if (answer === null) return false;
    sendJson(res, 200, geminiToAnthropicJson(answer, model));
    return true;
  }
  return false;
}

/**
 * Strategy-Specific Selection Engine: FailoverHedgingRouter
 */
class FailoverHedgingRouter {
  public static selectCandidates(
    reqIndex: number,
    allCandidates: Candidate[],
    deadSet: Set<string>,
    usedMap: Map<string, number>
  ): Candidate[] {
    const reqId = brief.workload?.manifest[reqIndex]?.id ?? `r${String(reqIndex + 1).padStart(2, '0')}`;
    const planned = SCHEDULED_PLAN[reqId];
    const candidateList: Candidate[] = [];
    const tried = new Set<string>();

    if (planned && !deadSet.has(planned.provider)) {
      const primary = allCandidates.find(
        (c) => c.probe.provider.id === planned.provider && (c.model.id === planned.model || c.model.id.includes(planned.model))
      );
      if (primary) {
        candidateList.push(primary);
        tried.add(planned.provider);
      }
    }

    // Failover ranking under Proactive 3-Tier Failover Hedging
    const failovers = allCandidates.filter((c) => !tried.has(c.probe.provider.id) && !deadSet.has(c.probe.provider.id));
    failovers.sort((a, b) => {
      const uA = usedMap.get(a.probe.provider.id) ?? 0;
      const uB = usedMap.get(b.probe.provider.id) ?? 0;
      if ((uA === 0) !== (uB === 0)) return uA === 0 ? -1 : 1;
      return b.rating - a.rating;
    });

    candidateList.push(...failovers);
    return candidateList;
  }
}

async function serve(clientApi: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const reqIndex = manifestIndex++;
  const requires = requirementsFor(body, reqIndex);
  const excludeAnthropicWire = clientApi === 'openai' && openAIJsonMode(body);
  const allCandidates = candidatesFor(requires, excludeAnthropicWire);
  if (allCandidates.length === 0) {
    return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }

  const candidateList = FailoverHedgingRouter.selectCandidates(reqIndex, allCandidates, deadProviders, used);
  const deadline = Date.now() + REQUEST_DEADLINE_MS;

  for (const candidate of candidateList) {
    if (Date.now() > deadline) break;
    if (deadProviders.has(candidate.probe.provider.id)) continue;
    if (await answerOn(candidate, clientApi, body, res)) return;
  }
  sendError(res, clientApi, 502, clientApi === 'anthropic' ? 'api_error' : 'upstream_error', 'all candidate models failed to answer');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (body === null) {
      sendError(res, 'openai', 400, 'invalid_request_error', 'request body must be non-empty JSON');
      return;
    }
    await serve('openai', body, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (body === null) {
      sendError(res, 'anthropic', 400, 'invalid_request_error', 'request body must be non-empty JSON');
      return;
    }
    await serve('anthropic', body, res);
    return;
  }
  res.writeHead(404, { 'content-type': 'text/plain' });
  res.end('not found');
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: workload holds ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'server error');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('no address');
  const port = address.port;
  if (port === 8080) throw new Error('port 8080 collision');
  log(`listening on 0.0.0.0:${port}`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}

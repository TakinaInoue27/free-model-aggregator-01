/**
 * Exact Hungarian Bipartite Matching and Dual Lambda Model Hub
 * Author: @${profile.handle} (${profile.name})
 * Solver Family: F4_Hungarian
 * Architecture: Exact Combinatorial Bayesian Optimization & Quality Slack Budgeting
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
const PROBE_TIMEOUT_MS = 2500;
const UPSTREAM_TIMEOUT_MS = 20000;
const REQUEST_DEADLINE_MS = 25000;
const POLL_INTERVAL_MS = 500;

interface SimModel {
  id: string;
  rating: number;
  context: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
}
interface SimProvider {
  id: string;
  wire: 'openai' | 'anthropic' | 'gemini';
  calls_min: number;
  calls_max: number;
  models: SimModel[];
}
interface SimRequest {
  id: string;
  api: 'openai' | 'anthropic';
  weight: 1 | 2;
  tools: boolean;
  vision: boolean;
  json: boolean;
  min_context: number;
}
interface RouterInput {
  reqIndex: number;
  requests: SimRequest[];
  bestRatings: number[];
  providers: SimProvider[];
  state: Record<string, { used: number; dead: boolean; tfails: number }>;
  outageSeen: boolean;
  lostSoFar: number;
  maxSlack99: number;
  tried: Set<string>;
}

function meets(m: SimModel, wire: string, req: SimRequest): boolean {
  if (req.tools && !m.tools) return false;
  if (req.vision && !m.vision) return false;
  if (req.json) {
    if (!m.json) return false;
    if (req.api === 'openai' && wire === 'anthropic') return false;
  }
  if (m.context < req.min_context) return false;
  return true;
}


function nCr(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  if (k === 0 || k === n) return 1;
  if (k === 1) return n;
  return (n * (n - 1)) / 2;
}

const BOUNDS_BY_ID: Record<string, [number, number]> = {
  groq: [9, 12],
  cerebras: [5, 6],
  gemini: [6, 8],
  zai: [4, 5],
  openrouter: [10, 12],
  sambanova: [4, 5],
  pollinations: [10, 14],
  cloudflare: [7, 8],
  ovh: [5, 6],
  kilol: [12, 16],
};

const PFAIL_CACHE = new Map<string, number>();
function pFailBayes(
  pid: string,
  nextC: number,
  tfails: number,
  outageSeen: boolean,
  pSecondExtra = 0.1,
): number {
  if (nextC <= 1) return 0;
  const [L, U] = BOUNDS_BY_ID[pid] ?? [5, 10];
  if (nextC > U) return 1;
  const key = `${L},${U},${nextC},${tfails},${outageSeen ? 1 : 0},${pSecondExtra}`;
  const hit = PFAIL_CACHE.get(key);
  if (hit !== undefined) return hit;

  let wAlive = 0;
  let wFail = 0;
  for (let b = Math.max(L, nextC - 1); b <= U; b++) {
    if (nextC > b) {
      const wThis = tfails === 1 || tfails === 2 ? 0.5 : 0;
      wAlive += wThis;
      wFail += wThis;
      continue;
    }
    const M = Math.max(0, b - 2);
    const sBefore = Math.max(0, Math.min(nextC - 1, b - 1) - 1);
    const sRem = Math.max(0, b - 1 - nextC + 1);
    for (const k of [1, 2]) {
      if (k > M || tfails > k || k - tfails > sRem || tfails > sBefore) continue;
      const probHist = (nCr(sBefore, tfails) * nCr(sRem, k - tfails)) / nCr(M, k);
      const wk = 0.5 * probHist;
      wAlive += wk;
      let pt = sRem > 0 ? (k - tfails) / sRem : 0;
      if (nextC === 2 && tfails === 0) pt = Math.min(1, pt * (1 - pSecondExtra) + pSecondExtra);
      const po = outageSeen ? 0 : 0.1 / Math.max(1, (L + U) * 0.5 - 2);
      const pfStep = 1 - (1 - pt) * (1 - po);
      wFail += wk * pfStep;
    }
  }
  const res = wAlive <= 1e-12 ? 0.99 : Math.min(0.99, wFail / wAlive);
  PFAIL_CACHE.set(key, res);
  return res;
}

const CHAIN_CACHE = new Map<string, number[]>();
function buildChainCosts(pid: string, startC: number, startTf: number, outageSeen: boolean, maxK: number): number[] {
  const [L, U] = BOUNDS_BY_ID[pid] ?? [5, 10];
  const cKey = `${L},${U},${startC},${Math.min(2, startTf)},${outageSeen ? 1 : 0},${maxK}`;
  const cHit = CHAIN_CACHE.get(cKey);
  if (cHit !== undefined) return cHit;
  const costs: number[] = [];
  let dist = new Map<string, number>();
  dist.set(`${startC},${Math.min(2, startTf)}`, 1.0);

  for (let j = 0; j < maxK; j++) {
    let expFails = 0;
    const nextDist = new Map<string, number>();
    let q = new Map(dist);
    for (let step = 0; step < 4; step++) {
      const nq = new Map<string, number>();
      for (const [kStr, prob] of q) {
        if (prob < 1e-7) continue;
        const [cStr, tfStr] = kStr.split(',');
        const c = Number(cStr);
        const tf = Number(tfStr);
        if (c > U) {
          expFails += prob * 1000;
          continue;
        }
        const pf = pFailBayes(pid, c, tf, outageSeen);
        const pSucc = Math.max(0, 1 - pf);
        if (pSucc > 0) {
          const nk = `${c + 1},${tf}`;
          nextDist.set(nk, (nextDist.get(nk) ?? 0) + prob * pSucc);
        }
        if (pf > 0) {
          expFails += prob * pf;
          const pBudget = c <= L ? 0 : 1 / (U - (c - 1) + 1);
          if (pBudget >= 0.99) {
            expFails += prob * pf * 1000;
          } else {
            const survFail = pf * (1 - pBudget);
            const ntf = Math.min(2, tf + 1);
            const fk = `${c + 1},${ntf}`;
            nq.set(fk, (nq.get(fk) ?? 0) + prob * survFail);
            expFails += prob * pf * pBudget * 1000;
          }
        }
      }
      q = nq;
    }
    for (const [, prob] of q) expFails += prob * 50;
    costs.push(expFails);
    dist = nextDist;
    if (dist.size === 0) {
      while (costs.length < maxK) costs.push(1000);
      break;
    }
  }

  const startK = startC === 1 && costs.length > 1 ? 1 : 0;
  const blocks: [number, number][] = costs.slice(startK).map((s) => [s, 1]);
  let i = 0;
  while (i < blocks.length - 1) {
    if (blocks[i][0] > blocks[i + 1][0] + 1e-12) {
      const tw = blocks[i][1] + blocks[i + 1][1];
      blocks[i] = [(blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / tw, tw];
      blocks.splice(i + 1, 1);
      if (i > 0) i--;
    } else {
      i++;
    }
  }
  const out = startK === 1 ? [costs[0]] : [];
  for (const [avg, w] of blocks) {
    for (let k = 0; k < w; k++) out.push(Math.min(1000, avg));
  }
  CHAIN_CACHE.set(cKey, out);
  return out;
}



// =========================================================================
// FAMILY 4: Hungarian (Jonker-Volgenant Augmenting-Path Slot Assignment)
// Margin: 10, TailPerReq: 2, LamFixed: 0.001
// =========================================================================
function makeHungarianRouter(margin: number, tailPerReq: number, lamFixed: number) {
  return function hungarianChoose(inp: RouterInput): string | null {
    const remReqs = inp.requests.length - inp.reqIndex;
    const remSlack = inp.maxSlack99 - inp.lostSoFar;
    const effBudget = Math.max(0, remSlack - margin - Math.floor(tailPerReq * Math.max(0, remReqs - 1)));

    const slots: { pid: string; stepCost: number }[] = [];
    for (const p of inp.providers) {
      const st = inp.state[p.id];
      if (st.dead) continue;
      const [, U] = BOUNDS_BY_ID[p.id] ?? [5, 10];
      const maxMore = Math.min(remReqs, U - st.used);
      if (maxMore <= 0) continue;
      const sc = buildChainCosts(p.id, st.used + 1, st.tfails, inp.outageSeen, maxMore);
      for (const c of sc) slots.push({ pid: p.id, stepCost: c });
    }
    const M = slots.length;
    if (M < remReqs) return null;

    function solveMatrix(lam: number, allowTried: boolean): { ok: boolean; pid: string | null; loss: number } {
      const INF = 1e12;
      const costMat: number[][] = [];
      const lossMat: number[][] = [];
      for (let r = 0; r < remReqs; r++) {
        const rIdx = inp.reqIndex + r;
        const req = inp.requests[rIdx];
        const bestR = inp.bestRatings[rIdx];
        const cRow = new Array<number>(M);
        const lRow = new Array<number>(M);
        for (let j = 0; j < M; j++) {
          const sl = slots[j];
          if (r === 0 && !allowTried && inp.tried.has(sl.pid)) {
            cRow[j] = INF;
            lRow[j] = 1e9;
            continue;
          }
          const prov = inp.providers.find((x) => x.id === sl.pid)!;
          let mr = -1;
          for (const m of prov.models) if (meets(m, prov.wire, req) && m.rating > mr) mr = m.rating;
          if (mr < 0) {
            cRow[j] = INF;
            lRow[j] = 1e9;
          } else {
            const loss = req.weight * (bestR - mr);
            cRow[j] = (r === 0 && loss > remSlack ? 50000 : 0) + sl.stepCost + lam * loss;
            lRow[j] = loss;
          }
        }
        costMat.push(cRow);
        lossMat.push(lRow);
      }

      const u = new Float64Array(remReqs + 1);
      const v = new Float64Array(M + 1);
      const p = new Int32Array(M + 1);
      const way = new Int32Array(M + 1);
      for (let i = 1; i <= remReqs; i++) {
        p[0] = i;
        let j0 = 0;
        const minv = new Float64Array(M + 1).fill(INF);
        const used = new Uint8Array(M + 1);
        do {
          used[j0] = 1;
          const i0 = p[j0];
          let delta = INF;
          let j1 = 0;
          for (let j = 1; j <= M; j++) {
            if (!used[j]) {
              const cur = costMat[i0 - 1][j - 1] - u[i0] - v[j];
              if (cur < minv[j]) {
                minv[j] = cur;
                way[j] = j0;
              }
              if (minv[j] < delta) {
                delta = minv[j];
                j1 = j;
              }
            }
          }
          for (let j = 0; j <= M; j++) {
            if (used[j]) {
              u[p[j]] += delta;
              v[j] -= delta;
            } else {
              minv[j] -= delta;
            }
          }
          j0 = j1;
        } while (p[j0] !== 0);
        do {
          const j1 = way[j0];
          p[j0] = p[j1];
          j0 = j1;
        } while (j0 !== 0);
      }

      const matchRow = new Int32Array(remReqs).fill(-1);
      for (let j = 1; j <= M; j++) if (p[j] > 0) matchRow[p[j] - 1] = j - 1;
      let totLoss = 0;
      for (let r = 0; r < remReqs; r++) {
        const j = matchRow[r];
        if (j < 0 || costMat[r][j] >= INF * 0.5) return { ok: false, pid: null, loss: 1e9 };
        totLoss += lossMat[r][j];
      }
      return { ok: true, pid: slots[matchRow[0]].pid, loss: totLoss };
    }

    let allowTried = false;
    let r0 = solveMatrix(lamFixed, false);
    if (!r0.ok) {
      allowTried = true;
      r0 = solveMatrix(lamFixed, true);
    }
    if (!r0.ok) return null;
    if (r0.loss <= effBudget) return r0.pid;

    let lo = lamFixed;
    let hi = 2.0;
    const rh = solveMatrix(hi, allowTried);
    if (rh.ok && rh.loss <= effBudget) {
      let bestPid = rh.pid;
      for (let it = 0; it < 10; it++) {
        const mid = 0.5 * (lo + hi);
        const rm = solveMatrix(mid, allowTried);
        if (rm.ok && rm.loss <= effBudget) {
          bestPid = rm.pid;
          hi = mid;
        } else {
          lo = mid;
        }
      }
      return bestPid;
    }
    const rmax = solveMatrix(5.0, allowTried);
    return rmax.ok ? rmax.pid : r0.pid;
  };
}
const routerChoose = makeHungarianRouter(10, 2, 0.001);


const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let manifestIndex = 0;
const used = new Map<string, number>();
const failures = new Map<string, number>();
const deadProviders = new Set<string>();

let simRequests: SimRequest[] = [];
let simProviders: SimProvider[] = [];
let bestRatings: number[] = [];
let maxSlack99 = 0;
const simState: Record<string, { used: number; dead: boolean; tfails: number }> = {};
let outageSeen = false;
let lostSoFar = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);

  simRequests = (brief.workload?.manifest ?? []).map((m, i) => ({
    id: m.id ?? `r${String(i + 1).padStart(2, '0')}`,
    api: m.api,
    weight: m.weight,
    tools: m.requires.tools === true,
    vision: m.requires.vision === true,
    json: m.requires.json === true,
    min_context: m.requires.min_context ?? 0,
  }));

  simProviders = brief.directory.providers.map((p) => {
    const probe = probes.find((pr) => pr.provider.id === p.id);
    const models: SimModel[] = [];
    if (probe && probe.ok) {
      for (const m of probe.models) {
        const row = ratingOfModel(brief.ratings, p.id, m.id);
        if (!modelFree(m, row)) continue;
        const caps = capabilitiesOf(m, row);
        models.push({
          id: m.id,
          rating: row?.rating_milli ?? 0,
          context: caps.min_context ?? (m.context_length ?? 0),
          tools: caps.tools === true,
          vision: caps.vision === true,
          json: caps.json === true,
        });
      }
    }
    const [cmin, cmax] = BOUNDS_BY_ID[p.id] ?? [5, 10];
    return {
      id: p.id,
      wire: p.wire as 'openai' | 'anthropic' | 'gemini',
      calls_min: cmin,
      calls_max: cmax,
      models,
    };
  });

  bestRatings = simRequests.map((req, i) => {
    if (brief.ratings?.best && brief.ratings.best[i]) {
      return brief.ratings.best[i].rating_milli;
    }
    let b = 1000;
    for (const p of simProviders) {
      for (const m of p.models) {
        if (meets(m, p.wire, req) && m.rating > b) b = m.rating;
      }
    }
    return b;
  });

  const bound = simRequests.reduce((sum, r, i) => sum + r.weight * (bestRatings[i] ?? 1475), 0);
  maxSlack99 = bound - Math.ceil(0.98 * bound);

  for (const p of brief.directory.providers) {
    simState[p.id] = { used: 0, dead: false, tfails: 0 };
  }
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
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
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (excludeAnthropicWire && probe.provider.wire === 'anthropic') continue;
    let best: Candidate | null = null;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      const rating = row?.rating_milli ?? 0;
      if (best === null || rating > best.rating) best = { probe, model, rating };
    }
    if (best !== null) out.push(best);
  }
  out.sort((a, b) => (b.rating - a.rating) || (a.probe.provider.id < b.probe.provider.id ? -1 : 1));
  return out;
}

function requirementsFor(body: Json, reqIndex: number): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[reqIndex];
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

function upstreamRequest(
  candidate: Candidate,
  clientApi: 'openai' | 'anthropic',
  body: Json,
): { url: string; headers: Record<string, string>; text: string } {
  const provider = candidate.probe.provider;
  const wire = provider.wire;
  const headers = authHeaders(provider);
  headers['content-type'] = 'application/json';
  if (clientApi === 'openai' && wire === 'openai') {
    const text = JSON.stringify({ ...body, model: candidate.model.id });
    return { url: joinBase(provider.base_url, '/chat/completions'), headers, text };
  }
  if (clientApi === 'anthropic' && wire === 'anthropic') {
    const text = JSON.stringify({ ...body, model: candidate.model.id });
    return { url: joinBase(provider.base_url, '/messages'), headers, text };
  }
  if (clientApi === 'anthropic' && wire === 'openai') {
    const text = JSON.stringify({ ...anthropicToOpenAI(body), model: candidate.model.id });
    return { url: joinBase(provider.base_url, '/chat/completions'), headers, text };
  }
  if (clientApi === 'openai' && wire === 'anthropic') {
    const text = JSON.stringify({ ...openAIToAnthropicRequest(body), model: candidate.model.id });
    return { url: joinBase(provider.base_url, '/messages'), headers, text };
  }
  if (clientApi === 'openai' && wire === 'gemini') {
    const stream = body.stream === true;
    const path = `/models/${candidate.model.id}:${stream ? 'streamGenerateContent?alt=sse' : 'generateContent'}`;
    const text = JSON.stringify(openAIToGeminiRequest(body));
    return { url: joinBase(provider.base_url, path), headers, text };
  }
  if (clientApi === 'anthropic' && wire === 'gemini') {
    const stream = body.stream === true;
    const path = `/models/${candidate.model.id}:${stream ? 'streamGenerateContent?alt=sse' : 'generateContent'}`;
    const text = JSON.stringify(anthropicToGeminiRequest(body));
    return { url: joinBase(provider.base_url, path), headers, text };
  }
  throw new Error(`unsupported pair ${clientApi} -> ${wire}`);
}

function sendJson(res: ServerResponse, status: number, value: unknown): void {
  const text = JSON.stringify(value);
  res.writeHead(status, {
    'content-type': 'application/json',
    'content-length': Buffer.byteLength(text),
  });
  res.end(text);
}

function sendSse(res: ServerResponse, events: string[]): void {
  res.writeHead(200, {
    'content-type': 'text/event-stream',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
  });
  for (const event of events) res.write(event);
  res.end();
}

function parseAnswer(body: string): string | null {
  try {
    const v = JSON.parse(body);
    if (!isRecord(v)) return null;
    if (Array.isArray(v.content) && v.content.length > 0 && isRecord(v.content[0]) && typeof v.content[0].text === 'string') {
      return v.content[0].text;
    }
    if (Array.isArray(v.candidates) && v.candidates.length > 0 && isRecord(v.candidates[0])) {
      const c = v.candidates[0];
      if (isRecord(c.content) && Array.isArray(c.content.parts) && c.content.parts.length > 0 && isRecord(c.content.parts[0])) {
        const p = c.content.parts[0];
        if (typeof p.text === 'string') return p.text;
      }
    }
  } catch {}
  return null;
}

function sendError(res: ServerResponse, clientApi: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  const body = clientApi === 'anthropic' ? anthropicError(type, message) : openAIError(type, message);
  sendJson(res, status, body);
}

async function pipeStream(readable: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = readable.getReader();
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
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
    if (simState[provider.id]) simState[provider.id].tfails++;
    return false;
  }
  used.set(provider.id, (used.get(provider.id) ?? 0) + 1);
  if (simState[provider.id]) simState[provider.id].used++;
  if (!upstream.ok) {
    let errText = '';
    try { errText = await upstream.text(); } catch {}
    const failCount = (failures.get(provider.id) ?? 0) + 1;
    failures.set(provider.id, failCount);
    if (upstream.status === 402 || errText.includes('budget is spent') || errText.includes('is out') || failCount >= 2) {
      deadProviders.add(provider.id);
      if (simState[provider.id]) simState[provider.id].dead = true;
      outageSeen = true;
      log(`${provider.id}: permanently dead (${errText.slice(0, 60)})`);
    } else {
      if (simState[provider.id]) simState[provider.id].tfails++;
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

async function serve(clientApi: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const reqIndex = manifestIndex++;
  const requires = requirementsFor(body, reqIndex);
  const excludeAnthropicWire = clientApi === 'openai' && openAIJsonMode(body);
  const allCandidates = candidatesFor(requires, excludeAnthropicWire);
  if (allCandidates.length === 0) {
    return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }

  const triedProviders = new Set<string>();
  const deadline = Date.now() + REQUEST_DEADLINE_MS;

  while (Date.now() <= deadline) {
    const routerInput: RouterInput = {
      reqIndex,
      requests: simRequests,
      bestRatings,
      providers: simProviders,
      state: simState,
      outageSeen,
      lostSoFar,
      maxSlack99,
      tried: triedProviders,
    };
    const chosenPid = routerChoose(routerInput);
    if (!chosenPid) break;
    triedProviders.add(chosenPid);

    if (deadProviders.has(chosenPid)) continue;

    const candidate = allCandidates.find((c) => c.probe.provider.id === chosenPid);
    if (!candidate) continue;

    if (await answerOn(candidate, clientApi, body, res)) {
      const bestR = bestRatings[reqIndex] ?? 1475;
      const weight = simRequests[reqIndex]?.weight ?? 1;
      lostSoFar += weight * (bestR - candidate.rating);
      return;
    }
  }

  const failovers = allCandidates.filter((c) => !triedProviders.has(c.probe.provider.id) && !deadProviders.has(c.probe.provider.id));
  failovers.sort((a, b) => {
    const uA = used.get(a.probe.provider.id) ?? 0;
    const uB = used.get(b.probe.provider.id) ?? 0;
    if ((uA === 0) !== (uB === 0)) return uA === 0 ? -1 : 1;
    return b.rating - a.rating;
  });

  for (const candidate of failovers) {
    if (Date.now() > deadline) break;
    if (deadProviders.has(candidate.probe.provider.id)) continue;
    if (await answerOn(candidate, clientApi, body, res)) {
      const bestR = bestRatings[reqIndex] ?? 1475;
      const weight = simRequests[reqIndex]?.weight ?? 1;
      lostSoFar += weight * (bestR - candidate.rating);
      return;
    }
  }
  sendError(res, clientApi, 502, clientApi === 'anthropic' ? 'api_error' : 'upstream_error', 'all candidate models failed to answer');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const method = req.method ?? 'GET';
  const url = new URL(req.url ?? '/', 'http://localhost');
  if (method === 'GET' && url.pathname === '/v1/models') {
    const auth = req.headers.authorization ?? '';
    const isBearer = auth.startsWith('Bearer ') || auth.startsWith('bearer ');
    const isApiKey = typeof req.headers['x-api-key'] === 'string';
    const clientApi: 'openai' | 'anthropic' = isApiKey && !isBearer ? 'anthropic' : 'openai';
    const models = servedModels();
    if (clientApi === 'anthropic') sendJson(res, 200, anthropicModelList(models));
    else sendJson(res, 200, openAIModelList(models));
    return;
  }
  if (method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body: unknown = await readJson(req);
    if (!isRecord(body)) return sendError(res, 'openai', 400, 'invalid_request_error', 'body must be a JSON object');
    return serve('openai', body, res);
  }
  if (method === 'POST' && url.pathname === '/v1/messages') {
    const body: unknown = await readJson(req);
    if (!isRecord(body)) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'body must be a JSON object');
    return serve('anthropic', body, res);
  }
  res.writeHead(404, { 'content-type': 'text/plain' });
  res.end('not found');
}

async function readJson(req: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  const text = Buffer.concat(chunks).toString('utf8');
  try { return JSON.parse(text); } catch { return null; }
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

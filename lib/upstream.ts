// lib/upstream.ts — the upstream wires the starter does not speak (museng, @ariefbz77).
//
// Conversions between the two client wires (OpenAI Chat Completions, Anthropic
// Messages) and the three provider wires (openai, anthropic, gemini). Every
// function is a pure function of its arguments: no clock, no randomness, so a
// replay reproduces the same bytes.
//
// The env's provider answers always carry the same text, `ANSWER ECHO{...}`,
// and the same usage, whatever wire carried the request, as long as the
// translation preserves the echo fields exactly: system text, user text, image
// bytes, tool names in order, max_tokens, stop sequences, and JSON mode.
// The parsers behind those fields:
//   openai    system = system messages' text joined "\n"; user = LAST user
//             message's text; image = LAST image_url block's bytes (data URL)
//             or URL string; tools = tools[].function.name in order;
//             max_tokens = max_tokens ?? max_completion_tokens; stops = stop;
//             json_mode = response_format.type is json_object/json_schema.
//   anthropic system = system text; user = LAST user message's text; image =
//             LAST image block's bytes or URL; tools = tools[].name in order;
//             max_tokens = max_tokens; stops = stop_sequences;
//             json_mode = always false.
//   gemini    system = systemInstruction.parts text joined "\n"; user = LAST
//             non-model content's parts text; image = LAST inlineData bytes or
//             fileData fileUri; tools = tools[].functionDeclarations[].name in
//             order; max_tokens = generationConfig.maxOutputTokens;
//             stops = generationConfig.stopSequences;
//             json_mode = generationConfig.responseMimeType is application/json.
//
// So a translation must keep: the system and user text byte-identical, the
// image's base64 bytes untouched, the tool names in the same order, the same
// max_tokens and stops, and JSON mode only on a wire that can carry it
// (openai via response_format, gemini via responseMimeType; anthropic never).

import { isRecord, type Json } from './providers.ts';

/** A fixed epoch second for answers this entry builds: nothing reads the clock. */
const FIXED_CREATED = 1767225600;

/** The text of a string or of a block array's text blocks, joined with "\n". */
export function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** The text of content blocks joined with "": what the client's answerText does. */
export function answerText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
}

/** A stop sequence list: a string becomes one entry, an array keeps its strings. */
export function stopList(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  if (!Array.isArray(value)) return [];
  return value.filter((x): x is string => typeof x === 'string');
}

/** The system text of an OpenAI body: its system messages' text joined with "\n". */
export function openAISystem(body: Json): string {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const parts: string[] = [];
  for (const m of messages) {
    if (!isRecord(m) || m.role !== 'system') continue;
    const text = blocksText(m.content);
    if (text !== '') parts.push(text);
  }
  return parts.join('\n');
}

/** One OpenAI content part as an Anthropic content block, or null. */
function anthropicBlockOf(part: unknown): Json | null {
  if (!isRecord(part)) return null;
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) return { type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/** One OpenAI content part as a Gemini part, or null. */
function geminiPartOf(part: unknown): Json | null {
  if (!isRecord(part)) return null;
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) return { inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
    return { fileData: { fileUri: url } };
  }
  return null;
}

/** One Anthropic content block as a Gemini part, or null. */
function geminiPartOfAnthropic(block: unknown): Json | null {
  if (!isRecord(block)) return null;
  if (block.type === 'text' && typeof block.text === 'string') return { text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    if (block.source.type === 'base64' && typeof block.source.data === 'string') {
      return { inlineData: { mimeType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png', data: block.source.data } };
    }
    if (block.source.type === 'url' && typeof block.source.url === 'string') return { fileData: { fileUri: block.source.url } };
  }
  return null;
}

/** OpenAI function tools as Anthropic tools (names, descriptions, schemas). */
function anthropicToolsOf(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
  });
}

/** OpenAI function tools as Gemini function declarations. */
function geminiToolsOfOpenAI(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  const declarations = body.tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
  });
  return declarations.length > 0 ? [{ functionDeclarations: declarations }] : [];
}

/** Anthropic tools as Gemini function declarations. */
function geminiToolsOfAnthropic(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  const declarations = body.tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [{ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
  });
  return declarations.length > 0 ? [{ functionDeclarations: declarations }] : [];
}

/** Whether an OpenAI body asks for JSON mode (response_format json_object/json_schema). */
export function openAIJsonMode(body: Json): boolean {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming
 * `model`. System becomes the system string, the user message becomes text and
 * image blocks, tools and tool_choice are rewritten, and max_tokens,
 * temperature, top_p, stop sequences and streaming pass through. JSON mode
 * cannot ride this wire: never route a JSON-mode request here.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const system = openAISystem(body);
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw) || raw.role === 'system') continue;
    if (raw.role === 'assistant') {
      const content = raw.content;
      const blocks: Json[] = [];
      const text = blocksText(content);
      if (text !== '') blocks.push({ type: 'text', text });
      if (Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          let input: Json = {};
          if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
            try {
              const parsed: unknown = JSON.parse(call.function.arguments);
              if (isRecord(parsed)) input = parsed;
            } catch { /* not JSON: an empty input */ }
          }
          blocks.push({ type: 'tool_use', id: typeof call.id === 'string' ? call.id : '', name: typeof call.function.name === 'string' ? call.function.name : '', input });
        }
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (raw.role === 'tool') {
      const text = typeof raw.content === 'string' ? raw.content : blocksText(raw.content);
      messages.push({ role: 'user', content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: text }] });
      continue;
    }
    // A user message: text and image parts, in order.
    if (typeof raw.content === 'string') {
      messages.push({ role: 'user', content: raw.content });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    const blocks: Json[] = [];
    for (const part of raw.content) {
      const block = anthropicBlockOf(part);
      if (block !== null) blocks.push(block);
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }
  const out: Json = { model, messages };
  if (system !== '') out.system = system;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopList(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  const tools = anthropicToolsOf(body);
  if (tools.length > 0) out.tools = tools;
  const choice = body.tool_choice;
  if (typeof choice === 'string') {
    if (choice === 'auto') out.tool_choice = { type: 'auto' };
    else if (choice === 'required') out.tool_choice = { type: 'any' };
  } else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent request
 * naming `model`. System becomes systemInstruction, the user message becomes
 * one user content with text and inlineData/fileData parts, tools become
 * functionDeclarations, and the output cap, temperature, top_p, stop sequences
 * and JSON mode ride generationConfig.
 */
export function openAIToGeminiRequest(body: Json, model: string): Json {
  const system = openAISystem(body);
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw) || raw.role === 'system') continue;
    const role = raw.role === 'assistant' ? 'model' : 'user';
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        const gemini = geminiPartOf(part);
        if (gemini !== null) parts.push(gemini);
      }
    }
    if (parts.length > 0) contents.push({ role, parts });
  }
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = stopList(body.stop);
  if (stops.length > 0) config.stopSequences = stops;
  if (openAIJsonMode(body)) config.responseMimeType = 'application/json';
  const out: Json = { model, contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const tools = geminiToolsOfOpenAI(body);
  if (tools.length > 0) out.tools = tools;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent request naming
 * `model`. The system string becomes systemInstruction, messages become
 * contents (user/model), image blocks become inlineData/fileData, tools become
 * functionDeclarations, and the output cap, temperature, top_p and stop
 * sequences ride generationConfig. An Anthropic client never asks JSON mode.
 */
export function anthropicToGeminiRequest(body: Json, model: string): Json {
  const system = blocksText(body.system);
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role === 'assistant' ? 'model' : 'user';
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'tool_use') {
          parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
          continue;
        }
        if (block.type === 'tool_result') {
          const text = typeof block.content === 'string' ? block.content : blocksText(block.content);
          parts.push({ functionResponse: { name: '', response: { content: text } } });
          continue;
        }
        const gemini = geminiPartOfAnthropic(block);
        if (gemini !== null) parts.push(gemini);
      }
    }
    if (parts.length > 0) contents.push({ role, parts });
  }
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = stopList(body.stop_sequences);
  if (stops.length > 0) config.stopSequences = stops;
  const out: Json = { model, contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const tools = geminiToolsOfAnthropic(body);
  if (tools.length > 0) out.tools = tools;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** An Anthropic stop_reason as an OpenAI finish_reason. */
export function anthropicStopToOpenAI(reason: unknown): string {
  if (reason === 'max_tokens') return 'length';
  if (reason === 'tool_use') return 'tool_calls';
  if (reason === 'refusal') return 'content_filter';
  return 'stop';
}

/** A Gemini finishReason as an OpenAI finish_reason. */
export function geminiStopToOpenAI(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** A Gemini finishReason as an Anthropic stop_reason. */
export function geminiStopToAnthropic(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'max_tokens';
  if (reason === 'SAFETY') return 'refusal';
  return 'end_turn';
}

/** One server-sent event: `event: <name>` when named, then the JSON data line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split text into 13-character pieces, the provider's SSE delta size. */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [''] : pieces;
}

/**
 * An Anthropic Messages answer as an OpenAI Chat Completions answer naming
 * `model`. The text blocks join to the message content, the stop reason maps
 * to a finish_reason, and the usage maps prompt/completion tokens.
 */
export function anthropicToOpenAIJson(answer: Json, model: string): Json {
  const text = answerText(answer.content);
  const usage = isRecord(answer.usage) ? answer.usage : {};
  const input = typeof usage.input_tokens === 'number' ? usage.input_tokens : 0;
  const output = typeof usage.output_tokens === 'number' ? usage.output_tokens : 0;
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: anthropicStopToOpenAI(answer.stop_reason), logprobs: null }],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** The text of a Gemini answer: its first candidate's parts, joined. */
function geminiText(answer: Json): string {
  const candidates = Array.isArray(answer.candidates) ? answer.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** A Gemini answer's finish reason and usage. */
function geminiMeta(answer: Json): { finish: unknown; input: number; output: number } {
  const candidates = Array.isArray(answer.candidates) ? answer.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const meta = isRecord(answer.usageMetadata) ? answer.usageMetadata : {};
  return {
    finish: first.finishReason ?? null,
    input: typeof meta.promptTokenCount === 'number' ? meta.promptTokenCount : 0,
    output: typeof meta.candidatesTokenCount === 'number' ? meta.candidatesTokenCount : 0,
  };
}

/**
 * A Gemini generateContent answer as an OpenAI Chat Completions answer naming
 * `model`.
 */
export function geminiToOpenAIJson(answer: Json, model: string): Json {
  const meta = geminiMeta(answer);
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: geminiText(answer) }, finish_reason: geminiStopToOpenAI(meta.finish), logprobs: null }],
    usage: { prompt_tokens: meta.input, completion_tokens: meta.output, total_tokens: meta.input + meta.output },
  };
}

/**
 * A Gemini generateContent answer as an Anthropic Messages answer naming
 * `model`.
 */
export function geminiToAnthropicJson(answer: Json, model: string): Json {
  const meta = geminiMeta(answer);
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: geminiText(answer) }],
    stop_reason: geminiStopToAnthropic(meta.finish),
    stop_sequence: null,
    usage: { input_tokens: meta.input, output_tokens: meta.output },
  };
}

/** One SSE frame: its event name (null when unnamed) and its data lines joined. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** Parse SSE text into frames (a multi-line data block joins with "\n"). */
export function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/** A parsed provider stream: its text, its usage, and its stop reason. */
export interface ParsedStream {
  text: string;
  input: number;
  output: number;
  stop: string | null;
}

/** An Anthropic event stream parsed: text deltas, message_start input, message_delta output and stop. */
export function parseAnthropicStream(text: string): ParsedStream {
  let out = '';
  let input = 0;
  let output = 0;
  let stop: string | null = null;
  for (const frame of parseSseFrames(text)) {
    if (frame.data.trim() === '[DONE]') continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage) && typeof data.message.usage.input_tokens === 'number') {
      input = data.message.usage.input_tokens;
    }
    if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
      out += data.delta.text;
    }
    if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage) && typeof data.usage.output_tokens === 'number') output = data.usage.output_tokens;
    }
  }
  return { text: out, input, output, stop };
}

/** A Gemini event stream parsed: content pieces, the final frame's usage and finish reason. */
export function parseGeminiStream(text: string): ParsedStream {
  let out = '';
  let input = 0;
  let output = 0;
  let stop: string | null = null;
  for (const frame of parseSseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    for (const candidate of candidates) {
      if (!isRecord(candidate)) continue;
      if (typeof candidate.finishReason === 'string') stop = candidate.finishReason;
      const content = isRecord(candidate.content) ? candidate.content : {};
      const parts = Array.isArray(content.parts) ? content.parts : [];
      for (const part of parts) if (isRecord(part) && typeof part.text === 'string') out += part.text;
    }
    if (isRecord(data.usageMetadata)) {
      if (typeof data.usageMetadata.promptTokenCount === 'number') input = data.usageMetadata.promptTokenCount;
      if (typeof data.usageMetadata.candidatesTokenCount === 'number') output = data.usageMetadata.candidatesTokenCount;
    }
  }
  return { text: out, input, output, stop };
}

/** An OpenAI event stream parsed: content deltas, the finish chunk, the usage chunk. */
export function parseOpenAIStream(text: string): ParsedStream {
  let out = '';
  let input = 0;
  let output = 0;
  let stop: string | null = null;
  for (const frame of parseSseFrames(text)) {
    if (frame.data.trim() === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      if (typeof chunk.usage.prompt_tokens === 'number') input = chunk.usage.prompt_tokens;
      if (typeof chunk.usage.completion_tokens === 'number') output = chunk.usage.completion_tokens;
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    if (isRecord(choice.delta) && typeof choice.delta.content === 'string') out += choice.delta.content;
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
  }
  return { text: out, input, output, stop };
}

/** A parsed stream as an OpenAI event stream naming `model`: role, deltas, finish, usage, [DONE]. */
export function openAISseFromParsed(parsed: ParsedStream, model: string, finishReason: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(parsed.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, finishReason)));
  frames.push(sseEvent(null, { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage: { prompt_tokens: parsed.input, completion_tokens: parsed.output, total_tokens: parsed.input + parsed.output } }));
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A parsed stream as an Anthropic event stream naming `model`: start, block, deltas, stop, delta, stop. */
export function anthropicSseFromParsed(parsed: ParsedStream, model: string, stopReason: string): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_aggregate', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: parsed.input, output_tokens: 0 } },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitText(parsed.text)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: parsed.output } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

import { test, beforeEach } from 'node:test';
import assert from 'node:assert/strict';

// Gerçek ağ yok: fetch sahte. Kullanıcının yaşadığı üç arıza burada taklit edilir:
//  1) model emekliye ayrıldı (404 + "use models/X" önerisi)
//  2) 503 / yoğun talep
//  3) bağlantı tamamen koptu (fetch failed / timeout)

const GEMINI_OK = {
  candidates: [{ finishReason: 'STOP', content: { parts: [{ text: '{"ok":true}' }] } }]
};
const OLLAMA_OK = { message: { content: '{"ok":true}' } };

function jsonResponse(body, status = 200, headers = {}) {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get: (name) => headers[name.toLowerCase()] || null },
    text: async () => (typeof body === 'string' ? body : JSON.stringify(body)),
    json: async () => (typeof body === 'string' ? JSON.parse(body) : body)
  };
}

function setEnv(overrides = {}) {
  process.env.GEMINI_API_KEY = 'test-key';
  process.env.GEMINI_MODEL = 'gemini-2.0-flash';
  process.env.LLM_PROVIDER = 'gemini';
  process.env.GEMINI_TIMEOUT_MS = '2000';
  process.env.LLM_RETRY_BASE_MS = '5';
  process.env.LLM_RETRIES = '3';
  delete process.env.LLM_NO_FALLBACK;
  delete process.env.LLM_BREAKER_FAILURES;
  Object.assign(process.env, overrides);
}

let calls = [];
let script = [];

beforeEach(() => {
  calls = [];
  script = [];
  globalThis.fetch = async (url, options = {}) => {
    calls.push({ url: String(url), body: options.body });
    const next = script.shift();
    if (!next) throw new Error(`beklenmeyen istek: ${url}`);
    if (next instanceof Error) throw next;
    return next;
  };
});

const geminiUrl = (model) => `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`;

test('emekli model adı 404 dönerse API önerisi otomatik kullanılır', async () => {
  setEnv();
  script = [
    jsonResponse({
      error: {
        code: 404,
        message: 'This model models/gemini-2.0-flash is no longer available. '
          + 'Please update your code to use models/gemini-3.8-flash for the latest features.'
      }
    }, 404),
    jsonResponse(GEMINI_OK)
  ];

  const { askLLM } = await import('../src/llm/provider.js?case=404');
  const result = await askLLM({ prompt: 'test', json: true });

  assert.deepEqual(result, { ok: true });
  assert.equal(calls.length, 2);
  assert.match(calls[1].url, /gemini-3\.8-flash/);
});

test('503 gelirse bekleyip yeniden dener, sonunda başarılı olur', async () => {
  setEnv();
  script = [jsonResponse({ error: 'overloaded' }, 503), jsonResponse(GEMINI_OK)];

  const { askLLM } = await import('../src/llm/provider.js?case=503');
  const result = await askLLM({ prompt: 'test', json: true });
  assert.deepEqual(result, { ok: true });
  assert.equal(calls.length, 2);
});

test('bağlantı koparsa denemeler tükenince anlaşılır hata verir ve Ollama denenir', async () => {
  setEnv();
  // 3 deneme: hepsi ağ hatası; sonra Ollama'ya düşer
  script = [
    new Error('fetch failed'),
    new Error('fetch failed'),
    new Error('fetch failed'),
    jsonResponse(OLLAMA_OK)
  ];

  const { askLLM } = await import('../src/llm/provider.js?case=conn');
  const result = await askLLM({ prompt: 'test', json: true });

  assert.deepEqual(result, { ok: true });         // Ollama yanıtı
  assert.equal(calls.length, 4);
  assert.match(calls[0].url, /generativelanguage/);
  assert.match(calls[3].url, /127\.0\.0\.1:11434/);
});

test('LLM_NO_FALLBACK=1 ise yavaş yerel modele düşmeden hata verir', async () => {
  setEnv({ LLM_NO_FALLBACK: '1' });
  script = [new Error('fetch failed'), new Error('fetch failed'), new Error('fetch failed')];

  const { askLLM } = await import('../src/llm/provider.js?case=nofb');
  await assert.rejects(
    () => askLLM({ prompt: 'test', json: true }),
    (err) => {
      assert.match(err.message, /LLM_NO_FALLBACK=1/);
      return true;
    }
  );
  assert.ok(calls.every((call) => /generativelanguage/.test(call.url)), 'Ollama çağrılmamalı');
});

test('üst üste hata sonrası devre kesici: sonraki çağrı boşuna beklemez', async () => {
  setEnv({ LLM_BREAKER_FAILURES: '2' });
  script = [
    // 1. çağrı: 3 deneme de kopuyor + Ollama
    new Error('fetch failed'), new Error('fetch failed'), new Error('fetch failed'),
    jsonResponse(OLLAMA_OK),
    // 2. çağrı: 3 deneme de kopuyor + Ollama
    new Error('fetch failed'), new Error('fetch failed'), new Error('fetch failed'),
    jsonResponse(OLLAMA_OK),
    // 3. çağrı: devre açık → doğrudan Ollama
    jsonResponse(OLLAMA_OK)
  ];

  const { askLLM } = await import('../src/llm/provider.js?case=breaker');
  await askLLM({ prompt: 'a', json: true });
  await askLLM({ prompt: 'b', json: true });
  const before = calls.filter((call) => /generativelanguage/.test(call.url)).length;
  await askLLM({ prompt: 'c', json: true });
  const after = calls.filter((call) => /generativelanguage/.test(call.url)).length;

  assert.equal(before, 6);
  assert.equal(after, before, 'devre açıkken Gemini hiç denenmemeli');
});

test('yarım kalan JSON onarılır', async () => {
  const { parseJsonContent } = await import('../src/llm/gemini.js');
  const broken = '{"scenes":[{"narration":"Bal bozulmaz.","visual_objects":["bal","kavanoz"]}';
  const parsed = parseJsonContent(broken);
  assert.equal(parsed.scenes[0].narration, 'Bal bozulmaz.');
});

test('model adı ya da API sürümü değişirse öneri ayrıştırılır', async () => {
  const { suggestedModelFromError, pickModelName } = await import('../src/llm/gemini.js');
  assert.equal(
    suggestedModelFromError('This model models/gemini-2.0-flash is no longer available. use models/gemini-3.8-flash'),
    'gemini-3.8-flash'
  );
  assert.equal(
    pickModelName([
      { name: 'models/gemini-2.5-pro' },
      { name: 'models/gemini-3.8-flash' },
      { name: 'models/antigravity-preview-09-2026' },
      { name: 'models/gemini-99.0-flash-preview' },
      { name: 'models/text-embedding-004' },
      { name: 'models/imagen-3.0' }
    ]),
    'gemini-3.8-flash'
  );
});

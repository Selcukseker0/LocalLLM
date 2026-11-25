/**
 * Gemini (Google AI Studio) istemcisi.
 *
 * Bu dosyanın işi tek bir şey: Gemini çağrılarını **öngörülebilir** yapmak.
 * Önceki sürümde yaşanan üç sorun burada çözülüyor:
 *
 *  1. Model adı emekliye ayrılınca (404 "use models/X") kod ölüyordu.
 *     Artık API'nin önerdiği ad otomatik alınır, hiç yoksa model listesinden
 *     "en yeni flash" seçilir. .env'i düzenlemek gerekmez.
 *  2. Bağlantı hatasında istek sonsuza kadar asılı kalıyordu (TimeoutError).
 *     Artık her denemede `GEMINI_TIMEOUT_MS` kadar beklenir, 429/5xx/ağ
 *     hatasında üstel bekleme ile tekrar denenir.
 *  3. `fetch failed` gerçek nedeni saklıyordu. Artık ECONNRESET/ENOTFOUND
 *     gibi kod ve IPv4/IPv6 bilgisi hata mesajına yazılır.
 */

import dns from 'node:dns';

// Windows'ta IPv6 çözümlemesi Google'a bağlanırken sık sık asılı kalıyor.
try {
  dns.setDefaultResultOrder(process.env.LLM_DNS_ORDER || 'ipv4first');
} catch {
  // Eski Node sürümlerinde yok; zararsız
}

const HOST = 'https://generativelanguage.googleapis.com';
const MIN_OUTPUT_TOKENS = 8192;   // düşünme (thinking) token'ları da bu limite sayılır
const MAX_OUTPUT_TOKENS = 32768;

/** Tekrar denemeler arasındaki bekleme (testlerde kısaltılabilir). */
function backoffMs(attempt) {
  const base = Number(process.env.LLM_RETRY_BASE_MS || 2000);
  return Math.min(Number(process.env.LLM_RETRY_MAX_MS || 30_000), base * 2 ** attempt);
}

let cachedModel = null;

export function geminiKey() {
  return process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY || '';
}

export function isGeminiConfigured() {
  return Boolean(geminiKey());
}

/** .env'de GEMINI_MODEL boşsa çözülen adı döndürür (varsa). */
export function currentGeminiModel() {
  return process.env.GEMINI_MODEL || cachedModel || '';
}

export function setGeminiModel(model) {
  if (model) cachedModel = model;
}

/** 404 gövdesinden "use models/X" önerisini çıkarır. */
export function suggestedModelFromError(text) {
  const body = String(text || '');
  const patterns = [
    /use models\/([\w.\-]+)/i,
    /models\/([\w.\-]+) is no longer available/i,
    /model:?\s*models\/([\w.\-]+)/i
  ];
  for (const pattern of patterns) {
    const match = body.match(pattern);
    if (match) return match[1];
  }
  return null;
}

/**
 * Kullanılabilir modeller listesinden en uygun sohbet modelini seçer.
 * Tercih: flash-lite > flash > diğer > pro; sürüm numarası büyük olan önce.
 */
export function pickModelName(models) {
  const usable = (models || [])
    .map((item) => String(item?.name || '').replace(/^models\//, ''))
    .filter(Boolean)
    // AI Studio's model list can include internal/preview endpoints (for example
    // antigravity-preview) that do not accept the public Gemini API request body.
    // They often have a higher date/version number and would otherwise win below.
    .filter((name) => !/embedding|aqa|imagen|veo|tts|image|antigravity|preview|experimental|^gemma/i.test(name));
  if (!usable.length) return null;

  const version = (name) => {
    const match = name.match(/(\d+)(?:[.\-](\d+))?/);
    return match ? Number(match[1]) * 100 + Number(match[2] || 0) : 0;
  };
  const rank = (name) => {
    if (/flash-lite/i.test(name)) return 3;
    if (/flash/i.test(name)) return 2;
    if (/pro/i.test(name)) return 1;
    return 0;
  };
  return usable.sort((a, b) => (version(b) - version(a)) || (rank(b) - rank(a)))[0];
}

async function geminiFetch(url, options = {}, timeoutMs = null) {
  const timeout = timeoutMs || Number(process.env.GEMINI_TIMEOUT_MS || 45_000);
  try {
    return await fetch(url, { ...options, signal: AbortSignal.timeout(timeout) });
  } catch (err) {
    const cause = err?.cause?.code || err?.cause?.message || err?.message || err?.name;
    const hint = /Timeout|abort/i.test(String(cause))
      ? ` (${timeout / 1000} sn içinde yanıt yok — internet/VPN/güvenlik duvarı; GEMINI_TIMEOUT_MS ile süre değişir)`
      : '';
    const error = new Error(`Gemini bağlantı hatası: ${cause}${hint}`);
    error.connection = true;
    throw error;
  }
}

/**
 * Model adını çözer: env → 404 önerisi (cache) → canlı model listesi.
 * @returns {Promise<string>}
 */
export async function resolveGeminiModel() {
  if (process.env.GEMINI_MODEL) return process.env.GEMINI_MODEL;
  if (cachedModel) return cachedModel;

  const key = geminiKey();
  if (!key) throw new Error('GEMINI_API_KEY tanımlı değil');

  const res = await geminiFetch(`${HOST}/v1beta/models?pageSize=200`, {
    headers: { 'x-goog-api-key': key }
  });
  if (!res.ok) {
    throw new Error(`Gemini model listesi alınamadı (${res.status}): ${(await res.text()).slice(0, 200)}`);
  }
  const data = await res.json();
  const chosen = pickModelName(data.models || []);
  if (!chosen) throw new Error('Gemini hesabında kullanılabilir sohbet modeli bulunamadı');
  cachedModel = chosen;
  return chosen;
}

/**
 * generateContent çağrısı. 429/5xx/ağ hatasında üstel bekleme ile tekrar dener,
 * 404'te API'nin önerdiği model adına geçip bir kez daha dener.
 */
export async function geminiGenerate({ model, body, timeoutMs, retries = null, label = 'gemini' } = {}) {
  const key = geminiKey();
  if (!key) throw new Error('GEMINI_API_KEY tanımlı değil');

  const maxAttempts = retries ?? Number(process.env.LLM_RETRIES || 2);
  let activeModel = model || (await resolveGeminiModel());
  let lastError = null;
  let modelSwitched = false;

  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    const res = await geminiFetch(
      `${HOST}/v1beta/models/${activeModel}:generateContent`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'x-goog-api-key': key },
        body: JSON.stringify(body)
      },
      timeoutMs
    ).catch((err) => {
      lastError = err;
      return null;
    });

    if (!res) {
      // Ağ hatası: bekle ve tekrar dene
      const wait = backoffMs(attempt);
      console.warn(`[llm] ${label}: ${lastError.message}; ${Math.round(wait / 1000)} sn sonra tekrar (${attempt + 1}/${maxAttempts}).`);
      await new Promise((r) => setTimeout(r, wait));
      continue;
    }

    if (res.status === 404) {
      const text = await res.text();
      const suggestion = suggestedModelFromError(text);
      if (suggestion && !modelSwitched) {
        console.warn(`[llm] ${activeModel} artık yok; API'nin önerdiği model kullanılacak: ${suggestion}`);
        activeModel = suggestion;
        setGeminiModel(suggestion);
        modelSwitched = true;
        attempt -= 1;   // model değişimi deneme hakkı yemesin
        continue;
      }
      throw new Error(
        `Gemini modeli bulunamadı (${activeModel}). GEMINI_MODEL=${activeModel} geçerli mi? ` +
        `Hata: ${text.slice(0, 200)}`
      );
    }

    if (!res.ok) {
      const text = (await res.text()).slice(0, 300);
      lastError = new Error(`Gemini API hatası (${res.status}): ${text}`);
      const retryable = res.status === 429 || res.status >= 500;
      if (!retryable) throw lastError;
      const retryAfter = Number(res.headers?.get?.('retry-after') || 0) * 1000;
      const wait = retryAfter || backoffMs(attempt);
      console.warn(`[llm] ${label}: ${res.status}; ${Math.round(wait / 1000)} sn sonra tekrar (${attempt + 1}/${maxAttempts}).`);
      await new Promise((r) => setTimeout(r, wait));
      continue;
    }

    const data = await res.json();
    const candidate = data?.candidates?.[0];
    const finish = candidate?.finishReason;
    const content = (candidate?.content?.parts || [])
      .map((part) => part.text || '')
      .join('')
      .trim();

    if (finish && finish !== 'STOP') {
      console.warn(`[llm] ${label}: finishReason=${finish}`);
    }
    if (!content) {
      lastError = new Error(`Gemini boş yanıt döndürdü (finishReason=${finish || 'yok'})`);
      continue;
    }

    return { content, finishReason: finish, model: activeModel, raw: data };
  }

  throw lastError || new Error('Gemini çağrısı başarısız');
}

/** JSON yanıtını ayrıştırır; yarım kalan JSON'u onarmayı dener. */
export function parseJsonContent(content) {
  const text = String(content || '').trim();
  const strip = text.replace(/^```(?:json)?/i, '').replace(/```$/, '').trim();
  try {
    return JSON.parse(strip);
  } catch {
    const match = strip.match(/\{[\s\S]*\}/);
    if (match) {
      try {
        return JSON.parse(match[0]);
      } catch {
        // Yarım JSON: kapanmamış süslü parantezleri tamamla
        const repaired = match[0].replace(/,\s*$/, '');
        const open = (repaired.match(/\{/g) || []).length - (repaired.match(/\}/g) || []).length;
        const openArr = (repaired.match(/\[/g) || []).length - (repaired.match(/\]/g) || []).length;
        const patched = repaired + ']'.repeat(Math.max(0, openArr)) + '}'.repeat(Math.max(0, open));
        try {
          return JSON.parse(patched);
        } catch {
          /* aşağıda hata */
        }
      }
    }
    throw new Error(`Model geçerli JSON döndürmedi: ${strip.slice(0, 200)}`);
  }
}

export function geminiOutputTokens(requested) {
  return Math.min(MAX_OUTPUT_TOKENS, Math.max(MIN_OUTPUT_TOKENS, Number(requested) || MIN_OUTPUT_TOKENS));
}

const DB_NAME = "Aperture-model-cache";
const DB_VERSION = 1;
const STORE_NAME = "files";

/**
 * Files larger than this are never written to the cache — fetched fresh
 * from Hugging Face every time instead. Keeps one oversized checkpoint from
 * burning through a large share of the browser's per-origin storage quota,
 * and avoids a slow structured-clone IndexedDB write on the main thread for
 * something that isn't one of this app's small "tiny-random" presets.
 */
export const MAX_CACHEABLE_BYTES = 50 * 1024 * 1024; // 50 MB

interface CachedFile {
  url: string;
  bytes: ArrayBuffer;
  size: number;
  cachedAt: number;
}

let dbPromise: Promise<IDBDatabase | null> | null = null;

/** Opens (or creates) the cache database. Resolves to null rather than rejecting on any failure — IndexedDB can be unavailable or disabled (private browsing in some browsers, storage restrictions), and that should just mean "no cache", never a load failure. */
function openDb(): Promise<IDBDatabase | null> {
  if (!dbPromise) {
    dbPromise = new Promise((resolve) => {
      if (typeof indexedDB === "undefined") {
        resolve(null);
        return;
      }
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = () => req.result.createObjectStore(STORE_NAME, { keyPath: "url" });
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
    });
  }
  return dbPromise;
}

async function getCached(url: string): Promise<ArrayBuffer | undefined> {
  const db = await openDb();
  if (!db) return undefined;
  return new Promise((resolve) => {
    const req = db.transaction(STORE_NAME, "readonly").objectStore(STORE_NAME).get(url);
    req.onsuccess = () => resolve((req.result as CachedFile | undefined)?.bytes);
    req.onerror = () => resolve(undefined);
  });
}

async function putCached(url: string, bytes: ArrayBuffer): Promise<void> {
  const db = await openDb();
  if (!db) return;
  return new Promise((resolve) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    const entry: CachedFile = { url, bytes, size: bytes.byteLength, cachedAt: Date.now() };
    tx.objectStore(STORE_NAME).put(entry);
    // A write failure (quota exceeded despite the size check above, private
    // mode restrictions) should never break loading the model that's
    // already sitting in memory — just means it won't be cached this time.
    tx.oncomplete = () => resolve();
    tx.onerror = () => resolve();
  });
}

/**
 * Fetches `url` as raw bytes, transparently caching the result in
 * IndexedDB (skipping anything over MAX_CACHEABLE_BYTES) so a re-click of
 * the same preset loads instantly with no network request at all —
 * `fetchJson` and `fetchArrayBuffer` both route through this, so
 * config.json/tokenizer.json/model.safetensors are all covered uniformly.
 */
export async function fetchCachedArrayBuffer(url: string): Promise<ArrayBuffer> {
  const cached = await getCached(url);
  if (cached) return cached;

  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status} ${res.statusText}`);
  const bytes = await res.arrayBuffer();

  if (bytes.byteLength <= MAX_CACHEABLE_BYTES) {
    await putCached(url, bytes);
  }
  return bytes;
}

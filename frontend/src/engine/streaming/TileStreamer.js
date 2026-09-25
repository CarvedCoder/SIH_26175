/**
 * DepthWizard Geospatial Engine — TileStreamer
 *
 * Transport + cache layer for async terrain tile requests (heights or textures).
 *
 * Responsibilities (spec §13/§14):
 *   - Bounded LRU cache of decoded tile payloads (never grows indefinitely)
 *   - In-flight request de-duplication (one network fetch per tile key)
 *   - Concurrency-limited dispatch with priority ordering
 *     (higher `priority` values dispatch first — callers pass 1/(1+distance))
 *   - Retry with backoff on transient failures
 *   - AbortController cancellation on dispose
 *
 * The streamer is payload-agnostic: the caller supplies `fetchTile` which
 * performs the actual fetch + decode for one tile and returns the decoded
 * payload (Float32Array patch, ImageBitmap, ...).
 */

const DEFAULT_MAX_CONCURRENT = 6;
const DEFAULT_MAX_CACHE_ENTRIES = 64;
const DEFAULT_MAX_RETRIES = 1;
const DEFAULT_RETRY_DELAY_MS = 250;

export class TileStreamer {
  /**
   * @param {Object} options
   * @param {Function} options.fetchTile - async ({ z, x, y, size }, signal) => payload
   * @param {number} [options.maxConcurrent=6] - Max simultaneous network fetches
   * @param {number} [options.maxCacheEntries=64] - LRU bound on decoded payloads
   * @param {number} [options.maxRetries=1] - Retries per tile after the first attempt
   * @param {number} [options.retryDelayMs=250]
   */
  constructor(options) {
    if (typeof options?.fetchTile !== 'function') {
      throw new Error('TileStreamer requires a fetchTile callback');
    }
    this._fetchTile = options.fetchTile;
    this._maxConcurrent = Math.max(1, options.maxConcurrent ?? DEFAULT_MAX_CONCURRENT);
    this._maxCacheEntries = Math.max(1, options.maxCacheEntries ?? DEFAULT_MAX_CACHE_ENTRIES);
    this._maxRetries = Math.max(0, options.maxRetries ?? DEFAULT_MAX_RETRIES);
    this._retryDelayMs = Math.max(0, options.retryDelayMs ?? DEFAULT_RETRY_DELAY_MS);

    /** @type {Map<string, {payload: any, key: string}>} LRU (insertion order = recency) */
    this._cache = new Map();
    /** @type {Map<string, Object>} key -> pending entry */
    this._pending = new Map();
    /** @type {Set<Object>} queued entries waiting for a concurrency slot */
    this._queue = new Set();
    this._running = 0;
    this.disposed = false;

    this.stats = {
      requests: 0,
      deduped: 0,
      cacheHits: 0,
      evictions: 0,
      failures: 0,
      inflight: 0,
    };
  }

  _key(z, x, y, size) {
    return `${z}/${x}/${y}/${size}`;
  }

  /**
   * Request a tile. Resolves with the decoded payload.
   *
   * @param {number} z - Quadtree level
   * @param {number} x - Tile column
   * @param {number} y - Tile row
   * @param {number} size - Tile resolution
   * @param {Object} [opts]
   * @param {number} [opts.priority=0] - Higher dispatches sooner
   * @returns {Promise<any>}
   */
  request(z, x, y, size, opts = {}) {
    if (this.disposed) {
      return Promise.reject(new Error('TileStreamer disposed'));
    }
    const key = this._key(z, x, y, size);
    this.stats.requests++;

    // 1. Cache hit — refresh LRU recency
    const cached = this._cache.get(key);
    if (cached !== undefined) {
      this._cache.delete(key);
      this._cache.set(key, cached);
      this.stats.cacheHits++;
      return Promise.resolve(cached.payload);
    }

    // 2. In-flight — dedupe and escalate priority if the new request is hotter
    const existing = this._pending.get(key);
    if (existing) {
      existing.priority = Math.max(existing.priority, opts.priority ?? 0);
      this.stats.deduped++;
      return existing.promise;
    }

    // 3. New request
    const controller = new AbortController();
    const entry = {
      key, z, x, y, size,
      priority: opts.priority ?? 0,
      controller,
      resolve: null,
      reject: null,
      promise: null,
    };
    entry.promise = new Promise((resolve, reject) => {
      entry.resolve = resolve;
      entry.reject = reject;
    });
    this._pending.set(key, entry);
    this._queue.add(entry);
    this.stats.inflight = this._running + this._queue.size;
    this._pump();
    return entry.promise;
  }

  _pump() {
    while (this._running < this._maxConcurrent && this._queue.size > 0) {
      // Highest priority first (linear scan — queue is tiny, typically < 100)
      let best = null;
      for (const entry of this._queue) {
        if (!best || entry.priority > best.priority) best = entry;
      }
      this._queue.delete(best);
      this._run(best);
    }
    this.stats.inflight = this._running + this._queue.size;
  }

  async _run(entry) {
    this._running++;
    this.stats.inflight = this._running + this._queue.size;

    let attempt = 0;
    for (;;) {
      try {
        const payload = await this._fetchTile(
          { z: entry.z, x: entry.x, y: entry.y, size: entry.size },
          entry.controller.signal
        );
        // Store in cache (most-recently-used position = last)
        this._cache.set(entry.key, { payload, key: entry.key });
        while (this._cache.size > this._maxCacheEntries) {
          const oldestKey = this._cache.keys().next().value;
          this._cache.delete(oldestKey);
          this.stats.evictions++;
        }
        this._finish(entry);
        entry.resolve(payload);
        return;
      } catch (err) {
        if (entry.controller.signal.aborted || this.disposed) {
          this._finish(entry);
          entry.reject(new Error(`Tile request aborted: ${entry.key}`));
          return;
        }
        if (attempt < this._maxRetries) {
          attempt++;
          await new Promise((r) => setTimeout(r, this._retryDelayMs * attempt));
          continue;
        }
        this.stats.failures++;
        this._finish(entry);
        entry.reject(err);
        return;
      }
    }
  }

  _finish(entry) {
    this._pending.delete(entry.key);
    this._queue.delete(entry);
    this._running = Math.max(0, this._running - 1);
    this.stats.inflight = this._running + this._queue.size;
    this._pump();
  }

  /** Drop all cached payloads and abort everything in flight. */
  dispose() {
    this.disposed = true;
    for (const entry of this._pending.values()) {
      entry.controller.abort();
    }
    this._pending.clear();
    this._queue.clear();
    this._cache.clear();
    this._running = 0;
    this.stats.inflight = 0;
  }
}

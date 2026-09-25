/**
 * Client-side semantic data loader, NPY parser, and UV sampler.
 * Provides sub-millisecond hover lookups in memory with zero backend roundtrips.
 */

import {
  getSemanticMeta,
  getSemanticLabelsUrl,
  SEMANTIC_CLASSES,
  SEMANTIC_COLORS,
  SEMANTIC_COLORS_HEX,
} from '../api/semantic.js';

/**
 * Robust parser for NumPy .npy format (v1.0 & v2.0).
 * @param {ArrayBuffer} buffer
 * @returns {{ descr: string, shape: number[], data: TypedArray }}
 */
export function parseNpy(buffer) {
  const u8 = new Uint8Array(buffer);
  if (
    u8[0] !== 0x93 ||
    u8[1] !== 0x4e || // N
    u8[2] !== 0x55 || // U
    u8[3] !== 0x4d || // M
    u8[4] !== 0x50 || // P
    u8[5] !== 0x59    // Y
  ) {
    throw new Error('Invalid NPY magic header');
  }

  const major = u8[6];
  let headerLen = 0;
  let offset = 0;
  if (major === 1) {
    headerLen = u8[8] | (u8[9] << 8);
    offset = 10;
  } else if (major === 2) {
    headerLen = u8[8] | (u8[9] << 8) | (u8[10] << 16) | (u8[11] << 24);
    offset = 12;
  } else {
    throw new Error(`Unsupported NPY major version: ${major}`);
  }

  const headerStr = new TextDecoder('ascii').decode(
    u8.subarray(offset, offset + headerLen)
  );
  offset += headerLen;

  const descrMatch = headerStr.match(/'descr':\s*'([^']+)'/);
  const descr = descrMatch ? descrMatch[1] : '';

  const shapeMatch = headerStr.match(/'shape':\s*\(([^)]*)\)/);
  let shape = [];
  if (shapeMatch && shapeMatch[1].trim()) {
    shape = shapeMatch[1]
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean)
      .map(Number);
  }

  const dataBuffer = buffer.slice(offset);
  let data;
  if (descr.includes('u1') || descr.includes('b1')) {
    data = new Uint8Array(dataBuffer);
  } else if (descr.includes('f4')) {
    data = new Float32Array(dataBuffer);
  } else if (descr.includes('f8')) {
    data = new Float64Array(dataBuffer);
  } else if (descr.includes('i4')) {
    data = new Int32Array(dataBuffer);
  } else {
    data = new Uint8Array(dataBuffer);
  }

  return { descr, shape, data };
}

const semanticCache = new Map();

/**
 * Load and cache semantic labels and confidence arrays for a scene.
 * @param {string} sceneId
 * @returns {Promise<object>}
 */
export async function loadSemanticData(sceneId) {
  if (semanticCache.has(sceneId)) {
    return semanticCache.get(sceneId);
  }

  try {
    const meta = await getSemanticMeta(sceneId);
    if (!meta || !meta.available) {
      const res = {
        available: false,
        reason: 'Semantic segmentation unavailable for this scene.',
      };
      semanticCache.set(sceneId, res);
      return res;
    }

    const labelsUrl = meta.labels_url || getSemanticLabelsUrl(sceneId);
    const labelsResp = await fetch(labelsUrl);
    if (!labelsResp.ok) {
      throw new Error(`Failed to fetch labels: ${labelsResp.statusText}`);
    }
    const labelsBuf = await labelsResp.arrayBuffer();
    const parsedLabels = parseNpy(labelsBuf);

    let height = 0;
    let width = 0;
    if (parsedLabels.shape.length >= 2) {
      height = parsedLabels.shape[parsedLabels.shape.length - 2];
      width = parsedLabels.shape[parsedLabels.shape.length - 1];
    }

    let confData = null;
    if (meta.confidence_url) {
      try {
        const confResp = await fetch(meta.confidence_url);
        if (confResp.ok) {
          const confBuf = await confResp.arrayBuffer();
          const parsedConf = parseNpy(confBuf);
          confData = parsedConf.data;
        }
      } catch (err) {
        console.warn('Could not load semantic confidence array:', err);
      }
    }

    const result = {
      available: true,
      height,
      width,
      labels: parsedLabels.data,
      confidence: confData,
      meta,
    };
    semanticCache.set(sceneId, result);
    return result;
  } catch (err) {
    console.error('loadSemanticData error:', err);
    return { available: false, error: err.message };
  }
}

/**
 * Sample semantic class, confidence, and colors at normalized terrain (u, v) coordinates.
 * @param {object} semanticData
 * @param {number} u in [0, 1]
 * @param {number} v in [0, 1]
 * @returns {object|null}
 */
export function sampleSemanticAtUV(semanticData, u, v) {
  if (!semanticData || !semanticData.available || !semanticData.labels) {
    return null;
  }
  const { width, height, labels, confidence, meta } = semanticData;
  if (!width || !height) return null;

  const px = Math.min(Math.max(Math.round(u * (width - 1)), 0), width - 1);
  const py = Math.min(Math.max(Math.round(v * (height - 1)), 0), height - 1);
  const idx = py * width + px;

  const classId = labels[idx];
  const className = SEMANTIC_CLASSES[classId] || 'other';
  const conf = confidence ? confidence[idx] : (meta?.mean_confidence ?? 0.85);

  return {
    px,
    py,
    u,
    v,
    classId,
    className,
    confidence: conf,
    colorHex: SEMANTIC_COLORS_HEX[className] || '#9582A8',
    colorRgba: SEMANTIC_COLORS[className] || [0.584, 0.510, 0.659, 1.0],
    classFractions: meta?.class_fractions || null,
  };
}

/**
 * Compute the 4-connected building region (connected component) for a hovered pixel.
 * @param {object} semanticData
 * @param {number} startPx
 * @param {number} startPy
 * @param {number} maxCells
 * @returns {object|null}
 */
export function getBuildingConnectedComponent(semanticData, startPx, startPy, maxCells = 25000) {
  if (!semanticData || !semanticData.labels) return null;
  const { width, height, labels } = semanticData;
  const targetClass = 0; // 0 = building

  const startIndex = startPy * width + startPx;
  if (labels[startIndex] !== targetClass) return null;

  const visited = new Uint8Array(width * height);
  const queue = new Int32Array(maxCells * 2);
  let head = 0;
  let tail = 0;

  queue[tail++] = startPx;
  queue[tail++] = startPy;
  visited[startIndex] = 1;

  let minX = startPx;
  let maxX = startPx;
  let minY = startPy;
  let maxY = startPy;
  let count = 0;

  while (head < tail && count < maxCells) {
    const cx = queue[head++];
    const cy = queue[head++];
    count++;

    if (cx < minX) minX = cx;
    if (cx > maxX) maxX = cx;
    if (cy < minY) minY = cy;
    if (cy > maxY) maxY = cy;

    const neighbors = [
      [cx + 1, cy],
      [cx - 1, cy],
      [cx, cy + 1],
      [cx, cy - 1],
    ];

    for (let i = 0; i < 4; i++) {
      const nx = neighbors[i][0];
      const ny = neighbors[i][1];
      if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
        const nIdx = ny * width + nx;
        if (!visited[nIdx] && labels[nIdx] === targetClass) {
          visited[nIdx] = 1;
          if (tail + 2 < queue.length) {
            queue[tail++] = nx;
            queue[tail++] = ny;
          }
        }
      }
    }
  }

  return {
    pixelCount: count,
    bounds: { minX, maxX, minY, maxY, width: maxX - minX + 1, height: maxY - minY + 1 },
    componentMask: visited,
  };
}

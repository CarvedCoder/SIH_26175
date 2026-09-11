"""
Minimal mock backend for browser-testing the DepthWizard frontend camera modes.

Serves the endpoints the terrain workspace touches:
  GET /api/v1/scenes/demo_scene/terrain         -> heightmap/texture URLs + elevation metadata
  GET /api/v1/scenes/demo_scene/minimap         -> minimap metadata
  GET /api/v1/scenes/demo_scene/results         -> results payload
  GET /api/v1/health                            -> health probe
  GET /files/heightmap.png, /files/texture.png  -> generated images

Run:  python tools/mock_backend_for_camera_test.py  (listens on :8000)
"""
import io
import json
import math
import struct
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SIZE = 256  # heightmap/texture pixels


def build_heightmap_png():
    """Normalised-height PNG (red channel) with ridges and a valley."""
    rows = []
    for y in range(SIZE):
        row = bytearray()
        for x in range(SIZE):
            u, v = x / SIZE, y / SIZE
            h = (
                0.42 * math.exp(-(((u - 0.35) ** 2 + (v - 0.4) ** 2) / 0.03))
                + 0.30 * math.exp(-(((u - 0.7) ** 2 + (v - 0.65) ** 2) / 0.05))
                + 0.22 * math.sin(u * 9.0) * math.sin(v * 7.0) * 0.5
                + 0.5
            )
            val = max(0.0, min(1.0, h / 1.3))
            c = int(val * 255)
            row += bytes((c, c, c, 255))
        rows.append(bytes(row))

    def chunk(tag, data):
        c = struct.pack('>I', len(data)) + tag + data
        return c + struct.pack('>I', zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b''.join(b'\x00' + r for r in rows)
    return (
        b'\x89PNG\r\n\x1a\n'
        + chunk(b'IHDR', struct.pack('>IIBBBBB', SIZE, SIZE, 8, 6, 0, 0, 0))
        + chunk(b'IDAT', zlib.compress(raw, 6))
        + chunk(b'IEND', b'')
    )


def build_texture_png():
    rows = []
    for y in range(SIZE):
        row = bytearray()
        for x in range(SIZE):
            u, v = x / SIZE, y / SIZE
            shade = 0.45 + 0.3 * math.sin(u * 12.0) * math.sin(v * 10.0)
            row += bytes((int(110 * shade + 60), int(140 * shade + 50), int(90 * shade + 40), 255))
        rows.append(bytes(row))

    def chunk(tag, data):
        c = struct.pack('>I', len(data)) + tag + data
        return c + struct.pack('>I', zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b''.join(b'\x00' + r for r in rows)
    return (
        b'\x89PNG\r\n\x1a\n'
        + chunk(b'IHDR', struct.pack('>IIBBBBB', SIZE, SIZE, 8, 6, 0, 0, 0))
        + chunk(b'IDAT', zlib.compress(raw, 6))
        + chunk(b'IEND', b'')
    )


HEIGHTMAP = build_heightmap_png()
TEXTURE = build_texture_png()

TERRAIN_META = {
    'scene_id': 'demo_scene',
    'heightmap_url': '/files/heightmap.png',
    'texture_url': '/files/texture.png',
    'height_scale': 120.0,
    'min_elevation': 142.5,
    'max_elevation': 846.2,
    'width': SIZE,
    'height': SIZE,
    'georeferenced': True,
    'crs': 'EPSG:32643',
}

RESULTS = {
    'scene_id': 'demo_scene',
    'elevation_mode': 'absolute',
    'units': 'm',
    'reference_source': 'SRTM GL1 30m',
    'reference_dem_available': True,
    'min_elevation': 142.5,
    'max_elevation': 846.2,
    'outputs': ['rgb', 'depth', 'dsm', 'reference_dem', 'error', 'terrain'],
}

MINIMAP = {
    'bg_url': '/files/texture.png',
    'extent': {'min_x': 0.0, 'min_z': 0.0, 'max_x': 1.0, 'max_z': 1.0},
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype='application/json'):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split('?')[0]
        if path == '/api/v1/health':
            self._send(200, json.dumps({'status': 'ok', 'version': 'mock', 'model_loaded': True}).encode())
        elif path.startswith('/api/v1/scenes/') and path.endswith('/terrain'):
            meta = dict(TERRAIN_META, scene_id=path.split('/')[4])
            self._send(200, json.dumps(meta).encode())
        elif path.startswith('/api/v1/scenes/') and path.endswith('/results'):
            results = dict(RESULTS, scene_id=path.split('/')[4])
            self._send(200, json.dumps(results).encode())
        elif path.startswith('/api/v1/scenes/') and path.endswith('/minimap'):
            self._send(200, json.dumps(MINIMAP).encode())
        elif path == '/files/heightmap.png':
            self._send(200, HEIGHTMAP, 'image/png')
        elif path == '/files/texture.png':
            self._send(200, TEXTURE, 'image/png')
        else:
            self._send(404, json.dumps({'error': {'code': 'NOT_FOUND', 'message': path, 'recoverable': False}}).encode())


if __name__ == '__main__':
    print('Mock backend listening on http://localhost:8000')
    ThreadingHTTPServer(('127.0.0.1', 8000), Handler).serve_forever()

/**
 * DepthWizard — Upload Zone
 *
 * Large drag-and-drop target. Calls uploadScene() on file selection.
 * Transitions state: NO_SCENE → UPLOADING → SCENE_READY (on success) | FAILED (on error)
 *
 * Multi-file support (§27-adjacent tiles): up to MAX_FILES images at once.
 * When every file is a GeoTIFF the backend's mosaic endpoint decides
 * whether they are genuinely ADJACENT tiles (shared CRS + ground
 * resolution + touching footprints): adjacent tiles merge into ONE
 * seamless scene; any rejection (non-adjacent, mixed formats) falls back
 * to one scene per image, switchable via the terrain-workspace switcher.
 *
 * DESIGN.md: no card chrome, instrument-blue dashed border on drag-over.
 * Spec §4 Upload Area, §46 Upload Endpoint.
 */
import { useState, useRef, useCallback } from 'react';
import { Upload, FileImage, AlertCircle } from 'lucide-react';
import { useApp } from '../../store/appStore.jsx';
import { uploadScene, mosaicScenes } from '../../api/upload.js';

const ACCEPTED = ['.png', '.jpg', '.jpeg', '.tif', '.tiff'];
const ACCEPT_MIME = 'image/png,image/jpeg,image/tiff,.tif,.tiff';

/** Max images per multi-upload batch. 6 keeps the mosaic merge, the
 * sequential scene processing and the switcher row readable on a laptop
 * viewport, and bounds total GPU/inference time per batch. */
export const MAX_FILES = 6;

function isAccepted(file) {
  const ext = '.' + file.name.split('.').pop().toLowerCase();
  return ACCEPTED.includes(ext) || file.type.startsWith('image/');
}

function isGeoTiff(file) {
  return /\.tiff?$/i.test(file.name);
}

export default function UploadZone() {
  const { state, actions } = useApp();
  const [dragOver, setDragOver]   = useState(false);
  const [localErr, setLocalErr]   = useState(null);
  const [progressNote, setProgressNote] = useState(null);
  const inputRef = useRef(null);

  const isUploading = state.status === 'UPLOADING';

  const fail = (err) => actions.uploadFail({
    code:        err.code        ?? 'UPLOAD_ERROR',
    message:     err.message     ?? 'Upload failed.',
    recoverable: err.recoverable ?? true,
  });

  /** One file per scene, in order; first becomes the active scene. */
  const uploadSeparateScenes = useCallback(async (files) => {
    const scenes = [];
    for (const file of files) {
      setProgressNote(`Uploading ${file.name} (${scenes.length + 1}/${files.length})…`);
      scenes.push(await uploadScene(file));
    }
    setProgressNote(null);
    actions.uploadBatch(scenes[0], scenes);
  }, [actions]);

  /** Multi-file entry point: try a seamless mosaic first, else split. */
  const handleFiles = useCallback(async (fileList) => {
    const files = Array.from(fileList ?? []);
    const rejected = files.filter(f => !isAccepted(f));
    if (rejected.length) {
      setLocalErr(`Unsupported file type: ${rejected[0].name}. Use PNG, JPG, or GeoTIFF.`);
      return;
    }
    if (!files.length) return;
    if (files.length > MAX_FILES) {
      setLocalErr(`Up to ${MAX_FILES} images per batch — you selected ${files.length}.`);
      return;
    }

    setLocalErr(null);
    setProgressNote(null);
    actions.startUpload();

    try {
      if (files.length === 1) {
        const scene = await uploadScene(files[0]);
        actions.uploadSuccess(scene);
        return;
      }

      if (files.every(isGeoTiff)) {
        // Adjacent-tile attempt: upload the first file, then hand the
        // backend the whole set — it merges only if they genuinely touch.
        setProgressNote(`Checking ${files.length} GeoTIFFs for adjacent tiles…`);
        const first = await uploadScene(files[0]);
        try {
          const merged = await mosaicScenes(first.scene_id, files);
          setProgressNote(null);
          actions.uploadSuccess({
            ...merged,
            mosaic_of: files.map(f => f.name),
          });
          return;
        } catch (mosaicErr) {
          // 400 = not adjacent tiles (or CRS/GSD mismatch) — each image
          // stays its own scene. Other errors (network etc.) re-throw.
          if (mosaicErr?.status !== 400) throw mosaicErr;
          console.info('[upload] mosaic rejected — inputs are not adjacent tiles:', mosaicErr.message);
        }
      }

      await uploadSeparateScenes(files);
    } catch (err) {
      setProgressNote(null);
      fail(err);
    }
  }, [actions, uploadSeparateScenes]);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files?.length) handleFiles(e.dataTransfer.files);
  }, [handleFiles]);

  const onDragOver = useCallback((e) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const onDragLeave = useCallback(() => setDragOver(false), []);

  const onInputChange = useCallback((e) => {
    if (e.target.files?.length) handleFiles(e.target.files);
    e.target.value = '';
  }, [handleFiles]);

  return (
    <div
      role="region"
      aria-label="Image upload area"
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onClick={() => !isUploading && inputRef.current?.click()}
      style={{
        position: 'relative',
        width: '100%',
        maxWidth: '100%',
        padding: '56px 36px',
        borderRadius: 'var(--dw-radius)',
        border: `1px dashed ${dragOver ? 'var(--dw-accent)' : 'var(--dw-rim)'}`,
        background: dragOver ? 'var(--dw-accent-soft)' : 'var(--dw-surface)',
        cursor: isUploading ? 'wait' : 'pointer',
        transition: 'border-color 150ms ease, background 150ms ease',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 14,
        outline: 'none',
      }}
      tabIndex={0}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click(); }}
      onFocus={e => { e.currentTarget.style.outline = '2px solid var(--dw-accent)'; e.currentTarget.style.outlineOffset = '2px'; }}
      onBlur={e => { e.currentTarget.style.outline = 'none'; }}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT_MIME}
        multiple
        onChange={onInputChange}
        style={{ position: 'absolute', opacity: 0, pointerEvents: 'none' }}
        aria-hidden="true"
        tabIndex={-1}
      />

      {/* Icon */}
      <div style={{ color: isUploading ? 'var(--dw-accent)' : dragOver ? 'var(--dw-accent)' : 'var(--dw-fg-muted)' }}>
        {isUploading
          ? <UploadSpinner />
          : <Upload size={36} strokeWidth={1.5} />}
      </div>

      {/* Primary label */}
      <p style={{
        fontFamily: 'var(--dw-font-ui)',
        fontSize: 17,
        fontWeight: 600,
        color: 'var(--dw-fg)',
        margin: 0,
        textAlign: 'center',
      }}>
        {isUploading
          ? (progressNote ?? 'Uploading…')
          : dragOver ? 'Drop to upload' : 'Drop your image here'}
      </p>

      {/* or / browse */}
      {!isUploading && (
        <>
          <p style={{ fontFamily: 'var(--dw-font-ui)', fontSize: 14, color: 'var(--dw-fg-muted)', margin: 0 }}>or</p>
          <button
            onClick={(e) => { e.stopPropagation(); inputRef.current?.click(); }}
            style={{
              background: 'none',
              border: '1px solid var(--dw-rim)',
              borderRadius: 'var(--dw-radius-sm)',
              padding: '8px 20px',
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 14,
              fontWeight: 500,
              color: 'var(--dw-fg)',
              cursor: 'pointer',
              transition: 'border-color 150ms ease, background 150ms ease',
              outline: 'none',
            }}
            onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--dw-accent)'; e.currentTarget.style.background = 'var(--dw-panel)'; }}
            onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--dw-rim)'; e.currentTarget.style.background = 'none'; }}
            onFocus={e => { e.currentTarget.style.outline = '2px solid var(--dw-accent)'; e.currentTarget.style.outlineOffset = '2px'; }}
            onBlur={e => { e.currentTarget.style.outline = 'none'; }}
          >
            Browse Files
          </button>
          <p style={{
            fontFamily: 'var(--dw-font-data)',
            fontSize: 12.5,
            color: 'var(--dw-fg-muted)',
            margin: 0,
            letterSpacing: '0.04em',
          }}>
            PNG · JPG · GeoTIFF — up to {MAX_FILES} images ·
            adjacent GeoTIFF tiles are auto-merged into one scene
          </p>
        </>
      )}

      {/* Error */}
      {localErr && (
        <div
          role="alert"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            color: 'var(--dw-fault)',
            fontFamily: 'var(--dw-font-ui)',
            fontSize: 13.5,
            marginTop: 4,
          }}
        >
          <AlertCircle size={16} strokeWidth={1.5} />
          <span>{localErr}</span>
        </div>
      )}
    </div>
  );
}

function UploadSpinner() {
  return (
    <svg width="36" height="36" viewBox="0 0 28 28" fill="none" aria-label="Uploading…">
      <circle cx="14" cy="14" r="11" stroke="var(--dw-rim)" strokeWidth="2" />
      <path
        d="M14 3 A11 11 0 0 1 25 14"
        stroke="var(--dw-accent)"
        strokeWidth="2"
        strokeLinecap="round"
      >
        <animateTransform
          attributeName="transform"
          type="rotate"
          from="0 14 14"
          to="360 14 14"
          dur="0.8s"
          repeatCount="indefinite"
        />
      </path>
    </svg>
  );
}

/**
 * DepthWizard — Upload Zone
 *
 * Large drag-and-drop target. Calls uploadScene() on file selection.
 * Transitions state: NO_SCENE → UPLOADING → SCENE_READY (on success) | FAILED (on error)
 *
 * DESIGN.md: no card chrome, instrument-blue dashed border on drag-over.
 * Spec §4 Upload Area, §46 Upload Endpoint.
 */
import { useState, useRef, useCallback } from 'react';
import { Upload, FileImage, AlertCircle } from 'lucide-react';
import { useApp } from '../../store/appStore.jsx';
import { uploadScene } from '../../api/upload.js';

const ACCEPTED = ['.png', '.jpg', '.jpeg', '.tif', '.tiff'];
const ACCEPT_MIME = 'image/png,image/jpeg,image/tiff,.tif,.tiff';

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(1)} MB`;
}

function isAccepted(file) {
  const ext = '.' + file.name.split('.').pop().toLowerCase();
  return ACCEPTED.includes(ext) || file.type.startsWith('image/');
}

export default function UploadZone() {
  const { state, actions } = useApp();
  const [dragOver, setDragOver]   = useState(false);
  const [localErr, setLocalErr]   = useState(null);
  const inputRef = useRef(null);

  const isUploading = state.status === 'UPLOADING';

  const handleFile = useCallback(async (file) => {
    if (!isAccepted(file)) {
      setLocalErr(`Unsupported file type: ${file.name}. Use PNG, JPG, or GeoTIFF.`);
      return;
    }
    setLocalErr(null);
    actions.startUpload();
    try {
      const scene = await uploadScene(file);
      actions.uploadSuccess(scene);
    } catch (err) {
      actions.uploadFail({
        code:        err.code        ?? 'UPLOAD_ERROR',
        message:     err.message     ?? 'Upload failed.',
        recoverable: err.recoverable ?? true,
      });
    }
  }, [actions]);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const onDragOver = useCallback((e) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const onDragLeave = useCallback(() => setDragOver(false), []);

  const onInputChange = useCallback((e) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    e.target.value = '';
  }, [handleFile]);

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
        maxWidth: 580,
        padding: '56px 36px',
        borderRadius: 'var(--dw-radius)',
        border: `1px dashed ${dragOver ? 'var(--dw-accent)' : 'var(--dw-rim)'}`,
        background: dragOver ? 'rgba(59,130,246,0.06)' : 'var(--dw-surface)',
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
        {isUploading ? 'Uploading…' : dragOver ? 'Drop to upload' : 'Drop your image here'}
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
            PNG · JPG · GeoTIFF
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

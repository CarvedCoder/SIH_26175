/**
 * DepthWizard — Global Error Boundary (Phase 18, Task 18.1)
 *
 * Catches any render errors anywhere in the component tree.
 * Implements structured recovery message adhering to spec §31 Rule 6:
 *   - What happened
 *   - Why it happened
 *   - What can I do next?
 *
 * DESIGN.md: Operate mode. Instrument panel register. No card shadows. Monospace technical readout.
 */
import React from 'react';
import { AlertTriangle, RotateCcw, Home, Copy, Check } from 'lucide-react';

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      copied: false,
    };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('[DepthWizard ErrorBoundary caught]', error, errorInfo);
    this.setState({ errorInfo });
  }

  handleReset = () => {
    try {
      localStorage.removeItem('dw_active_scene');
    } catch {}
    window.location.href = '/';
  };

  handleReload = () => {
    window.location.reload();
  };

  handleCopy = () => {
    const text = `DepthWizard Interface Error Report:
Error: ${this.state.error?.message || 'Unknown error'}
Stack: ${this.state.error?.stack || 'No stack trace'}
Component Stack: ${this.state.errorInfo?.componentStack || 'None'}
Timestamp: ${new Date().toISOString()}`;
    navigator.clipboard?.writeText(text);
    this.setState({ copied: true });
    setTimeout(() => this.setState({ copied: false }), 2500);
  };

  render() {
    if (this.state.hasError) {
      const error = this.state.error;

      return (
        <div
          role="alert"
          aria-live="assertive"
          style={{
            minHeight: '100vh',
            width: '100vw',
            background: 'var(--dw-void)',
            color: 'var(--dw-fg)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: 24,
            boxSizing: 'border-box',
          }}
        >
          <div
            style={{
              width: '100%',
              maxWidth: 620,
              background: 'var(--dw-panel)',
              border: '1px solid var(--dw-rim)',
              borderRadius: 'var(--dw-radius-md)',
              padding: 24,
              display: 'flex',
              flexDirection: 'column',
              gap: 16,
            }}
          >
            {/* Header */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: 10,
              paddingBottom: 12,
              borderBottom: '1px solid var(--dw-rim)',
            }}>
              <div style={{
                width: 28,
                height: 28,
                borderRadius: 'var(--dw-radius-sm)',
                background: 'rgba(239,68,68,0.12)',
                border: '1px solid rgba(239,68,68,0.3)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'var(--dw-fault)',
              }}>
                <AlertTriangle size={16} strokeWidth={1.5} />
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <span style={{
                  fontFamily: 'var(--dw-font-ui)',
                  fontSize: 14,
                  fontWeight: 600,
                  color: 'var(--dw-fg)',
                }}>
                  Application Interface Interrupted
                </span>
                <span style={{
                  fontFamily: 'var(--dw-font-data)',
                  fontSize: 10,
                  color: 'var(--dw-fg-ghost)',
                  letterSpacing: '0.04em',
                }}>
                  RECOVERY SYSTEM (§31 RULE 6)
                </span>
              </div>
            </div>

            {/* 3-Part Structured Message */}
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              gap: 12,
              fontFamily: 'var(--dw-font-ui)',
              fontSize: 13,
              lineHeight: 1.5,
            }}>
              <div>
                <span style={{
                  fontFamily: 'var(--dw-font-ui)',
                  fontSize: 10,
                  fontWeight: 500,
                  letterSpacing: '0.06em',
                  textTransform: 'uppercase',
                  color: 'var(--dw-fg-ghost)',
                  display: 'block',
                  marginBottom: 2,
                }}>
                  What happened
                </span>
                <span style={{ color: 'var(--dw-fg)' }}>
                  A runtime rendering exception occurred while mounting or updating the 3D terrain workspace.
                </span>
              </div>

              <div>
                <span style={{
                  fontFamily: 'var(--dw-font-ui)',
                  fontSize: 10,
                  fontWeight: 500,
                  letterSpacing: '0.06em',
                  textTransform: 'uppercase',
                  color: 'var(--dw-fg-ghost)',
                  display: 'block',
                  marginBottom: 2,
                }}>
                  Why it happened
                </span>
                <span style={{
                  fontFamily: 'var(--dw-font-data)',
                  fontSize: 12,
                  color: 'var(--dw-fault)',
                  display: 'block',
                  background: 'var(--dw-surface)',
                  padding: '6px 10px',
                  borderRadius: 'var(--dw-radius-sm)',
                  border: '1px solid var(--dw-rim)',
                  wordBreak: 'break-all',
                }}>
                  {error?.message || 'Component failed to render.'}
                </span>
              </div>

              <div>
                <span style={{
                  fontFamily: 'var(--dw-font-ui)',
                  fontSize: 10,
                  fontWeight: 500,
                  letterSpacing: '0.06em',
                  textTransform: 'uppercase',
                  color: 'var(--dw-fg-ghost)',
                  display: 'block',
                  marginBottom: 2,
                }}>
                  What can I do next?
                </span>
                <span style={{ color: 'var(--dw-fg-muted)' }}>
                  You can reload this page to re-initialize the WebGL context, or return to the landing page to select a different project.
                </span>
              </div>
            </div>

            {/* Actions strip */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: 10,
              paddingTop: 12,
              borderTop: '1px solid var(--dw-rim)',
              flexWrap: 'wrap',
            }}>
              <button
                onClick={this.handleReload}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 6,
                  height: 32,
                  padding: '0 12px',
                  background: 'var(--dw-accent)',
                  border: 'none',
                  borderRadius: 'var(--dw-radius-sm)',
                  fontFamily: 'var(--dw-font-ui)',
                  fontSize: 12,
                  fontWeight: 500,
                  color: '#fff',
                  cursor: 'pointer',
                  outline: 'none',
                }}
              >
                <RotateCcw size={13} strokeWidth={1.5} />
                Reload Page
              </button>

              <button
                onClick={this.handleReset}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 6,
                  height: 32,
                  padding: '0 12px',
                  background: 'var(--dw-surface)',
                  border: '1px solid var(--dw-rim)',
                  borderRadius: 'var(--dw-radius-sm)',
                  fontFamily: 'var(--dw-font-ui)',
                  fontSize: 12,
                  color: 'var(--dw-fg)',
                  cursor: 'pointer',
                  outline: 'none',
                }}
              >
                <Home size={13} strokeWidth={1.5} />
                Return to Landing
              </button>

              <button
                onClick={this.handleCopy}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 6,
                  height: 32,
                  padding: '0 12px',
                  background: 'none',
                  border: '1px solid var(--dw-rim)',
                  borderRadius: 'var(--dw-radius-sm)',
                  fontFamily: 'var(--dw-font-ui)',
                  fontSize: 12,
                  color: this.state.copied ? 'var(--dw-confirm)' : 'var(--dw-fg-muted)',
                  cursor: 'pointer',
                  marginLeft: 'auto',
                  outline: 'none',
                }}
              >
                {this.state.copied ? (
                  <>
                    <Check size={13} strokeWidth={2} />
                    <span>Copied</span>
                  </>
                ) : (
                  <>
                    <Copy size={13} strokeWidth={1.5} />
                    <span>Copy Diagnostics</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

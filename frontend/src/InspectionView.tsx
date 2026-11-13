import { useCallback, useEffect, useRef, useState, type DragEvent } from "react";
import Viewer from "./Viewer";
import { ApiError, getModel, predict, type ModelInfo, type Prediction } from "./api";
import { maskToOverlayUrl, pngDataUrl } from "./overlay";

type Status =
  | { kind: "idle" }
  | { kind: "uploading"; name: string }
  | { kind: "done"; result: Prediction }
  | { kind: "error"; code: string; title: string; message: string };

const ACCEPTED = ["image/jpeg", "image/png"];
// 100% is only shown for an exact 1.0; a score of 0.9996 reads ">99.9%" so it is never rounded up to certainty.
const pct = (x: number) => (x >= 1 ? "100%" : x >= 0.9995 ? ">99.9%" : `${(x * 100).toFixed(1)}%`);

function errorCopy(e: unknown): { code: string; title: string; message: string } {
  if (e instanceof ApiError) {
    switch (e.code) {
      case "unsupported_file_type":
        return { code: e.code, title: "Unsupported file", message: e.message };
      case "invalid_dimensions":
      case "empty_file":
        return { code: e.code, title: "Image cannot be analysed", message: e.message };
      case "file_too_large":
        return { code: e.code, title: "File too large", message: e.message };
      case "network":
        return { code: e.code, title: "Service unreachable", message: e.message };
    }
    if (e.status === 503)
      return { code: e.code, title: "Model unavailable", message: e.message };
    return { code: e.code, title: "Server error", message: e.message };
  }
  return { code: "unknown", title: "Unexpected error", message: (e as Error)?.message ?? "Unknown failure." };
}

export default function InspectionView() {
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [modelErr, setModelErr] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  const [preview, setPreview] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [mode, setMode] = useState<"original" | "overlay">("overlay");
  const [opacity, setOpacity] = useState(0.6);
  const [overlayUrl, setOverlayUrl] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const abort = useRef<AbortController | null>(null);

  const loadModel = useCallback(() => {
    setModelErr(null);
    getModel()
      .then(setModel)
      .catch((e: Error) => {
        setModel(null);
        setModelErr(e.message);
      });
  }, []);
  useEffect(loadModel, [loadModel]);

  useEffect(
    () => () => {
      if (preview) URL.revokeObjectURL(preview);
    },
    [preview],
  );

  const result = status.kind === "done" ? status.result : null;

  useEffect(() => {
    let live = true;
    setOverlayUrl(null);
    if (result?.predicted_mask) {
      maskToOverlayUrl(pngDataUrl(result.predicted_mask.data), [255, 86, 64])
        .then((u) => live && setOverlayUrl(u))
        .catch(() => live && setOverlayUrl(null));
    }
    return () => {
      live = false;
    };
  }, [result]);

  const analyse = useCallback(async (file: File) => {
    abort.current?.abort();
    if (!ACCEPTED.includes(file.type)) {
      setStatus({
        kind: "error",
        code: "unsupported_file_type",
        title: "Unsupported file",
        message: `“${file.name}” is ${file.type || "an unknown type"}. Upload a JPEG or PNG SEM image.`,
      });
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    setFileName(file.name);
    setStatus({ kind: "uploading", name: file.name });
    const ctl = new AbortController();
    abort.current = ctl;
    try {
      const r = await predict(file, ctl.signal);
      setStatus({ kind: "done", result: r });
    } catch (e) {
      if ((e as Error).name === "AbortError") return;
      setStatus({ kind: "error", ...errorCopy(e) });
    }
  }, []);

  const cancel = () => {
    abort.current?.abort();
    setStatus({ kind: "idle" });
  };

  const reset = () => {
    abort.current?.abort();
    setPreview(null);
    setFileName(null);
    setStatus({ kind: "idle" });
    if (input.current) input.current.value = "";
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) void analyse(f);
  };

  const unavailable = model !== null && !model.ready;
  const hasMask = !!result?.predicted_mask;
  const showOverlay = hasMask && mode === "overlay" && overlayUrl;
  const busy = status.kind === "uploading";

  return (
    <div className="inspection">
      <div
        className={`stage-col${dragging ? " dragging" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <Viewer
          src={preview}
          alt={fileName ? `SEM image ${fileName}` : "SEM image"}
          overlayUrl={showOverlay ? overlayUrl : null}
          overlayOpacity={opacity}
          overlayLabel={result?.predicted_mask?.defect_pixels === 0 ? "Predicted defect: none" : "Predicted defect"}
        >
          {!preview && (
            <div className="empty-state">
              <button type="button" className="dropzone" onClick={() => input.current?.click()} disabled={unavailable}>
                <span className="dz-title">Drop a SEM image here</span>
                <span className="dz-sub">or choose a file · JPEG or PNG, native 480 × 480</span>
              </button>
            </div>
          )}
          {busy && (
            <div className="busy" role="status">
              <span className="spinner" aria-hidden="true" />
              Analysing {status.kind === "uploading" ? status.name : ""}…
            </div>
          )}
          {dragging && <div className="drop-hint">Release to analyse</div>}
        </Viewer>
      </div>

      <aside className="side" aria-label="Inspection result">
        <section className="card">
          <h2 className="card-title">Image</h2>
          <input
            ref={input}
            type="file"
            accept="image/jpeg,image/png"
            hidden
            aria-label="Choose SEM image file"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void analyse(f);
              e.target.value = "";
            }}
          />
          <div className="row">
            <button type="button" className="btn primary" onClick={() => input.current?.click()} disabled={unavailable || busy}>
              Choose image…
            </button>
            {busy ? (
              <button type="button" className="btn" onClick={cancel}>
                Cancel
              </button>
            ) : (
              <button type="button" className="btn" onClick={reset} disabled={!preview && status.kind === "idle"}>
                Clear
              </button>
            )}
          </div>
          {fileName && <p className="muted mono truncate" title={fileName}>{fileName}</p>}
        </section>

        {modelErr && (
          <div className="notice error" role="alert">
            <strong>Service unreachable.</strong> {modelErr}
            <div>
              <button type="button" className="btn small" onClick={loadModel}>
                Retry
              </button>
            </div>
          </div>
        )}
        {unavailable && model && (
          <div className="notice warn" role="alert">
            <strong>Model unavailable ({model.reason}).</strong> {model.message}
            <div>
              <button type="button" className="btn small" onClick={loadModel}>
                Re-check model
              </button>
            </div>
          </div>
        )}

        {status.kind === "error" && (
          <div className="notice error" role="alert">
            <strong>{status.title}.</strong> {status.message}
          </div>
        )}

        <section className="card" aria-live="polite">
          <h2 className="card-title">Visual inspection</h2>
          {!result && status.kind !== "uploading" && <p className="muted">No result yet. Upload an image to classify this imaged site.</p>}
          {busy && <p className="muted">Waiting for the model…</p>}
          {result && (
            <>
              <div className={`verdict ${result.inspection_decision.toLowerCase()}`}>
                <span className="verdict-icon" aria-hidden="true">
                  {result.inspection_decision === "PASS" ? "✓" : "✕"}
                </span>
                <div>
                  <div className="verdict-label" data-testid="decision">
                    {result.inspection_decision}
                  </div>
                  <div className="verdict-sub">
                    {result.inspection_decision === "PASS" ? "No visible defect" : "Visible defect"} · this imaged site
                  </div>
                </div>
              </div>
              <p className="reason">{result.decision_reason}</p>
              <dl className="facts">
                <div>
                  <dt>Predicted class</dt>
                  <dd data-testid="predicted-class">
                    {result.predicted_class_name}
                    <span className="muted"> (source ID {result.predicted_class_id})</span>
                  </dd>
                </div>
                <div>
                  <dt>Softmax score</dt>
                  <dd className="mono">{pct(result.predicted_class_score)}</dd>
                </div>
                <div>
                  <dt>Model</dt>
                  <dd className="mono wrap">{result.model_version}</dd>
                </div>
                <div>
                  <dt>Processing time</dt>
                  <dd className="mono">{result.processing_time_ms.toFixed(0)} ms</dd>
                </div>
                <div>
                  <dt>Image</dt>
                  <dd className="mono">
                    {result.original_width} × {result.original_height} px
                  </dd>
                </div>
              </dl>
              <details className="scores">
                <summary>All six class scores</summary>
                <ul>
                  {Object.entries(result.class_scores)
                    .sort(([a], [b]) => Number(a) - Number(b))
                    .map(([id, s]) => (
                      <li key={id} className={Number(id) === result.predicted_class_id ? "top" : ""}>
                        <span className="sc-name">Class {id}</span>
                        <span className="sc-bar" aria-hidden="true">
                          <i style={{ width: `${Math.max(s * 100, 0.5)}%` }} />
                        </span>
                        <span className="sc-val mono">{pct(s)}</span>
                      </li>
                    ))}
                </ul>
              </details>
            </>
          )}
        </section>

        {result && (
          <section className="card">
            <h2 className="card-title">Defect mask</h2>
            {hasMask ? (
              <>
                <div className="seg" role="radiogroup" aria-label="Image display">
                  <button
                    type="button"
                    role="radio"
                    aria-checked={mode === "original"}
                    className={mode === "original" ? "on" : ""}
                    onClick={() => setMode("original")}
                  >
                    Original
                  </button>
                  <button
                    type="button"
                    role="radio"
                    aria-checked={mode === "overlay"}
                    className={mode === "overlay" ? "on" : ""}
                    onClick={() => setMode("overlay")}
                  >
                    Predicted defect
                  </button>
                </div>
                <label className="slider">
                  <span>Overlay opacity</span>
                  <input
                    type="range"
                    min={5}
                    max={100}
                    value={Math.round(opacity * 100)}
                    disabled={mode !== "overlay"}
                    onChange={(e) => setOpacity(Number(e.target.value) / 100)}
                    aria-valuetext={`${Math.round(opacity * 100)} percent`}
                  />
                  <output className="mono">{Math.round(opacity * 100)}%</output>
                </label>
                <p className="muted small-text">
                  {result.predicted_mask!.defect_pixels.toLocaleString()} px predicted defect. The mask is a model output, not an
                  expert annotation.
                </p>
              </>
            ) : (
              <p className="muted">This model ({result.model_kind}) returns classification only; it has no mask output.</p>
            )}
          </section>
        )}

        {model?.ready && model.model && (
          <p className="muted small-text meta">
            Active model <span className="mono">{model.model.model_version}</span> · {model.model.kind} · {model.device}
            <br />
            Visual inspection of the imaged site only; not die yield or wafer disposition.
          </p>
        )}
      </aside>
    </div>
  );
}

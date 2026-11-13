import { useCallback, useEffect, useLayoutEffect, useRef, useState, type KeyboardEvent, type PointerEvent, type ReactNode } from "react";

interface Props {
  src: string | null;
  alt: string;
  overlayUrl?: string | null;
  overlayOpacity?: number;
  overlayLabel?: string;
  children?: ReactNode; // empty / status content drawn on top of the viewport
  compact?: boolean;
}

interface T {
  x: number;
  y: number;
  k: number;
}

const MAX_K = 32;

export default function Viewer({ src, alt, overlayUrl, overlayOpacity = 0.55, overlayLabel, children, compact }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [nat, setNat] = useState<{ w: number; h: number } | null>(null);
  const [t, setT] = useState<T>({ x: 0, y: 0, k: 1 });
  const [auto, setAuto] = useState(true); // follow fit-to-view until the user zooms or pans
  const drag = useRef<{ px: number; py: number; x: number; y: number } | null>(null);

  const fitT = useCallback(
    (n = nat, s = size): T => {
      if (!n || !s.w || !s.h) return { x: 0, y: 0, k: 1 };
      const k = Math.min(s.w / n.w, s.h / n.h, MAX_K);
      return { k, x: (s.w - n.w * k) / 2, y: (s.h - n.h * k) / 2 };
    },
    [nat, size],
  );

  useLayoutEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    setSize({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    setNat(null);
    setAuto(true);
  }, [src]);

  useEffect(() => {
    if (auto) setT(fitT());
  }, [auto, fitT]);

  const zoomAt = useCallback(
    (factor: number, cx: number, cy: number) => {
      setAuto(false);
      setT((p) => {
        const minK = nat ? Math.min(fitT().k, 1) * 0.5 : 0.1;
        const k = Math.min(MAX_K, Math.max(minK, p.k * factor));
        const f = k / p.k;
        return { k, x: cx - (cx - p.x) * f, y: cy - (cy - p.y) * f };
      });
    },
    [fitT, nat],
  );

  // Wheel zoom around the cursor (needs a non-passive listener to prevent page scroll).
  useEffect(() => {
    const el = box.current;
    if (!el || !src) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      zoomAt(Math.exp(-e.deltaY * 0.0015), e.clientX - r.left, e.clientY - r.top);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [src, zoomAt]);

  const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    if (!src || e.button !== 0) return;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    drag.current = { px: e.clientX, py: e.clientY, x: t.x, y: t.y };
  };
  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    if (!d) return;
    setAuto(false);
    setT((p) => ({ ...p, x: d.x + e.clientX - d.px, y: d.y + e.clientY - d.py }));
  };
  const endDrag = () => {
    drag.current = null;
  };

  const center = () => [size.w / 2, size.h / 2] as const;
  const zoomIn = () => zoomAt(1.4, ...center());
  const zoomOut = () => zoomAt(1 / 1.4, ...center());
  const fit = () => setAuto(true);
  const actual = () => {
    if (!nat) return;
    setAuto(false);
    setT({ k: 1, x: (size.w - nat.w) / 2, y: (size.h - nat.h) / 2 });
  };

  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    if (!src) return;
    const step = 48;
    const map: Record<string, () => void> = {
      "+": zoomIn,
      "=": zoomIn,
      "-": zoomOut,
      "0": fit,
      "1": actual,
      ArrowLeft: () => (setAuto(false), setT((p) => ({ ...p, x: p.x + step }))),
      ArrowRight: () => (setAuto(false), setT((p) => ({ ...p, x: p.x - step }))),
      ArrowUp: () => (setAuto(false), setT((p) => ({ ...p, y: p.y + step }))),
      ArrowDown: () => (setAuto(false), setT((p) => ({ ...p, y: p.y - step }))),
    };
    const fn = map[e.key];
    if (fn) {
      e.preventDefault();
      fn();
    }
  };

  return (
    <div className={`viewer${compact ? " compact" : ""}`}>
      <div
        ref={box}
        className={`viewport${src ? " has-image" : ""}`}
        tabIndex={src ? 0 : -1}
        role="group"
        aria-label={src ? `${alt}. Zoom with plus and minus, pan with arrow keys, 0 to fit, 1 for 100 percent.` : "Image viewport"}
        onKeyDown={onKey}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
      >
        {src && (
          <div
            className="stage"
            style={{
              width: nat?.w ?? 0,
              height: nat?.h ?? 0,
              transform: `translate(${t.x}px, ${t.y}px) scale(${t.k})`,
              visibility: nat ? "visible" : "hidden",
            }}
          >
            <img
              src={src}
              alt={alt}
              draggable={false}
              onLoad={(e) => setNat({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
              style={{ imageRendering: t.k >= 2 ? "pixelated" : "auto" }}
            />
            {overlayUrl && nat && (
              <img
                className="overlay"
                src={overlayUrl}
                alt=""
                aria-hidden="true"
                draggable={false}
                style={{ opacity: overlayOpacity, imageRendering: t.k >= 2 ? "pixelated" : "auto" }}
              />
            )}
          </div>
        )}
        {overlayUrl && overlayLabel && (
          <span className="overlay-chip">
            <i aria-hidden="true" />
            {overlayLabel}
          </span>
        )}
        {children}
      </div>
      {src && (
        <div className="viewer-tools" role="toolbar" aria-label="Viewer controls">
          <button type="button" onClick={zoomOut} aria-label="Zoom out" title="Zoom out (-)">
            −
          </button>
          <span className="zoom-readout" aria-live="polite">
            {Math.round(t.k * 100)}%
          </span>
          <button type="button" onClick={zoomIn} aria-label="Zoom in" title="Zoom in (+)">
            +
          </button>
          <button type="button" onClick={fit} aria-label="Fit image to view" title="Fit to view (0)" aria-pressed={auto}>
            Fit
          </button>
          <button type="button" onClick={actual} aria-label="Show at 100 percent" title="100% (1)">
            1:1
          </button>
        </div>
      )}
    </div>
  );
}

import { useEffect, useMemo, useState } from "react";
import { assetUrl, getExperiments, type EvalResult, type Experiments, type Frac, type GalleryEntry } from "./api";
import { maskToOverlayUrl } from "./overlay";

const MODEL_IDS = ["model_a", "model_b"] as const;
const MODEL_LABEL: Record<string, string> = { model_a: "Model A · baseline", model_b: "Model B · mask-supervised" };
const SPLIT_LABEL: Record<string, string> = { test: "Test", val: "Validation" };
const KIND_LABEL: Record<string, string> = {
  false_pass: "False passes (defect → class 6)",
  false_fail: "False fails (class 6 → defect)",
  wrong_defect_type: "Wrong defect type (still FAIL)",
};
const SIZES = ["small", "medium", "large"] as const;

const pct = (x: number | null | undefined) => (x === null || x === undefined ? "n/a" : `${(x * 100).toFixed(1)}%`);
const scorePct = (x: number) => (x >= 1 ? "100%" : x >= 0.9995 ? ">99.9%" : pct(x));
const frac = (f: Frac) => `${f.num} of ${f.den}${f.rate === null ? "" : ` (${pct(f.rate)})`}`;
const num = (x: number | null | undefined, d = 3) => (x === null || x === undefined ? "n/a" : x.toFixed(d));

function useExperiments() {
  const [data, setData] = useState<Experiments | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const ctl = new AbortController();
    setError(null);
    getExperiments(ctl.signal)
      .then(setData)
      .catch((e: Error) => e.name !== "AbortError" && setError(e.message));
    return () => ctl.abort();
  }, [tick]);
  return { data, error, reload: () => setTick((t) => t + 1) };
}

function EmptyState({ command, message, onReload }: { command: string; message?: string; onReload: () => void }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="card empty-card">
      <h2 className="card-title">No evaluation results yet</h2>
      <p className="muted">{message ?? "Train and evaluate both models to generate the comparison."}</p>
      <p className="muted">Run this in PowerShell from the wafer-inspection folder (with the virtual environment active):</p>
      <pre className="cmd" tabIndex={0} aria-label="Command to generate results">
        {command}
      </pre>
      <div className="row">
        <button
          type="button"
          className="btn"
          onClick={() => {
            void navigator.clipboard?.writeText(command).then(() => setCopied(true));
            setTimeout(() => setCopied(false), 1800);
          }}
        >
          {copied ? "Copied" : "Copy command"}
        </button>
        <button type="button" className="btn primary" onClick={onReload}>
          Reload results
        </button>
      </div>
    </div>
  );
}

function Head({ m, split, r }: { m: string; split: string; r: EvalResult | null }) {
  return (
    <th scope="col" className="num">
      <div>{MODEL_LABEL[m]}</div>
      <div className="th-sub">
        {SPLIT_LABEL[split]} split{r ? ` · n=${r.n_images}` : ""}
      </div>
    </th>
  );
}

function CompareTable({ split, res }: { split: string; res: Record<string, EvalResult | null> }) {
  const rows: [string, (r: EvalResult) => string, string?][] = [
    ["Overall accuracy", (r) => frac(r.six_class.accuracy)],
    ["Macro-F1", (r) => `${num(r.six_class.macro_f1, 4)} (classes ${r.six_class.macro_f1_classes.join(",")})`],
    [
      "Majority-class baseline (acc / macro-F1)",
      (r) => `${pct(r.six_class.majority_class_baseline.accuracy.rate)} / ${num(r.six_class.majority_class_baseline.macro_f1, 4)}`,
    ],
    ["False passes: defect → class 6", (r) => frac(r.inspection.false_pass), "danger"],
    ["False fails: class 6 → defect", (r) => frac(r.inspection.false_fail)],
    ["Wrong defect type (decision still FAIL)", (r) => frac(r.inspection.wrong_defect_type)],
    ["Defects flagged FAIL", (r) => frac(r.inspection.defects_flagged_fail)],
    [
      "Mask Dice / IoU (non-empty expert masks)",
      (r) => (r.mask ? `${num(r.mask.mean_dice)} / ${num(r.mask.mean_iou)}` : "no mask output"),
    ],
  ];
  return (
    <div className="table-wrap">
      <table>
        <caption className="sr-only">Side-by-side comparison on the {SPLIT_LABEL[split]} split</caption>
        <thead>
          <tr>
            <th scope="col">Metric</th>
            {MODEL_IDS.map((m) => (
              <Head key={m} m={m} split={split} r={res[m]} />
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(([label, fn, cls]) => (
            <tr key={label}>
              <th scope="row">{label}</th>
              {MODEL_IDS.map((m) => (
                <td key={m} className={`num mono ${cls ?? ""}`}>
                  {res[m] ? fn(res[m]!) : <span className="muted">no results</span>}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PerClassTable({ split, res }: { split: string; res: Record<string, EvalResult | null> }) {
  return (
    <div className="table-wrap">
      <table>
        <caption className="sr-only">Per-class recall and F1 on the {SPLIT_LABEL[split]} split</caption>
        <thead>
          <tr>
            <th scope="col">Source class</th>
            {MODEL_IDS.map((m) => (
              <Head key={m} m={m} split={split} r={res[m]} />
            ))}
          </tr>
        </thead>
        <tbody>
          {[1, 2, 3, 4, 5, 6].map((id) => (
            <tr key={id}>
              <th scope="row">
                Class {id}
                {id === 6 && <span className="muted"> · no visible defect</span>}
              </th>
              {MODEL_IDS.map((m) => {
                const e = res[m]?.six_class.per_class[id - 1];
                return (
                  <td key={m} className="num mono">
                    {e ? (
                      <>
                        {e.recall_text}
                        <span className="muted"> · F1 {e.f1 === null ? "n/a" : e.f1.toFixed(3)}</span>
                      </>
                    ) : (
                      <span className="muted">no results</span>
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Confusion({ m, split, r }: { m: string; split: string; r: EvalResult | null }) {
  if (!r)
    return (
      <div className="cm-card">
        <h3>{MODEL_LABEL[m]} · {SPLIT_LABEL[split]}</h3>
        <p className="muted">No results for this model and split.</p>
      </div>
    );
  const cm = r.six_class.confusion_matrix;
  const max = Math.max(1, ...cm.flat());
  return (
    <div className="cm-card">
      <h3>
        {MODEL_LABEL[m]} · {SPLIT_LABEL[split]}
      </h3>
      <table className="cm">
        <caption className="sr-only">Confusion matrix, rows true class, columns predicted class</caption>
        <thead>
          <tr>
            <th scope="col" className="corner">
              true ↓ / pred →
            </th>
            {r.six_class.confusion_matrix_order.map((c) => (
              <th scope="col" key={c}>
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {cm.map((row, i) => (
            <tr key={i}>
              <th scope="row">{r.six_class.confusion_matrix_order[i]}</th>
              {row.map((v, j) => (
                <td
                  key={j}
                  className={i === j ? "diag" : v > 0 ? "off" : ""}
                  style={{ ["--a" as string]: v / max }}
                  aria-label={`true ${r.six_class.confusion_matrix_order[i]}, predicted ${r.six_class.confusion_matrix_order[j]}: ${v}`}
                >
                  {v}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SizePanel({ split, res }: { split: string; res: Record<string, EvalResult | null> }) {
  const cut = (res.model_a ?? res.model_b)?.size_groups.cutoffs;
  return (
    <>
      {cut && (
        <p className="muted small-text">
          Defect area = expert-mask fraction of native pixels. Cutoffs from positive training images only: small ≤{" "}
          {(cut.small_max * 100).toFixed(2)}% &lt; medium ≤ {(cut.medium_max * 100).toFixed(2)}% &lt; large. Applied unchanged to{" "}
          {SPLIT_LABEL[split].toLowerCase()}.
        </p>
      )}
      <div className="size-grid">
        {(["flagged_fail", "exact_class_correct"] as const).map((key) => (
          <div key={key} className="size-card">
            <h3>{key === "flagged_fail" ? "Flagged FAIL" : "Exact class correct"}</h3>
            {SIZES.map((s) => (
              <div key={s} className="size-row">
                <div className="size-name">
                  <span className="cap">{s}</span>
                  <span className="muted"> · n={(res.model_a ?? res.model_b)?.size_groups.groups[s].n ?? 0}</span>
                </div>
                {MODEL_IDS.map((m) => {
                  const g = res[m]?.size_groups.groups[s][key];
                  return (
                    <div key={m} className="bar-line" title={`${MODEL_LABEL[m]} · ${SPLIT_LABEL[split]}`}>
                      <span className="bar-tag">{m === "model_a" ? "A" : "B"}</span>
                      <span className="bar-track" role="img" aria-label={g ? `${MODEL_LABEL[m]} ${SPLIT_LABEL[split]} ${s}: ${g.num} of ${g.den}` : "no results"}>
                        <i className={m === "model_a" ? "a" : "b"} style={{ width: `${(g?.rate ?? 0) * 100}%` }} />
                      </span>
                      <span className="bar-val mono">{g ? `${g.num}/${g.den} · ${pct(g.rate)}` : "n/a"}</span>
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        ))}
      </div>
    </>
  );
}

function Thumb({ g }: { g: GalleryEntry }) {
  const [expert, setExpert] = useState(true);
  const [pred, setPred] = useState(false);
  const [exUrl, setExUrl] = useState<string | null>(null);
  const [prUrl, setPrUrl] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    maskToOverlayUrl(assetUrl(g.expert_mask), [64, 220, 160]).then((u) => live && setExUrl(u)).catch(() => undefined);
    if (g.predicted_mask)
      maskToOverlayUrl(assetUrl(g.predicted_mask), [255, 86, 64]).then((u) => live && setPrUrl(u)).catch(() => undefined);
    return () => {
      live = false;
    };
  }, [g]);
  const id = `${g.model}-${g.kind}-${g.filename}`;
  return (
    <figure className="thumb">
      <div className="thumb-img">
        <img src={assetUrl(g.image)} alt={`SEM image ${g.filename.slice(0, 8)}, true class ${g.true_class_id}, predicted class ${g.predicted_class_id}`} />
        {expert && exUrl && <img className="ov" src={exUrl} alt="" aria-hidden="true" style={{ opacity: 0.5 }} />}
        {pred && prUrl && <img className="ov" src={prUrl} alt="" aria-hidden="true" style={{ opacity: 0.5 }} />}
      </div>
      <figcaption>
        <div className="cap-main">
          True <strong>Class {g.true_class_id}</strong> → predicted <strong>Class {g.predicted_class_id}</strong>
        </div>
        <div className="muted small-text mono">
          {g.inspection_decision} (reference {g.reference_decision}) · score {scorePct(g.predicted_class_score)}
        </div>
        <div className="toggles">
          <label>
            <input id={`${id}-e`} type="checkbox" checked={expert} onChange={(e) => setExpert(e.target.checked)} /> Expert mask
          </label>
          {g.predicted_mask && (
            <label>
              <input type="checkbox" checked={pred} onChange={(e) => setPred(e.target.checked)} /> Predicted defect
            </label>
          )}
        </div>
      </figcaption>
    </figure>
  );
}

function Gallery({ data, split }: { data: Experiments; split: string }) {
  const [model, setModel] = useState<string>("model_a");
  const [kind, setKind] = useState<string>("false_pass");
  const entries = useMemo(
    () => (data.gallery ?? []).filter((g) => g.model === model && g.kind === kind && g.split === split),
    [data, model, kind, split],
  );
  return (
    <>
      <div className="filters">
        <div className="seg" role="radiogroup" aria-label="Gallery model">
          {MODEL_IDS.map((m) => (
            <button key={m} type="button" role="radio" aria-checked={model === m} className={model === m ? "on" : ""} onClick={() => setModel(m)}>
              {MODEL_LABEL[m]}
            </button>
          ))}
        </div>
        <div className="seg" role="radiogroup" aria-label="Error type">
          {Object.entries(KIND_LABEL).map(([k, label]) => (
            <button key={k} type="button" role="radio" aria-checked={kind === k} className={kind === k ? "on" : ""} onClick={() => setKind(k)}>
              {label}
            </button>
          ))}
        </div>
      </div>
      <p className="muted small-text">
        Showing {entries.length} actual {KIND_LABEL[kind].toLowerCase()} from {MODEL_LABEL[model]} on the {SPLIT_LABEL[split].toLowerCase()} split
        (most confident first). Expert masks are shown here for evaluation only.
      </p>
      {split !== "test" ? (
        <p className="muted">The error gallery is generated from the test split. Switch to Test.</p>
      ) : entries.length === 0 ? (
        <p className="muted">No errors of this type for this model, or the model has no test results.</p>
      ) : (
        <div className="gallery">
          {entries.map((g) => (
            <Thumb key={g.image} g={g} />
          ))}
        </div>
      )}
    </>
  );
}

export default function ExperimentsView() {
  const { data, error, reload } = useExperiments();
  const [split, setSplit] = useState<"test" | "val">("test");

  if (error)
    return (
      <div className="notice error" role="alert">
        <strong>Could not load experiments.</strong> {error}
        <div>
          <button type="button" className="btn small" onClick={reload}>
            Retry
          </button>
        </div>
      </div>
    );
  if (!data) return <p className="muted" role="status">Loading evaluation artifacts…</p>;
  if (!data.available) return <EmptyState command={data.command} message={data.message} onReload={reload} />;

  const res: Record<string, EvalResult | null> = {};
  for (const m of MODEL_IDS) res[m] = data.results?.[m]?.[split] ?? null;
  const any = MODEL_IDS.some((m) => res[m]);
  const sr = data.split_report;

  return (
    <div className="experiments">
      <div className="exp-head">
        <div>
          <h1 className="h1">Experiments</h1>
          <p className="muted small-text">
            Split <span className="mono">{data.split_id}</span> · image-level split (no wafer or acquisition IDs in the dataset) · visible-defect
            decisions refer to the imaged SEM site only.
          </p>
        </div>
        <div className="seg" role="radiogroup" aria-label="Evaluation split">
          {(["test", "val"] as const).map((s) => (
            <button key={s} type="button" role="radio" aria-checked={split === s} className={split === s ? "on" : ""} onClick={() => setSplit(s)}>
              {SPLIT_LABEL[s]}
            </button>
          ))}
        </div>
      </div>

      {!any ? (
        <EmptyState command={data.command} message={`No ${SPLIT_LABEL[split]} evaluation results were found for either model.`} onReload={reload} />
      ) : (
        <>
          <section className="card">
            <h2 className="card-title">Model comparison · {SPLIT_LABEL[split]} split</h2>
            <CompareTable split={split} res={res} />
          </section>
          <section className="card">
            <h2 className="card-title">Per-class recall (raw counts) · {SPLIT_LABEL[split]} split</h2>
            <PerClassTable split={split} res={res} />
          </section>
          <section className="card">
            <h2 className="card-title">Confusion matrices · {SPLIT_LABEL[split]} split</h2>
            <div className="cm-grid">
              {MODEL_IDS.map((m) => (
                <Confusion key={m} m={m} split={split} r={res[m]} />
              ))}
            </div>
          </section>
          <section className="card">
            <h2 className="card-title">Defect size · {SPLIT_LABEL[split]} split</h2>
            <SizePanel split={split} res={res} />
          </section>
          <section className="card">
            <h2 className="card-title">Error gallery</h2>
            <Gallery data={data} split={split} />
          </section>
          {sr && (
            <section className="card">
              <h2 className="card-title">Split composition (images / distinct originals)</h2>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th scope="col">Split</th>
                      {[1, 2, 3, 4, 5, 6].map((c) => (
                        <th scope="col" className="num" key={c}>
                          Class {c}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(["train", "val", "test"] as const).map((s) => (
                      <tr key={s}>
                        <th scope="row">{s}</th>
                        {[1, 2, 3, 4, 5, 6].map((c) => (
                          <td className="num mono" key={c}>
                            {sr.counts_per_class[s][String(c)]} / {sr.distinct_originals_per_class[s][String(c)]}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}
        </>
      )}
    </div>
  );
}

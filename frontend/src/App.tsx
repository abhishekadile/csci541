import { useRef, useState, type KeyboardEvent } from "react";
import InspectionView from "./InspectionView";
import ExperimentsView from "./ExperimentsView";

type Tab = "inspection" | "experiments";
const TABS: { id: Tab; label: string }[] = [
  { id: "inspection", label: "Inspection" },
  { id: "experiments", label: "Experiments" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("inspection");
  const refs = useRef<Record<Tab, HTMLButtonElement | null>>({ inspection: null, experiments: null });

  // Roving tabindex + arrow keys, as in the WAI-ARIA tabs pattern.
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const i = TABS.findIndex((t) => t.id === tab);
    let next = i;
    if (e.key === "ArrowRight") next = (i + 1) % TABS.length;
    else if (e.key === "ArrowLeft") next = (i - 1 + TABS.length) % TABS.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = TABS.length - 1;
    else return;
    e.preventDefault();
    setTab(TABS[next].id);
    refs.current[TABS[next].id]?.focus();
  };

  return (
    <div className="app">
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <span className="brand-name">Wafer Defect Inspection</span>
          <span className="brand-sub">SEM site classification</span>
        </div>
        <div role="tablist" aria-label="Views" className="tabs" onKeyDown={onKey}>
          {TABS.map((t) => (
            <button
              key={t.id}
              ref={(el) => {
                refs.current[t.id] = el;
              }}
              role="tab"
              id={`tab-${t.id}`}
              aria-selected={tab === t.id}
              aria-controls={`panel-${t.id}`}
              tabIndex={tab === t.id ? 0 : -1}
              className={`tab${tab === t.id ? " active" : ""}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>
      </header>
      <main id="main" className="main">
        <section role="tabpanel" id="panel-inspection" aria-labelledby="tab-inspection" hidden={tab !== "inspection"}>
          <InspectionView />
        </section>
        <section role="tabpanel" id="panel-experiments" aria-labelledby="tab-experiments" hidden={tab !== "experiments"}>
          {tab === "experiments" && <ExperimentsView />}
        </section>
      </main>
    </div>
  );
}

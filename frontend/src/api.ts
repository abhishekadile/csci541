// Typed client for the FastAPI service. The UI displays exactly what the API returns:
// the PASS/FAIL decision is computed server-side (ml/decision.py), never in the browser.

export interface ClassInfo {
  id: number;
  name: string;
}

export interface ModelDetails {
  model_version: string;
  name: string;
  kind: string;
  has_mask_output: boolean;
  input_size: [number, number];
  split_id: string;
  provenance: string;
  epoch: number | null;
  checkpoint_sha256: string;
  task: string;
  val_macro_f1: number | null;
}

export interface ModelInfo {
  ready: boolean;
  reason: string | null;
  message: string;
  classes: ClassInfo[];
  decision_rule: string;
  scope: string;
  device: string;
  model: ModelDetails | null;
}

export interface MaskPayload {
  encoding: "png_base64";
  width: number;
  height: number;
  data: string;
  defect_pixels: number;
  meaning: string;
}

export type Decision = "PASS" | "FAIL";

export interface Prediction {
  predicted_class_id: number;
  predicted_class_name: string;
  predicted_class_score: number;
  class_scores: Record<string, number>; // keyed by ORIGINAL source class IDs "1".."6"
  inspection_decision: Decision;
  decision_reason: string;
  scope: string;
  model_version: string;
  model_kind: string;
  original_width: number;
  original_height: number;
  predicted_mask: MaskPayload | null;
  processing_time_ms: number;
}

export class ApiError extends Error {
  code: string;
  status: number;
  constructor(status: number, code: string, message: string) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

async function readError(res: Response): Promise<ApiError> {
  try {
    const body = await res.json();
    const d = body?.detail;
    if (d && typeof d === "object" && "code" in d) return new ApiError(res.status, d.code, d.message);
    if (typeof d === "string") return new ApiError(res.status, "http_error", d);
  } catch {
    /* fall through */
  }
  return new ApiError(res.status, "http_error", `Server responded with HTTP ${res.status}.`);
}

export async function getModel(signal?: AbortSignal): Promise<ModelInfo> {
  let res: Response;
  try {
    res = await fetch("/api/model", { signal });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError(0, "network", "Cannot reach the inspection service.");
  }
  if (!res.ok) throw await readError(res);
  return res.json();
}

export async function predict(file: File, signal?: AbortSignal): Promise<Prediction> {
  const form = new FormData();
  form.append("file", file);
  let res: Response;
  try {
    res = await fetch("/api/predict", { method: "POST", body: form, signal });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError(0, "network", "Cannot reach the inspection service.");
  }
  if (!res.ok) throw await readError(res);
  return res.json();
}

// ---- Experiments -----------------------------------------------------------------------

export interface Frac {
  num: number;
  den: number;
  rate: number | null;
}

export interface PerClass {
  class_id: number;
  class_name: string;
  support: number;
  correct: number;
  predicted_count: number;
  recall: number | null;
  precision: number | null;
  f1: number | null;
  recall_text: string;
}

export interface SizeGroup {
  n: number;
  flagged_fail: Frac;
  missed_as_pass: Frac;
  exact_class_correct: Frac;
}

export interface EvalResult {
  model: string;
  model_kind: string;
  model_version: string;
  split: "val" | "test";
  split_id: string;
  n_images: number;
  checkpoint_epoch: number | null;
  aux_weight: number;
  class_weights: number[];
  six_class: {
    per_class: PerClass[];
    accuracy: Frac;
    macro_f1: number | null;
    macro_f1_classes: number[];
    macro_f1_note: string;
    confusion_matrix: number[][];
    confusion_matrix_order: number[];
    majority_class_baseline: { predicts_class_id: number; accuracy: Frac; macro_f1: number | null };
  };
  inspection: {
    n_actual_defect: number;
    n_actual_pass: number;
    false_pass: Frac & { definition: string };
    false_fail: Frac & { definition: string };
    wrong_defect_type: Frac & { definition: string };
    defects_flagged_fail: Frac;
    decision_accuracy: Frac;
  };
  size_groups: {
    cutoffs: { small_max: number; medium_max: number; definition: string };
    groups: Record<"small" | "medium" | "large", SizeGroup>;
  };
  mask: null | {
    images_with_expert_mask: number;
    mean_dice: number | null;
    mean_iou: number | null;
    empty_expert_mask_images: number;
    empty_mask_images_with_any_predicted_pixels: Frac;
    note: string;
  };
}

export interface GalleryEntry {
  model: string;
  split: string;
  kind: "false_pass" | "false_fail" | "wrong_defect_type";
  filename: string;
  true_class_id: number;
  predicted_class_id: number;
  inspection_decision: Decision;
  reference_decision: Decision;
  predicted_class_score: number;
  class_scores: Record<string, number>;
  image: string;
  expert_mask: string;
  predicted_mask: string | null;
}

export interface Experiments {
  available: boolean;
  command: string;
  message?: string;
  split_id?: string;
  models?: Record<string, { label: string; available_splits: string[]; model_version: string | null }>;
  results?: Record<string, Record<string, EvalResult | null>>;
  gallery?: GalleryEntry[];
  split_report?: {
    counts_per_class: Record<string, Record<string, number>>;
    distinct_originals_per_class: Record<string, Record<string, number>>;
    size_group_counts: Record<string, Record<string, number>>;
    unit: string;
  } | null;
  plots?: Record<string, string>;
}

export async function getExperiments(signal?: AbortSignal): Promise<Experiments> {
  let res: Response;
  try {
    res = await fetch("/api/experiments", { signal });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError(0, "network", "Cannot reach the inspection service.");
  }
  if (!res.ok) throw await readError(res);
  return res.json();
}

export const assetUrl = (rel: string) => `/api/experiments/assets/${rel}`;

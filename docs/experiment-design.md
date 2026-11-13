# Experiment design: Wafer Defect Inspection (CSCI 441/541)

The sections follow the project requirements as given to us: research questions, dataset, preprocessing,
independent/dependent/controlled variables, models, baseline, training configuration, procedure, metrics, analysis.
Results are **not** in this file; they are generated into `artifacts/results/` and go into the report
(`docs/report-outline.md`). Nothing here states a finding.

## 1. Research questions

* **RQ1.** Does adding expert-mask supervision improve six-class macro-F1 and per-class recall compared with training the
  same full-resolution CNN on class labels alone?
* **RQ2.** As visible defect area decreases, how do the two models differ in their ability to flag a site as FAIL and to
  identify its exact defect class?

**Scope of claims.** The unit of prediction is one SEM inspection image. PASS/FAIL is a *visual inspection* decision
about the imaged site. The dataset does not establish electrical die yield or the disposition of an entire wafer, and
no result in this project should be described that way.

## 2. Dataset

Carinthia-S (Kofler and Hasic, 2025; Zenodo record 16895427, CC BY 4.0), derived from the original Carinthia dataset
(Zenodo 10696644). Verified by `python -m ml.audit` on the extracted copy:

| Property | Verified value |
|---|---|
| Images | 4,591 JPEG, 480 x 480, 8-bit grayscale; 4,591 PNG masks of the same size |
| Index | `carinthia-s.csv`, `;`-separated: `image_path;mask_path;filename;label` |
| Labels | source classes 1-6: 1 = 55, 2 = 8, 3 = 4008, 4 = 289, 5 = 4, 6 = 227 |
| Class meanings | the dataset page states only that **class 6 images contain no visible defects**; it publishes no names for classes 1-5, so the UI shows "Class N" |
| Masks | binary with anti-aliased edges (values 0/255 plus intermediate values); binarised with value > 127. 4 mask files are stored as RGB/RGBA (all black) and are converted to grayscale |
| Label/mask agreement | every class 1-5 image has a non-empty mask; 226 of 227 class-6 masks are empty, and 1 class-6 mask has 50 pixels (0.02% of the image) |
| Exact duplicates | none (MD5) |
| Near duplicates | 103 groups covering 467 images (differences of about 1.5-4 gray levels out of 255); no group mixes labels. Class 6: 181 of 227 images are in such groups, one group alone has 157 images. Class 3: 286 images are in such groups |
| Group identifiers | none: no wafer or acquisition IDs, so evaluation is **image-level** |

Near-duplicates are found with a 64-bit difference hash (Hamming distance <= 3) confirmed by a 32 x 32 thumbnail
correlation >= 0.98, then merged with union-find (`ml/audit.py`).

## 3. Preprocessing and split

* **Native resolution.** Images are never resized or cropped; tensors are 1 x 480 x 480. The only batching code is
  `pad_collate`, which zero-pads to the largest height/width in a batch and records the valid extent. With the verified
  uniform data this is a no-op. Normalization: `(pixel/255 - mean) / std`, with mean and std computed from the **training
  split only** (stored in the manifest folder and in every checkpoint).
* **Split** (`python -m ml.split`, seed 42, manifest in `artifacts/splits/<split_id>/manifest.csv`): target 70/15/15,
  stratified per class, duplicate groups never split. Because class 6 has a 157-image group that would distort a split,
  any group larger than a split's quota is placed in train. Actual counts (images / distinct originals after duplicate
  grouping):

| Split | Class 1 | Class 2 | Class 3 | Class 4 | Class 5 | Class 6 | Total |
|---|---|---|---|---|---|---|---|
| train | 39 / 39 | 6 / 6 | 2806 / 2679 | 203 / 203 | 2 / 2 | 159 / 3 | 3215 |
| val | 8 / 8 | 1 / 1 | 601 / 581 | 43 / 43 | 1 / 1 | 34 / 19 | 688 |
| test | 8 / 8 | 1 / 1 | 601 / 559 | 43 / 43 | 1 / 1 | 34 / 30 | 688 |

* **Augmentation** (train split only, after splitting): vertical flip p = 0.5, translation up to +/-16 px, brightness and
  contrast jitter +/-10%. Horizontal flip is off by default because every image has a fixed 3-pixel black border on the right.
  Image and mask are transformed together; flips and integer shifts move masks without interpolation (nearest by construction).
* **Defect area and size groups.** Area = fraction of native-resolution pixels covered by the expert mask. Small/medium/large
  cutoffs are the 33.3rd and 66.7th percentiles of area over **positive (class 1-5) training images only**: small <= 1.354% <
  medium <= 2.288% < large. Saved in `size_cutoffs.json` and applied unchanged to validation and test. Positive images per
  group: train 1019/1018/1019, val 218/219/217, test 207/217/230.

## 4. Variables

| Kind | Variable |
|---|---|
| **Independent** | mask supervision: Model A (`aux_weight = 0`, labels only) vs Model B (`aux_weight = 1.0`, labels + expert masks, plus a segmentation decoder) |
| **Dependent** | six-class macro-F1; per-class recall and F1; accuracy; false-pass, false-fail and wrong-defect-type rates; % flagged FAIL and % exact class by size group; Dice/IoU (Model B only, descriptive) |
| **Controlled** | split manifest; normalization; class weights (computed from the train split only, identical for both); classification loss (class-weighted cross-entropy); augmentation; optimizer and learning rate schedule; batch size; maximum epochs and early-stopping patience; model-selection rule; random seed; input resolution; decision rule |

## 5. Models

* **Model A (baseline).** `torchvision` ResNet-18 architecture, `weights=None` (random initialization; nothing is downloaded),
  first convolution changed to 1 input channel, final layer with 6 logits, trained with multiclass cross-entropy. 11.17 M parameters.
* **Model B (mask-supervised).** Identical encoder and 6-logit head, plus a lightweight U-Net-style decoder using skip
  connections from the stride 2/4/8/16/32 encoder features and, in the last stage, the raw input image, so the mask logits
  are produced at full 480 x 480 resolution (small defects stay representable). 12.52 M parameters. Loss =
  class-weighted cross-entropy + `aux_weight` x (BCE + soft Dice on the mask), `aux_weight = 1.0` (explicit in `configs/model_b.yaml`).
  At inference only the image is needed; B returns six scores and a predicted mask.
* **Decision rule** (`ml/decision.py`): softmax over six logits, take the highest-scoring class (original source ID),
  then class 6 -> PASS, classes 1-5 -> FAIL. The same function is used in evaluation, CLI and API, so the class and the
  decision always agree. There is no binary head and no threshold; older binary checkpoints are rejected.
* **Label mapping.** Source IDs 1-6 map to indices 0-5 for cross-entropy; the mapping is stored in every checkpoint and
  validated at load, and predictions are converted back to source IDs before any metric, display or API response.

## 6. Training configuration

Common to both models (`configs/common.yaml`): seed 42; batch size 8 (4 or 2 if memory is short, same resolution);
AdamW, learning rate 1e-3, weight decay 1e-4, cosine schedule over the epoch budget; at most 50 epochs; early stopping with
patience 10 on the selection metric; gradient clipping 5.0; FP16 autocast on CUDA (CPU fallback for smoke tests and
inference); `num_workers = 0` (Windows-safe); class weighting `sqrt_inverse` from train counts (normalized so the mean
weight over training samples is 1; the recorded weights are `[5.724, 14.594, 0.675, 2.509, 25.277, 2.835]` for classes 1-6);
**selection rule: highest validation macro-F1, ties broken by lower validation cross-entropy.** Best and latest
checkpoints (`best.pt`, `latest.pt`) and resumable state are written to `weight/<model>/`; the per-epoch CSV log, history, run summary and evaluation results go to `artifacts/runs/<model>/`.
Each checkpoint stores the model config, normalization, label mapping, class weights, split ID, decision rule and training settings.

## 7. Hardware measurements (RTX 2080, 8 GB, batch 8, 480 x 480, FP16, `python -m ml.benchmark`)

| Model | Train img/s | Val img/s | Peak GPU memory | Estimated 50-epoch run (upper bound) |
|---|---|---|---|---|
| A | 110 | 160 | 614 MB | about 28 min |
| B | 46 | 87 | 1576 MB | about 65 min |

A full real Model A epoch (train + validation) measured **42 s** with `num_workers = 0`, and **99 s** with
`num_workers = 4` (worker start-up dominates on Windows), so 0 is the default. Estimates assume the PC is otherwise idle; epochs
were about 2.5x slower while other heavy programs ran. These are timings, not results; the benchmark trains nothing that is kept.

## 8. Procedure

1. `python -m ml.audit`; `python -m ml.split` (shared manifest).
2. `python -m ml.smoke_test` (plumbing only) and `python -m ml.benchmark`.
3. `python -m ml.run_all`: trains A then B under identical settings; evaluates validation, then evaluates the **test** split once per frozen checkpoint (`best.pt`, hashed in the result file; re-evaluation needs `--force`); builds the comparison.
4. Report results from `artifacts/results/comparison.md`, plots and the error gallery; inspect errors in the Experiments view.

## 9. Metrics (every metric is tagged with model and split)

**Six-class identification** (source IDs): 6 x 6 confusion matrix (rows true, columns predicted); per-class support, recall
(shown as "k of n correct"), precision, F1; macro-F1; overall accuracy; the majority-class baseline (always predict the most
frequent *train* class). A class with no examples in a split is marked unavailable; **macro-F1 is the unweighted mean of
per-class F1 over classes with at least one true example in that split**, with F1 = 2TP/(2TP+FP+FN) so a present class that is
never predicted counts as 0.

**Inspection decision** (reference PASS/FAIL derived from the true label with the same rule):
* *False pass*: actual class 1-5 predicted as class 6, divided by all actual defect images.
* *False fail*: actual class 6 predicted as any of 1-5, divided by all actual class-6 images.
* *Wrong defect type*: an actual defect predicted as a **different defect** class (the decision is still FAIL), divided by all actual defect images; reported separately.
* Raw numerators and denominators are always shown; a rate is "n/a" when its denominator is 0.

**Size analysis** (positive images only): for small/medium/large, % flagged FAIL (predicted class != 6) and % assigned the exact correct class, with raw counts.

**Mask quality** (Model B): Dice and IoU over images whose expert mask is non-empty; images with an empty expert mask are reported
separately (how many have any predicted pixels). Masks are thresholded at probability 0.5.

## 10. Analysis plan

* RQ1: compare macro-F1, accuracy and per-class recall (with raw counts) of A and B on the test split, and report validation
  alongside. Also report false passes, false fails and wrong-type counts.
* RQ2: compare "flagged FAIL" and "exact class" rates across the three size groups, and which model misses fewer small defects.
* The analysis is **descriptive**. We do not run significance tests: classes 2 and 5 have 1 test image each and one seed is one
  sample. Any claim about a difference must be tied to the raw counts and qualified accordingly.

## 11. Known limitations and threats to validity

* **Image-level evaluation.** No wafer or acquisition IDs exist, so images from the same wafer may sit in different splits. Near-duplicates are grouped, but visually similar non-duplicates can still straddle splits.
* **Tiny classes.** Class 2 (8 images) and class 5 (4 images) have 1 test image each, so their recall is 0 or 100% and macro-F1 (which weights all six classes equally) is very noisy.
* **Class 6 diversity.** The training class-6 images come from only 3 distinct clusters (159 images), while test class 6 spans 30. Class-6 performance on test measures generalization to differently looking "no visible defect" images; false-fail rates have a denominator of 34.
* **One label per image, one split, one seed.** No confidence intervals are computed.
* **Class names.** The dataset documents only class 6; classes 1-5 are not named here.
* **Single production layer.** The images come from one production layer of unstructured wafers; transfer to other layers or tools is untested.

## 12. Sources identified so far

* Kofler, C., and Hasic, V. (2025). *Carinthia-S dataset.* Zenodo. doi:10.5281/zenodo.16895427 (CC BY 4.0).
* Kofler, C. et al. (2024). *Carinthia dataset.* Zenodo. doi:10.5281/zenodo.10696644 (original dataset from which Carinthia-S is derived).

The team should add any other references it actually reads; none have been added here.

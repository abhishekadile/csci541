# Final report outline

Fill each placeholder with real outputs from `artifacts/results/comparison.md`, `artifacts/runs/*/eval_*.json`
and your own work. Do not paste numbers from the smoke test or the benchmark as results. Nothing below is pre-filled with
findings, citations, names or contributions on purpose.

## 1. Title and team
* Title: Wafer Defect Inspection: six-class SEM classification with a derived PASS/FAIL visual decision
* Team members and course section: **[TEAM TO FILL IN]**

## 2. Abstract (150-200 words)
* **[WRITE AFTER RESULTS]** Task, data, two models, headline comparison (macro-F1 and false passes), main caveat.

## 3. Introduction and motivation
* Why automated visual review of SEM inspection images matters.
* State the scope precisely: visual inspection of an imaged site; **not** die yield or wafer disposition.
* Research questions RQ1 and RQ2 (verbatim from `docs/experiment-design.md`).

## 4. Related work
* **[CITATIONS TO BE ADDED BY THE TEAM FROM SOURCES THEY HAVE READ]** (the dataset record and paper are listed in section 11
  of `docs/experiment-design.md` as the only sources already identified; add anything else you actually read).

## 5. Dataset
* Source: Carinthia-S (Kofler and Hasic, Zenodo 16895427); 4,591 images, expert masks, six classes; class 6 = no visible defects.
* Audit results (paste the `python -m ml.audit` output): size, class counts, mask encoding, label/mask disagreements, duplicates.
* Split: paste `split_report.json` counts (images / distinct originals per class) and the size-group counts.
* Limitations: no wafer IDs, near-duplicate clusters in classes 3 and 6, tiny classes 2 and 5.

## 6. Methods
* Preprocessing and augmentation (native 480 x 480, train-only augmentation, masks moved identically).
* Model A, Model B (architecture figure), loss, class weights (paste the recorded values), auxiliary weight.
* Decision rule (class 6 -> PASS, 1-5 -> FAIL) and why it is applied to the original source ID.
* Training configuration and hardware (RTX 2080, FP16, batch size actually used, epochs run, early-stopping epoch).
* Model selection rule and the frozen-checkpoint test protocol.

## 7. Results (all numbers from the artifacts; label every table with model and split)
| Item | Source file | Where it goes |
|---|---|---|
| Six-class confusion matrices (A and B) | `confusion_test.png`, `eval_test.json` | **[INSERT]** |
| Per-class support, recall (k of n), F1, macro-F1, accuracy, majority baseline | `comparison.md` | **[INSERT]** |
| False passes / false fails / wrong defect type (with numerators and denominators) | `eval_test.json` | **[INSERT]** |
| Recall by defect size: % flagged FAIL and % exact class, with raw counts | `recall_by_size_test.png` | **[INSERT]** |
| Mask Dice / IoU (Model B), empty-mask handling | `eval_test.json` | **[INSERT]** |
| Training curves | `artifacts/runs/*/log.csv` | **[INSERT]** |
| Error gallery (actual misses and wrong types) | Experiments view | **[INSERT]** |

## 8. Analysis
* **RQ1**: macro-F1 and per-class recall, A vs B. Say plainly which differences are within noise given support of 1 test image for classes 2 and 5.
* **RQ2**: flagged-FAIL and exact-class rates across small/medium/large; which model misses fewer small defects.
* Error analysis from the gallery. Distinguish false passes (the costly error) from wrong defect type (decision still FAIL).
* Threats to validity (section 9 of the design doc) and anything that surprised you.

## 9. System implementation
* Architecture diagram (training, checkpoint, API, frontend), single-origin deployment, Docker CPU image.
* Screenshots of the Inspection view (PASS and FAIL cases) and the Experiments view.
* Testing: what `pytest` covers and what was verified with trained weights versus fixtures.

## 10. Conclusion and future work
* **[WRITE AFTER RESULTS]**. Candidate extensions to evaluate only if time permits: pretrained comparison, multiple seeds, a custom GPU kernel.

## 11. Team contributions
| Member | Contribution (actual work performed) |
|---|---|
| **[NAME]** | **[DESCRIBE]** |
| **[NAME]** | **[DESCRIBE]** |

## 12. References
* **[TEAM TO FILL IN]**

## Appendix
* Commands used (copy from the README), environment (`uv pip freeze`, GPU, driver), seeds, run directories.

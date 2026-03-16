# Thesis-to-Book Conversion Strategy
## Autonomous USV Navigation — IIUM Press Research Book

---

## 1. Source Material Summary

### Thesis 1 — Muhammad Aiman Bin Norazaruddin (2026)
| Field | Detail |
|---|---|
| Title | A Mid-Level LiDAR-Stereo Fusion Framework for Unmanned Surface Vessel Geospatial Awareness in Near-Shore Environments |
| Degree | PhD in Engineering, Kulliyyah of Engineering, IIUM |
| Supervisor | Assoc. Prof. Dr Zulkifli Bin Zainal Abidin |
| Pages | 173 |
| Core Contribution | M-ACF Net — Maritime Adaptive Confidence Fusion Network (mid-level feature fusion of LiDAR + stereo vision with dynamic confidence-based weighting, PCA + K-Means clustering) |
| Platform | Suraya Riptide USV (IIUM CUTe) |
| Key Methods | YOLOv8, Pseudo-LiDAR, PCA, K-Means, GNC integration |

### Thesis 2 — Yousef A Y Alhattab (2025)
| Field | Detail |
|---|---|
| Title | Enhanced Situational Awareness for Unmanned Surface Vehicles Using LiDAR-Camera Fusion with Hybrid KD-Accelerated Spatial Clustering and Supervised Learning |
| Degree | PhD in Engineering, Kulliyyah of Engineering, IIUM |
| Supervisor | Assoc. Prof. Dr Zulkifli Bin Zainal Abidin |
| Pages | 264 |
| Core Contribution | ESA system — Enhanced Situational Awareness with decision-level LiDAR-camera fusion, hybrid KD-accelerated Euclidean clustering, MFLD unified dataset, 99% operational accuracy |
| Platform | Suraya USV family (Surveyor v2, Suraya-I, Suraya-II, Suraya Riptide) |
| Key Methods | YOLOv5s, VIP Module, MOOS-IvP, NMEA reporting, Malaysia Patent PI2023001845 |

### Thematic Link
Both theses emerge from the same research group (Kulliyyah of Engineering, IIUM, supervisor Dr. Zulkifli) and address the same problem — autonomous USV navigation in cluttered near-shore maritime environments using LiDAR-camera sensor fusion. Together they represent an **evolutionary arc**:
- Alhattab (2025): decision-level fusion, established SURAYA platform, ESA framework, real-world field-tested system
- Aiman (2026): mid-level fusion (deeper integration), M-ACF Net, advanced confidence weighting, extended same platform

This makes them ideally suited for an **edited Research Book** showing two complementary approaches to the same challenge.

---

## 2. Publisher Requirements (IIUM Press)

### Book Category
**Research Book** (presents research findings based on doctoral research) — or **Edited Book** (two distinct authors, one thematic framework).

> Since the two works are from different researchers with different methods/models but the same research group and topic, the correct IIUM Press category is **Edited Book with a Research Book structure**.

### Hard Requirements
| Item | Requirement |
|---|---|
| Minimum chapters | 6 chapters |
| Words per chapter | 8,000 – 10,000 words |
| Pages per chapter | 20 – 30 pages |
| Total pages | 200 – 250 pages (recommended) |
| Editors | Max 2 editors; each must contribute at least 1 chapter |
| Authors per chapter | Max 4 authors |
| Page size | 6" × 9" |
| Margins | 1 inch (all sides) |
| Font | Times New Roman |
| Body text size | 11 pt |
| Spacing | Single |
| Heading levels | Max 3 levels (X.1 → X.1.1 → X.1.1.1) |
| Citation style | APA 7th Edition |
| Figures/Tables | 9–10 pt, double-numerated (e.g., Figure 3.2) |
| Language | British English |
| Plagiarism | Turnitin similarity < 20% |
| Index | Compulsory (nouns/noun phrases only) |
| Blurb | 200–300 words (synopsis + author bios) |
| ISBN | Required (applied by IIUM Press) |

### Mandatory Front Matter (in order, Roman numeral pages)
1. Half title page
2. Title page (title, subtitle, editor names, institution)
3. Table of contents
4. List of figures
5. List of tables
6. List of abbreviations/symbols (merged from both theses)
7. Foreword *(optional — recommend requesting from a senior maritime robotics expert)*
8. **Preface** *(compulsory — converted from thesis abstracts + book rationale)*
9. Acknowledgements *(merged from both theses, shortened)*

### Mandatory Back Matter
1. References (consolidated, APA 7th, only cited works)
2. Glossary *(recommended — both theses use heavy domain jargon)*
3. **Index** *(compulsory)*
4. Author biodata (for both researchers + editors)

---

## 3. Thesis → Book: Key Structural Transformations

| Thesis Element | Book Treatment |
|---|---|
| Abstract (EN/Arabic/Malay) | Remove — use English abstract content to build Preface |
| Approval page | Remove entirely |
| Declaration page | Remove entirely |
| Copyright page | Replace with standard IIUM Press copyright page |
| Dedication | Remove (or incorporate as a short note in Acknowledgements) |
| Problem Statement (standalone section) | Fold into Chapter 1 Introduction narrative |
| Hypothesis section | Fold into Introduction or relevant research chapter |
| Objectives (list) | Convert to paragraph form; include in Introduction |
| Detailed methodology chapter | Condense — retain design rationale and key decisions; remove low-level step-by-step details; methodology sub-sections become a short sub-section within each research chapter |
| Wide literature review (Chapters 2) | Merge both theses' literature reviews into 2 focused background chapters (Chapters 2 and 3 of the book) |
| 5-level heading hierarchy | Reduce to max 3 levels |
| Repeated content across both theses | Identify and eliminate duplication (especially in USV background and sensor descriptions); canonical versions appear once in background chapters |
| Full-paragraph quotations | Replace with paraphrased text + citation |
| Appendices / supplementary code | Move to online repository or omit |
| Bibliography (end of thesis) | Merge into single consolidated References list; remove uncited entries; verify all are APA 7th |
| All figures (thesis-formatted) | Renumber using book double-numeration (Figure 2.1, Figure 2.2 …); verify copyright for any figures sourced from other publications |
| Thesis-specific preamble language ("This thesis presents…") | Rewrite as book chapter opening ("This chapter presents…") |

---

## 4. Proposed Book Structure

### Book Title (Recommended)
> **Autonomous Navigation and Situational Awareness for Unmanned Surface Vessels:
> LiDAR-Camera Sensor Fusion Approaches in Near-Shore Maritime Environments**

### Subtitle
> *Doctoral Research from the Kulliyyah of Engineering, International Islamic University Malaysia*

### Editors
> Assoc. Prof. Dr Zulkifli Bin Zainal Abidin *(lead editor, supervisor of both researchers)*
> + one co-editor (e.g., Dr Hasan Firdaus Bin Mohd Zaki, co-supervisor on both theses)

---

### Chapter Map

```
FRONT MATTER
  Half Title
  Title Page
  Table of Contents
  List of Figures
  List of Tables
  List of Abbreviations and Symbols
  Foreword (optional)
  Preface
  Acknowledgements

BODY

Chapter 1: Introduction — Autonomous Maritime Navigation: Challenges, Opportunities,
           and Scope of This Book
  [Written by the editors — sets the stage, introduces both research streams,
   presents book structure. ~25 pages.]
  Sources: Intro sections of both theses + editors' framing text.

Chapter 2: Unmanned Surface Vehicles — Platforms, Navigation Architectures, and
           Near-Shore Operational Challenges
  [Merged and condensed literature from both theses Ch. 2.
   Covers: USV evolution, SURAYA family, GNC architecture, near-shore
   environmental challenges (occlusion, reflections, wave motion, domain shift).
   ~28 pages.]
  Primary source: Alhattab Ch. 2 (more comprehensive USV survey) + Aiman Ch. 2.

Chapter 3: Sensor Modalities and Fusion Strategies for Maritime Perception
  [Merged literature from both theses on sensors and fusion.
   Covers: LiDAR, stereo camera, radar, fusion taxonomy (early/mid/late),
   deep learning for multimodal perception, maritime datasets (SMD, SeaShips, MFLD).
   ~28 pages.]
  Primary source: Both theses Ch. 2 sensor and fusion review sections.

Chapter 4: A Practical Navigation System for Enhanced Situational Awareness
           Using Decision-Level LiDAR-Camera Fusion
  [Alhattab's primary contribution.
   Covers: ESA framework (perception/comprehension/projection), SURAYA system
   architecture, VIP module, sensor layers, KD-accelerated Euclidean clustering,
   YOLOv5s object detection, MFLD dataset, NMEA reporting, MOOS-IvP integration,
   field test results (99% accuracy), collision avoidance scenarios.
   ~35 pages.]
  Source: Alhattab Ch. 3 + Ch. 4 + Ch. 5 (condensed, findings-focused).

Chapter 5: Mid-Level LiDAR-Stereo Geospatial Feature Fusion with Adaptive
           Confidence Weighting: The M-ACF Net Framework
  [Aiman's primary contribution.
   Covers: Suraya Riptide platform and sensor setup, M-ACF Net architecture,
   pseudo-LiDAR generation, YOLOv8 detection, dynamic confidence weighting,
   PCA + K-Means clustering, dataset collection (multi-modal, multi-environment),
   results (obstacle detection, ranging accuracy, GNC integration, navigation trials).
   ~35 pages.]
  Source: Aiman Ch. 3 + Ch. 4 (condensed, findings-focused).

Chapter 6: Comparative Analysis — Decision-Level vs Mid-Level Fusion in
           Near-Shore USV Navigation
  [New analytical chapter synthesizing both works.
   Covers: Side-by-side comparison of ESA (Alhattab) and M-ACF Net (Aiman)
   on detection accuracy, computational cost, latency, robustness to environmental
   conditions, ease of deployment; cross-dataset evaluation; lessons learned;
   design guidelines for practitioners.
   ~25 pages.]
  Source: Results chapters of both theses + new editorial analysis.

Chapter 7: Conclusions and Future Directions
  [Synthesising conclusions chapter.
   Covers: Summary of contributions, open challenges (lightweight models,
   GNSS-denied environments, multi-USV cooperation, real-time edge AI),
   recommended future research directions.
   ~18 pages.]
  Source: Conclusion sections of both theses + editorial extension.

BACK MATTER
  References (consolidated APA 7th)
  Glossary
  Index
  Author Biodata
```

---

## 5. Word Count & Page Estimates

| Chapter | Estimated Words | Estimated Pages (6×9, 11pt, single) |
|---|---|---|
| Ch 1 — Introduction | 9,000 | 25 |
| Ch 2 — USV Platforms & Challenges | 10,000 | 28 |
| Ch 3 — Sensors & Fusion Strategies | 10,000 | 28 |
| Ch 4 — Alhattab ESA System | 12,000 | 35 |
| Ch 5 — Aiman M-ACF Net | 12,000 | 35 |
| Ch 6 — Comparative Analysis | 9,000 | 25 |
| Ch 7 — Conclusions | 6,500 | 18 |
| Front matter | — | ~20 |
| Back matter (Refs, Glossary, Index) | — | ~25 |
| **Total** | **~68,500** | **~239 pages** |

This falls within the IIUM Press recommended range of 200–250 pages. ✓

---

## 6. Content Rewriting Guidelines (AI Pipeline)

### Rewrite Level: `medium`
Preserve technical accuracy and quantitative results exactly. Restructure for book prose flow.
Do NOT change any reported metric (accuracy %, mAP scores, RMSE values, etc.).

### Specific Instructions per Chapter Type

**Background chapters (Ch 2 & 3):**
- Eliminate redundant coverage between the two theses' literature reviews
- Retain only the most authoritative citations; remove marginal name-dropping
- Convert passive thesis voice ("it was found that…") to active scholarly voice
- Group related studies thematically, not chronologically

**Research chapters (Ch 4 & 5):**
- Open with a clear statement of the research contribution
- Condense methodology to the "what and why" — remove step-by-step procedural detail
- Lead with results; use discussion to interpret
- All tables and figures retain their original data; only formatting and captions are revised
- Convert "this thesis…" / "the proposed system…" → "this chapter…" / "the system…"

**Synthesis chapters (Ch 6 & 7):**
- Draw direct cross-references between both systems using consistent notation
- Quantitative comparisons must cite specific tables from Chapters 4 and 5
- Future directions should be grounded in the specific limitations identified in both systems

### Language & Style
- British English throughout (e.g., "utilise", "analyse", "recognise")
- Avoid first-person singular; use "this study", "the authors", or "the research"
- Define all abbreviations at first use within each chapter (even if defined globally)
- Limit subsection depth to 3 levels: X.1 → X.1.1 → X.1.1.1

---

## 7. Illustration & Table Policy

- **Renumber** all figures and tables using double-numeration: Figure Ch.N (e.g., Figure 4.3)
- **Verify copyright** for any figures reproduced from external papers; obtain written permission
- **Original data figures** (from authors' own experiments) are freely usable; caption as "Source: Authors"
- **Simplify** complex multi-panel figures where possible; split if needed
- **Minimum resolution** for print: 300 DPI
- Tables: 9–10 pt, single spaced, numbered (Table 4.2), with caption above the table
- Figures: 9–10 pt caption below the figure

---

## 8. Reference Consolidation Plan

Both theses use **APA 7th Edition** in-text citation style — no conversion needed.

Steps:
1. Extract all references from both theses
2. De-duplicate (same paper cited in both)
3. Verify completeness of each entry (DOI, volume, page numbers)
4. Sort alphabetically
5. Remove any thesis-internal references that are not cited in the final book chapters
6. Maintain a single References section at the end of the book (not per-chapter)

*Note: IIUM Press allows per-chapter references for edited books. Given that Chapters 4 and 5 are from different authors, consider per-chapter references for Chapters 4 and 5, and a unified list for Chapters 1, 2, 3, 6, and 7.*

---

## 9. Glossary Recommendations

Both theses share dense domain terminology. A unified glossary (~40–60 terms) should cover:

**USV / Navigation**: USV, GNC, GNSS, IMU, MRU, AIS, ESA, MOOS-IvP, NMEA, RTK, INS, EKF

**Perception / Sensing**: LiDAR, Stereo Camera, Point Cloud, Voxel, Disparity Map, Pseudo-LiDAR, Depth Estimation, SOR (Statistical Outlier Removal), Bounding Box

**Algorithms / ML**: YOLOv5s, YOLOv8, CNN, PCA, K-Means, KD-Tree, Euclidean Clustering, mAP, Precision, Recall, F1 Score, RMSE

**Fusion**: Early Fusion, Mid-Level Fusion, Late/Decision-Level Fusion, M-ACF Net, VIP Module, Confidence Weighting

**Datasets**: SMD (Singapore Maritime Dataset), SeaShips, MFLD (Maritime Fusion LiDAR-Depth)

---

## 10. Pipeline Configuration

The project uses `main.py` with `book_config.yaml`. The ready-to-run configuration
is provided in `book_config.yaml` at the root of this project directory.

### Recommended Execution

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: Dry run to verify extraction without API cost
python main.py --config book_config.yaml --dry-run

# Step 2: Full run with AI rewriting (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY="your_key_here"
python main.py --config book_config.yaml

# Output will be at: output/USV_Book_Draft.docx
```

### Estimated API Usage
- 7 chapters × avg 6 sections × ~800 words input = ~34,000 tokens input
- Output ~4,000 tokens per section × 42 sections = ~168,000 tokens output
- **Estimated total: ~200,000 tokens (Claude Opus 4.6)**

---

## 11. Editorial Checklist (Pre-Submission to IIUM Press)

- [ ] All chapters reviewed by both authors and editors
- [ ] Turnitin similarity report < 20% (each chapter and full manuscript)
- [ ] All figures at ≥ 300 DPI, saved separately (JPEG/PNG/PDF)
- [ ] Written copyright permission for all externally sourced figures
- [ ] Unified list of abbreviations covers all acronyms in all chapters
- [ ] APA 7th Edition verified for all references (use Zotero/Mendeley)
- [ ] Index covers key terms (nouns/noun phrases only; no verbs)
- [ ] Blurb written (200–300 words): synopsis + 2-line bios for each researcher
- [ ] All chapter headings use max 3 levels
- [ ] British English spell-check completed
- [ ] Page size set to 6" × 9" in final Word document
- [ ] Manuscript Submission Form completed
- [ ] Cover concept (optional) prepared for IIUM Press design team
- [ ] Foreword obtained from invited senior expert
- [ ] Editors' letter of approval from Head of Department/Dean of Kulliyyah attached

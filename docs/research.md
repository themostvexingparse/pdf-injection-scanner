# Research notes

How the scanner's design and its instruction-likeness signal were arrived at. Numbers come from the
scripts in `research/` and the notebook in `notebooks/`; "local documents" are a handful of private,
benign PDFs (course assignments, lecture slides, a software readme) that are not part of this repository.

## 1. Visibility: measure pixels, explain with drawing state

Early versions decided visibility from drawing-state heuristics (render mode, colour, font size,
glyph boxes). These rely on metrics PDF producers write inconsistently: ligature continuation glyphs,
combining marks and broken width tables all have zero-width boxes, and were reported as hidden text.

The scanner now makes every text object transparent in turn (pdfium) and re-renders the page: the
pixels that change are exactly what the object contributes to what a person sees. Objects that do not
overlap are measured in the same render, so a page costs a handful of renders. Drawing-state checks
only explain *why* something is invisible.

Bugs found on real files and fixed: PyMuPDF's raw text API splits ligature glyphs (U+FB01) unless
asked not to; several subsets of one font sharing a base name broke the font-cmap check; extractors
glue words without spaces ("thestructuralstyle", "11void" for a code line number + code), which the
extractor-disagreement and unexplained-text checks now handle by comparing whitespace-free text and
by splitting tokens into words the extractor itself returned.

## 2. OCR is a safety net, not the engine

| Stage (6-page PDF) | Time |
|---|---|
| Everything except OCR | ~1.6 s |
| Full-page OCR + faint-text OCR | ~61 s |

OCR now runs only on crops where nothing else can decide (hidden text over a scanned image: does the
OCR layer match the scan?), or on whole pages with `--ocr full`.

## 3. Redundancy must be order-aware (audit finding)

An audit (`docs/audit/`) showed a payload padded with a copy of the visible text evading the
"at most 20% new words" redundancy rule (4 new words of 91). A word-set fix would still miss a payload
built only from words that appear somewhere on the page. Hidden text is now aligned against the
visible text by 3-word runs; the stretches it adds are extracted and judged on their own.

## 4. Embedding model: speed and accuracy (bge-small-en-v1.5)

| | fp32 ONNX | int8 (dynamic) |
|---|---|---|
| Short sentence, batch 1 | 3.6 ms | 1.5 ms |
| Throughput, batch 32 | 280-340 sentences/s | 480-570 sentences/s |
| Agreement with PyTorch | exact (cos 1.0) | mean cos 0.979; 18% of nearest neighbours change |

int8 cost accuracy in downstream tests, so fp32 is used.

Findings with frozen embeddings:

- **A learned direction beats nearest-neighbour lookup.** At 1% false positives, a logistic probe
  caught 95% of held-out injections; "max cosine to a database of known injections" caught 77% and
  missed a real hidden instruction. Nearest neighbours are kept only to explain a flag.
- **No single dimension encodes "injection".** The best single dimension reaches AUC 0.75-0.93; about
  50-100 dimensions are needed for full accuracy; the mean-difference direction and the probe agree
  (cosine 0.8-0.9).
- **The benign side decides real-world false positives.** Adding chat-style public datasets raised
  false positives on real documents from 1% to 22%; with only descriptive text (Wikipedia) the probe
  learned "imperative mood = injection" (28%); adding instructions addressed to human readers fixed most of it.
- **Score short units.** Embedding an injection sentence together with 1 / 10 benign sentences cut
  detection from 97% to 68% / 32%.
- **Visible text: precision first.** At a per-sentence 1% threshold a 100-300 sentence document gets
  1-8 unrelated flags; visible sentences are therefore flagged only at the strong tier and only when
  close to a known injection (0 unrelated flags on six real documents, 60% of planted injections).

## 5. Fine-tuning

GPU notebook (Colab T4, ~20 min): public data only, 1-3 sentence windows of at most 64 tokens,
multiple-instance learning (an example is a bag of windows scored by its maximum, so the greeting
sentences of an injection email are not mislabelled as attacks).

**Run 1** (formal prose as the only benign documents): test AUC 0.999 and ~100% detection on held-out
injection sets, but 42-77% of sentences in real assignments and manuals reached the strong tier. With
every imperative text in training being an attack, the model learned "addresses the reader".

**Run 2** (+ FineWeb-Edu and C4 as real instruction-heavy benign text; four variants): flags on real
documents fell to 5-7%, still above the 1% acceptance limit with notebook-calibrated thresholds.
Comparing at *equal* false-positive rates on real documents (thresholds re-set on half of them,
measured on the other half) showed the actual trade-off:

| | held-out real PDFs | document-embedded injections | chat-style injections |
|---|---|---|---|
| base probe | 2.4% | 56% | 83% |
| fine-tuned (`ft_e2_lowlr`) | 0.2% | 79% | 17% |

**Shipped: base OR fine-tuned**, the member's thresholds re-set on local documents, applied to hidden
and unreviewed text (visible sentences use the base probe only):

| strong tier | real PDFs | hand-written / generated benign instructions | chat-style injections | document-embedded injections |
|---|---|---|---|---|
| base alone | 0.7% | 1.8% / 2.6% | 78% | 39% |
| base OR fine-tuned | 0.9% | 1.8% / 2.9% | 82% | 79% |

Lessons: never accept a model on its test-set false-positive rate alone (`research/accept_models.py`
checks real documents and hand-written instructions); recalibrate thresholds on the target document
genre; prefer adding a model as an ensemble member over replacing a calibrated one.

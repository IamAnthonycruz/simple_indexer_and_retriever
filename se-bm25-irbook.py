"""
CS4422 Assignment 2 — Search Engine: BM25+ vs TF-IDF
=====================================================
Single-file implementation covering:
  • Document parsing & text preprocessing (tokenization, stopwords, lemmatization)
  • Vocabulary building (tok2idx / idx2tok)
  • Inverted index with postings lists
  • Pickle-based persistence with FORCE_REBUILD flag
  • BM25+ scoring  (k1=1.2, b=0.75, δ=1.0)
  • TF-IDF scoring  (log-normalized TF × IDF)
  • SearchAgent class with method dispatch
  • Comparative analysis utilities
  • PDF report generation
"""

import os
import re
import math
import pickle
import statistics
import nltk
from collections import defaultdict, Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords',  quiet=True)
nltk.download('wordnet',    quiet=True)
nltk.download('punkt_tab',  quiet=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CORE CLASSES                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class SearchAgent:
    """Unified search interface supporting BM25+ and TF-IDF scoring."""

    def __init__(self, inverted_index, doc_metadata, doc_lengths,
                 avgdl, tok2idx, num_docs, method="bm25+"):
        self.inverted_index = inverted_index
        self.doc_metadata   = doc_metadata
        self.doc_lengths    = doc_lengths
        self.avgdl          = avgdl
        self.tok2idx        = tok2idx
        self.num_docs       = num_docs
        self.method         = method

    def query(self, query_string, top_k=10):
        """Return ranked list of (doc_id, score) tuples."""
        if self.method == "tfidf":
            arr = tfidf(query_string, self.inverted_index,
                        self.doc_lengths, self.avgdl,
                        self.tok2idx, self.num_docs)
        else:
            arr = bm25_plus(query_string, self.inverted_index,
                            self.doc_lengths, self.avgdl,
                            self.tok2idx, self.num_docs)
        return arr[:top_k]

    def display_results(self, query_string, results):
        """Pretty-print ranked search results."""
        print(f"\nQuery: '{query_string}'  [method: {self.method}]")
        print(f"Results found: {len(results)}")
        print("-" * 60)

        if not results:
            print("  No matching documents found.")
            return

        for rank, (doc_id, score) in enumerate(results, 1):
            meta = self.doc_metadata[doc_id]
            print(f"  Rank {rank} | Doc {doc_id} | Score: {score:.4f}")
            print(f"    URL:      {meta.get('url', 'N/A')}")
            print(f"    File:     {meta.get('filename', 'N/A')}")
            print(f"    Title:    {meta.get('title', 'N/A')}")
            print()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PARSING & PREPROCESSING                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def parse_file(file_path):
    """Parse a crawled webpage file into a key-value dictionary."""
    hash_map = {}
    with open(file_path, "r") as file:
        key, val = "", ""
        for line in file:
            line = line.strip()
            if ":" in line:
                if key:
                    hash_map[key] = val
                key, val = line.split(":", 1)
            else:
                val += " " + line
        if key:
            hash_map[key] = val
    return hash_map


def text_preprocessing(file_dict):
    """Tokenize, lowercase, remove punctuation/stopwords, lemmatize."""
    processed_dict = {}
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    for key, val in file_dict.items():
        val = val.lower()
        val = re.sub(r'[^\w\s]', "", val)
        tokens = nltk.word_tokenize(val)
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        processed_dict[key] = tokens
    return processed_dict


def build_vocabulary(preprocessed_dict):
    """Build bidirectional token↔index mappings from all preprocessed tokens."""
    all_tokens = set()
    for tokens in preprocessed_dict.values():
        all_tokens.update(tokens)
    vocab = sorted(all_tokens)
    tok2idx = {token: idx for idx, token in enumerate(vocab)}
    idx2tok = {idx: token for idx, token in enumerate(vocab)}
    return tok2idx, idx2tok


def preprocess_query(query_string):
    """Run query through the SAME pipeline as documents."""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    query = query_string.lower()
    query = re.sub(r'[^\w\s]', "", query)
    tokens = nltk.word_tokenize(query)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return tokens


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SCORING FUNCTIONS                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def bm25_plus(query_string, inverted_index, doc_lengths, avgdl,
              tok2idx, num_docs, k1=1.2, b=0.75, delta=1.0):
    """Score all documents against the query using BM25+.

    Formula per query term t:
      IDF(t) = log((N+1) / df_t)
      TF_comp = tf*(k1+1) / (tf + k1*(1 - b + b*|d|/avgdl))
      score  += IDF(t) * (TF_comp + δ)
    """
    query_tokens = preprocess_query(query_string)
    N = num_docs
    scores = {}

    for token in query_tokens:
        if token not in tok2idx:
            continue
        term_idx = tok2idx[token]
        if term_idx not in inverted_index:
            continue

        postings = inverted_index[term_idx]
        df_t = len(postings)
        idf = math.log((N + 1) / df_t)

        for doc_id, tf in postings:
            doc_len = doc_lengths[doc_id]
            numerator   = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
            tf_component = numerator / denominator
            term_score = idf * (tf_component + delta)
            scores[doc_id] = scores.get(doc_id, 0.0) + term_score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


def tfidf(query_string, inverted_index, doc_lengths, avgdl,
          tok2idx, num_docs):
    """Score all documents using log-normalized TF × IDF.

    Variant: tf_weight = 1 + log(tf),  idf = log(N / df_t)
    """
    query_tokens = preprocess_query(query_string)
    N = num_docs
    scores = {}

    for token in query_tokens:
        if token not in tok2idx:
            continue
        term_idx = tok2idx[token]
        if term_idx not in inverted_index:
            continue

        postings = inverted_index[term_idx]
        df_t = len(postings)
        idf = math.log(N / df_t)

        for doc_id, tf in postings:
            tf_weight  = 1 + math.log(tf) if tf > 0 else 0
            term_score = tf_weight * idf
            scores[doc_id] = scores.get(doc_id, 0.0) + term_score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  COMPARATIVE ANALYSIS  (Day 10)                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def top_k_overlap(ranked_a, ranked_b, k=10):
    """Jaccard overlap of the top-k doc sets from two rankings."""
    set_a = {doc_id for doc_id, _ in ranked_a[:k]}
    set_b = {doc_id for doc_id, _ in ranked_b[:k]}
    if not set_a and not set_b:
        return 1.0
    intersection = set_a & set_b
    union        = set_a | set_b
    return len(intersection) / len(union)


def rank_displacement(ranked_a, ranked_b, k=10):
    """Average rank displacement for documents appearing in both top-k lists.

    For each document in the intersection of the two top-k sets, compute
    |rank_in_A − rank_in_B| and return the mean.  Returns 0.0 if there
    is no overlap (nothing to compare).
    """
    rank_map_a = {doc_id: rank for rank, (doc_id, _) in enumerate(ranked_a[:k], 1)}
    rank_map_b = {doc_id: rank for rank, (doc_id, _) in enumerate(ranked_b[:k], 1)}
    common = set(rank_map_a) & set(rank_map_b)
    if not common:
        return 0.0
    total_disp = sum(abs(rank_map_a[d] - rank_map_b[d]) for d in common)
    return total_disp / len(common)


def score_distribution_stats(ranked):
    """Return dict of score statistics from a ranked list."""
    if not ranked:
        return {"count": 0, "max": 0, "min": 0, "mean": 0, "stdev": 0, "range": 0}
    scores = [s for _, s in ranked]
    return {
        "count": len(scores),
        "max":   round(max(scores), 4),
        "min":   round(min(scores), 4),
        "mean":  round(statistics.mean(scores), 4),
        "stdev": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
        "range": round(max(scores) - min(scores), 4),
    }


def compare_methods(bm25_agent, tfidf_agent, queries, top_k=10):
    """Run a battery of queries through both models and collect comparison data.

    Returns a list of dicts, one per query, containing:
      - query, bm25_results, tfidf_results
      - overlap (Jaccard at top_k)
      - avg_rank_displacement
      - bm25_stats, tfidf_stats  (score distribution)
      - same_top_1  (bool: do both models agree on the #1 doc?)
    """
    comparisons = []

    for q in queries:
        bm25_res  = bm25_agent.query(q, top_k=top_k)
        tfidf_res = tfidf_agent.query(q, top_k=top_k)

        overlap    = top_k_overlap(bm25_res, tfidf_res, k=top_k)
        avg_disp   = rank_displacement(bm25_res, tfidf_res, k=top_k)
        bm25_stats = score_distribution_stats(bm25_res)
        tf_stats   = score_distribution_stats(tfidf_res)

        same_top_1 = (
            bool(bm25_res) and bool(tfidf_res) and
            bm25_res[0][0] == tfidf_res[0][0]
        )

        comparisons.append({
            "query":              q,
            "bm25_results":       bm25_res,
            "tfidf_results":      tfidf_res,
            "overlap":            overlap,
            "avg_rank_disp":      avg_disp,
            "same_top_1":         same_top_1,
            "bm25_stats":         bm25_stats,
            "tfidf_stats":        tf_stats,
        })

    return comparisons


def print_comparison_report(comparisons, top_k=10):
    """Print a formatted comparison report to the console."""
    print("\n" + "=" * 70)
    print("  COMPARATIVE ANALYSIS: BM25+ vs TF-IDF")
    print("=" * 70)

    for i, comp in enumerate(comparisons, 1):
        q = comp["query"]
        print(f"\n{'─' * 70}")
        print(f"  Query {i}: '{q}'")
        print(f"{'─' * 70}")

        # Side-by-side top results
        bm = comp["bm25_results"]
        tf = comp["tfidf_results"]
        max_rows = min(top_k, max(len(bm), len(tf)))

        print(f"\n  {'Rank':<6}{'BM25+ Doc':>10}{'Score':>10}   {'TF-IDF Doc':>10}{'Score':>10}")
        print(f"  {'─'*6}{'─'*10}{'─'*10}   {'─'*10}{'─'*10}")

        for r in range(max_rows):
            bm_str = f"{bm[r][0]:>10}{bm[r][1]:>10.4f}" if r < len(bm) else " " * 20
            tf_str = f"{tf[r][0]:>10}{tf[r][1]:>10.4f}" if r < len(tf) else " " * 20
            print(f"  {r+1:<6}{bm_str}   {tf_str}")

        # Metrics
        print(f"\n  Jaccard overlap (top {top_k}):  {comp['overlap']:.2%}")
        print(f"  Avg rank displacement:        {comp['avg_rank_disp']:.2f}")
        print(f"  Agree on #1 document:         {'Yes' if comp['same_top_1'] else 'No'}")

        # Score distributions
        bs, ts = comp["bm25_stats"], comp["tfidf_stats"]
        print(f"\n  Score distribution:")
        print(f"    {'':15}{'BM25+':>12}{'TF-IDF':>12}")
        for key in ("count", "max", "min", "mean", "stdev", "range"):
            print(f"    {key:<15}{bs[key]:>12}{ts[key]:>12}")

    # Summary
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    overlaps     = [c["overlap"] for c in comparisons]
    displacements = [c["avg_rank_disp"] for c in comparisons]
    agree_count  = sum(1 for c in comparisons if c["same_top_1"])
    total        = len(comparisons)

    print(f"  Queries tested:              {total}")
    print(f"  Mean Jaccard overlap:        {statistics.mean(overlaps):.2%}")
    print(f"  Mean rank displacement:      {statistics.mean(displacements):.2f}")
    print(f"  Top-1 agreement:             {agree_count}/{total} "
          f"({agree_count/total:.0%})")
    print(f"{'=' * 70}\n")

    return {
        "num_queries":          total,
        "mean_overlap":         statistics.mean(overlaps),
        "mean_rank_disp":       statistics.mean(displacements),
        "top1_agreement_rate":  agree_count / total if total else 0,
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PDF REPORT GENERATION  (Day 11)                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def generate_report_pdf(comparisons, summary, num_docs, vocab_size, avgdl,
                        output_path="report.pdf", top_k=10):
    """Generate a PDF report summarising the implementation and comparison.

    Requires: pip install reportlab
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    )

    doc = SimpleDocTemplate(output_path, pagesize=letter,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    story  = []

    # Custom styles
    styles.add(ParagraphStyle(
        name='SectionHead',
        parent=styles['Heading2'],
        spaceAfter=6,
        textColor=HexColor('#1a1a2e'),
    ))

    def heading(text):
        story.append(Spacer(1, 12))
        story.append(Paragraph(text, styles['SectionHead']))
        story.append(Spacer(1, 4))

    def body(text):
        story.append(Paragraph(text, styles['Normal']))
        story.append(Spacer(1, 6))

    # ── Title ──
    story.append(Paragraph(
        "Search Engine Comparative Analysis: BM25+ vs TF-IDF",
        styles['Title']
    ))
    story.append(Paragraph("CS4422 — Information Retrieval", styles['Normal']))
    story.append(Spacer(1, 20))

    # ── 1. Implementation Overview ──
    heading("1. Implementation Overview")
    body(
        "This search engine is implemented as a single Python file using NLTK "
        "for text preprocessing. The pipeline consists of: document parsing from "
        "crawled webpage data, text preprocessing (lowercasing, punctuation removal, "
        "stopword filtering, lemmatization), vocabulary building with bidirectional "
        "tok2idx/idx2tok mappings, and inverted index construction with postings "
        "lists storing (doc_id, term_frequency) tuples."
    )
    body(
        f"The corpus contains <b>{num_docs}</b> documents with a vocabulary "
        f"of <b>{vocab_size}</b> unique terms after preprocessing. "
        f"The average document length is <b>{avgdl:.2f}</b> tokens. "
        "The index is persisted to disk using pickle with a FORCE_REBUILD flag "
        "for cache invalidation during development."
    )

    # ── 2. Scoring Functions ──
    heading("2. Scoring Functions")

    heading("2.1 BM25+")
    body(
        "BM25+ (Lv &amp; Zhai, 2011) is a probabilistic ranking function "
        "that extends classic BM25 by adding a lower-bound constant δ to prevent "
        "over-penalisation of long documents. Parameters used: k1=1.2, b=0.75, δ=1.0."
    )
    body(
        "IDF is computed as log((N+1) / df<sub>t</sub>), where the +1 smoothing "
        "keeps IDF positive for high-frequency terms. The TF saturation component "
        "gives diminishing returns for repeated term occurrences, and the "
        "length-normalisation factor adjusts for document length relative to avgdl."
    )

    heading("2.2 TF-IDF")
    body(
        "The TF-IDF variant uses log-normalised term frequency: "
        "tf_weight = 1 + log(tf), combined with IDF = log(N / df<sub>t</sub>). "
        "This is the standard log-normalisation row from the classic TF-IDF weighting "
        "scheme. Unlike BM25+, this variant does not incorporate document length "
        "normalisation or TF saturation beyond the logarithmic dampening."
    )

    # ── 3. Comparison Methodology ──
    heading("3. Comparison Methodology")
    body(
        f"Both scoring functions were evaluated on {summary['num_queries']} "
        f"test queries. For each query, the top {top_k} results from each method "
        "were compared using three metrics: Jaccard overlap of the top-k document "
        "sets, average rank displacement for documents appearing in both lists, "
        "and top-1 agreement (whether both methods select the same #1 document)."
    )

    # ── 4. Results per Query ──
    heading("4. Results")

    for i, comp in enumerate(comparisons, 1):
        q = comp["query"]
        body(f"<b>Query {i}: \"{q}\"</b>")

        # Build results table
        table_data = [["Rank", "BM25+ Doc", "BM25+ Score",
                        "TF-IDF Doc", "TF-IDF Score"]]
        bm, tf = comp["bm25_results"], comp["tfidf_results"]
        rows = min(top_k, max(len(bm), len(tf)))
        for r in range(rows):
            row = [str(r + 1)]
            if r < len(bm):
                row += [str(bm[r][0]), f"{bm[r][1]:.4f}"]
            else:
                row += ["—", "—"]
            if r < len(tf):
                row += [str(tf[r][0]), f"{tf[r][1]:.4f}"]
            else:
                row += ["—", "—"]
            table_data.append(row)

        t = Table(table_data, colWidths=[0.6*inch, 1*inch, 1*inch, 1*inch, 1*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2d3436')),
            ('TEXTCOLOR',  (0, 0), (-1, 0), HexColor('#ffffff')),
            ('FONTSIZE',   (0, 0), (-1, -1), 8),
            ('GRID',       (0, 0), (-1, -1), 0.5, HexColor('#b2bec3')),
            ('ALIGN',      (0, 0), (-1, -1), 'CENTER'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [HexColor('#f5f6fa'), HexColor('#ffffff')]),
        ]))
        story.append(t)
        story.append(Spacer(1, 4))

        body(
            f"Jaccard overlap: {comp['overlap']:.2%} · "
            f"Avg rank displacement: {comp['avg_rank_disp']:.2f} · "
            f"Top-1 agree: {'Yes' if comp['same_top_1'] else 'No'}"
        )
        story.append(Spacer(1, 8))

    # ── 5. Summary & Discussion ──
    story.append(PageBreak())
    heading("5. Summary")

    summary_data = [
        ["Metric", "Value"],
        ["Queries tested",         str(summary["num_queries"])],
        ["Mean Jaccard overlap",   f"{summary['mean_overlap']:.2%}"],
        ["Mean rank displacement", f"{summary['mean_rank_disp']:.2f}"],
        ["Top-1 agreement rate",   f"{summary['top1_agreement_rate']:.0%}"],
    ]
    st = Table(summary_data, colWidths=[2.5*inch, 2*inch])
    st.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2d3436')),
        ('TEXTCOLOR',  (0, 0), (-1, 0), HexColor('#ffffff')),
        ('FONTSIZE',   (0, 0), (-1, -1), 9),
        ('GRID',       (0, 0), (-1, -1), 0.5, HexColor('#b2bec3')),
        ('ALIGN',      (1, 0), (1, -1), 'CENTER'),
    ]))
    story.append(st)
    story.append(Spacer(1, 12))

    heading("6. Discussion")
    body(
        "BM25+ and TF-IDF share the same fundamental intuition — rare terms "
        "that appear frequently in a document should contribute more to relevance — "
        "but they differ in how they handle term frequency saturation and document "
        "length normalisation."
    )
    body(
        "BM25+ explicitly controls TF saturation through the k1 parameter and "
        "normalises for document length via the b parameter and avgdl. This makes "
        "it more robust when the corpus contains documents of widely varying lengths. "
        "The δ floor further ensures that long documents are not unfairly penalised."
    )
    body(
        "TF-IDF with log-normalised TF dampens the effect of high term counts "
        "but does so less aggressively than BM25+, and it applies no document length "
        "correction. As a result, longer documents that naturally accumulate more "
        "term occurrences may be ranked higher than they should be."
    )
    body(
        "In practice, the two methods often agree on the top-ranked document for "
        "well-targeted queries, but diverge more in the middle ranks and on broader "
        "queries where length normalisation plays a larger role."
    )

    heading("7. Challenges")
    body(
        "Key challenges encountered during development included: preserving document "
        "boundaries during parsing (an early flat-dict merge with update() destroyed "
        "per-document structure), ensuring postings lists store raw term frequencies "
        "rather than normalised values, and correctly wiring the num_docs attribute "
        "in the SearchAgent class (a tuple-unpacking bug initially created a local "
        "variable instead of an instance attribute)."
    )

    # Build
    doc.build(story)
    print(f"\nReport saved → {output_path}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CONSTANTS                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

PICKLE_PATH   = "index_data.pkl"
FOLDER_PATH   = r"C:\Users\cruzi\OneDrive\Documents\CS4422 Assignment 2\data"
CONTENT_FIELD = "Text"
FORCE_REBUILD = False

# Test queries for comparative analysis
TEST_QUERIES = [
    "python data structures",
    "machine learning algorithms",
    "web scraping tutorial",
    "maximum likelihood estimation",
    "database management systems",
    "neural network deep learning",
]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MAIN EXECUTION                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":

    # ── Load or Build Index ──────────────────────────────────────────────

    if os.path.exists(PICKLE_PATH) and not FORCE_REBUILD:
        print("Found index_data.pkl — loading from disk (skipping indexing)...")
        with open(PICKLE_PATH, "rb") as f:
            index_data = pickle.load(f)

        inverted_index = index_data["inverted_index"]
        doc_metadata   = index_data["doc_metadata"]
        doc_lengths    = index_data["doc_lengths"]
        avgdl          = index_data["avgdl"]
        tok2idx        = index_data["tok2idx"]
        idx2tok        = index_data["idx2tok"]
        num_docs       = index_data["num_docs"]

        print(f"Loaded index: {len(inverted_index)} terms, "
              f"{num_docs} docs, avgdl={avgdl:.2f}")

    else:
        raw_docs = []
        for filename in os.listdir(FOLDER_PATH):
            full_path = os.path.join(FOLDER_PATH, filename)
            if os.path.isfile(full_path):
                parsed = parse_file(full_path)
                parsed["filename"] = filename
                raw_docs.append(parsed)

        print(f"Loaded {len(raw_docs)} documents")

        all_preprocessed_tokens = {}
        for doc_id, doc in enumerate(raw_docs):
            text = doc.get(CONTENT_FIELD, "")
            result = text_preprocessing({CONTENT_FIELD: text})
            all_preprocessed_tokens[doc_id] = result[CONTENT_FIELD]

        tok2idx, idx2tok = build_vocabulary(all_preprocessed_tokens)
        print(f"Vocabulary size: {len(tok2idx)}")

        inverted_index = defaultdict(list)
        doc_metadata   = {}
        doc_lengths    = []

        for doc_id, doc in enumerate(raw_docs):
            tokens    = all_preprocessed_tokens[doc_id]
            tf_counts = Counter(tokens)

            for token, freq in tf_counts.items():
                if token in tok2idx:
                    term_idx = tok2idx[token]
                    inverted_index[term_idx].append((doc_id, freq))

            doc_lengths.append(len(tokens))

            doc_metadata[doc_id] = {
                "url":      doc.get("URL",          ""),
                "title":    doc.get("Title",         ""),
                "content":  doc.get(CONTENT_FIELD,   ""),
                "filename": doc.get("filename",      ""),
            }

        avgdl    = sum(doc_lengths) / len(doc_lengths)
        num_docs = len(raw_docs)

        print(f"\nIndex built:")
        print(f"  Unique terms indexed : {len(inverted_index)}")
        print(f"  Total documents      : {num_docs}")
        print(f"  avgdl                : {avgdl:.2f} tokens")

        index_data = {
            "inverted_index": dict(inverted_index),
            "doc_metadata":   doc_metadata,
            "doc_lengths":    doc_lengths,
            "avgdl":          avgdl,
            "tok2idx":        tok2idx,
            "idx2tok":        idx2tok,
            "num_docs":       num_docs,
        }

        with open(PICKLE_PATH, "wb") as f:
            pickle.dump(index_data, f)
        print("Saved → index_data.pkl")

    # ── Sanity Checks ────────────────────────────────────────────────────

    def inspect_term(term):
        if term not in tok2idx:
            print(f"  '{term}' — not in vocabulary")
            return
        idx      = tok2idx[term]
        postings = inverted_index[idx]
        top5     = sorted(postings, key=lambda x: -x[1])[:5]
        print(f"  '{term}' (idx {idx}) → {len(postings)} docs | "
              f"top 5 TF: {top5}")

    print()
    inspect_term("python")
    inspect_term("the")
    inspect_term("data")

    print(f"\n  doc_metadata sample (doc_id=0):")
    for k, v in doc_metadata[0].items():
        print(f"    {k:10}: {str(v)[:80]}")

    # ── Create Search Agents ─────────────────────────────────────────────

    bm25_agent = SearchAgent(inverted_index, doc_metadata, doc_lengths,
                             avgdl, tok2idx, num_docs, method="bm25+")

    tfidf_agent = SearchAgent(inverted_index, doc_metadata, doc_lengths,
                              avgdl, tok2idx, num_docs, method="tfidf")

    # ── Individual Search Demo ───────────────────────────────────────────

    q = "python data structures"
    bm25_agent.display_results(q, bm25_agent.query(q))
    tfidf_agent.display_results(q, tfidf_agent.query(q))

    # ── Comparative Analysis (Day 10) ────────────────────────────────────

    comparisons = compare_methods(bm25_agent, tfidf_agent, TEST_QUERIES, top_k=10)
    summary = print_comparison_report(comparisons, top_k=10)

    # ── Generate PDF Report (Day 11) ─────────────────────────────────────

    try:
        generate_report_pdf(
            comparisons, summary,
            num_docs    = num_docs,
            vocab_size  = len(tok2idx),
            avgdl       = avgdl,
            output_path = "report.pdf",
            top_k       = 10,
        )
    except ImportError:
        print("\nreportlab not installed — skipping PDF generation.")
        print("Install with: pip install reportlab")
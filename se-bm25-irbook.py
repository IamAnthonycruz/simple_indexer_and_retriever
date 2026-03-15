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


# ── Parsing & Preprocessing ──────────────────────────────────────────────────

def parse_file(file_path):
   #This parses the files into a dict
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
    #This helper function serves to take the file dict and do some preprocessing like stemming and lemitization
    processed_dict = {}
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    for key, val in file_dict.items():
        val = val.lower()
        val = re.sub(r'[^\w\s]', "", val)   # remove punctuation
        val = re.sub(r'\d+', "", val)        # remove digits
        tokens = nltk.word_tokenize(val)
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        processed_dict[key] = tokens
    return processed_dict


def build_vocabulary(preprocessed_dict):
    #Build the vocabulary
    all_tokens = set()
    for tokens in preprocessed_dict.values():
        all_tokens.update(tokens)
    vocab = sorted(all_tokens)
    tok2idx = {token: idx for idx, token in enumerate(vocab)}
    idx2tok = {idx: token for idx, token in enumerate(vocab)}
    return tok2idx, idx2tok


def preprocess_query(query_string):
    #Helper function to clean query
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    query = query_string.lower()
    query = re.sub(r'[^\w\s]', "", query)
    query = re.sub(r'\d+', "", query)
    tokens = nltk.word_tokenize(query)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return tokens


# ── Scoring Functions ────────────────────────────────────────────────────────

def bm25_plus(query_string, inverted_index, doc_lengths, avgdl,
              tok2idx, num_docs, k1=1.2, b=0.75, delta=1.0):
    #bm25 scoring func impl
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
    #tfidf impl
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


# ── Search Agent ─────────────────────────────────────────────────────────────

class SearchAgent:
    #Search agent class to have some form of centralization

    def __init__(self, inverted_index, doc_metadata, doc_lengths,
                 avgdl, tok2idx, num_docs, method="bm25+"):
        self.inverted_index = inverted_index
        self.doc_metadata   = doc_metadata
        self.doc_lengths    = doc_lengths
        self.avgdl          = avgdl
        self.tok2idx        = tok2idx
        self.num_docs       = num_docs
        self.method         = method

    def query(self, query_string, top_k=5):
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




def top_k_overlap(ranked_a, ranked_b, k=10):
    # Jaccard overlap of the top-k doc sets from two rankings.
    set_a = {doc_id for doc_id, _ in ranked_a[:k]}
    set_b = {doc_id for doc_id, _ in ranked_b[:k]}
    if not set_a and not set_b:
        return 1.0
    intersection = set_a & set_b
    union        = set_a | set_b
    return len(intersection) / len(union)


def rank_displacement(ranked_a, ranked_b, k=10):
    #avg rank
    rank_map_a = {doc_id: rank for rank, (doc_id, _) in enumerate(ranked_a[:k], 1)}
    rank_map_b = {doc_id: rank for rank, (doc_id, _) in enumerate(ranked_b[:k], 1)}
    common = set(rank_map_a) & set(rank_map_b)
    if not common:
        return 0.0
    total_disp = sum(abs(rank_map_a[d] - rank_map_b[d]) for d in common)
    return total_disp / len(common)


def score_distribution_stats(ranked):
    #Ranked output from ranked list
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
    #comparison test
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
    
    print("BM25+ vs TF-IDF")
  

    for i, comp in enumerate(comparisons, 1):
        q = comp["query"]
        print(f"\n{'─' * 70}")
        print(f"  Query {i}: '{q}'")
        print(f"{'─' * 70}")

        bm = comp["bm25_results"]
        tf = comp["tfidf_results"]
        max_rows = min(top_k, max(len(bm), len(tf)))

        print(f"\n  {'Rank':<6}{'BM25+ Doc':>10}{'Score':>10}   {'TF-IDF Doc':>10}{'Score':>10}")
        print(f"  {'─'*6}{'─'*10}{'─'*10}   {'─'*10}{'─'*10}")

        for r in range(max_rows):
            bm_str = f"{bm[r][0]:>10}{bm[r][1]:>10.4f}" if r < len(bm) else " " * 20
            tf_str = f"{tf[r][0]:>10}{tf[r][1]:>10.4f}" if r < len(tf) else " " * 20
            print(f"  {r+1:<6}{bm_str}   {tf_str}")

        print(f"\n  Jaccard overlap (top {top_k}):  {comp['overlap']:.2%}")
        print(f"  Avg rank displacement:        {comp['avg_rank_disp']:.2f}")
        print(f"  Agree on #1 document:         {'Yes' if comp['same_top_1'] else 'No'}")

        bs, ts = comp["bm25_stats"], comp["tfidf_stats"]
        print(f"\n  Score distribution:")
        print(f"    {'':15}{'BM25+':>12}{'TF-IDF':>12}")
        for key in ("count", "max", "min", "mean", "stdev", "range"):
            print(f"    {key:<15}{bs[key]:>12}{ts[key]:>12}")

    # Summary
   
    print("Summary")
    
    overlaps      = [c["overlap"] for c in comparisons]
    displacements = [c["avg_rank_disp"] for c in comparisons]
    agree_count   = sum(1 for c in comparisons if c["same_top_1"])
    total         = len(comparisons)

    print(f"  Queries tested:              {total}")
    print(f"  Mean Jaccard overlap:        {statistics.mean(overlaps):.2%}")
    print(f"  Mean rank displacement:      {statistics.mean(displacements):.2f}")
    print(f"  Top-1 agreement:             {agree_count}/{total} "
          f"({agree_count/total:.0%})")
    print(f"{'=' * 70}\n")



PICKLE_PATH   = "index_data.pkl"
FOLDER_PATH   = r"C:\Users\cruzi\OneDrive\Documents\CS4422 Assignment 2\data"
CONTENT_FIELD = "Text"
FORCE_REBUILD = False

TEST_QUERIES = [
    "research",
    "test results",
    "information about animals",
    "machine learnings",
    "web scraping tutorial",
    "information retrieval",
]




if __name__ == "__main__":
    #main func

    if os.path.exists(PICKLE_PATH) and not FORCE_REBUILD:
        print("Found index_data.pkl")
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


    bm25_agent = SearchAgent(inverted_index, doc_metadata, doc_lengths,
                             avgdl, tok2idx, num_docs, method="bm25+")

    tfidf_agent = SearchAgent(inverted_index, doc_metadata, doc_lengths,
                              avgdl, tok2idx, num_docs, method="tfidf")


    print("BM25+ Search")
    for q in ["bm25", "maximum likelihood estimation"]:
        results = bm25_agent.query(q)
        bm25_agent.display_results(q, results)


    print("\n── TF-IDF Search ─────────────────────────────────────────")
    for q in ["bm25", "maximum likelihood estimation"]:
        results = tfidf_agent.query(q)
        tfidf_agent.display_results(q, results)

    #Comparative Analysis 

    comparisons = compare_methods(bm25_agent, tfidf_agent, TEST_QUERIES, top_k=10)
    print_comparison_report(comparisons, top_k=10)
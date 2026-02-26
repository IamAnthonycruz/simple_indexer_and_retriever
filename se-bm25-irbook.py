import os
import re
import pickle
import nltk
from collections import defaultdict, Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# ── Functions ────────────────────────────────────────────────────────────────

def parse_file(file_path):
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
    all_tokens = set()
    for tokens in preprocessed_dict.values():
        all_tokens.update(tokens)
    vocab = sorted(all_tokens)
    tok2idx = {token: idx for idx, token in enumerate(vocab)}
    idx2tok = {idx: token for idx, token in enumerate(vocab)}
    return tok2idx, idx2tok

# ── Constants ─────────────────────────────────────────────────────────────────

PICKLE_PATH   = "index_data.pkl"
FOLDER_PATH   = r"C:\Users\cruzi\OneDrive\Documents\CS4422 Assignment 2\data"
CONTENT_FIELD = "Text"
FORCE_REBUILD = False   # set to True to regenerate the index from scratch

# ── Load or Build ─────────────────────────────────────────────────────────────

if os.path.exists(PICKLE_PATH) and not FORCE_REBUILD:
    # ── Fast path: load from disk ─────────────────────────────────────────
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

    print(f"Loaded index: {len(inverted_index)} terms, {num_docs} docs, avgdl={avgdl:.2f}")

else:
    # ── Slow path: build from raw files ──────────────────────────────────
    raw_docs = []

    for filename in os.listdir(FOLDER_PATH):
        full_path = os.path.join(FOLDER_PATH, filename)
        if os.path.isfile(full_path):
            parsed = parse_file(full_path)
            parsed["filename"] = filename
            raw_docs.append(parsed)

    print(f"Loaded {len(raw_docs)} documents")

    # ── Preprocess & build vocabulary ────────────────────────────────────
    all_preprocessed_tokens = {}
    for doc_id, doc in enumerate(raw_docs):
        text = doc.get(CONTENT_FIELD, "")
        result = text_preprocessing({CONTENT_FIELD: text})
        all_preprocessed_tokens[doc_id] = result[CONTENT_FIELD]

    tok2idx, idx2tok = build_vocabulary(all_preprocessed_tokens)
    print(f"Vocabulary size: {len(tok2idx)}")

    # ── Build inverted index ──────────────────────────────────────────────
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
            "filename": doc.get("filename",      "")
        }

    avgdl   = sum(doc_lengths) / len(doc_lengths)
    num_docs = len(raw_docs)

    print(f"\nIndex built:")
    print(f"  Unique terms indexed : {len(inverted_index)}")
    print(f"  Total documents      : {num_docs}")
    print(f"  avgdl                : {avgdl:.2f} tokens")

    # ── Save to disk ──────────────────────────────────────────────────────
    index_data = {
        "inverted_index" : dict(inverted_index),
        "doc_metadata"   : doc_metadata,
        "doc_lengths"    : doc_lengths,
        "avgdl"          : avgdl,
        "tok2idx"        : tok2idx,
        "idx2tok"        : idx2tok,
        "num_docs"       : num_docs
    }

    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(index_data, f)

    print("\nSaved → index_data.pkl")

# ── Sanity checks (always run, regardless of load vs build) ──────────────────

def inspect_term(term):
    if term not in tok2idx:
        print(f"  '{term}' — not in vocabulary (expected if it's a stopword)")
        return
    idx      = tok2idx[term]
    postings = inverted_index[idx]
    top5     = sorted(postings, key=lambda x: -x[1])[:5]
    print(f"  '{term}' (idx {idx}) → {len(postings)} docs | top 5 TF: {top5}")

print()
inspect_term("python")
inspect_term("the")
inspect_term("data")

print(f"\n  doc_metadata sample (doc_id=0):")
for k, v in doc_metadata[0].items():
    print(f"    {k:10}: {str(v)[:80]}") 
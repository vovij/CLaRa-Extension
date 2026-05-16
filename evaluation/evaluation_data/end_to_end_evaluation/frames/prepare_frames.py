import json
import re
import time
import random
import wikipediaapi
from datasets import load_dataset
from tqdm import tqdm
from rank_bm25 import BM25Okapi  # pip install rank-bm25

wiki = wikipediaapi.Wikipedia(language="en", user_agent="CLaRa-FRAMES/1.0")
dataset = load_dataset("google/frames-benchmark", split="test")

TARGET_DOCS = 20  # match other datasets

def tokenize(text):
    return text.lower().split()

def fetch_wiki_summary(url):
    match = re.search(r"/wiki/(.+)$", url)
    if not match:
        return None
    title = match.group(1).replace("_", " ")
    page = wiki.page(title)
    time.sleep(0.3)
    return page.summary[:2048] if page.exists() else None

def get_distractors(question, gold_docs, n_needed, all_summaries):
    """Use BM25 to retrieve distractor docs from a pool."""
    if not all_summaries or n_needed <= 0:
        return []
    
    # Exclude gold docs
    pool = [d for d in all_summaries if d not in gold_docs]
    if not pool:
        return []
    
    # BM25 retrieval
    tokenized_pool = [tokenize(d) for d in pool]
    bm25 = BM25Okapi(tokenized_pool)
    scores = bm25.get_scores(tokenize(question))
    
    # Get top n_needed distractors
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_needed]
    return [pool[i] for i in top_indices]

# First pass — collect all gold summaries to use as distractor pool
print("First pass: collecting all Wikipedia summaries for distractor pool...")
all_summaries = []
raw_data = []

for row in tqdm(dataset):
    links = [row[f"wikipedia_link_{i}"] for i in range(1, 12)
             if row.get(f"wikipedia_link_{i}")]
    
    docs = []
    for url in links:
        summary = fetch_wiki_summary(url)
        if summary:
            docs.append(summary)
            if summary not in all_summaries:
                all_summaries.append(summary)
    
    if docs:
        raw_data.append({
            "question": row["Prompt"],
            "answer":   row["Answer"],
            "gold_docs": docs,
        })

print(f"Collected {len(all_summaries)} unique summaries from {len(raw_data)} samples")

# Second pass — pad each sample to TARGET_DOCS with BM25 distractors
print("Second pass: padding with BM25 distractors...")
with open("frames_clara.jsonl", "w") as f:
    for item in tqdm(raw_data):
        gold_docs = item["gold_docs"]
        n_gold = len(gold_docs)
        n_needed = max(0, TARGET_DOCS - n_gold)
        
        # Get distractors
        distractors = get_distractors(item["question"], gold_docs, n_needed, all_summaries)
        
        # Combine: gold docs first, then distractors, shuffle
        all_docs = gold_docs + distractors
        
        # Track which indices are gold (pos_index) before shuffling
        indices = list(range(len(all_docs)))
        random.shuffle(indices)
        shuffled_docs = [all_docs[i] for i in indices]
        pos_index = [indices.index(i) for i in range(n_gold)]  # positions of gold docs after shuffle
        
        f.write(json.dumps({
            "question":  item["question"],
            "answer":    item["answer"],
            "docs":      shuffled_docs,
            "pos_index": pos_index,    # matches other datasets
            "data_type": "qa",
        }) + "\n")

print("Done! frames_clara.jsonl written.")
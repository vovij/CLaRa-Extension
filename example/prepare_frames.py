import json
import re
import time
import wikipediaapi
from datasets import load_dataset
from tqdm import tqdm

wiki = wikipediaapi.Wikipedia(language="en", user_agent="CLaRa-FRAMES/1.0")
dataset = load_dataset("google/frames-benchmark", split="test")

with open("frames_clara.jsonl", "w") as f:
    for row in tqdm(dataset):
        links = [row[f"wikipedia_link_{i}"] for i in range(1, 12)
                 if row.get(f"wikipedia_link_{i}")]

        docs = []
        for url in links:
            match = re.search(r"/wiki/(.+)$", url)
            if not match:
                continue
            title = match.group(1).replace("_", " ")
            page = wiki.page(title)
            if page.exists():
                docs.append(page.summary[:1024])
            time.sleep(0.3)

        if not docs:
            continue

        f.write(json.dumps({
            "question":  row["Prompt"],
            "answer":    row["Answer"],
            "docs":      docs,
            "data_type": "qa",
        }) + "\n")
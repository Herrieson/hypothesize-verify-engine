import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import pathlib
import random
import os

# --- Configuration ---
SEED = 42
DISTRACTOR_DOCS_COUNT = 20000
CORPUS_OUTPUT_FILENAME = "./data/hotpotqa_corpus.jsonl"
TEST_SET_OUTPUT_FILENAME = "./data/hotpotqa_test_set.jsonl"
NUM_PROCESSORS = os.cpu_count() or 1

# --- Setup ---
random.seed(SEED)

def normalize_title(title: str) -> str:
    return title.strip().lower()

if __name__ == '__main__':
    # --- Step 1: Get Required Document Titles from HotpotQA ---
    print("Loading HotpotQA validation set...")
    hotpotqa_validation_split = load_dataset("hotpot_qa", "fullwiki", trust_remote_code=True)['validation']
    
    print("Extracting and normalizing unique supporting document titles...")
    required_titles_normalized = set()
    for item in tqdm(hotpotqa_validation_split, desc="Scanning HotpotQA"):
        for title in item['supporting_facts']['title']:
            required_titles_normalized.add(normalize_title(title))

    # --- Step 2: Select Distractor Documents ---
    print("\nLoading Wikipedia corpus...")
    wiki_corpus = load_dataset("wikimedia/wikipedia", "20231101.en", trust_remote_code=True)['train']
    
    print("Normalizing all Wikipedia titles to create the distractor pool...")
    all_wiki_titles_normalized = {normalize_title(title) for title in tqdm(wiki_corpus['title'], desc="Normalizing Wiki Titles")}
    
    distractor_pool = list(all_wiki_titles_normalized - required_titles_normalized)
    
    print(f"Randomly selecting {DISTRACTOR_DOCS_COUNT:,} distractor articles...")
    distractor_titles_normalized = set(random.sample(distractor_pool, DISTRACTOR_DOCS_COUNT))
    
    final_corpus_titles_normalized = required_titles_normalized.union(distractor_titles_normalized)

    # --- Step 3: Build and Save the Document Corpus ---
    print(f"\nBuilding the final corpus by filtering the dataset (using {NUM_PROCESSORS} processes)...")
    
    def is_in_final_corpus(example):
        return normalize_title(example['title']) in final_corpus_titles_normalized

    filtered_corpus_dataset = wiki_corpus.filter(is_in_final_corpus, num_proc=NUM_PROCESSORS)
    
    corpus_df = filtered_corpus_dataset.to_pandas()
    
    pathlib.Path(CORPUS_OUTPUT_FILENAME).parent.mkdir(parents=True, exist_ok=True)
    corpus_df.to_json(CORPUS_OUTPUT_FILENAME, orient='records', lines=True, force_ascii=False)

    # --- Step 4: Save the Corresponding HotpotQA Test Set ---
    # Since our corpus is guaranteed to contain all documents needed for the validation set,
    # we can save the entire validation set as our corresponding test file.
    print(f"\nSaving the full HotpotQA validation set as the test set...")
    test_set_df = hotpotqa_validation_split.to_pandas()
    test_set_df.to_json(TEST_SET_OUTPUT_FILENAME, orient='records', lines=True, force_ascii=False)
    
    # --- Step 5: Print Metadata Summary ---
    print("\n" + "="*50)
    print("              Data Generation Complete")
    print("="*50)
    print("\n--- Metadata Summary ---")
    print(f"Total Articles in Corpus:   {len(corpus_df):,}")
    print(f"  - Required Articles:      {len(required_titles_normalized):,}")
    print(f"  - Distractor Articles:    {len(distractor_titles_normalized):,}")
    print(f"Total Questions in Test Set: {len(test_set_df):,}")
    print("-" * 28)
    print(f"Corpus saved to:            '{CORPUS_OUTPUT_FILENAME}'")
    print(f"Test Set saved to:          '{TEST_SET_OUTPUT_FILENAME}'")
    print("="*50)
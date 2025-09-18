# First, ensure the necessary libraries are installed
# pip install openai pandas datasets tqdm nltk

import json
import random
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import nltk

# ------------------- Configuration -------------------
# To run, ensure your OpenAI API key is set as an environment variable
# e.g., export OPENAI_API_KEY='your_key_here'
client = OpenAI() 
CONTRADICTION_MODEL = "gpt-4o" # Using a powerful model is best for this task
STANDARD_TEST_SET_FILE = "./data/test_set_standard.jsonl" # Adjusted path
WIKI_CORPUS_NAME = "wikimedia/wikipedia"
WIKI_CORPUS_VERSION = "20231101.en"

# ------------------- Prompt Templates -------------------

CONTRADICTION_PROMPT_L1 = """
You are an expert at creating subtly incorrect information.
Your task is to take the following sentence and change a key fact (like a name, date, number, or specific term) to something plausible but incorrect.
Only output the single, modified sentence. Do not add explanations.

Original Sentence: "{sentence}"
Modified Sentence:
"""

CONTRADICTION_PROMPT_L2 = """
You are an expert in logical reasoning.
Based on the following fact, please generate another fact that logically contradicts the first one. The contradiction should be indirect and subtle.
For example, if the original fact is 'A is B's father', a contradictory fact could be 'A and B are not related by blood'.

Original Fact: "{sentence}"
Contradictory Fact:
"""

# --- IMPROVEMENT 3: OPTIMIZED LLM CALLS ---
# This new prompt combines two tasks into a single API call.
CONTRADICTION_PROMPT_L2_WITH_SOURCE = """
You are an expert in logical reasoning and creative writing.
Your task is to perform two steps:
1. First, based on the 'Original Fact', generate another fact that logically contradicts it. The contradiction should be indirect and subtle.
2. Second, add a fictional but plausible-sounding citation in the format (Source, Year) to the end of the contradictory fact you just created.

Only output the final, single, modified sentence with its citation.

Original Fact: "{sentence}"
Contradictory Fact with Fictional Citation:
"""

# ------------------- Core Functions -------------------

def call_llm(prompt, model, temperature=0.7):
    """A robust wrapper for making LLM calls."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=100
        )
        content = response.choices[0].message.content
        if content:
            # Clean up common artifacts from LLM responses
            return content.strip().replace('"', '')
        return None
    except Exception as e:
        print(f"LLM call failed: {e}")
        return None

class ContradictionAgent:
    def __init__(self, model=CONTRADICTION_MODEL):
        self.model = model

    def create_contradiction(self, sentence, level=1, add_fake_source=False):
        """Generates a contradictory sentence based on the specified level and options."""
        # Use the optimized, combined prompt when possible
        if level == 2 and add_fake_source:
            prompt = CONTRADICTION_PROMPT_L2_WITH_SOURCE.format(sentence=sentence)
            return call_llm(prompt, self.model, temperature=0.9)

        # Otherwise, follow the original logic
        if level == 1:
            prompt = CONTRADICTION_PROMPT_L1.format(sentence=sentence)
        else: # level 2
            prompt = CONTRADICTION_PROMPT_L2.format(sentence=sentence)
            
        contradictory_sentence = call_llm(prompt, self.model)
        
        # This part is now only needed for Level 1 with a fake source
        if contradictory_sentence and add_fake_source and level == 1:
            # Note: For L1, adding a source still requires a second call.
            # To combine them, you would need another specific prompt.
            # FAKE_SOURCE_PROMPT would be needed here if you wanted to support this.
            pass

        return contradictory_sentence

# --- IMPROVEMENT 1: RELIABLE SENTENCE SPLITTING ---
def find_meaningful_sentence(wiki_article):
    """
    Uses NLTK to safely tokenize sentences and find a suitable one.
    """
    if not wiki_article or 'text' not in wiki_article:
        return None
    # Use nltk's recommended sentence tokenizer
    sentences = nltk.sent_tokenize(wiki_article['text'])
    # Filter for sentences that are likely to contain a complete fact
    meaningful_sentences = [s.strip() for s in sentences if 15 < len(s.split()) < 100]
    
    if not meaningful_sentences:
        return None
    
    return random.choice(meaningful_sentences)

# ------------------- Main Flow -------------------

if __name__ == "__main__":
    # Download the NLTK sentence tokenizer model if not already present
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK's 'punkt' model for sentence tokenization...")
        nltk.download('punkt')

    # Load data
    print("Loading Wikipedia corpus...")
    wiki_corpus = load_dataset(WIKI_CORPUS_NAME, WIKI_CORPUS_VERSION, split='train')
    print("Loading standard test set...")
    standard_test_set = pd.read_json(STANDARD_TEST_SET_FILE, lines=True)

    # --- IMPROVEMENT 2: EFFICIENT ARTICLE LOOKUP ---
    # Build a title-to-article index once for instantaneous lookups.
    print("Building a title-to-article index for fast lookups...")
    title_to_article_map = {item['title']: item for item in tqdm(wiki_corpus, desc="Indexing articles")}
    print("Index built.")

    agent = ContradictionAgent()
    
    # Demonstrate by generating contradictions for the first test sample
    sample = standard_test_set.iloc[0]
    supporting_titles = sample['supporting_facts_titles']
    
    print(f"\n--- Generating Contradictions for Question ID: {sample['id']} ---")
    
    title = supporting_titles[0]
    # Use the fast index instead of the slow .filter() method
    article = title_to_article_map.get(title)
    
    if article:
        original_sentence = find_meaningful_sentence(article)
        
        if original_sentence:
            print(f"Original Sentence: {original_sentence}")

            # Generate Level 1 contradiction
            print("\n[Strategy B1: Direct Contradiction]")
            contradiction_l1 = agent.create_contradiction(original_sentence, level=1)
            print(f"Poisoned L1: {contradiction_l1}")

            # Generate Level 2 contradiction with a fake source (in one API call)
            print("\n[Strategy B2: Indirect Contradiction with Fake Source]")
            contradiction_l2_sourced = agent.create_contradiction(original_sentence, level=2, add_fake_source=True)
            print(f"Poisoned L2: {contradiction_l2_sourced}")
        else:
            print(f"Could not find a suitable sentence in the article '{title}'.")
    else:
        print(f"Article '{title}' not found in the index.")
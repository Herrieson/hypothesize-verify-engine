import pandas as pd
from tqdm import tqdm
import torch
import os
import pickle # B_NEW: For saving and loading the BM25 retriever

# LangChain core components
from langchain_core.documents import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever

# Supporting LangChain components
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ------------------- Configuration -------------------
CORPUS_FILE = "./data/hotpotqa_corpus.jsonl"
TEST_SET_FILE = "./data/hotpotqa_test_set.jsonl"
# EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHROMA_DB_PATH = "./data/chroma_db_hotpotqa_lc"
COLLECTION_NAME = "hotpotqa_corpus_lc"
BM25_INDEX_PATH = "./data/bm25_retriever.pkl" # B_NEW: Path to save the serialized BM25 index

# ------------------- Main Flow -------------------

if __name__ == '__main__':
    # --- Verify Device and Setup Models ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Running on device: {device} ---")
    if device == "cpu":
        print("Warning: CUDA not available, falling back to CPU. This will be very slow.")

    embedding_function = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': device}
    )

    # --- Setup ChromaDB and Load or Build the Index ---
    vectorstore = None
    if os.path.exists(CHROMA_DB_PATH):
        print(f"Loading existing index from ChromaDB: {CHROMA_DB_PATH}")
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embedding_function,
            collection_name=COLLECTION_NAME
        )
    else:
        print(f"No index found. Building new index from: {CORPUS_FILE}")
        corpus_df = pd.read_json(CORPUS_FILE, lines=True)
        documents = [
            Document(page_content=row['text'], metadata={"title": row['title']})
            for _, row in tqdm(corpus_df.iterrows(), total=len(corpus_df), desc="Creating LangChain Documents")
        ]
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
        chunked_documents = text_splitter.split_documents(documents)

        print("Building new index and storing in ChromaDB (this will take a while)...")
        
        # Initialize an empty Chroma vector store first
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embedding_function,
            collection_name=COLLECTION_NAME
        )
        
        # Add documents in batches with a progress bar
        batch_size = 256 # You can tune this batch size
        for i in tqdm(range(0, len(chunked_documents), batch_size), desc="Adding documents to Chroma"):
            batch = chunked_documents[i:i + batch_size]
            vectorstore.add_documents(documents=batch)
        
        # Persist the database to disk after adding all documents
        vectorstore.persist()
    
    print(f"Index ready. Total documents in store: {vectorstore._collection.count()}")
    
    # --- Setup the Advanced Retrieval Pipeline ---
    print("Setting up the advanced retrieval pipeline...")
    
    # 1. Vector Retriever (Dense Search)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    # 2. BM25 Retriever (Keyword Search)
    # OPTIMIZATION: Load from disk if it exists, otherwise build and save it.
    bm25_retriever = None
    if os.path.exists(BM25_INDEX_PATH):
        print(f"Loading existing BM25 index from: {BM25_INDEX_PATH}")
        with open(BM25_INDEX_PATH, "rb") as f:
            bm25_retriever = pickle.load(f)
    else:
        print("No BM25 index found. Building new one...")
        # OPTIMIZATION: Call vectorstore.get() only once.
        vs_data = vectorstore.get(include=["metadatas", "documents"])
        all_docs = vs_data['documents']
        all_metadatas = vs_data['metadatas']
        doc_list = [Document(page_content=doc, metadata=meta) for doc, meta in zip(all_docs, all_metadatas)]
        
        bm25_retriever = BM25Retriever.from_documents(doc_list, k=10)
        print(f"Saving BM25 index to: {BM25_INDEX_PATH}")
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump(bm25_retriever, f)

    # 3. Ensemble Retriever for Hybrid Search
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    
    # 4. Reranker
    reranker = FlashrankRerank(model=RERANKER_MODEL, top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker, 
        base_retriever=ensemble_retriever
    )

    # --- Demonstrate Retrieval ---
    print("\n--- Advanced Retrieval Demonstration ---")
    test_set_df = pd.read_json(TEST_SET_FILE, lines=True)
    sample_question = test_set_df.iloc[0]['question']

    print(f"Using a sample query: '{sample_question}'")
    retrieved_docs = compression_retriever.invoke(sample_question)

    print(f"\nSuccessfully retrieved and reranked {len(retrieved_docs)} relevant document chunks.")
    if retrieved_docs:
        print("Top 3 results after reranking:")
        for i, doc in enumerate(retrieved_docs):
            # OPTIMIZATION: Safely format the score to prevent errors.
            score = doc.metadata.get('_compressor_score')
            if isinstance(score, (int, float)):
                score_str = f"{score:.4f}"
            else:
                score_str = "N/A"
            
            print(f"\n--- Result {i+1} (Reranked Score: {score_str}) ---")
            print(f"Source Article: {doc.metadata.get('title', 'N/A')}")
            print("Retrieved Chunk:")
            print(doc.page_content.strip())
            print("-" * 20)
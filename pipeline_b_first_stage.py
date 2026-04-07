import pyterrier as pt
from src.indexing import load_bm25_index
from config import BM25_INDEX_DIR, BM25_K1, BM25_B

# Load the index using the shared loader
index_ref = load_bm25_index(str(BM25_INDEX_DIR))

# Create the BM25 retriever with specified hyperparameters
bm25 = pt.BatchRetrieve(
    index_ref, 
    wmodel="BM25", 
    controls={"bm25.k_1": BM25_K1, "bm25.b": BM25_B}
)

# Initialising the Bo1 query expander
q_expander = pt.rewrite.Bo1QueryExpansion(index_ref, fb_docs=10, fb_terms=10)

# Construct Pipeline B (First Stage) capping output at 100 documents 
pipeline_b = bm25 >> q_expander >> (bm25 % 100)

# Test block: only runs if this file is executed directly, not when imported
if __name__ == "__main__":
    if not pt.java.started():
        pt.java.init()
    
    test_query = "What is a safe withdrawal rate for retirement?"
    results = pipeline_b.search(test_query)
    print("Top 10 results for test query:")
    print(results.head(10))
import pandas as pd
import src.evaluation as ev
import src.data_loader as dl
from pipeline_b_first_stage import pipeline_b

# Load and format queries
queries_dict = dl.load_local_queries()
queries_list = [{"qid": str(qid), "query": text} for qid, text in queries_dict.items()]
queries_df = pd.DataFrame(queries_list)

# Sanitise queries to remove characters that confuse the Terrier parser
queries_df["query"] = queries_df["query"].str.replace(r'[^\w\s]', ' ', regex=True)

# Load relevance judgements (qrels)
qrels = dl.load_local_qrels()

# Run Pipeline B on the full query set
print("Running pipeline on all queries...")
all_results = pipeline_b.transform(queries_df)

# Evaluate results using the project's standard metrics
stage1_scores = ev.evaluate_stage1(all_results, qrels)

# Display results
print("\nEvaluation Results:")
print(ev.format_results_table({"Pipeline B First Stage (Bo1)": stage1_scores}))
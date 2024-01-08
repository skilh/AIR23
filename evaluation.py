import numpy as np


def calculate_precision_recall_f1(retrieved_docs, relevant_docs, k):
    retrieved_at_k = set(retrieved_docs[:k])
    relevant = set(relevant_docs)
    tp = len(retrieved_at_k & relevant)
    precision = tp / len(retrieved_at_k) if retrieved_at_k else 0
    recall = tp / len(relevant) if relevant else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1


def calculate_mrr(re_ranked_results, qrels):

    evaluation_queries = qrels['qid'].unique().tolist()

    mrr_total = 0

    for query_id in evaluation_queries:
        retrieved_docs = re_ranked_results.get(query_id, [])
        relevant_docs = qrels[qrels['qid'] == query_id]['docid'].tolist()  # Get relevant documents

        # Find the rank of the first relevant document
        for rank, doc_id in enumerate(retrieved_docs, start=1):
            if doc_id in relevant_docs:
                mrr_total += 1 / rank
                break

    mrr_score = mrr_total / len(evaluation_queries) if evaluation_queries else 0

    print(f"Mean Reciprocal Rank (MRR): {mrr_score}")


def eval(qrels, re_ranked_results):

    evaluation_queries = qrels['qid'].unique().tolist()

    # Initialize accumulators for metrics
    total_precision = 0
    total_recall = 0
    total_f1 = 0

    # Number of queries
    num_queries = len(evaluation_queries)

    # Loop through each query in the evaluation set
    for query_id in evaluation_queries:
        retrieved_docs = re_ranked_results.get(query_id, [])  # Get ranked documents for the query
        relevant_docs = qrels[qrels['qid'] == query_id]['docid'].tolist()  # Get relevant documents

        precision, recall, f1 = calculate_precision_recall_f1(retrieved_docs, relevant_docs, k=1)

        # Add the metrics to the accumulators
        total_precision += precision
        total_recall += recall
        total_f1 += f1

    # Calculate average metrics
    avg_precision = total_precision / num_queries
    avg_recall = total_recall / num_queries
    avg_f1 = total_f1 / num_queries

    # Print or return the average metrics
    print(f"Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average F1 Score: {avg_f1}")


if __name__ == '__main__':
    print("Evaluation")
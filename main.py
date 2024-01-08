import load_dataset
from sentence_transformers import SentenceTransformer
import pickle
import initial_ret
import cross_encoder
import pandas as pd

def two_stage(collection, queries, qrels):
    # Initial retrival

    doc_embeddings, query_embeddings = initial_ret.create_embeddings(collection, queries)

    print("Embeddings done!")

    initial_retrieval_results = {}

    initial_retrieval_results = initial_ret.perform_initial_ret(doc_embeddings, query_embeddings, queries, collection)

    # Save results from first retrival
    df = pd.DataFrame([(k, v) for k, v in initial_retrieval_results.items()], columns=['query_id', 'doc_ids'])
    df.to_csv('ranked_results.csv', index=False)

    # Re-ranking

    print("Start re-ranking!")

    cross_encoder, tokenizer = cross_encoder.create_model()

    re_ranked_results = {}

    for qid, doc_data in initial_retrieval_results.items():
        query = queries[queries['qid'] == qid]['query'].iloc[0]
        scores = []

        for doc_id, _ in doc_data:  # Unpack the tuple to get the doc_id
            # Retrieve the document text using the doc_id
            if doc_id in collection['docid'].values:
                doc_text = collection[collection['docid'] == doc_id]['text'].iloc[0]

                # Prepare the input for the cross-encoder
                inputs = tokenizer.encode_plus(query, doc_text, return_tensors='pt', truncation=True, max_length=512)

                # Get the relevance score
                outputs = cross_encoder(**inputs)
                score = outputs.logits[0][1].item()  # Assuming binary classification (relevant, not relevant)

                # Store the score with the document ID
                scores.append((doc_id, score))
            else:
                print(f"Document ID {doc_id} not found in collection.")

        # Sort the documents for the query based on the scores
        scores.sort(key=lambda x: x[1], reverse=True)
        re_ranked_results[qid] = [doc_id for doc_id, _ in scores]

    df = pd.DataFrame([(k, v) for k, v in re_ranked_results.items()], columns=['query_id', 'doc_ids'])
    df.to_csv('re_ranked_results.csv', index=False)


def main():
    collection, queries, qrels = load_dataset.load_subset()

    if collection.empty or queries.empty or qrels.empty:
        raise ValueError("Dataset was not loaded correctly!")

    # Two-Stage Retrival

    two_stage(collection, queries, qrels)


if __name__ == '__main__':
    main()

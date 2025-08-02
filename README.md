# Overview
## This project builds a document retrieval system over Alice in Wonderland, using:
    * LangChain — unified framework for chunking documents and managing retrieval
    * Ollama — local embeddings via the mxbai-embed-large model (runs on your machine)
    * Chroma — lightweight open-source vector database for efficient storage and similarity search

The system splits the book into overlapping, fixed-size chunks, creates embeddings, and stores them for retrieval-augmented search — ideal for chatbots, semantic search, or QA.


## Why use Chroma?
    1. Chroma is a lightweight, open-source vector database designed for AI/LLM use cases:
    2. Fast, easy to install and run locally.
    3. Built-in support for embeddings, filters, metadata, and fast similarity search.
    4. Works seamlessly with LangChain, making it popular for RAG and retrieval-based workflows.
    5. Great for experimentation and scaling up as your project grows.

Summary:
You should always use **embedding_function**. Using persist_directory and collection_name is considered best practice for development, reuse, and organization. Chroma is a top choice for storing and searching your document embeddings efficiently in Python and LangChain workflows.


```
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5},
    filter={"rating": {"$gte": 4.0}}
)
```

## Definitions:
    + as_retriever: Converts your vector store (Chroma) into a retriever object for similarity search. 
    + search_kwargs={"k": 5}: When you query with this retriever, it returns the 5 most similar documents to your query based on vector similarity.
    + filter={"rating": {"$gte": 4.0}}: Applies a filter at query time so that only documents with metadata field "rating" ≥ 4.0 are eligible for retrieval (documents with lower ratings will not be considered) .

## Are there additional function variables you should know about?
    Yes! The as_retriever() call (and its underlying search/query methods) accepts several useful parameters:

### Common ones include:
1. k: Number of results to return for each query ("k": 5).
2. filter: Filter results by document metadata (e.g., by "rating", "user", etc.). You can use advanced operators like $gte, $lte, $eq, $in, $and, $or, etc. 
3. where_document: Filter based on the document content, e.g., {"$contains": "Alice"} to only get chunks containing "Alice" .
4. score_threshold: Float between 0 and 1; only returns documents with similarity score above threshold .
5. fetch_k: (For advanced use) Number of documents to fetch before filtering for "maximal marginal relevance" queries .
6. lambda_mult: Degree of diversity in returned results for Maximal Marginal Relevance (MMR) search .


**Example of combining multiple filters:**

python
```
search_kwargs={
  "k": 10,
  "filter": {
    "$and": [
      {"rating": {"$gte": 4.0}},
      {"author": {"$eq": "Lewis Carroll"}}
    ]
  }
}
```

**Example of using where_document:**

python
```
search_kwargs={
  "k": 5,
  "where_document": {"$contains": "rabbit"}
}
```

## When should you use these options?
+ Use k to control how many top results you want returned for each query.
+ Use filter for precise control based on your metadata fields (works just like filters in a database).
+ Use logical filters ($and, $or, $in, etc.) for more complex retrieval logic.
+ Use where_document to filter based on content inside each chunk/document (not just metadata).
+ Use score_threshold if you want only highly similar results.

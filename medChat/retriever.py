from langchain_pinecone import PineconeVectorStore
from langchain_core.vectorstores import VectorStoreRetriever

def get_retriever(vectorstore: PineconeVectorStore) -> VectorStoreRetriever:
    return vectorstore.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k": 5}
    )
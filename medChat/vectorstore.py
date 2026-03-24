from langchain_pinecone import PineconeVectorStore
from medChat.config import PINECONE_INDEX_NAME
from medChat.embeddings import get_embedding_model

def get_vectorstore() -> PineconeVectorStore:
    embedding_model = get_embedding_model()
    
    # connects to existing index, does NOT re-upload embeddings
    vectorstore = PineconeVectorStore(
        index_name = PINECONE_INDEX_NAME,
        embedding = embedding_model
    )
    return vectorstore
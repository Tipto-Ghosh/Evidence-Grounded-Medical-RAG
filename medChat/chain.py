from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from medChat.config import LLM_MODEL_NAME, HF_TOKEN
from medChat.prompts import get_contextualize_prompt, get_qa_prompt

store = {}  # session_id -> ChatMessageHistory

def get_llm():
    llm = HuggingFaceEndpoint(
        repo_id = LLM_MODEL_NAME,
        huggingfacehub_api_token = HF_TOKEN
    )
    return ChatHuggingFace(llm=llm)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def build_conversational_rag(retriever):
    chat_model = get_llm()

    history_aware_retriever = create_history_aware_retriever(
        llm = chat_model,
        retriever = retriever,
        prompt = get_contextualize_prompt()
    )

    qa_chain = create_stuff_documents_chain(chat_model, get_qa_prompt())
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    conversational_rag = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key = "input",
        history_messages_key = "chat_history",
        output_messages_key = "answer"
    )
    return conversational_rag
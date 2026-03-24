from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def get_contextualize_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and the latest user question, "
         "reformulate the question to be standalone and clear. "
         "Do NOT answer it, just rephrase if needed."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

def get_qa_prompt() -> ChatPromptTemplate:
    system_prompt = (
        "You are an assistant for medical question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer or the context is insufficient, "
        "simply say you don't know. Keep the answer concise within 3-5 sentences.\n\n"
        "{context}"
    )
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
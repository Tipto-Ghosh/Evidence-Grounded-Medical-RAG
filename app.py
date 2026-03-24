import streamlit as st
from medChat.vectorstore import get_vectorstore
from medChat.retriever import get_retriever
from medChat.chain import build_conversational_rag

st.set_page_config(page_title="Medical RAG", page_icon="🏥")
st.title("🏥 Medical Encyclopedia Q&A")

# build pipeline once and cache it
@st.cache_resource
def load_pipeline():
    vectorstore = get_vectorstore()
    retriever = get_retriever(vectorstore)
    rag_chain = build_conversational_rag(retriever)
    return rag_chain

rag_chain = load_pipeline()

# session state for chat history display
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = "user_1"

# render existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# chat input
if user_input := st.chat_input("Ask a medical question..."):

    # show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"session_id": st.session_state.session_id}}
            response = rag_chain.invoke({"input": user_input}, config=config)
            answer = response["answer"]
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
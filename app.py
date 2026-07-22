import streamlit as st
from src.llm_chain import build_chain

st.set_page_config(page_title="RAG Assistant", page_icon="📄")
st.title("RAG Assistant")
st.caption("Ask questions about the documents in the knowledge base.")


@st.cache_resource
def get_chain():
    return build_chain()


chain = get_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = chain.invoke(question)
            answer = result["answer"]
            sources = sorted({doc.metadata.get("source", "unknown") for doc in result["context"]})

        st.markdown(answer)
        if sources:
            st.caption(f"Sources: {', '.join(sources)}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
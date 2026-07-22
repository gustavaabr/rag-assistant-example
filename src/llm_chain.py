import os
from dotenv import load_dotenv 
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

load_dotenv()

PROMPT_TEMPLATE = """Answer the question based only on the following context. \
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}
"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_chain():
    """Assemble the RAG chain: retrieve chunks from Pinecone, then answer with the LLM."""
    embeddings = OpenAIEmbeddings()
    vector_store = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        embedding=embeddings,
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    answer_chain = (
        {"context": lambda x: format_docs(x["context"]), "question": lambda x: x["question"]}
        | prompt
        | llm
        | StrOutputParser())

    return RunnableParallel(context=retriever, question=RunnablePassthrough()).assign(answer=answer_chain)

def ask(question: str):
    """Answer a question and return (answer, sorted list of source PDF filenames)."""
    chain = build_chain()
    result = chain.invoke(question)

    sources = sorted({doc.metadata.get("source", "unknown") for doc in result["context"]})
    return result["answer"], sources

if __name__ == "__main__":
    question = input("Ask a question: ")
    answer, sources = ask(question)
    print(f"\nAnswer: {answer}")
    print(f"Sources: {', '.join(sources)}")
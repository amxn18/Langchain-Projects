import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
api = os.environ["GROQ_API_KEY"]

st.set_page_config(
    page_title="RAG Using Groq Inferencing",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "vectorstore" not in st.session_state:
    embeddings = HuggingFaceEndpointEmbeddings(
        repo_id="sentence-transformers/all-MiniLM-L6-v2"
    )

    loader = WebBaseLoader("https://console.groq.com/docs/overview")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    st.session_state.vectorstore = vectorstore

llm = ChatGroq(
    groq_api_key=api,
    model="llama-3.1-8b-instant",
    temperature=0.2
)

parser = StrOutputParser()

prompt_template = PromptTemplate(
    template="""
    Answer the question strictly based on the context below.
    If the answer is not present, say "Answer not found in context."

    <context>
    {context}
    </context>

    Question: {question}
    """,
    input_variables=["context", "question"]
)

llm_chain = prompt_template | llm | parser

st.title("Chat Using Groq Inferencing")

query = st.text_input("Please write your question here")
btn = st.button("Submit Question")

if query and btn:
    start = time.process_time()

    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(query)

    context = "\n\n".join(doc.page_content for doc in docs)

    response = llm_chain.invoke({
        "context": context,
        "question": query
    })

    st.write(response)

    st.write("Response time:", time.process_time() - start)

    with st.expander("Retrieved Documents"):
        for doc in docs:
            st.write(doc.page_content)
            st.write("-----")

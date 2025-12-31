import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_objectbox.vectorstores import ObjectBox
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    temperature=0.2
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

st.set_page_config(
    page_title="RAG Using ObjectBox",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("ObjectBox VectorDB")

def vector_embedding():
    if "vectorstore" not in st.session_state:
        embeddings = HuggingFaceEndpointEmbeddings(
            repo_id="sentence-transformers/all-MiniLM-L6-v2"
        )

        loader = PyPDFDirectoryLoader("./documents")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        vectorstore = ObjectBox.from_documents(
            chunks,
            embeddings,
            embedding_dimension=384
        )

        st.session_state.vectorstore = vectorstore

def retrieve_similar_documents(query: str, k: int = 4):
    if "vectorstore" in st.session_state:
        vectorstore = st.session_state.vectorstore
        return vectorstore.similarity_search(query, k=k)
    return []

def generate_answer(query: str):
    similar_docs = retrieve_similar_documents(query, k=3)

    context = "\n\n".join([doc.page_content for doc in similar_docs])

    prompt = PromptTemplate(
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

    chain = prompt | model | parser

    response = chain.invoke({
        "context": context,
        "question": query
    })

    return response, similar_docs

user_query = st.text_input("Enter Your Question From Documents")
btn = st.button("Submit Question")

if btn and user_query:
    vector_embedding()
    with st.spinner("Generating response..."):
        answer, docs = generate_answer(user_query)

    st.write(answer)

    with st.expander("Document Similarity Search"):
        for doc in docs:
            st.write(doc.page_content)
            st.write("-------------------------------")

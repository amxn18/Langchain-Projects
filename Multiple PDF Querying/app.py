import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
    temperature=0.2
)

model = ChatHuggingFace(llm=llm)

embeddings = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
)

parser = StrOutputParser()

st.set_page_config(
    page_title="Multiple PDF Querying",
    layout="wide",
    initial_sidebar_state="expanded",
)

def loadDocs(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    os.remove(tmp_path)
    return docs

def chunkDocs(docs, chunk_size=1500, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

def VectorStore(textChunks, embeddings):
    vectorStore = FAISS.from_documents(textChunks, embeddings)
    vectorStore.save_local("faiss_index")

def conversationChain():
    template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not available in the context, say:
    "answer is not available in the context".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    return prompt | model | parser

def userInput(query: str):
    docs = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    similar_docs = docs.similarity_search(query, k=4)
    context = "\n\n".join([doc.page_content for doc in similar_docs])

    chain = conversationChain()
    response = chain.invoke({
        "context": context,
        "question": query
    })

    st.write("Retrieved Information:", response)

def main():
    st.title("Multiple PDF Querying Application")
    st.write("Upload multiple PDF documents and ask questions related to their content.")

    query = st.text_input("Enter your question related to the uploaded documents:")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files and query:
        if st.button("Process Documents and Get Answer"):
            with st.spinner("Processing documents..."):
                all_docs = []
                for uploaded_file in uploaded_files:
                    docs = loadDocs(uploaded_file)
                    all_docs.extend(docs)

                text_chunks = chunkDocs(all_docs)
                VectorStore(text_chunks, embeddings)

            with st.spinner("Generating answer..."):
                userInput(query)

            st.success("Answer generated.")

if __name__ == "__main__":
    main()

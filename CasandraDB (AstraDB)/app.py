from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_astradb import AstraDBVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from astrapy import DataAPIClient
from dotenv import load_dotenv
from datasets import load_dataset
import os

# https://astra.datastax.com/
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    temperature=0.2,
    max_new_tokens=256
)

model = ChatHuggingFace(llm=llm)


embeddings = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
)
client = DataAPIClient(os.environ["ASTRA_DB_APPLICATION_TOKEN"])
db = client.get_database_by_api_endpoint(
    os.environ["ASTRA_DB_API_ENDPOINT"]
)

# Vector store
vectorstore = AstraDBVectorStore(
    embedding=embeddings,
    collection_name="vectorSearch",
    api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
    token=os.environ["ASTRA_DB_APPLICATION_TOKEN"]
)

# print("Astra DB vector store ready") 

def loadDocs(path: str):
    loader = PyPDFLoader(path)
    docs = loader.load()
    return docs

def chunkDocs(docs, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    return chunks

documents = loadDocs("CasandraDB (AstraDB)\documents\docsss.pdf")
chunks = chunkDocs(documents)

# print(len(documents))
# print(len(chunks))
# print(chunks[0])

vectorstore.add_documents(chunks)
# print("Documents have been embedded and stored in Astra DB.")

def retrieve_similar_documents(query: str, k: int = 4):
    similar_docs = vectorstore.similarity_search(query, k=k)
    return similar_docs

def generate_answer(query: str):
    similar_docs = retrieve_similar_documents(query, k=3)
    
    # Combining context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in similar_docs])
    
    # Create prompt
    prompt = f"""Based on the following context, answer the question. 
    Context:{context}
    Question: {query}
    Answer:"""

    
    response = model.invoke(prompt)
    return response.content

while True:
    query = input("Enter your question related to the document: ")
    
    if not query.strip():
        print("Please enter a valid question.\n")
        continue
    
    print("\nGenerating answer...\n")
    answer = generate_answer(query)
    print("Answer:", answer)
    continue_chat = input("Do you want to ask another question? (yes/no): ").strip().lower()
    
    if continue_chat not in ['yes', 'y']:
        print("\nThank you for using the document Q&A system. Goodbye!")
        break


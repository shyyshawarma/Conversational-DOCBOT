import os
import json
import chromadb
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="pdf_rag")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


with open("processed_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Store text and image insights in ChromaDB
for page, content in data.items():
    text = content.get("text", "")
    image_insights = " ".join([img["insight"] for img in content.get("image_insights", [])])
    combined_text = text + " " + image_insights
    
    if combined_text.strip():
        vector = embeddings.embed_query(combined_text)
        collection.add(
            ids=[str(page)], 
            embeddings=[vector], 
            metadatas=[{"page": page, "text": combined_text}]
        )

# Configure LLM (Gemma via Groq API)
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="Gemma2-9b-It")

def retrieve_and_generate(query):
    """Retrieve relevant documents from ChromaDB and generate response using Gemma."""
    query_vector = embeddings.embed_query(query)
    results = collection.query(query_embeddings=[query_vector], n_results=3)
    
    retrieved_text = "\n".join([doc["text"] for doc in results["metadatas"]])
    prompt = f"Context:\n{retrieved_text}\n\nUser Query: {query}\n\nAnswer:"
    
    response = llm.invoke(prompt)
    return response

# Example query
query = "How did COVID-19 impact stock markets?"
response = retrieve_and_generate(query)
print("Generated Response:", response)

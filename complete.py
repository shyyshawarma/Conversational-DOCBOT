import os
import streamlit as st
import fitz
import json
import shutil
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()


genai.configure(api_key=os.getenv("GEMINI_API_KEY"), transport="rest")


chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="pdf_rag")


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="deepseek-r1-distill-llama-70b")


def clear_image_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)

def extract_text(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = loader.load_and_split(text_splitter)
    return {i + 1: doc.page_content for i, doc in enumerate(docs)}


def extract_images(pdf_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_data = {}

    for page_number in range(len(doc)):
        images = doc[page_number].get_images(full=True)
        page_images = []
        
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            if not base_image:
                continue

            image_bytes = base_image["image"]
            image_ext = base_image.get("ext", "png")
            image_filename = os.path.join(output_folder, f"page_{page_number + 1}_img_{img_index + 1}.{image_ext}")
            
            with open(image_filename, "wb") as img_file:
                img_file.write(image_bytes)
            
            page_images.append(image_filename)
        
        if page_images:
            image_data[page_number + 1] = page_images

    return image_data

#gemini
def send_to_gemini(image_path):
    prompt = "Extract key insights from this image, including any text and meaningful patterns; if it is a table extract and summarize the data in this table and if it is graph analyze this graph and describe the trends, patterns, and key insights."

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        img = Image.open(image_path)
        response = model.generate_content([prompt, img])
        return response.text if response else "No response"
    except Exception as e:
        return f"Error processing {image_path}: {e}"


def process_pdf(pdf_path, image_folder):
    text_content = extract_text(pdf_path)
    image_data = extract_images(pdf_path, image_folder)

    extracted_data = {}

    for page_num, text in text_content.items():
        extracted_data[page_num] = {"text": text, "image_insights": []}

        if page_num in image_data:
            for img_path in image_data[page_num]:
                insight = send_to_gemini(img_path)
                extracted_data[page_num]["image_insights"].append({"image_path": img_path, "insight": insight})

    return extracted_data


def store_in_chromadb(data):
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


def retrieve_and_generate(query):
    query_vector = embeddings.embed_query(query)
    results = collection.query(query_embeddings=[query_vector], n_results=3)

    retrieved_text = "\n".join([doc["text"] for doc in results["metadatas"][0]])
    prompt = f"Context:\n{retrieved_text}\n\nUser Query: {query}\n\nAnswer:"

    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error: {e}"


st.title("CONVERSATIONAL RAG CHATBOT")
st.write("Upload a PDF and chat with it!")


uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    pdf_path = "uploaded_document.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    image_folder = "extracted_images"
    clear_image_folder(image_folder)

    st.write("🔍 Processing PDF...")
    extracted_data = process_pdf(pdf_path, image_folder)

    with open("processed_data.json", "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=4)

    st.success("✅ PDF processed successfully!")

    st.write("📥 Storing extracted data in ChromaDB...")
    store_in_chromadb(extracted_data)
    st.success("✅ Data stored successfully!")


st.subheader("💬 Chat with your PDF")

query = st.text_input("Ask a question about the document:")
if query:
    response = retrieve_and_generate(query)
    st.write("🤖 Response:")
    st.write(response)

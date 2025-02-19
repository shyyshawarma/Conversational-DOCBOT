# Conversational-DOCBOT
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🚀 Conversational RAG Chatbot is an interactive Retrieval-Augmented Generation (RAG) application that allows users to upload PDFs and chat with them using Streamlit. It leverages Gemini Vision for image analysis, ChromaDB for vector storage, and Groq's DeepSeek model for response generation.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🔥 Features

✅ PDF Processing – Extracts text and images from PDFs<br/>
✅ Image Analysis – Uses Gemini Vision to analyze images, tables, and graphs<br/>
✅ Vector Storage – Stores extracted data in ChromaDB for efficient retrieval<br/>
✅ Conversational AI – Enables chat-based interactions with PDFs<br/>
✅ Streamlit UI – Simple and interactive web interface<br/>

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🛠️ Tech Stack

Python, Streamlit<br/>
LangChain, ChromaDB, Hugging Face Embeddings<br/>
Gemini Vision for image insights<br/>
DeepSeek LLM (via Groq) for response generation<br/>
PyMuPDF (Fitz) for PDF text/image extraction<br/>

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🚀 How to Run

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

1️⃣ Clone the repository
```
git clone https://github.com/shyyshawarma/Conversational-DOCBOT
cd conversational-rag-chatbot
```
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

2️⃣ Install dependencies
```
pip install -r requirements.txt

```
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

3️⃣ Set up environment variables (.env file)
```
GEMINI_API_KEY=your_gemini_api_key  
GROQ_API_KEY=your_groq_api_key  
HF_TOKEN=your_huggingface_token
```
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

4️⃣ Run the Streamlit app
```
streamlit run app.py

```
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📌 Usage

1️⃣ Upload a PDF<br />
2️⃣ The system extracts text & images<br />
3️⃣ Images are analyzed using Gemini Vision<br />
4️⃣ Extracted data is stored in ChromaDB<br />
5️⃣ Ask questions about the document in the chat interface<br />
6️⃣ Get accurate, context-aware responses<br />
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🎯 Future Improvements

📊 Advanced Data Visualization for extracted insights<br />
🤖 Multi-LLM Support for diverse responses<br />

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🛠 Built for seamless document understanding and interactive AI-driven insights. 🚀✨


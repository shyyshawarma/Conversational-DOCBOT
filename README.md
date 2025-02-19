# 🚀 Conversational-DOCBOT  

Conversational-DOCBOT is an **interactive Retrieval-Augmented Generation (RAG) application** that enables users to **upload PDFs and chat with them** using **Streamlit**. It integrates **Gemini Vision** for image analysis, **ChromaDB** for vector storage, and **Groq's DeepSeek model** for AI-powered responses.  


## 🔥 Features  

✅ **PDF Processing** – Extracts **text** and **images** from PDFs  
✅ **Image Analysis** – Uses **Gemini Vision** to analyze images, tables, and graphs  
✅ **Vector Storage** – Stores extracted data in **ChromaDB** for efficient retrieval  
✅ **Conversational AI** – Enables **chat-based interactions** with PDFs  
✅ **Streamlit UI** – Simple and interactive **web interface**  



## 🛠️ Tech Stack  

- **Python, Streamlit**  
- **LangChain, ChromaDB, Hugging Face Embeddings**  
- **Gemini Vision** for image insights  
- **DeepSeek LLM (via Groq)** for response generation  
- **PyMuPDF (Fitz)** for PDF text/image extraction  



## 🚀 How to Run  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/shyyshawarma/Conversational-DOCBOT.git  
cd Conversational-DOCBOT  
```  

### 2️⃣ Install dependencies  
```bash
pip install -r requirements.txt  
```  

### 3️⃣ Set up environment variables (`.env` file)  
```bash
GEMINI_API_KEY=your_gemini_api_key  
GROQ_API_KEY=your_groq_api_key  
HF_TOKEN=your_huggingface_token  
```  

### 4️⃣ Run the Streamlit app  
Go to complete.py and run 
```bash
streamlit run app.py  
```  



## 📌 Usage  

1️⃣ **Upload a PDF**  
2️⃣ **Extract text & images** automatically  
3️⃣ **Analyze images** using **Gemini Vision**  
4️⃣ **Store extracted data** in **ChromaDB**  
5️⃣ **Chat with your document** using an AI-powered interface  
6️⃣ **Receive accurate, context-aware responses**  



## 🎯 Future Improvements  

📊 **Advanced Data Visualization** for extracted insights  
🤖 **Multi-LLM Support** for diverse responses  

---

🛠 **Built for seamless document understanding and AI-powered insights.** 🚀✨  

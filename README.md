
# Conversational RAG: Context-Aware Chat with Your PDFs  

🚀 **Conversational RAG** is a Streamlit-based application that allows you to chat with your PDFs using **Retrieval-Augmented Generation (RAG)**. It provides **context-aware responses**, making interactions more relevant and insightful.  

## ✨ Features  
- 📄 **Upload PDFs** and extract meaningful content  
- 🧠 **Context-aware responses** using conversation history  
- 🔍 **FAISS vector store** for efficient retrieval  
- 🤗 **Hugging Face embeddings** for semantic search  
- ⚡ **Ollama-powered LLM** for intelligent responses  
- 🔄 **Chat history management** for better conversation flow  

## 🛠️ Installation  

1. **Clone the repository**  
   ```
   git clone https://github.com/vikrambhat2/conversational-rag.git
   cd conversational-rag
   ```

2. **Create a virtual environment (optional but recommended)**  
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies**  
   ```
   pip install -r requirements.txt
   ```

## 🚀 Running the App  

```
streamlit run context_chat_rag_deepseek.py
```

## 📂 File Structure  
```
├── context_chat_rag_deepseek.py                 # Main Streamlit app  
├── requirements.txt        # Required dependencies  
├── README.md               # Project documentation  
└── uploads/                # Directory for uploaded PDFs  
```

## 📝 Usage  
1. Upload a **PDF** from the sidebar.  
2. Ask questions about the document in the chat input.  
3. Get **context-aware responses** based on document content.  
4. View **reasoning behind responses** (optional).  

## 🏗️ Technologies Used  
- **LangChain** (Document processing, retrieval chains)  
- **Streamlit** (UI for interaction)  
- **FAISS** (Efficient similarity search)  
- **Ollama** (LLM model execution)  
- **Hugging Face Embeddings** (Text embeddings)  

## 🤝 Contributing  
Feel free to **fork** the repo, create a branch, and submit a pull request!  


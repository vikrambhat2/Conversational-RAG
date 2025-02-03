import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
import os
import re
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains import create_history_aware_retriever

llm_model = "deepseek-r1:1.5b"

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to process the uploaded PDF
def process_pdf(pdf_path):
    st.info("Processing PDF... ⏳")
    
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=256)
    split_docs = text_splitter.split_documents(docs)

    # Create embeddings and FAISS vector store
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embedding_model)

    return vectorstore

# Function to extract reasoning inside <think> tags
def extract_think_content(text):
    match = re.search(r'<think>(.*?)</think>', text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""

# Function to remove everything between <think> and </think> tags
def remove_think_content(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

# Function to get conversation chain with history
def get_conversation_chain(retriever):
    llm = OllamaLLM(model=llm_model)

    contextualize_q_system_prompt = (
        "Given the chat history and the latest user question, "
        "provide a response that directly addresses the user's query based on the provided documents. "
        "Do not rephrase the question or ask follow-up questions."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    ### **Fix: Include `context` in the system prompt**
    system_prompt = (
        "As a personal chat assistant, provide accurate and relevant information based on the provided documents. "
        "Limit answers to 2-3 sentences and 50 words max. "
        "Do not form standalone questions, suggest selections, or ask further questions."
        "\n\nContext:\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    ### **Statefully manage chat history**
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    print("Conversational chain created ✅")
    return conversational_rag_chain

# Streamlit UI
st.set_page_config(page_title="Chat with PDF", layout="wide")
st.title("📄 Chat with Your PDF using RAG 💬")
st.sidebar.header("Upload Your PDF")

uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    pdf_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)
    
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    vectorstore = process_pdf(pdf_path)
    retriever = vectorstore.as_retriever()
    st.session_state.chatbot = get_conversation_chain(retriever)
    st.success("PDF uploaded and processed! ✅ Start asking questions below.")

show_think_content = st.sidebar.checkbox("Show Reasoning")
st.subheader("Chat with the PDF 🤖")

if st.session_state.messages:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

if prompt := st.chat_input("Ask a question about your document"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    try:
        response = st.session_state.chatbot.invoke(
            {"input": prompt, "chat_history": st.session_state.messages},
            {"configurable": {"session_id": "user_session"}}  # Add session_id here
        )
        answer = response["answer"]
        think_content = extract_think_content(answer)
        answer = remove_think_content(answer)

        if "I could not find relevant information" in answer:
            st.warning("⚠️ No relevant information found in the uploaded PDF.")
    except Exception as e:
        answer = f"An error occurred: {e}"
        think_content = ""

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        st.write(answer)
        if show_think_content and think_content:
            with st.expander("🤔 Show/Hide Reasoning"):
                st.write(think_content)
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM as Ollama
from langchain.prompts import ChatPromptTemplate

# Constants
CHROMA_PATH = "chroma"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Load embedding + vector DB
@st.cache_resource
def load_vector_store():
    embedding_function = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    return db

# Run RAG
def generate_answer(query_text):
    db = load_vector_store()
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    if len(results) == 0 or results[0][1] < 0.3:
        return "❌ No relevant context found.", [], []

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text, question=query_text
    )

    model = Ollama(model="gemma3")
    response = model.invoke(prompt)

    sources = [doc.metadata.get("source", "unknown") for doc, _ in results]
    scores = [score for _, score in results]

    return response, sources, scores

# ------------------ Streamlit UI ------------------

st.set_page_config(page_title="📚 Local RAG with Ollama", layout="wide")
st.title("📚 Ask Alice in Wonderland")
st.markdown("Ask questions based on the local markdown file.")

query = st.text_input("Enter your question:", value="Who is Alice?")

if st.button("Get Answer"):
    with st.spinner("Thinking..."):
        answer, sources, scores = generate_answer(query)
        st.subheader("🧠 Answer")
        st.markdown(answer)

        st.subheader("📚 Sources")
        for src, score in zip(sources, scores):
            st.markdown(f"- **{src}** (score: `{score:.4f}`)")

st.markdown("---")
st.caption("Powered by Ollama + HuggingFace + LangChain + Chroma")

import os
import streamlit as st
import fitz
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import tempfile
#.
load_dotenv()

if "page_number" not in st.session_state:
    st.session_state.page_number = 0
if "zoom" not in st.session_state:
    st.session_state.zoom = 2.5

def load_db(file, chain_type, k):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(file.getbuffer())

    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)

    google_api_key = os.getenv("GOOGLE_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            google_api_key=google_api_key
        ),
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True
    )

    return qa, documents

def render_pdf_page(file, page_number, zoom=1.0, highlight_texts=[]):
    doc = fitz.open(stream=file.getvalue(), filetype="pdf")
    page = doc.load_page(page_number)
    mat = fitz.Matrix(zoom, zoom)

    # Apply highlighting
    for text, color in highlight_texts:
        areas = page.search_for(text)
        for area in areas:
            highlight = page.add_highlight_annot(area)
            highlight.set_colors(stroke=color)
            highlight.update()

    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")

def generate_synopsis(file, qa_chain):
    doc = fitz.open(stream=file.getvalue(), filetype="pdf")
    full_text = ""

    for i in range(min(5, len(doc))):
        full_text += doc[i].get_text()

    prompt = "Summarize this document briefly in 5-8 lines:\n\n" + full_text
    response = qa_chain({"question": prompt, "chat_history": []})
    return response['answer'].strip()

def app():
    st.set_page_config(layout="wide")
    st.title("📖 Scriptura: Your Literary Companion")

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file:
        if 'qa_chain' not in st.session_state:
            st.session_state.qa_chain, st.session_state.documents = load_db(uploaded_file, chain_type="stuff", k=4)

        total_pages = len(st.session_state.documents)

        col1, col2 = st.columns([5, 1])

        with col2:
            st.subheader("Navigation & Zoom")
            page_number = st.slider("Select a page", 0, total_pages - 1, st.session_state.page_number, 1)
            st.session_state.page_number = page_number

            zoom_display = st.radio("Zoom Mode", ["Manual Zoom", "Fit to Page", "Fit to Width"])
            click_to_zoom = st.checkbox("Click to Zoom", value=False)

            if zoom_display == "Manual Zoom":
                st.session_state.zoom = st.slider("Zoom", 0.5, 3.0, st.session_state.zoom, step=0.1)
            elif zoom_display == "Fit to Page":
                st.session_state.zoom = 1.3
            elif zoom_display == "Fit to Width":
                st.session_state.zoom = 2.0

            st.write(f"Current Zoom: {st.session_state.zoom:.1f}x")

        with col1:
            # Define text to highlight
            highlight_texts = [("example text", (1, 0, 0))]  # Format: (text_to_highlight, (R, G, B))

            img_data = render_pdf_page(uploaded_file, st.session_state.page_number, zoom=st.session_state.zoom, highlight_texts=highlight_texts)

            st.markdown(
                "<div style='overflow-x:auto; white-space: nowrap;'>",
                unsafe_allow_html=True
            )
            if click_to_zoom:
                st.image(img_data, caption=f"Page {st.session_state.page_number + 1}", use_column_width=False)
            else:
                st.image(img_data, caption=f"Page {st.session_state.page_number + 1}")
            st.markdown("</div>", unsafe_allow_html=True)

        progress = (st.session_state.page_number + 1) / total_pages
        st.progress(progress)

        st.subheader("📄 Document Synopsis:")
        synopsis = generate_synopsis(uploaded_file, st.session_state.qa_chain)
        st.write(synopsis)

        st.subheader("💬 Ask a question about the document:")
        user_input = st.text_input("Your question:")
        if user_input:
            st.write(f"Query: {user_input}")
            response = st.session_state.qa_chain({"question": user_input, "chat_history": []})
            st.write(f"**Answer:** {response['answer']}")

            # st.write("📄 **Sources:**")
            # unique_sources = set()
            # for source in response["source_documents"]:
            #     source_file = source.metadata.get('source', 'Unknown')
            #     human_readable_source = os.path.basename(source_file) if source_file != 'Unknown' else 'Unknown'
            #     unique_sources.add(human_readable_source)

            # # Display each unique source only once
            # for src in unique_sources:
            #     st.markdown(f"- {src}")


if __name__ == "__main__":
    app()

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from langchain_core.messages import HumanMessage



# Page configuration


st.set_page_config(
    page_title="Intelligent Hybrid Analysis Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)



# Imports


from graph import create_graph
from utils.dataset_loader import DatasetLoader
from utils.dataset_pofiler import DatasetProfiler
from services.rag_ingestion import RAGIngestion



# Paths


UPLOAD_DIR = Path("uploads")
DATASET_DIR = UPLOAD_DIR / "datasets"
DOCUMENT_DIR = UPLOAD_DIR / "documents"

DATASET_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DOCUMENT_DIR.mkdir(
    parents=True,
    exist_ok=True
)



# Custom CSS


st.markdown(
    """
    <style>

    /* Main container */

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Sidebar */

    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.2);
    }

    /* Header */

    .app-header {
        padding: 1rem 0 1.5rem 0;
    }

    .app-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .app-subtitle {
        color: #777;
        font-size: 1rem;
    }

    /* Route badge */

    .route-badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(128, 128, 128, 0.12);
        margin-bottom: 1rem;
    }

    /* Source box */

    .source-box {
        padding: 0.8rem;
        border-radius: 8px;
        background: rgba(128, 128, 128, 0.08);
        margin-top: 0.5rem;
        font-size: 0.85rem;
    }

    /* Empty state */

    .empty-state {
        text-align: center;
        padding: 5rem 1rem;
        color: #777;
    }

    .empty-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }

    /* Stats */

    .stat-card {
        padding: 1rem;
        border-radius: 10px;
        background: rgba(128, 128, 128, 0.08);
        text-align: center;
    }

    .stat-value {
        font-size: 1.4rem;
        font-weight: 700;
    }

    .stat-label {
        font-size: 0.8rem;
        color: #777;
    }

    </style>
    """,
    unsafe_allow_html=True
)



# Session state


if "messages" not in st.session_state:
    st.session_state.messages = []

if "graph" not in st.session_state:
    st.session_state.graph = None

if "dataset" not in st.session_state:
    st.session_state.dataset = None

if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None

if "dataset_profile" not in st.session_state:
    st.session_state.dataset_profile = None

if "last_route" not in st.session_state:
    st.session_state.last_route = None

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

if "document_names" not in st.session_state:
    st.session_state.document_names = []



# Helper functions


def initialize_dataset(df, filename="survey.csv"):
    """
    Build a new graph around the supplied DataFrame.
    """

    profiler = DatasetProfiler()

    profile = profiler.profile(df)

    new_graph = create_graph(df)

    st.session_state.dataset = df
    st.session_state.dataset_name = filename
    st.session_state.dataset_profile = profile
    st.session_state.graph = new_graph


def save_uploaded_file(uploaded_file, directory):
    """
    Save a Streamlit UploadedFile to disk.
    """

    file_path = directory / uploaded_file.name

    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    return file_path



# Sidebar


with st.sidebar:

    st.markdown("## 🤖 Hybrid Agent")

    st.caption(
        "Analytics + RAG + Hybrid reasoning"
    )

    st.divider()


    # Dataset section
  

    st.markdown("### 📊 Dataset")

    uploaded_dataset = st.file_uploader(
        "Upload a CSV dataset",
        type=["csv"],
        help="Upload a CSV file for analytical questions."
    )

    if uploaded_dataset is not None:

        if (
            st.session_state.dataset_name
            != uploaded_dataset.name
        ):

            try:

                with st.spinner("Loading dataset..."):

                    dataset_path = save_uploaded_file(
                        uploaded_dataset,
                        DATASET_DIR
                    )

                    loader = DatasetLoader()

                    df = loader.load_csv(
                        str(dataset_path)
                    )

                    initialize_dataset(
                        df,
                        uploaded_dataset.name
                    )

                st.success(
                    f"Loaded {len(df):,} rows."
                )

            except Exception as e:

                st.error(
                    f"Could not load dataset: {e}"
                )

    # Dataset information

    if st.session_state.dataset is not None:

        profile = st.session_state.dataset_profile

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Rows",
                f"{profile['rows']:,}"
            )

        with col2:
            st.metric(
                "Columns",
                f"{profile['columns']:,}"
            )

        st.caption(
            f"Current: **{st.session_state.dataset_name}**"
        )

    else:

        st.info(
            "Using the default dataset."
        )

    st.divider()

    
    # Document section
  

    st.markdown("### 📚 Knowledge Base")

    uploaded_document = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt"],
        help="PDF and TXT files are supported."
    )

    if uploaded_document is not None:

        if uploaded_document.name not in st.session_state.document_names:

            try:

                with st.spinner(
                    "Processing document..."
                ):

                    document_path = save_uploaded_file(
                        uploaded_document,
                        DOCUMENT_DIR
                    )

                    ingestion = RAGIngestion()

                    ingestion.ingest(
                        str(document_path)
                    )

                    st.session_state.document_names.append(
                        uploaded_document.name
                    )

                st.success(
                    "Document added to knowledge base."
                )

            except Exception as e:

                st.error(
                    f"Could not process document: {e}"
                )

    if st.session_state.document_names:

        st.caption("Uploaded documents:")

        for document in st.session_state.document_names:
            st.write(f"• {document}")

    else:

        st.caption(
            "No additional documents uploaded."
        )

    st.divider()

    
    # Chat controls
    

    st.markdown("### ⚙️ Controls")

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True
    ):

        st.session_state.messages = []
        st.session_state.last_route = None
        st.session_state.last_sources = []

        st.rerun()

    st.divider()

    st.caption(
        "Intelligent Hybrid Analysis Agent"
    )

    st.caption(
        "LangGraph • LangChain • Groq • "
        "Pandas • ChromaDB"
    )



# Main header


st.markdown(
    """
    <div class="app-header">
        <div class="app-title">
            🤖 Intelligent Hybrid Analysis Agent
        </div>
        <div class="app-subtitle">
            Ask questions about your dataset and
            supporting documentation using
            Analytics, RAG, or Hybrid reasoning.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# Default dataset initialization


if st.session_state.graph is None:

    try:

        default_df = pd.read_csv(
            "survey.csv"
        )

        initialize_dataset(
            default_df,
            "survey.csv"
        )

    except Exception as e:

        st.error(
            f"Could not load the default dataset: {e}"
        )

        st.stop()



# Dataset overview


with st.expander(
    "📊 Dataset overview",
    expanded=False
):

    df = st.session_state.dataset

    profile = st.session_state.dataset_profile

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Rows",
            f"{profile['rows']:,}"
        )

    with col2:
        st.metric(
            "Columns",
            f"{profile['columns']:,}"
        )

    with col3:
        st.metric(
            "Numeric",
            len(profile["numeric_columns"])
        )

    with col4:
        st.metric(
            "Categorical",
            len(profile["categorical_columns"])
        )

    st.markdown("#### Dataset preview")

    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True
    )



# Chat history

if not st.session_state.messages:

    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">💬</div>
            <h3>Ask your first question</h3>
            <p>
                Try an analytical, documentation,
                or hybrid question.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("route")
        ):

            st.markdown(
                f"""
                <div class="route-badge">
                    Route: {message["route"]}
                </div>
                """,
                unsafe_allow_html=True
            )

            sources = message.get(
                "sources",
                []
            )

            if sources:

                with st.expander(
                    "📚 Sources"
                ):

                    for source in sources:

                        source_name = source.get(
                            "source",
                            "Unknown"
                        )

                        page = source.get(
                            "page"
                        )

                        if page is not None:

                            st.write(
                                f"• {source_name} "
                                f"— page {page + 1}"
                            )

                        else:

                            st.write(
                                f"• {source_name}"
                            )


# Chat input

question = st.chat_input(
    "Ask a question about your data..."
)


if question:

  
    # User message
   

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

 
    # Assistant response
   

    with st.chat_message("assistant"):

        try:

            with st.spinner(
                "Analyzing..."
            ):

                result = st.session_state.graph.invoke(
                    {
                        "messages": [
                            HumanMessage(
                                content=question
                            )
                        ],
                        "question": question,
                        "route": "",
                        "analytics_result": "",
                        "rag_result": "",
                        "final_answer": "",
                        "sources": []
                    }
                )

            answer = result.get(
                "final_answer",
                ""
            )

            route = result.get(
                "route",
                "UNKNOWN"
            )

            sources = result.get(
                "sources",
                []
            )

            if not answer:

                answer = (
                    "I could not generate an answer."
                )

            # Display answer

            st.markdown(answer)

            # Route

            st.markdown(
                f"""
                <div class="route-badge">
                    Route: {route}
                </div>
                """,
                unsafe_allow_html=True
            )

            # Sources

            if sources:

                with st.expander(
                    "📚 Sources"
                ):

                    for source in sources:

                        source_name = source.get(
                            "source",
                            "Unknown"
                        )

                        page = source.get(
                            "page"
                        )

                        if page is not None:

                            st.write(
                                f"• {source_name} "
                                f"— page {page + 1}"
                            )

                        else:

                            st.write(
                                f"• {source_name}"
                            )

            # Save response

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "route": route,
                    "sources": sources
                }
            )

            st.session_state.last_route = route
            st.session_state.last_sources = sources

        except Exception as e:

            error_message = (
                f"Something went wrong while "
                f"processing your question: {e}"
            )

            st.error(error_message)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "route": "ERROR",
                    "sources": []
                }
            )
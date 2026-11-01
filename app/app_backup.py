import os
import sys
import streamlit as st


# ------------------------------------------------
# FIX SRC PATH
# ------------------------------------------------

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from src.rag_engine import create_rag_engine


# ------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------

st.set_page_config(
    page_title="AI Research Paper Assistant",
    page_icon="📚",
    layout="wide"
)


# ------------------------------------------------
# TITLE
# ------------------------------------------------

st.title("📚 AI Research Paper Assistant")

st.caption(
    "RAG-powered research paper analysis and summarization"
)


# ------------------------------------------------
# SESSION STATE
# ------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "ask_question" not in st.session_state:
    st.session_state.ask_question = None

if "summarize_paper" not in st.session_state:
    st.session_state.summarize_paper = None

if "summary" not in st.session_state:
    st.session_state.summary = None

if "summary_sources" not in st.session_state:
    st.session_state.summary_sources = []


# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

with st.sidebar:

    st.header("📄 Research Papers")

    uploaded_files = st.file_uploader(
        "Upload your research papers",
        type=["pdf"],
        accept_multiple_files=True
    )


    # ------------------------------------------------
    # PROCESS PDFs
    # ------------------------------------------------

    if uploaded_files:

        current_files = [
            file.name
            for file in uploaded_files
        ]


        if current_files != st.session_state.processed_files:

            os.makedirs(
                "data/papers",
                exist_ok=True
            )

            pdf_paths = []


            for uploaded_file in uploaded_files:

                pdf_path = os.path.join(
                    "data",
                    "papers",
                    uploaded_file.name
                )


                with open(
                    pdf_path,
                    "wb"
                ) as f:

                    f.write(
                        uploaded_file.getbuffer()
                    )


                pdf_paths.append(
                    pdf_path
                )


            # ------------------------------------------------
            # BUILD RAG ENGINE
            # ------------------------------------------------

            with st.spinner(
                "🔄 Reading papers and building knowledge base..."
            ):

                try:

                    (
                        st.session_state.ask_question,
                        st.session_state.summarize_paper
                    ) = create_rag_engine(
                        pdf_paths
                    )

                except Exception as e:

                    st.error(
                        "❌ Could not process the research papers."
                    )

                    st.info(
                        "Please check the PDF files and try again."
                    )

                    st.stop()


            st.session_state.processed_files = (
                current_files
            )

            st.session_state.chat_history = []

            st.session_state.summary = None

            st.session_state.summary_sources = []


            st.success(
                f"✅ {len(pdf_paths)} research paper(s) are ready!"
            )


        else:

            st.success(
                f"✅ {len(uploaded_files)} research paper(s) are ready!"
            )


    # ------------------------------------------------
    # UPLOADED PAPERS
    # ------------------------------------------------

    if st.session_state.processed_files:

        st.divider()

        st.subheader("📚 Uploaded Papers")

        for file_name in st.session_state.processed_files:

            st.write(
                f"📄 {file_name}"
            )


        # ------------------------------------------------
        # SUMMARIZE BUTTON
        # ------------------------------------------------

        st.divider()

        st.subheader("📋 Paper Summary")

        if st.button(
            "📋 Summarize Research Paper",
            use_container_width=True
        ):

            with st.spinner(
                "🤖 Generating research paper summary..."
            ):

                try:

                    summary_result = (
                        st.session_state.summarize_paper()
                    )

                    st.session_state.summary = (
                        summary_result["summary"]
                    )

                    st.session_state.summary_sources = (
                        summary_result["sources"]
                    )

                except Exception as e:

                    st.error(
                        "❌ Could not generate the summary."
                    )

                    st.info(
                        "Please make sure Ollama is running "
                        "and try again."
                    )


    # ------------------------------------------------
    # CLEAR CHAT
    # ------------------------------------------------

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.chat_history = []

        st.rerun()


# ------------------------------------------------
# SUMMARY SECTION
# ------------------------------------------------

if st.session_state.summary:

    st.header("📋 Research Paper Summary")

    st.markdown(
        st.session_state.summary
    )


    # ------------------------------------------------
    # SUMMARY SOURCES
    # ------------------------------------------------

    if st.session_state.summary_sources:

        with st.expander(
            "📚 Summary Sources"
        ):

            for source in (
                st.session_state.summary_sources
            ):

                st.markdown(
                    f"📄 **{source['pdf']}** — "
                    f"Page **{source['page']}**"
                )


    st.divider()


# ------------------------------------------------
# MAIN CHAT
# ------------------------------------------------

if st.session_state.ask_question:

    st.subheader(
        "💬 Chat with Your Research Papers"
    )


    # ------------------------------------------------
    # DISPLAY CHAT HISTORY
    # ------------------------------------------------

    for chat in st.session_state.chat_history:

        # User message
        with st.chat_message("user"):

            st.write(
                chat["question"]
            )


        # Assistant message
        with st.chat_message("assistant"):

            st.markdown(
                "### 🤖 AI Answer"
            )

            st.markdown(
                chat["answer"]
            )


            # Sources
            if chat.get("sources"):

                st.markdown(
                    "### 📚 Sources"
                )

                for source in chat["sources"]:

                    st.markdown(
                        f"📄 **{source['pdf']}** — "
                        f"Page **{source['page']}**"
                    )


    # ------------------------------------------------
    # CHAT INPUT
    # ------------------------------------------------

    question = st.chat_input(
        "Ask something about your research papers..."
    )


    # ------------------------------------------------
    # NEW QUESTION
    # ------------------------------------------------

    if question:

        # User message
        with st.chat_message("user"):

            st.write(
                question
            )


        # Assistant message
        with st.chat_message("assistant"):

            with st.spinner(
                "🤔 Searching the papers..."
            ):

                try:

                    result = (
                        st.session_state.ask_question(
                            question,
                            st.session_state.chat_history
                        )
                    )

                except Exception as e:

                    st.error(
                        "❌ Something went wrong while "
                        "processing your question."
                    )

                    st.info(
                        "Please make sure Ollama is running "
                        "and try again."
                    )

                    st.stop()


            answer = result["answer"]

            sources = result["sources"]


            # ------------------------------------------------
            # AI ANSWER
            # ------------------------------------------------

            st.markdown(
                "### 🤖 AI Answer"
            )

            st.markdown(
                answer
            )


            # ------------------------------------------------
            # SOURCES
            # ------------------------------------------------

            if sources:

                st.markdown(
                    "### 📚 Sources"
                )

                for source in sources:

                    st.markdown(
                        f"📄 **{source['pdf']}** — "
                        f"Page **{source['page']}**"
                    )


        # ------------------------------------------------
        # SAVE CHAT
        # ------------------------------------------------

        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": answer,
                "sources": sources
            }
        )


else:

    if not st.session_state.summary:

        st.info(
            "👈 Upload one or more research papers "
            "from the sidebar to start."
        )
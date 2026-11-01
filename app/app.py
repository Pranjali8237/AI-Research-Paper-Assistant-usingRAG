
import os
import sys
import streamlit as st

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from src.rag_engine import create_rag_engine

st.set_page_config(
    page_title="AI Research Paper Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Research Paper Assistant")
st.caption("RAG-powered research paper analysis and summarization")

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

if "compare_papers" not in st.session_state:
    st.session_state.compare_papers = None

if "summary" not in st.session_state:
    st.session_state.summary = None

if "summary_sources" not in st.session_state:
    st.session_state.summary_sources = []

if "comparison_result" not in st.session_state:
    st.session_state.comparison_result = None

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

    if uploaded_files:

        current_files = [
            file.name for file in uploaded_files
        ]

        if current_files != st.session_state.processed_files:

            os.makedirs("data/papers", exist_ok=True)
            pdf_paths = []

            for uploaded_file in uploaded_files:

                pdf_path = os.path.join(
                    "data",
                    "papers",
                    uploaded_file.name
                )

                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                pdf_paths.append(pdf_path)

            with st.spinner(
                "🔄 Reading papers and building knowledge base..."
            ):

                try:
                    (
                        st.session_state.ask_question,
                        st.session_state.summarize_paper,
                        st.session_state.compare_papers
                    ) = create_rag_engine(pdf_paths)

                except Exception as error:
                    st.error(
                        "❌ Could not process the research papers."
                    )
                    st.exception(error)
                    st.stop()

            st.session_state.processed_files = current_files
            st.session_state.chat_history = []
            st.session_state.summary = None
            st.session_state.summary_sources = []
            st.session_state.comparison_result = None

            st.success(
                f"✅ {len(pdf_paths)} research paper(s) are ready!"
            )

        else:
            st.success(
                f"✅ {len(uploaded_files)} research paper(s) are ready!"
            )

    # --------------------------------------------
    # UPLOADED PAPERS
    # --------------------------------------------

    if st.session_state.processed_files:

        st.divider()
        st.subheader("📚 Uploaded Papers")

        for file_name in st.session_state.processed_files:
            st.write(f"📄 {file_name}")

        # ----------------------------------------
        # EXISTING SUMMARY FEATURE
        # ----------------------------------------

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

                except Exception as error:
                    st.error("❌ Could not generate the summary.")
                    st.exception(error)

        # ----------------------------------------
        # NEW PAPER COMPARISON FEATURE
        # ----------------------------------------

        st.divider()
        st.subheader("🔍 Compare Research Papers")

        if len(st.session_state.processed_files) < 2:

            st.info(
                "Upload at least 2 research papers to compare them."
            )

        else:

            paper_options = st.session_state.processed_files

            paper_1 = st.selectbox(
                "Select Paper 1",
                options=paper_options,
                index=0,
                key="comparison_paper_1"
            )

            available_paper_2 = [
                name for name in paper_options
                if name != paper_1
            ]

            paper_2 = st.selectbox(
                "Select Paper 2",
                options=available_paper_2,
                index=0,
                key="comparison_paper_2"
            )

            if st.button(
                "🔍 Compare Research Papers",
                use_container_width=True
            ):

                with st.spinner(
                    "🤖 Comparing both research papers..."
                ):

                    try:
                        result = (
                            st.session_state.compare_papers(
                                paper_1,
                                paper_2
                            )
                        )

                        st.session_state.comparison_result = result

                    except Exception as error:
                        st.error(
                            "❌ Could not compare the papers."
                        )
                        st.exception(error)

        # ----------------------------------------
        # CLEAR CHAT
        # ----------------------------------------

        st.divider()

        if st.button(
            "🗑️ Clear Chat",
            use_container_width=True
        ):

            st.session_state.chat_history = []
            st.rerun()

# ------------------------------------------------
# SUMMARY DISPLAY
# Existing summary display format preserved
# ------------------------------------------------

if st.session_state.summary:

    st.header("📋 Research Paper Summary")

    st.caption(
        "AI-generated structured summary based only "
        "on the uploaded research paper"
    )

    st.markdown(st.session_state.summary)

    if st.session_state.summary_sources:

        with st.expander("📚 Summary Sources"):

            for source in st.session_state.summary_sources:

                st.markdown(
                    f"📄 **{source['pdf']}** — "
                    f"Page **{source['page']}**"
                )

    st.divider()

# ------------------------------------------------
# COMPARISON RESULT DISPLAY
# ------------------------------------------------

if st.session_state.comparison_result:

    st.header("🔍 Research Paper Comparison")

    result = st.session_state.comparison_result

    st.caption(
        f"Paper 1: {result['papers'][0]}  |  "
        f"Paper 2: {result['papers'][1]}"
    )

    st.markdown(result["comparison"])
    st.divider()

# ------------------------------------------------
# CHAT WITH RESEARCH PAPERS
# ------------------------------------------------

if st.session_state.ask_question:

    st.subheader("💬 Chat with Your Research Papers")

    for chat in st.session_state.chat_history:

        with st.chat_message("user"):
            st.write(chat["question"])

        with st.chat_message("assistant"):

            st.markdown("### 🤖 AI Answer")
            st.markdown(chat["answer"])

            if chat.get("sources"):

                st.markdown("### 📚 Sources")

                for source in chat["sources"]:

                    st.markdown(
                        f"📄 **{source['pdf']}** — "
                        f"Page **{source['page']}**"
                    )

    question = st.chat_input(
        "Ask something about your research papers..."
    )

    if question:

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):

            with st.spinner("🤔 Searching the papers..."):

                try:
                    result = st.session_state.ask_question(
                        question,
                        st.session_state.chat_history
                    )

                except Exception as error:
                    st.error(
                        "❌ Something went wrong while "
                        "processing your question."
                    )
                    st.exception(error)
                    st.stop()

            answer = result["answer"]
            sources = result["sources"]

            st.markdown("### 🤖 AI Answer")
            st.markdown(answer)

            if sources:

                st.markdown("### 📚 Sources")

                for source in sources:

                    st.markdown(
                        f"📄 **{source['pdf']}** — "
                        f"Page **{source['page']}**"
                    )

        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": answer,
                "sources": sources
            }
        )

else:

    if (
        not st.session_state.summary
        and not st.session_state.comparison_result
    ):
        st.info(
            "👈 Upload one or more research papers "
            "from the sidebar to start."
        )

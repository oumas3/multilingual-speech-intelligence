import sys
import tempfile
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


st.set_page_config(
    page_title="Multilingual Speech Intelligence",
    page_icon="🎙️",
    layout="wide"
)


st.title("🎙️ Multilingual Speech Intelligence")

st.write(
    """
    Explore speech with Whisper and WavLM,
    and interact with research documents using
    Retrieval-Augmented Generation.
    """
)


speech_tab, document_tab = st.tabs(
    [
        "🎙️ Speech Intelligence",
        "📄 Document Intelligence"
    ]
)


# =================================================
# SPEECH
# =================================================

with speech_tab:

    st.header("Speech Intelligence")

    st.write(
        """
        Upload audio to transcribe speech,
        extract WavLM representations,
        and analyze the transcript.
        """
    )

    uploaded_audio = st.file_uploader(
        "Upload audio",
        type=[
            "wav",
            "mp3",
            "m4a",
            "flac"
        ],
        key="speech_audio"
    )

    if uploaded_audio:

        st.audio(uploaded_audio)

        if st.button(
            "Analyze Audio",
            type="primary"
        ):

            suffix = Path(
                uploaded_audio.name
            ).suffix

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp:

                temp.write(
                    uploaded_audio.getbuffer()
                )

                audio_path = temp.name

            try:

                # -------------------------
                # Whisper
                # -------------------------

                from transcribe import (
                    transcribe_audio
                )

                with st.spinner(
                    "Transcribing with Whisper..."
                ):

                    transcript = (
                        transcribe_audio(
                            audio_path
                        )
                    )

                st.subheader("Transcript")

                st.success(transcript)


                # -------------------------
                # WavLM
                # -------------------------

                from audio_embeddings import (
                    extract_audio_embeddings
                )

                with st.spinner(
                    "Extracting WavLM features..."
                ):

                    audio_embedding = (
                        extract_audio_embeddings(
                            audio_path
                        )
                    )

                st.subheader(
                    "WavLM Speech Representation"
                )

                col1, col2 = st.columns(2)

                col1.metric(
                    "Embedding dimensions",
                    int(
                        audio_embedding.shape[0]
                    )
                )

                col2.metric(
                    "Model",
                    "WavLM Base+"
                )

                with st.expander(
                    "Inspect embedding"
                ):

                    values = (
                        audio_embedding[:20]
                        .detach()
                        .cpu()
                        .numpy()
                        .tolist()
                    )

                    st.write(values)


                # -------------------------
                # Qwen
                # -------------------------

                from llm import (
                    analyze_transcript
                )

                with st.spinner(
                    "Analyzing transcript..."
                ):

                    analysis = (
                        analyze_transcript(
                            transcript
                        )
                    )

                st.subheader(
                    "Transcript Analysis"
                )

                st.info(analysis)

            except Exception as error:

                st.error(
                    f"Audio analysis failed: "
                    f"{error}"
                )


# =================================================
# DOCUMENT INTELLIGENCE
# =================================================

with document_tab:

    st.header(
        "Document Intelligence"
    )

    st.write(
        """
        Upload a PDF and ask questions about it.
        The document is chunked, embedded,
        retrieved, reranked, and passed to the
        language model as grounded context.
        """
    )

    uploaded_pdf = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        key="rag_pdf"
    )


    if uploaded_pdf:

        st.success(
            f"Selected: {uploaded_pdf.name}"
        )

        # Detect PDF changes
        pdf_id = (
            uploaded_pdf.name,
            uploaded_pdf.size
        )

        if (
            st.session_state.get(
                "current_pdf_id"
            )
            != pdf_id
        ):

            st.session_state.pop(
                "knowledge_base",
                None
            )

            st.session_state[
                "current_pdf_id"
            ] = pdf_id


        if st.button(
            "Process PDF",
            type="primary"
        ):

            try:

                from rag import (
                    build_knowledge_base
                )

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp:

                    temp.write(
                        uploaded_pdf.getbuffer()
                    )

                    pdf_path = temp.name


                with st.spinner(
                    "Reading, chunking and "
                    "embedding the PDF..."
                ):

                    knowledge_base = (
                        build_knowledge_base(
                            pdf_path
                        )
                    )

                # Preserve real uploaded filename
                knowledge_base[
                    "source"
                ] = uploaded_pdf.name

                st.session_state[
                    "knowledge_base"
                ] = knowledge_base


                st.success(
                    "Knowledge base created!"
                )

                st.metric(
                    "Document chunks",
                    len(
                        knowledge_base[
                            "chunks"
                        ]
                    )
                )

            except Exception as error:

                st.error(
                    f"PDF processing failed: "
                    f"{error}"
                )


        if (
            "knowledge_base"
            in st.session_state
        ):

            st.divider()

            st.subheader(
                "Ask the document"
            )

            question = st.text_input(
                "Question",
                placeholder=(
                    "What methodology "
                    "does this paper use?"
                )
            )


            if st.button(
                "Ask Question"
            ):

                if not question.strip():

                    st.warning(
                        "Enter a question first."
                    )

                else:

                    try:

                        from rag import (
                            rag_answer
                        )

                        with st.spinner(
                            "Retrieving and "
                            "reranking evidence..."
                        ):

                            result = (
                                rag_answer(
                                    question,
                                    st.session_state[
                                        "knowledge_base"
                                    ]
                                )
                            )


                        st.subheader(
                            "Answer"
                        )

                        st.success(
                            result["answer"]
                        )


                        st.subheader(
                            "Source"
                        )

                        st.write(
                            f"📄 "
                            f"{result['source']}"
                        )


                        with st.expander(
                            "View retrieved evidence"
                        ):

                            st.write(
                                result[
                                    "context"
                                ]
                            )


                        with st.expander(
                            "View retrieval scores"
                        ):

                            for i, item in enumerate(
                                result["results"],
                                start=1
                            ):

                                st.write(
                                    f"Result {i}"
                                )

                                st.write(
                                    "Similarity:",
                                    round(
                                        item["score"],
                                        4
                                    )
                                )

                                st.write(
                                    "Reranker:",
                                    round(
                                        item[
                                            "rerank_score"
                                        ],
                                        4
                                    )
                                )

                                st.write(
                                    item["text"]
                                )

                                st.divider()


                    except Exception as error:

                        st.error(
                            f"Question answering "
                            f"failed: {error}"
                        )


st.divider()

st.caption(
    "Whisper • WavLM • Qwen • "
    "Sentence Transformers • "
    "CrossEncoder • RAG"
)
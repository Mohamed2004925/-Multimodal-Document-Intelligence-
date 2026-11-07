"""Streamlit web interface for the document OCR pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

import app as pipeline


st.set_page_config(page_title="Document Intelligence", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading the document layout model...")
def get_model(weights_path: str):
    return pipeline.load_model(Path(weights_path))


st.title("Document Intelligence")
st.write("Upload an image or PDF to detect document regions, run OCR, and extract fields.")

with st.sidebar:
    st.header("Processing settings")
    weights_path = st.text_input("YOLO model weights", value=str(pipeline.DEFAULT_WEIGHTS))
    confidence = st.slider("Detection confidence", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
    image_size = st.selectbox("Model image size", options=[640, 1024], index=0)
    pdf_dpi = st.slider("PDF rendering resolution (DPI)", min_value=100, max_value=300, value=150, step=25)

uploaded_file = st.file_uploader(
    "Choose an image or PDF",
    type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp", "pdf"],
    accept_multiple_files=False,
)

if uploaded_file is not None:
    st.info(f"Selected: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")
    if st.button("Analyze document", type="primary", use_container_width=True):
        st.session_state.pop("document_result", None)
        st.session_state.pop("question_answers", None)
        if pipeline.configure_tesseract() is None:
            st.error("Tesseract was not found. Install Tesseract and restart the app.")
            st.stop()

        weights = Path(weights_path).expanduser()
        if not weights.is_file():
            st.error(f"Model weights file not found: {weights}")
            st.stop()

        try:
            suffix = Path(uploaded_file.name).suffix.lower()
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
                temporary.write(uploaded_file.getvalue())
                temp_path = Path(temporary.name)
            try:
                model = get_model(str(weights.resolve()))
                with st.spinner("Analyzing the document..."):
                    result = pipeline.process_document(
                        temp_path,
                        model,
                        confidence=confidence,
                        imgsz=image_size,
                        pdf_dpi=pdf_dpi,
                    )
            finally:
                temp_path.unlink(missing_ok=True)
            st.session_state["document_result"] = result
        except Exception as exc:
            st.error(f"Could not process this document: {exc}")

result = st.session_state.get("document_result")
if result:
    st.divider()
    st.subheader("Extracted fields")
    if result.get("needs_review"):
        st.warning("This document needs review. One or more required fields are missing or uncertain.")
    else:
        st.success("Required fields were extracted. Please verify them against the source document.")

    fields = result.get("fields", {})
    columns = st.columns(min(max(len(fields), 1), 3))
    for index, (name, field) in enumerate(fields.items()):
        with columns[index % len(columns)]:
            status = field.get("extraction_status", "unknown")
            st.markdown(f"**{name.replace('_', ' ').title()}** · `{status}`")
            st.write(field.get("value") or "—")
            if field.get("evidence"):
                st.caption(f"Evidence: {field['evidence']}")

    st.subheader("Recognized text")
    st.text_area("OCR output", value=result.get("full_text", ""), height=260, label_visibility="collapsed")

    st.subheader("Ask about this document")
    with st.form("document_question_form", clear_on_submit=True):
        question = st.text_input("Your question", placeholder="What is the total amount?")
        ask = st.form_submit_button("Ask", type="primary")
    if ask:
        answer = pipeline.answer_question(question, result)
        st.session_state.setdefault("question_answers", []).append(answer)
    for answer in reversed(st.session_state.get("question_answers", [])):
        st.markdown(f"**Q: {answer['question']}**")
        if answer["answer"] is not None:
            st.write(answer["answer"])
        else:
            st.info(answer["message"] or "No answer found.")
        if answer.get("evidence"):
            st.caption(f"Evidence: {answer['evidence']}")
        st.caption(f"Answer status: {answer['status']}")

    with st.expander("Detected regions"):
        for page in result.get("pages", []):
            st.markdown(f"**Page {page.get('page_number', '?')}**")
            regions = page.get("regions", [])
            if regions:
                st.dataframe(
                    [
                        {
                            "Region": region.get("layout_label"),
                            "Detection confidence": region.get("layout_confidence"),
                            "Text": region.get("text", ""),
                            "Bounding box": region.get("bbox_xyxy"),
                        }
                        for region in regions
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.write("No text regions detected.")

    result_with_answers = dict(result)
    result_with_answers["question_answers"] = st.session_state.get("question_answers", [])
    result_with_answers = pipeline.DocumentResult.model_validate(result_with_answers).model_dump(mode="python")
    rendered_json = json.dumps(result_with_answers, indent=2, ensure_ascii=False)
    st.download_button(
        "Download structured JSON",
        data=rendered_json.encode("utf-8"),
        file_name=f"{Path(result['source_file']).stem}_result.json",
        mime="application/json",
        use_container_width=True,
    )
    with st.expander("View complete JSON"):
        st.code(rendered_json, language="json")
else:
    st.caption("Your file stays in this local app session and is processed by the model and Tesseract installed on this computer.")

with st.expander("About the system and its limitations"):
    st.markdown(
        """
        - Layout detection uses a general DocLayNet YOLO **detection** model, not an invoice-specific or segmentation model.
        - OCR uses Tesseract. Recognition quality depends on scan quality, language, font, and page layout.
        - Field extraction and question answering use conservative rules and OCR evidence; they do not use an LLM and can miss paraphrases or values.
        - Answers quote a field or matching OCR line. The app returns an uncertain or not-found status when it cannot identify unique supporting evidence.
        - Tables are OCRed as regions; rows and columns are not reconstructed. Multi-column reading order may be incorrect.
        - Check extracted values and answers against the original document before relying on them.
        """
    )

import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db
from medquad_helper import get_medical_qa_chain, create_medical_vector_db
from arxiv_helper import get_arxiv_qa_chain, create_arxiv_vector_db, search_papers
from multimodal_helper import analyze_image, chat_with_image, generate_text_explanation
from sentiment_helper import detect_sentiment
from language_helper import (
    detect_language, get_language_name,
    get_language_sentiment_prefix, SUPPORTED_LANGUAGES
)
from entity_recognition import extract_medical_entities, has_medical_entities, format_entities_for_display
from knowledge_updater import (
    update_nullclass_knowledge, update_medical_knowledge,
    auto_update_if_needed, get_last_updated, start_scheduler
)
from visualizer import extract_keywords, get_category_labels, build_bar_chart_data
import csv
import io
from collections import Counter

# Start background scheduler for periodic external updates
start_scheduler(interval_minutes=30)

st.title("CUSTOMER SERVICE & RESEARCH CHATBOT 🤖")

# Mode selector
mode = st.selectbox(
    "Select Mode:",
    [
        "🏫 Customer Service (Nullclass)",
        "🏥 Medical Q&A (MedQuAD)",
        "🔬 Research Expert (arXiv CS)",
        "🎨 Multi-Modal Chat"
    ]
)

st.divider()

# ── KNOWLEDGE BASE MANAGEMENT ──
if "Multi-Modal" not in mode:
    with st.expander("⚙️ Knowledge Base Management"):
        current_mode = "nullclass" if "Nullclass" in mode else "medical" if "Medical" in mode else "arxiv"

        if current_mode != "arxiv":
            last_updated, row_count, last_fetch = get_last_updated(current_mode)
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"🕒 Last updated: {last_updated if last_updated else 'Never'}")
            with col2:
                st.caption(f"📄 Documents indexed: {row_count if row_count else 0}")
            if last_fetch:
                st.caption(f"🌐 Last external fetch: {last_fetch}")

        st.subheader("Create / Rebuild Knowledgebase")
        btn = st.button("🔨 Create Knowledgebase")
        if btn:
            with st.spinner("Building knowledgebase... this may take a few minutes."):
                try:
                    if "Nullclass" in mode:
                        create_vector_db()
                    elif "Medical" in mode:
                        create_medical_vector_db()
                    else:
                        create_arxiv_vector_db()
                    st.success("✅ Knowledgebase created successfully!")
                except FileNotFoundError as e:
                    st.error(str(e))

        if current_mode != "arxiv":
            st.divider()
            st.subheader("📤 Upload New Data")
            st.caption("Upload a CSV file with columns: `prompt`, `response`")
            uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

            if uploaded_file is not None:
                try:
                    content = uploaded_file.read().decode("utf-8", errors="ignore")
                    reader = csv.DictReader(io.StringIO(content))
                    new_rows = [row for row in reader]
                    st.info(f"Found {len(new_rows)} rows in uploaded file.")

                    if st.button("➕ Add to Knowledge Base"):
                        with st.spinner("Merging and rebuilding index..."):
                            if "Nullclass" in mode:
                                added, total = update_nullclass_knowledge(new_rows)
                            else:
                                added, total = update_medical_knowledge(new_rows)

                        if added > 0:
                            st.success(f"✅ Added {added} new entries. Total documents: {total}")
                        else:
                            st.warning("⚠️ No new unique entries found.")

                except Exception as e:
                    st.error(f"Error reading file: {e}")

            st.divider()
            st.subheader("🔄 Auto Update")
            auto_col1, auto_col2 = st.columns(2)
            with auto_col1:
                if st.button("🔍 Check & Auto Update Now"):
                    with st.spinner("Checking for updates..."):
                        updated, message = auto_update_if_needed(current_mode)
                    if updated:
                        st.success(f"✅ {message}")
                    else:
                        st.info(f"ℹ️ {message}")
            with auto_col2:
                refresh_interval = st.selectbox(
                    "Auto-refresh interval",
                    ["Off", "5 minutes", "30 minutes", "1 hour"],
                    index=0
                )

            if refresh_interval != "Off":
                intervals = {"5 minutes": 300, "30 minutes": 1800, "1 hour": 3600}
                seconds = intervals[refresh_interval]
                st.markdown(
                    f"""<script>setTimeout(function() {{window.location.reload();}}, {seconds * 1000});</script>""",
                    unsafe_allow_html=True
                )
                updated, message = auto_update_if_needed(current_mode)
                if updated:
                    st.success(f"🔄 Auto-updated: {message}")

    st.divider()

# ── MULTI-MODAL MODE ──
if "Multi-Modal" in mode:
    st.subheader("🎨 Multi-Modal Chat")
    st.caption("Upload an image and ask questions about it.")

    if "multimodal_chat_history" not in st.session_state:
        st.session_state.multimodal_chat_history = []
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "gif", "webp"],
        key="image_upload"
    )

    if uploaded_image:
        st.session_state.uploaded_image = uploaded_image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        if len(st.session_state.multimodal_chat_history) == 0:
            with st.spinner("Analyzing image..."):
                analysis = analyze_image(uploaded_image)
            st.session_state.multimodal_chat_history.append({
                "role": "assistant",
                "content": f"**Image Analysis:**\n\n{analysis}"
            })

    st.divider()

    for msg in st.session_state.multimodal_chat_history:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").markdown(msg["content"])

    question = st.chat_input("Ask a question about the image or any topic...")

    if question:
        # Detect language
        lang_code = detect_language(question)
        lang_name = get_language_name(lang_code)

        st.chat_message("user").write(question)
        st.session_state.multimodal_chat_history.append({"role": "user", "content": question})

        if lang_code != "en":
            st.caption(f"🌐 Detected language: {lang_name}")

        with st.spinner("Thinking..."):
            if st.session_state.uploaded_image:
                answer = chat_with_image(st.session_state.uploaded_image, st.session_state.multimodal_chat_history)
            else:
                answer = generate_text_explanation(question)

        st.chat_message("assistant").markdown(answer)
        st.session_state.multimodal_chat_history.append({"role": "assistant", "content": answer})

    if st.button("🗑️ Clear Chat & Image"):
        st.session_state.multimodal_chat_history = []
        st.session_state.uploaded_image = None
        st.rerun()

# ── ARXIV MODE ──
elif "arXiv" in mode:
    if "arxiv_chat_history" not in st.session_state:
        st.session_state.arxiv_chat_history = []

    st.subheader("🔎 Search Research Papers")
    search_query = st.text_input("Search for papers by topic or keyword:", key="search")

    if search_query:
        with st.spinner("Searching papers..."):
            papers = search_papers(search_query, k=5)

        st.markdown(f"**Top {len(papers)} papers found:**")
        abstracts = []
        categories_list = []

        for i, paper in enumerate(papers):
            with st.expander(f"📄 {paper['title']}"):
                st.markdown(f"**Authors:** {paper['authors']}")
                st.markdown(f"**Categories:** {paper['categories']}")
                if paper['id']:
                    st.markdown(f"**arXiv ID:** [{paper['id']}](https://arxiv.org/abs/{paper['id']})")
                st.markdown("**Abstract:**")
                st.write(paper['abstract'])
            abstracts.append(paper['abstract'])
            categories_list.append(paper['categories'])

        st.divider()
        st.subheader("📊 Concept Visualization")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top Keywords in Results**")
            keywords = extract_keywords(abstracts, top_n=10)
            words, counts = build_bar_chart_data(keywords)
            st.bar_chart(dict(zip(words, counts)))
        with col2:
            st.markdown("**Research Areas**")
            labels = get_category_labels(categories_list)
            if labels:
                st.bar_chart(dict(Counter(labels)))
            else:
                st.info("No category data available.")

    st.divider()
    st.subheader("💬 Ask the Research Expert")
    st.caption("Supports English, Malayalam, Arabic, Spanish and Hindi.")

    for msg in st.session_state.arxiv_chat_history:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])

    question = st.chat_input("Ask a research question in any supported language...")

    if question:
        lang_code = detect_language(question)
        lang_name = get_language_name(lang_code)

        st.chat_message("user").write(question)
        st.session_state.arxiv_chat_history.append({"role": "user", "content": question})

        if lang_code != "en":
            st.caption(f"🌐 Detected language: {lang_name}")

        if len(st.session_state.arxiv_chat_history) > 2:
            last_exchange = st.session_state.arxiv_chat_history[-3:-1]
            full_query = " ".join([m["content"] for m in last_exchange]) + " " + question
        else:
            full_query = question

        with st.spinner("Thinking..."):
            chain = get_arxiv_qa_chain(lang_code=lang_code)
            response = chain(full_query)
            answer = response["result"]

        st.chat_message("assistant").write(answer)
        st.session_state.arxiv_chat_history.append({"role": "assistant", "content": answer})

        if st.button("🗑️ Clear Chat History"):
            st.session_state.arxiv_chat_history = []
            st.rerun()

# ── NULLCLASS MODE ──
elif "Nullclass" in mode:
    st.caption("💬 Supports English, Malayalam, Arabic, Spanish and Hindi.")
    question = st.text_input("Ask a question:")

    if question:
        lang_code = detect_language(question)
        lang_name = get_language_name(lang_code)
        sentiment = detect_sentiment(question)
        prefix = get_language_sentiment_prefix(sentiment, lang_code)

        if lang_code != "en":
            st.info(f"🌐 Detected language: {lang_name}")

        with st.spinner("Thinking..."):
            chain = get_qa_chain(lang_code=lang_code)
            response = chain(question)
            final_answer = prefix + response["result"]

        st.header("Answer")
        st.write(final_answer)
        st.caption(f"Detected sentiment: `{sentiment}` | Language: `{lang_name}`")

# ── MEDICAL MODE ──
else:
    st.caption("💬 Supports English, Malayalam, Arabic, Spanish and Hindi.")
    question = st.text_input("Ask a question:")

    if question:
        lang_code = detect_language(question)
        lang_name = get_language_name(lang_code)
        sentiment = detect_sentiment(question)
        prefix = get_language_sentiment_prefix(sentiment, lang_code)

        if lang_code != "en":
            st.info(f"🌐 Detected language: {lang_name}")

        entities = extract_medical_entities(question)
        if has_medical_entities(entities):
            st.subheader("🔍 Medical Entities Detected")
            for line in format_entities_for_display(entities):
                st.markdown(line)
            st.divider()

        with st.spinner("Thinking..."):
            chain = get_medical_qa_chain(lang_code=lang_code)
            response = chain(question)
            final_answer = prefix + response["result"]

        st.header("Medical Answer")
        st.write(final_answer)
        st.warning("⚠️ This information is for educational purposes only. Always consult a qualified healthcare professional for personal medical advice.")
        st.caption(f"Detected sentiment: `{sentiment}` | Language: `{lang_name}`")

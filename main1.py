import streamlit as st
try:
    from streamlit_extras.colored_header import colored_header
except Exception:
    # Fallback simple implementation to avoid ImportError if the package is missing
    def colored_header(title, description="", color_name="blue"):
        color_map = {
            "violet-70": "#A78BFA",
            "blue": "#4C9AFF",
            "green": "#34D399"
        }
        color = color_map.get(color_name, color_name)
        st.markdown(f"<h2 style='color:{color}; margin:0'>{title}</h2>", unsafe_allow_html=True)
        if description:
            st.markdown(f"<p style='color:gray; margin-top:0'>{description}</p>", unsafe_allow_html=True)

from preprocessor import preprocess_text, load_model, build_matcher, extract_triples, normalize_entity, NeoGraph
import pdfplumber
from docx import Document
import time
import streamlit as st

st.set_page_config(page_title="NeuroTrait Graph", layout="wide")

# --- HEADER ---
colored_header("🧠 NeuroTrait Knowledge Graph Generator", 
               description="Explore how human traits, environments, and neural patterns interconnect.",
               color_name="violet-70")

# --- LAYOUT COLUMNS ---
col1, col2, col3 = st.columns([1, 2, 2])

# --- LEFT PANEL: INPUT ---
with col1:
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "docx"])
    uploaded_file_doc = uploaded_file.name.lower().endswith(('.docx', '.pdf', '.txt')) if uploaded_file is not None else False
    st.markdown("### Actions")

    def extract_text_from_file(uploaded_file):
        if uploaded_file.type == "text/plain":
            return str(uploaded_file.read(), "utf-8")
        elif uploaded_file.type == "application/pdf":
            with pdfplumber.open(uploaded_file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text()
                return text
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(uploaded_file)
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            return text
        else:
            return ""

    preprocess_btn = st.button("🔧 Preprocess", disabled=not uploaded_file_doc)
    # Neo4j connection inputs (defaults may be overridden)
    neo_uri = st.text_input("Neo4j URI", value="bolt://localhost:7687")
    neo_user = st.text_input("Neo4j User", value="neo4j")
    neo_pwd = st.text_input("Neo4j Password", type="password", value="password")
    # sending files to preprocess only when button is clicked
    if preprocess_btn:
        # Create per-run progress UI elements
        progress_text = st.empty()
        progress_bar = st.progress(0)

        # Stage 1: load model
        progress_text.text("🔄 Loading NLP model...")
        with st.spinner("Loading NLP model (this can take a while for large models)..."):
            try:
                nlp = load_model()
            except Exception:
                # If model fails to load, still attempt preprocessing (preprocess_text may handle it)
                nlp = None
        progress_bar.progress(20)

        # Stage 2: extract raw text from uploaded file
        progress_text.text("🔄 Extracting text from file...")
        text = extract_text_from_file(uploaded_file)
        progress_bar.progress(40)

        # Stage 3: run preprocessing (cleaning + entity extraction)
        progress_text.text("🔄 Cleaning text and extracting entities...")
        cleaned_text, entities = preprocess_text(text)
        progress_bar.progress(85)

        # Stage 4: finalize
        st.session_state.cleaned_text = cleaned_text
        st.session_state.entities = entities
        st.session_state.raw_text = text
        progress_bar.progress(100)
        progress_text.text("✅ Preprocessing complete!")
        st.balloons()

    # only enable graph button if a valid file is preprocessed
    if preprocess_btn:
        uploaded_file_doc = True
        graph_btn = st.button("🕸️ Generate Graph", disabled=not uploaded_file_doc)
        st.progress(0)

        if graph_btn:
            # Ensure we have the raw text (from preprocessing or freshly extracted)
            raw = st.session_state.get("raw_text")
            if raw is None:
                raw = extract_text_from_file(uploaded_file)

            # Progress UI
            gen_status = st.empty()
            gen_progress = st.progress(0)

            gen_status.text("Loading model and matcher...")
            with st.spinner("Loading NLP model and building matcher..."):
                nlp = load_model()
                matcher = build_matcher(nlp)
            gen_progress.progress(10)

            gen_status.text("Extracting triples...")
            triples = extract_triples(nlp, matcher, raw)
            gen_progress.progress(40)

            gen_status.text("Normalizing entities and writing to Neo4j...")
            graph = NeoGraph(uri=neo_uri, user=neo_user, pwd=neo_pwd)
            inserted = 0
            for i, t in enumerate(triples):
                subj_norm, subj_cui, _ = normalize_entity(nlp, t["subject"])
                obj_norm, obj_cui, _ = normalize_entity(nlp, t["object"])

                def label_for(term):
                    term_l = term.lower()
                    if any(x in term_l for x in ["hippocamp", "cortex", "amygdala", "pfc"]):
                        return "NEURAL_REGION"
                    if any(x in term_l for x in ["theta", "activation", "spike", "oscillation"]):
                        return "NEURAL_PATTERN"
                    if any(x in term_l for x in ["stress", "novelty", "reward", "environment"]):
                        return "ENVIRONMENT"
                    return "TRAIT"

                s_label = label_for(subj_norm)
                o_label = label_for(obj_norm)

                graph.upsert_node(s_label, subj_norm, subj_cui)
                graph.upsert_node(o_label, obj_norm, obj_cui)
                graph.upsert_relation(s_label, subj_norm, o_label, obj_norm, t["relation"], t["dir"], t["confidence"], t["sentence"])
                inserted += 1
                if triples:
                    gen_progress.progress(40 + int(50 * (i + 1) / len(triples)))

            graph.close()
            gen_progress.progress(100)
            gen_status.text(f"Done — inserted {inserted} triples into Neo4j.")
            st.success("✅ Graph generated and written to Neo4j.")

# --- MIDDLE PANEL: TEXT PREVIEW ---
with col2:
    st.subheader("Processed Text")
    st.info("Preview will appear here after preprocessing.")
    st.markdown("**Trait:** <span style='color: #4C9AFF'>Blue</span>  |  "
                "**Environment:** <span style='color: #34D399'>Green</span>  |  "
                "**Neural Pattern:** <span style='color: #A78BFA'>Violet</span>", unsafe_allow_html=True)
    if 'cleaned_text' in st.session_state:
        st.success("✅ Preprocessing Complete!")
        st.text_area("Cleaned and Annotated Text", value=st.session_state.cleaned_text, height=500)
        if 'entities' in st.session_state and st.session_state.entities:
            st.subheader("Detected Entities")
            st.write(st.session_state.entities)
    else:
        st.text_area("Cleaned and Annotated Text", placeholder="Waiting for input...", height=500)


# --- RIGHT PANEL: GRAPH AREA ---
with col3:
    st.subheader("Knowledge Graph View")
    st.info("Interactive graph will appear here.")
    st.empty()  

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
    
    # Extract text as soon as file is uploaded
    if uploaded_file is not None:
        try:
            raw_text = extract_text_from_file(uploaded_file)
            st.session_state.raw_text = raw_text
            if len(raw_text.strip()) == 0:
                st.error("⚠️ No text could be extracted from the uploaded file.")
                st.session_state.raw_text = None
            else:
                st.success("✅ Text extracted successfully.")
        except Exception as e:
            st.error(f"⚠️ Error extracting text: {str(e)}")
            st.session_state.raw_text = None
    
    st.markdown("### Actions")

    # Function moved to top of file

    # Add preprocess button
    preprocess_btn = st.button("🔧 Preprocess", disabled=not (uploaded_file_doc and 'raw_text' in st.session_state))
    
    # Run preprocessing when button is clicked
    if preprocess_btn and 'raw_text' in st.session_state:
        # Create per-run progress UI elements
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        progress_text.text("🔄 Loading NLP model...")
        try:
            with st.spinner("Loading NLP model and preprocessing..."):
                nlp = load_model()
                progress_bar.progress(30)
                
                # Run preprocessing on the extracted text
                progress_text.text("🔄 Analyzing text...")
                cleaned_text, entities = preprocess_text(st.session_state.raw_text)
                progress_bar.progress(90)
                
                # Store results
                st.session_state.cleaned_text = cleaned_text
                st.session_state.entities = entities
                progress_bar.progress(100)
                progress_text.text("✅ Analysis complete!")
                st.success("✅ Preprocessing complete!")
                st.balloons()
        except Exception as e:
            st.error(f"⚠️ Error during preprocessing: {str(e)}")
            if "No module named 'en_core_sci" in str(e):
                st.warning("📦 The scientific NLP model is not installed. Run:\n```\npip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_lg-0.5.1.tar.gz\n```")
    
    # Neo4j connection inputs (defaults may be overridden)
    neo_uri = st.text_input("Neo4j URI", value="bolt://localhost:7687")
    neo_user = st.text_input("Neo4j User", value="neo4j")
    neo_pwd = st.text_input("Neo4j Password", type="password", value="password")
    
    # Add Neo4j connection test button
    if st.button("🔌 Test Neo4j Connection"):
        try:
            with st.spinner("Testing connection..."):
                graph = NeoGraph(uri=neo_uri, user=neo_user, pwd=neo_pwd)
                graph.close()
            st.success("✅ Successfully connected to Neo4j!")
        except Exception as e:
            st.error(f"❌ Failed to connect to Neo4j: {str(e)}")
            if "Connection refused" in str(e):
                st.warning("Make sure Neo4j is running and accessible at the specified URI.")
            elif "unauthorized" in str(e).lower():
                st.warning("Check your username and password.")
    
    # Enable graph generation if text is preprocessed
    can_generate_graph = 'cleaned_text' in st.session_state and st.session_state.cleaned_text is not None
            
    # Enable graph generation if text is preprocessed
    graph_btn = st.button("🕸️ Generate Graph", disabled=not can_generate_graph)
    
    if graph_btn:
            # Ensure we have the raw text (from preprocessing or freshly extracted)
            raw = st.session_state.get("raw_text")
            if not raw:
                st.error("⚠️ No text available for processing. Please upload and preprocess a document first.")
                st.stop()

            # Test Neo4j connection before starting
            try:
                with st.spinner("Testing Neo4j connection..."):
                    graph = NeoGraph(uri=neo_uri, user=neo_user, pwd=neo_pwd)
                    graph.close()
            except Exception as e:
                st.error(f"❌ Cannot connect to Neo4j: {str(e)}")
                st.warning("Please check your Neo4j connection settings and try again.")
                st.stop()

            # Progress UI
            gen_status = st.empty()
            gen_progress = st.progress(0)

            gen_status.text("Loading model and matcher...")
            with st.spinner("Loading NLP model and building matcher..."):
                nlp = load_model()
                matcher = build_matcher(nlp)
            gen_progress.progress(10)

            try:
                gen_status.text("Extracting triples...")
                triples = extract_triples(nlp, matcher, raw)
                if not triples:
                    st.warning("⚠️ No relationship triples found in the text. The graph may be empty.")
                gen_progress.progress(40)

                gen_status.text("Normalizing entities and writing to Neo4j...")
                graph = NeoGraph(uri=neo_uri, user=neo_user, pwd=neo_pwd)
                inserted = 0
                
                # Store triples for display
                triple_data = []
                
                for i, t in enumerate(triples):
                    try:
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

                        # Store triple info for display
                        triple_data.append({
                            "subject": subj_norm,
                            "subject_type": s_label,
                            "relation": t["relation"],
                            "direction": t["dir"],
                            "object": obj_norm,
                            "object_type": o_label,
                            "confidence": t["confidence"],
                            "sentence": t["sentence"]
                        })

                        graph.upsert_node(s_label, subj_norm, subj_cui)
                        graph.upsert_node(o_label, obj_norm, obj_cui)
                        graph.upsert_relation(s_label, subj_norm, o_label, obj_norm, t["relation"], t["dir"], t["confidence"], t["sentence"])
                        inserted += 1
                        if triples:
                            gen_progress.progress(40 + int(50 * (i + 1) / len(triples)))
                    
                    except Exception as e:
                        st.warning(f"⚠️ Skipped a triple due to error: {str(e)}")
                        continue

                graph.close()
                gen_progress.progress(100)
                gen_status.text(f"Done — inserted {inserted} triples into Neo4j.")
                
                if inserted > 0:
                    st.success("✅ Graph generated and written to Neo4j.")
                    
                    # Display extracted triples in a nice format
                    st.subheader("📊 Extracted Relationships")
                    for t in triple_data:
                        st.markdown(f"""
                        **{t['subject']}** ({t['subject_type']}) 
                        → *{t['relation']}* ({t['direction']}) → 
                        **{t['object']}** ({t['object_type']})
                        
                        Confidence: {t['confidence']:.2f}
                        Context: _{t['sentence']}_
                        ---
                        """)
                else:
                    st.warning("⚠️ No relationships were written to the graph. Check if the text contains relevant patterns.")
                    
            except Exception as e:
                st.error(f"❌ Error during graph generation: {str(e)}")
                gen_status.text("Failed to generate graph.")

# --- MIDDLE PANEL: TEXT PREVIEW ---
with col2:
    st.subheader("Processed Text")
    st.info("Preview will appear here after preprocessing.")
    st.markdown("**Trait:** <span style='color: #4C9AFF'>Blue</span>  |  "
                "**Environment:** <span style='color: #34D399'>Green</span>  |  "
                "**Neural Pattern:** <span style='color: #A78BFA'>Violet</span>", unsafe_allow_html=True)
    
    # Show raw text immediately after upload
    raw_preview = None
    if 'raw_text' in st.session_state and st.session_state.raw_text:
        raw_preview = st.text_area("Extracted Text", value=st.session_state.raw_text, height=200)
    
    # Show preprocessed text when available
    if 'cleaned_text' in st.session_state:
        st.success("✅ Preprocessing Complete!")
        st.text_area("Cleaned and Annotated Text", value=st.session_state.cleaned_text, height=500)
        if 'entities' in st.session_state and st.session_state.entities:
            st.subheader("Detected Entities")
            st.write(st.session_state.entities)
    elif not raw_preview:  # Only show placeholder if no raw text
        st.text_area("Cleaned and Annotated Text", placeholder="Waiting for input...", height=500)


# --- RIGHT PANEL: GRAPH AREA ---
with col3:
    st.subheader("Knowledge Graph View")
    st.info("Interactive graph will appear here.")
    st.empty()  

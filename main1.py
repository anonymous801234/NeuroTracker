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
    
    # Neo4j connection section
    st.markdown("### Neo4j Connection")
    
    storage_type = st.radio(
        "Storage Type",
        ["Local File", "Neo4j Database"],
        help="Choose Local File to save data without requiring a database, or Neo4j Database for graph database storage"
    )
    
    # Initialize variables with defaults
    save_format = "JSON"
    save_path = "./output"
    connection_type = "Local Installation"
    neo_uri = "bolt://localhost:7687"
    neo_user = "neo4j"
    neo_pwd = ""
    
    # Show appropriate options based on storage type
    if storage_type == "Local File":
        st.info("""
        ℹ️ Local File Storage:
        - Saves graph data as JSON files
        - No database required
        - Easy to share and backup
        - Suitable for testing and development
        """)
        save_format = st.selectbox(
            "File Format",
            ["JSON", "CSV"],
            help="Choose the format to save the extracted relationships"
        )
        save_path = st.text_input(
            "Save Directory",
            value="./output",
            help="Directory where files will be saved"
        )
    else:
        connection_type = st.radio(
            "Database Type",
            ["Neo4j AuraDB (Cloud)", "Local Installation"],
            help="Choose AuraDB for cloud-hosted database or Local for Neo4j Desktop installation"
        )
    
    if connection_type == "Neo4j AuraDB (Cloud)":
        st.info("""
        ℹ️ To use Neo4j AuraDB (Free tier):
        1. Go to [Neo4j AuraDB Console](https://console.neo4j.io)
        2. Sign up for free account
        3. Create a new free database
        4. Copy connection details from AuraDB console
        """)
        
        # AuraDB connection inputs
        neo_uri = st.text_input(
            "Database URI", 
            value="neo4j+s://xxxxxxxx.databases.neo4j.io",
            help="Copy the connection URI from AuraDB console"
        )
        neo_user = st.text_input(
            "Database User", 
            value="neo4j",
            help="Username from AuraDB console"
        )
        neo_pwd = st.text_input(
            "Database Password", 
            type="password", 
            value="",
            help="Copy the password from AuraDB console"
        )
    else:
        st.info("""
        ℹ️ For local Neo4j Desktop:
        1. Install Neo4j Desktop
        2. Create and start a database
        3. Use local connection details
        """)
        
        # Local connection inputs
        neo_uri = st.text_input(
            "Database URI", 
            value="bolt://localhost:7687",
            help="Usually 'bolt://localhost:7687' for local installations"
        )
        neo_user = st.text_input(
            "Database User", 
            value="neo4j",
            help="Default is 'neo4j' for new installations"
        )
        neo_pwd = st.text_input(
            "Database Password", 
            type="password", 
            value="",
            help="The password you set during database creation"
        )
    
    # Detailed connection status section
    st.markdown("#### Connection Status")
    if st.button("🔌 Test Neo4j Connection"):
        try:
            with st.spinner("Testing connection..."):
                graph = NeoGraph(uri=neo_uri, user=neo_user, pwd=neo_pwd)
                graph.close()
            st.success("✅ Successfully connected to Neo4j!")
        except Exception as e:
            st.error(f"❌ Failed to connect to Neo4j: {str(e)}")
            
            if "Connection refused" in str(e):
                if storage_type == "Neo4j Database" and connection_type == "Neo4j AuraDB (Cloud)":
                    st.error("Unable to connect to AuraDB.")
                    st.markdown("""
                    **Troubleshooting Steps:**
                    1. Verify the connection URI is correct
                    2. Check if the database is active in AuraDB console
                    3. Make sure you're using the correct URI format (neo4j+s://...)
                    4. Verify your internet connection
                    
                    Need help? Go to the [AuraDB Quick Start Guide](https://neo4j.com/docs/aura/current/getting-started/overview/)
                    """)
                else:
                    st.error("Connection refused. Local Neo4j instance not running.")
                    st.markdown("""
                    **Troubleshooting Steps:**
                    1. Start Neo4j Desktop
                    2. Check if your database is running
                    3. Verify the connection URI (usually bolt://localhost:7687)
                    4. Check if port 7687 is available
                    """)
            elif "unauthorized" in str(e).lower():
                if connection_type == "Neo4j AuraDB (Cloud)":
                    st.error("Authentication failed. Check your AuraDB credentials.")
                    st.markdown("""
                    **Troubleshooting Steps:**
                    1. Go to [AuraDB Console](https://console.neo4j.io)
                    2. Open your database details
                    3. Copy the exact username and password
                    4. Make sure to use neo4j+s:// URI format
                    """)
                else:
                    st.error("Authentication failed. Check your local Neo4j credentials.")
                    st.markdown("""
                    **Troubleshooting Steps:**
                    1. Verify your username (default is 'neo4j')
                    2. Make sure you're using the correct password
                    3. Try resetting the password if needed
                    """)
            else:
                st.error("Unexpected connection error.")
                st.markdown("""
                **General Troubleshooting:**
                1. Verify your connection details
                2. Check database status in console/desktop
                3. Try copying credentials again
                4. Check network connectivity
                """)
    
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

            # Progress UI
            gen_status = st.empty()
            gen_progress = st.progress(0)

            gen_status.text("Loading model and matcher...")
            with st.spinner("Loading NLP model and building matcher..."):
                nlp = load_model()
                matcher = build_matcher(nlp)
            gen_progress.progress(10)
            
            # Verify storage setup
            if storage_type == "Neo4j Database":
                try:
                    with st.spinner("Testing Neo4j connection..."):
                        graph = NeoGraph(uri=neo_uri, user=neo_user, pwd=neo_pwd)
                        graph.close()
                except Exception as e:
                    st.error(f"❌ Cannot connect to Neo4j: {str(e)}")
                    st.warning("Please check your Neo4j connection settings and try again.")
                    st.stop()
            else:
                import os
                import json
                import csv
                from datetime import datetime
                
                # Create output directory if it doesn't exist
                os.makedirs(save_path, exist_ok=True)

            try:
                gen_status.text("Extracting triples...")
                triples = extract_triples(nlp, matcher, raw)
                if not triples:
                    st.warning("⚠️ No relationship triples found in the text. The graph may be empty.")
                gen_progress.progress(40)

                # Store triples for display and saving
                triple_data = []
                inserted = 0
                
                gen_status.text("Normalizing entities and storing relationships...")
                
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

                        # Store triple info
                        triple_info = {
                            "subject": subj_norm,
                            "subject_type": s_label,
                            "relation": t["relation"],
                            "direction": t["dir"],
                            "object": obj_norm,
                            "object_type": o_label,
                            "confidence": t["confidence"],
                            "sentence": t["sentence"]
                        }
                        triple_data.append(triple_info)
                        
                        if storage_type == "Neo4j Database":
                            graph = NeoGraph(uri=neo_uri, user=neo_user, pwd=neo_pwd)
                            graph.upsert_node(s_label, subj_norm, subj_cui)
                            graph.upsert_node(o_label, obj_norm, obj_cui)
                            graph.upsert_relation(s_label, subj_norm, o_label, obj_norm, t["relation"], t["dir"], t["confidence"], t["sentence"])
                            graph.close()
                        
                        inserted += 1
                        if triples:
                            gen_progress.progress(40 + int(50 * (i + 1) / len(triples)))
                    
                    except Exception as e:
                        st.warning(f"⚠️ Skipped a triple due to error: {str(e)}")
                        continue

                # Save to file if using local storage
                if storage_type == "Local File":
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    if save_format == "JSON":
                        output_file = os.path.join(save_path, f"relationships_{timestamp}.json")
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump({
                                "metadata": {
                                    "timestamp": timestamp,
                                    "total_relationships": len(triple_data)
                                },
                                "relationships": triple_data
                            }, f, indent=2, ensure_ascii=False)
                    else:  # CSV
                        output_file = os.path.join(save_path, f"relationships_{timestamp}.csv")
                        with open(output_file, 'w', newline='', encoding='utf-8') as f:
                            writer = csv.DictWriter(f, fieldnames=[
                                "subject", "subject_type", "relation", "direction",
                                "object", "object_type", "confidence", "sentence"
                            ])
                            writer.writeheader()
                            writer.writerows(triple_data)
                    
                    st.success(f"✅ Saved relationships to {output_file}")
                else:
                    gen_status.text(f"Done — inserted {inserted} triples into Neo4j.")
                
                gen_progress.progress(100)
                
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

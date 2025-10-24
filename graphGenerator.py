"""
Streamlit App for NeuroGraph
----------------------------
Takes preprocessed neuroscience text, extracts triples,
builds a Neo4j knowledge graph, and shows real-time progress & results.
"""

import streamlit as st
import pandas as pd
import time
import networkx as nx
import matplotlib.pyplot as plt
from neo4j import GraphDatabase
from neurograph_pipeline import load_scispacy_model, build_matcher, extract_triples, normalize_entity, NeoGraph


# --------------------------------------------------------------------
# Helper: Visualize graph using NetworkX
# --------------------------------------------------------------------
def visualize_graph(uri, user, pwd, limit=30):
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    query = """
    MATCH (a)-[r]->(b)
    RETURN a.name AS source, labels(a)[0] AS source_label,
           type(r) AS relation, r.dir AS dir, r.conf AS conf,
           b.name AS target, labels(b)[0] AS target_label
    LIMIT $limit
    """
    edges = []
    with driver.session() as session:
        for rec in session.run(query, limit=limit):
            edges.append(rec.data())
    driver.close()

    if not edges:
        st.warning("No edges found in Neo4j yet.")
        return

    G = nx.DiGraph()
    for e in edges:
        G.add_edge(f"{e['source']} ({e['source_label']})",
                   f"{e['target']} ({e['target_label']})",
                   relation=e['relation'], weight=e['conf'])

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.6)
    nx.draw(G, pos, with_labels=True, node_size=1800, node_color="skyblue",
            font_size=9, font_weight="bold", arrows=True)
    st.pyplot(plt)


# --------------------------------------------------------------------
# Streamlit layout
# --------------------------------------------------------------------
st.set_page_config(page_title="🧠 NeuroGraph Generator", layout="wide")
st.title("🧠 NeuroGraph: Neural Trait–Pattern Knowledge Graph")
st.caption("Upload or paste neuroscience text to extract patterns, normalize via UMLS, and visualize as a graph.")

uri = st.text_input("Neo4j URI", value="bolt://localhost:7687")
user = st.text_input("Neo4j User", value="neo4j")
pwd = st.text_input("Neo4j Password", type="password")

input_text = st.text_area("Paste your neuroscience text here", height=200,
                          placeholder="e.g., Exposure to novelty increases hippocampal theta oscillations...")

if st.button("🚀 Generate Knowledge Graph"):
    if not input_text.strip():
        st.warning("Please provide some text first.")
        st.stop()

    st.subheader("Progress Monitor")
    progress_text = st.empty()
    progress_bar = st.progress(0)
    steps = [
        ("Loading SciSpacy model (UMLS)...", 20),
        ("Extracting triples...", 40),
        ("Normalizing entities with UMLS...", 60),
        ("Writing data to Neo4j...", 90),
        ("Finalizing visualization...", 100)
    ]

    # Stage 1: Load model
    progress_text.text(steps[0][0])
    nlp = load_scispacy_model()
    matcher = build_matcher(nlp)
    time.sleep(1)
    progress_bar.progress(steps[0][1])

    # Stage 2: Extract triples
    progress_text.text(steps[1][0])
    triples = extract_triples(nlp, matcher, input_text)
    df = pd.DataFrame(triples)
    progress_bar.progress(steps[1][1])
    time.sleep(0.5)

    # Stage 3: Normalize entities
    progress_text.text(steps[2][0])
    for t in triples:
        subj_norm, subj_cui, _ = normalize_entity(nlp, t["subject"])
        obj_norm, obj_cui, _ = normalize_entity(nlp, t["object"])
        t["subject"] = subj_norm
        t["subject_cui"] = subj_cui
        t["object"] = obj_norm
        t["object_cui"] = obj_cui
    progress_bar.progress(steps[2][1])
    time.sleep(0.5)

    # Stage 4: Write to Neo4j
    progress_text.text(steps[3][0])
    graph = NeoGraph(uri, user, pwd)
    for t in triples:
        # Heuristic label assignment
        def label_for(term):
            tl = term.lower()
            if any(x in tl for x in ["hippocamp", "cortex", "amygdala", "pfc"]):
                return "NEURAL_REGION"
            if any(x in tl for x in ["theta", "activation", "spike", "oscillation"]):
                return "NEURAL_PATTERN"
            if any(x in tl for x in ["stress", "novelty", "reward", "environment"]):
                return "ENVIRONMENT"
            return "TRAIT"

        s_label = label_for(t["subject"])
        o_label = label_for(t["object"])
        graph.upsert_node(s_label, t["subject"], t.get("subject_cui"))
        graph.upsert_node(o_label, t["object"], t.get("object_cui"))
        graph.upsert_relation(s_label, t["subject"], o_label, t["object"],
                              t["relation"], t["dir"], t["confidence"], t["sentence"])
    graph.close()
    progress_bar.progress(steps[3][1])

    # Stage 5: Visualization
    progress_text.text(steps[4][0])
    time.sleep(1)
    st.success("✅ Knowledge graph successfully generated!")
    st.balloons()
    progress_bar.progress(steps[4][1])

    # Display triples table
    st.subheader("Extracted Triples")
    st.dataframe(df)

    # Visualize Neo4j subgraph
    st.subheader("Graph Visualization")
    visualize_graph(uri, user, pwd)

"""
Streamlit App for NeuroGraph
----------------------------
Takes preprocessed neuroscience text, extracts triples,
builds a Neo4j knowledge graph, and shows real-time progress & results.
"""

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from neo4j import GraphDatabase


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

    plt.figure(figsize=(11,8), facecolor='#0A0E14')
    pos = nx.spring_layout(G, k=0.75)

    # Fix edge_colors - it was referencing non-existent "color" attribute
    edge_colors = ["#DB510C" for _ in G.edges()]  # Default color for edges
    node_colors = ["#8FCDF4" if "(NEURAL_REGION)" in n else "#706F5C" for n in G.nodes()]

    # Ensure weight is a float to avoid comparison errors
    widths = []
    for u, v in G.edges():
        weight = G[u][v].get("weight", 0.5)
        if isinstance(weight, list):
            weight = float(weight[0]) if weight else 0.5
        else:
            weight = float(weight)
        widths.append(max(0.8, weight * 2))

    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=widths, alpha=0.85)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, edgecolors="#0A0E14", linewidths=1.2, node_size=1500)

    nx.draw_networkx_labels(G, pos, font_color="white", font_size=9, font_family="monospace")
    
    plt.gca().set_facecolor('#0A0E14')

    st.pyplot(plt)

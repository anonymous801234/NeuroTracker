import scispacy
import spacy
from spacy.matcher import DependencyMatcher
from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking import EntityLinker
from neo4j import GraphDatabase
from tqdm import tqdm
import re
import functools
import streamlit as st

@functools.lru_cache(maxsize=1)
def load_model():
    nlp = spacy.load("en_core_sci_sm")
    nlp.add_pipe("abbreviation_detector")
    linker = EntityLinker(resolve_abbreviations=True, name="umls")
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
    return nlp

def normalize_with_scispacy(text):
    nlp = load_model()
    text = " ".join(text.split())  # faster than regex
    doc = nlp(text)

    cleaned_tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
    entities = [(ent.text, ent.label_, [umls_ent for umls_ent in ent._.kb_ents]) for ent in doc.ents]

    return " ".join(cleaned_tokens), entities

@st.cache_data
def preprocess_text(text):
    cleaned_text, entities = normalize_with_scispacy(text)
    return cleaned_text, entities

def build_matcher(nlp):
    matcher = DependencyMatcher(nlp.vocab)

    def pattern(lemma_list):
        return [
            {"RIGHT_ID": "verb",
             "RIGHT_ATTRS": {"LEMMA": {"IN": lemma_list}}},
            {"LEFT_ID": "verb", "REL_OP": ">",
             "RIGHT_ID": "subj",
             "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "nsubjpass"]}}},
            {"LEFT_ID": "verb", "REL_OP": ">",
             "RIGHT_ID": "obj",
             "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "pobj"]}}}
        ]

    matcher.add("MODULATES", [pattern(["modulate", "influence", "affect"])])
    matcher.add("AFFECTS", [pattern(["affect", "alter", "increase", "decrease"])])
    matcher.add("EXHIBITS", [pattern(["exhibit", "show", "display"])])
    return matcher

# ---------------------------------------------------------------------
# 3. Entity normalization via UMLS
# ---------------------------------------------------------------------

def normalize_entity(nlp, text):
    """Return (canonical_name, cui, score)."""
    doc = nlp(text)
    for ent in doc.ents:
        if ent._.kb_ents:
            cui, score = ent._.kb_ents[0]
            canonical = nlp.get_pipe("scispacy_linker").kb.cui_to_entity[cui].canonical_name
            return canonical, cui, score
    return text, None, None

def estimate_trait_intensity(sentence):
    """Assigns an intensity score based on keywords."""
    intensity_keywords = {
        "strongly": 0.9,
        "highly": 0.8,
        "significantly": 0.7,
        "moderately": 0.5,
        "weakly": 0.3,
        "slightly": 0.2
    }
    for word, score in intensity_keywords.items():
        if word in sentence.lower():
            return score
    return 0.5  # default

def extract_triples(nlp, matcher, text):
    doc = nlp(text)
    triples = []

    for match_id, token_ids in matcher(doc):
        verb = doc[token_ids[0]]
        subj = doc[token_ids[1]]
        obj = doc[token_ids[2]]

        relation = doc.vocab.strings[match_id]
        confidence = 0.8  
        intensity = estimate_trait_intensity(text)
        condition = "baseline"  

        triples.append({
            "subject": subj.text,
            "subject_type": "TRAIT",  
            "relation": relation,
            "object": obj.text,
            "object_type": "NEURAL_PATTERN",  
            "confidence": confidence,
            "condition": condition,
            "intensity": intensity,
            "sentence": text,
            "dir": "+"
        })

    return triples

# ---------------------------------------------------------------------
# 5. Neo4j connector and upsert functions
# ---------------------------------------------------------------------

class NeoGraph:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", pwd="password"):
        self.driver = GraphDatabase.driver(uri, auth=(user, pwd))
        self._create_schema()

    def _create_schema(self):
        # Neo4j now requires exactly one statement per run() call. Split
        # the schema DDL into individual statements and run them separately.
        schema = """
        CREATE CONSTRAINT IF NOT EXISTS FOR (t:TRAIT) REQUIRE t.name IS UNIQUE;
        CREATE CONSTRAINT IF NOT EXISTS FOR (e:ENVIRONMENT) REQUIRE e.name IS UNIQUE;
        CREATE CONSTRAINT IF NOT EXISTS FOR (r:NEURAL_REGION) REQUIRE r.name IS UNIQUE;
        CREATE CONSTRAINT IF NOT EXISTS FOR (p:NEURAL_PATTERN) REQUIRE p.name IS UNIQUE;
        """
        statements = [s.strip() for s in schema.split(";")]
        with self.driver.session() as s:
            for stmt in statements:
                if stmt:
                    # run one statement at a time to avoid the "Expected exactly
                    # one statement per query" error from Neo4j
                    s.run(stmt)

    def upsert_node(self, label, name, cui):
        with self.driver.session() as s:
            s.run(f"MERGE (n:{label} {{name:$name}}) SET n.cui=$cui", name=name, cui=cui)

    def upsert_relation(self, start_label, start_name, end_label, end_name,
                        rel_type, dir_sign, conf, sent):
        cypher = f"""
        MATCH (a:{start_label} {{name:$s_name}}),
              (b:{end_label} {{name:$e_name}})
        MERGE (a)-[r:{rel_type}]->(b)
        ON CREATE SET r.dir=$dir, r.conf=$conf, r.example=$sent
        ON MATCH SET r.conf = CASE
            WHEN r.conf IS NULL THEN $conf
            WHEN r.conf < $conf THEN $conf
            ELSE r.conf
        END
        """
        with self.driver.session() as s:
            s.run(cypher, s_name=start_name, e_name=end_name,
                  dir=dir_sign, conf=conf, sent=sent)

    def close(self):
        self.driver.close()

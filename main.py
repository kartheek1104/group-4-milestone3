import streamlit as st
import pandas as pd
import spacy
import networkx as nx
from pyvis.network import Network
from sentence_transformers import SentenceTransformer, util
import streamlit.components.v1 as components

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("bbc-news-data.csv", sep="\t")
    return df

df = load_data()
st.write("### BBC News Dataset Sample", df.head())

# Load SpaCy Model
nlp = spacy.load("en_core_web_sm")

# Triple Extraction Function
def extract_triples(text):
    doc = nlp(text)
    triples = []
    for sent in doc.sents:
        ents = [(ent.text, ent.start, ent.end, ent.label_) for ent in sent.ents 
                if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT"]]
        for i in range(len(ents)-1):
            e1, e2 = ents[i], ents[i+1]
            span = doc[e1[2]:e2[1]]
            relation = [token.lemma_ for token in span if token.pos_ == "VERB"]
            if relation:
                triples.append((e1[0], relation[0], e2[0]))
    return triples

# Extract triples from first 200 articles for speed
all_triples = []
for text in df["content"].dropna().head(200):
    all_triples.extend(extract_triples(text))

all_triples = list(set(all_triples))  # deduplicate

st.write("### Sample Extracted Triples", all_triples[:10])

# Build Knowledge Graph
G = nx.DiGraph()
for head, relation, tail in all_triples:
    G.add_node(head, label=head)
    G.add_node(tail, label=tail)
    G.add_edge(head, tail, label=relation)

nodes = list(G.nodes)

# Semantic Search
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
node_embeddings = model.encode(nodes, convert_to_tensor=True)

st.title("BBC Semantic Knowledge Graph")
query = st.text_input("Enter your search query:", "Apple")

if st.button("Search"):
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(query_embedding, node_embeddings)[0]
    top_k = 5
    results = sorted(zip(nodes, cosine_scores), key=lambda x: x[1], reverse=True)[:top_k]

    st.write("### Top Matches:")
    for node, score in results:
        st.write(f"{node} (Score: {score:.4f})")

    # PyVis Visualization
    net = Network(height="600px", width="100%", directed=True, bgcolor="white", font_color="black")

    for node in nodes:
        if node in [r[0] for r in results]:
            net.add_node(node, color='red', size=30, label=node)
        else:
            net.add_node(node, label=node)

    for edge in G.edges(data=True):
        net.add_edge(edge[0], edge[1], label=edge[2]['label'])

    net.save_graph("semantic_graph.html")
    HtmlFile = open("semantic_graph.html", "r", encoding="utf-8")
    components.html(HtmlFile.read(), height=600)

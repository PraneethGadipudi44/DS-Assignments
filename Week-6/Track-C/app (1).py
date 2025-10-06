import os, time, json, io
import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Week 6: Next-Level RAG", layout="wide")
st.title("Week 6 • Graph-RAG + Multi-Hop Demo")

# -----------------------------
# 0) Load graph artifacts
# -----------------------------
def load_graph():
    """
    Tries to load, in this order:
    - ./graph_week6.graphml
    - ./graph_week6.edgelist (expects 'u v' per line, evidence not required)
    - demo tiny graph if nothing is available
    """
    # Search current dir first, then app dir
    CAND_ROOTS = [".", os.path.dirname(__file__)]
    gml, edgelist = None, None
    for root in CAND_ROOTS:
        p = os.path.join(root, "graph_week6.graphml")
        if os.path.exists(p):
            gml = p; break
    if gml is None:
        for root in CAND_ROOTS:
            p = os.path.join(root, "graph_week6.edgelist")
            if os.path.exists(p):
                edgelist = p; break

    if gml:
        try:
            G = nx.read_graphml(gml)
            src = "graphml"
            return G, src
        except Exception as e:
            st.warning(f"Could not read {gml}: {e}")

    if edgelist:
        try:
            G = nx.read_edgelist(edgelist)
            src = "edgelist"
            return G, src
        except Exception as e:
            st.warning(f"Could not read {edgelist}: {e}")

    # Demo fallback (same spirit as professor’s examples)
    G = nx.Graph()
    nodes = [("Method X","METHOD"),("Author A","AUTHOR"),("Dataset D1","DATASET"),
             ("Paper P3","PAPER"),("Metric F1","METRIC")]
    for n,t in nodes: G.add_node(n, type=t)
    # attach evidence on edges as attributes
    G.add_edge("Method X","Author A",  doc_id="doc1", sentence="Method X was introduced by Author A.")
    G.add_edge("Method X","Dataset D1",doc_id="doc1", sentence="Method X compared on Dataset D1 with F1=0.78.")
    G.add_edge("Method X","Paper P3",  doc_id="doc4", sentence="Paper P3 applies Method X to D2 and reports Accuracy 0.82.")
    G.add_edge("Dataset D1","Metric F1",doc_id="doc1", sentence="F1 reported for D1.")
    return G, "demo"

G, GRAPH_SRC = load_graph()

with st.sidebar:
    st.header("Settings")
    mode = st.selectbox("Mode", ["Graph-RAG","Multi-Hop"], index=1)
    hops = st.slider("Hop limit (Graph-RAG)", 1, 3, 2, 1)
    max_spans = st.slider("Top-k spans (Graph-RAG)", 4, 20, 12, 1)
    show_graph = st.checkbox("Show neighborhood graph", value=True)

st.caption(f"Graph loaded: {len(G.nodes())} nodes, {len(G.edges())} edges (source: {GRAPH_SRC})")

# Optional tables if user placed these near the app
def try_read_csv(name):
    for root in [".", os.path.dirname(__file__)]:
        p = os.path.join(root, name)
        if os.path.exists(p):
            try:
                return pd.read_csv(p)
            except Exception:
                pass
    return None

entities_df  = try_read_csv("entities.csv")   # optional, for quick inspection
relations_df = try_read_csv("relations.csv")  # optional, for quick inspection

# -----------------------------
# 1) Graph-RAG utilities
# -----------------------------
def detect_seed_entities(query: str):
    seeds = []
    qlow = (query or "").lower()
    for n in G.nodes():
        if n.lower().split()[-1] in qlow:
            seeds.append(n)
    # also match type words in query
    for n, data in G.nodes(data=True):
        t = (data.get("type") or "").lower()
        if t and t in qlow:
            seeds.append(n)
    # dedupe, preserve order
    seen = set(); out=[]
    for s in seeds:
        if s not in seen:
            seen.add(s); out.append(s)
    return out

def neighborhood_evidence(seeds, hops=1, max_spans=12):
    spans, seen_edges = [], set()
    for s in seeds:
        if s not in G: 
            continue
        nodes = nx.single_source_shortest_path_length(G, s, cutoff=hops).keys()
        for u in nodes:
            for v in G.neighbors(u):
                e = tuple(sorted([u, v]))
                if e in seen_edges: 
                    continue
                seen_edges.add(e)
                data = G.get_edge_data(u, v) or {}
                spans.append({
                    "u": u, "v": v,
                    "doc_id":   data.get("doc_id", ""),
                    "sentence": data.get("sentence", f"{u} — {v}")
                })
                if len(spans) >= max_spans:
                    return spans
    return spans

def graph_rag(query: str, hops=1, max_spans=12):
    t0 = time.perf_counter()
    seeds = detect_seed_entities(query)
    spans = neighborhood_evidence(seeds, hops=hops, max_spans=max_spans)
    # synthesize a tiny textual answer (LLM could be used here)
    if spans:
        answer = "Based on neighborhood evidence: " + "; ".join(
            f"({s['doc_id']}) {s['sentence']}" if s['doc_id'] else s['sentence'] 
            for s in spans[:2]
        )
    else:
        answer = "No evidence found in graph."
    return {
        "seeds": seeds, "spans": spans, "answer": answer,
        "latency": round(time.perf_counter() - t0, 3)
    }

# -----------------------------
# 2) Multi-hop (self-ask style)
# -----------------------------
def decompose(query: str):
    q = (query or "").lower()
    # tiny hand-made patterns that line up with our demo graph
    if "method x" in q and ("dataset" in q or "f1" in q):
        return ["Which paper or author introduced the method?",
                "Which dataset did that method/paper use for F1 or evaluation?"]
    return [query]

def neighbors_for(node):
    spans=[]
    if node not in G: return spans
    for u, v, data in G.edges(node, data=True):
        spans.append({"doc_id": data.get("doc_id",""), "sentence": data.get("sentence","")})
    return spans

def answer_subq(subq, memory):
    s = (subq or "").lower()
    # Hop 1
    if "introduced the method" in s:
        ev = neighbors_for("Method X")
        ans = ""
        # pick a paper/author if present in evidence
        for e in ev:
            line = e["sentence"].lower()
            if "introduced" in line and "author" in line:
                ans = "Author A"
                break
            if "paper" in line or "p3" in line:
                ans = "Paper P3"
        ans = ans or "Paper P3"  # default if ambiguous
        return {"subq": subq, "answer": ans, "evidence": ev, "memory_update": {"intro_ref": ans}}

    # Hop 2
    if "dataset" in s and ("method" in s or "paper" in s):
        # see previous hop memory
        ref = memory.get("intro_ref","Method X")
        # two quick checks for demo
        if ref in ["Method X","Paper P3","Author A"]:
            ev = neighbors_for("Dataset D1")
            ans = "Dataset D1" if ev else ""
            return {"subq": subq, "answer": ans, "evidence": ev, "memory_update": {}}

    # Fallback
    return {"subq": subq, "answer": "", "evidence": [], "memory_update": {}}

def multi_hop(query: str, hops_limit=2):
    t0 = time.perf_counter()
    subs = decompose(query)[:hops_limit]
    memory = {}
    hops = []
    trace = [("decompose", f"{len(subs)} hops")]
    for s in subs:
        h = answer_subq(s, memory)
        hops.append(h)
        memory.update(h.get("memory_update", {}))
    final = " ; ".join([h["answer"] for h in hops if h["answer"]]) or "No evidence found in graph."
    cites = sorted(set([e["doc_id"] for h in hops for e in h.get("evidence",[]) if e.get("doc_id")]))
    return {
        "final": final, "subqs": subs, "hops": hops,
        "citations": cites, "trace": trace,
        "latency": round(time.perf_counter() - t0, 3)
    }

# -----------------------------
# 3) UI
# -----------------------------
q = st.text_area("Ask a question:", "Which dataset did the paper that introduced Method X use for F1?")
run = st.button("Run")

if run and q.strip():
    if mode == "Graph-RAG":
        out = graph_rag(q, hops=hops, max_spans=max_spans)
        st.subheader("Answer")
        st.write(out["answer"])

        st.markdown("**Evidence (spans)**")
        if out["spans"]:
            for s in out["spans"]:
                st.markdown(f"- ({s['doc_id']}) {s['sentence']}")
        else:
            st.caption("(none)")

        st.caption(f"Latency: {out['latency']}s")

        if show_graph:
            st.markdown("**Neighborhood graph (preview)**")
            pos = nx.spring_layout(G, seed=7)
            type_to_color = {"METHOD":"#6aa84f","AUTHOR":"#3c78d8","DATASET":"#cc0000","PAPER":"#674ea7","METRIC":"#e69138"}
            colors = [type_to_color.get(G.nodes[n].get('type',''), "#999") for n in G.nodes()]
            fig, ax = plt.subplots(figsize=(6,4))
            nx.draw(G, pos, with_labels=True, node_color=colors, node_size=900, font_size=9, edge_color="#bbb", ax=ax)
            st.pyplot(fig)

    else:
        res = multi_hop(q, hops_limit=hops)
        st.subheader("Final")
        st.write(res["final"])

        st.markdown("**Citations:** " + (", ".join(res["citations"]) if res["citations"] else "(none)"))
        st.caption(f"Latency: {res['latency']}s")

        with st.expander("Sub-questions & Evidence", expanded=True):
            for i, h in enumerate(res["hops"], 1):
                st.write(f"**Hop {i}** — {h['subq']} → {h['answer'] or '(no answer)'}")
                if h.get("evidence"):
                    for ev in h["evidence"][:3]:
                        st.markdown(f"- ({ev.get('doc_id','')}) {ev.get('sentence','')}")
                else:
                    st.caption("(no evidence found in graph)")

        with st.expander("Trace", expanded=False):
            for tag, info in res["trace"]:
                st.write(f"- {tag}: {info}")

# Optional quick look at artifacts (if present)
if entities_df is not None or relations_df is not None:
    st.divider()
    st.subheader("Artifacts (optional)")
    tabs = st.tabs(["entities.csv","relations.csv"])
    with tabs[0]:
        if entities_df is not None: st.dataframe(entities_df.head(50))
        else: st.caption("entities.csv not found")
    with tabs[1]:
        if relations_df is not None: st.dataframe(relations_df.head(50))
        else: st.caption("relations.csv not found")

"""
app/streamlit_app.py
======================
Professional Streamlit UI for the Intelligent E-Commerce Recommendation System.
AIE425 — Alamein International University

Page 1 — Recommendations
  · Sidebar: select approach, method, user, constraints
  · Recommendation cards with scores and explanations
  · Side-by-side comparison mode for all approaches

Page 2 — Evaluation Dashboard
  · CF methods RMSE / MAE bar charts (cross-validation)
  · Approach comparison: Precision@K, Recall@K, F1@K, Coverage, Diversity
  · Written analysis report: which method wins and why

Run with:
    cd recommender_system
    streamlit run app/streamlit_app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Intelligent Recommender System | AIE425",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  CUSTOM CSS  (dark professional theme)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Cards */
  .rec-card {
    background: linear-gradient(135deg,#1a1a2e 0%,#25253e 100%);
    border:1px solid #3a3a5e; border-radius:12px;
    padding:16px; margin-bottom:12px;
    box-shadow:0 4px 12px rgba(0,0,0,.35);
  }
  .rec-title  { font-size:1.05rem; font-weight:700; color:#e0e0f0; margin-bottom:4px; }
  .rec-score  { font-size:.85rem;  color:#a0cfff; font-weight:600; }
  .rec-explain{ font-size:.8rem;   color:#9090b0; margin-top:8px; line-height:1.45; }
  .rec-meta   { font-size:.78rem;  color:#6878a0; margin-top:4px; }
  /* Badges */
  .badge { display:inline-block; padding:2px 8px; border-radius:20px;
           font-size:.72rem; font-weight:600; margin-right:4px; }
  .badge-cf      { background:#122a48; color:#7ec8e3; }
  .badge-content { background:#12312a; color:#7edd9e; }
  .badge-kb      { background:#2e1f10; color:#e3b07e; }
  /* Metric cards */
  .metric-card { background:#141428; border-radius:10px; padding:14px;
                 text-align:center; border:1px solid #28285a; }
  .metric-val  { font-size:1.8rem; font-weight:800; color:#7ec8ff; }
  .metric-lbl  { font-size:.8rem;  color:#7080a0; margin-top:2px; }
  /* Sidebar */
  section[data-testid="stSidebar"] { background:#0c0c1e; }
  h1,h2,h3 { color:#c8d8f0 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  CACHED DATA & MODEL LOADING
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    from utils.data_loader import load_dataset
    return load_dataset(source="synthetic")


@st.cache_resource(show_spinner="Training all CF models…")
def train_all_cf(_ratings_df):
    """Train all 6 CF algorithms once; cache the objects."""
    from models.collaborative.user_knn   import UserKNNAlgo
    from models.collaborative.item_knn   import ItemKNNAlgo
    from models.collaborative.svd_model  import SVDAlgo
    from models.collaborative.svdpp_model import SVDppAlgo
    from models.collaborative.knn_means  import KNNMeansAlgo
    from models.collaborative.nmf_model  import NMFAlgo
    return {
        "User-KNN":  UserKNNAlgo().fit(_ratings_df),
        "Item-KNN":  ItemKNNAlgo().fit(_ratings_df),
        "SVD":       SVDAlgo().fit(_ratings_df),
        "SVD++":     SVDppAlgo().fit(_ratings_df),
        "KNN-Means": KNNMeansAlgo().fit(_ratings_df),
        "NMF":       NMFAlgo().fit(_ratings_df),
    }


# ── Load everything ──────────────────────────────────────────
dataset    = load_data()
ratings_df = dataset["ratings"]
items_df   = dataset["items"]

cf_models  = train_all_cf(ratings_df)   # dict of trained algo objects


# ─────────────────────────────────────────────────────────────
#  RECOMMEND HELPERS
# ─────────────────────────────────────────────────────────────

def cf_recommend(method_name: str, user_id: int, n: int):
    """Run inference with a pre-trained CF model."""
    from models.collaborative.cf_base import get_unrated_items
    from utils.result_format import format_results
    algo = cf_models[method_name]
    unrated = get_unrated_items(ratings_df, items_df, user_id)
    scores  = sorted([(iid, algo.predict(user_id, iid)) for iid in unrated],
                     key=lambda x: x[1], reverse=True)[:n]
    def exp(row):
        return (f"Recommended by {method_name} collaborative filtering "
                f"based on community rating patterns.")
    return format_results(items_df, scores, "collaborative",
                          method_name.lower().replace("-","_").replace("+","p"), exp)


def cb_recommend(method_name: str, user_id: int, n: int):
    from models.content import tfidf_genre, tfidf_keywords, tfidf_description
    if   method_name == "TF-IDF Genre":        return tfidf_genre.recommend(ratings_df, items_df, user_id, n)
    elif method_name == "TF-IDF Keywords":     return tfidf_keywords.recommend(ratings_df, items_df, user_id, n)
    else:                                       return tfidf_description.recommend(ratings_df, items_df, user_id, n)


def kb_recommend(method_name: str, user_id: int, n: int,
                 category=None, budget=200.0, brand=None,
                 min_rating=3.0, keywords=""):
    # ===== INPUT HANDLING =====
    # Routes all user-supplied KB constraints to the appropriate backend module.
    from models.knowledge import simple_filter, rule_based, constraint_based
    cat = None if (not category or category == "Any") else category
    brd = None if (not brand    or brand    == "Any") else brand
    if method_name == "Simple Filter":
        return simple_filter.recommend(items_df, ratings_df, user_id,
                                       category=cat, max_price=budget,
                                       brand=brd, min_rating=min_rating, n=n)
    elif method_name == "Rule-Based":
        return rule_based.recommend(items_df, ratings_df, user_id, budget=budget, n=n)
    else:
        return constraint_based.recommend(items_df, ratings_df, user_id,
                                          preferred_category=cat, max_budget=budget,
                                          preferred_brand=brd, min_rating=min_rating, n=n)


# ─────────────────────────────────────────────────────────────
#  DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────

BADGE_CLS   = {"collaborative":"badge-cf","content":"badge-content","knowledge":"badge-kb"}
BADGE_LABEL = {"collaborative":"🤝 CF",  "content":"🔤 Content",   "knowledge":"🧠 Knowledge"}
STARS = ["","⭐","⭐⭐","⭐⭐⭐","⭐⭐⭐⭐","⭐⭐⭐⭐⭐"]


def render_card(r, rank: int):
    stars_n = min(5, max(1, round(r.predicted_score)))
    bcls    = BADGE_CLS.get(r.approach,"badge-cf")
    blbl    = BADGE_LABEL.get(r.approach, r.approach)
    price   = f"${r.price:.2f}" if r.price else ""
    st.markdown(f"""
    <div class="rec-card">
      <div class="rec-title">#{rank} — {r.item_name}</div>
      <div class="rec-score">
        <span class="badge {bcls}">{blbl}</span>
        <span class="badge" style="background:#1c1c2e;color:#aaa;">{r.method}</span>
        &nbsp;{STARS[stars_n]}&nbsp;
        <b style="color:#7ec8ff">{r.predicted_score:.3f}</b>
      </div>
      <div class="rec-meta">🏷️ {r.category} &nbsp;|&nbsp; 🏢 {r.brand} &nbsp;|&nbsp; 💰 {price}</div>
      <div class="rec-explain">💡 {r.explanation}</div>
    </div>""", unsafe_allow_html=True)


def score_chart(recs, color_scale, x_label="Score"):
    scores = [r.predicted_score for r in recs]
    names  = [r.item_name[:22] for r in recs]
    fig = px.bar(x=scores, y=names, orientation="h",
                 color=scores, color_continuous_scale=color_scale,
                 template="plotly_dark")
    fig.update_layout(margin=dict(l=0,r=0,t=20,b=0), height=340,
                      coloraxis_showscale=False,
                      yaxis_title="", xaxis_title=x_label,
                      yaxis=dict(autorange="reversed"))
    return fig


def category_pie(recs):
    cats = pd.Series([r.category for r in recs]).value_counts()
    fig  = px.pie(values=cats.values, names=cats.index,
                  template="plotly_dark", hole=0.4)
    fig.update_layout(margin=dict(l=0,r=0,t=20,b=0), height=240,
                      showlegend=True,
                      legend=dict(font=dict(size=10)))
    return fig


# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛒 Recommender\n`AIE425 · AIU`")
    st.divider()

    page = st.radio("Navigate", ["🎯 Recommendations", "📊 Evaluation Dashboard"],
                    label_visibility="collapsed")
    st.divider()

    # ── Common controls ──────────────────────────────────────
    all_users = sorted(ratings_df["user_id"].unique().tolist())
    sel_user  = st.selectbox("👤 User", all_users,
                             format_func=lambda x: f"User #{x}")
    n_recs = st.slider("🔢 Top-N", 3, 15, 8)
    st.divider()

    if "Recommendations" in page:
        st.markdown("#### Approach")
        approach = st.selectbox("", [
            "All Approaches (Compare)",
            "Collaborative Filtering",
            "Content-Based",
            "Knowledge-Based",
        ], label_visibility="collapsed")

        if approach == "Collaborative Filtering":
            cf_method = st.selectbox("CF Method",
                ["User-KNN","Item-KNN","SVD","SVD++","KNN-Means","NMF"])

        elif approach == "Content-Based":
            cb_method = st.selectbox("Content Method",
                ["TF-IDF Genre","TF-IDF Keywords","TF-IDF Description"])

        elif approach == "Knowledge-Based":
            kb_method = st.selectbox("KB Method",
                ["Simple Filter","Rule-Based","Constraint-Based"])
            st.markdown("#### Constraints")
            cats_list  = ["Any"] + sorted(items_df["category"].unique())
            brands_list= ["Any"] + sorted(items_df["brand"].unique())
            # ===== INPUT HANDLING =====
            kb_cat    = st.selectbox("Category", cats_list,
                help="Leave 'Any' to infer from your rating history")
            kb_budget = st.slider("Max Budget ($)", 10, 500, 200)
            kb_brand  = st.selectbox("Brand", brands_list)
            kb_min_rating = st.slider("Min Community Rating ⭐", 1.0, 5.0, 3.0, 0.5,
                help="Filter out items with a community average below this threshold")
            kb_keywords = st.text_input("Keyword Preference (optional)",
                placeholder="e.g. wireless, ergonomic, organic",
                help="Free-text keywords — biases content scoring toward matching products")


# ─────────────────────────────────────────────────────────────
#  PAGE 1 — RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────

if "Recommendations" in page:
    st.title("🛒 Intelligent E-Commerce Recommender")

    # ── User stats bar ────────────────────────────────────────
    ur = ratings_df[ratings_df["user_id"] == sel_user]
    merged_ur = ur.merge(items_df[["item_id","category"]], on="item_id", how="left")
    top_cat  = merged_ur["category"].value_counts().index[0] if not merged_ur.empty else "N/A"
    avg_r    = round(ur["rating"].mean(), 2) if not ur.empty else 0.0

    c1,c2,c3,c4 = st.columns(4)
    for col, val, lbl in zip(
        [c1,c2,c3,c4],
        [f"#{sel_user}", len(ur), avg_r, top_cat],
        ["User ID","Items Rated","Avg Rating","Top Category"]
    ):
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-val">{val}</div>'
            f'<div class="metric-lbl">{lbl}</div></div>',
            unsafe_allow_html=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════
    #  ALL APPROACHES COMPARE
    # ══════════════════════════════════════════════════════════
    if approach == "All Approaches (Compare)":
        st.subheader("🔀 Side-by-Side: All Three Approaches")
        col_cf, col_cb, col_kb = st.columns(3)

        with col_cf:
            st.markdown("#### 🤝 Collaborative (SVD)")
            with st.spinner("SVD…"):
                cf_recs = cf_recommend("SVD", sel_user, n_recs)
            for i,r in enumerate(cf_recs,1): render_card(r,i)

        with col_cb:
            st.markdown("#### 🔤 Content (TF-IDF Keywords)")
            with st.spinner("TF-IDF…"):
                cb_recs = cb_recommend("TF-IDF Keywords", sel_user, n_recs)
            for i,r in enumerate(cb_recs,1): render_card(r,i)

        with col_kb:
            st.markdown("#### 🧠 Knowledge (Constraint-Based)")
            with st.spinner("Constraints…"):
                kb_recs = kb_recommend("Constraint-Based", sel_user, n_recs)
            for i,r in enumerate(kb_recs,1): render_card(r,i)

        # ── Overlap analysis ─────────────────────────────────
        st.markdown("---")
        st.subheader("📐 Result Overlap Analysis")
        cf_ids = {r.item_id for r in cf_recs}
        cb_ids = {r.item_id for r in cb_recs}
        kb_ids = {r.item_id for r in kb_recs}
        o1,o2,o3,o4 = st.columns(4)
        o1.metric("CF ∩ Content",           len(cf_ids & cb_ids))
        o2.metric("CF ∩ Knowledge",          len(cf_ids & kb_ids))
        o3.metric("Content ∩ Knowledge",     len(cb_ids & kb_ids))
        o4.metric("All Three Agree",         len(cf_ids & cb_ids & kb_ids))

        st.info("""
**Why the overlap is low — and why that's a feature, not a bug:**

• **Collaborative Filtering** (SVD) uses latent patterns from the entire user community — it finds items popular among similar users, even if those items don't match the user's explicit preferences.

• **Content-Based** (TF-IDF Keywords) matches items by their textual feature similarity to things you've already rated — it stays close to known interests but risks a "filter bubble."

• **Knowledge-Based** (Constraint-Based) applies explicit constraints (category, budget, brand) with weighted scoring — it satisfies stated requirements without needing any rating history.

Low overlap means each approach adds unique, complementary value. A **hybrid system** combining all three would produce the best real-world results.
        """)

        # ── Diversity comparison mini-chart ──────────────────
        from evaluation.evaluator import intra_list_diversity
        div_data = {
            "CF (SVD)": intra_list_diversity(cf_recs),
            "Content (KW)": intra_list_diversity(cb_recs),
            "Knowledge": intra_list_diversity(kb_recs),
        }
        fig_div = px.bar(
            x=list(div_data.keys()), y=list(div_data.values()),
            title="Intra-List Diversity (this user)",
            template="plotly_dark",
            color=list(div_data.values()),
            color_continuous_scale="Viridis",
        )
        fig_div.update_layout(height=280, coloraxis_showscale=False,
                              yaxis_title="Diversity (0–1)", xaxis_title="")
        st.plotly_chart(fig_div, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    #  COLLABORATIVE FILTERING
    # ══════════════════════════════════════════════════════════
    elif approach == "Collaborative Filtering":
        st.subheader(f"🤝 Collaborative Filtering — {cf_method}")

        # Method explainer
        explainers = {
            "User-KNN":  "Finds users with similar rating patterns and recommends items they liked.",
            "Item-KNN":  "Finds items similar to what you've rated and recommends those.",
            "SVD":       "Factorises the rating matrix into latent factors via gradient descent.",
            "SVD++":     "SVD extended with implicit feedback (which items you interacted with).",
            "KNN-Means": "KNN with mean-rating bias correction for more accurate similarity.",
            "NMF":       "Non-negative matrix factorisation — interpretable additive latent topics.",
        }
        st.caption(f"ℹ️ {explainers.get(cf_method,'')}")

        with st.spinner(f"Running {cf_method}…"):
            recs = cf_recommend(cf_method, sel_user, n_recs)

        if not recs:
            st.warning("No recommendations generated. Try a different user.")
        else:
            col_main, col_viz = st.columns([3,2])
            with col_main:
                for i,r in enumerate(recs,1): render_card(r,i)
            with col_viz:
                st.markdown("#### 📈 Score Distribution")
                st.plotly_chart(score_chart(recs,"Blues","Predicted Rating"),
                                use_container_width=True)
                st.markdown("#### 🗂️ Category Mix")
                st.plotly_chart(category_pie(recs), use_container_width=True)

        # ── CF method switcher ────────────────────────────────
        st.markdown("---")
        st.subheader("🔄 Compare All CF Methods for This User")
        if st.button("Run All 6 CF Methods", type="secondary"):
            all_cf = {}
            prog = st.progress(0)
            for idx, m in enumerate(["User-KNN","Item-KNN","SVD","SVD++","KNN-Means","NMF"]):
                all_cf[m] = cf_recommend(m, sel_user, 5)
                prog.progress((idx+1)/6)

            # Show top item per method
            rows = []
            for m, rs in all_cf.items():
                if rs:
                    top = rs[0]
                    rows.append({"Method": m, "Top Rec": top.item_name,
                                 "Score": top.predicted_score,
                                 "Category": top.category})
            df_cmp = pd.DataFrame(rows)
            st.dataframe(df_cmp, use_container_width=True, hide_index=True)

            fig_cmp = px.bar(df_cmp, x="Method", y="Score",
                             color="Score", color_continuous_scale="Blues",
                             template="plotly_dark",
                             title="Top Recommendation Score by CF Method")
            fig_cmp.update_layout(height=300, coloraxis_showscale=False)
            st.plotly_chart(fig_cmp, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    #  CONTENT-BASED
    # ══════════════════════════════════════════════════════════
    elif approach == "Content-Based":
        st.subheader(f"🔤 Content-Based Filtering — {cb_method}")
        explainers = {
            "TF-IDF Genre":       "Matches items by category/genre using TF-IDF cosine similarity.",
            "TF-IDF Keywords":    "Matches items by product tags/keywords — captures cross-category similarity.",
            "TF-IDF Description": "Matches items using full descriptions (name + keywords + description text).",
        }
        st.caption(f"ℹ️ {explainers.get(cb_method,'')}")

        with st.spinner(f"Running {cb_method}…"):
            recs = cb_recommend(cb_method, sel_user, n_recs)

        col_main, col_viz = st.columns([3,2])
        with col_main:
            for i,r in enumerate(recs,1): render_card(r,i)
        with col_viz:
            st.markdown("#### 📈 Similarity Scores")
            st.plotly_chart(score_chart(recs,"Greens","Cosine Similarity"),
                            use_container_width=True)
            st.markdown("#### 🗂️ Category Mix")
            st.plotly_chart(category_pie(recs), use_container_width=True)

        # ── All 3 CB methods comparison ───────────────────────
        st.markdown("---")
        st.subheader("🔄 Compare All Content-Based Methods")
        if st.button("Run All 3 CB Methods", type="secondary"):
            rows = []
            for m in ["TF-IDF Genre","TF-IDF Keywords","TF-IDF Description"]:
                rs = cb_recommend(m, sel_user, 5)
                if rs:
                    rows.append({"Method": m, "Top Rec": rs[0].item_name,
                                 "Score": rs[0].predicted_score,
                                 "Category": rs[0].category})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════
    #  KNOWLEDGE-BASED
    # ══════════════════════════════════════════════════════════
    elif approach == "Knowledge-Based":
        st.subheader(f"🧠 Knowledge-Based — {kb_method}")
        explainers = {
            "Simple Filter":    "Hard pass/fail on category, price, brand. Ranked by community avg rating.",
            "Rule-Based":       "8 domain IF-THEN rules (R1–R8). Weighted rule firing score.",
            "Constraint-Based": "Soft weighted constraint satisfaction: category, budget, brand, quality, value.",
        }
        st.caption(f"ℹ️ {explainers.get(kb_method,'')}")
        cat_disp   = kb_cat    if kb_cat   != "Any" else "Inferred from history"
        brand_disp = kb_brand  if kb_brand != "Any" else "Any"
        kw_disp    = kb_keywords.strip() if kb_keywords.strip() else "none"
        st.caption(
            f"Constraints: Category={cat_disp} | Budget≤${kb_budget} | "
            f"Brand={brand_disp} | Min Rating≥{kb_min_rating} | Keywords={kw_disp}"
        )

        with st.spinner("Applying constraints…"):
            # ===== MODEL EXECUTION =====
            recs = kb_recommend(kb_method, sel_user, n_recs,
                                category=kb_cat, budget=kb_budget, brand=kb_brand,
                                min_rating=kb_min_rating, keywords=kb_keywords)

        if not recs:
            st.warning("No items matched your constraints. Try relaxing them (e.g. increase budget or select 'Any' category).")
        else:
            col_main, col_viz = st.columns([3,2])
            with col_main:
                for i,r in enumerate(recs,1): render_card(r,i)
            with col_viz:
                st.markdown("#### 💰 Price Distribution")
                st.plotly_chart(score_chart(recs,"Oranges","Price ($)"), use_container_width=True)

                within_budget = sum(1 for r in recs if r.price <= kb_budget)
                st.markdown(f"**Budget compliance:** {within_budget}/{len(recs)} items within ${kb_budget}")
                st.progress(within_budget / len(recs))

                st.markdown("#### 🗂️ Category Mix")
                st.plotly_chart(category_pie(recs), use_container_width=True)

    # ── Rating history expander ──────────────────────────────
    with st.expander(f"📋 User #{sel_user} — Full Rating History ({len(ur)} items)"):
        hist = ur.merge(items_df[["item_id","name","category","price"]], on="item_id", how="left")
        hist = hist.sort_values("rating", ascending=False)
        st.dataframe(hist[["name","category","price","rating"]].rename(columns={
            "name":"Product","category":"Category","price":"Price","rating":"Rating ⭐"
        }), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────
#  PAGE 2 — EVALUATION DASHBOARD
# ─────────────────────────────────────────────────────────────

else:
    st.title("📊 Evaluation & Comparison Dashboard")
    st.markdown("""
    > Cross-validates all models and compares the three approaches on 7 metrics:
    > RMSE, MAE, Precision@K, Recall@K, F1@K, Coverage, Diversity.
    """)

    col_btn, col_hint = st.columns([2,5])
    with col_btn:
        run_eval = st.button("▶️  Run Full Evaluation", type="primary")
    with col_hint:
        st.caption("⏱️ Estimated time: ~60–90 seconds (3-fold CV on all 6 CF methods + ranking metrics)")

    # ── Quick diversity (always available, instant) ───────────
    st.markdown("---")
    st.subheader("⚡ Quick Diversity & Coverage Preview")
    st.caption("No training needed — computed from recommendations for User #0.")

    from evaluation.evaluator import intra_list_diversity, catalog_coverage
    all_item_ids = items_df["item_id"].tolist()

    quick_data = []
    # Two-arg lambdas: (uid, ratings_df) — matches catalog_coverage API
    preview_configs = [
        ("CF — User-KNN",    lambda uid, df: cf_recommend("User-KNN",  uid, 10)),
        ("CF — Item-KNN",    lambda uid, df: cf_recommend("Item-KNN",  uid, 10)),
        ("CF — SVD",         lambda uid, df: cf_recommend("SVD",       uid, 10)),
        ("CF — SVD++",       lambda uid, df: cf_recommend("SVD++",     uid, 10)),
        ("CF — KNN-Means",   lambda uid, df: cf_recommend("KNN-Means", uid, 10)),
        ("CF — NMF",         lambda uid, df: cf_recommend("NMF",       uid, 10)),
        ("Content — Genre",  lambda uid, df: cb_recommend("TF-IDF Genre",        uid, 10)),
        ("Content — KW",     lambda uid, df: cb_recommend("TF-IDF Keywords",     uid, 10)),
        ("Content — Desc",   lambda uid, df: cb_recommend("TF-IDF Description",  uid, 10)),
        ("Knowledge — Filter",     lambda uid, df: kb_recommend("Simple Filter",    uid, 10)),
        ("Knowledge — Rules",      lambda uid, df: kb_recommend("Rule-Based",       uid, 10)),
        ("Knowledge — Constraint", lambda uid, df: kb_recommend("Constraint-Based", uid, 10)),
    ]

    sample_for_cov = ratings_df["user_id"].unique()[:20].tolist()

    for label, fn in preview_configs:
        try:
            recs = fn(0, ratings_df)   # sample recs for user 0
            div  = intra_list_diversity(recs)
            cov  = catalog_coverage(fn, all_item_ids, sample_for_cov[:10], ratings_df, k=10)
            quick_data.append({"Method": label, "Diversity": div, "Coverage": cov})
        except Exception:
            pass

    if quick_data:
        qdf = pd.DataFrame(quick_data)

        col_d, col_c = st.columns(2)
        with col_d:
            fig_d = px.bar(qdf, x="Method", y="Diversity",
                           color="Diversity", color_continuous_scale="Viridis",
                           template="plotly_dark",
                           title="Intra-List Diversity (higher = more varied list)")
            fig_d.update_layout(height=380, coloraxis_showscale=False,
                                xaxis_tickangle=-45, xaxis_title="")
            st.plotly_chart(fig_d, use_container_width=True)

        with col_c:
            fig_c = px.bar(qdf, x="Method", y="Coverage",
                           color="Coverage", color_continuous_scale="Plasma",
                           template="plotly_dark",
                           title="Catalog Coverage (higher = more of catalog surfaced)")
            fig_c.update_layout(height=380, coloraxis_showscale=False,
                                xaxis_tickangle=-45, xaxis_title="")
            st.plotly_chart(fig_c, use_container_width=True)

        st.dataframe(qdf.style.format({"Diversity":"{:.3f}","Coverage":"{:.3f}"}),
                     use_container_width=True, hide_index=True)

    # ── Full evaluation (triggered by button) ─────────────────
    if run_eval:
        from evaluation.evaluator import run_full_evaluation
        with st.spinner("Running 3-fold cross-validation on all 6 CF methods + ranking metrics…"):
            st.session_state.eval_results = run_full_evaluation(ratings_df, items_df, cv=3)
        st.success("✅ Evaluation complete!")

    if "eval_results" in st.session_state:
        results = st.session_state.eval_results
        cf_df   = results["cf_comparison"]
        app_df  = results["approach_comparison"]

        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(
            ["🤝 CF Method Comparison", "🔀 Approach Comparison", "📝 Analysis Report"]
        )

        # ── TAB 1: CF Comparison ──────────────────────────────
        with tab1:
            st.subheader("Collaborative Filtering — 6 Methods, 3-Fold Cross-Validation")

            col_a, col_b = st.columns(2)
            with col_a:
                fig_rmse = go.Figure()
                fig_rmse.add_trace(go.Bar(
                    x=cf_df["Method"], y=cf_df["RMSE"],
                    error_y=dict(type="data", array=cf_df["RMSE±"].tolist(), visible=True),
                    marker_color="#4a90d9", name="RMSE"
                ))
                fig_rmse.update_layout(
                    title="RMSE by CF Method (lower = better)",
                    template="plotly_dark", height=380,
                    yaxis_title="RMSE", xaxis_title=""
                )
                st.plotly_chart(fig_rmse, use_container_width=True)

            with col_b:
                fig_mae = go.Figure()
                fig_mae.add_trace(go.Bar(
                    x=cf_df["Method"], y=cf_df["MAE"],
                    error_y=dict(type="data", array=cf_df["MAE±"].tolist(), visible=True),
                    marker_color="#50c878", name="MAE"
                ))
                fig_mae.update_layout(
                    title="MAE by CF Method (lower = better)",
                    template="plotly_dark", height=380,
                    yaxis_title="MAE", xaxis_title=""
                )
                st.plotly_chart(fig_mae, use_container_width=True)

            st.markdown("#### Raw Results")
            st.dataframe(cf_df.style.format({
                "RMSE":"{:.4f}","RMSE±":"{:.4f}","MAE":"{:.4f}","MAE±":"{:.4f}"
            }).highlight_min(subset=["RMSE","MAE"], color="#1a3e1a"),
                         use_container_width=True, hide_index=True)

            if not cf_df["RMSE"].isnull().all():
                best = cf_df.loc[cf_df["RMSE"].idxmin()]
                st.success(f"🏆 Best CF Method (RMSE): **{best['Method']}** — "
                           f"RMSE={best['RMSE']:.4f}, MAE={best['MAE']:.4f}")

        # ── TAB 2: Approach Comparison ────────────────────────
        with tab2:
            st.subheader("CF vs Content-Based vs Knowledge-Based")

            pk = [c for c in app_df.columns if "Precision" in c][0]
            rk = [c for c in app_df.columns if "Recall"    in c][0]
            fk = [c for c in app_df.columns if "F1"        in c][0]

            fig_rank = go.Figure()
            x = app_df["Approach"].tolist()
            fig_rank.add_trace(go.Bar(name=pk, x=x, y=app_df[pk], marker_color="#4a90d9"))
            fig_rank.add_trace(go.Bar(name=rk, x=x, y=app_df[rk], marker_color="#50c878"))
            fig_rank.add_trace(go.Bar(name=fk, x=x, y=app_df[fk], marker_color="#e07040"))
            fig_rank.update_layout(
                barmode="group",
                title="Ranking Metrics by Approach",
                template="plotly_dark", height=380,
                yaxis_title="Score (0–1)"
            )
            st.plotly_chart(fig_rank, use_container_width=True)

            col_c, col_d = st.columns(2)
            with col_c:
                fig_cov = px.bar(app_df, x="Approach", y="Coverage",
                                 color="Coverage", color_continuous_scale="Purples",
                                 template="plotly_dark",
                                 title="Catalog Coverage")
                fig_cov.update_layout(height=300, coloraxis_showscale=False)
                st.plotly_chart(fig_cov, use_container_width=True)
            with col_d:
                fig_dv = px.bar(app_df, x="Approach", y="Diversity",
                                color="Diversity", color_continuous_scale="Oranges",
                                template="plotly_dark",
                                title="Intra-List Diversity")
                fig_dv.update_layout(height=300, coloraxis_showscale=False)
                st.plotly_chart(fig_dv, use_container_width=True)

            st.markdown("#### Raw Results")
            st.dataframe(app_df, use_container_width=True, hide_index=True)

            if not app_df[fk].isnull().all():
                best_app = app_df.loc[app_df[fk].idxmax()]
                st.success(f"🏆 Best Approach (F1@K): **{best_app['Approach']}** — "
                           f"F1={best_app[fk]:.4f}")

        # ── TAB 3: Written Analysis ───────────────────────────
        with tab3:
            st.subheader("📝 Analysis Report")

            best_cf_name = (cf_df.loc[cf_df["RMSE"].idxmin(), "Method"]
                            if not cf_df["RMSE"].isnull().all() else "SVD")
            best_app_row = app_df.loc[app_df[fk].idxmax()]

            st.markdown(f"""
### 1. Best Collaborative Filtering Method
**🏆 Winner: {best_cf_name}**

| Method | Key Characteristic | Typical Strength |
|--------|-------------------|-----------------|
| User-KNN | Neighborhood of similar users | Intuitive, transparent |
| Item-KNN | Neighborhood of similar items | Stable, item cold-start tolerant |
| SVD | Latent factor decomposition (SGD) | Accurate, scalable |
| SVD++ | SVD + implicit feedback term | Best accuracy on dense data |
| KNN-Means | KNN with per-user bias correction | Better than plain KNN |
| NMF | Non-negative factorization (sklearn) | Interpretable factors |

**Why SVD/SVD++ typically win on RMSE/MAE:**
Latent factor models learn dense representations that generalise across all users simultaneously, avoiding the sparsity problem that degrades KNN performance. SVD++ additionally benefits from knowing *which* items a user interacted with, even when ratings are missing.

---

### 2. Best Overall Approach
**🏆 Winner: {best_app_row['Approach']}** (highest F1@K = {best_app_row[fk]:.4f})

---

### 3. When to Use Each Approach

| Scenario | Best Choice | Reason |
|----------|-------------|--------|
| User has 20+ ratings | Collaborative Filtering | Enough data for patterns |
| New user (cold start) | Knowledge-Based | No history needed |
| New item (cold item) | Content-Based | Uses only item metadata |
| User states explicit needs | Knowledge-Based | Directly models constraints |
| Maximize catalog coverage | Content-Based | Explores full feature space |
| Maximize accuracy | Collaborative (SVD) | Learns from whole community |
| Need interpretability | Knowledge-Based | Every decision is traceable |

---

### 4. Why Differences Occur

**Collaborative Filtering** learns entirely from rating patterns — it finds what users with similar taste enjoyed, without looking at item content at all. It suffers from the *cold-start problem* (needs data) and *popularity bias* (trends toward popular items).

**Content-Based Filtering** only uses item metadata (category, keywords, description). It works immediately for new users and new items, but can fall into a *filter bubble* — recommending more of the same rather than discovering new categories.

**Knowledge-Based** is driven by explicit domain rules and user-supplied constraints. It is fully transparent, requires no historical data, and never has a cold-start problem. Its weakness is that quality depends heavily on how well the rules and constraints are defined.

---

### 5. Evaluation Metric Interpretation

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| **RMSE** | √(Σ(pred−actual)²/n) | Rating prediction accuracy (lower = better) |
| **MAE** | Σ|pred−actual|/n | Rating prediction accuracy (lower = better) |
| **Precision@K** | hits / K | Quality: are top-K results relevant? |
| **Recall@K** | hits / total_relevant | Completeness: do we find all relevant items? |
| **F1@K** | 2·P·R / (P+R) | Balance between precision and recall |
| **Coverage** | unique_recs / catalog | Catalog exploration breadth |
| **Diversity** | avg pairwise category dissimilarity | Variety within a recommendation list |

---

### 6. Recommended Deployment Strategy (Hybrid)

```
New user signup   →  Knowledge-Based (use stated preferences)
After 5 ratings   →  50% Content-Based + 50% Knowledge-Based
After 15 ratings  →  70% CF (SVD) + 20% Content + 10% Knowledge
After 30 ratings  →  90% CF (SVD) + 10% Content diversity injection
```

This graduated approach solves the cold-start problem while maximising accuracy for engaged users.
""")

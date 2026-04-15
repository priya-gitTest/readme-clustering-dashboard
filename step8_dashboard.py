#!/usr/bin/env python3
"""
Research Software Metadata Analyser
Interactive visualization and exploration of research software metadata
"""

import json
import sys
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).parent.parent))

from contextlib import contextmanager

from database import get_session
from database.models import (
    Cluster,
    ExcludedRepository,
    HeaderClusterAssignment,
    HeaderEmbedding,
    ReadmeHeader,
    Repository,
    SomefResult,
    UnsupportedRepository,
)

# ClusterKSearchResult was added in migration 007 — import gracefully so the
# dashboard keeps working on deployments where the migration hasn't run yet.
try:
    from database.models import ClusterKSearchResult

    _k_search_available = True
except ImportError:
    ClusterKSearchResult = None  # type: ignore
    _k_search_available = False
import os

from sqlalchemy import func, text


@contextmanager
def get_session_context():
    """Fresh session per call — auto-rollback on error, always closed.

    Defined locally so the dashboard works with the repo's minimal
    database module (which only provides get_session()).
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# Get database URL from Streamlit secrets or environment
if "DATABASE_URL" in st.secrets:
    os.environ["DATABASE_URL"] = st.secrets["DATABASE_URL"]

# Page config
st.set_page_config(
    page_title="Research Software Metadata Analyser",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .cluster-card {
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #f8f9fa;
    }
    .cluster-name {
        font-weight: bold;
        color: #1f77b4;
        font-size: 1.2rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=300)
def load_overview_stats():
    """Load overview statistics"""
    with get_session_context() as session:
        stats = {
            "total_repos": session.query(Repository).count(),
            "total_headers": session.query(ReadmeHeader).count(),
            "total_embeddings": session.query(HeaderEmbedding).count(),
            "total_clusters": session.query(Cluster).count(),
            "total_assignments": session.query(HeaderClusterAssignment).count(),
        }

        latest_run = session.query(Cluster.run_id).order_by(Cluster.created_at.desc()).first()
        if latest_run:
            stats["run_id"] = latest_run[0]

    return stats


@st.cache_data(ttl=300)
def load_cluster_data(run_id=None):
    """Load cluster information"""
    with get_session_context() as session:
        query = session.query(Cluster).order_by(Cluster.cluster_size.desc())
        if run_id:
            query = query.filter_by(run_id=run_id)
        clusters = query.all()
        cluster_data = []
        for c in clusters:
            rep_headers = json.loads(c.representative_headers)
            cluster_data.append(
                {
                    "id": c.id,
                    "cluster_id": c.cluster_id,
                    "name": c.cluster_name,
                    "size": c.cluster_size,
                    "representative_headers": rep_headers,
                    "sample": rep_headers[0] if rep_headers else "N/A",
                }
            )
    return pd.DataFrame(cluster_data)


import re as _re

# Applied at display time — independent of what is stored in normalized_text.
_DISPLAY_SECTION_RE = _re.compile(r'^\d+(\.\d+)*\.?\s+')
_DISPLAY_EMOJI_RE = _re.compile(
    u'[\U0001F000-\U0001FFFF'   # broad emoji block
    u'\U00002600-\U000027BF'    # misc symbols + dingbats
    u'\u2300-\u23FF'            # misc technical
    u'\uFE00-\uFE0F'            # variation selectors
    u'\u200D'                   # zero-width joiner
    u']+',
    _re.UNICODE,
)


def _clean_header_display(text: str) -> str:
    """Normalise a header text purely for display grouping.

    Strips: leading section numbers, leading/trailing emojis,
    trailing punctuation (:, ?, .), and surrounding whitespace.
    """
    if not text:
        return ''
    t = _DISPLAY_SECTION_RE.sub('', text)
    t = _DISPLAY_EMOJI_RE.sub('', t)
    t = t.strip().rstrip(':?. ')
    return t.strip()


@st.cache_data(ttl=300)
def load_cluster_members(cluster_db_id: int, limit: int = 20):
    """Load deduplicated member headers for a cluster, ordered by distance to centroid.

    SQL groups by raw normalized_text (fast, accurate counts). Python then applies
    comprehensive display normalisation (section numbers, emojis, trailing punctuation)
    and re-merges groups, so counts stay accurate and any stored-data quirks are handled.
    Returns (display_text, min_distance, count) tuples.
    Refreshed automatically on each workflow run (ttl=300s cache)."""
    with get_session_context() as session:
        from sqlalchemy import func as sa_func

        # Fetch all unique normalized_text values for this cluster (no LIMIT —
        # there are typically only a few hundred unique texts per cluster).
        rows = (
            session.query(
                ReadmeHeader.normalized_text,
                sa_func.min(HeaderClusterAssignment.distance).label("min_dist"),
                sa_func.count(ReadmeHeader.id).label("cnt"),
            )
            .join(HeaderClusterAssignment, HeaderClusterAssignment.header_id == ReadmeHeader.id)
            .filter(HeaderClusterAssignment.cluster_id == cluster_db_id)
            .group_by(ReadmeHeader.normalized_text)
            .all()
        )

    # Re-group in Python with full normalisation.
    merged: dict[str, list] = {}  # key → [min_dist, total_count]
    for text, min_dist, cnt in rows:
        key = _clean_header_display(text or '')
        if not key:
            continue
        if key not in merged:
            merged[key] = [min_dist, cnt]
        else:
            merged[key][0] = min(merged[key][0], min_dist)
            merged[key][1] += cnt

    result = sorted(merged.items(), key=lambda x: x[1][0])
    return [(text, data[0], data[1]) for text, data in result[:limit]]


@st.cache_data(ttl=300)
def load_k_search_results(run_id=None):
    """Load k-selection sweep results for the given run_id (or the latest run).
    Returns an empty DataFrame if the table doesn't exist yet (pre-migration 007)."""
    if not _k_search_available:
        return pd.DataFrame()
    try:
        with get_session_context() as session:
            query = session.query(ClusterKSearchResult)
            if run_id:
                query = query.filter_by(run_id=run_id)
            else:
                # Fall back to the most recent run that has k-search data
                latest = (
                    session.query(ClusterKSearchResult.run_id)
                    .order_by(ClusterKSearchResult.created_at.desc())
                    .first()
                )
                if latest:
                    query = query.filter_by(run_id=latest[0])
            rows = query.order_by(ClusterKSearchResult.k.asc()).all()
            return pd.DataFrame(
                [
                    {
                        "k": r.k,
                        "inertia": r.inertia,
                        "silhouette": r.silhouette_score,
                        "is_best": r.is_best,
                    }
                    for r in rows
                ]
            )
    except Exception:
        # Table doesn't exist yet (migration 007 pending) — show info in UI instead
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_license_data():
    """Load per-repo license information from the repositories table."""
    with get_session_context() as session:
        repos = session.query(Repository).all()
        rows = []
        for repo in repos:
            lic = getattr(repo, "license_from_api", None)
            rows.append(
                {
                    "name": repo.name,
                    "url": repo.url,
                    "platform": repo.platform or "",
                    "source": repo.source or "",
                    "license": lic if lic and lic != "NOASSERTION" else "No License",
                }
            )
    return pd.DataFrame(rows)


def categorize_license(license_name):
    """Map a raw SPDX id to a broad category string."""
    if not license_name or license_name == "No License":
        return "No License"
    permissive = [
        "MIT",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "ISC",
        "Artistic",
        "Zlib",
        "Python-2.0",
        "PSF",
        "WTFPL",
        "0BSD",
        "CC0",
        "Unlicense",
        "BSL-1.0",
        "CC-BY-4.0",
        "CC-BY-3.0",
    ]
    weak_copyleft = ["LGPL", "MPL-2.0", "CDDL", "EPL", "EUPL", "CC-BY-SA"]
    strong_copyleft = ["GPL-2.0", "GPL-3.0", "AGPL", "CPAL", "OSL"]
    u = license_name.upper()
    for p in strong_copyleft:
        if p.upper() in u:
            return "Strong Copyleft"
    for p in weak_copyleft:
        if p.upper() in u:
            return "Weak Copyleft"
    for p in permissive:
        if p.upper() in u:
            return "Permissive"
    return "Other"


@st.cache_data(ttl=300)
def load_repository_details():
    """Load full repository rows for the browser page."""
    with get_session_context() as session:
        repos = session.query(Repository).all()
        rows = []
        for repo in repos:
            lic = getattr(repo, "license_from_api", None)
            desc = getattr(repo, "description", "") or ""
            rows.append(
                {
                    "name": repo.name,
                    "url": repo.url,
                    "license": lic if lic and lic != "NOASSERTION" else "No License",
                    "stars": getattr(repo, "stars", 0) or 0,
                    "language": getattr(repo, "language", "Unknown") or "Unknown",
                    "description": desc[:100] + "..." if len(desc) > 100 else desc,
                    "source": getattr(repo, "source", None) or "unknown",
                    "platform": getattr(repo, "platform", None) or "unknown",
                }
            )
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def load_excluded_repos():
    """Load excluded repositories for the browser page."""
    with get_session_context() as session:
        rows_q = (
            session.query(
                ExcludedRepository.url,
                ExcludedRepository.exclusion_reason,
                ExcludedRepository.exclusion_stage,
                ExcludedRepository.source,
                ExcludedRepository.is_retryable,
                ExcludedRepository.excluded_at,
            )
            .order_by(ExcludedRepository.excluded_at.desc())
            .all()
        )
        rows = [
            {
                "url": r.url,
                "exclusion_reason": r.exclusion_reason,
                "exclusion_stage": r.exclusion_stage,
                "source": r.source or "unknown",
                "retryable": r.is_retryable,
                "excluded_at": r.excluded_at,
            }
            for r in rows_q
        ]
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def load_unsupported_repos():
    """Load unsupported-platform repositories for the browser page."""
    try:
        with get_session_context() as session:
            rows_q = (
                session.query(
                    UnsupportedRepository.url,
                    UnsupportedRepository.source,
                    UnsupportedRepository.host,
                    UnsupportedRepository.platform,
                    UnsupportedRepository.occurrence_count,
                    UnsupportedRepository.first_seen_at,
                    UnsupportedRepository.last_seen_at,
                )
                .order_by(
                    UnsupportedRepository.occurrence_count.desc(),
                    UnsupportedRepository.host,
                )
                .all()
            )
            rows = [
                {
                    "url": r.url,
                    "source": r.source,
                    "host": r.host,
                    "platform": r.platform,
                    "seen": r.occurrence_count,
                    "first_seen": r.first_seen_at,
                    "last_seen": r.last_seen_at,
                }
                for r in rows_q
            ]
    except Exception:
        # Table may not exist yet on first run before migration.
        # get_session_context() auto-rolls back on exception so no shared
        # session is left in a failed state.
        return pd.DataFrame()
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def load_somef_stats():
    """Load SOMEF validation results for the SOMEF Validation page."""
    CATEGORIES = [
        "description",
        "installation",
        "invocation",
        "citation",
        "requirement",
        "documentation",
        "contributor",
        "license",
        "usage",
        "acknowledgement",
        "run",
        "support",
    ]
    with get_session_context() as session:
        total = session.query(SomefResult).count()
        if total == 0:
            return {"total": 0, "categories": CATEGORIES}

        errors = session.query(SomefResult).filter(SomefResult.error.isnot(None)).count()

        cat_counts = {}
        for cat in CATEGORIES:
            col = getattr(SomefResult, cat)
            cat_counts[cat] = session.query(SomefResult).filter(col.isnot(None)).count()

        rows_q = (
            session.query(
                Repository.name,
                Repository.url,
                SomefResult.categories_found,
                SomefResult.processing_time_s,
                SomefResult.somef_version,
                SomefResult.error,
                SomefResult.run_date,
            )
            .join(SomefResult, SomefResult.repository_id == Repository.id)
            .all()
        )

        rows = []
        for name, url, cats_json, proc_time, version, error, run_date in rows_q:
            try:
                cats = json.loads(cats_json) if cats_json else []
            except (ValueError, TypeError):
                cats = []
            rows.append(
                {
                    "name": name,
                    "url": url,
                    "categories_found": len(cats),
                    "categories": ", ".join(cats) if cats else "",
                    "processing_s": round(proc_time, 1) if proc_time else None,
                    "somef_version": version,
                    "error": error,
                    "run_date": run_date,
                }
            )

    df = pd.DataFrame(rows)
    avg_cats = df["categories_found"].mean() if not df.empty else 0
    avg_time = df["processing_s"].mean() if not df.empty else 0

    return {
        "total": total,
        "errors": errors,
        "avg_cats": avg_cats,
        "avg_time": avg_time,
        "cat_counts": cat_counts,
        "categories": CATEGORIES,
        "df": df,
    }


@st.cache_data(ttl=300)
def load_embeddings_for_viz(max_samples=5000):
    """Load embeddings for visualization (sample if too large)"""
    with get_session_context() as session:
        total = session.query(HeaderEmbedding).count()

        if total > max_samples:
            query = (
                session.query(
                    HeaderEmbedding.header_id,
                    HeaderEmbedding.embedding_vector,
                    HeaderEmbedding.embedding_dim,
                )
                .order_by(func.random())
                .limit(max_samples)
            )
        else:
            query = session.query(
                HeaderEmbedding.header_id,
                HeaderEmbedding.embedding_vector,
                HeaderEmbedding.embedding_dim,
            )

        data = query.all()

        header_ids = []
        embeddings = []
        for header_id, emb_bytes, _dim in data:
            header_ids.append(header_id)
            emb_array = np.frombuffer(emb_bytes, dtype=np.float32)
            embeddings.append(emb_array)

        embeddings = np.array(embeddings)

        rows = (
            session.query(HeaderClusterAssignment.header_id, Cluster.cluster_name)
            .join(Cluster, HeaderClusterAssignment.cluster_id == Cluster.id)
            .filter(HeaderClusterAssignment.header_id.in_(header_ids))
            .all()
        )
        assignments = {r.header_id: r.cluster_name for r in rows}

    return header_ids, embeddings, assignments


@st.cache_data(ttl=600)
def load_cluster_similarity_matrix(run_id: str | None = None):
    """Compute pairwise cosine similarity between cluster centroids.

    For each cluster in the given run, averages the embeddings of all assigned
    headers to get a centroid vector, then computes the full NxN cosine
    similarity matrix.  Returns (cluster_names, cluster_sizes, sim_matrix).
    """
    with get_session_context() as session:
        # Get clusters for this run
        if run_id:
            clusters = (
                session.query(Cluster)
                .filter(Cluster.run_id == run_id)
                .order_by(Cluster.cluster_size.desc())
                .all()
            )
        else:
            clusters = (
                session.query(Cluster)
                .order_by(Cluster.cluster_size.desc())
                .all()
            )
        if not clusters:
            return [], [], np.array([])

        cluster_names = []
        cluster_sizes = []
        centroids = []

        for c in clusters:
            # Get all embeddings for headers in this cluster
            rows = (
                session.query(HeaderEmbedding.embedding_vector)
                .join(HeaderClusterAssignment, HeaderClusterAssignment.header_id == HeaderEmbedding.header_id)
                .filter(HeaderClusterAssignment.cluster_id == c.id)
                .all()
            )
            if not rows:
                continue

            vecs = np.array([np.frombuffer(r[0], dtype=np.float32) for r in rows])
            centroid = vecs.mean(axis=0)
            # L2 normalise for cosine similarity
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm

            display_name = _clean_header_display(c.cluster_name) if c.cluster_name else f"Cluster {c.cluster_id}"
            cluster_names.append(display_name)
            cluster_sizes.append(c.cluster_size or 0)
            centroids.append(centroid)

        if len(centroids) < 2:
            return cluster_names, cluster_sizes, np.array([])

        centroid_matrix = np.array(centroids)
        # Cosine similarity = dot product of L2-normalised vectors
        sim_matrix = centroid_matrix @ centroid_matrix.T
        return cluster_names, cluster_sizes, sim_matrix


@st.cache_data(ttl=300)
def load_gap_analysis_data():
    """Load cluster_codemeta_mappings and unmapped_clusters for the Gap Analysis page."""
    with get_session_context() as session:
        # ── Mapped clusters ────────────────────────────────────────────────
        mapped_rows = session.execute(
            text("""
            SELECT c.cluster_name, c.cluster_size,
                   m.codemeta_property, m.confidence, m.mapping_method
              FROM cluster_codemeta_mappings m
              JOIN clusters c ON c.id = m.cluster_id
             ORDER BY c.cluster_size DESC
        """)
        ).fetchall()

        # ── Unmapped clusters ──────────────────────────────────────────────
        unmapped_rows = session.execute(
            text("""
            SELECT c.cluster_name, c.cluster_size,
                   u.proposed_property_name, u.priority, u.status, u.justification
              FROM unmapped_clusters u
              JOIN clusters c ON c.id = u.cluster_id
             ORDER BY
                 CASE u.priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                 c.cluster_size DESC
        """)
        ).fetchall()

    mapped_df = (
        pd.DataFrame(
            mapped_rows,
            columns=[
                "cluster_name",
                "cluster_size",
                "codemeta_property",
                "confidence",
                "mapping_method",
            ],
        )
        if mapped_rows
        else pd.DataFrame()
    )

    unmapped_df = (
        pd.DataFrame(
            unmapped_rows,
            columns=[
                "cluster_name",
                "cluster_size",
                "proposed_property_name",
                "priority",
                "status",
                "justification",
            ],
        )
        if unmapped_rows
        else pd.DataFrame()
    )

    return mapped_df, unmapped_df


@st.cache_data
def compute_umap(embeddings, n_components=2):
    """Compute UMAP projection"""
    from umap import UMAP

    reducer = UMAP(
        n_components=n_components, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42
    )

    return reducer.fit_transform(embeddings)


def main():
    """Main dashboard"""

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        [
            "Overview",
            "Repository Browser",
            "License Analysis",
            "Data Quality",
            "Header Cluster Lookup",
            "Cluster Explorer",
            "Experiment History",
            # "SOMEF Validation",  # disabled — SOMEF pipeline paused
            "Gap Analysis",
            "Cluster Similarity Graph",
            "Visualization",
            "Export",
            "Architecture",
        ],
    )

    # Page title — dynamic per page
    PAGE_TITLES = {
        "Overview": "📊 Research Software Metadata Analyser",
        "Cluster Explorer": "🔍 Cluster Explorer",
        "Experiment History": "🧪 Experiment History",
        "License Analysis": "📄 License Analysis",
        "Repository Browser": "📚 Repository Browser",
        "Data Quality": "📋 Data Quality Report",
        "Cluster Similarity Graph": "🕸️ Cluster Similarity Graph",
        "Visualization": "📊 Embedding Visualization",
        "Header Cluster Lookup": "🔎 Header Cluster Lookup",
        "SOMEF Validation": "🔬 SOMEF Metadata Validation",
        "Gap Analysis": "🗺️ CodeMeta Gap Analysis",
        "Export": "📥 Export Results",
        "Architecture": "🏗️ Architecture",
    }
    st.markdown(
        f'<div class="main-header">{PAGE_TITLES.get(page, "Research Software Metadata Analyser")}</div>',
        unsafe_allow_html=True,
    )

    # Load data — gracefully degrade if DB is unreachable so static pages still work
    _db_error = None
    try:
        stats = load_overview_stats()
    except Exception as _e:
        _db_error = _e
        stats = {"total_repos": 0, "total_headers": 0, "total_embeddings": 0,
                 "total_clusters": 0, "total_assignments": 0}

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("### Dataset Statistics")
    if _db_error:
        st.sidebar.warning("⚠️ DB unavailable")
    else:
        st.sidebar.metric("Repositories", f"{stats['total_repos']:,}")
        st.sidebar.metric("Headers", f"{stats['total_headers']:,}")
        st.sidebar.metric("Clusters", stats["total_clusters"])

    if "run_id" in stats:
        st.sidebar.markdown(f"**Run ID:** `{stats['run_id']}`")

    # License info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📄 License & Attribution")
    st.sidebar.markdown("""
    **Research Project**
    AI-Assisted Code Metadata Pipeline

    **License:** MIT License
    **Data Source:** Research software repositories

    **Citation:**
    If you use this work, please cite:
    ```
    [Priyanka Ojha] [ORCID](https://orcid.org/0000-0002-6844-6493) (2026)
    Research Software Metadata Analyser
    AI-Assisted Code Metadata Pipeline
    ```

    **Code:** [GitHub Repository](https://github.com/priya-gitTest/readme-clustering-dashboard)
    """)

    # Page routing
    # Architecture page never needs DB — always render it directly
    if page == "Architecture":
        show_architecture()
        # Skip footer duplication — fall through to footer below
    elif _db_error and page != "Architecture":
        st.error(f"⚠️ Database unavailable: {_db_error}\n\nStatic pages (Architecture) still work — use the sidebar to navigate there.")
    elif page == "Overview":
        show_overview(stats)
    elif page == "Cluster Explorer":
        show_cluster_explorer(stats.get("run_id"))
    elif page == "Experiment History":
        show_experiment_history()
    elif page == "License Analysis":
        show_license_analysis()
    elif page == "Repository Browser":
        show_repository_browser()
    elif page == "SOMEF Validation":
        show_somef_validation()
    elif page == "Gap Analysis":
        show_gap_analysis()
    elif page == "Data Quality":
        show_data_quality()
    elif page == "Cluster Similarity Graph":
        show_cluster_network(stats.get("run_id"))
    elif page == "Visualization":
        show_visualization()
    elif page == "Header Cluster Lookup":
        show_search()
    elif page == "Export":
        show_export(stats.get("run_id"))

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><strong>Research Software Metadata Analyser</strong></p>
        <p>Analyzing documentation structure patterns in research software</p>
        <p>© 2026 | MIT License | <a href='https://github.com/priya-gitTest/readme-clustering-dashboard'>View on GitHub</a></p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def show_overview(stats):
    """Overview page"""

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Repositories Analyzed", value=f"{stats['total_repos']:,}", delta="100% coverage"
        )

    with col2:
        st.metric(
            label="Headers Extracted",
            value=f"{stats['total_headers']:,}",
            delta=f"{stats['total_headers'] / stats['total_repos']:.1f} per repo"
            if stats["total_repos"] > 0
            else "0.0 per repo",
        )

    with col3:
        st.metric(label="Clusters Discovered", value=stats["total_clusters"], delta="K-Means")

    with col4:
        coverage = (
            (stats["total_assignments"] / stats["total_headers"] * 100)
            if stats["total_headers"] > 0
            else 0
        )
        st.metric(
            label="Clustering Coverage",
            value=f"{coverage:.1f}%",
            delta=f"{stats['total_assignments']:,} assigned",
        )

    # Flashing banner for Cluster Similarity Graph
    st.markdown("""
    <style>
    @keyframes pulse-glow {
        0%   { opacity: 1; box-shadow: 0 0 5px #4CAF50; }
        50%  { opacity: 0.85; box-shadow: 0 0 20px #4CAF50, 0 0 40px #4CAF5066; }
        100% { opacity: 1; box-shadow: 0 0 5px #4CAF50; }
    }
    .network-banner {
        animation: pulse-glow 2s ease-in-out infinite;
        background: linear-gradient(135deg, #1a472a, #2d5a3f);
        border: 1px solid #4CAF50;
        border-radius: 10px;
        padding: 12px 20px;
        margin: 10px 0;
        text-align: center;
        color: #81C784;
        font-size: 1.05em;
    }
    </style>
    <div class="network-banner">
        🕸️ <b>Explore the Cluster Similarity Graph</b>
        <span style="color:#C8E6C9;">— interactive graph showing how documentation topic clusters relate to each other.
        Use the sidebar to navigate there.</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Key findings
    st.subheader("🔍 Key Findings")

    cluster_df = load_cluster_data(stats.get("run_id"))

    if cluster_df.empty:
        st.info(
            "No clustering data yet — the pipeline hasn't completed a full run. Check back after the next scheduled scrape."
        )
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Top 10 Largest Clusters")
        top_10 = cluster_df.head(10)

        fig = px.bar(
            top_10,
            x="size",
            y="name",
            orientation="h",
            title="Cluster Sizes",
            labels={"size": "Number of Headers", "name": "Cluster"},
        )
        fig.update_layout(height=400, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Cluster Size Distribution")

        fig = px.histogram(
            cluster_df,
            x="size",
            nbins=20,
            title="Distribution of Cluster Sizes",
            labels={"size": "Cluster Size", "count": "Frequency"},
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    st.markdown("---")
    st.subheader("📊 Cluster Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Mean Cluster Size", f"{cluster_df['size'].mean():.0f}")
    with col2:
        st.metric("Median Cluster Size", f"{cluster_df['size'].median():.0f}")
    with col3:
        st.metric("Smallest Cluster", f"{cluster_df['size'].min()}")
    with col4:
        st.metric("Largest Cluster", f"{cluster_df['size'].max()}")


@st.cache_data(ttl=300)
def load_cluster_names_for_runs(run_ids: tuple) -> dict:
    """Return {run_id: DataFrame(name, size)} for each run_id.

    Used by the Compare Cluster Contents section in Experiment History.
    run_ids is a tuple (not list) so st.cache_data can hash it.
    """
    result = {}
    try:
        with get_session_context() as session:
            for rid in run_ids:
                rows = (
                    session.query(Cluster.cluster_name, Cluster.cluster_size)
                    .filter(Cluster.run_id == rid)
                    .order_by(Cluster.cluster_size.desc())
                    .all()
                )
                result[rid] = pd.DataFrame(rows, columns=["name", "size"])
    except Exception:
        pass
    return result


@st.cache_data(ttl=300)
def load_experiment_history() -> "pd.DataFrame":
    """Load all AnalysisRun rows with a populated config_snapshot, newest first.

    Returns a flat DataFrame with columns:
        run_id, run_date, min_level, method, merge_threshold,
        best_k, best_silhouette, n_clusters_after_merge, n_merges
    Returns an empty DataFrame gracefully if the table has no records yet.
    """
    try:
        from database.models import AnalysisRun

        with get_session_context() as session:
            rows = (
                session.query(AnalysisRun)
                .filter(AnalysisRun.config_snapshot.isnot(None))
                .order_by(AnalysisRun.run_date.desc())
                .all()
            )
            records = []
            for r in rows:
                try:
                    snap = json.loads(r.config_snapshot)
                except (ValueError, TypeError):
                    continue
                inp = snap.get("input", {})
                out = snap.get("outcome", {})
                records.append(
                    {
                        "run_id": r.run_id,
                        "run_date": r.run_date,
                        "min_level": inp.get("min_level"),
                        "method": inp.get("method"),
                        "merge_threshold": inp.get("merge_threshold", 0.0),
                        "best_k": out.get("best_k"),
                        "best_silhouette": out.get("best_silhouette"),
                        "n_clusters_before_merge": out.get("n_clusters_before_merge"),
                        "n_clusters_after_merge": out.get("n_clusters_after_merge"),
                        "n_merges": out.get("n_merges", 0),
                        "total_headers": out.get("total_headers_clustered"),
                        "mean_cluster_size": out.get("mean_cluster_size"),
                        "_config_snapshot": r.config_snapshot,  # kept for drill-down
                    }
                )
        return pd.DataFrame(records)
    except Exception:
        return pd.DataFrame()


def show_experiment_history():
    """Experiment History page — tracks clustering runs for comparison and reporting."""
    df = load_experiment_history()

    if df.empty:
        st.info(
            "No experiment history yet. Run step 6 at least once — "
            "experiment data will appear here automatically."
        )
        return

    st.markdown(f"**{len(df)} clustering run(s) recorded**")

    display_cols = [
        "run_id", "run_date", "min_level", "method",
        "merge_threshold", "best_k", "best_silhouette",
        "n_clusters_before_merge", "n_clusters_after_merge", "n_merges",
    ]
    fmt = {"best_silhouette": "{:.4f}", "merge_threshold": "{:.2f}"}
    st.dataframe(
        df[display_cols].style.format(fmt, na_rep="—"),
        use_container_width=True,
    )

    # Comparison charts — only when there are ≥ 2 runs
    if len(df) >= 2:
        chart_df = df.sort_values("run_date")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Silhouette Score over Time")
            fig = px.line(
                chart_df,
                x="run_date",
                y="best_silhouette",
                markers=True,
                hover_data=["run_id", "min_level", "merge_threshold"],
                labels={"run_date": "Run Date", "best_silhouette": "Best Silhouette"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Cluster Count over Time")
            fig2 = px.bar(
                chart_df,
                x="run_date",
                y="n_clusters_after_merge",
                color=chart_df["min_level"].astype(str),
                hover_data=["run_id", "merge_threshold", "n_merges"],
                labels={
                    "n_clusters_after_merge": "Final Cluster Count",
                    "run_date": "Run Date",
                    "color": "min_level",
                },
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Per-run drill-down
    st.subheader("Run Detail")
    selected = st.selectbox("Select run", df["run_id"].tolist())
    if selected:
        row = df[df["run_id"] == selected].iloc[0]
        try:
            snap = json.loads(row["_config_snapshot"])
        except (ValueError, TypeError):
            snap = {}

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Input parameters**")
            st.json(snap.get("input", {}))
        with col2:
            st.markdown("**Outcome metrics**")
            st.json(snap.get("outcome", {}))

        merge_log = snap.get("merge_log", [])
        if merge_log:
            st.markdown("**Merge log**")
            st.dataframe(pd.DataFrame(merge_log), use_container_width=True)
        else:
            st.caption("No merges applied in this run.")

    # ── Compare Cluster Contents ──────────────────────────────────────────────
    st.markdown("---")
    with st.expander("Compare Cluster Contents across runs", expanded=False):
        if df.empty or len(df) < 2:
            st.info("Run the clustering pipeline at least twice (e.g. k=30, k=45, k=60) to compare cluster contents here.")
        else:
            run_labels = {}
            for _, row in df.iterrows():
                k_str = str(int(row["best_k"])) if pd.notna(row["best_k"]) else "?"
                sil_str = f"{float(row['best_silhouette']):.4f}" if row["best_silhouette"] is not None else "?"
                run_labels[row["run_id"]] = f"{row['run_id']}  (k={k_str}, silhouette={sil_str})"

            selected_labels = st.multiselect(
                "Select up to 3 runs to compare",
                options=list(run_labels.values()),
                default=list(run_labels.values())[:min(3, len(run_labels))],
                max_selections=3,
            )
            selected_run_ids = [rid for rid, lbl in run_labels.items() if lbl in selected_labels]

            if not selected_run_ids:
                st.info("Select at least one run above.")
            else:
                # Metrics comparison table
                st.markdown("#### Metrics comparison")
                metric_rows = []
                for rid in selected_run_ids:
                    row = df[df["run_id"] == rid].iloc[0]
                    metric_rows.append({
                        "Run": run_labels[rid],
                        "best_k": int(row["best_k"]) if pd.notna(row["best_k"]) else None,
                        "silhouette": round(float(row["best_silhouette"]), 4) if pd.notna(row["best_silhouette"]) else None,
                        "clusters (after merge)": int(row["n_clusters_after_merge"]) if pd.notna(row["n_clusters_after_merge"]) else None,
                        "merges": int(row["n_merges"]) if pd.notna(row["n_merges"]) else 0,
                    })
                st.dataframe(pd.DataFrame(metric_rows).set_index("Run"), use_container_width=True)

                # Probe pairs comparison
                probe_data = {}
                for rid in selected_run_ids:
                    row = df[df["run_id"] == rid].iloc[0]
                    try:
                        snap = json.loads(row["_config_snapshot"])
                        pairs = snap.get("outcome", {}).get("probe_pairs", [])
                        probe_data[rid] = {p["category"]: p["same_cluster"] for p in pairs}
                    except Exception:
                        probe_data[rid] = {}

                if any(probe_data.values()):
                    st.markdown("#### Probe pair quality check")
                    st.caption("✅ co-clustered  ⚠️ split  ⚪ not found")
                    all_categories = list(dict.fromkeys(
                        cat for d in probe_data.values() for cat in d
                    ))
                    probe_rows = []
                    for cat in all_categories:
                        probe_row = {"Pair": cat}
                        for rid in selected_run_ids:
                            val = probe_data[rid].get(cat)
                            if val is True:
                                icon = "✅"
                            elif val is False:
                                icon = "⚠️"
                            else:
                                icon = "⚪"
                            probe_row[run_labels[rid].split("  ")[0]] = icon
                        probe_rows.append(probe_row)
                    st.dataframe(pd.DataFrame(probe_rows).set_index("Pair"), use_container_width=True)

                # Cluster name comparison (top 30 per run)
                st.markdown("#### Top 30 clusters by size")
                clusters_by_run = load_cluster_names_for_runs(tuple(selected_run_ids))
                if clusters_by_run:
                    col_dfs = []
                    for rid in selected_run_ids:
                        short_label = run_labels[rid].split("  ")[0]
                        name_col = f"Cluster ({short_label})"
                        size_col = f"Size ({short_label})"
                        run_df = clusters_by_run.get(rid, pd.DataFrame())
                        if not run_df.empty:
                            run_df = run_df.head(30).reset_index(drop=True)
                            run_df.columns = [name_col, size_col]
                        else:
                            run_df = pd.DataFrame(columns=[name_col, size_col])
                        col_dfs.append(run_df)
                    combined = pd.concat(col_dfs, axis=1)
                    # Fill NaN (from runs with different cluster counts) and
                    # cast Size columns to object so Arrow serialises cleanly
                    combined = combined.fillna("")
                    for col in combined.columns:
                        if col.startswith("Size"):
                            combined[col] = combined[col].astype(str).replace("nan", "")
                    st.dataframe(combined, use_container_width=True)


def show_cluster_explorer(run_id):
    """Cluster explorer page"""

    # ── Fetch latest run config for dynamic methodology text ──────────────────
    _latest_min_level = 2  # sensible default
    _latest_merge_t = 0.0
    _latest_best_k = None
    try:
        hist_df = load_experiment_history()
        if not hist_df.empty:
            _latest = hist_df.iloc[0]
            _latest_min_level = int(_latest["min_level"]) if _latest["min_level"] is not None else 1
            _latest_merge_t = float(_latest["merge_threshold"]) if _latest["merge_threshold"] else 0.0
            _latest_best_k = int(_latest["best_k"]) if _latest["best_k"] is not None else None
    except Exception:
        pass

    _header_range = f"H{_latest_min_level}–H5" if _latest_min_level >= 2 else "H1–H5"
    _h1_excluded = _latest_min_level >= 2

    # ── Researcher context ────────────────────────────────────────────────────
    with st.expander("📖 How these clusters were built — decisions, limitations & quality checks", expanded=True):

        # Build the design decisions table dynamically
        _decisions = []
        if _h1_excluded:
            _decisions.append(
                "| **Exclude H1 headers** | `--min-level {lvl}` excludes H1 from clustering "
                "| H1 is almost always the project name (e.g. \"MyTool / A Python library for…\"). "
                "Including it creates noise clusters like \"cutadapt / sdmbench / name\" that don't "
                "reflect documentation structure. |".format(lvl=_latest_min_level)
            )
        _decisions.append(
            "| **Strip section numbers** | \"1. Installation\" → \"installation\" before embedding "
            "| Many READMEs number their sections. Without stripping, \"1. Installation\" and "
            "\"2. Installation\" appear as different headers and split the same concept across clusters. |"
        )
        _decisions.append(
            "| **Strip emojis & shortcodes** | \"🚀 Installation\" → \"installation\", "
            "\":pray:\" removed | Emojis and GitHub shortcodes add visual noise to headers. "
            "Without stripping, the same concept gets different embeddings depending on emoji usage. |"
        )
        _decisions.append(
            "| **Automatic k selection** | Silhouette score sweep over a k range "
            "| Rather than fixing k manually, we sweep a range and pick the k with the highest "
            "silhouette score — a measure of how well-separated the clusters are. |"
        )
        if _latest_merge_t > 0:
            _decisions.append(
                "| **Post-hoc merging** | `--merge-threshold {mt}` cosine similarity merge "
                "| Even with optimal k, semantically near-identical clusters sometimes split "
                "(e.g. \"Getting Started\" and \"Quick Start\"). Post-hoc merging computes centroid "
                "cosine similarity and merges pairs above the threshold. |".format(mt=_latest_merge_t)
            )
        _decisions_table = "\n".join(_decisions)

        st.markdown(f"""
### What is being clustered?

All README section headers ({_header_range}) extracted from research software repositories
across three sources: **JOSS**, **Research Software Directory**, and **Helmholtz**.
Each header is treated as a short text and embedded into a 384-dimensional semantic
vector using the `all-MiniLM-L6-v2` sentence-transformer model. K-Means is then run
on those vectors to group semantically similar headers into clusters.

---

### Design decisions made (and why)

| Decision | What we did | Why |
|----------|-------------|-----|
{_decisions_table}

---

### Known limitations

**Misspellings land in wrong clusters**
The embedding model has no spell-correction. `ackowledgements` (missing 'n') produces
a different vector from `acknowledgements` and can drift into an unrelated cluster
(e.g. "Functionality"). This is tracked automatically — see the **Spelling Variants**
section in each run's `.history/clustering/run_*.md` report.

**Project-specific headers don't generalise**
Headers like `"obiba acknowledgments"` or `"how to build and run the rsd"` contain
project names that pull their embedding away from the generic concept cluster.
These show up as outliers near the edge of a cluster (high distance to centroid).

**Cluster names are auto-generated**
Each cluster name is computed from the 3 most frequent words across the 5 headers
closest to the centroid. The name is a heuristic, not a manual label — it reflects
the dominant vocabulary but may not capture every member.

**Short or ambiguous headers**
Single-word headers like `"code"` or `"methods"` are ambiguous and may appear in
multiple semantically distinct clusters depending on surrounding context.

---

### Quality checks run after each pipeline execution

| Check | What it does |
|-------|-------------|
| **Probe pairs** | 8 known-similar header pairs (e.g. "installation" / "how to install") are looked up after each run to confirm they co-clustered. A ⚠️ split flags a potential over-split that merging or a lower k might fix. |
| **Spelling variant detection** | Headers with ≥ 0.75 string similarity to a probe word that landed in a *different* cluster are flagged. These are likely typos (e.g. `ackowledgements`, `acknowlegdements`) whose embeddings drifted away from the correct cluster. |
| **Merge log** | Records every cluster pair merged, their similarity, and size before/after — so you can trace why the final cluster count differs from the k selected. |

All checks are written to `.history/clustering/run_<run_id>.md` after each run and
are visible in the **Experiment History** page for comparison across runs.

---

### How to read the cluster members

- **Distance to centroid** — lower = more representative of the cluster's core concept
- **×N** — how many repositories use this exact normalized header text
- **Raw text shown** — the original header before normalisation (e.g. `"1. Installation"`)
  is shown where it differs from the normalized form
""")

    # ── Run selector ─────────────────────────────────────────────────────────
    hist_df = load_experiment_history()
    selected_hist = None
    if not hist_df.empty:
        def _run_label(r):
            parts = []
            if pd.notna(r["min_level"]):
                parts.append(f"min_level={int(r['min_level'])}")
            if pd.notna(r["best_k"]):
                parts.append(f"k={int(r['best_k'])}")
            if pd.notna(r["merge_threshold"]) and r["merge_threshold"]:
                parts.append(f"merge={r['merge_threshold']}")
            return f"{r['run_id']}  —  {', '.join(parts)}" if parts else r["run_id"]

        options = hist_df["run_id"].tolist()
        labels = [_run_label(hist_df[hist_df["run_id"] == rid].iloc[0]) for rid in options]
        label_to_run = dict(zip(labels, options))

        default_label = next(
            (lbl for lbl, rid in label_to_run.items() if rid == run_id),
            labels[0],
        )
        selected_label = st.selectbox(
            "Select clustering run",
            options=labels,
            index=labels.index(default_label),
            help="Choose which clustering run to explore. Each run may use different parameters.",
        )
        run_id = label_to_run[selected_label]
        selected_hist = hist_df[hist_df["run_id"] == run_id].iloc[0]

    cluster_df = load_cluster_data(run_id)

    if cluster_df.empty:
        st.info("No cluster data yet — run the full pipeline first.")
        return

    # ── Config summary metrics ────────────────────────────────────────────────
    if selected_hist is not None:
        col1, col2, col3, col4, col5 = st.columns(5)
        min_lvl = selected_hist["min_level"]
        col1.metric("Min level", f"H{int(min_lvl)}" if min_lvl is not None else "H1")
        best_k_val = selected_hist["best_k"]
        col2.metric("k (clusters)", int(best_k_val) if best_k_val is not None else "—")
        sil = selected_hist["best_silhouette"]
        col3.metric("Silhouette", f"{sil:.4f}" if sil is not None else "—")
        merge_t = selected_hist["merge_threshold"]
        col4.metric("Merge threshold", merge_t if merge_t else "off")
        n_merges = selected_hist["n_merges"]
        col5.metric("Merges applied", int(n_merges) if n_merges else 0)

    # ── Methodology / rationale ──────────────────────────────────────────────
    with st.expander("ℹ️ About this analysis", expanded=True):
        total_headers = sum(cluster_df["size"])
        total_clusters = len(cluster_df)
        avg_size = cluster_df["size"].mean()

        # Pull k-search range from the DB if available
        k_df_info = load_k_search_results(run_id)
        if not k_df_info.empty:
            k_min_actual = int(k_df_info["k"].min())
            k_max_actual = int(k_df_info["k"].max())
            k_range_str = f"k = {k_min_actual} … {k_max_actual}"
            k_method_str = "chosen automatically by maximising the silhouette score"
        else:
            k_range_str = "sweep not yet recorded"
            k_method_str = "set manually"

        # Derive dynamic values from selected run config
        if selected_hist is not None:
            _min_lvl = selected_hist["min_level"]
            _merge_t = selected_hist["merge_threshold"]
            _n_merges = int(selected_hist["n_merges"]) if selected_hist["n_merges"] else 0
        else:
            _min_lvl = None
            _merge_t = None
            _n_merges = 0

        if _min_lvl is not None and int(_min_lvl) >= 2:
            header_levels_str = f"H{int(_min_lvl)} – H5 (H1 project-name headings excluded)"
            h1_note = "- **H1 excluded:** Project-name headings (e.g. \"MyTool / Description\") were omitted — they inflate cluster noise without reflecting documentation structure."
        else:
            header_levels_str = "H1 – H5 (all Markdown heading levels)"
            h1_note = ("- **Why H1 headers are included:** H1 headings sometimes carry meaningful topic labels\n"
                       "  (e.g. \"Getting started\", \"API reference\") alongside project-name uses — including them\n"
                       "  lets the clustering vocabulary reflect the full heading practice in the corpus.\n"
                       "  To exclude H1 headers, re-run step 6 with `--min-level 2`.")

        merge_row = ""
        if _merge_t and float(_merge_t) > 0:
            merge_row = f"\n| Post-hoc merging | threshold {_merge_t} — {_n_merges} cluster pair(s) merged |"

        st.markdown(f"""
**What was clustered**

| Parameter | Value |
|-----------|-------|
| Total headers clustered | {total_headers:,} |
| Number of clusters (k) | {total_clusters} |
| Average cluster size | {avg_size:.0f} headers |
| Header levels included | {header_levels_str} |
| Source | README files from JOSS, Research Software Directory, and Helmholtz repositories |{merge_row}

**Header preprocessing (step 5)**

Before clustering, each raw header was filtered and normalised:

- **Excluded:** headers shorter than 3 characters or longer than 120 characters
- **Excluded:** version strings (`1.2.0`, `v1.0`), dates (`2024-01-05`), and pure digit strings
- **Stripped:** leading section numbers ("1. Installation" → "installation")
- **Stripped:** emojis and GitHub shortcodes (`:pray:`, `:rocket:`) before embedding
- **Normalised:** lowercased and whitespace-trimmed
{h1_note}

**Embedding model — `all-MiniLM-L6-v2`**

- Architecture: 6-layer MiniLM Transformer, **384-dimensional** sentence vectors
- Training: fine-tuned on 1 billion sentence pairs for semantic similarity
- Embeddings are **L2-normalised** before clustering so that Euclidean distance
  approximates cosine similarity — clusters are topic-based, not length-based
- Chosen for speed and accuracy on short-text inputs (README headers are typically 2–8 words)

**Clustering algorithm — K-Means (k = {total_clusters})**

- **k was {k_method_str}** from a silhouette + inertia sweep over {k_range_str}
  (see the **K-selection analysis** chart below)
- The silhouette score measures how well each header fits its own cluster vs. the
  nearest neighbouring cluster (range −1 … +1; higher = better separation)
- The inertia elbow is shown as a secondary signal
- `random_state = 42` ensures the run is reproducible

**Cluster naming**

Each cluster label was generated automatically:
1. Find the **5 headers** closest to the K-Means centroid (most representative)
2. Select the **3 most frequent words** across those 5 headers as the cluster name

No manual labelling was applied — names reflect the dominant vocabulary in each cluster.
""")

    # ── K-selection analysis chart ───────────────────────────────────────────
    st.subheader("K-selection analysis")
    k_df = load_k_search_results(run_id)

    if k_df.empty:
        st.info(
            "No k-selection data yet — re-run Step 6 "
            "(the k-search sweep runs by default; use `--no-search-k` to skip it)."
        )
    else:
        best_row = k_df[k_df["is_best"]].iloc[0] if k_df["is_best"].any() else None
        best_k = int(best_row["k"]) if best_row is not None else None

        # Dual-axis chart: inertia (bar, left) + silhouette (line, right)
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=k_df["k"],
                y=k_df["inertia"],
                name="Inertia (within-cluster variance)",
                marker_color="steelblue",
                opacity=0.6,
                yaxis="y1",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=k_df["k"],
                y=k_df["silhouette"],
                name="Silhouette score",
                mode="lines+markers",
                line={"color": "darkorange", "width": 2},
                marker={"size": 7},
                yaxis="y2",
            )
        )

        if best_k is not None:
            fig.add_vline(
                x=best_k,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Best k={best_k}",
                annotation_position="top right",
            )

        fig.update_layout(
            title="K-Means k-selection: inertia (elbow) vs silhouette score",
            xaxis={"title": "Number of clusters (k)", "dtick": k_df["k"].diff().median()},
            yaxis={"title": "Inertia", "side": "left", "showgrid": False},
            yaxis2={
                "title": "Silhouette score",
                "side": "right",
                "overlaying": "y",
                "showgrid": True,
                "range": [0, max(0.5, k_df["silhouette"].max() * 1.1)],
            },
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        if best_k is not None:
            best_sil = float(k_df[k_df["is_best"]]["silhouette"].iloc[0])
            actual_clusters = len(cluster_df)
            merge_note = ""
            if actual_clusters < best_k:
                merge_note = (
                    f" After post-hoc merging of highly similar cluster pairs, "
                    f"the final count is **{actual_clusters}** clusters."
                )
            st.caption(
                f"k = **{best_k}** was selected as the value with the highest "
                f"silhouette score ({best_sil:.4f}) across the sweep. "
                f"Silhouette ranges from -1 (poor) to 1 (perfect separation); "
                f"values above 0.1 indicate meaningful cluster structure for "
                f"high-dimensional text embeddings.{merge_note}"
            )

    # Filters
    col1, col2 = st.columns([2, 1])

    with col1:
        search_term = st.text_input("🔎 Search clusters by name", "")

    with col2:
        min_size = st.number_input("Min cluster size", min_value=0, value=0)

    # Filter
    if search_term:
        cluster_df = cluster_df[cluster_df["name"].str.contains(search_term, case=False, na=False)]

    if min_size > 0:
        cluster_df = cluster_df[cluster_df["size"] >= min_size]

    st.markdown(f"**Showing {len(cluster_df)} clusters**")

    # Display clusters
    for _, row in cluster_df.iterrows():
        cluster_name = _clean_header_display(row["name"])
        with st.expander(f"**{cluster_name}** ({row['size']} headers)"):
            st.markdown(f"**Cluster ID:** {row['cluster_id']}")
            st.markdown(f"**Size:** {row['size']} headers")

            st.markdown("**30 most representative unique headers (closest to cluster centre):**")
            members = load_cluster_members(int(row["id"]), limit=30)
            if members:
                for i, (header_text, dist, cnt) in enumerate(members, 1):
                    count_str = (
                        f"  <span style='color:#888;font-size:0.8em'>×{cnt}</span>"
                        if cnt > 1
                        else ""
                    )
                    dist_str = (
                        f"  <span style='color:grey;font-size:0.8em'>dist={dist:.3f}</span>"
                        if dist is not None
                        else ""
                    )
                    st.markdown(f"{i}. `{header_text}`{count_str}{dist_str}", unsafe_allow_html=True)
            else:
                # fallback: use stored representative headers if live query returns nothing
                for i, header in enumerate(row["representative_headers"][:30], 1):
                    st.markdown(f"{i}. `{_clean_header_display(header)}`")

    # ── References ───────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        "**References**\n\n"
        "- **Embedding model:** all-MiniLM-L6-v2 — "
        "[Hugging Face model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) "
        "· [sentence-transformers library](https://www.sbert.net/)\n"
        "- **Clustering:** scikit-learn K-Means — "
        "[documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)\n"
        "- **Dimensionality reduction (visualisation):** UMAP — "
        "[documentation](https://umap-learn.readthedocs.io/)\n"
    )


def show_visualization():
    """Visualization page"""

    st.info("⚠️ Loading and projecting embeddings may take a minute...")

    # Options
    col1, col2 = st.columns(2)

    with col1:
        max_samples = st.slider("Max samples to visualize", 1000, 10000, 5000, 500)

    with col2:
        dimensions = st.radio("Projection", ["2D", "3D"])

    n_components = 2 if dimensions == "2D" else 3

    if st.button("Generate Visualization", type="primary"):
        with st.spinner("Loading embeddings..."):
            header_ids, embeddings, assignments = load_embeddings_for_viz(max_samples)

        with st.spinner(f"Computing {dimensions} UMAP projection..."):
            projection = compute_umap(embeddings, n_components)

        # Create dataframe
        if n_components == 2:
            df = pd.DataFrame(
                {
                    "header_id": header_ids,
                    "x": projection[:, 0],
                    "y": projection[:, 1],
                    "cluster": [assignments.get(hid, "Unassigned") for hid in header_ids],
                }
            )

            fig = px.scatter(
                df,
                x="x",
                y="y",
                color="cluster",
                title=f"{dimensions} UMAP Projection of Header Embeddings",
                labels={"x": "UMAP 1", "y": "UMAP 2"},
                hover_data=["header_id"],
            )
            fig.update_traces(marker={"size": 5, "opacity": 0.7})

        else:  # 3D
            df = pd.DataFrame(
                {
                    "header_id": header_ids,
                    "x": projection[:, 0],
                    "y": projection[:, 1],
                    "z": projection[:, 2],
                    "cluster": [assignments.get(hid, "Unassigned") for hid in header_ids],
                }
            )

            fig = px.scatter_3d(
                df,
                x="x",
                y="y",
                z="z",
                color="cluster",
                title=f"{dimensions} UMAP Projection of Header Embeddings",
                labels={"x": "UMAP 1", "y": "UMAP 2", "z": "UMAP 3"},
                hover_data=["header_id"],
            )
            fig.update_traces(marker={"size": 3, "opacity": 0.6})

        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)

        st.success(f"✅ Visualized {len(df):,} headers across {df['cluster'].nunique()} clusters")


def show_cluster_network(run_id: str | None = None):
    """Interactive force-directed network of cluster similarities."""
    import networkx as nx

    sim_threshold = st.slider(
        "Similarity threshold (edges shown above this value)",
        min_value=0.3, max_value=0.95, value=0.6, step=0.05,
        help="Only connections with cosine similarity above this threshold are shown.",
    )

    with st.spinner("Computing cluster centroids and similarity matrix..."):
        names, sizes, sim_matrix = load_cluster_similarity_matrix(run_id)

    if len(names) < 2:
        st.info("Not enough cluster data to build a network graph.")
        return

    # Build networkx graph
    G = nx.Graph()
    for i, name in enumerate(names):
        G.add_node(i, label=name, size=sizes[i])

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            sim = float(sim_matrix[i][j])
            if sim >= sim_threshold:
                G.add_edge(i, j, weight=sim)

    if G.number_of_edges() == 0:
        st.warning(f"No cluster pairs have similarity >= {sim_threshold:.2f}. Try lowering the threshold.")
        return

    # Force-directed layout
    pos = nx.spring_layout(G, k=2.5, iterations=80, seed=42, weight="weight")

    # Edge traces
    edge_x, edge_y = [], []
    edge_mid_x, edge_mid_y, edge_labels = [], [], []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_mid_x.append((x0 + x1) / 2)
        edge_mid_y.append((y0 + y1) / 2)
        edge_labels.append(f"{names[u]} — {names[v]}<br>similarity: {d['weight']:.3f}")

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color="#aaa"),
        hoverinfo="none",
        mode="lines",
    )

    # Edge hover (invisible midpoints)
    edge_hover_trace = go.Scatter(
        x=edge_mid_x, y=edge_mid_y,
        mode="markers",
        marker=dict(size=10, color="rgba(0,0,0,0)"),
        hoverinfo="text",
        text=edge_labels,
    )

    # Node trace
    node_x = [pos[i][0] for i in range(len(names))]
    node_y = [pos[i][1] for i in range(len(names))]

    # Scale node sizes: min 15, max 60
    max_size = max(sizes) if sizes else 1
    node_sizes = [max(15, int(s / max_size * 60)) for s in sizes]

    # Color by number of connections
    node_degrees = [G.degree(i) for i in range(len(names))]

    node_hover = [
        f"<b>{names[i]}</b><br>"
        f"Headers: {sizes[i]:,}<br>"
        f"Connections: {node_degrees[i]}"
        for i in range(len(names))
    ]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=[n if sizes[i] > max_size * 0.05 else "" for i, n in enumerate(names)],
        textposition="top center",
        textfont=dict(size=9),
        hoverinfo="text",
        hovertext=node_hover,
        marker=dict(
            size=node_sizes,
            color=node_degrees,
            colorscale="YlOrRd",
            colorbar=dict(title="Connections", thickness=15),
            line=dict(width=1, color="#333"),
        ),
    )

    fig = go.Figure(
        data=[edge_trace, edge_hover_trace, node_trace],
        layout=go.Layout(
            title=f"Cluster Similarity Network ({len(names)} clusters, "
                  f"{G.number_of_edges()} edges at threshold >= {sim_threshold:.2f})",
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=750,
            margin=dict(l=20, r=20, t=50, b=20),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Clusters", len(names))
    col2.metric("Edges (connections)", G.number_of_edges())
    avg_sim = np.mean([d["weight"] for _, _, d in G.edges(data=True)]) if G.number_of_edges() > 0 else 0
    col3.metric("Avg similarity", f"{avg_sim:.3f}")

    # Show strongest connections table
    with st.expander("Strongest cluster connections", expanded=False):
        edge_data = [
            {"Cluster A": names[u], "Cluster B": names[v], "Similarity": round(d["weight"], 4)}
            for u, v, d in G.edges(data=True)
        ]
        edge_df = pd.DataFrame(edge_data).sort_values("Similarity", ascending=False)
        st.dataframe(edge_df, use_container_width=True, hide_index=True)

    # Show isolated clusters (no connections)
    isolated = [names[i] for i in range(len(names)) if G.degree(i) == 0]
    if isolated:
        with st.expander(f"Isolated clusters ({len(isolated)} — no connections at this threshold)", expanded=False):
            for name in isolated:
                st.write(f"- {name}")


def show_search():
    """Header Cluster Lookup page — keyword search grouped by unique header text."""

    st.caption(
        "**Keyword search** across all README header texts in the corpus. "
        "Type a word or phrase and see every unique header variant that contains it, "
        "how many repositories use it, and which cluster it was assigned to. "
        "Expand any row to browse the individual repositories."
    )

    search_query = st.text_input("Search headers by keyword", "", placeholder="e.g.  install,  usage,  contributing,  docker")

    if not search_query:
        return

    term = search_query.lower().strip()

    from sqlalchemy import text as sa_text

    groups = []
    repo_details: dict[str, list] = {}

    with get_session_context() as session:
        # ── Level 1: aggregated by (normalized_text, cluster) ────────────────
        agg_rows = session.execute(
            sa_text("""
                SELECT
                    rh.normalized_text,
                    cl.cluster_name,
                    COUNT(DISTINCT rh.repository_id) AS repo_count
                FROM readme_headers rh
                LEFT JOIN header_cluster_assignments hca ON hca.header_id = rh.id
                LEFT JOIN clusters cl ON cl.id = hca.cluster_id
                WHERE rh.normalized_text ILIKE :term
                GROUP BY rh.normalized_text, cl.cluster_name
                ORDER BY repo_count DESC
                LIMIT 50
            """),
            {"term": f"%{term}%"},
        ).fetchall()

        if not agg_rows:
            st.info(f"No headers found containing **{search_query}**.")
            return

        groups = [(r[0], r[1], r[2]) for r in agg_rows]
        unique_texts = list({r[0] for r in groups})

        # ── Level 2: individual repos per normalized_text ────────────────────
        detail_rows = session.execute(
            sa_text("""
                SELECT
                    rh.normalized_text,
                    rh.header_text,
                    r.url,
                    rh.level,
                    rh.position
                FROM readme_headers rh
                JOIN repositories r ON r.id = rh.repository_id
                WHERE rh.normalized_text = ANY(:texts)
                ORDER BY rh.normalized_text, r.url
                LIMIT 2000
            """),
            {"texts": unique_texts},
        ).fetchall()

        for norm_text, header_text, url, level, pos in detail_rows:
            repo_details.setdefault(norm_text, []).append((header_text, url, level, pos))

    total_repos = sum(count for _, _, count in groups)
    st.markdown(
        f"**{len(groups)} unique header variant(s)** matched — "
        f"used across **{total_repos:,} repository occurrences** (showing top 50 variants)"
    )

    for norm_text, cluster_name, repo_count in groups:
        display_name = _clean_header_display(norm_text or "")
        cluster_label = cluster_name or "Unassigned"
        with st.expander(f"`{display_name}` → **{cluster_label}** × {repo_count:,} repos"):
            repos = repo_details.get(norm_text, [])
            for header_text, url, level, pos in repos[:25]:
                raw_label = (
                    f"  <span style='color:#888;font-size:0.8em'>"
                    f"(raw: `{header_text}`)</span>"
                    if header_text.lower() != norm_text
                    else ""
                )
                level_pos = (
                    f"  <span style='color:#aaa;font-size:0.8em'>H{level} · pos {pos}</span>"
                )
                if url:
                    st.markdown(
                        f"- [{url}]({url}){raw_label}{level_pos}",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"- *(repo URL unavailable)*{raw_label}{level_pos}",
                        unsafe_allow_html=True,
                    )
            if len(repos) > 25:
                st.caption(f"…and {len(repos) - 25} more repositories")
            if cluster_label == "Unassigned":
                st.caption("ℹ️ Not assigned to a cluster — header may be H1 or from a run without clustering.")


def show_export(run_id):
    """Export page"""

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    cluster_df = load_cluster_data(run_id)

    if cluster_df.empty:
        st.info("No data to export yet — run the full pipeline first.")
        return

    st.markdown("### Download Options")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Cluster Summary (CSV)")
        _dl = cluster_df[["cluster_id", "name", "size", "sample"]].copy()
        if run_id:
            _dl["run_id"] = run_id
        st.download_button(
            label="Download CSV",
            data=_dl.to_csv(index=False),
            file_name=f"cluster_summary_{ts}.csv",
            mime="text/csv",
        )

    with col2:
        st.markdown("#### Full Report (JSON)")
        json_data = cluster_df.to_json(orient="records", indent=2)
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name=f"cluster_report_{ts}.json",
            mime="application/json",
        )

    st.markdown("---")

    st.markdown("### Research Summary")

    stats = load_overview_stats()

    summary = f"""
## Research Software Metadata Analyser

**Date:** {pd.Timestamp.now().strftime("%Y-%m-%d")}
**Run ID:** {run_id}

### Dataset
- **Repositories Analyzed:** {stats["total_repos"]:,}
- **Headers Extracted:** {stats["total_headers"]:,}
- **Average Headers per Repository:** {(stats["total_headers"] / stats["total_repos"] if stats["total_repos"] > 0 else 0.0):.1f}

### Clustering Results
- **Number of Clusters:** {stats["total_clusters"]}
- **Headers Clustered:** {stats["total_assignments"]:,} ({(stats["total_assignments"] / stats["total_headers"] * 100 if stats["total_headers"] > 0 else 0.0):.1f}%)
- **Average Cluster Size:** {cluster_df["size"].mean():.0f}
- **Median Cluster Size:** {cluster_df["size"].median():.0f}

### Top 10 Clusters
{cluster_df[["name", "size"]].head(10).to_markdown(index=False)}

### Methodology
- **Embedding Model:** all-MiniLM-L6-v2 (384 dimensions)
- **Clustering Algorithm:** K-Means (k={len(cluster_df)})
- **Distance Metric:** Euclidean distance on L2-normalised embeddings
"""

    st.markdown(summary)

    st.download_button(
        label="Download Summary (Markdown)",
        data=summary,
        file_name=f"research_summary_{ts}.md",
        mime="text/markdown",
    )


def show_license_analysis():
    """License analysis page"""

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    license_df = load_license_data()

    if license_df.empty:
        st.info("No repository data yet — run the pipeline first.")
        return

    total_repos = len(license_df)
    with_license = len(license_df[license_df["license"] != "No License"])
    without_license = total_repos - with_license

    st.subheader("📊 Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Repositories", f"{total_repos:,}")
    with col2:
        st.metric(
            "With License",
            f"{with_license:,}",
            delta=f"{with_license / total_repos * 100:.1f}%" if total_repos else "0%",
        )
    with col3:
        st.metric(
            "Without License",
            f"{without_license:,}",
            delta=f"-{without_license / total_repos * 100:.1f}%" if total_repos else "0%",
            delta_color="inverse",
        )
    with col4:
        unique_licenses = license_df[license_df["license"] != "No License"]["license"].nunique()
        st.metric("Unique Licenses", unique_licenses)

    # ── Drilldown: repos without a license ────────────────────────────────
    if without_license > 0:
        with st.expander(f"View {without_license} repositories without a license"):
            no_lic_df = (
                license_df[license_df["license"] == "No License"][
                    ["name", "url", "platform", "source"]
                ]
                .copy()
                .reset_index(drop=True)
            )

            # Platform filter
            platforms = sorted(no_lic_df["platform"].unique().tolist())
            if len(platforms) > 1:
                sel = st.multiselect(
                    "Filter by platform", platforms, default=platforms, key="no_lic_platform_filter"
                )
                no_lic_df = no_lic_df[no_lic_df["platform"].isin(sel)]

            st.dataframe(
                no_lic_df,
                column_config={
                    "url": st.column_config.LinkColumn("URL"),
                    "name": "Name",
                    "platform": "Platform",
                    "source": "Source",
                },
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download list (CSV)",
                data=no_lic_df.to_csv(index=False),
                file_name=f"repos_without_license_{ts}.csv",
                mime="text/csv",
                key="no_lic_csv_dl",
            )

    st.markdown("---")
    st.subheader("🏆 Top 10 Licenses")

    license_counts = license_df["license"].value_counts().head(10)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            x=license_counts.values,
            y=license_counts.index,
            orientation="h",
            title="Most Common Licenses",
            labels={"x": "Number of Repositories", "y": "License"},
            color=license_counts.values,
            color_continuous_scale="Blues",
        )
        fig.update_layout(height=400, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.pie(
            values=license_counts.values,
            names=license_counts.index,
            title="License Distribution",
            hole=0.3,
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📂 License Categories")

    license_df["category"] = license_df["license"].apply(categorize_license)
    category_counts = license_df["category"].value_counts()

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            x=category_counts.index,
            y=category_counts.values,
            title="Licenses by Category",
            labels={"x": "Category", "y": "Count"},
            color=category_counts.values,
            color_continuous_scale="Viridis",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("#### Category Breakdown")
        for category, count in category_counts.items():
            pct = (count / total_repos * 100) if total_repos else 0
            st.metric(category, f"{count:,}", delta=f"{pct:.1f}%")

    st.markdown("---")
    st.download_button(
        label="Download License Report (CSV)",
        data=license_df.to_csv(index=False),
        file_name=f"license_analysis_{ts}.csv",
        mime="text/csv",
    )


def show_repository_browser():
    """Repository browser page"""

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    repo_df = load_repository_details()

    if repo_df.empty:
        st.info("No repository data yet — run the pipeline first.")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Repositories", f"{len(repo_df):,}")
    with col2:
        st.metric("With License", f"{len(repo_df[repo_df['license'] != 'No License']):,}")
    with col3:
        st.metric("Languages", repo_df["language"].nunique())
    with col4:
        st.metric("Total Stars", f"{repo_df['stars'].sum():,}")

    # ── Platform summary ──────────────────────────────────────────────────
    platform_counts = repo_df["platform"].value_counts()
    plat_cols = st.columns(len(platform_counts))
    for col, (plat, count) in zip(plat_cols, platform_counts.items()):
        col.metric(plat.capitalize(), f"{count:,}")

    st.markdown("---")
    st.subheader("🔍 Filters")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        selected_license = st.selectbox(
            "License", ["All"] + sorted(repo_df["license"].unique().tolist())
        )
    with col2:
        selected_language = st.selectbox(
            "Language", ["All"] + sorted(repo_df["language"].unique().tolist())
        )
    with col3:
        source_options = ["All"] + sorted(repo_df["source"].unique().tolist())
        selected_source = st.selectbox("Source", source_options)
    with col4:
        platform_options = ["All"] + sorted(repo_df["platform"].unique().tolist())
        selected_platform = st.selectbox("Platform", platform_options)
    with col5:
        search_term = st.text_input("Search by name", "")

    filtered = repo_df.copy()
    if selected_license != "All":
        filtered = filtered[filtered["license"] == selected_license]
    if selected_language != "All":
        filtered = filtered[filtered["language"] == selected_language]
    if selected_source != "All":
        filtered = filtered[filtered["source"] == selected_source]
    if selected_platform != "All":
        filtered = filtered[filtered["platform"] == selected_platform]
    if search_term:
        filtered = filtered[filtered["name"].str.contains(search_term, case=False, na=False)]

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**Showing {len(filtered):,} of {len(repo_df):,} repositories**")
    with col2:
        sort_by = st.selectbox("Sort by", ["Name", "Stars", "License", "Source"])

    if sort_by == "Stars":
        filtered = filtered.sort_values("stars", ascending=False)
    elif sort_by == "License":
        filtered = filtered.sort_values("license")
    elif sort_by == "Source":
        filtered = filtered.sort_values("source")
    else:
        filtered = filtered.sort_values("name")

    st.dataframe(
        filtered[["name", "platform", "license", "language", "stars", "source", "url"]].head(500),
        column_config={
            "name": st.column_config.TextColumn("Repository", width="medium"),
            "platform": st.column_config.TextColumn("Platform", width="small"),
            "license": st.column_config.TextColumn("License", width="medium"),
            "language": st.column_config.TextColumn("Language", width="small"),
            "stars": st.column_config.NumberColumn("⭐ Stars", width="small"),
            "source": st.column_config.TextColumn("Source", width="small"),
            "url": st.column_config.LinkColumn("Link", width="medium"),
        },
        hide_index=True,
        use_container_width=True,
        height=600,
    )
    if len(filtered) > 500:
        st.info("Showing first 500 results. Use filters to narrow down.")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Repository Sources**")
        filtered["domain"] = filtered["url"].apply(
            lambda x: urlparse(x).netloc if pd.notna(x) else "Unknown"
        )
        for domain, count in filtered["domain"].value_counts().head(5).items():
            st.write(f"- **{domain}**: {count} repos")
    with col2:
        st.markdown("**Top 5 Licenses**")
        for lic, count in filtered["license"].value_counts().head(5).items():
            st.write(f"- **{lic}**: {count} repos")

    st.markdown("---")
    st.download_button(
        label=f"Download {len(filtered)} Repositories (CSV)",
        data=filtered.drop(columns=["domain"], errors="ignore").to_csv(index=False),
        file_name=f"repository_list_{ts}.csv",
        mime="text/csv",
    )

    # ── Excluded Repositories ─────────────────────────────────────────────
    st.markdown("---")
    excl_df = load_excluded_repos()
    with st.expander(f"🚫 Excluded Repositories ({len(excl_df):,} total)", expanded=False):
        if excl_df.empty:
            st.info("No excluded repositories recorded.")
        else:
            excl_col1, excl_col2, excl_col3 = st.columns(3)
            with excl_col1:
                excl_source = st.selectbox(
                    "Filter by source",
                    ["All"] + sorted(excl_df["source"].unique().tolist()),
                    key="excl_source",
                )
            with excl_col2:
                excl_stage = st.selectbox(
                    "Filter by stage",
                    ["All"] + sorted(excl_df["exclusion_stage"].unique().tolist()),
                    key="excl_stage",
                )
            with excl_col3:
                excl_search = st.text_input("Search URL", "", key="excl_search")

            excl_filtered = excl_df.copy()
            if excl_source != "All":
                excl_filtered = excl_filtered[excl_filtered["source"] == excl_source]
            if excl_stage != "All":
                excl_filtered = excl_filtered[excl_filtered["exclusion_stage"] == excl_stage]
            if excl_search:
                excl_filtered = excl_filtered[
                    excl_filtered["url"].str.contains(excl_search, case=False, na=False)
                ]

            st.markdown(
                f"**Showing {len(excl_filtered):,} of {len(excl_df):,} excluded repositories**"
            )
            st.dataframe(
                excl_filtered[
                    [
                        "url",
                        "exclusion_reason",
                        "exclusion_stage",
                        "source",
                        "retryable",
                        "excluded_at",
                    ]
                ].head(500),
                column_config={
                    "url": st.column_config.LinkColumn("URL", width="large"),
                    "exclusion_reason": st.column_config.TextColumn("Reason", width="large"),
                    "exclusion_stage": st.column_config.TextColumn("Stage", width="small"),
                    "source": st.column_config.TextColumn("Source", width="small"),
                    "retryable": st.column_config.CheckboxColumn("Retryable", width="small"),
                    "excluded_at": st.column_config.DatetimeColumn("Excluded At", width="medium"),
                },
                hide_index=True,
                use_container_width=True,
                height=400,
            )
            st.download_button(
                label=f"Download {len(excl_filtered)} Excluded Repos (CSV)",
                data=excl_filtered.to_csv(index=False),
                file_name=f"excluded_repositories_{ts}.csv",
                mime="text/csv",
            )

    # ── Unsupported-Platform Repositories ─────────────────────────────────
    st.markdown("---")
    unsup_df = load_unsupported_repos()
    with st.expander(
        f"⚠️ Unsupported Platform Repositories ({len(unsup_df):,} total)", expanded=False
    ):
        if unsup_df.empty:
            st.info(
                "No unsupported-platform repositories recorded yet. "
                "They are collected from step0 sources (RSD, Helmholtz, JOSS) "
                "when a URL's hosting platform is not supported by the pipeline scraper."
            )
        else:
            st.caption(
                "These URLs were found by the step0 collectors but their hosting platform "
                "is not yet supported by the pipeline scraper (step2). "
                "Hosts appearing frequently are the best candidates for future support — "
                "add them to `_GITLAB_SUBSTRINGS` in `scrapers/utils.py` or build a new scraper."
            )

            # ── Summary metrics ──────────────────────────────────────────
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Unique URLs", f"{len(unsup_df):,}")
            with col2:
                st.metric("Distinct Hosts", f"{unsup_df['host'].nunique():,}")
            with col3:
                st.metric("Distinct Platforms", f"{unsup_df['platform'].nunique():,}")

            # ── Host frequency bar chart ─────────────────────────────────
            host_counts = (
                unsup_df.groupby("host")["seen"]
                .sum()
                .sort_values(ascending=False)
                .head(20)
                .reset_index()
                .rename(columns={"seen": "occurrences"})
            )
            fig = px.bar(
                host_counts,
                x="occurrences",
                y="host",
                orientation="h",
                title="Top 20 Unsupported Hosts (by total occurrences)",
                labels={"host": "Host", "occurrences": "Occurrences"},
                height=max(300, len(host_counts) * 26),
            )
            fig.update_layout(yaxis={"autorange": "reversed"}, margin={"l": 10, "r": 10})
            st.plotly_chart(fig, use_container_width=True)

            # ── Filters ──────────────────────────────────────────────────
            ucol1, ucol2, ucol3 = st.columns(3)
            with ucol1:
                u_platform = st.selectbox(
                    "Filter by platform",
                    ["All"] + sorted(unsup_df["platform"].unique().tolist()),
                    key="unsup_platform",
                )
            with ucol2:
                u_source = st.selectbox(
                    "Filter by source",
                    ["All"] + sorted(unsup_df["source"].unique().tolist()),
                    key="unsup_source",
                )
            with ucol3:
                u_search = st.text_input("Search URL / host", "", key="unsup_search")

            unsup_filtered = unsup_df.copy()
            if u_platform != "All":
                unsup_filtered = unsup_filtered[unsup_filtered["platform"] == u_platform]
            if u_source != "All":
                unsup_filtered = unsup_filtered[unsup_filtered["source"] == u_source]
            if u_search:
                mask = unsup_filtered["url"].str.contains(
                    u_search, case=False, na=False
                ) | unsup_filtered["host"].str.contains(u_search, case=False, na=False)
                unsup_filtered = unsup_filtered[mask]

            st.markdown(f"**Showing {len(unsup_filtered):,} of {len(unsup_df):,} records**")
            st.dataframe(
                unsup_filtered[
                    ["url", "source", "host", "platform", "seen", "first_seen", "last_seen"]
                ].head(500),
                column_config={
                    "url": st.column_config.LinkColumn("URL", width="large"),
                    "source": st.column_config.TextColumn("Source", width="small"),
                    "host": st.column_config.TextColumn("Host", width="medium"),
                    "platform": st.column_config.TextColumn("Platform", width="small"),
                    "seen": st.column_config.NumberColumn("# Seen", width="small"),
                    "first_seen": st.column_config.DatetimeColumn("First Seen", width="medium"),
                    "last_seen": st.column_config.DatetimeColumn("Last Seen", width="medium"),
                },
                hide_index=True,
                use_container_width=True,
                height=400,
            )
            if len(unsup_filtered) > 500:
                st.info("Showing first 500 results. Use filters to narrow down.")

            st.download_button(
                label=f"Download {len(unsup_filtered)} Unsupported Repos (CSV)",
                data=unsup_filtered.to_csv(index=False),
                file_name=f"unsupported_repositories_{ts}.csv",
                mime="text/csv",
            )


def show_data_quality():
    """Data quality and processing transparency report"""

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    with get_session_context() as session:
        total_repos = session.query(Repository).count()
        total_readmes = session.query(ReadmeHeader.repository_id).distinct().count()
        total_headers = session.query(ReadmeHeader).count()
        total_embeddings = session.query(HeaderEmbedding).count()
        total_clustered = session.query(HeaderClusterAssignment).count()
        total_clusters = session.query(Cluster).count()
        total_excluded = (
            session.execute(text("SELECT COUNT(*) FROM excluded_repositories")).scalar() or 0
        )

    if total_repos == 0:
        st.info("No data yet — run the pipeline first.")
        return

    repos_without_readme = total_repos - total_readmes
    avg_headers = total_headers / total_readmes if total_readmes > 0 else 0
    ORIGINAL_SCRAPED = total_repos + total_excluded
    invalid_removed = total_excluded
    retention = total_repos / ORIGINAL_SCRAPED * 100 if ORIGINAL_SCRAPED else 100.0
    readme_coverage = total_readmes / total_repos * 100 if total_repos else 0
    cluster_coverage = total_clustered / total_headers * 100 if total_headers else 0

    st.subheader("🔄 Processing Pipeline")

    pipeline_df = pd.DataFrame(
        {
            "Stage": [
                "1. Initial Scraping",
                "2. Valid Repositories",
                "3. README Extracted",
                "4. Headers Extracted",
                "5. Embeddings Generated",
                "6. Clustered",
            ],
            "Count": [
                ORIGINAL_SCRAPED,
                total_repos,
                total_readmes,
                total_headers,
                total_embeddings,
                total_clustered,
            ],
            "Lost": [
                0,
                invalid_removed,
                repos_without_readme,
                0,
                total_headers - total_embeddings,
                total_embeddings - total_clustered,
            ],
            "Retention %": [
                100.0,
                round(total_repos / ORIGINAL_SCRAPED * 100, 1),
                round(total_readmes / total_repos * 100, 1) if total_repos else 0,
                100.0,
                round(total_embeddings / total_headers * 100, 1) if total_headers else 0,
                round(total_clustered / total_embeddings * 100, 1) if total_embeddings else 0,
            ],
        }
    )

    st.dataframe(
        pipeline_df,
        column_config={
            "Stage": st.column_config.TextColumn("Processing Stage", width="large"),
            "Count": st.column_config.NumberColumn("Count", format="%d"),
            "Lost": st.column_config.NumberColumn("Lost", format="%d"),
            "Retention %": st.column_config.NumberColumn("Retention", format="%.1f%%"),
        },
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("📊 Data Flow")

    fig = go.Figure(
        data=[
            go.Sankey(
                node={
                    "pad": 15,
                    "thickness": 20,
                    "line": {"color": "black", "width": 0.5},
                    "label": [
                        f"Initial ({ORIGINAL_SCRAPED:,})",
                        f"Valid ({total_repos:,})",
                        f"With README ({total_readmes:,})",
                        f"Headers ({total_headers:,})",
                        f"Embeddings ({total_embeddings:,})",
                        f"Clustered ({total_clustered:,})",
                    ],
                    "color": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
                },
                link={
                    "source": [0, 1, 2, 3, 4],
                    "target": [1, 2, 3, 4, 5],
                    "value": [
                        total_repos,
                        total_readmes,
                        total_headers,
                        total_embeddings,
                        total_clustered,
                    ],
                },
            )
        ]
    )
    fig.update_layout(title="Repository Processing Flow", height=400, font_size=12)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Retention Rate", f"{retention:.1f}%")
    with col2:
        st.metric("README Coverage", f"{readme_coverage:.1f}%")
    with col3:
        st.metric("Cluster Coverage", f"{cluster_coverage:.1f}%")
    with col4:
        st.metric("Avg Headers/README", f"{avg_headers:.1f}")

    st.markdown("---")
    st.subheader("⚖️ Potential Biases & Limitations")
    st.markdown("""
1. **Format bias** — only Markdown READMEs are parsed; RST / AsciiDoc / plain text may be under-represented.
2. **Platform bias** — mixed sources (GitHub, GitLab, Zenodo, 4TU); documentation conventions differ.
3. **Temporal bias** — snapshot from a specific date; active vs. archived repositories not distinguished.
4. **Completeness bias** — repositories without a README are excluded from clustering.

**Mitigations:** high retention rate, large sample size, transparent exclusion reporting, cross-platform deduplication.
""")

    st.markdown("---")
    report_md = f"""# Data Quality Report
Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}

## Pipeline

{pipeline_df.to_markdown(index=False)}

## Summary
- Initial repositories: {ORIGINAL_SCRAPED:,}
- Final repositories: {total_repos:,} ({retention:.1f}% retention)
- README coverage: {readme_coverage:.1f}% ({total_readmes:,}/{total_repos:,})
- Headers extracted: {total_headers:,} (avg {avg_headers:.1f}/README)
- Clustering coverage: {cluster_coverage:.1f}% ({total_clustered:,}/{total_headers:,})
- Clusters: {total_clusters}
"""
    st.download_button(
        label="Download Data Quality Report (Markdown)",
        data=report_md,
        file_name=f"data_quality_report_{ts}.md",
        mime="text/markdown",
    )


def show_somef_validation():
    """SOMEF metadata extraction results page."""
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    data = load_somef_stats()

    if data["total"] == 0:
        st.info(
            "No SOMEF results yet. Run step 7 to populate this page:\n\n"
            "```bash\n"
            'pip install -e ".[somef]"\n'
            "python -X utf8 step7_somef_validation.py --limit 200 --threshold 0.8\n"
            "```"
        )
        return

    # ── Summary metrics ───────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Repos Validated", f"{data['total']:,}")
    with col2:
        st.metric("Successful", f"{data['total'] - data['errors']:,}")
    with col3:
        st.metric("Avg Categories Found", f"{data['avg_cats']:.1f} / {len(data['categories'])}")
    with col4:
        st.metric("Avg Processing Time", f"{data['avg_time']:.1f}s")

    # ── Category coverage bar chart ───────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Category Coverage")
    st.caption("How many repos had each SOMEF category detected (out of total validated).")

    cat_df = pd.DataFrame(
        {
            "Category": [c.capitalize() for c in data["categories"]],
            "Repos": [data["cat_counts"][c] for c in data["categories"]],
            "Coverage %": [
                round(data["cat_counts"][c] / data["total"] * 100, 1) for c in data["categories"]
            ],
        }
    ).sort_values("Repos", ascending=False)

    fig = px.bar(
        cat_df,
        x="Category",
        y="Coverage %",
        text="Repos",
        color="Coverage %",
        color_continuous_scale="Blues",
        labels={"Coverage %": "Coverage (%)"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=420, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        cat_df,
        column_config={
            "Category": st.column_config.TextColumn("Category", width="medium"),
            "Repos": st.column_config.NumberColumn("# Repos", width="small"),
            "Coverage %": st.column_config.NumberColumn(
                "Coverage %", format="%.1f%%", width="small"
            ),
        },
        hide_index=True,
        use_container_width=True,
    )

    # ── Per-repo browser ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔍 Per-Repository Results")

    df = data["df"]
    col1, col2 = st.columns(2)
    with col1:
        min_cats = st.slider("Minimum categories found", 0, len(data["categories"]), 0)
    with col2:
        show_errors = st.checkbox("Show only errored repos", value=False)

    filtered = df[df["categories_found"] >= min_cats]
    if show_errors:
        filtered = filtered[filtered["error"].notna()]

    st.markdown(f"**Showing {len(filtered):,} of {len(df):,} repos**")
    st.dataframe(
        filtered[
            [
                "name",
                "categories_found",
                "categories",
                "processing_s",
                "somef_version",
                "error",
                "url",
            ]
        ].head(500),
        column_config={
            "name": st.column_config.TextColumn("Repository", width="medium"),
            "categories_found": st.column_config.NumberColumn("# Categories", width="small"),
            "categories": st.column_config.TextColumn("Categories Found", width="large"),
            "processing_s": st.column_config.NumberColumn("Time (s)", format="%.1f", width="small"),
            "somef_version": st.column_config.TextColumn("SOMEF Version", width="small"),
            "error": st.column_config.TextColumn("Error", width="medium"),
            "url": st.column_config.LinkColumn("Link", width="medium"),
        },
        hide_index=True,
        use_container_width=True,
        height=500,
    )

    st.markdown("---")
    st.download_button(
        label=f"Download {len(filtered)} SOMEF Results (CSV)",
        data=filtered.drop(columns=["run_date"], errors="ignore").to_csv(index=False),
        file_name=f"somef_results_{ts}.csv",
        mime="text/csv",
    )


def _render_mermaid(diagram: str, height: int = 500) -> None:
    """Render a Mermaid diagram using the Mermaid JS CDN."""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
      <script>mermaid.initialize({{startOnLoad: true, theme: 'default'}});</script>
      <style>
        body {{ margin: 0; background: transparent; }}
        .mermaid {{ width: 100%; overflow-x: auto; }}
      </style>
    </head>
    <body>
      <div class="mermaid">
{diagram}
      </div>
    </body>
    </html>
    """
    components.html(html, height=height, scrolling=True)


PIPELINE_DIAGRAM = """
flowchart TD
    subgraph Sources["📥 Input Sources"]
        SRC1[JOSS API]
        SRC2[Helmholtz RSD]
        SRC3[Research Software Directory]
        SRC4[Manual CSV]
    end

    Sources --> S0["Step 0 — step0_collect_repos.py
    Collect and deduplicate URLs
    adds source label per URL"]

    S0 --> CSV["output/code_repository_list.csv
    (url · source)"]

    CSV --> S2PRE["Step 2 Preflight — step2_preflight.py
    Validate config and DB connection"]

    S2PRE --> S2["Step 2 — step2_scrape_repos.py
    Scrape GitHub · GitLab · Bitbucket · Codeberg
    Creates analysis_run row"]

    S2 -->|scrape succeeded| REPOS[(repositories
    repository_history
    analysis_runs)]
    S2 -->|scrape failed| EXCL[(excluded_repositories)]

    REPOS --> S3["Step 3 — extract_all.py
    Parse READMEs: badges · DOIs · sections"]
    S3 --> META[(readme_metadata)]

    REPOS --> S4["Step 4 — step4_license_analysis.py
    SPDX license classification"]
    S4 --> S4B["Step 4b — step4b_recheck_licenses.py
    Re-check unknown / custom licenses"]
    S4B --> S4C["Step 4c — step4c_fetch_license_files.py
    Fetch raw LICENSE files · seed SPDX catalog"]
    S4C --> LIC[(repository_licenses
    license_files · licenses)]

    REPOS --> S5["Step 5 — step5_header_extraction.py
    Extract H1-H6 headers from READMEs"]
    S5 --> HDR[(readme_headers)]

    HDR --> S6["Step 6 — step6_clustering.py
    Sentence embeddings all-MiniLM-L6-v2
    + sklearn K-Means clustering"]
    S6 --> CLUST[(header_embeddings
    clusters
    header_cluster_assignments)]

    REPOS --> S7["Step 7 optional — step7_somef_validation.py
    SOMEF metadata extraction"]
    S7 --> SOMEF[(somef_results)]

    CLUST --> S11["Step 11 — step11_gap_analysis.py
    CodeMeta gap analysis
    keyword + alias matching"]
    S11 --> GAP[(cluster_codemeta_mappings
    unmapped_clusters)]

    META & LIC & CLUST & SOMEF & GAP --> S8["Step 8 — step8_dashboard.py
    Streamlit interactive dashboard"]

    CLUST -.->|future: Step 10| FUTURE10["cluster_reuse_scenarios"]
"""

ERD_DIAGRAM = """
erDiagram
    repositories {
        int     id              PK
        text    url
        text    name
        text    platform
        text    source
        text    license_from_api
        text    license_category
        datetime scraped_at
    }
    excluded_repositories {
        int     id              PK
        text    url
        text    exclusion_reason
        text    exclusion_stage
        text    source
        bool    is_retryable
    }
    readme_metadata {
        int     id              PK
        int     repository_id   FK
        text    readme_title
        int     badge_count
        bool    has_doi
        bool    has_citation_section
    }
    readme_headers {
        int     id              PK
        int     repository_id   FK
        text    header_text
        int     level
        text    normalized_text
    }
    header_embeddings {
        int     id              PK
        int     header_id       FK
        bytes   embedding_vector
        text    model_name
    }
    clusters {
        int     id              PK
        text    run_id          FK
        int     cluster_id
        text    cluster_name
        int     cluster_size
    }
    header_cluster_assignments {
        int     id              PK
        int     header_id       FK
        int     cluster_id      FK
        float   distance
    }
    cluster_codemeta_mappings {
        int     id              PK
        int     cluster_id      FK
        text    codemeta_property
        float   confidence
    }
    cluster_reuse_scenarios {
        int     id              PK
        int     cluster_id      FK
        text    scenario
        float   relevance_score
    }
    unmapped_clusters {
        int     id              PK
        int     cluster_id      FK
        text    proposed_property_name
        text    priority
    }
    licenses {
        int     id              PK
        text    spdx_id
        text    name
        text    category
    }
    repository_licenses {
        int     id              PK
        int     repository_id   FK
        int     license_id      FK
        text    source
    }
    license_files {
        int     id              PK
        int     repository_id   FK
        text    filename
        text    detected_license
    }
    analysis_runs {
        text    run_id          PK
        datetime run_date
        int     total_repos
        int     n_clusters
    }
    repository_history {
        int     id              PK
        int     repository_id   FK
        text    run_id          FK
        text    status
        text    readme_hash
    }
    somef_results {
        int     id              PK
        int     repository_id   FK
        text    categories_found
        float   processing_time_s
    }

    repositories ||--o{ readme_headers              : "has headers"
    repositories ||--o|  readme_metadata             : "has metadata"
    repositories ||--o{ repository_licenses          : "has licenses"
    repositories ||--o{ license_files                : "has license files"
    repositories ||--o{ repository_history           : "has history"
    repositories ||--o|  somef_results               : "has SOMEF result"
    readme_headers  ||--o|  header_embeddings         : "has embedding"
    readme_headers  ||--o|  header_cluster_assignments : "assigned to cluster"
    header_cluster_assignments }o--||  clusters           : "belongs to"
    clusters        ||--o{ cluster_codemeta_mappings  : "mapped to"
    clusters        ||--o{ cluster_reuse_scenarios    : "tagged with"
    clusters        ||--o{ unmapped_clusters          : "flagged as gap"
    clusters        }o--||  analysis_runs              : "created in run"
    repository_licenses }o--||  licenses               : "references"
    repository_history  }o--||  analysis_runs           : "recorded in run"
"""


def show_gap_analysis():
    """CodeMeta Gap Analysis page — mapped clusters and identified gaps."""

    mapped_df, unmapped_df = load_gap_analysis_data()

    if mapped_df.empty and unmapped_df.empty:
        st.info("No gap analysis data yet. Run **Step 11 — CodeMeta Gap Analysis** workflow first.")
        return

    total_clusters = len(mapped_df) + len(unmapped_df)
    n_mapped = len(mapped_df)
    n_unmapped = len(unmapped_df)
    pct_mapped = n_mapped / total_clusters * 100 if total_clusters else 0

    # ── Summary metrics ───────────────────────────────────────────────────
    st.subheader("📊 Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Clusters", total_clusters)
    with col2:
        st.metric("Mapped to CodeMeta", n_mapped, delta=f"{pct_mapped:.0f}% of clusters")
    with col3:
        st.metric(
            "Unmapped Gaps",
            n_unmapped,
            delta=f"{100 - pct_mapped:.0f}% of clusters",
            delta_color="inverse",
        )
    with col4:
        high_pri = (
            len(unmapped_df[unmapped_df["priority"] == "high"]) if not unmapped_df.empty else 0
        )
        st.metric("High-Priority Gaps", high_pri)

    st.markdown("---")

    # ── Section 1: CodeMeta property coverage ────────────────────────────
    st.subheader("📌 CodeMeta Property Coverage")
    st.caption(
        "Each bar shows the total number of README headers (cluster_size) "
        "covered by that CodeMeta property across all mapped clusters."
    )

    if not mapped_df.empty:
        coverage = (
            mapped_df.groupby("codemeta_property")["cluster_size"]
            .sum()
            .reset_index()
            .rename(columns={"cluster_size": "headers_covered"})
            .sort_values("headers_covered", ascending=True)
        )
        fig = px.bar(
            coverage,
            x="headers_covered",
            y="codemeta_property",
            orientation="h",
            color="headers_covered",
            color_continuous_scale="Blues",
            labels={
                "headers_covered": "README Headers Covered",
                "codemeta_property": "CodeMeta Property",
            },
            height=max(300, len(coverage) * 35),
        )
        fig.update_layout(
            coloraxis_showscale=False,
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Section 2 & 3 in two columns ─────────────────────────────────────
    left, right = st.columns([3, 2])

    with left:
        st.subheader("✅ Mapped Clusters")
        st.caption("Clusters classified to an existing CodeMeta property.")
        if not mapped_df.empty:
            display_mapped = mapped_df.copy()
            display_mapped["confidence"] = display_mapped["confidence"].map(lambda x: f"{x:.0%}")
            st.dataframe(
                display_mapped.rename(
                    columns={
                        "cluster_name": "Cluster",
                        "cluster_size": "Headers",
                        "codemeta_property": "CodeMeta Property",
                        "confidence": "Confidence",
                        "mapping_method": "Method",
                    }
                )[["Cluster", "Headers", "CodeMeta Property", "Confidence", "Method"]],
                use_container_width=True,
                hide_index=True,
            )

    with right:
        st.subheader("🔍 Gap Clusters (Unmapped)")
        st.caption("No existing CodeMeta property covers these concepts.")
        if not unmapped_df.empty:
            PRIORITY_COLOUR = {"high": "🔴", "medium": "🟡", "low": "🟢"}
            display_unmapped = unmapped_df.copy()
            display_unmapped["priority"] = display_unmapped["priority"].map(
                lambda p: f"{PRIORITY_COLOUR.get(p, '')} {p}"
            )
            st.dataframe(
                display_unmapped.rename(
                    columns={
                        "cluster_name": "Cluster",
                        "cluster_size": "Headers",
                        "proposed_property_name": "Proposed Property",
                        "priority": "Priority",
                    }
                )[["Cluster", "Headers", "Proposed Property", "Priority"]],
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("---")

    # ── Section 3: Gap treemap ────────────────────────────────────────────
    if not unmapped_df.empty:
        st.subheader("🗺️ Gap Map — Size of Undocumented Concepts")
        st.caption(
            "Each tile is an unmapped cluster. Size = number of README headers "
            "in that cluster. Larger tiles = more researchers document this concept "
            "but CodeMeta has no property for it."
        )
        treemap_df = unmapped_df.copy()
        treemap_df["label"] = treemap_df.apply(
            lambda r: f"{r['cluster_name']}<br>{r['cluster_size']:,} headers", axis=1
        )
        PRIORITY_ORDER = {"high": 1, "medium": 2, "low": 3}
        treemap_df["priority_order"] = treemap_df["priority"].map(PRIORITY_ORDER)

        fig2 = px.treemap(
            treemap_df,
            path=["priority", "cluster_name"],
            values="cluster_size",
            color="priority",
            color_discrete_map={"high": "#e74c3c", "medium": "#f39c12", "low": "#27ae60"},
            custom_data=["proposed_property_name", "cluster_size"],
            height=420,
        )
        fig2.update_traces(
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Headers: %{customdata[1]:,}<br>"
                "Proposed property: <i>%{customdata[0]}</i>"
                "<extra></extra>"
            )
        )
        fig2.update_layout(margin={"l": 10, "r": 10, "t": 10, "b": 10})
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ── Download ──────────────────────────────────────────────────────────
    st.subheader("📥 Export")
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    dl1, dl2 = st.columns(2)
    with dl1:
        if not mapped_df.empty:
            st.download_button(
                "Download Mapped Clusters (CSV)",
                data=mapped_df.to_csv(index=False),
                file_name=f"codemeta_mapped_{ts}.csv",
                mime="text/csv",
            )
    with dl2:
        if not unmapped_df.empty:
            st.download_button(
                "Download Gap Clusters (CSV)",
                data=unmapped_df.drop(columns=["justification"]).to_csv(index=False),
                file_name=f"codemeta_gaps_{ts}.csv",
                mime="text/csv",
            )


def show_architecture():
    """Architecture diagrams — pipeline flow and database ERD."""
    st.markdown(
        "Visual overview of the pipeline steps and database schema. "
        "Both diagrams are kept in sync with the codebase."
    )

    st.markdown("---")
    st.subheader("🔄 Pipeline Flow")
    st.caption(
        "End-to-end flow from input sources through all pipeline steps to the dashboard. "
        "Dashed arrow indicates the planned Step 10 (reuse scenario tagging)."
    )
    _render_mermaid(PIPELINE_DIAGRAM, height=780)

    st.markdown("---")
    st.subheader("🗄️ Database Schema (ERD)")
    st.caption(
        "All 15 database tables with key columns and foreign-key relationships. "
        "cluster_codemeta_mappings and unmapped_clusters are populated by Step 11 "
        "(Gap Analysis). cluster_reuse_scenarios is schema-ready for future use."
    )
    _render_mermaid(ERD_DIAGRAM, height=900)


if __name__ == "__main__":
    main()

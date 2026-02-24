#!/usr/bin/env python3
"""
Research Software Metadata Analyser
Interactive visualization and exploration of research software metadata
"""
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import json
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from contextlib import contextmanager
from database import get_session
from database.models import (
    Repository, ReadmeHeader, HeaderEmbedding,
    Cluster, HeaderClusterAssignment, ExcludedRepository, SomefResult,
    UnsupportedRepository,
)
from sqlalchemy import func, text

import os


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
if 'DATABASE_URL' in st.secrets:
    os.environ['DATABASE_URL'] = st.secrets['DATABASE_URL']
    
# Page config
st.set_page_config(
    page_title="Research Software Metadata Analyser",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
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
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def load_overview_stats():
    """Load overview statistics"""
    with get_session_context() as session:
        stats = {
            'total_repos': session.query(Repository).count(),
            'total_headers': session.query(ReadmeHeader).count(),
            'total_embeddings': session.query(HeaderEmbedding).count(),
            'total_clusters': session.query(Cluster).count(),
            'total_assignments': session.query(HeaderClusterAssignment).count(),
        }

        latest_run = session.query(Cluster.run_id).order_by(
            Cluster.created_at.desc()
        ).first()
        if latest_run:
            stats['run_id'] = latest_run[0]

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
            cluster_data.append({
                'id': c.id,
                'cluster_id': c.cluster_id,
                'name': c.cluster_name,
                'size': c.cluster_size,
                'representative_headers': rep_headers,
                'sample': rep_headers[0] if rep_headers else "N/A"
            })
    return pd.DataFrame(cluster_data)


@st.cache_data(ttl=300)
def load_license_data():
    """Load per-repo license information from the repositories table."""
    with get_session_context() as session:
        repos = session.query(Repository).all()
        rows = []
        for repo in repos:
            lic = getattr(repo, 'license_from_api', None)
            rows.append({
                'name': repo.name,
                'url': repo.url,
                'platform': repo.platform or '',
                'source': repo.source or '',
                'license': lic if lic and lic != 'NOASSERTION' else 'No License',
            })
    return pd.DataFrame(rows)


def categorize_license(license_name):
    """Map a raw SPDX id to a broad category string."""
    if not license_name or license_name == 'No License':
        return 'No License'
    permissive    = ['MIT', 'Apache-2.0', 'BSD-2-Clause', 'BSD-3-Clause', 'ISC', 'Artistic',
                     'Zlib', 'Python-2.0', 'PSF', 'WTFPL', '0BSD', 'CC0', 'Unlicense', 'BSL-1.0',
                     'CC-BY-4.0', 'CC-BY-3.0']
    weak_copyleft = ['LGPL', 'MPL-2.0', 'CDDL', 'EPL', 'EUPL', 'CC-BY-SA']
    strong_copyleft = ['GPL-2.0', 'GPL-3.0', 'AGPL', 'CPAL', 'OSL']
    u = license_name.upper()
    for p in strong_copyleft:
        if p.upper() in u:
            return 'Strong Copyleft'
    for p in weak_copyleft:
        if p.upper() in u:
            return 'Weak Copyleft'
    for p in permissive:
        if p.upper() in u:
            return 'Permissive'
    return 'Other'


@st.cache_data(ttl=300)
def load_repository_details():
    """Load full repository rows for the browser page."""
    with get_session_context() as session:
        repos = session.query(Repository).all()
        rows = []
        for repo in repos:
            lic = getattr(repo, 'license_from_api', None)
            desc = getattr(repo, 'description', '') or ''
            rows.append({
                'name': repo.name,
                'url': repo.url,
                'license': lic if lic and lic != 'NOASSERTION' else 'No License',
                'stars': getattr(repo, 'stars', 0) or 0,
                'language': getattr(repo, 'language', 'Unknown') or 'Unknown',
                'description': desc[:100] + '...' if len(desc) > 100 else desc,
                'source': getattr(repo, 'source', None) or 'unknown',
            })
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def load_excluded_repos():
    """Load excluded repositories for the browser page."""
    with get_session_context() as session:
        rows_q = session.query(
            ExcludedRepository.url,
            ExcludedRepository.exclusion_reason,
            ExcludedRepository.exclusion_stage,
            ExcludedRepository.source,
            ExcludedRepository.is_retryable,
            ExcludedRepository.excluded_at,
        ).order_by(ExcludedRepository.excluded_at.desc()).all()
        rows = [
            {
                'url': r.url,
                'exclusion_reason': r.exclusion_reason,
                'exclusion_stage': r.exclusion_stage,
                'source': r.source or 'unknown',
                'retryable': r.is_retryable,
                'excluded_at': r.excluded_at,
            }
            for r in rows_q
        ]
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def load_unsupported_repos():
    """Load unsupported-platform repositories for the browser page."""
    try:
        with get_session_context() as session:
            rows_q = session.query(
                UnsupportedRepository.url,
                UnsupportedRepository.source,
                UnsupportedRepository.host,
                UnsupportedRepository.platform,
                UnsupportedRepository.occurrence_count,
                UnsupportedRepository.first_seen_at,
                UnsupportedRepository.last_seen_at,
            ).order_by(
                UnsupportedRepository.occurrence_count.desc(),
                UnsupportedRepository.host,
            ).all()
            rows = [
                {
                    "url":       r.url,
                    "source":    r.source,
                    "host":      r.host,
                    "platform":  r.platform,
                    "seen":      r.occurrence_count,
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
        "description", "installation", "invocation", "citation",
        "requirement", "documentation", "contributor", "license",
        "usage", "acknowledgement", "run", "support",
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
            rows.append({
                "name":             name,
                "url":              url,
                "categories_found": len(cats),
                "categories":       ", ".join(cats) if cats else "",
                "processing_s":     round(proc_time, 1) if proc_time else None,
                "somef_version":    version,
                "error":            error,
                "run_date":         run_date,
            })

    df = pd.DataFrame(rows)
    avg_cats = df["categories_found"].mean() if not df.empty else 0
    avg_time = df["processing_s"].mean() if not df.empty else 0

    return {
        "total":      total,
        "errors":     errors,
        "avg_cats":   avg_cats,
        "avg_time":   avg_time,
        "cat_counts": cat_counts,
        "categories": CATEGORIES,
        "df":         df,
    }


@st.cache_data(ttl=300)
def load_embeddings_for_viz(max_samples=5000):
    """Load embeddings for visualization (sample if too large)"""
    with get_session_context() as session:
        total = session.query(HeaderEmbedding).count()

        if total > max_samples:
            query = session.query(
                HeaderEmbedding.header_id,
                HeaderEmbedding.embedding_vector,
                HeaderEmbedding.embedding_dim
            ).order_by(func.random()).limit(max_samples)
        else:
            query = session.query(
                HeaderEmbedding.header_id,
                HeaderEmbedding.embedding_vector,
                HeaderEmbedding.embedding_dim
            )

        data = query.all()

        header_ids = []
        embeddings = []
        for header_id, emb_bytes, dim in data:
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


@st.cache_data
def compute_umap(embeddings, n_components=2):
    """Compute UMAP projection"""
    from umap import UMAP
    
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    
    return reducer.fit_transform(embeddings)


def main():
    """Main dashboard"""
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Repository Browser", "License Analysis", "Data Quality",
         "Search", "Cluster Explorer", "SOMEF Validation", "Visualization", "Export",
         "Architecture"]
    )

    # Page title — dynamic per page
    PAGE_TITLES = {
        "Overview":            "📊 Research Software Metadata Analyser",
        "Cluster Explorer":    "🔍 Cluster Explorer",
        "License Analysis":    "📄 License Analysis",
        "Repository Browser":  "📚 Repository Browser",
        "Data Quality":        "📋 Data Quality Report",
        "Visualization":       "📊 Embedding Visualization",
        "Search":              "🔎 Header Search",
        "SOMEF Validation":    "🔬 SOMEF Metadata Validation",
        "Export":              "📥 Export Results",
        "Architecture":        "🏗️ Architecture",
    }
    st.markdown(
        f'<div class="main-header">{PAGE_TITLES.get(page, "Research Software Metadata Analyser")}</div>',
        unsafe_allow_html=True,
    )
    
    # Load data
    stats = load_overview_stats()
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("### Dataset Statistics")
    st.sidebar.metric("Repositories", f"{stats['total_repos']:,}")
    st.sidebar.metric("Headers", f"{stats['total_headers']:,}")
    st.sidebar.metric("Clusters", stats['total_clusters'])
    
    if 'run_id' in stats:
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
    if page == "Overview":
        show_overview(stats)
    elif page == "Cluster Explorer":
        show_cluster_explorer(stats.get('run_id'))
    elif page == "License Analysis":
        show_license_analysis()
    elif page == "Repository Browser":
        show_repository_browser()
    elif page == "SOMEF Validation":
        show_somef_validation()
    elif page == "Data Quality":
        show_data_quality()
    elif page == "Visualization":
        show_visualization()
    elif page == "Search":
        show_search()
    elif page == "Export":
        show_export(stats.get('run_id'))
    elif page == "Architecture":
        show_architecture()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><strong>Research Software Metadata Analyser</strong></p>
        <p>Analyzing documentation structure patterns in research software</p>
        <p>© 2026 | MIT License | <a href='https://github.com/priya-gitTest/readme-clustering-dashboard'>View on GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)


def show_overview(stats):
    """Overview page"""

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Repositories Analyzed",
            value=f"{stats['total_repos']:,}",
            delta="100% coverage"
        )
    
    with col2:
        st.metric(
            label="Headers Extracted",
            value=f"{stats['total_headers']:,}",
            delta=f"{stats['total_headers']/stats['total_repos']:.1f} per repo" if stats['total_repos'] > 0 else "0.0 per repo"
        )
    
    with col3:
        st.metric(
            label="Clusters Discovered",
            value=stats['total_clusters'],
            delta="K-Means"
        )
    
    with col4:
        coverage = (stats['total_assignments'] / stats['total_headers'] * 100) if stats['total_headers'] > 0 else 0
        st.metric(
            label="Clustering Coverage",
            value=f"{coverage:.1f}%",
            delta=f"{stats['total_assignments']:,} assigned"
        )
    
    st.markdown("---")
    
    # Key findings
    st.subheader("🔍 Key Findings")
    
    cluster_df = load_cluster_data(stats.get('run_id'))

    if cluster_df.empty:
        st.info("No clustering data yet — the pipeline hasn't completed a full run. Check back after the next scheduled scrape.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Top 10 Largest Clusters")
        top_10 = cluster_df.head(10)

        fig = px.bar(
            top_10,
            x='size',
            y='name',
            orientation='h',
            title="Cluster Sizes",
            labels={'size': 'Number of Headers', 'name': 'Cluster'}
        )
        fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Cluster Size Distribution")

        fig = px.histogram(
            cluster_df,
            x='size',
            nbins=20,
            title="Distribution of Cluster Sizes",
            labels={'size': 'Cluster Size', 'count': 'Frequency'}
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


def show_cluster_explorer(run_id):
    """Cluster explorer page"""

    cluster_df = load_cluster_data(run_id)

    if cluster_df.empty:
        st.info("No cluster data yet — run the full pipeline first.")
        return

    # Filters
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_term = st.text_input("🔎 Search clusters by name", "")
    
    with col2:
        min_size = st.number_input("Min cluster size", min_value=0, value=0)
    
    # Filter
    if search_term:
        cluster_df = cluster_df[
            cluster_df['name'].str.contains(search_term, case=False, na=False)
        ]
    
    if min_size > 0:
        cluster_df = cluster_df[cluster_df['size'] >= min_size]
    
    st.markdown(f"**Showing {len(cluster_df)} clusters**")
    
    # Display clusters
    for _, row in cluster_df.iterrows():
        with st.expander(f"**{row['name']}** ({row['size']} headers)"):
            st.markdown(f"**Cluster ID:** {row['cluster_id']}")
            st.markdown(f"**Size:** {row['size']} headers")
            
            st.markdown("**Representative Headers:**")
            for i, header in enumerate(row['representative_headers'][:10], 1):
                st.markdown(f"{i}. `{header}`")
            
            if len(row['representative_headers']) > 10:
                st.markdown(f"*... and {len(row['representative_headers']) - 10} more*")


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
            df = pd.DataFrame({
                'header_id': header_ids,
                'x': projection[:, 0],
                'y': projection[:, 1],
                'cluster': [assignments.get(hid, "Unassigned") for hid in header_ids]
            })
            
            fig = px.scatter(
                df,
                x='x',
                y='y',
                color='cluster',
                title=f"{dimensions} UMAP Projection of Header Embeddings",
                labels={'x': 'UMAP 1', 'y': 'UMAP 2'},
                hover_data=['header_id']
            )
            fig.update_traces(marker=dict(size=5, opacity=0.7))
            
        else:  # 3D
            df = pd.DataFrame({
                'header_id': header_ids,
                'x': projection[:, 0],
                'y': projection[:, 1],
                'z': projection[:, 2],
                'cluster': [assignments.get(hid, "Unassigned") for hid in header_ids]
            })
            
            fig = px.scatter_3d(
                df,
                x='x',
                y='y',
                z='z',
                color='cluster',
                title=f"{dimensions} UMAP Projection of Header Embeddings",
                labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'z': 'UMAP 3'},
                hover_data=['header_id']
            )
            fig.update_traces(marker=dict(size=3, opacity=0.6))
        
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)
        
        st.success(f"✅ Visualized {len(df):,} headers across {df['cluster'].nunique()} clusters")


def show_search():
    """Search page"""

    search_query = st.text_input("Search for headers", "")

    if search_query:
        # Collect all data within the session context, then render UI outside
        display_results = []
        with get_session_context() as session:
            results = session.query(ReadmeHeader).filter(
                ReadmeHeader.normalized_text.contains(search_query.lower())
            ).limit(100).all()

            result_ids = [r.id for r in results]
            cluster_map: dict[int, str] = {}
            if result_ids:
                rows = (
                    session.query(HeaderClusterAssignment.header_id, Cluster.cluster_name)
                    .join(Cluster, HeaderClusterAssignment.cluster_id == Cluster.id)
                    .filter(HeaderClusterAssignment.header_id.in_(result_ids))
                    .all()
                )
                cluster_map = {r.header_id: r.cluster_name for r in rows}

            # Convert to plain dicts before session closes
            for result in results:
                display_results.append({
                    "id":            result.id,
                    "header_text":   result.header_text,
                    "repository_id": result.repository_id,
                    "level":         result.level,
                    "position":      result.position,
                    "cluster":       cluster_map.get(result.id, "Unassigned"),
                })

        st.markdown(f"**Found {len(display_results)} results** (showing top 100)")
        for r in display_results:
            with st.expander(f"`{r['header_text']}` → **{r['cluster']}**"):
                st.markdown(f"**Header ID:** {r['id']}")
                st.markdown(f"**Repository ID:** {r['repository_id']}")
                st.markdown(f"**Level:** H{r['level']}")
                st.markdown(f"**Position:** {r['position']}")
                st.markdown(f"**Cluster:** {r['cluster']}")


def show_export(run_id):
    """Export page"""

    ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
    cluster_df = load_cluster_data(run_id)

    if cluster_df.empty:
        st.info("No data to export yet — run the full pipeline first.")
        return

    st.markdown("### Download Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Cluster Summary (CSV)")
        _dl = cluster_df[['cluster_id', 'name', 'size', 'sample']].copy()
        if run_id:
            _dl['run_id'] = run_id
        st.download_button(
            label="Download CSV",
            data=_dl.to_csv(index=False),
            file_name=f"cluster_summary_{ts}.csv",
            mime="text/csv"
        )

    with col2:
        st.markdown("#### Full Report (JSON)")
        json_data = cluster_df.to_json(orient='records', indent=2)
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name=f"cluster_report_{ts}.json",
            mime="application/json"
        )
    
    st.markdown("---")
    
    st.markdown("### Research Summary")
    
    stats = load_overview_stats()
    
    summary = f"""
## Research Software Metadata Analyser

**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}
**Run ID:** {run_id}

### Dataset
- **Repositories Analyzed:** {stats['total_repos']:,}
- **Headers Extracted:** {stats['total_headers']:,}
- **Average Headers per Repository:** {(stats['total_headers']/stats['total_repos'] if stats['total_repos'] > 0 else 0.0):.1f}

### Clustering Results
- **Number of Clusters:** {stats['total_clusters']}
- **Headers Clustered:** {stats['total_assignments']:,} ({(stats['total_assignments']/stats['total_headers']*100 if stats['total_headers'] > 0 else 0.0):.1f}%)
- **Average Cluster Size:** {cluster_df['size'].mean():.0f}
- **Median Cluster Size:** {cluster_df['size'].median():.0f}

### Top 10 Clusters
{cluster_df[['name', 'size']].head(10).to_markdown(index=False)}

### Methodology
- **Embedding Model:** all-MiniLM-L6-v2 (384 dimensions)
- **Clustering Algorithm:** K-Means (k=30)
- **Distance Metric:** Euclidean distance on normalized embeddings
"""
    
    st.markdown(summary)
    
    st.download_button(
        label="Download Summary (Markdown)",
        data=summary,
        file_name=f"research_summary_{ts}.md",
        mime="text/markdown"
    )


def show_license_analysis():
    """License analysis page"""

    ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
    license_df = load_license_data()

    if license_df.empty:
        st.info("No repository data yet — run the pipeline first.")
        return

    total_repos = len(license_df)
    with_license = len(license_df[license_df['license'] != 'No License'])
    without_license = total_repos - with_license

    st.subheader("📊 Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Repositories", f"{total_repos:,}")
    with col2:
        st.metric("With License", f"{with_license:,}",
                  delta=f"{with_license/total_repos*100:.1f}%" if total_repos else "0%")
    with col3:
        st.metric("Without License", f"{without_license:,}",
                  delta=f"-{without_license/total_repos*100:.1f}%" if total_repos else "0%",
                  delta_color="inverse")
    with col4:
        unique_licenses = license_df[license_df['license'] != 'No License']['license'].nunique()
        st.metric("Unique Licenses", unique_licenses)

    # ── Drilldown: repos without a license ────────────────────────────────
    if without_license > 0:
        with st.expander(f"View {without_license} repositories without a license"):
            no_lic_df = license_df[license_df['license'] == 'No License'][
                ['name', 'url', 'platform', 'source']
            ].copy().reset_index(drop=True)

            # Platform filter
            platforms = sorted(no_lic_df['platform'].unique().tolist())
            if len(platforms) > 1:
                sel = st.multiselect("Filter by platform", platforms, default=platforms,
                                     key="no_lic_platform_filter")
                no_lic_df = no_lic_df[no_lic_df['platform'].isin(sel)]

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
                "Download list (CSV)", data=no_lic_df.to_csv(index=False),
                file_name=f"repos_without_license_{ts}.csv", mime="text/csv",
                key="no_lic_csv_dl",
            )

    st.markdown("---")
    st.subheader("🏆 Top 10 Licenses")

    license_counts = license_df['license'].value_counts().head(10)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            x=license_counts.values, y=license_counts.index, orientation='h',
            title="Most Common Licenses",
            labels={'x': 'Number of Repositories', 'y': 'License'},
            color=license_counts.values, color_continuous_scale='Blues'
        )
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.pie(
            values=license_counts.values, names=license_counts.index,
            title="License Distribution", hole=0.3
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📂 License Categories")

    license_df['category'] = license_df['license'].apply(categorize_license)
    category_counts = license_df['category'].value_counts()

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            x=category_counts.index, y=category_counts.values,
            title="Licenses by Category",
            labels={'x': 'Category', 'y': 'Count'},
            color=category_counts.values, color_continuous_scale='Viridis'
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
        label="Download License Report (CSV)", data=license_df.to_csv(index=False),
        file_name=f"license_analysis_{ts}.csv", mime="text/csv"
    )


def show_repository_browser():
    """Repository browser page"""

    ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
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
        st.metric("Languages", repo_df['language'].nunique())
    with col4:
        st.metric("Total Stars", f"{repo_df['stars'].sum():,}")

    st.markdown("---")
    st.subheader("🔍 Filters")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        selected_license = st.selectbox(
            "License", ['All'] + sorted(repo_df['license'].unique().tolist()))
    with col2:
        selected_language = st.selectbox(
            "Language", ['All'] + sorted(repo_df['language'].unique().tolist()))
    with col3:
        source_options = ['All'] + sorted(repo_df['source'].unique().tolist())
        selected_source = st.selectbox("Source", source_options)
    with col4:
        search_term = st.text_input("Search by name", "")

    filtered = repo_df.copy()
    if selected_license != 'All':
        filtered = filtered[filtered['license'] == selected_license]
    if selected_language != 'All':
        filtered = filtered[filtered['language'] == selected_language]
    if selected_source != 'All':
        filtered = filtered[filtered['source'] == selected_source]
    if search_term:
        filtered = filtered[filtered['name'].str.contains(search_term, case=False, na=False)]

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**Showing {len(filtered):,} of {len(repo_df):,} repositories**")
    with col2:
        sort_by = st.selectbox("Sort by", ["Name", "Stars", "License", "Source"])

    if sort_by == "Stars":
        filtered = filtered.sort_values('stars', ascending=False)
    elif sort_by == "License":
        filtered = filtered.sort_values('license')
    elif sort_by == "Source":
        filtered = filtered.sort_values('source')
    else:
        filtered = filtered.sort_values('name')

    st.dataframe(
        filtered[['name', 'license', 'language', 'stars', 'source', 'url']].head(500),
        column_config={
            "name":     st.column_config.TextColumn("Repository", width="medium"),
            "license":  st.column_config.TextColumn("License",    width="medium"),
            "language": st.column_config.TextColumn("Language",   width="small"),
            "stars":    st.column_config.NumberColumn("⭐ Stars",  width="small"),
            "source":   st.column_config.TextColumn("Source",     width="small"),
            "url":      st.column_config.LinkColumn("Link",        width="medium"),
        },
        hide_index=True, use_container_width=True, height=600
    )
    if len(filtered) > 500:
        st.info("Showing first 500 results. Use filters to narrow down.")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Repository Sources**")
        filtered['domain'] = filtered['url'].apply(
            lambda x: urlparse(x).netloc if pd.notna(x) else 'Unknown')
        for domain, count in filtered['domain'].value_counts().head(5).items():
            st.write(f"- **{domain}**: {count} repos")
    with col2:
        st.markdown("**Top 5 Licenses**")
        for lic, count in filtered['license'].value_counts().head(5).items():
            st.write(f"- **{lic}**: {count} repos")

    st.markdown("---")
    st.download_button(
        label=f"Download {len(filtered)} Repositories (CSV)",
        data=filtered.drop(columns=['domain'], errors='ignore').to_csv(index=False),
        file_name=f"repository_list_{ts}.csv", mime="text/csv"
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
                    "Filter by source", ['All'] + sorted(excl_df['source'].unique().tolist()),
                    key="excl_source")
            with excl_col2:
                excl_stage = st.selectbox(
                    "Filter by stage", ['All'] + sorted(excl_df['exclusion_stage'].unique().tolist()),
                    key="excl_stage")
            with excl_col3:
                excl_search = st.text_input("Search URL", "", key="excl_search")

            excl_filtered = excl_df.copy()
            if excl_source != 'All':
                excl_filtered = excl_filtered[excl_filtered['source'] == excl_source]
            if excl_stage != 'All':
                excl_filtered = excl_filtered[excl_filtered['exclusion_stage'] == excl_stage]
            if excl_search:
                excl_filtered = excl_filtered[
                    excl_filtered['url'].str.contains(excl_search, case=False, na=False)]

            st.markdown(f"**Showing {len(excl_filtered):,} of {len(excl_df):,} excluded repositories**")
            st.dataframe(
                excl_filtered[['url', 'exclusion_reason', 'exclusion_stage', 'source', 'retryable', 'excluded_at']].head(500),
                column_config={
                    "url":               st.column_config.LinkColumn("URL",             width="large"),
                    "exclusion_reason":  st.column_config.TextColumn("Reason",          width="large"),
                    "exclusion_stage":   st.column_config.TextColumn("Stage",           width="small"),
                    "source":            st.column_config.TextColumn("Source",          width="small"),
                    "retryable":         st.column_config.CheckboxColumn("Retryable",   width="small"),
                    "excluded_at":       st.column_config.DatetimeColumn("Excluded At", width="medium"),
                },
                hide_index=True, use_container_width=True, height=400
            )
            st.download_button(
                label=f"Download {len(excl_filtered)} Excluded Repos (CSV)",
                data=excl_filtered.to_csv(index=False),
                file_name=f"excluded_repositories_{ts}.csv", mime="text/csv"
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
                unsup_df.groupby("host")["seen"].sum()
                .sort_values(ascending=False)
                .head(20)
                .reset_index()
                .rename(columns={"seen": "occurrences"})
            )
            fig = px.bar(
                host_counts, x="occurrences", y="host", orientation="h",
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
                mask = (
                    unsup_filtered["url"].str.contains(u_search, case=False, na=False)
                    | unsup_filtered["host"].str.contains(u_search, case=False, na=False)
                )
                unsup_filtered = unsup_filtered[mask]

            st.markdown(f"**Showing {len(unsup_filtered):,} of {len(unsup_df):,} records**")
            st.dataframe(
                unsup_filtered[["url", "source", "host", "platform", "seen", "first_seen", "last_seen"]].head(500),
                column_config={
                    "url":        st.column_config.LinkColumn("URL",          width="large"),
                    "source":     st.column_config.TextColumn("Source",       width="small"),
                    "host":       st.column_config.TextColumn("Host",         width="medium"),
                    "platform":   st.column_config.TextColumn("Platform",     width="small"),
                    "seen":       st.column_config.NumberColumn("# Seen",     width="small"),
                    "first_seen": st.column_config.DatetimeColumn("First Seen", width="medium"),
                    "last_seen":  st.column_config.DatetimeColumn("Last Seen",  width="medium"),
                },
                hide_index=True, use_container_width=True, height=400,
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

    ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
    with get_session_context() as session:
        total_repos      = session.query(Repository).count()
        total_readmes    = session.query(ReadmeHeader.repository_id).distinct().count()
        total_headers    = session.query(ReadmeHeader).count()
        total_embeddings = session.query(HeaderEmbedding).count()
        total_clustered  = session.query(HeaderClusterAssignment).count()
        total_clusters   = session.query(Cluster).count()
        total_excluded   = session.execute(
            text("SELECT COUNT(*) FROM excluded_repositories")
        ).scalar() or 0

    if total_repos == 0:
        st.info("No data yet — run the pipeline first.")
        return

    repos_without_readme = total_repos - total_readmes
    avg_headers = total_headers / total_readmes if total_readmes > 0 else 0
    ORIGINAL_SCRAPED = total_repos + total_excluded
    invalid_removed  = total_excluded
    retention        = total_repos / ORIGINAL_SCRAPED * 100 if ORIGINAL_SCRAPED else 100.0
    readme_coverage  = total_readmes / total_repos * 100 if total_repos else 0
    cluster_coverage = total_clustered / total_headers * 100 if total_headers else 0

    st.subheader("🔄 Processing Pipeline")

    pipeline_df = pd.DataFrame({
        'Stage': [
            '1. Initial Scraping', '2. Valid Repositories', '3. README Extracted',
            '4. Headers Extracted', '5. Embeddings Generated', '6. Clustered'
        ],
        'Count': [ORIGINAL_SCRAPED, total_repos, total_readmes,
                  total_headers, total_embeddings, total_clustered],
        'Lost': [0, invalid_removed, repos_without_readme,
                 0, total_headers - total_embeddings, total_embeddings - total_clustered],
        'Retention %': [
            100.0,
            round(total_repos / ORIGINAL_SCRAPED * 100, 1),
            round(total_readmes / total_repos * 100, 1) if total_repos else 0,
            100.0,
            round(total_embeddings / total_headers * 100, 1) if total_headers else 0,
            round(total_clustered / total_embeddings * 100, 1) if total_embeddings else 0,
        ],
    })

    st.dataframe(
        pipeline_df,
        column_config={
            "Stage":       st.column_config.TextColumn("Processing Stage", width="large"),
            "Count":       st.column_config.NumberColumn("Count",          format="%d"),
            "Lost":        st.column_config.NumberColumn("Lost",           format="%d"),
            "Retention %": st.column_config.NumberColumn("Retention",      format="%.1f%%"),
        },
        hide_index=True, use_container_width=True
    )

    st.markdown("---")
    st.subheader("📊 Data Flow")

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20,
            line=dict(color="black", width=0.5),
            label=[
                f"Initial ({ORIGINAL_SCRAPED:,})", f"Valid ({total_repos:,})",
                f"With README ({total_readmes:,})", f"Headers ({total_headers:,})",
                f"Embeddings ({total_embeddings:,})", f"Clustered ({total_clustered:,})"
            ],
            color=["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"]
        ),
        link=dict(
            source=[0, 1, 2, 3, 4],
            target=[1, 2, 3, 4, 5],
            value=[total_repos, total_readmes, total_headers, total_embeddings, total_clustered]
        )
    )])
    fig.update_layout(title="Repository Processing Flow", height=400, font_size=12)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Retention Rate",    f"{retention:.1f}%")
    with col2:
        st.metric("README Coverage",   f"{readme_coverage:.1f}%")
    with col3:
        st.metric("Cluster Coverage",  f"{cluster_coverage:.1f}%")
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
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

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
        label="Download Data Quality Report (Markdown)", data=report_md,
        file_name=f"data_quality_report_{ts}.md", mime="text/markdown"
    )


def show_somef_validation():
    """SOMEF metadata extraction results page."""
    ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
    data = load_somef_stats()

    if data["total"] == 0:
        st.info(
            "No SOMEF results yet. Run step 7 to populate this page:\n\n"
            "```bash\n"
            "pip install -e \".[somef]\"\n"
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

    cat_df = pd.DataFrame({
        "Category": [c.capitalize() for c in data["categories"]],
        "Repos":    [data["cat_counts"][c] for c in data["categories"]],
        "Coverage %": [
            round(data["cat_counts"][c] / data["total"] * 100, 1)
            for c in data["categories"]
        ],
    }).sort_values("Repos", ascending=False)

    fig = px.bar(
        cat_df, x="Category", y="Coverage %",
        text="Repos",
        color="Coverage %", color_continuous_scale="Blues",
        labels={"Coverage %": "Coverage (%)"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=420, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        cat_df,
        column_config={
            "Category":   st.column_config.TextColumn("Category",    width="medium"),
            "Repos":      st.column_config.NumberColumn("# Repos",   width="small"),
            "Coverage %": st.column_config.NumberColumn("Coverage %", format="%.1f%%", width="small"),
        },
        hide_index=True, use_container_width=True,
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
        filtered[["name", "categories_found", "categories", "processing_s", "somef_version", "error", "url"]].head(500),
        column_config={
            "name":             st.column_config.TextColumn("Repository",        width="medium"),
            "categories_found": st.column_config.NumberColumn("# Categories",    width="small"),
            "categories":       st.column_config.TextColumn("Categories Found",  width="large"),
            "processing_s":     st.column_config.NumberColumn("Time (s)",        format="%.1f", width="small"),
            "somef_version":    st.column_config.TextColumn("SOMEF Version",     width="small"),
            "error":            st.column_config.TextColumn("Error",             width="medium"),
            "url":              st.column_config.LinkColumn("Link",              width="medium"),
        },
        hide_index=True, use_container_width=True, height=500,
    )

    st.markdown("---")
    st.download_button(
        label=f"Download {len(filtered)} SOMEF Results (CSV)",
        data=filtered.drop(columns=["run_date"], errors="ignore").to_csv(index=False),
        file_name=f"somef_results_{ts}.csv", mime="text/csv",
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

    META & LIC & CLUST & SOMEF --> S8["Step 8 — step8_dashboard.py
    Streamlit interactive dashboard"]

    CLUST -.->|future Steps 9-11| FUTURE["cluster_codemeta_mappings
    cluster_reuse_scenarios
    unmapped_clusters"]
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
        "Dashed arrow indicates planned future steps."
    )
    _render_mermaid(PIPELINE_DIAGRAM, height=780)

    st.markdown("---")
    st.subheader("🗄️ Database Schema (ERD)")
    st.caption(
        "All 15 database tables with key columns and foreign-key relationships. "
        "Tables shown with dashed borders (cluster_codemeta_mappings, "
        "cluster_reuse_scenarios, unmapped_clusters) are schema-ready but not yet populated."
    )
    _render_mermaid(ERD_DIAGRAM, height=900)


if __name__ == "__main__":
    main()

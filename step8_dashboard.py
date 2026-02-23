#!/usr/bin/env python3
"""
Research Dashboard: README Header Clustering Analysis
Interactive visualization and exploration of clustering results
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import json
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import get_session
from database.models import (
    Repository, ReadmeHeader, HeaderEmbedding, 
    Cluster, HeaderClusterAssignment
)
from sqlalchemy import func, text

import os

# Get database URL from Streamlit secrets or environment
if 'DATABASE_URL' in st.secrets:
    os.environ['DATABASE_URL'] = st.secrets['DATABASE_URL']
    
# Page config
st.set_page_config(
    page_title="README Clustering Analysis",
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


@st.cache_resource
def get_db_session():
    """Get database session (cached)"""
    return get_session()


@st.cache_data
def load_overview_stats():
    """Load overview statistics"""
    session = get_db_session()
    
    stats = {
        'total_repos': session.query(Repository).count(),
        'total_headers': session.query(ReadmeHeader).count(),
        'total_embeddings': session.query(HeaderEmbedding).count(),
        'total_clusters': session.query(Cluster).count(),
        'total_assignments': session.query(HeaderClusterAssignment).count(),
    }
    
    # Get latest run
    latest_run = session.query(Cluster.run_id).order_by(
        Cluster.created_at.desc()
    ).first()
    
    if latest_run:
        stats['run_id'] = latest_run[0]
    
    return stats


@st.cache_data
def load_cluster_data(run_id=None):
    """Load cluster information"""
    session = get_db_session()
    
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


@st.cache_data
def load_license_data():
    """Load per-repo license information from the repositories table."""
    session = get_db_session()
    repos = session.query(Repository).all()
    rows = []
    for repo in repos:
        lic = getattr(repo, 'license_from_api', None)
        rows.append({
            'name': repo.name,
            'url': repo.url,
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


@st.cache_data
def load_repository_details():
    """Load full repository rows for the browser page."""
    session = get_db_session()
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
        })
    return pd.DataFrame(rows)


@st.cache_data
def load_embeddings_for_viz(max_samples=5000):
    """Load embeddings for visualization (sample if too large)"""
    session = get_db_session()
    
    # Get total count
    total = session.query(HeaderEmbedding).count()
    
    # Sample if needed
    if total > max_samples:
        # Random sampling
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
    
    # Convert to arrays
    header_ids = []
    embeddings = []
    
    for header_id, emb_bytes, dim in data:
        header_ids.append(header_id)
        emb_array = np.frombuffer(emb_bytes, dtype=np.float32)
        embeddings.append(emb_array)
    
    embeddings = np.array(embeddings)
    
    # Get cluster assignments
    assignments = {}
    for header_id in header_ids:
        assignment = session.query(HeaderClusterAssignment).filter_by(
            header_id=header_id
        ).first()
        
        if assignment:
            cluster = session.query(Cluster).get(assignment.cluster_id)
            assignments[header_id] = cluster.cluster_name if cluster else "Unassigned"
        else:
            assignments[header_id] = "Unassigned"
    
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
    
    # Header
    st.markdown('<div class="main-header">📊 README Header Clustering Analysis</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Cluster Explorer", "License Analysis", "Repository Browser", "Data Quality", "Visualization", "Search", "Export"]
    )
    
    # Load data
    stats = load_overview_stats()
    
    st.sidebar.markdown("---")
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
    [Your Name] (2026)
    README Header Clustering Analysis
    AI-Assisted Code Metadata Pipeline
    ```
    
    **Code:** [GitHub Repository](https://github.com/your-username/your-repo)
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
    elif page == "Data Quality":
        show_data_quality()
    elif page == "Visualization":
        show_visualization()
    elif page == "Search":
        show_search()
    elif page == "Export":
        show_export(stats.get('run_id'))
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><strong>README Header Clustering Analysis Dashboard</strong></p>
        <p>Analyzing documentation structure patterns in research software</p>
        <p>© 2026 | MIT License | <a href='https://github.com/your-username/your-repo'>View on GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)


def show_overview(stats):
    """Overview page"""
    st.header("📈 Overview")
    
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
    st.header("🔍 Cluster Explorer")

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
    for idx, row in cluster_df.iterrows():
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
    st.header("📊 Embedding Visualization")
    
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
    st.header("🔎 Search Headers")
    
    session = get_db_session()
    
    search_query = st.text_input("Search for headers", "")
    
    if search_query:
        # Search headers
        results = session.query(ReadmeHeader).filter(
            ReadmeHeader.normalized_text.contains(search_query.lower())
        ).limit(100).all()
        
        st.markdown(f"**Found {len(results)} results** (showing top 100)")
        
        for result in results:
            # Get cluster assignment
            assignment = session.query(HeaderClusterAssignment).filter_by(
                header_id=result.id
            ).first()
            
            cluster_name = "Unassigned"
            if assignment:
                cluster = session.query(Cluster).get(assignment.cluster_id)
                cluster_name = cluster.cluster_name if cluster else "Unknown"
            
            with st.expander(f"`{result.header_text}` → **{cluster_name}**"):
                st.markdown(f"**Header ID:** {result.id}")
                st.markdown(f"**Repository ID:** {result.repository_id}")
                st.markdown(f"**Level:** H{result.level}")
                st.markdown(f"**Position:** {result.position}")
                st.markdown(f"**Cluster:** {cluster_name}")


def show_export(run_id):
    """Export page"""
    st.header("📥 Export Results")

    cluster_df = load_cluster_data(run_id)

    if cluster_df.empty:
        st.info("No data to export yet — run the full pipeline first.")
        return

    st.markdown("### Download Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Cluster Summary (CSV)")
        csv = cluster_df[['cluster_id', 'name', 'size', 'sample']].to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="cluster_summary.csv",
            mime="text/csv"
        )
    
    with col2:
        st.markdown("#### Full Report (JSON)")
        json_data = cluster_df.to_json(orient='records', indent=2)
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name="cluster_report.json",
            mime="application/json"
        )
    
    st.markdown("---")
    
    st.markdown("### Research Summary")
    
    stats = load_overview_stats()
    
    summary = f"""
## README Header Clustering Analysis

**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}
**Run ID:** {run_id}

### Dataset
- **Repositories Analyzed:** {stats['total_repos']:,}
- **Headers Extracted:** {stats['total_headers']:,}
- **Average Headers per Repository:** {stats['total_headers']/stats['total_repos']:.1f if stats['total_repos'] > 0 else 0.0}

### Clustering Results
- **Number of Clusters:** {stats['total_clusters']}
- **Headers Clustered:** {stats['total_assignments']:,} ({stats['total_assignments']/stats['total_headers']*100:.1f if stats['total_headers'] > 0 else 0.0}%)
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
        file_name="research_summary.md",
        mime="text/markdown"
    )


def show_license_analysis():
    """License analysis page"""
    st.header("📄 License Analysis")

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
    csv = license_df.to_csv(index=False)
    st.download_button(
        label="Download License Report (CSV)", data=csv,
        file_name="license_analysis.csv", mime="text/csv"
    )


def show_repository_browser():
    """Repository browser page"""
    st.header("📚 Repository Browser")

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

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_license = st.selectbox(
            "License", ['All'] + sorted(repo_df['license'].unique().tolist()))
    with col2:
        selected_language = st.selectbox(
            "Language", ['All'] + sorted(repo_df['language'].unique().tolist()))
    with col3:
        search_term = st.text_input("Search by name", "")

    filtered = repo_df.copy()
    if selected_license != 'All':
        filtered = filtered[filtered['license'] == selected_license]
    if selected_language != 'All':
        filtered = filtered[filtered['language'] == selected_language]
    if search_term:
        filtered = filtered[filtered['name'].str.contains(search_term, case=False, na=False)]

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**Showing {len(filtered):,} of {len(repo_df):,} repositories**")
    with col2:
        sort_by = st.selectbox("Sort by", ["Name", "Stars", "License"])

    if sort_by == "Stars":
        filtered = filtered.sort_values('stars', ascending=False)
    elif sort_by == "License":
        filtered = filtered.sort_values('license')
    else:
        filtered = filtered.sort_values('name')

    st.dataframe(
        filtered[['name', 'license', 'language', 'stars', 'url']].head(500),
        column_config={
            "name":     st.column_config.TextColumn("Repository", width="medium"),
            "license":  st.column_config.TextColumn("License",    width="medium"),
            "language": st.column_config.TextColumn("Language",   width="small"),
            "stars":    st.column_config.NumberColumn("⭐ Stars",  width="small"),
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
    csv = filtered.drop(columns=['domain'], errors='ignore').to_csv(index=False)
    st.download_button(
        label=f"Download {len(filtered)} Repositories (CSV)", data=csv,
        file_name="repository_list.csv", mime="text/csv"
    )


def show_data_quality():
    """Data quality and processing transparency report"""
    st.header("📋 Data Quality Report")

    session = get_db_session()
    total_repos      = session.query(Repository).count()
    total_readmes    = session.query(ReadmeHeader.repository_id).distinct().count()
    total_headers    = session.query(ReadmeHeader).count()
    total_embeddings = session.query(HeaderEmbedding).count()
    total_clustered  = session.query(HeaderClusterAssignment).count()
    total_clusters   = session.query(Cluster).count()

    if total_repos == 0:
        st.info("No data yet — run the pipeline first.")
        return

    repos_without_readme = total_repos - total_readmes
    avg_headers = total_headers / total_readmes if total_readmes > 0 else 0

    # The original scraping count from the source CSV
    ORIGINAL_SCRAPED = 4266
    invalid_removed  = max(ORIGINAL_SCRAPED - total_repos, 0)
    retention        = total_repos / ORIGINAL_SCRAPED * 100
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
        file_name="data_quality_report.md", mime="text/markdown"
    )


if __name__ == "__main__":
    main()

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
        ["Overview", "Cluster Explorer", "Visualization", "Search", "Export"]
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
            delta=f"{stats['total_headers']/stats['total_repos']:.1f} per repo"
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
- **Average Headers per Repository:** {stats['total_headers']/stats['total_repos']:.1f}

### Clustering Results
- **Number of Clusters:** {stats['total_clusters']}
- **Headers Clustered:** {stats['total_assignments']:,} ({stats['total_assignments']/stats['total_headers']*100:.1f}%)
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


if __name__ == "__main__":
    main()

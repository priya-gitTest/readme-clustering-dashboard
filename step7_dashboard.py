#!/usr/bin/env python3
import streamlit as st
#st.write("✅ Dashboard is loading!")
#"""
#Dashboard: README Header Clustering + License Analysis
#"""

import os

# CRITICAL: Read database URL from Streamlit secrets
if 'DATABASE_URL' in st.secrets:
    os.environ['DATABASE_URL'] = st.secrets['DATABASE_URL']
elif not os.getenv('DATABASE_URL'):
    st.error("❌ Database connection not configured.")
    st.stop()

# Rest of your dashboard code...
import pandas as pd
import numpy as np
# ... (rest of the imports and code from step7_dashboard.py)
#!/usr/bin/env python3
#"""
#Dashboard: README Header Clustering + License Analysis
#Interactive visualization and exploration of clustering results
#"""
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
#if 'DATABASE_URL' in st.secrets:
#    os.environ['DATABASE_URL'] = st.secrets['DATABASE_URL']
    
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
    """Load license information"""
    session = get_db_session()
    
    repos = session.query(Repository).all()
    
    license_data = []
    for repo in repos:
        license_info = repo.license_from_api if hasattr(repo, 'license_from_api') else None
        license_data.append({
            'name': repo.name,
            'url': repo.url,
            'license': license_info if license_info and license_info != 'NOASSERTION' else 'No License',
        })
    
    return pd.DataFrame(license_data)


def categorize_license(license_name):
    """Categorize license into type"""
    if not license_name or license_name == 'No License':
        return 'No License'
    
    permissive = ['MIT', 'Apache-2.0', 'BSD-2-Clause', 'BSD-3-Clause', 'ISC']
    copyleft_strong = ['GPL-3.0', 'GPL-2.0', 'AGPL-3.0']
    copyleft_weak = ['LGPL-3.0', 'LGPL-2.1', 'MPL-2.0']
    
    license_upper = license_name.upper()
    
    for lic in permissive:
        if lic.upper() in license_upper:
            return 'Permissive'
    for lic in copyleft_strong:
        if lic.upper() in license_upper:
            return 'Strong Copyleft'
    for lic in copyleft_weak:
        if lic.upper() in license_upper:
            return 'Weak Copyleft'
    
    return 'Other'

@st.cache_data
def load_repository_details():
    """Load detailed repository information"""
    session = get_db_session()
    
    repos = session.query(Repository).all()
    
    repo_data = []
    for repo in repos:
        license_info = repo.license_from_api if hasattr(repo, 'license_from_api') else None
        
        repo_data.append({
            'name': repo.name,
            'url': repo.url,
            'license': license_info if license_info and license_info != 'NOASSERTION' else 'No License',
            'stars': repo.stars if hasattr(repo, 'stars') else 0,
            'description': repo.description[:100] + '...' if hasattr(repo, 'description') and repo.description and len(repo.description) > 100 else (repo.description if hasattr(repo, 'description') else ''),
            'language': repo.primary_language if hasattr(repo, 'primary_language') else 'Unknown',
        })
    
    return pd.DataFrame(repo_data)

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
    st.markdown('<div class="main-header">📊 README Header Clustering + License Analysis</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Cluster Explorer", "License Analysis","Repository Browser", "Visualization", "Search", "Export"]
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
    [Priyanka O] (2026)
    README Header Clustering + License Analysis
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
        <p>© 2026 | MIT License | <a href='https://github.com/priya-gitTest/readme-clustering-dashboard'>View on GitHub</a></p>
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

def show_license_analysis():
    """License analysis page"""
    st.header("📄 License Analysis")
    
    license_df = load_license_data()
    
    # Overall statistics
    st.subheader("📊 Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_repos = len(license_df)
    with_license = len(license_df[license_df['license'] != 'No License'])
    without_license = total_repos - with_license
    
    with col1:
        st.metric("Total Repositories", f"{total_repos:,}")
    
    with col2:
        st.metric("With License", f"{with_license:,}", 
                 delta=f"{with_license/total_repos*100:.1f}%")
    
    with col3:
        st.metric("Without License", f"{without_license:,}",
                 delta=f"-{without_license/total_repos*100:.1f}%", delta_color="inverse")
    
    with col4:
        unique_licenses = license_df[license_df['license'] != 'No License']['license'].nunique()
        st.metric("Unique Licenses", unique_licenses)
    
    st.markdown("---")
    
    # Top licenses
    st.subheader("🏆 Top 10 Licenses")
    
    license_counts = license_df['license'].value_counts().head(10)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            x=license_counts.values,
            y=license_counts.index,
            orientation='h',
            title="Most Common Licenses",
            labels={'x': 'Number of Repositories', 'y': 'License'},
            color=license_counts.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(
            values=license_counts.values,
            names=license_counts.index,
            title="License Distribution",
            hole=0.3
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # License categories
    st.subheader("📂 License Categories")
    
    license_df['category'] = license_df['license'].apply(categorize_license)
    category_counts = license_df['category'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            x=category_counts.index,
            y=category_counts.values,
            title="Licenses by Category",
            labels={'x': 'Category', 'y': 'Count'},
            color=category_counts.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Category Breakdown")
        for category, count in category_counts.items():
            percentage = (count / total_repos) * 100
            st.metric(category, f"{count:,}", delta=f"{percentage:.1f}%")
    
    st.markdown("---")
    
    # Export
    st.subheader("📥 Export")
    
    csv = license_df.to_csv(index=False)
    st.download_button(
        label="Download License Report (CSV)",
        data=csv,
        file_name="license_analysis.csv",
        mime="text/csv"
    )

def show_repository_browser():
    """Repository browser page"""
    st.header("📚 Repository Browser")
    
    st.markdown("""
    Browse all repositories analyzed in this study. 
    View licenses, descriptions, and verify the dataset used for clustering analysis.
    """)
    
    repo_df = load_repository_details()
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Repositories", f"{len(repo_df):,}")
    
    with col2:
        with_license = len(repo_df[repo_df['license'] != 'No License'])
        st.metric("With License", f"{with_license:,}")
    
    with col3:
        languages = repo_df['language'].nunique()
        st.metric("Programming Languages", languages)
    
    with col4:
        total_stars = repo_df['stars'].sum()
        st.metric("Total Stars", f"{total_stars:,}")
    
    st.markdown("---")
    
    # Filters
    st.subheader("🔍 Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # License filter
        all_licenses = ['All'] + sorted(repo_df['license'].unique().tolist())
        selected_license = st.selectbox("License", all_licenses)
    
    with col2:
        # Language filter
        all_languages = ['All'] + sorted(repo_df['language'].unique().tolist())
        selected_language = st.selectbox("Language", all_languages)
    
    with col3:
        # Search
        search_term = st.text_input("Search by name", "")
    
    # Apply filters
    filtered_df = repo_df.copy()
    
    if selected_license != 'All':
        filtered_df = filtered_df[filtered_df['license'] == selected_license]
    
    if selected_language != 'All':
        filtered_df = filtered_df[filtered_df['language'] == selected_language]
    
    if search_term:
        filtered_df = filtered_df[
            filtered_df['name'].str.contains(search_term, case=False, na=False)
        ]
    
    # Sort options
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"**Showing {len(filtered_df):,} of {len(repo_df):,} repositories**")
    
    with col2:
        sort_by = st.selectbox("Sort by", ["Name", "Stars", "License"])
    
    # Sort
    if sort_by == "Name":
        filtered_df = filtered_df.sort_values('name')
    elif sort_by == "Stars":
        filtered_df = filtered_df.sort_values('stars', ascending=False)
    elif sort_by == "License":
        filtered_df = filtered_df.sort_values('license')
    
    st.markdown("---")
    
    # Display as table
    st.subheader("📋 Repository List")
    
    # Make URL clickable
    display_df = filtered_df[['name', 'license', 'language', 'stars', 'url']].head(500)
    
    st.dataframe(
        display_df,
        column_config={
            "name": st.column_config.TextColumn("Repository", width="medium"),
            "license": st.column_config.TextColumn("License", width="medium"),
            "language": st.column_config.TextColumn("Language", width="small"),
            "stars": st.column_config.NumberColumn("⭐ Stars", width="small"),
            "url": st.column_config.LinkColumn("Link", width="medium"),
        },
        hide_index=True,
        use_container_width=True,
        height=600
    )
    
    if len(filtered_df) > 500:
        st.info(f"💡 Showing first 500 results. Use filters to narrow down.")
    
    st.markdown("---")
    
    # Export filtered results
    st.subheader("📥 Export Filtered Results")
    
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label=f"Download {len(filtered_df)} Repositories (CSV)",
        data=csv,
        file_name="repository_list.csv",
        mime="text/csv"
    )
    
    # Summary statistics
    st.markdown("---")
    st.subheader("📊 Quick Stats")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Repository Sources**")
        # Parse domains from URLs
        filtered_df['domain'] = filtered_df['url'].apply(lambda x: urlparse(x).netloc if pd.notna(x) else 'Unknown')
        top_domains = filtered_df['domain'].value_counts().head(5)
        for domain, count in top_domains.items():
            st.write(f"- **{domain}**: {count} repos")
    
    with col2:
        st.markdown("**Top 5 Licenses**")
        top_licenses = filtered_df['license'].value_counts().head(5)
        for lic, count in top_licenses.items():
            st.write(f"- **{lic}**: {count} repos")

if __name__ == "__main__":
    main()

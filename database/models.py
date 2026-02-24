"""
Database models for AI_assisted_Code_metadata_pipeline
SQLAlchemy ORM models for all tables
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    UniqueConstraint,
    LargeBinary,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Repository(Base):
    """Main repository metadata table - CORRECTED to match actual DB schema"""

    __tablename__ = "repositories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    url = Column(String(500), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    owner = Column(String(255))
    platform = Column(String(50))  # github, gitlab, bitbucket
    description = Column(Text)
    language = Column(String(100))
    stars = Column(Integer, default=0)
    forks = Column(Integer, default=0)
    license_from_api = Column(String(100))
    readme_content = Column(Text)
    readme_format = Column(String(20), default="markdown")  # markdown, rst, txt
    scraped_at = Column(DateTime)

    # Source tracking (which CSV / input file this repo came from)
    source = Column(Text)

    # License compliance columns (from migration 001_license_compliance_columns.sql)
    license_spdx_ok = Column(Boolean)
    license_filename = Column(Text)
    license_note = Column(Text)

    # License enrichment columns (from migration 002_license_category_source.sql)
    license_category = Column(String(30))  # Permissive/WeakCopyleft/StrongCopyleft/PublicDomain/NonOSS/Custom/NoLicense/Other
    license_source   = Column(String(30))  # github_api / spdx_header / file_scan

    # Relationships
    headers = relationship("ReadmeHeader", back_populates="repository", cascade="all, delete-orphan")
    licenses = relationship("RepositoryLicense", back_populates="repository", cascade="all, delete-orphan")
    history = relationship("RepositoryHistory", back_populates="repository", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_repo_platform_owner", "platform", "owner"),
        Index("idx_repo_language", "language"),
    )

class ReadmeMetadataExtracted(Base):
    """Extracted README metadata (Step 3)"""
    
    __tablename__ = "readme_metadata"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    repository_id = Column(Integer, ForeignKey("repositories.id", ondelete="CASCADE"), unique=True, nullable=False)
    readme_title = Column(Text)
    readme_description = Column(Text)
    documentation_url = Column(Text)
    badge_count = Column(Integer, default=0)
    section_count = Column(Integer, default=0)
    has_doi = Column(Boolean, default=False)
    has_installation_section = Column(Boolean, default=False)
    has_usage_section = Column(Boolean, default=False)
    has_citation_section = Column(Boolean, default=False)
    readme_metadata_json = Column(Text)  # JSON
    doi_list = Column(Text)  # JSON
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    repository = relationship("Repository", backref="readme_metadata_extracted")
    
class ReadmeHeader(Base):
    """Extracted README headers (H1-H5)"""

    __tablename__ = "readme_headers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    repository_id = Column(Integer, ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False)
    header_text = Column(Text, nullable=False)
    level = Column(Integer, nullable=False)  # 1-6 for H1-H6
    normalized_text = Column(Text, index=True)
    position = Column(Integer, default=0)  # Position in README (order)
    has_content = Column(Boolean, default=False)  # TRUE if section has non-empty body
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    repository = relationship("Repository", back_populates="headers")
    embedding = relationship("HeaderEmbedding", back_populates="header", uselist=False, cascade="all, delete-orphan")
    cluster_assignment = relationship("HeaderClusterAssignment", back_populates="header", uselist=False, cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_readme_headers_repo", "repository_id"),
        Index("idx_header_level", "level"),
        Index("idx_header_normalized", "normalized_text"),
    )


class HeaderEmbedding(Base):
    """Sentence embeddings for headers (Step 6)"""

    __tablename__ = "header_embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    header_id = Column(Integer, ForeignKey("readme_headers.id", ondelete="CASCADE"), unique=True, nullable=False)
    embedding_vector = Column(LargeBinary, nullable=False)  # Stored as bytes (numpy array)
    model_name = Column(String(100), default="all-MiniLM-L6-v2")
    embedding_dim = Column(Integer, default=384)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    header = relationship("ReadmeHeader", back_populates="embedding")

    __table_args__ = (Index("idx_embedding_model", "model_name"),)


class Cluster(Base):
    """Clustering results (Step 7)"""

    __tablename__ = "clusters"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(50), nullable=False)  # Links to analysis_runs
    cluster_id = Column(Integer, nullable=False)  # Cluster number (0, 1, 2, ...)
    cluster_name = Column(String(255))
    cluster_size = Column(Integer, default=0)
    representative_headers = Column(Text)  # JSON array of top headers
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    codemeta_mappings = relationship("ClusterCodemetaMapping", back_populates="cluster", cascade="all, delete-orphan")
    reuse_scenarios = relationship("ClusterReuseScenario", back_populates="cluster", cascade="all, delete-orphan")
    assignments = relationship("HeaderClusterAssignment", back_populates="cluster", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("run_id", "cluster_id", name="uq_run_cluster"),
        Index("idx_cluster_run", "run_id"),
    )


class HeaderClusterAssignment(Base):
    """Many-to-many: headers to clusters"""

    __tablename__ = "header_cluster_assignments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    header_id = Column(Integer, ForeignKey("readme_headers.id", ondelete="CASCADE"), nullable=False)
    cluster_id = Column(Integer, ForeignKey("clusters.id", ondelete="CASCADE"), nullable=False)
    distance = Column(Float)  # Distance to cluster centroid
    assigned_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    header = relationship("ReadmeHeader", back_populates="cluster_assignment")
    cluster = relationship("Cluster", back_populates="assignments")

    __table_args__ = (
        UniqueConstraint("header_id", name="uq_header_cluster"),  # One cluster per header
        Index("idx_assignment_cluster", "cluster_id"),
    )


class ClusterCodemetaMapping(Base):
    """Maps clusters to CodeMeta vocabulary properties"""

    __tablename__ = "cluster_codemeta_mappings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cluster_id = Column(Integer, ForeignKey("clusters.id", ondelete="CASCADE"), nullable=False)
    codemeta_property = Column(String(100), nullable=False)
    confidence = Column(Float)  # 0.0-1.0
    mapping_method = Column(String(50))  # manual, automatic, hybrid
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    cluster = relationship("Cluster", back_populates="codemeta_mappings")

    __table_args__ = (Index("idx_codemeta_property", "codemeta_property"),)


class ClusterReuseScenario(Base):
    """Maps clusters to FAIR4RS reuse scenarios"""

    __tablename__ = "cluster_reuse_scenarios"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cluster_id = Column(Integer, ForeignKey("clusters.id", ondelete="CASCADE"), nullable=False)
    scenario = Column(String(50), nullable=False)  # execution, understanding, modifying, etc.
    relevance_score = Column(Float)  # 0.0-1.0
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    cluster = relationship("Cluster", back_populates="reuse_scenarios")

    __table_args__ = (Index("idx_scenario", "scenario"),)


class UnmappedCluster(Base):
    """Clusters with no CodeMeta equivalent (gap analysis)"""

    __tablename__ = "unmapped_clusters"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cluster_id = Column(Integer, ForeignKey("clusters.id", ondelete="CASCADE"), nullable=False)
    proposed_property_name = Column(String(100))
    justification = Column(Text)
    priority = Column(String(20))  # high, medium, low
    status = Column(String(20), default="proposed")  # proposed, accepted, rejected
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("idx_unmapped_priority", "priority"),)


class License(Base):
    """License catalog (SPDX identifiers)"""

    __tablename__ = "licenses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    spdx_id = Column(String(50), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    category = Column(String(50))  # permissive, weak_copyleft, strong_copyleft, etc.
    is_osi_approved = Column(Boolean, default=False)
    url = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    repository_licenses = relationship("RepositoryLicense", back_populates="license")

    __table_args__ = (Index("idx_license_category", "category"),)


class LicenseFile(Base):
    """Detected license files in repositories"""

    __tablename__ = "license_files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    repository_id = Column(Integer, ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False)
    filename = Column(String(100), nullable=False)  # LICENSE, LICENSE.md, COPYING, etc.
    content = Column(Text)
    detected_license = Column(String(50))
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("idx_licensefile_repo", "repository_id"),)


class RepositoryLicense(Base):
    """Many-to-many: repositories to licenses (with source tracking)"""

    __tablename__ = "repository_licenses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    repository_id = Column(Integer, ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False)
    license_id = Column(Integer, ForeignKey("licenses.id", ondelete="CASCADE"), nullable=False)
    source = Column(String(50), nullable=False)  # api, license_file, readme, package_metadata
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    repository = relationship("Repository", back_populates="licenses")
    license = relationship("License", back_populates="repository_licenses")

    __table_args__ = (
        Index("idx_repo_license_join", "repository_id", "license_id"),
        Index("idx_license_source", "source"),
    )


class AnalysisRun(Base):
    """Metadata for each analysis run (temporal tracking)"""

    __tablename__ = "analysis_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(50), unique=True, nullable=False)
    run_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    total_repos = Column(Integer)
    processed_repos = Column(Integer)
    failed_repos = Column(Integer)
    n_clusters = Column(Integer)
    avg_codemeta_coverage = Column(Float)
    config_snapshot = Column(Text)  # JSON of config used
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("idx_run_date", "run_date"),)


class RepositoryHistory(Base):
    """Snapshots for temporal delta analysis"""

    __tablename__ = "repository_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    repository_id = Column(Integer, ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False)
    run_id = Column(String(50), ForeignKey("analysis_runs.run_id"), nullable=False)
    status = Column(String(20), nullable=False)  # active, deleted, private, error
    readme_hash = Column(String(64))  # SHA256 of README content
    license = Column(String(100))
    header_count = Column(Integer)
    codemeta_coverage = Column(Float)
    snapshot_date = Column(DateTime, default=datetime.utcnow)

    # Relationships
    repository = relationship("Repository", back_populates="history")

    __table_args__ = (
        Index("idx_history_run", "run_id"),
        Index("idx_history_repo_run", "repository_id", "run_id"),
    )

class ProcessingMetadata(Base):
    """Track processing pipeline metadata and statistics"""
    __tablename__ = 'processing_metadata'

    id = Column(Integer, primary_key=True)
    stage = Column(String(100), nullable=False, unique=True)
    count = Column(Integer, nullable=False)
    description = Column(Text)
    processed_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SomefResult(Base):
    """SOMEF metadata extraction results (Step 8 — optional validation).

    Each row holds the structured output of running SOMEF on a repository's
    README.  The per-category columns store JSON arrays of the form:
        [{"excerpt": "<text>", "confidence": 0.9}, ...]
    so the full detail is preserved while remaining queryable.
    """

    __tablename__ = "somef_results"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    repository_id   = Column(Integer, ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False)
    run_date        = Column(DateTime, default=datetime.utcnow)
    somef_version   = Column(String(20))

    # SOMEF extraction categories (JSON: [{excerpt, confidence}])
    description     = Column(Text)
    installation    = Column(Text)
    invocation      = Column(Text)
    citation        = Column(Text)
    requirement     = Column(Text)
    documentation   = Column(Text)
    contributor     = Column(Text)
    license         = Column(Text)
    usage           = Column(Text)
    acknowledgement = Column(Text)
    run             = Column(Text)
    support         = Column(Text)

    categories_found    = Column(Text)   # JSON list of detected category names
    processing_time_s   = Column(Float)
    error               = Column(Text)   # NULL on success

    # Relationship
    repository = relationship("Repository", backref="somef_results")

    __table_args__ = (
        UniqueConstraint("repository_id", name="uq_somef_repo"),
        Index("idx_somef_repo", "repository_id"),
    )


class ExcludedRepository(Base):
    """Audit log of every URL filtered out during the pipeline, with reason and stage.

    Lifecycle:
    - is_retryable=True  → re-attempted on the next run (e.g. 404, timeout, scraping exception)
    - is_retryable=False → permanent exclusion, never retried (e.g. unknown platform, bad URL)
    - resolved_at        → set when the URL is later successfully scraped; row kept for audit trail
    Multiple rows per URL are allowed (one per exclusion event across runs).
    """

    __tablename__ = "excluded_repositories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    url = Column(Text, nullable=False)
    exclusion_reason = Column(String(255), nullable=False)  # e.g. 'Repository not found (404)'
    exclusion_stage = Column(String(100), nullable=False, default="scraping")  # scraping | validation
    notes = Column(Text)
    excluded_at = Column(DateTime, default=datetime.utcnow)
    source = Column(Text)  # which CSV / input file this URL came from

    # Retry management
    is_retryable = Column(Boolean, default=True)   # False for permanent errors (unknown platform, bad URL)
    resolved_at  = Column(DateTime)                # Set when URL is successfully scraped in a later run

    __table_args__ = (
        Index("idx_excluded_url", "url"),
        Index("idx_excluded_reason", "exclusion_reason"),
        Index("idx_excluded_stage", "exclusion_stage"),
        Index("idx_excluded_retryable", "is_retryable", "resolved_at"),
    )


class UnsupportedRepository(Base):
    """URLs collected by step0 whose platform is not yet supported by step2.

    Each row is one unique URL.  On repeated step0 runs the occurrence_count
    is incremented and last_seen_at updated, making it easy to see which
    unsupported hosts appear most often and are therefore the best candidates
    for future platform support (scrapers/utils.py _GITLAB_SUBSTRINGS or a
    new dedicated scraper).
    """

    __tablename__ = "unsupported_repositories"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    url              = Column(Text, nullable=False)
    source           = Column(String(100), nullable=False)   # rsd / helmholtz / joss
    host             = Column(String(255), nullable=False)   # urlparse(url).netloc
    platform         = Column(String(100), nullable=False)   # detect_platform() or "unrecognised"
    first_seen_at    = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_seen_at     = Column(DateTime, nullable=False, default=datetime.utcnow)
    occurrence_count = Column(Integer, nullable=False, default=1)

    __table_args__ = (
        UniqueConstraint("url", name="uq_unsupported_url"),
        Index("idx_unsupported_host",     "host"),
        Index("idx_unsupported_platform", "platform"),
        Index("idx_unsupported_source",   "source"),
    )
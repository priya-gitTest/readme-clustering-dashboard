#!/usr/bin/env python3
"""
Research Dashboard: README Header Clustering Analysis
"""
import streamlit as st
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
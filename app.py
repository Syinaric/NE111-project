import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, Optional, Dict, Any

st.set_page_config(
    page_title="Distribution Fitting Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_theme_css(is_dark):
    if is_dark:
        return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .main-header {
        font-size: 2.25rem;
        font-weight: 600;
        color: #FAFAFA !important;
        text-align: center;
        margin-bottom: 2.5rem;
        letter-spacing: -0.01em;
    }
    
    .stExpander {
        border: 1px solid #2d3748;
        border-radius: 0px;
        background-color: #1A1C23;
    }
    
    .stExpander > div > div {
        background-color: #1A1C23;
    }
    
    .stButton > button {
        font-weight: 500;
        border-radius: 0px;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
    }
    
    .stSelectbox > div > div {
        border-radius: 0px;
    }
    
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: 0px;
        color: #FAFAFA !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-weight: 600;
        color: #FAFAFA !important;
    }
    
    h3 {
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-size: 1.5rem;
        color: #FAFAFA !important;
    }
    
    h4 {
        font-weight: 600;
        color: #E2E8F0 !important;
        font-size: 1.1rem;
        margin-top: 1rem;
        margin-bottom: 0.75rem;
    }
    
    .stMarkdown {
        color: #FAFAFA !important;
    }
    
    .stMarkdown p, .stMarkdown li, .stMarkdown div {
        color: #FAFAFA !important;
    }
    
    .stMarkdown hr {
        margin: 2rem 0;
        border: none;
        border-top: 1px solid #2d3748;
    }
    
    [data-testid="stMetricValue"] {
        font-weight: 600;
        color: #FAFAFA !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-weight: 500;
        color: #CBD5E0 !important;
        font-size: 0.875rem;
    }
    
    .stSuccess {
        border-left: 3px solid #4caf50;
        background-color: rgba(76, 175, 80, 0.1);
        color: #FAFAFA !important;
    }
    
    .stError {
        border-left: 3px solid #f44336;
        background-color: rgba(244, 67, 54, 0.1);
        color: #FAFAFA !important;
    }
    
    .stWarning {
        border-left: 3px solid #ff9800;
        background-color: rgba(255, 152, 0, 0.1);
        color: #FAFAFA !important;
    }
    
    .stInfo {
        border-left: 3px solid #2196f3;
        background-color: rgba(33, 150, 243, 0.1);
        color: #FAFAFA !important;
    }
    
    .metric-card {
        background-color: #1A1C23;
        padding: 1rem;
        border-radius: 0px;
        margin: 0.5rem 0;
        border: 1px solid #2d3748;
    }
    
    label {
        color: #FAFAFA !important;
    }
    
    .stSelectbox label, .stTextInput label, .stTextArea label {
        color: #FAFAFA !important;
    }
    
    .theme-switcher {
        position: fixed;
        top: 10px;
        right: 10px;
        z-index: 999;
    }
    
    * {
        border-radius: 0px !important;
    }
    
    button, input, select, textarea, div, section, article, aside, header, footer, nav, main {
        border-radius: 0px !important;
    }
    
    [class*="st"] {
        border-radius: 0px !important;
    }
    
    [data-testid] {
        border-radius: 0px !important;
    }
    </style>
"""
    else:
        return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .stApp {
        background-color: #ffffff !important;
    }
    
    .main {
        background-color: #ffffff !important;
    }
    
    [data-testid="stAppViewContainer"] {
        background-color: #ffffff !important;
    }
    
    .main-header {
        font-size: 2.25rem;
        font-weight: 600;
        color: #1a1a1a !important;
        text-align: center;
        margin-bottom: 2.5rem;
        letter-spacing: -0.01em;
    }
    
    .stExpander {
        border: 1px solid #e0e0e0 !important;
        border-radius: 0px;
        background-color: #ffffff !important;
    }
    
    .stExpander > div > div {
        background-color: #ffffff !important;
    }
    
    .stExpander label {
        color: #1a1a1a !important;
    }
    
    .stButton > button {
        font-weight: 500;
        border-radius: 0px;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
    }
    
    .stSelectbox > div > div {
        border-radius: 0px;
        background-color: #ffffff !important;
    }
    
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: 0px;
        color: #1a1a1a !important;
        background-color: #ffffff !important;
    }
    
    .stCheckbox label {
        color: #1a1a1a !important;
    }
    
    .stRadio label {
        color: #1a1a1a !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-weight: 600;
        color: #1a1a1a !important;
    }
    
    h3 {
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-size: 1.5rem;
        color: #1a1a1a !important;
    }
    
    h4 {
        font-weight: 600;
        color: #2c3e50 !important;
        font-size: 1.1rem;
        margin-top: 1rem;
        margin-bottom: 0.75rem;
    }
    
    .stMarkdown {
        color: #1a1a1a !important;
    }
    
    .stMarkdown p, .stMarkdown li, .stMarkdown div, .stMarkdown span {
        color: #1a1a1a !important;
    }
    
    .stMarkdown hr {
        margin: 2rem 0;
        border: none;
        border-top: 1px solid #e0e0e0;
    }
    
    [data-testid="stMetricValue"] {
        font-weight: 600;
        color: #1a1a1a !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-weight: 500;
        color: #666666 !important;
        font-size: 0.875rem;
    }
    
    .stSuccess {
        border-left: 3px solid #4caf50;
        background-color: #f1f8f4 !important;
        color: #1a1a1a !important;
    }
    
    .stSuccess p, .stSuccess div {
        color: #1a1a1a !important;
    }
    
    .stError {
        border-left: 3px solid #f44336;
        background-color: #ffebee !important;
        color: #1a1a1a !important;
    }
    
    .stError p, .stError div {
        color: #1a1a1a !important;
    }
    
    .stWarning {
        border-left: 3px solid #ff9800;
        background-color: #fff3e0 !important;
        color: #1a1a1a !important;
    }
    
    .stWarning p, .stWarning div {
        color: #1a1a1a !important;
    }
    
    .stInfo {
        border-left: 3px solid #2196f3;
        background-color: #e3f2fd !important;
        color: #1a1a1a !important;
    }
    
    .stInfo p, .stInfo div {
        color: #1a1a1a !important;
    }
    
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0px;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
    }
    
    label {
        color: #1a1a1a !important;
    }
    
    .stSelectbox label, .stTextInput label, .stTextArea label {
        color: #1a1a1a !important;
    }
    
    [data-testid="stToggle"] label {
        color: #1a1a1a !important;
    }
    
    [data-testid="stToggle"] {
        color: #1a1a1a !important;
        background-color: #ffffff !important;
    }
    
    [data-testid="stToggle"] > label {
        color: #1a1a1a !important;
        background-color: transparent !important;
    }
    
    [data-testid="stToggle"] label {
        color: #1a1a1a !important;
        background-color: transparent !important;
    }
    
    .stToggle {
        color: #1a1a1a !important;
        background-color: #ffffff !important;
    }
    
    .stToggle label {
        color: #1a1a1a !important;
        background-color: transparent !important;
    }
    
    .stToggle > label {
        color: #1a1a1a !important;
        background-color: transparent !important;
    }
    
    .stToggle * {
        color: #1a1a1a !important;
    }
    
    div[data-baseweb="toggle"] {
        color: #1a1a1a !important;
        background-color: #ffffff !important;
    }
    
    div[data-baseweb="toggle"] label {
        color: #1a1a1a !important;
        background-color: transparent !important;
    }
    
    [data-baseweb="toggle"] {
        color: #1a1a1a !important;
        background-color: #ffffff !important;
    }
    
    [data-baseweb="toggle"] label {
        color: #1a1a1a !important;
        background-color: transparent !important;
    }
    
    button[data-baseweb="toggle"] {
        background-color: #4A90E2 !important;
    }
    
    [data-baseweb="base-input"] {
        background-color: #ffffff !important;
    }
    
    .element-container [data-testid="stToggle"] {
        background-color: #ffffff !important;
        padding: 10px !important;
        border: 1px solid #e0e0e0 !important;
    }
    
    .element-container [data-testid="stToggle"] label {
        color: #1a1a1a !important;
        font-weight: 500 !important;
    }
    
    .theme-switcher {
        position: fixed;
        top: 10px;
        right: 10px;
        z-index: 999;
    }
    
    p, span, div {
        color: #1a1a1a !important;
    }
    
    .element-container {
        color: #1a1a1a !important;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] * {
        color: #1a1a1a !important;
    }
    
    * {
        border-radius: 0px !important;
    }
    
    button, input, select, textarea, div, section, article, aside, header, footer, nav, main {
        border-radius: 0px !important;
    }
    
    [class*="st"] {
        border-radius: 0px !important;
    }
    
    [data-testid] {
        border-radius: 0px !important;
    }
    </style>
"""

@st.cache_data
def parse_text_input(text: str) -> np.ndarray:
    if not text or not text.strip():
        return np.array([])
    
    text = text.replace(',', ' ').replace('\n', ' ').replace('\t', ' ')
    
    values = []
    for item in text.split():
        try:
            val = float(item.strip())
            if not np.isnan(val) and np.isfinite(val):
                values.append(val)
        except (ValueError, TypeError):
            continue
    
    return np.array(values)


def clean_data(data: np.ndarray) -> np.ndarray:
    if data is None or len(data) == 0:
        return np.array([])
    
    data = np.asarray(data, dtype=float)
    
    mask = np.isfinite(data)
    cleaned = data[mask]
    
    return cleaned


def get_data_summary(data: np.ndarray) -> Dict[str, float]:
    if data is None or len(data) == 0:
        return {}
    
    return {
        'count': len(data),
        'min': np.min(data),
        'max': np.max(data),
        'mean': np.mean(data),
        'std': np.std(data),
        'median': np.median(data)
    }


DISTRIBUTIONS = {
    'Normal': {
        'dist': stats.norm,
        'params': ['loc', 'scale'],
        'param_names': ['Mean (μ)', 'Std Dev (σ)'],
        'param_bounds': lambda data: {
            'loc': (np.min(data) - 0.1 * (np.max(data) - np.min(data)), 
                   np.max(data) + 0.1 * (np.max(data) - np.min(data))),
            'scale': (0.01, np.std(data) * 3)
        }
    },
    'Gamma': {
        'dist': stats.gamma,
        'params': ['a', 'loc', 'scale'],
        'param_names': ['Shape (α)', 'Location', 'Scale'],
        'param_bounds': lambda data: {
            'a': (0.1, 10.0),
            'loc': (np.min(data) - 0.1 * (np.max(data) - np.min(data)), 
                   np.max(data) + 0.1 * (np.max(data) - np.min(data))),
            'scale': (0.01, np.std(data) * 3)
        }
    },
    'Weibull': {
        'dist': stats.weibull_min,
        'params': ['c', 'loc', 'scale'],
        'param_names': ['Shape (c)', 'Location', 'Scale'],
        'param_bounds': lambda data: {
            'c': (0.1, 10.0),
            'loc': (np.min(data) - 0.1 * (np.max(data) - np.min(data)), 
                   np.max(data) + 0.1 * (np.max(data) - np.min(data))),
            'scale': (0.01, np.std(data) * 3)
        }
    },
    'Exponential': {
        'dist': stats.expon,
        'params': ['loc', 'scale'],
        'param_names': ['Location', 'Scale (λ)'],
        'param_bounds': lambda data: {
            'loc': (np.min(data) - 0.1 * (np.max(data) - np.min(data)), 
                   np.max(data) + 0.1 * (np.max(data) - np.min(data))),
            'scale': (0.01, np.std(data) * 3)
        }
    },
    'Lognormal': {
        'dist': stats.lognorm,
        'params': ['s', 'loc', 'scale'],
        'param_names': ['Shape (σ)', 'Location', 'Scale'],
        'param_bounds': lambda data: {
            's': (0.01, 3.0),
            'loc': (np.min(data) - 0.1 * (np.max(data) - np.min(data)), 
                   np.max(data) + 0.1 * (np.max(data) - np.min(data))),
            'scale': (0.01, np.max(data) * 2)
        }
    },
    'Beta': {
        'dist': stats.beta,
        'params': ['a', 'b', 'loc', 'scale'],
        'param_names': ['Shape α', 'Shape β', 'Location', 'Scale'],
        'param_bounds': lambda data: {
            'a': (0.1, 10.0),
            'b': (0.1, 10.0),
            'loc': (np.min(data) - 0.1 * (np.max(data) - np.min(data)), 
                   np.max(data) + 0.1 * (np.max(data) - np.min(data))),
            'scale': (0.01, np.max(data) - np.min(data) + 0.1 * (np.max(data) - np.min(data)))
        }
    },
    'Chi-square': {
        'dist': stats.chi2,
        'params': ['df', 'loc', 'scale'],
        'param_names': ['Degrees of Freedom', 'Location', 'Scale'],
        'param_bounds': lambda data: {
            'df': (0.1, 50.0),
            'loc': (np.min(data) - 0.1 * (np.max(data) - np.min(data)), 
                   np.max(data) + 0.1 * (np.max(data) - np.min(data))),
            'scale': (0.01, np.std(data) * 3)
        }
    },
    'Rayleigh': {
        'dist': stats.rayleigh,
        'params': ['loc', 'scale'],
        'param_names': ['Location', 'Scale'],
        'param_bounds': lambda data: {
            'loc': (np.min(data) - 0.1 * (np.max(data) - np.min(data)), 
                   np.max(data) + 0.1 * (np.max(data) - np.min(data))),
            'scale': (0.01, np.std(data) * 3)
        }
    },
    'Uniform': {
        'dist': stats.uniform,
        'params': ['loc', 'scale'],
        'param_names': ['Location (a)', 'Scale (b-a)'],
        'param_bounds': lambda data: {
            'loc': (np.min(data) - 0.1 * (np.max(data) - np.min(data)), 
                   np.max(data) + 0.1 * (np.max(data) - np.min(data))),
            'scale': (0.01, np.max(data) - np.min(data) + 0.1 * (np.max(data) - np.min(data)))
        }
    },
    'Student t': {
        'dist': stats.t,
        'params': ['df', 'loc', 'scale'],
        'param_names': ['Degrees of Freedom', 'Location', 'Scale'],
        'param_bounds': lambda data: {
            'df': (0.1, 50.0),
            'loc': (np.min(data) - 0.1 * (np.max(data) - np.min(data)), 
                   np.max(data) + 0.1 * (np.max(data) - np.min(data))),
            'scale': (0.01, np.std(data) * 3)
        }
    }
}


@st.cache_data
def fit_distribution(data: np.ndarray, dist_name: str) -> Tuple[Optional[tuple], Optional[Dict[str, float]]]:
    if data is None or len(data) == 0:
        return None, None
    
    if dist_name not in DISTRIBUTIONS:
        return None, None
    
    dist_info = DISTRIBUTIONS[dist_name]
    dist = dist_info['dist']
    
    try:
        params = dist.fit(data)
        
        param_dict = {}
        param_names = dist_info['params']
        
        if isinstance(params, tuple):
            for i, param_name in enumerate(param_names):
                param_dict[param_name] = params[i]
        else:
            param_dict[param_names[0]] = params
        
        return params, param_dict
    except Exception as e:
        st.warning(f"Error fitting {dist_name}: {str(e)}")
        return None, None


def compute_pdf(dist_name: str, params: tuple, x: np.ndarray) -> np.ndarray:
    if dist_name not in DISTRIBUTIONS:
        return np.array([])
    
    dist = DISTRIBUTIONS[dist_name]['dist']
    
    try:
        pdf_values = dist.pdf(x, *params)
        pdf_values = np.nan_to_num(pdf_values, nan=0.0, posinf=0.0, neginf=0.0)
        return pdf_values
    except Exception as e:
        return np.array([])


def compute_fit_metrics(data: np.ndarray, dist_name: str, params: tuple, 
                        bins: int = 50) -> Dict[str, float]:
    if data is None or len(data) == 0 or params is None:
        return {}
    
    try:
        hist, bin_edges = np.histogram(data, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        pdf_values = compute_pdf(dist_name, params, bin_centers)
        
        if len(pdf_values) == 0 or len(hist) == 0:
            return {}
        
        mae = np.mean(np.abs(hist - pdf_values))
        max_error = np.max(np.abs(hist - pdf_values))
        
        rmse = np.sqrt(np.mean((hist - pdf_values) ** 2))
        
        ss_res = np.sum((hist - pdf_values) ** 2)
        ss_tot = np.sum((hist - np.mean(hist)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return {
            'MAE': mae,
            'Max Error': max_error,
            'RMSE': rmse,
            'R²': r_squared
        }
    except Exception as e:
        return {}


def plot_distribution_fit(data: np.ndarray, dist_name: str, params: tuple, 
                         manual_params: Optional[Dict[str, float]] = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if manual_params is not None:
        dist_info = DISTRIBUTIONS[dist_name]
        param_list = []
        for param_name in dist_info['params']:
            param_list.append(manual_params.get(param_name, 0.0))
        params = tuple(param_list)
    
    n_bins = min(50, max(10, len(data) // 10))
    hist, bin_edges, patches = ax.hist(data, bins=n_bins, density=True, 
                                       alpha=0.7, color='skyblue', 
                                       edgecolor='black', label='Data Histogram')
    
    x_min, x_max = np.min(data), np.max(data)
    x_range = x_max - x_min
    x_grid = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 1000)
    
    pdf_values = compute_pdf(dist_name, params, x_grid)
    
    if len(pdf_values) > 0:
        ax.plot(x_grid, pdf_values, 'r-', linewidth=2, label=f'{dist_name} PDF')
    
    ax.set_xlabel('Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Distribution Fit: {dist_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = True
    
    st.markdown(get_theme_css(st.session_state.dark_mode), unsafe_allow_html=True)
    
    col_header, col_theme = st.columns([5, 1])
    
    with col_header:
        st.markdown('<h1 class="main-header">Distribution Fitting Tool</h1>', 
                    unsafe_allow_html=True)
    
    with col_theme:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <style>
            div[data-testid="column"]:nth-child(2) {{
                background-color: {'#1A1C23' if st.session_state.dark_mode else '#ffffff'} !important;
            }}
            div[data-testid="column"]:nth-child(2) * {{
                color: {'#FAFAFA' if st.session_state.dark_mode else '#1a1a1a'} !important;
            }}
            [data-testid="stToggle"] {{
                display: block !important;
                visibility: visible !important;
                opacity: 1 !important;
            }}
            [data-testid="stToggle"] * {{
                color: {'#FAFAFA' if st.session_state.dark_mode else '#1a1a1a'} !important;
            }}
            [data-testid="stToggle"] label {{
                color: {'#FAFAFA' if st.session_state.dark_mode else '#1a1a1a'} !important;
                font-weight: 500 !important;
                visibility: visible !important;
                opacity: 1 !important;
            }}
            [data-baseweb="toggle"] label {{
                color: {'#FAFAFA' if st.session_state.dark_mode else '#1a1a1a'} !important;
                visibility: visible !important;
                opacity: 1 !important;
            }}
            .stToggle label {{
                color: {'#FAFAFA' if st.session_state.dark_mode else '#1a1a1a'} !important;
                visibility: visible !important;
                opacity: 1 !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
        dark_mode = st.toggle(
            "Dark Mode",
            value=st.session_state.dark_mode,
            key="theme_toggle"
        )
        if dark_mode != st.session_state.dark_mode:
            st.session_state.dark_mode = dark_mode
            st.rerun()
    
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'data_summary' not in st.session_state:
        st.session_state.data_summary = None
    if 'manual_mode' not in st.session_state:
        st.session_state.manual_mode = False
    
    with st.expander("Data Input", expanded=True):
        input_method = st.radio(
            "Choose input method:",
            ["Manual Entry", "CSV Upload"],
            horizontal=True
        )
        
        if input_method == "Manual Entry":
            st.markdown("**Enter your data below (separated by commas, spaces, or line breaks):**")
            text_input = st.text_area(
                "Data Input",
                height=150,
                placeholder="Example: 1.2, 3.4, 5.6, 7.8\nOr: 1.2 3.4 5.6 7.8"
            )
            
            if st.button("Load Data", type="primary"):
                if text_input:
                    parsed_data = parse_text_input(text_input)
                    cleaned_data = clean_data(parsed_data)
                    
                    if len(cleaned_data) > 0:
                        st.session_state.data = cleaned_data
                        st.session_state.data_summary = get_data_summary(cleaned_data)
                        st.success(f"Successfully loaded {len(cleaned_data)} data points.")
                    else:
                        st.error("No valid numeric data found. Please check your input.")
                else:
                    st.warning("Please enter some data.")
        
        else:
            uploaded_file = st.file_uploader(
                "Upload CSV file",
                type=['csv'],
                help="Upload a CSV file with numeric data"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    
                    if len(numeric_cols) == 0:
                        st.error("No numeric columns found in the CSV file.")
                    elif len(numeric_cols) == 1:
                        selected_col = numeric_cols[0]
                        cleaned_data = clean_data(df[selected_col].values)
                        
                        if len(cleaned_data) > 0:
                            st.session_state.data = cleaned_data
                            st.session_state.data_summary = get_data_summary(cleaned_data)
                            st.success(f"Successfully loaded {len(cleaned_data)} data points from column '{selected_col}'.")
                        else:
                            st.error("No valid numeric data found in the selected column.")
                    else:
                        selected_col = st.selectbox(
                            "Select column to use:",
                            numeric_cols,
                            help="Multiple numeric columns found. Please select one."
                        )
                        
                        if st.button("Load Selected Column", type="primary"):
                            cleaned_data = clean_data(df[selected_col].values)
                            
                            if len(cleaned_data) > 0:
                                st.session_state.data = cleaned_data
                                st.session_state.data_summary = get_data_summary(cleaned_data)
                                st.success(f"Successfully loaded {len(cleaned_data)} data points from column '{selected_col}'.")
                            else:
                                st.error("No valid numeric data found in the selected column.")
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
    
    if st.session_state.data is not None and st.session_state.data_summary is not None:
        st.markdown("---")
        st.markdown("### Data Summary")
        summary = st.session_state.data_summary
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("Count", f"{summary['count']:,}")
        with col2:
            st.metric("Min", f"{summary['min']:.4f}")
        with col3:
            st.metric("Max", f"{summary['max']:.4f}")
        with col4:
            st.metric("Mean", f"{summary['mean']:.4f}")
        with col5:
            st.metric("Std Dev", f"{summary['std']:.4f}")
        with col6:
            st.metric("Median", f"{summary['median']:.4f}")
    
    if st.session_state.data is not None and len(st.session_state.data) > 0:
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_dist = st.selectbox(
                "Select Distribution:",
                list(DISTRIBUTIONS.keys()),
                help="Choose a distribution to fit to your data"
            )
        
        with col2:
            st.session_state.manual_mode = st.checkbox(
                "Manual Fitting Mode",
                value=st.session_state.manual_mode,
                help="Toggle to manually adjust parameters with sliders"
            )
        
        dist_info = DISTRIBUTIONS[selected_dist]
        
        if not st.session_state.manual_mode:
            if st.button("Fit Distribution", type="primary"):
                with st.spinner(f"Fitting {selected_dist} distribution..."):
                    params, param_dict = fit_distribution(st.session_state.data, selected_dist)
                    
                    if params is not None and param_dict is not None:
                        st.session_state.fitted_params = params
                        st.session_state.fitted_param_dict = param_dict
                        st.session_state.fitted_dist = selected_dist
                    else:
                        st.error(f"Failed to fit {selected_dist} distribution.")
            
            if 'fitted_params' in st.session_state and 'fitted_dist' in st.session_state:
                if st.session_state.fitted_dist == selected_dist:
                    st.markdown("### Fitting Results")
                    
                    col_plot, col_metrics = st.columns([2, 1])
                    
                    with col_plot:
                        fig = plot_distribution_fit(
                            st.session_state.data,
                            selected_dist,
                            st.session_state.fitted_params
                        )
                        st.pyplot(fig)
                    
                    with col_metrics:
                        st.markdown("#### Parameters")
                        param_names = dist_info['param_names']
                        params_list = dist_info['params']
                        
                        for i, (param_name, param_key) in enumerate(zip(param_names, params_list)):
                            value = st.session_state.fitted_param_dict.get(param_key, 0.0)
                            st.metric(param_name, f"{value:.6f}")
                        
                        st.markdown("---")
                        st.markdown("#### Fit Quality Metrics")
                        
                        metrics = compute_fit_metrics(
                            st.session_state.data,
                            selected_dist,
                            st.session_state.fitted_params
                        )
                        
                        if metrics:
                            st.metric("MAE", f"{metrics.get('MAE', 0):.6f}")
                            st.metric("Max Error", f"{metrics.get('Max Error', 0):.6f}")
                            st.metric("RMSE", f"{metrics.get('RMSE', 0):.6f}")
                            st.metric("R²", f"{metrics.get('R²', 0):.6f}")
        
        else:
            st.markdown("### Manual Parameter Adjustment")
            
            bounds = dist_info['param_bounds'](st.session_state.data)
            param_names = dist_info['param_names']
            params_list = dist_info['params']
            
            manual_params = {}
            slider_cols = st.columns(len(params_list))
            
            for i, (param_name, param_key) in enumerate(zip(param_names, params_list)):
                with slider_cols[i]:
                    if param_key in bounds:
                        min_val, max_val = bounds[param_key]
                        
                        if param_key == 'loc':
                            default_val = np.mean(st.session_state.data)
                        elif param_key == 'scale':
                            default_val = np.std(st.session_state.data)
                        elif param_key == 'a' or param_key == 'c' or param_key == 's':
                            default_val = 1.0
                        elif param_key == 'b':
                            default_val = 1.0
                        elif param_key == 'df':
                            default_val = 10.0
                        else:
                            default_val = (min_val + max_val) / 2
                        
                        default_val = max(min_val, min(max_val, default_val))
                        
                        manual_params[param_key] = st.slider(
                            param_name,
                            min_value=float(min_val),
                            max_value=float(max_val),
                            value=float(default_val),
                            step=(max_val - min_val) / 1000,
                            key=f"slider_{param_key}_{selected_dist}"
                        )
            
            if manual_params:
                col_plot, col_info = st.columns([2, 1])
                
                with col_plot:
                    param_tuple = tuple([manual_params[p] for p in params_list])
                    fig = plot_distribution_fit(
                        st.session_state.data,
                        selected_dist,
                        param_tuple,
                        manual_params=manual_params
                    )
                    st.pyplot(fig)
                
                with col_info:
                    st.markdown("#### Current Parameters")
                    for param_name, param_key in zip(param_names, params_list):
                        value = manual_params.get(param_key, 0.0)
                        st.metric(param_name, f"{value:.6f}")
                    
                    st.markdown("---")
                    st.markdown("#### Fit Quality Metrics")
                    
                    param_tuple = tuple([manual_params[p] for p in params_list])
                    metrics = compute_fit_metrics(
                        st.session_state.data,
                        selected_dist,
                        param_tuple
                    )
                    
                    if metrics:
                        st.metric("MAE", f"{metrics.get('MAE', 0):.6f}")
                        st.metric("Max Error", f"{metrics.get('Max Error', 0):.6f}")
                        st.metric("RMSE", f"{metrics.get('RMSE', 0):.6f}")
                        st.metric("R²", f"{metrics.get('R²', 0):.6f}")
    
    else:
        st.info("Please load data using the Data Input section above to begin fitting distributions.")
    


if __name__ == "__main__":
    main()


import os
import glob
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Autonomous Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #9CA3AF;
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        background-color: #4F46E5;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        width: 100%;
        border: none;
    }
    .stButton>button:hover {
        background-color: #4338CA;
        color: white;
    }
    .status-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #1E293B;
        border: 1px solid #334155;
        color: #E2E8F0;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

from main import run_analysis_workflow, set_current_data_path

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/combo-chart.png", width=64)
    st.title("Configuration")

    env_groq_key = os.getenv("GROQ_API_KEY", "")
    api_key = st.text_input(
        "Groq API Key",
        value=env_groq_key,
        type="password",
        help="Enter your Groq API key to power CrewAI agents."
    )

    st.markdown(
        "[👉 Get a Free Groq API Key](https://console.groq.com/keys)",
        unsafe_allow_html=True
    )

    st.markdown("---")

    model_choice = st.selectbox(
        "Select LLM Model",
        options=["llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b"],
        index=0,
        help="Choose the Groq LLM model for the autonomous analysis crew."
    )

    st.markdown("---")
    st.markdown("### CrewAI Multi-Agent Team")
    st.markdown("1. 🕵️‍♂️ **Senior Data Analyst**: Schema inspection, stat computation & chart code execution.")
    st.markdown("2. 🔍 **QA Reviewer**: Verification of figures, math, and chart existence.")
    st.markdown("3. ✍️ **Executive Deliverable Writer**: Final client-ready Markdown report generation.")

# Main Panel
st.markdown('<div class="main-title">🤖 Autonomous Data Analyst</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload your raw CSV or Excel file, state your client requirements, and let autonomous AI agents deliver a client-ready report with charts.</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Upload Dataset")
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Supported formats: .csv, .xlsx, .xls"
    )

    if uploaded_file is not None:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"Uploaded `{uploaded_file.name}` successfully!")

        # Display Data Preview
        try:
            if file_ext == ".csv":
                df_preview = pd.read_csv(file_path)
            else:
                df_preview = pd.read_excel(file_path)

            with st.expander("👀 Preview Raw Dataset", expanded=False):
                st.write(f"**Shape:** {df_preview.shape[0]} rows × {df_preview.shape[1]} columns")
                st.dataframe(df_preview.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"Error loading preview: {str(e)}")

with col2:
    st.subheader("2. Client Instructions & Focus")
    client_instructions = st.text_area(
        "Client Requirements (Optional)",
        placeholder="e.g. Focus on sales trends by region, identify top 3 performing product categories, and highlight churn risks...",
        height=180
    )

    st.subheader("3. Execute Analysis")
    start_button = st.button("🚀 Start Autonomous Analysis", use_container_width=True)

# Processing & Execution
if start_button:
    if not api_key:
        st.error("Please provide a valid Groq API Key in the sidebar.")
    elif uploaded_file is None:
        st.error("Please upload a CSV or Excel dataset to proceed.")
    else:
        status_box = st.empty()

        def update_status(msg: str):
            status_box.markdown(f'<div class="status-box">🔄 <b>Status:</b> {msg}</div>', unsafe_allow_html=True)

        try:
            with st.spinner("CrewAI Agents are analyzing data and generating charts..."):
                update_status("Initialising autonomous agent team...")

                temp_file_path = os.path.join("temp_uploads", uploaded_file.name)

                result = run_analysis_workflow(
                    file_path=temp_file_path,
                    client_instructions=client_instructions,
                    api_key=api_key,
                    model_name=model_choice,
                    status_callback=update_status
                )

                update_status("Analysis Complete! Report generated.")
                st.balloons()

        except Exception as e:
            st.error(f"An error occurred during execution: {str(e)}")

# Display Results Section
st.markdown("---")
st.subheader("📊 Deliverables & Results")

report_path = "outputs/final_report.md"
chart_dir = "outputs/charts"

tab_report, tab_charts = st.tabs(["📝 Final Report", "🖼️ Interactive Chart Gallery"])

with tab_report:
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report_content = f.read()

        st.markdown(report_content)

        st.download_button(
            label="📥 Download Final Report (.md)",
            data=report_content,
            file_name="final_report.md",
            mime="text/markdown"
        )
    else:
        st.info("No report generated yet. Upload a dataset and click 'Start Autonomous Analysis'.")

with tab_charts:
    if os.path.exists(chart_dir):
        image_files = [
            os.path.join(chart_dir, f) for f in os.listdir(chart_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
        ]

        if image_files:
            st.write(f"Generated **{len(image_files)}** analytical chart(s):")
            gallery_cols = st.columns(2)
            for idx, img_path in enumerate(image_files):
                col_target = gallery_cols[idx % 2]
                with col_target:
                    st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
        else:
            st.info("No chart images found in `outputs/charts/`.")
    else:
        st.info("Chart gallery will appear here after analysis execution.")

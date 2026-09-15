import os
import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from crewai.tools import tool

# Global variable to keep track of the current dataset path
_CURRENT_DATA_PATH = None

def set_current_data_path(file_path: str):
    global _CURRENT_DATA_PATH
    _CURRENT_DATA_PATH = file_path

def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from CSV or Excel file."""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload a .csv or .xlsx file.")

@tool("Inspect Dataset Schema and Summary Statistics")
def inspect_dataset_schema(file_path: str = "") -> str:
    """
    Inspects the CSV or Excel file to return column names, data types, missing value counts,
    and summary statistics (mean, std, min, max, quantiles).
    """
    target_path = file_path if file_path else _CURRENT_DATA_PATH
    if not target_path or not os.path.exists(target_path):
        return "Error: Data file path not provided or file does not exist."

    try:
        df = load_data(target_path)
        buffer = io.StringIO()

        buffer.write(f"Dataset Overview for {os.path.basename(target_path)}:\n")
        buffer.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n")

        buffer.write("Columns and Data Types:\n")
        for col in df.columns:
            null_count = df[col].isnull().sum()
            buffer.write(f" - {col} ({df[col].dtype}): {null_count} missing values\n")

        buffer.write("\nSummary Statistics (Numeric):\n")
        buffer.write(df.describe().to_string())

        buffer.write("\n\nSample First 5 Rows:\n")
        buffer.write(df.head(5).to_string())

        return buffer.getvalue()
    except Exception as e:
        return f"Error inspecting dataset schema: {str(e)}"

@tool("Execute Python Analytics Code and Generate Chart")
def execute_python_analytics(python_code: str) -> str:
    """
    Executes Python data analysis and visualization code using Pandas, Matplotlib, and Seaborn.
    The code has access to `df` loaded from the current dataset, and module aliases `pd`, `np`, `plt`, `sns`, `os`.
    Any generated plots must be saved using `plt.savefig('outputs/charts/<chart_name>.png')`.
    Returns the print output or execution status.
    """
    target_path = _CURRENT_DATA_PATH
    if not target_path or not os.path.exists(target_path):
        return "Error: Dataset is not set or file does not exist."

    try:
        df = load_data(target_path)
        os.makedirs("outputs/charts", exist_ok=True)

        output_buffer = io.StringIO()

        # Local execution namespace
        import numpy as np
        exec_globals = {
            "pd": pd,
            "np": np,
            "plt": plt,
            "sns": sns,
            "os": os,
            "df": df,
            "print": lambda *args, **kwargs: print(*args, file=output_buffer, **kwargs)
        }

        # Clear matplotlib figure state before running
        plt.clf()
        plt.close('all')

        # Run code
        exec(python_code, exec_globals)

        # Ensure matplotlib figures are closed after execution
        plt.close('all')

        printed_output = output_buffer.getvalue().strip()
        chart_files = os.listdir("outputs/charts")
        charts_str = ", ".join(chart_files) if chart_files else "None"

        result_msg = "Python Analytics Execution Succeeded.\n"
        if printed_output:
            result_msg += f"Console Output:\n{printed_output}\n"
        result_msg += f"Saved Charts in outputs/charts/: {charts_str}"
        return result_msg
    except Exception as e:
        plt.close('all')
        return f"Execution Error: {str(e)}"

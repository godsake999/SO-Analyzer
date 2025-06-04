# app.py
import streamlit as st
import pandas as pd
import time
import os
import queue
import threading
from typing import Tuple, Any

# Import the analysis functions from the engine file
from billing_analysis_engine import analyze_and_highlight_billing

# Configure Streamlit page
st.set_page_config(
    page_title="Monthly Billing Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create a progress queue to communicate between threads
progress_queue = queue.Queue()

def run_analysis_in_thread(df: pd.DataFrame, top_n: int, progress_queue: queue.Queue) -> None:
    """
    Wrapper function to run the analysis in a separate thread,
    sending progress updates to the main thread via the queue.
    """
    try:
        result_df = analyze_and_highlight_billing(df, top_n=top_n, progress_queue=progress_queue)
        progress_queue.put(('complete', result_df))
    except Exception as e:
        progress_queue.put(('error', str(e)))

def display_progress(progress_bar: st.delta_generator.DeltaGenerator, status_text: st.delta_generator.DeltaGenerator) -> pd.DataFrame:
    """
    Displays progress updates received from the queue.
    """
    while True:
        try:
            message = progress_queue.get(timeout=0.1) # Short timeout to avoid blocking indefinitely
            if message[0] == 'progress':
                current, total, desc = message[1]
                progress = current / total
                progress_bar.progress(min(progress, 1.0))
                status_text.text(f"{desc} - {int(progress*100)}%")
            elif message[0] == 'complete':
                progress_bar.progress(1.0)
                status_text.text("Analysis complete!")
                return message[1] # Return the result DataFrame
            elif message[0] == 'error':
                raise Exception(message[1])
            progress_queue.task_done()
        except queue.Empty:
            continue # No message, continue waiting

def main():
    st.title("📊 Monthly Billing Difference Analyzer")
    st.markdown("Upload your monthly billing data (containing two consecutive months) to analyze changes.")

    # Initialize session state for persistent results
    if 'analysis_results_df' not in st.session_state:
        st.session_state.analysis_results_df = None
        st.session_state.file_processed = False
        st.session_state.timestamp = str(int(time.time())) # Unique identifier for temp files

    # File Upload
    with st.expander("📁 Upload Billing Data", expanded=True):
        uploaded_file = st.file_uploader(
            "Upload your Excel (.xlsx) or CSV (.csv) file",
            type=["xlsx", "csv"],
            key="billing_file_uploader"
        )
        
        # Display file info if uploaded
        if uploaded_file is not None and not st.session_state.file_processed:
            st.info(f"File uploaded: **{uploaded_file.name}**")

    # Parameters
    with st.expander("⚙️ Analysis Parameters", expanded=True if st.session_state.analysis_results_df is None else False):
        top_n = st.slider(
            "Number of Top Increases/Decreases to Highlight (Top N)",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            key="top_n_slider"
        )

    # Run Analysis Button
    if st.button("🚀 Run Analysis", type="primary", key="run_analysis_button"):
        if uploaded_file is None:
            st.error("Please upload a file to start the analysis!")
            return

        # Read the uploaded file into a DataFrame
        try:
            if uploaded_file.name.endswith('.xlsx'):
                df_to_analyze = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.csv'):
                df_to_analyze = pd.read_csv(uploaded_file)
            else:
                st.error("Unsupported file type. Please upload an .xlsx or .csv file.")
                return
        except Exception as e:
            st.error(f"Error reading file: {e}. Please ensure it's a valid Excel or CSV file.")
            return

        # Initialize progress display
        progress_bar = st.progress(0)
        status_text = st.empty()
        result_placeholder = st.empty()

        # Start analysis in a separate thread
        analysis_thread = threading.Thread(
            target=run_analysis_in_thread,
            args=(df_to_analyze, top_n, progress_queue)
        )
        analysis_thread.start()

        # Display progress in the main thread and wait for completion
        try:
            final_result_df = display_progress(progress_bar, status_text)
            analysis_thread.join() # Wait for the thread to finish

            if 'Error' in final_result_df.columns:
                st.error(f"Analysis Error: {final_result_df['Error'].iloc[0]}")
                st.session_state.analysis_results_df = None # Clear results on error
                st.session_state.file_processed = False
            else:
                st.session_state.analysis_results_df = final_result_df
                st.session_state.file_processed = True
                result_placeholder.success("✅ Analysis completed successfully!")
        except Exception as e:
            st.error(f"❌ An error occurred during analysis: {e}")
            st.session_state.analysis_results_df = None
            st.session_state.file_processed = False

    # Display Results and Download Button
    if st.session_state.analysis_results_df is not None and st.session_state.file_processed:
        st.subheader("Analysis Results")

        # Display key metrics
        total_records = len(st.session_state.analysis_results_df)
        total_customers = st.session_state.analysis_results_df['Customer Code'].nunique()
        ranked_customers = st.session_state.analysis_results_df[st.session_state.analysis_results_df['Customer Ranking'] != '']['Customer Code'].nunique()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Analyzed Records", total_records)
        col2.metric("Unique Customer Codes", total_customers)
        col3.metric("Ranked Customers", ranked_customers)

        st.dataframe(st.session_state.analysis_results_df)

        # Download Buttons
        st.markdown("---")
        st.subheader("Download Results")
        col_dl_csv, col_dl_xlsx = st.columns(2)

        # Download as CSV
        col_dl_csv.download_button(
            label="📥 Download Results as CSV",
            data=st.session_state.analysis_results_df.to_csv(index=False).encode('utf-8'),
            file_name=f"billing_analysis_results_{st.session_state.timestamp}.csv",
            mime="text/csv",
            key="download_csv_button"
        )

        # Download as XLSX
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            st.session_state.analysis_results_df.to_excel(writer, index=False, sheet_name='Analysis Results')
        processed_data = output.getvalue()

        col_dl_xlsx.download_button(
            label="📥 Download Results as XLSX",
            data=processed_data,
            file_name=f"billing_analysis_results_{st.session_state.timestamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_xlsx_button"
        )

        st.markdown("---")
        # Clear Results Button
        if st.button("🔄 Clear Results and Reset", key="clear_results_button"):
            st.session_state.analysis_results_df = None
            st.session_state.file_processed = False
            st.session_state.timestamp = str(int(time.time()))
            st.rerun() # Rerun to clear the UI

if __name__ == "__main__":
    # Ensure 'io' module is imported for BytesIO
    import io 
    main()
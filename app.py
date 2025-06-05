# Enhanced and styled version of app.py with improved visuals
# Unified chart styles, removed ugly background and made them consistent

import streamlit as st
import pandas as pd
import time
import os
import queue
import threading
from typing import Tuple, Any, Dict
import io
import altair as alt
from billing_analysis_engine import analyze_and_highlight_billing

st.set_page_config(page_title="Monthly Billing Analyzer", layout="wide", initial_sidebar_state="expanded")

progress_queue = queue.Queue()

def run_analysis_in_thread(df: pd.DataFrame, top_n: int, progress_queue: queue.Queue) -> None:
    try:
        result_df, revenue_summary = analyze_and_highlight_billing(df, top_n=top_n, progress_queue=progress_queue)
        progress_queue.put(('complete', result_df, revenue_summary))
    except Exception as e:
        progress_queue.put(('error', str(e)))

def display_progress(progress_bar: st.delta_generator.DeltaGenerator, status_text: st.delta_generator.DeltaGenerator) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    while True:
        try:
            message = progress_queue.get(timeout=0.1)
            if message[0] == 'progress':
                current, total, desc = message[1]
                progress = current / total
                progress_bar.progress(min(progress, 1.0))
                status_text.text(f"{desc} - {int(progress*100)}%")
            elif message[0] == 'complete':
                progress_bar.progress(1.0)
                status_text.text("Analysis complete!")
                return message[1], message[2]
            elif message[0] == 'error':
                raise Exception(message[1])
            progress_queue.task_done()
        except queue.Empty:
            continue

def highlight_negative(v):
    return 'color: red;' if v < 0 else ''

def main():
    st.title("\U0001F4C8 Monthly Billing Difference Analyzer")
    st.markdown("Upload your monthly billing data (containing two consecutive months) to analyze changes.")

    if 'analysis_results_df' not in st.session_state:
        st.session_state.analysis_results_df = None
        st.session_state.monthly_revenue_summary = None
        st.session_state.file_processed = False
        st.session_state.timestamp = str(int(time.time()))

    with st.expander("\U0001F4C1 Upload Billing Data", expanded=True):
        uploaded_file = st.file_uploader("Upload your Excel (.xlsx) or CSV (.csv) file", type=["xlsx", "csv"], key="billing_file_uploader")
        if uploaded_file is not None and not st.session_state.file_processed:
            st.info(f"File uploaded: **{uploaded_file.name}**")

    with st.expander("\u2699\ufe0f Analysis Parameters", expanded=True if st.session_state.analysis_results_df is None else False):
        top_n = st.slider("Number of Top Increases/Decreases to Highlight (Top N)", min_value=1, max_value=50, value=10, step=1, key="top_n_slider")

    if st.button("\U0001F680 Run Analysis", type="primary", key="run_analysis_button"):
        if uploaded_file is None:
            st.error("Please upload a file to start the analysis!")
            return

        try:
            df_to_analyze = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {e}.")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()
        result_placeholder = st.empty()

        analysis_thread = threading.Thread(target=run_analysis_in_thread, args=(df_to_analyze, top_n, progress_queue))
        analysis_thread.start()

        try:
            final_result_df, final_revenue_summary = display_progress(progress_bar, status_text)
            analysis_thread.join()

            if 'Error' in final_result_df.columns:
                st.error(f"Analysis Error: {final_result_df['Error'].iloc[0]}")
                st.session_state.analysis_results_df = None
            else:
                st.session_state.analysis_results_df = final_result_df
                st.session_state.monthly_revenue_summary = final_revenue_summary
                st.session_state.file_processed = True
                result_placeholder.success("\u2705 Analysis completed successfully!")
        except Exception as e:
            st.error(f"\u274C An error occurred during analysis: {e}")
            st.session_state.analysis_results_df = None
            st.session_state.monthly_revenue_summary = None
            st.session_state.file_processed = False

    if st.session_state.analysis_results_df is not None and st.session_state.file_processed:
        st.markdown("---")
        st.subheader("\U0001F4CA Analysis Results")

        st.dataframe(st.session_state.analysis_results_df, use_container_width=True)
        
        # Top 10 Customers by Revenue Contribution
        st.markdown("---")
        st.subheader("🏆 Top Customers Caused Revenue Changes")

        top_customers_df = st.session_state.analysis_results_df
        revenue_cols = [col for col in top_customers_df.columns if col.startswith('Total Revenue')]

        if revenue_cols:
            top_customers_df = top_customers_df[['Customer Code', 'Customer Ranking'] + revenue_cols]
            top_customers_df = top_customers_df[top_customers_df['Customer Ranking'].notna()]
            top_customers_df = top_customers_df[top_customers_df['Customer Ranking'].notnull()]
            top_customers_df = top_customers_df[top_customers_df['Customer Ranking'].astype(str).str.strip() != '']

            try:
                top_customers_summary = (
                    top_customers_df
                    .groupby(['Customer Ranking','Customer Code'], as_index=False)
                    .agg({revenue_cols[0]: 'sum', revenue_cols[1]: 'sum'})
                )
                st.dataframe(top_customers_summary)
            except Exception as e:
                st.error(f"Error during summary creation: {e}")
        else:
            st.error("No column starting with 'Total Revenue' was found.")

        st.markdown("---")
        st.subheader("\U0001F4C8 Monthly Revenue Trends")

        summary = st.session_state.monthly_revenue_summary
        older_month = summary['older_month_name']
        recent_month = summary['recent_month_name']
        older_total_rev = summary['older_total_revenue']
        recent_total_rev = summary['recent_total_revenue']
        total_rev_diff = recent_total_rev - older_total_rev

        with st.container():
            col_trend1, col_trend2, col_trend3 = st.columns(3)
            col_trend1.metric(f"\U0001F4B8 Revenue ({older_month})", f"{older_total_rev:,.0f}")
            col_trend2.metric(f"\U0001F4B8 Revenue ({recent_month})", f"{recent_total_rev:,.0f}")
            delta_color = "inverse" if total_rev_diff < 0 else "normal"
            col_trend3.metric("\U0001F4C9 Overall Difference", f"{total_rev_diff:,.0f}", delta=f"{total_rev_diff:,.0f}", delta_color=delta_color)

        st.markdown("### \U0001F5C3 Revenue Grouped by Change Type")

        revenue_by_type_data = []
        all_change_types = sorted(list(set(summary['older_revenue_by_change_type'].keys()) | set(summary['recent_revenue_by_change_type'].keys())))

        for c_type in all_change_types:
            older_rev = summary['older_revenue_by_change_type'].get(c_type, 0)
            recent_rev = summary['recent_revenue_by_change_type'].get(c_type, 0)
            diff_rev = recent_rev - older_rev
            revenue_by_type_data.append({
                'Change Type': c_type,
                f'Revenue ({older_month})': older_rev,
                f'Revenue ({recent_month})': recent_rev,
                'Difference': diff_rev
            })

        revenue_by_type_df = pd.DataFrame(revenue_by_type_data)
        styled_df = revenue_by_type_df.style.applymap(highlight_negative, subset=["Difference"])

        num_rows = revenue_by_type_df.shape[0]
        row_height = 35
        st.dataframe(styled_df, height=row_height * (num_rows + 1), use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Monthly Revenue by Change Type**")
            base_chart = alt.Chart(revenue_by_type_df).transform_fold([
                f"Revenue ({older_month})",
                f"Revenue ({recent_month})"
            ], as_=['Month', 'Value']).mark_bar().encode(
                x=alt.X('Change Type:N', title=None),
                y=alt.Y('Value:Q'),
                color=alt.Color('Month:N', scale=alt.Scale(range=['#90CAF9', '#1565C0']), legend=alt.Legend(orient='bottom'))
            ).properties(height=400)
            st.altair_chart(base_chart, use_container_width=True)

        with col2:
            st.markdown("**Revenue Difference by Change Type**")
            chart = alt.Chart(revenue_by_type_df).mark_bar(
                cornerRadiusTopLeft=4, cornerRadiusTopRight=4
            ).encode(
                x=alt.X('Change Type:N'),
                y=alt.Y('Difference:Q'),
                color=alt.condition(
                    alt.datum.Difference >= 0,
                    alt.value('#4CAF50'),
                    alt.value('#F44336')
                ),
                tooltip=['Change Type', 'Difference']
            ).properties(
                height=400
            ).configure_view(
                stroke=None
            )
            st.altair_chart(chart, use_container_width=True)

        st.markdown("---")
        st.subheader("\U0001F4E5 Download Results")
        col_dl_csv, col_dl_xlsx = st.columns(2)

        col_dl_csv.download_button("\U0001F4BE Download CSV", data=st.session_state.analysis_results_df.to_csv(index=False).encode('utf-8'), file_name=f"billing_analysis_results_{st.session_state.timestamp}.csv", mime="text/csv")

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            st.session_state.analysis_results_df.to_excel(writer, index=False, sheet_name='Analysis Results')
        processed_data = output.getvalue()

        col_dl_xlsx.download_button("\U0001F4BE Download XLSX", data=processed_data, file_name=f"billing_analysis_results_{st.session_state.timestamp}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        st.markdown("---")
        if st.button("\U0001F504 Clear Results and Reset", key="clear_results_button"):
            st.session_state.analysis_results_df = None
            st.session_state.monthly_revenue_summary = None
            st.session_state.file_processed = False
            st.session_state.timestamp = str(int(time.time()))
            st.rerun()

if __name__ == "__main__":
    main()

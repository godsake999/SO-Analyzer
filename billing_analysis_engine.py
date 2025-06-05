import pandas as pd
from scipy.optimize import linear_sum_assignment
import numpy as np
from typing import Optional, Tuple, Dict, Any
import queue

# Function to analyze monthly billing, with the REVISED Change Type logic
def analyze_monthly_billing_v3(df: pd.DataFrame, progress_queue: Optional[queue.Queue] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Analyzes monthly billing data, determines changes between two consecutive months,
    and returns a detailed DataFrame along with monthly revenue summaries.
    """
    EXCLUDED_TAGS = ["SLA", "Credit Note", "Spill Bill", "Quarterly/Yearly Bill"]
    df['Date'] = pd.to_datetime(df['Month Name'] + '-' + df['Year2'].astype(str), format='%b-%Y')
    u_dates = sorted(df['Date'].unique())
    if len(u_dates) != 2:
        return pd.DataFrame({"Error": ["Expected 2 months, found {}.".format(len(u_dates))]}), {}

    o_m_d = u_dates[0]
    r_m_d = u_dates[1]
    df_o = df[df['Date'] == o_m_d].copy()
    df_r = df[df['Date'] == r_m_d].copy()

    o_m_n = df_o['Month Name'].iloc[0] if not df_o.empty else 'N/A'
    o_y = str(df_o['Year2'].iloc[0]) if not df_o.empty else 'N/A'
    r_m_n = df_r['Month Name'].iloc[0] if not df_r.empty else 'N/A'
    r_y = str(df_r['Year2'].iloc[0]) if not df_r.empty else 'N/A'

    o_f_m_n = f"{o_m_n} {o_y}"
    r_f_m_n = f"{r_m_n} {r_y}"
    f_sum_list = []
    all_cust_codes = pd.concat([df_o['Customer Code'], df_r['Customer Code']]).unique()

    # Initialize monthly revenue summaries
    monthly_revenue_summary = {
        'older_month_name': o_f_m_n,
        'recent_month_name': r_f_m_n,
        'older_total_revenue': df_o['Total Revenue'].sum(),
        'recent_total_revenue': df_r['Total Revenue'].sum(),
        'older_revenue_by_change_type': {},
        'recent_revenue_by_change_type': {}
    }

    # Aggregate revenue for excluded tags for initial monthly summary
    for tag in EXCLUDED_TAGS:
        older_tag_revenue = df_o[df_o['Analysis Remark'] == tag]['Total Revenue'].sum()
        recent_tag_revenue = df_r[df_r['Analysis Remark'] == tag]['Total Revenue'].sum()
        if older_tag_revenue > 0:
            monthly_revenue_summary['older_revenue_by_change_type'][tag] = older_tag_revenue
        if recent_tag_revenue > 0:
            monthly_revenue_summary['recent_revenue_by_change_type'][tag] = recent_tag_revenue


    for c_code in all_cust_codes:
        cust_o_data = df_o[df_o['Customer Code'] == c_code].copy()
        cust_r_data = df_r[df_r['Customer Code'] == c_code].copy()

        o_excl = cust_o_data[cust_o_data['Analysis Remark'].isin(EXCLUDED_TAGS)].copy()
        o_incl = cust_o_data[~cust_o_data['Analysis Remark'].isin(EXCLUDED_TAGS)].copy()
        r_excl = cust_r_data[cust_r_data['Analysis Remark'].isin(EXCLUDED_TAGS)].copy()
        r_incl = cust_r_data[~cust_r_data['Analysis Remark'].isin(EXCLUDED_TAGS)].copy()

        for idx, row in o_excl.iterrows():
            f_sum_list.append({
                'Customer Code': c_code,
                f'SO No. ({o_f_m_n})': row['SO No.'], f'SO No. ({r_f_m_n})': 'N/A',
                f'Total Revenue ({o_f_m_n})': row['Total Revenue'], f'Total Revenue ({r_f_m_n})': 0,
                f'NetSale(USD) ({o_f_m_n})': row.get('NetSale(USD)', 0), f'NetSale(USD) ({r_f_m_n})': 0,
                'Difference (Recent - Older)': 0 - row['Total Revenue'],
                'Change Type': row['Analysis Remark']
            })
        for idx, row in r_excl.iterrows():
            f_sum_list.append({
                'Customer Code': c_code,
                f'SO No. ({o_f_m_n})': 'N/A', f'SO No. ({r_f_m_n})': row['SO No.'],
                f'Total Revenue ({o_f_m_n})': 0, f'Total Revenue ({r_f_m_n})': row['Total Revenue'],
                f'NetSale(USD) ({o_f_m_n})': 0, f'NetSale(USD) ({r_f_m_n})': row.get('NetSale(USD)', 0),
                'Difference (Recent - Older)': row['Total Revenue'] - 0,
                'Change Type': row['Analysis Remark']
            })

        if 'NetSale(USD)' not in o_incl.columns: o_incl['NetSale(USD)'] = 0
        if 'NetSale(USD)' not in r_incl.columns: r_incl['NetSale(USD)'] = 0

        o_incl_amts = o_incl[['SO No.', 'Total Revenue', 'NetSale(USD)']].values.tolist()
        r_incl_amts = r_incl[['SO No.', 'Total Revenue', 'NetSale(USD)']].values.tolist()

        if not o_incl_amts and not r_incl_amts:
            continue
        elif not o_incl_amts and r_incl_amts:
            for r_so, r_mmk, r_usd in r_incl_amts:
                f_sum_list.append({
                    'Customer Code': c_code,
                    f'SO No. ({o_f_m_n})': 'N/A', f'SO No. ({r_f_m_n})': r_so,
                    f'Total Revenue ({o_f_m_n})': 0, f'Total Revenue ({r_f_m_n})': r_mmk,
                    f'NetSale(USD) ({o_f_m_n})': 0, f'NetSale(USD) ({r_f_m_n})': r_usd,
                    'Difference (Recent - Older)': r_mmk - 0,
                    'Change Type': 'New Bill in ' + r_f_m_n
                })
        elif o_incl_amts and not r_incl_amts:
            for o_so, o_mmk, o_usd in o_incl_amts:
                f_sum_list.append({
                    'Customer Code': c_code,
                    f'SO No. ({o_f_m_n})': o_so, f'SO No. ({r_f_m_n})': 'N/A',
                    f'Total Revenue ({o_f_m_n})': o_mmk, f'Total Revenue ({r_f_m_n})': 0,
                    f'NetSale(USD) ({o_f_m_n})': o_usd, f'NetSale(USD) ({r_f_m_n})': 0,
                    'Difference (Recent - Older)': 0 - o_mmk,
                    'Change Type': 'Missing in ' + r_f_m_n
                })
        else:
            cost_matrix = np.zeros((len(o_incl_amts), len(r_incl_amts)))
            for i, (o_so, o_mmk, o_usd_val) in enumerate(o_incl_amts): # renamed o_usd to o_usd_val for clarity
                for j, (r_so, r_mmk, r_usd_val) in enumerate(r_incl_amts): # renamed r_usd to r_usd_val
                    cost = 0.0
                    if o_usd_val == r_usd_val:
                        cost = abs(o_mmk - r_mmk) * 0.001
                    else:
                        cost = abs(o_usd_val - r_usd_val) + 1000000
                    cost_matrix[i, j] = cost

            r_ind, c_ind = linear_sum_assignment(cost_matrix)
            m_o_idx = set()
            m_r_idx = set()

            for i, j in zip(r_ind, c_ind):
                o_so, o_mmk, o_usd = o_incl_amts[i] # Keep original variable names here for assignment
                r_so, r_mmk, r_usd = r_incl_amts[j]
                diff_mmk = r_mmk - o_mmk
                c_type = ""

                # --- REVISED Change Type Logic ---
                if o_usd == r_usd:
                    # If USD amounts are the same, classify as "Same".
                    # This means changes in MMK (Increase/Decrease) for the same USD base
                    # will also fall into this "Same" category, aligning with original script's implied aggregation.
                    # The true "Same" (where MMK also doesn't change) is a subset of this.
                    c_type = "Same"
                else: # o_usd != r_usd
                    if diff_mmk == 0:
                        # This is the specific case to isolate: MMK is same, but USD changed.
                        c_type = "Same" # Still 'Same' from MMK perspective, but USD changed
                    elif diff_mmk > 0:
                        c_type = "Increase" # USD changed, and MMK increased.
                    elif diff_mmk < 0:
                        c_type = "Decrease" # USD changed, and MMK decreased.
                # --- End of REVISED Change Type Logic ---

                f_sum_list.append({
                    'Customer Code': c_code,
                    f'SO No. ({o_f_m_n})': o_so, f'SO No. ({r_f_m_n})': r_so,
                    f'Total Revenue ({o_f_m_n})': o_mmk, f'Total Revenue ({r_f_m_n})': r_mmk,
                    f'NetSale(USD) ({o_f_m_n})': o_usd, f'NetSale(USD) ({r_f_m_n})': r_usd,
                    'Difference (Recent - Older)': diff_mmk,
                    'Change Type': c_type
                })
                m_o_idx.add(i)
                m_r_idx.add(j)

            for i_idx, (o_so, o_mmk, o_usd) in enumerate(o_incl_amts): # Renamed i to i_idx
                if i_idx not in m_o_idx:
                    f_sum_list.append({
                        'Customer Code': c_code,
                        f'SO No. ({o_f_m_n})': o_so, f'SO No. ({r_f_m_n})': 'N/A',
                        f'Total Revenue ({o_f_m_n})': o_mmk, f'Total Revenue ({r_f_m_n})': 0,
                        f'NetSale(USD) ({o_f_m_n})': o_usd, f'NetSale(USD) ({r_f_m_n})': 0,
                        'Difference (Recent - Older)': 0 - o_mmk,
                        'Change Type': 'Missing in ' + r_f_m_n
                    })

            for i_idx, (r_so, r_mmk, r_usd) in enumerate(r_incl_amts): # Renamed i to i_idx
                if i_idx not in m_r_idx:
                    f_sum_list.append({
                        'Customer Code': c_code,
                        f'SO No. ({o_f_m_n})': 'N/A', f'SO No. ({r_f_m_n})': r_so,
                        f'Total Revenue ({o_f_m_n})': 0, f'Total Revenue ({r_f_m_n})': r_mmk,
                        f'NetSale(USD) ({o_f_m_n})': 0, f'NetSale(USD) ({r_f_m_n})': r_usd,
                        'Difference (Recent - Older)': r_mmk - 0,
                        'Change Type': 'New Bill in ' + r_f_m_n
                    })

    sum_df = pd.DataFrame(f_sum_list)
    d_cols = [
        'Customer Code',
        f'SO No. ({o_f_m_n})', f'SO No. ({r_f_m_n})',
        f'Total Revenue ({o_f_m_n})', f'Total Revenue ({r_f_m_n})',
        f'NetSale(USD) ({o_f_m_n})', f'NetSale(USD) ({r_f_m_n})',
        'Difference (Recent - Older)', 'Change Type'
    ]
    for col in d_cols:
        if col not in sum_df.columns:
            sum_df[col] = np.nan
    sum_df = sum_df.reindex(columns=d_cols, fill_value=np.nan)

    # Calculate revenue by change type after the sum_df is finalized
    # This ensures "New Bill" and "Missing" are also included
    for change_type in sum_df['Change Type'].unique():
        older_rev = sum_df[sum_df['Change Type'] == change_type][f'Total Revenue ({o_f_m_n})'].sum()
        recent_rev = sum_df[sum_df['Change Type'] == change_type][f'Total Revenue ({r_f_m_n})'].sum()
        if older_rev > 0:
            monthly_revenue_summary['older_revenue_by_change_type'][change_type] = older_rev
        if recent_rev > 0:
            monthly_revenue_summary['recent_revenue_by_change_type'][change_type] = recent_rev

    return sum_df, monthly_revenue_summary

# Your analyze_and_highlight_billing function can then use this updated function:
def analyze_and_highlight_billing(df: pd.DataFrame, top_n: int = 10, progress_queue: Optional[queue.Queue] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Analyzes monthly billing and adds a 'Customer Ranking' column based on the magnitude of change.
    Returns the analyzed DataFrame and monthly revenue summaries.
    """
    # Call the updated analysis function
    analysis_df, monthly_revenue_summary = analyze_monthly_billing_v3(df, progress_queue)

    if 'Error' in analysis_df.columns: # Check if an error DataFrame was returned
        return analysis_df, {}

    # Initialize 'Customer Ranking' column as empty strings
    analysis_df['Customer Ranking'] = ''

    # Calculate the total absolute difference per customer code for ranking
    customer_diff_magnitude = analysis_df.groupby('Customer Code')[
        'Difference (Recent - Older)' # Ranking is still based on MMK difference
    ].apply(lambda x: x.abs().sum()).sort_values(ascending=False)

    top_n_customer_codes = customer_diff_magnitude.head(top_n).index.tolist()

    for rank, cust_code in enumerate(top_n_customer_codes):
        analysis_df.loc[analysis_df['Customer Code'] == cust_code, 'Customer Ranking'] = f'Top {rank + 1}'
    
    # Ensure 'Customer Ranking' is the last column
    cols = [col for col in analysis_df.columns if col != 'Customer Ranking'] + ['Customer Ranking']
    analysis_df = analysis_df[cols]

    return analysis_df, monthly_revenue_summary
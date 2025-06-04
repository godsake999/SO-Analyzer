import pandas as pd
from scipy.optimize import linear_sum_assignment
import numpy as np
from typing import Optional
import queue

# Function to analyze monthly billing, with the corrected Change Type logic
def analyze_monthly_billing_v3(df: pd.DataFrame, progress_queue: Optional[queue.Queue] = None) -> pd.DataFrame:
    EXCLUDED_TAGS = ["SLA", "Credit Note", "Spill Bill", "Quarterly/Yearly Bill"]
    df['Date'] = pd.to_datetime(df['Month Name'] + '-' + df['Year2'].astype(str), format='%b-%Y')
    u_dates = sorted(df['Date'].unique())
    if len(u_dates) != 2:
        return pd.DataFrame({"Error": ["Expected 2 months, found {}.".format(len(u_dates))]})

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

    for c_code in all_cust_codes:
        cust_o_data = df_o[df_o['Customer Code'] == c_code].copy()
        cust_r_data = df_r[df_r['Customer Code'] == c_code].copy()

        o_excl = cust_o_data[cust_o_data['Analysis Remark'].isin(EXCLUDED_TAGS)].copy()
        o_incl = cust_o_data[~cust_o_data['Analysis Remark'].isin(EXCLUDED_TAGS)].copy()
        r_excl = cust_r_data[cust_r_data['Analysis Remark'].isin(EXCLUDED_TAGS)].copy()
        r_incl = cust_r_data[~cust_r_data['Analysis Remark'].isin(EXCLUDED_TAGS)].copy()

        for idx, row in o_excl.iterrows():
            f_sum_list.append({
                'Customer Code': c_code, f'SO No. ({o_f_m_n})': row['SO No.'], f'SO No. ({r_f_m_n})': 'N/A',
                f'Total Revenue ({o_f_m_n})': row['Total Revenue'], f'Total Revenue ({r_f_m_n})': 0,
                'Difference (Recent - Older)': 0 - row['Total Revenue'], 'Change Type': row['Analysis Remark']
            })
        for idx, row in r_excl.iterrows():
            f_sum_list.append({
                'Customer Code': c_code, f'SO No. ({o_f_m_n})': 'N/A', f'SO No. ({r_f_m_n})': row['SO No.'],
                f'Total Revenue ({o_f_m_n})': 0, f'Total Revenue ({r_f_m_n})': row['Total Revenue'],
                'Difference (Recent - Older)': row['Total Revenue'] - 0, 'Change Type': row['Analysis Remark']
            })

        o_incl_amts = o_incl[['SO No.', 'Total Revenue', 'NetSale(USD)']].values.tolist()
        r_incl_amts = r_incl[['SO No.', 'Total Revenue', 'NetSale(USD)']].values.tolist()

        if not o_incl_amts and not r_incl_amts:
            continue
        elif not o_incl_amts and r_incl_amts:
            for r_so, r_mmk, r_usd in r_incl_amts:
                f_sum_list.append({
                    'Customer Code': c_code, f'SO No. ({o_f_m_n})': 'N/A', f'SO No. ({r_f_m_n})': r_so,
                    f'Total Revenue ({o_f_m_n})': 0, f'Total Revenue ({r_f_m_n})': r_mmk,
                    'Difference (Recent - Older)': r_mmk - 0, 'Change Type': 'New Bill in ' + r_f_m_n
                })
        elif o_incl_amts and not r_incl_amts:
            for o_so, o_mmk, o_usd in o_incl_amts:
                f_sum_list.append({
                    'Customer Code': c_code, f'SO No. ({o_f_m_n})': o_so, f'SO No. ({r_f_m_n})': 'N/A',
                    f'Total Revenue ({o_f_m_n})': o_mmk, f'Total Revenue ({r_f_m_n})': 0,
                    'Difference (Recent - Older)': 0 - o_mmk, 'Change Type': 'Missing in ' + r_f_m_n
                })
        else:
            cost_matrix = np.zeros((len(o_incl_amts), len(r_incl_amts)))
            for i, (o_so, o_mmk, o_usd) in enumerate(o_incl_amts):
                for j, (r_so, r_mmk, r_usd) in enumerate(r_incl_amts):
                    cost = 0.0
                    if o_usd == r_usd:
                        cost = abs(o_mmk - r_mmk) * 0.001
                    else:
                        cost = abs(o_usd - r_usd) + 1000000
                    cost_matrix[i, j] = cost

            r_ind, c_ind = linear_sum_assignment(cost_matrix)
            m_o_idx = set()
            m_r_idx = set()

            for i, j in zip(r_ind, c_ind):
                o_so, o_mmk, o_usd = o_incl_amts[i]
                r_so, r_mmk, r_usd = r_incl_amts[j]
                diff_mmk = r_mmk - o_mmk
                if o_usd == r_usd:
                    if diff_mmk > 0:
                        c_type = "Increase"
                    elif diff_mmk < 0:
                        c_type = "Decrease"
                    else:
                        c_type = "Same"
                else:
                    if diff_mmk > 0:
                        c_type = "Increase"
                    elif diff_mmk < 0:
                        c_type = "Decrease"
                    else:
                        c_type = "Same"

                f_sum_list.append({
                    'Customer Code': c_code, f'SO No. ({o_f_m_n})': o_so, f'SO No. ({r_f_m_n})': r_so,
                    f'Total Revenue ({o_f_m_n})': o_mmk, f'Total Revenue ({r_f_m_n})': r_mmk,
                    'Difference (Recent - Older)': diff_mmk, 'Change Type': c_type
                })
                m_o_idx.add(i)
                m_r_idx.add(j)

            for i, (o_so, o_mmk, o_usd) in enumerate(o_incl_amts):
                if i not in m_o_idx:
                    f_sum_list.append({
                        'Customer Code': c_code, f'SO No. ({o_f_m_n})': o_so, f'SO No. ({r_f_m_n})': 'N/A',
                        f'Total Revenue ({o_f_m_n})': o_mmk, f'Total Revenue ({r_f_m_n})': 0,
                        'Difference (Recent - Older)': 0 - o_mmk, 'Change Type': 'Missing in ' + r_f_m_n
                    })

            for i, (r_so, r_mmk, r_usd) in enumerate(r_incl_amts):
                if i not in m_r_idx:
                    f_sum_list.append({
                        'Customer Code': c_code, f'SO No. ({o_f_m_n})': 'N/A', f'SO No. ({r_f_m_n})': r_so,
                        f'Total Revenue ({o_f_m_n})': 0, f'Total Revenue ({r_f_m_n})': r_mmk,
                        'Difference (Recent - Older)': r_mmk - 0, 'Change Type': 'New Bill in ' + r_f_m_n
                    })

    sum_df = pd.DataFrame(f_sum_list)
    d_cols = [
        'Customer Code', f'SO No. ({o_f_m_n})', f'SO No. ({r_f_m_n})',
        f'Total Revenue ({o_f_m_n})', f'Total Revenue ({r_f_m_n})',
        'Difference (Recent - Older)', 'Change Type'
    ]
    return sum_df.reindex(columns=d_cols, fill_value=np.nan)

# This function now correctly calculates and adds the 'Customer Ranking' column
def analyze_and_highlight_billing(df: pd.DataFrame, top_n: int = 10, progress_queue: Optional[queue.Queue] = None) -> pd.DataFrame:
    """
    Analyzes monthly billing and adds a 'Customer Ranking' column based on the magnitude of change.
    """
    analysis_df = analyze_monthly_billing_v3(df, progress_queue)

    # Initialize 'Customer Ranking' column as empty strings
    analysis_df['Customer Ranking'] = ''

    # Calculate the total absolute difference per customer code for ranking
    # Group by 'Customer Code' and sum the absolute differences in 'Total Revenue'
    customer_diff_magnitude = analysis_df.groupby('Customer Code')[
        'Difference (Recent - Older)'
    ].apply(lambda x: x.abs().sum()).sort_values(ascending=False)

    # Get the top N customer codes based on the magnitude of change
    top_n_customer_codes = customer_diff_magnitude.head(top_n).index.tolist()

    # Assign ranks to the top N customers in the analysis_df
    # We iterate through the sorted list of top N customer codes
    for rank, cust_code in enumerate(top_n_customer_codes):
        # Locate rows belonging to the current customer code and assign their rank
        analysis_df.loc[analysis_df['Customer Code'] == cust_code, 'Customer Ranking'] = f'Top {rank + 1}'

    return analysis_df
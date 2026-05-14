import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from streamlit import runtime
from streamlit.web import cli as stcli

# ==========================================
# 1. DATA ENGINE & COLOR CONFIG
# ==========================================
COLOR_MAP = {
    'Crossword': '#00e5ff', 'Bingo': '#df20df', 'Spin the Wheel': '#f1c40f',
    'Race 6': '#10d25f', 'Spin Roulette': '#38b284', 'Crossword Paradise': '#f06277',
    'Terdrup': '#8b5cf6', 'Pick 3': '#3b82f6', 'Lotto': '#a855f7',
    'Pick 4': '#f97316', 'Free Roll': '#0ea5e9'
}

DARK_TEMPLATE = dict(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(
        color='#F8FAFC',          # Crisp, highly visible off-white (much better than harsh pure white)
        size=14,                  # Slightly larger for better readability
        family="Sora, sans-serif" # Modern, clean dashboard font
    )
)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Online sales1.csv')
    except FileNotFoundError:
        return pd.DataFrame(), []

    product_cols = list(COLOR_MAP.keys())

    cols_to_fix = product_cols + ['Wagers/sales']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(',', '').str.replace('"', ''),
                errors='coerce'
            )

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
    df = df.dropna(subset=['Date']).sort_values('Date')

    df['Month']     = df['Date'].dt.month_name()
    df['MonthNum']  = df['Date'].dt.month
    df['Week']      = df['Date'].dt.isocalendar().week.astype(int)
    df['DayOfWeek'] = df['Date'].dt.day_name()
    df['Year']      = df['Date'].dt.year
    df['Total Sales'] = df[product_cols].sum(axis=1)

    return df, product_cols

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def growth_badge(val):
    if val > 0:
        return f"+{val:.1f}%"
    elif val < 0:
        return f"{val:.1f}%"
    return "0.0%"

def compute_mom_growth(df, product_cols):
    monthly = df.groupby(['Year', 'MonthNum'])[product_cols + ['Total Sales']].sum().reset_index()
    monthly = monthly.sort_values(['Year', 'MonthNum'])
    if len(monthly) < 2:
        return None, None
    last = monthly.iloc[-1]
    prev = monthly.iloc[-2]
    growth = {}
    for col in product_cols + ['Total Sales']:
        if prev[col] != 0:
            growth[col] = ((last[col] - prev[col]) / prev[col]) * 100
        else:
            growth[col] = 0.0
    last_label = f"{int(last['Year'])} M{int(last['MonthNum'])}"
    prev_label = f"{int(prev['Year'])} M{int(prev['MonthNum'])}"
    return growth, (prev_label, last_label)

def get_product_stats(df, product_cols):
    stats = []
    total_all = df[product_cols].sum().sum()
    for p in product_cols:
        s = df[p].dropna()
        total    = s.sum()
        avg      = s.mean()
        peak     = s.max()
        peak_date = df.loc[df[p] == peak, 'Date'].values
        peak_date_str = str(pd.to_datetime(peak_date[0]).date()) if len(peak_date) > 0 else "N/A"
        share = (total / total_all * 100) if total_all > 0 else 0
        stats.append({
            'Product': p,
            'Total Revenue': total,
            'Avg Daily': avg,
            'Peak Day': peak,
            'Peak Date': peak_date_str,
            'Market Share %': share,
            'Color': COLOR_MAP[p]
        })
    return pd.DataFrame(stats).sort_values('Total Revenue', ascending=False).reset_index(drop=True)

# ==========================================
# 3. MAIN DASHBOARD
# ==========================================
def main():
    st.set_page_config(page_title="AI Sales Dashboard", layout="wide", page_icon="chart_with_upwards_trend")

    # ---- Custom CSS ----
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

    .metric-card {
        background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 8px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    }
    .metric-card .label { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 4px; }
    .metric-card .value { font-size: 26px; font-weight: 700; color: #f1f5f9; }
    .metric-card .delta-pos { font-size: 12px; color: #10b981; margin-top: 2px; }
    .metric-card .delta-neg { font-size: 12px; color: #ef4444; margin-top: 2px; }
    .metric-card .delta-neu { font-size: 12px; color: #64748b; margin-top: 2px; }

    .product-scorecard {
        background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 100%);
        border-radius: 12px;
        padding: 16px;
        border-left: 4px solid var(--accent);
        margin-bottom: 10px;
        border-top: 1px solid rgba(255,255,255,0.06);
        border-right: 1px solid rgba(255,255,255,0.06);
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .scorecard-name  { font-size: 13px; font-weight: 700; color: #f1f5f9; margin-bottom: 6px; }
    .scorecard-total { font-size: 20px; font-weight: 700; color: #f1f5f9; }
    .scorecard-sub   { font-size: 11px; color: #64748b; margin-top: 2px; }
    .scorecard-share { font-size: 12px; font-weight: 600; color: #94a3b8; margin-top: 4px; }
    .rank-badge      { display:inline-block; background: rgba(255,255,255,0.08); border-radius:6px; padding: 2px 8px; font-size:10px; color:#94a3b8; margin-right:6px; }

    .section-header {
        font-size: 20px; font-weight: 700; color: #f1f5f9;
        margin: 24px 0 16px 0; padding-bottom: 8px;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }

    div[data-testid="stTabs"] button {
        font-family: 'Sora', sans-serif !important;
        font-size: 13px !important;
        font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    df, product_cols = load_data()

    if df.empty:
        st.error("Data file not found. Please ensure 'Online sales1.csv' is available.")
        return

    # ==========================================
    # SIDEBAR — GLOBAL FILTERS
    # ==========================================
    with st.sidebar:
        st.markdown("### Filters")
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        if len(date_range) == 2:
            start_date, end_date = date_range
            df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]

        selected_products = st.multiselect(
            "Focus Products",
            options=product_cols,
            default=product_cols,
            help="Select products to include in all charts"
        )
        if not selected_products:
            selected_products = product_cols

        st.markdown("---")
        st.markdown("### Quick Stats")
        st.caption(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        st.caption(f"Total days: {df['Date'].nunique()}")
        st.caption(f"Products tracked: {len(selected_products)}")

    df_filtered = df.copy()
    active_cols  = [p for p in product_cols if p in selected_products]

    # ==========================================
    # HEADER
    # ==========================================
    st.markdown("""
    <div style='padding: 10px 0 20px 0;'>
        <div style='font-size:28px; font-weight:700; color:#f1f5f9; letter-spacing:-0.5px;'>
            AI Sales Intelligence Dashboard
        </div>
        <div style='font-size:13px; color:#64748b; margin-top:4px;'>
            Real-time product performance analysis across all game categories
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ==========================================
    # TOP KPI ROW
    # ==========================================
    mom_growth, mom_labels = compute_mom_growth(df_filtered, active_cols)
    total_rev    = df_filtered[active_cols].sum().sum()
    best_product = df_filtered[active_cols].sum().idxmax()
    worst_product= df_filtered[active_cols].sum().idxmin()
    avg_daily    = df_filtered['Total Sales'].mean()
    active_days  = df_filtered['Date'].nunique()
    total_mom    = mom_growth['Total Sales'] if mom_growth else 0

    def delta_class(val):
        if val > 0: return "delta-pos"
        if val < 0: return "delta-neg"
        return "delta-neu"

    def delta_arrow(val):
        if val > 0: return "&#8593;"
        if val < 0: return "&#8595;"
        return "-"

    c1, c2, c3, c4, c5 = st.columns(5)
    kpi_data = [
        (c1, "Total Revenue",   f"${total_rev:,.0f}",       total_mom, "MoM"),
        (c2, "Best Performer",  best_product,               None,      f"${df_filtered[best_product].sum():,.0f} total"),
        (c3, "Avg Daily Revenue", f"${avg_daily:,.0f}",     None,      f"Over {active_days} days"),
        (c4, "Needs Attention", worst_product,              None,      f"${df_filtered[worst_product].sum():,.0f} total"),
        (c5, "Products Tracked", str(len(active_cols)),     None,      f"{active_cols[0]} leading"),
    ]
    for col, label, value, delta, sub in kpi_data:
        if delta is not None:
            dc    = delta_class(delta)
            arrow = delta_arrow(delta)
            delta_html = f'<div class="{dc}">{arrow} {growth_badge(delta)} {sub}</div>'
        else:
            delta_html = f'<div class="delta-neu">{sub}</div>'
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ==========================================
    # TABS
    # ==========================================
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "All Products Performance",
        "Performance Overview",
        "Trends & Time Analysis",
        "Market Intelligence",
        "AI Search & Analyze",
        "Raw Data"
    ])

    # ==========================================
    # TAB 1 — ALL PRODUCTS PERFORMANCE
    # ==========================================
    with tab1:
        st.markdown('<div class="section-header">Individual Product Scorecards</div>', unsafe_allow_html=True)

        stats_df = get_product_stats(df_filtered, active_cols)

        rows = [active_cols[i:i+3] for i in range(0, len(active_cols), 3)]
        for row in rows:
            cols_row = st.columns(3)
            for idx, product in enumerate(row):
                p_stats  = stats_df[stats_df['Product'] == product].iloc[0]
                rank_val = stats_df.index[stats_df['Product'] == product].tolist()[0] + 1
                color    = COLOR_MAP[product]
                mom_val  = mom_growth[product] if mom_growth and product in mom_growth else 0
                mom_str  = growth_badge(mom_val)
                mom_color= "#10b981" if mom_val >= 0 else "#ef4444"

                with cols_row[idx]:
                    st.markdown(f"""
                    <div class="product-scorecard" style="--accent: {color};">
                        <div class="scorecard-name">
                            <span class="rank-badge">#{rank_val}</span>{product}
                        </div>
                        <div class="scorecard-total">${p_stats['Total Revenue']:,.0f}</div>
                        <div class="scorecard-sub">Avg/Day: ${p_stats['Avg Daily']:,.0f}</div>
                        <div class="scorecard-sub">Peak: ${p_stats['Peak Day']:,.0f} on {p_stats['Peak Date']}</div>
                        <div class="scorecard-share">
                            Market Share: {p_stats['Market Share %']:.1f}%
                            &nbsp;|&nbsp;
                            <span style='color:{mom_color};'>MoM {mom_str}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # Revenue Ranking & Market Share
        st.markdown('<div class="section-header">Revenue Ranking & Market Share</div>', unsafe_allow_html=True)
        col_rank, col_pie = st.columns([3, 2])

        with col_rank:
            totals = df_filtered[active_cols].sum().sort_values(ascending=True).reset_index()
            totals.columns = ['Product', 'Sales']
            fig_rank = go.Figure(go.Bar(
                x=totals['Sales'], y=totals['Product'], orientation='h',
                marker=dict(color=[COLOR_MAP.get(p) for p in totals['Product']], opacity=0.9, line=dict(width=0)),
                text=totals['Sales'], texttemplate='$%{text:,.0f}', textposition='outside',
                textfont=dict(size=11)
            ))
            fig_rank.update_layout(
                title="Total Revenue Ranking", **DARK_TEMPLATE, height=420,
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False),
                margin=dict(l=10, r=100, t=40, b=10)
            )
            st.plotly_chart(fig_rank, use_container_width=True)

        with col_pie:
            pie_data = df_filtered[active_cols].sum().reset_index()
            pie_data.columns = ['Product', 'Sales']
            fig_pie = px.pie(
                pie_data, names='Product', values='Sales',
                color='Product', color_discrete_map=COLOR_MAP,
                hole=0.5, title="Revenue Market Share"
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label', textfont_size=10)
            fig_pie.update_layout(**DARK_TEMPLATE, height=420, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

        # Deep-Dive: Individual Product Daily Performance
        st.markdown('<div class="section-header">Individual Product Deep Dive</div>', unsafe_allow_html=True)
        selected_deep = st.selectbox("Select a product to deep dive", options=active_cols)

        if selected_deep:
            deep_df = df_filtered[['Date', selected_deep]].copy()
            deep_df['7DMA'] = deep_df[selected_deep].rolling(7).mean()
            deep_df['30DMA'] = deep_df[selected_deep].rolling(30).mean()
            p_color = COLOR_MAP[selected_deep]

            fig_deep = go.Figure()
            fig_deep.add_trace(go.Bar(
                x=deep_df['Date'], y=deep_df[selected_deep],
                name='Daily Revenue', marker_color=p_color, opacity=0.5
            ))
            fig_deep.add_trace(go.Scatter(
                x=deep_df['Date'], y=deep_df['7DMA'],
                name='7-Day MA', line=dict(color='#f1c40f', width=2)
            ))
            fig_deep.add_trace(go.Scatter(
                x=deep_df['Date'], y=deep_df['30DMA'],
                name='30-Day MA', line=dict(color='#ef4444', width=2, dash='dot')
            ))
            fig_deep.update_layout(
                title=f"{selected_deep} — Daily Revenue + Moving Averages",
                **DARK_TEMPLATE, height=380,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
                legend=dict(orientation='h', y=1.08),
                margin=dict(l=10, r=10, t=60, b=10)
            )
            st.plotly_chart(fig_deep, use_container_width=True)

            # Stats summary row for selected product
            d_stats = stats_df[stats_df['Product'] == selected_deep].iloc[0]
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Total Revenue",  f"${d_stats['Total Revenue']:,.0f}")
            s2.metric("Avg Daily",      f"${d_stats['Avg Daily']:,.0f}")
            s3.metric("Peak Single Day",f"${d_stats['Peak Day']:,.0f}")
            s4.metric("Market Share",   f"{d_stats['Market Share %']:.2f}%")

    # ==========================================
    # TAB 2 — PERFORMANCE OVERVIEW
    # ==========================================
    with tab2:
        st.markdown('<div class="section-header">Overall Performance Snapshot</div>', unsafe_allow_html=True)

        # Stacked area chart - all products over time
        fig_area = go.Figure()
        for p in active_cols:
            fig_area.add_trace(go.Scatter(
                x=df_filtered['Date'], y=df_filtered[p],
                name=p, stackgroup='one',
                line=dict(width=0.5, color=COLOR_MAP[p]),
                fillcolor=COLOR_MAP[p],
                opacity=0.75
            ))
        fig_area.update_layout(
            title="Cumulative Daily Revenue — All Products (Stacked)",
            **DARK_TEMPLATE, height=400,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            legend=dict(orientation='h', y=-0.15),
            margin=dict(l=10, r=10, t=40, b=60)
        )
        st.plotly_chart(fig_area, use_container_width=True)

        # Side-by-side: grouped monthly bar & top-N table
        col_bar, col_tbl = st.columns([2, 1])

        with col_bar:
            month_order = ['January','February','March','April','May','June',
                           'July','August','September','October','November','December']
            monthly = df_filtered.groupby('Month')[active_cols].sum().reset_index()
            monthly['SortKey'] = monthly['Month'].apply(
                lambda x: month_order.index(x) if x in month_order else 99
            )
            monthly = monthly.sort_values('SortKey').drop(columns='SortKey')
            melted  = monthly.melt(id_vars='Month', var_name='Product', value_name='Sales')
            fig_monthly = px.bar(
                melted, x='Month', y='Sales', color='Product',
                barmode='group', title="Monthly Revenue per Product",
                color_discrete_map=COLOR_MAP
            )
            fig_monthly.update_layout(**DARK_TEMPLATE, height=380, legend=dict(orientation='h', y=-0.2),
                                      margin=dict(l=10, r=10, t=40, b=80))
            st.plotly_chart(fig_monthly, use_container_width=True)

        with col_tbl:
            st.markdown("**Product Revenue Summary**")
            summary_tbl = stats_df[['Product','Total Revenue','Market Share %','Avg Daily']].copy()
            summary_tbl['Total Revenue']   = summary_tbl['Total Revenue'].apply(lambda x: f"${x:,.0f}")
            summary_tbl['Market Share %']  = summary_tbl['Market Share %'].apply(lambda x: f"{x:.1f}%")
            summary_tbl['Avg Daily']        = summary_tbl['Avg Daily'].apply(lambda x: f"${x:,.0f}")
            st.dataframe(summary_tbl, use_container_width=True, hide_index=True)

    # ==========================================
    # TAB 3 — TRENDS & TIME ANALYSIS
    # ==========================================
    with tab3:
        st.markdown('<div class="section-header">Trends & Time-Series Analysis</div>', unsafe_allow_html=True)

        t_col1, t_col2 = st.columns(2)

        with t_col1:
            # 7-day moving average — total
            df_filtered['7DMA_Total']  = df_filtered['Total Sales'].rolling(7).mean()
            df_filtered['30DMA_Total'] = df_filtered['Total Sales'].rolling(30).mean()

            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=df_filtered['Date'], y=df_filtered['Total Sales'],
                name='Daily Total', mode='lines',
                line=dict(color='rgba(255,255,255,0.2)', width=1)
            ))
            fig_trend.add_trace(go.Scatter(
                x=df_filtered['Date'], y=df_filtered['7DMA_Total'],
                name='7-Day MA', line=dict(color='#00e5ff', width=2)
            ))
            fig_trend.add_trace(go.Scatter(
                x=df_filtered['Date'], y=df_filtered['30DMA_Total'],
                name='30-Day MA', line=dict(color='#f1c40f', width=2, dash='dash')
            ))
            fig_trend.update_layout(
                title="Total Revenue Trend",
                **DARK_TEMPLATE, height=350,
                legend=dict(orientation='h', y=1.1),
                margin=dict(l=10, r=10, t=50, b=10)
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        with t_col2:
            # Day-of-week heatmap
            dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
            dow_data  = df_filtered.groupby('DayOfWeek')[active_cols].sum().reindex(dow_order).reset_index()
            dow_melted= dow_data.melt(id_vars='DayOfWeek', var_name='Product', value_name='Sales')
            fig_dow = px.bar(
                dow_melted, x='DayOfWeek', y='Sales', color='Product',
                barmode='stack', title="Revenue by Day of Week",
                color_discrete_map=COLOR_MAP
            )
            fig_dow.update_layout(
                **DARK_TEMPLATE, height=350,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
                legend=dict(orientation='h', y=-0.25),
                margin=dict(l=10, r=10, t=40, b=80)
            )
            st.plotly_chart(fig_dow, use_container_width=True)

        # Correlation Heatmap
        st.markdown('<div class="section-header">Product Correlation Heatmap</div>', unsafe_allow_html=True)
        corr = df_filtered[active_cols].corr()
        fig_corr = px.imshow(
            corr, text_auto=".2f",
            color_continuous_scale='RdBu_r',
            title="Product Sales Correlation Matrix",
            aspect='auto'
        )
        fig_corr.update_layout(**DARK_TEMPLATE, height=450, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_corr, use_container_width=True)

        # Weekly Rolling Revenue
        st.markdown('<div class="section-header">Weekly Rolling Revenue per Product</div>', unsafe_allow_html=True)
        weekly = df_filtered.groupby(['Year','Week'])[active_cols].sum().reset_index()
        weekly['PeriodLabel'] = weekly['Year'].astype(str) + '-W' + weekly['Week'].astype(str).str.zfill(2)
        melted_w = weekly.melt(id_vars='PeriodLabel', value_vars=active_cols, var_name='Product', value_name='Sales')
        fig_weekly = px.line(
            melted_w, x='PeriodLabel', y='Sales', color='Product',
            title="Weekly Revenue Breakdown",
            color_discrete_map=COLOR_MAP
        )
        fig_weekly.update_layout(
            **DARK_TEMPLATE, height=380,
            xaxis=dict(showgrid=False, tickangle=-45),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            legend=dict(orientation='h', y=-0.3),
            margin=dict(l=10, r=10, t=40, b=100)
        )
        st.plotly_chart(fig_weekly, use_container_width=True)

    # ==========================================
    # TAB 4 — MARKET INTELLIGENCE
    # ==========================================
    with tab4:
        st.markdown('<div class="section-header">Competitive Market Intelligence</div>', unsafe_allow_html=True)

        mi_col1, mi_col2 = st.columns(2)

        with mi_col1:
            # Winner each month
            monthly_by_product = df_filtered.groupby('Month')[active_cols].sum()
            monthly_winner      = monthly_by_product.idxmax(axis=1).reset_index()
            monthly_winner.columns = ['Month','Top Product']
            monthly_winner['Revenue'] = [
                monthly_by_product.loc[r['Month'], r['Top Product']]
                for _, r in monthly_winner.iterrows()
            ]
            fig_winners = px.bar(
                monthly_winner, x='Month', y='Revenue',
                color='Top Product', text='Top Product',
                title="Monthly Champion Product",
                color_discrete_map=COLOR_MAP
            )
            fig_winners.update_layout(**DARK_TEMPLATE, height=370, showlegend=False,
                                      margin=dict(l=10, r=10, t=40, b=60))
            st.plotly_chart(fig_winners, use_container_width=True)

        with mi_col2:
            # Revenue concentration (Pareto analysis)
            sorted_rev = df_filtered[active_cols].sum().sort_values(ascending=False)
            cumulative = (sorted_rev.cumsum() / sorted_rev.sum() * 100).reset_index()
            cumulative.columns = ['Product', 'Cumulative %']
            cumulative['Revenue'] = sorted_rev.values

            fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
            fig_pareto.add_trace(
                go.Bar(x=cumulative['Product'], y=cumulative['Revenue'],
                       name='Revenue', marker_color=[COLOR_MAP.get(p,'#fff') for p in cumulative['Product']]),
                secondary_y=False
            )
            fig_pareto.add_trace(
                go.Scatter(x=cumulative['Product'], y=cumulative['Cumulative %'],
                           name='Cumulative %', line=dict(color='#f1c40f', width=2)),
                secondary_y=True
            )
            fig_pareto.update_layout(
                title="Pareto Revenue Analysis",
                **DARK_TEMPLATE, height=370,
                legend=dict(orientation='h', y=1.12),
                margin=dict(l=10, r=10, t=50, b=60)
            )
            fig_pareto.update_yaxes(title_text="Revenue", secondary_y=False)
            fig_pareto.update_yaxes(title_text="Cumulative %", secondary_y=True, range=[0, 110])
            st.plotly_chart(fig_pareto, use_container_width=True)

        # MoM Growth Table
        st.markdown('<div class="section-header">Month-over-Month Growth Table</div>', unsafe_allow_html=True)
        if mom_growth and mom_labels:
            growth_rows = []
            for p in active_cols:
                g = mom_growth.get(p, 0)
                growth_rows.append({
                    'Product': p,
                    f'Revenue ({mom_labels[0]})': f"${df_filtered[df_filtered['MonthNum'] == int(mom_labels[0].split('M')[1])][p].sum():,.0f}" if mom_labels else "N/A",
                    f'Revenue ({mom_labels[1]})': f"${df_filtered[df_filtered['MonthNum'] == int(mom_labels[1].split('M')[1])][p].sum():,.0f}" if mom_labels else "N/A",
                    'MoM Growth': growth_badge(g),
                    'Direction': "UP" if g > 0 else ("DOWN" if g < 0 else "FLAT")
                })
            growth_table = pd.DataFrame(growth_rows)
            st.dataframe(growth_table, use_container_width=True, hide_index=True)
        else:
            st.info("Not enough monthly data to compute MoM growth.")

        # Scatter: Avg Daily vs Total Revenue (bubble = market share)
        st.markdown('<div class="section-header">Efficiency Matrix — Avg Daily vs Total Revenue</div>', unsafe_allow_html=True)
        fig_scatter = px.scatter(
            stats_df,
            x='Avg Daily', y='Total Revenue',
            size='Market Share %', color='Product',
            text='Product',
            title="Product Efficiency Matrix (bubble = market share)",
            color_discrete_map={r['Product']: r['Color'] for _, r in stats_df.iterrows()}
        )
        fig_scatter.update_traces(textposition='top center', textfont_size=10)
        fig_scatter.update_layout(
            **DARK_TEMPLATE, height=420, showlegend=False,
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ==========================================
    # TAB 5 — AI SEARCH & ANALYZE
    # ==========================================
    with tab5:
        st.markdown('<div class="section-header">AI Search & Analyze</div>', unsafe_allow_html=True)
        st.markdown("Ask me anything about your sales data. Try: *'Compare Lotto and Bingo'*, *'Show monthly'*, *'Trend analysis'*, *'Risk'*, *'Best day'*")

        user_query = st.text_input(
            "Your question:",
            placeholder="e.g., Compare Lotto and Bingo | Show monthly breakdown | Trend analysis | Worst performers"
        )

        if user_query:
            q = user_query.lower()
            st.write("### Insight Result")

            # CASE 1: COMPARISON
            if "compare" in q or " vs " in q:
                found_products = [p for p in active_cols if p.lower() in q]
                if len(found_products) >= 2:
                    p1, p2 = found_products[0], found_products[1]
                    diff   = df_filtered[p1].sum() - df_filtered[p2].sum()
                    winner = p1 if diff > 0 else p2
                    st.info(f"Comparison: {p1} vs {p2}. **{winner}** leads by **${abs(diff):,.0f}**.")
                    fig_c = px.line(df_filtered, x='Date', y=[p1, p2],
                                    title=f"Timeline: {p1} vs {p2}",
                                    color_discrete_map={p1: COLOR_MAP[p1], p2: COLOR_MAP[p2]})
                    fig_c.update_layout(**DARK_TEMPLATE)
                    st.plotly_chart(fig_c, use_container_width=True)

                    # Extra stats
                    a1, a2 = st.columns(2)
                    a1.metric(p1, f"${df_filtered[p1].sum():,.0f}", f"Avg/day ${df_filtered[p1].mean():,.0f}")
                    a2.metric(p2, f"${df_filtered[p2].sum():,.0f}", f"Avg/day ${df_filtered[p2].mean():,.0f}")
                else:
                    st.warning("Mention at least two valid product names (e.g., 'Compare Lotto and Bingo').")

            # CASE 2: MONTHLY
            elif "month" in q:
                month_order = ['January','February','March','April','May','June',
                               'July','August','September','October','November','December']
                month_data = df_filtered.groupby('Month')[active_cols].sum().reset_index()
                month_data['Total'] = month_data[active_cols].sum(axis=1)
                month_data['SortKey'] = month_data['Month'].apply(
                    lambda x: month_order.index(x) if x in month_order else 99
                )
                month_data = month_data.sort_values('SortKey').drop(columns='SortKey')
                best_month = month_data.loc[month_data['Total'].idxmax(), 'Month']
                st.success(f"Your strongest month overall is **{best_month}**.")
                melted = month_data.drop(columns='Total').melt(id_vars='Month')
                fig_m = px.bar(melted, x='Month', y='value', color='variable',
                               barmode='group', title="Monthly Product Breakdown",
                               color_discrete_map=COLOR_MAP)
                fig_m.update_layout(**DARK_TEMPLATE)
                st.plotly_chart(fig_m, use_container_width=True)

            # CASE 3: TREND / MOVING AVERAGE
            elif "trend" in q or "moving average" in q:
                df_filtered['7DMA'] = df_filtered['Total Sales'].rolling(7).mean()
                st.info("Showing 7-day moving average to smooth out daily fluctuations.")
                fig_t = px.scatter(df_filtered, x='Date', y='Total Sales',
                                   title="Sales Trend & 7-Day Moving Average",
                                   opacity=0.4)
                fig_t.add_trace(go.Scatter(
                    x=df_filtered['Date'], y=df_filtered['7DMA'],
                    name='7-Day Avg', line=dict(color='limegreen', width=3)
                ))
                fig_t.update_layout(**DARK_TEMPLATE)
                st.plotly_chart(fig_t, use_container_width=True)

            # CASE 4: RISK / WORST
            elif "risk" in q or "worst" in q or "low" in q or "weak" in q:
                bottom_3 = df_filtered[active_cols].sum().sort_values().head(3)
                st.warning(f"Low performers: **{', '.join(bottom_3.index)}**")
                fig_risk = px.pie(
                    names=bottom_3.index, values=bottom_3.values, hole=0.4,
                    title="Revenue Share of Lowest Performers",
                    color_discrete_sequence=['#ff4b4b','#ff7676','#ffb1b1']
                )
                st.plotly_chart(fig_risk, use_container_width=True)

            # CASE 5: BEST DAY
            elif "best day" in q or "top day" in q or "highest day" in q:
                best_row = df_filtered.loc[df_filtered['Total Sales'].idxmax()]
                st.success(f"Best single day: **{str(best_row['Date'].date())}** with **${best_row['Total Sales']:,.0f}** in total sales.")
                best_detail = best_row[active_cols].sort_values(ascending=False).reset_index()
                best_detail.columns = ['Product', 'Revenue']
                fig_bd = px.bar(best_detail, x='Product', y='Revenue',
                                title=f"Revenue Breakdown on {str(best_row['Date'].date())}",
                                color='Product', color_discrete_map=COLOR_MAP)
                fig_bd.update_layout(**DARK_TEMPLATE)
                st.plotly_chart(fig_bd, use_container_width=True)

            # CASE 6: SPECIFIC PRODUCT QUERY
            elif any(p.lower() in q for p in active_cols):
                found = [p for p in active_cols if p.lower() in q][0]
                p_stats = stats_df[stats_df['Product'] == found].iloc[0]
                st.info(f"Showing full analysis for **{found}**")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Revenue",  f"${p_stats['Total Revenue']:,.0f}")
                m2.metric("Avg Daily",      f"${p_stats['Avg Daily']:,.0f}")
                m3.metric("Peak Day",       f"${p_stats['Peak Day']:,.0f}")
                m4.metric("Market Share",   f"{p_stats['Market Share %']:.2f}%")
                fig_sp = px.line(df_filtered, x='Date', y=found, title=f"{found} Daily Revenue",
                                 color_discrete_sequence=[COLOR_MAP[found]])
                fig_sp.update_layout(**DARK_TEMPLATE)
                st.plotly_chart(fig_sp, use_container_width=True)
             # CASE 7: OVERALL PRODUCT TRENDS
            elif "overall" in q or "all product" in q or "product trend" in q or "every product" in q:
                st.info("Showing individual trend lines for all active products over the selected period.")

                # --- Cumulative revenue per product (area chart) ---
                df_cum = df_filtered[['Date'] + active_cols].copy()
                for col in active_cols:
                    df_cum[col] = df_cum[col].cumsum()

                fig_cum = px.area(
                    df_cum, x='Date', y=active_cols,
                    title="Cumulative Revenue – All Products",
                    color_discrete_map=COLOR_MAP
                )
                fig_cum.update_layout(**DARK_TEMPLATE)
                st.plotly_chart(fig_cum, use_container_width=True)

                # --- 7-Day Moving Average per product (line chart) ---
                df_ma = df_filtered[['Date'] + active_cols].copy()
                for col in active_cols:
                    df_ma[col] = df_ma[col].rolling(7, min_periods=1).mean()

                fig_ma = px.line(
                    df_ma, x='Date', y=active_cols,
                    title="7-Day Moving Average – All Products",
                    color_discrete_map=COLOR_MAP
                )
                fig_ma.update_layout(**DARK_TEMPLATE)
                st.plotly_chart(fig_ma, use_container_width=True)

                # --- Growth %: first vs last 7-day window ---
                growth_data = []
                for col in active_cols:
                    series   = df_filtered[col]
                    early    = series.head(7).mean()
                    recent   = series.tail(7).mean()
                    pct      = ((recent - early) / early * 100) if early else 0
                    growth_data.append({'Product': col, 'Growth %': round(pct, 1)})

                growth_df = pd.DataFrame(growth_data).sort_values('Growth %', ascending=False)

                fig_growth = px.bar(
                    growth_df, x='Product', y='Growth %',
                    title="Period Growth % (First 7 Days vs Last 7 Days)",
                    color='Growth %',
                    color_continuous_scale=['#ff4b4b', '#ffdd57', '#00c896'],
                    text='Growth %'
                )
                fig_growth.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig_growth.update_layout(**DARK_TEMPLATE)
                st.plotly_chart(fig_growth, use_container_width=True)

                # --- Summary metrics row ---
                st.markdown("#### Product-Level Summary")
                cols_ui = st.columns(len(active_cols))
                for i, col in enumerate(active_cols):
                    series  = df_filtered[col]
                    early   = series.head(7).mean()
                    recent  = series.tail(7).mean()
                    delta   = ((recent - early) / early * 100) if early else 0
                    cols_ui[i].metric(
                        label=col,
                        value=f"${series.sum():,.0f}",
                        delta=f"{delta:+.1f}% trend"
                    )

            else:
                st.write("I'm not sure how to answer that yet. Try: **'compare X and Y'**, **'monthly sales'**, **'trend'**, **'risk'**, **'best day'**, or a product name.")

    # ==========================================
    # TAB 6 — RAW DATA
    # ==========================================
    with tab6:
        st.markdown('<div class="section-header">Raw Data Explorer</div>', unsafe_allow_html=True)

        search_term = st.text_input("Filter by date (YYYY-MM-DD) or any value:", placeholder="e.g. 2024-01")
        display_df  = df_filtered.copy()
        if search_term:
            mask = display_df.astype(str).apply(lambda col: col.str.contains(search_term, case=False)).any(axis=1)
            display_df = display_df[mask]

        st.caption(f"Showing {len(display_df):,} rows")
        st.dataframe(display_df, use_container_width=True)

        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name="sales_export.csv",
            mime="text/csv"
        )

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == '__main__':
    if runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())

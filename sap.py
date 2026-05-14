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
# 1. DATA ENGINE & COLOR CONFIG (Premium Palette)
# ==========================================
# Professional, high-contrast dashboard palette
COLOR_MAP = {
    'Crossword': '#6366f1',       # Indigo
    'Bingo': '#ec4899',           # Pink
    'Spin the Wheel': '#f59e0b',  # Amber
    'Race 6': '#10b981',          # Emerald
    'Spin Roulette': '#14b8a6',   # Teal
    'Crossword Paradise': '#f43f5e', # Rose
    'Terdrup': '#8b5cf6',         # Violet
    'Pick 3': '#3b82f6',          # Blue
    'Lotto': '#d946ef',           # Fuchsia
    'Pick 4': '#f97316',          # Orange
    'Free Roll': '#0ea5e9'        # Sky
}

DARK_TEMPLATE = dict(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#f8fafc', size=13, family="Inter, sans-serif") 
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
    st.set_page_config(page_title="Executive Sales Dashboard", layout="wide", page_icon="📈")

    # ---- Premium Dashboard CSS ----
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .metric-card {
        background: #111827; /* Dark slate background */
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.18);
    }
    .metric-card .label { font-size: 13px; color: #9ca3af; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
    .metric-card .value { font-size: 30px; font-weight: 700; color: #f9fafb; margin-bottom: 4px; }
    .metric-card .delta-pos { font-size: 14px; font-weight: 600; color: #10b981; }
    .metric-card .delta-neg { font-size: 14px; font-weight: 600; color: #ef4444; }
    .metric-card .delta-neu { font-size: 14px; font-weight: 600; color: #9ca3af; }

    .product-scorecard {
        background: #111827;
        border-radius: 12px;
        padding: 18px;
        border-left: 6px solid var(--accent);
        margin-bottom: 12px;
        border-top: 1px solid #1f2937;
        border-right: 1px solid #1f2937;
        border-bottom: 1px solid #1f2937;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .scorecard-name  { font-size: 15px; font-weight: 700; color: #f9fafb; margin-bottom: 8px; }
    .scorecard-total { font-size: 24px; font-weight: 700; color: #ffffff; }
    .scorecard-sub   { font-size: 13px; color: #9ca3af; margin-top: 4px; }
    .scorecard-share { font-size: 13px; font-weight: 600; color: #d1d5db; margin-top: 8px; }
    .rank-badge      { display:inline-block; background: #374151; border-radius:6px; padding: 2px 8px; font-size:11px; color:#ffffff; margin-right:8px; font-weight: 700;}

    .section-header {
        font-size: 22px; font-weight: 700; color: #f9fafb;
        margin: 32px 0 20px 0; padding-bottom: 12px;
        border-bottom: 2px solid #1f2937;
    }

    /* Tabs Styling */
    div[data-testid="stTabs"] button {
        font-family: 'Inter', sans-serif !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        color: #9ca3af !important;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #ffffff !important;
        border-bottom-color: #3b82f6 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    df, product_cols = load_data()

    if df.empty:
        st.error("Data file not found. Please ensure 'Online sales1.csv' is available.")
        return

    # ==========================================
    # SIDEBAR
    # ==========================================
    with st.sidebar:
        st.markdown("### Dashboard Controls")
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        date_range = st.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        if len(date_range) == 2:
            start_date, end_date = date_range
            df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]

        selected_products = st.multiselect("Products to Track", options=product_cols, default=product_cols)
        if not selected_products:
            selected_products = product_cols

    df_filtered = df.copy()
    active_cols  = [p for p in product_cols if p in selected_products]

    st.markdown("""
    <div style='padding: 0 0 24px 0;'>
        <div style='font-size:36px; font-weight:800; color:#f8fafc; letter-spacing:-0.5px;'>
            Executive Sales Intelligence
        </div>
        <div style='font-size:15px; color:#9ca3af; margin-top:4px;'>
            Real-time revenue monitoring and product performance analytics
        </div>
    </div>
    """, unsafe_allow_html=True)

    mom_growth, mom_labels = compute_mom_growth(df_filtered, active_cols)
    total_rev    = df_filtered[active_cols].sum().sum()
    best_product = df_filtered[active_cols].sum().idxmax()
    worst_product= df_filtered[active_cols].sum().idxmin()
    avg_daily    = df_filtered['Total Sales'].mean()
    active_days  = df_filtered['Date'].nunique()
    total_mom    = mom_growth['Total Sales'] if mom_growth else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    kpi_data = [
        (c1, "Total Revenue",   f"${total_rev:,.0f}",       total_mom, "MoM"),
        (c2, "Best Performer",  best_product,               None,      f"${df_filtered[best_product].sum():,.0f} total"),
        (c3, "Avg Daily Rev",   f"${avg_daily:,.0f}",       None,      f"Over {active_days} days"),
        (c4, "Lowest Performer",worst_product,              None,      f"${df_filtered[worst_product].sum():,.0f} total"),
        (c5, "Active Products", str(len(active_cols)),      None,      "Tracking"),
    ]
    for col, label, value, delta, sub in kpi_data:
        delta_html = f'<div class="delta-{"pos" if delta and delta>0 else ("neg" if delta and delta<0 else "neu")}">{("&#8593;" if delta and delta>0 else "&#8595;") if delta else "-"} {growth_badge(delta) if delta else ""} <span style="color:#6b7280; font-weight:400;">{sub}</span></div>'
        col.markdown(f'<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div>{delta_html}</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Performance Scorecards", "Revenue Overview", "Trend Analysis",
        "Market Intelligence", "AI Insights", "Raw Data Export"
    ])

    # ==========================================
    # TAB 1 — SCORECARDS & MARKET SHARE
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
                
                with cols_row[idx]:
                    st.markdown(f"""
                    <div class="product-scorecard" style="--accent: {color};">
                        <div class="scorecard-name"><span class="rank-badge">#{rank_val}</span>{product}</div>
                        <div class="scorecard-total">${p_stats['Total Revenue']:,.0f}</div>
                        <div class="scorecard-sub">Avg Daily: <b>${p_stats['Avg Daily']:,.0f}</b></div>
                        <div class="scorecard-sub">Record Day: <b>${p_stats['Peak Day']:,.0f}</b> ({p_stats['Peak Date']})</div>
                        <div class="scorecard-share">Share: {p_stats['Market Share %']:.1f}% &nbsp;|&nbsp; <span style='color:{"#10b981" if mom_val >= 0 else "#ef4444"};'>MoM {growth_badge(mom_val)}</span></div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Revenue Distribution</div>', unsafe_allow_html=True)
        col_rank, col_pie = st.columns([1.2, 1])

        with col_rank:
            totals = df_filtered[active_cols].sum().sort_values(ascending=True).reset_index()
            totals.columns = ['Product', 'Sales']
            
            fig_rank = go.Figure(go.Bar(
                x=totals['Sales'], y=totals['Product'], orientation='h',
                marker=dict(color=[COLOR_MAP.get(p) for p in totals['Product']], line=dict(color='#111827', width=1.5)),
                text=totals['Sales'], texttemplate=' $%{text:,.0f}', textposition='outside',
                textfont=dict(size=13, weight='bold', color='#f8fafc'), cliponaxis=False
            ))
            fig_rank.update_layout(
                title=dict(text="Revenue Ranking", font=dict(size=18, weight='bold')), 
                **DARK_TEMPLATE, height=450,
                xaxis=dict(showgrid=True, gridcolor='#374151', zeroline=False), 
                yaxis=dict(tickfont=dict(size=13, weight='bold')),
                margin=dict(l=10, r=120, t=50, b=10)
            )
            st.plotly_chart(fig_rank, use_container_width=True)

        with col_pie:
            pie_data = df_filtered[active_cols].sum().reset_index()
            pie_data.columns = ['Product', 'Sales']
            
            # PREMIUM DASHBOARD PIE CHART
            fig_pie = go.Figure(data=[go.Pie(
                labels=pie_data['Product'], values=pie_data['Sales'], hole=0.55,
                marker=dict(colors=[COLOR_MAP.get(p) for p in pie_data['Product']], line=dict(color='#111827', width=3)),
                textinfo='label+percent', textposition='outside', # Highly visible outside text
                pull=[0.02] * len(pie_data)
            )])
            fig_pie.update_layout(
                title=dict(text="Market Share", font=dict(size=18, weight='bold')), 
                **DARK_TEMPLATE, height=450, showlegend=False,
                margin=dict(l=50, r=50, t=50, b=50) # Generous margins to fit outside text
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # ==========================================
    # TAB 2 — REVENUE OVERVIEW
    # ==========================================
    with tab2:
        st.markdown('<div class="section-header">Cumulative Performance</div>', unsafe_allow_html=True)
        fig_area = go.Figure()
        for p in active_cols:
            fig_area.add_trace(go.Scatter(
                x=df_filtered['Date'], y=df_filtered[p], name=p, stackgroup='one',
                line=dict(width=2, color=COLOR_MAP[p], shape='spline'),
                fillcolor=COLOR_MAP[p], opacity=0.85
            ))
        fig_area.update_layout(
            title=dict(text="Daily Revenue Mix (Stacked)", font=dict(size=18, weight='bold')),
            **DARK_TEMPLATE, height=450, hovermode="x unified",
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#374151'),
            legend=dict(orientation='h', y=-0.2)
        )
        st.plotly_chart(fig_area, use_container_width=True)

        col_bar, col_tbl = st.columns([2, 1])
        with col_bar:
            monthly = df_filtered.groupby('Month')[active_cols].sum().reset_index()
            melted  = monthly.melt(id_vars='Month', var_name='Product', value_name='Sales')
            fig_monthly = px.bar(
                melted, x='Month', y='Sales', color='Product', barmode='group', 
                title="Monthly Comparison", color_discrete_map=COLOR_MAP
            )
            fig_monthly.update_traces(marker_line=dict(color='#111827', width=1))
            fig_monthly.update_layout(**DARK_TEMPLATE, height=400, hovermode="x unified", legend=dict(orientation='h', y=-0.2), yaxis=dict(gridcolor='#374151'))
            st.plotly_chart(fig_monthly, use_container_width=True)
            
        with col_tbl:
            st.markdown("<div style='font-size:18px; font-weight:700; margin-bottom:15px;'>Summary Table</div>", unsafe_allow_html=True)
            summary_tbl = stats_df[['Product','Total Revenue','Market Share %','Avg Daily']].copy()
            summary_tbl['Total Revenue'] = summary_tbl['Total Revenue'].apply(lambda x: f"${x:,.0f}")
            summary_tbl['Market Share %'] = summary_tbl['Market Share %'].apply(lambda x: f"{x:.1f}%")
            summary_tbl['Avg Daily'] = summary_tbl['Avg Daily'].apply(lambda x: f"${x:,.0f}")
            st.dataframe(summary_tbl, use_container_width=True, hide_index=True)

    # ==========================================
    # TAB 3 — TREND ANALYSIS
    # ==========================================
    with tab3:
        st.markdown('<div class="section-header">Trend & Seasonality Analysis</div>', unsafe_allow_html=True)
        t_col1, t_col2 = st.columns(2)

        with t_col1:
            df_filtered['7DMA_Total']  = df_filtered['Total Sales'].rolling(7).mean()
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=df_filtered['Date'], y=df_filtered['Total Sales'], name='Daily Actuals', 
                mode='lines+markers', marker=dict(size=4), line=dict(color='#4b5563', width=1.5)
            ))
            fig_trend.add_trace(go.Scatter(
                x=df_filtered['Date'], y=df_filtered['7DMA_Total'], name='7-Day Moving Avg', 
                line=dict(color='#3b82f6', width=4, shape='spline')
            ))
            fig_trend.update_layout(
                title=dict(text="Total Revenue Trajectory", font=dict(size=18, weight='bold')), 
                **DARK_TEMPLATE, height=400, hovermode="x unified", legend=dict(orientation='h', y=1.1),
                yaxis=dict(gridcolor='#374151')
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        with t_col2:
            dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
            dow_data  = df_filtered.groupby('DayOfWeek')[active_cols].sum().reindex(dow_order).reset_index()
            dow_melted= dow_data.melt(id_vars='DayOfWeek', var_name='Product', value_name='Sales')
            fig_dow = px.bar(
                dow_melted, x='DayOfWeek', y='Sales', color='Product', barmode='stack', 
                title="Revenue by Day of Week", color_discrete_map=COLOR_MAP
            )
            fig_dow.update_traces(marker_line=dict(color='#111827', width=1.5))
            fig_dow.update_layout(**DARK_TEMPLATE, height=400, hovermode="x unified", legend=dict(orientation='h', y=-0.25), yaxis=dict(gridcolor='#374151'))
            st.plotly_chart(fig_dow, use_container_width=True)

        st.markdown('<div class="section-header">Weekly Product Dynamics</div>', unsafe_allow_html=True)
        weekly = df_filtered.groupby(['Year','Week'])[active_cols].sum().reset_index()
        weekly['PeriodLabel'] = weekly['Year'].astype(str) + '-W' + weekly['Week'].astype(str).str.zfill(2)
        melted_w = weekly.melt(id_vars='PeriodLabel', value_vars=active_cols, var_name='Product', value_name='Sales')
        fig_weekly = px.line(
            melted_w, x='PeriodLabel', y='Sales', color='Product', title="Rolling Weekly Revenue", color_discrete_map=COLOR_MAP
        )
        fig_weekly.update_traces(line=dict(width=3, shape='spline'), mode="lines+markers", marker=dict(size=6, line=dict(color='#111827', width=1)))
        fig_weekly.update_layout(**DARK_TEMPLATE, height=450, hovermode="x unified", legend=dict(orientation='h', y=-0.2), yaxis=dict(gridcolor='#374151'))
        st.plotly_chart(fig_weekly, use_container_width=True)

    # ==========================================
    # TAB 4 — MARKET INTELLIGENCE
    # ==========================================
    with tab4:
        st.markdown('<div class="section-header">Market Intelligence</div>', unsafe_allow_html=True)
        mi_col1, mi_col2 = st.columns(2)

        with mi_col1:
            monthly_by_product = df_filtered.groupby('Month')[active_cols].sum()
            monthly_winner      = monthly_by_product.idxmax(axis=1).reset_index()
            monthly_winner.columns = ['Month','Top Product']
            monthly_winner['Revenue'] = [monthly_by_product.loc[r['Month'], r['Top Product']] for _, r in monthly_winner.iterrows()]
            fig_winners = px.bar(
                monthly_winner, x='Month', y='Revenue', color='Top Product', text='Top Product',
                title="Monthly Top Performers", color_discrete_map=COLOR_MAP
            )
            fig_winners.update_traces(textposition='inside', textfont=dict(size=14, color='white', weight='bold'), marker_line=dict(color='#111827', width=2))
            fig_winners.update_layout(**DARK_TEMPLATE, height=400, showlegend=False, yaxis=dict(gridcolor='#374151'))
            st.plotly_chart(fig_winners, use_container_width=True)

        with mi_col2:
            sorted_rev = df_filtered[active_cols].sum().sort_values(ascending=False)
            cumulative = (sorted_rev.cumsum() / sorted_rev.sum() * 100).reset_index()
            cumulative.columns = ['Product', 'Cumulative %']
            cumulative['Revenue'] = sorted_rev.values
            fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
            fig_pareto.add_trace(go.Bar(
                x=cumulative['Product'], y=cumulative['Revenue'], name='Revenue', 
                marker_color=[COLOR_MAP.get(p,'#fff') for p in cumulative['Product']], marker_line=dict(color='#111827', width=1.5)), secondary_y=False)
            fig_pareto.add_trace(go.Scatter(
                x=cumulative['Product'], y=cumulative['Cumulative %'], name='Cumulative %', 
                line=dict(color='#f9fafb', width=3, shape='spline'), mode='lines+markers', marker=dict(size=8, color='#3b82f6')), secondary_y=True)
            fig_pareto.update_layout(title=dict(text="Pareto Distribution", font=dict(size=18, weight='bold')), **DARK_TEMPLATE, height=400, hovermode="x unified", legend=dict(orientation='h', y=1.12), yaxis=dict(gridcolor='#374151'))
            st.plotly_chart(fig_pareto, use_container_width=True)

        st.markdown('<div class="section-header">Product Efficiency Matrix</div>', unsafe_allow_html=True)
        fig_scatter = px.scatter(
            stats_df, x='Avg Daily', y='Total Revenue', size='Market Share %', color='Product', text='Product',
            title="Total Revenue vs. Average Daily (Bubble Size = Market Share)", color_discrete_map={r['Product']: r['Color'] for _, r in stats_df.iterrows()}
        )
        fig_scatter.update_traces(textposition='top center', textfont=dict(size=13, weight='bold', color='#f8fafc'), marker=dict(line=dict(width=2, color='#111827')))
        fig_scatter.update_layout(**DARK_TEMPLATE, height=500, showlegend=False, xaxis=dict(showgrid=True, gridcolor='#374151'), yaxis=dict(showgrid=True, gridcolor='#374151'))
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ==========================================
    # TAB 5 — AI SEARCH & ANALYZE
    # ==========================================
    with tab5:
        st.markdown('<div class="section-header">Ask AI Analytics</div>', unsafe_allow_html=True)
        st.markdown("<span style='color:#9ca3af;'>Type a command to instantly generate a custom report: *'Compare Lotto and Bingo'*, *'Overall product trends'*, *'Show monthly'*, *'Trend analysis'*, *'Risk'*, *'Best day'*</span>", unsafe_allow_html=True)

        user_query = st.text_input("", placeholder="e.g. Overall product trends | Compare Lotto and Bingo | Worst performers...")

        if user_query:
            q = user_query.lower()
            st.markdown("### Insight Result")

            if "compare" in q or " vs " in q:
                found_products = [p for p in active_cols if p.lower() in q]
                if len(found_products) >= 2:
                    p1, p2 = found_products[0], found_products[1]
                    diff   = df_filtered[p1].sum() - df_filtered[p2].sum()
                    winner = p1 if diff > 0 else p2
                    st.success(f"**Comparison:** {p1} vs {p2}. **{winner}** leads by **${abs(diff):,.0f}**.")
                    fig_c = px.line(df_filtered, x='Date', y=[p1, p2], title=f"Head-to-Head Timeline: {p1} vs {p2}", color_discrete_map={p1: COLOR_MAP[p1], p2: COLOR_MAP[p2]})
                    fig_c.update_traces(line=dict(width=4, shape='spline'), mode='lines+markers')
                    fig_c.update_layout(**DARK_TEMPLATE, hovermode="x unified", yaxis=dict(gridcolor='#374151'))
                    st.plotly_chart(fig_c, use_container_width=True)
                else:
                    st.warning("Mention at least two valid product names (e.g., 'Compare Lotto and Bingo').")

            elif "overall" in q or "all product" in q or "product trend" in q or "every product" in q:
                st.info("Generating full product trend report...")
                df_ma = df_filtered[['Date'] + active_cols].copy()
                for col in active_cols: df_ma[col] = df_ma[col].rolling(7, min_periods=1).mean()
                fig_ma = px.line(df_ma, x='Date', y=active_cols, title="7-Day Moving Average – All Products", color_discrete_map=COLOR_MAP)
                fig_ma.update_traces(line=dict(width=3, shape='spline'))
                fig_ma.update_layout(**DARK_TEMPLATE, hovermode="x unified", yaxis=dict(gridcolor='#374151'))
                st.plotly_chart(fig_ma, use_container_width=True)

                st.markdown("#### Performance Deltas (Last 7 Days vs First 7 Days)")
                cols_ui = st.columns(4)
                for i, col in enumerate(active_cols):
                    series  = df_filtered[col]
                    early   = series.head(7).mean()
                    recent  = series.tail(7).mean()
                    delta   = ((recent - early) / early * 100) if early else 0
                    cols_ui[i % 4].metric(label=col, value=f"${series.sum():,.0f}", delta=f"{delta:+.1f}% trend")

            elif "trend" in q or "moving average" in q:
                df_filtered['7DMA'] = df_filtered['Total Sales'].rolling(7).mean()
                fig_t = px.scatter(df_filtered, x='Date', y='Total Sales', title="Total Sales & Trendline", opacity=0.4, color_discrete_sequence=['#9ca3af'])
                fig_t.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['7DMA'], name='7-Day Avg', line=dict(color='#3b82f6', width=4, shape='spline')))
                fig_t.update_layout(**DARK_TEMPLATE, hovermode="x unified", yaxis=dict(gridcolor='#374151'))
                st.plotly_chart(fig_t, use_container_width=True)

            elif "risk" in q or "worst" in q or "low" in q or "weak" in q:
                bottom_3 = df_filtered[active_cols].sum().sort_values().head(3)
                st.error(f"Attention Required: **{', '.join(bottom_3.index)}** are underperforming.")
                fig_risk = px.bar(x=bottom_3.index, y=bottom_3.values, color=bottom_3.index, color_discrete_map=COLOR_MAP, title="Bottom 3 Products by Revenue")
                fig_risk.update_layout(**DARK_TEMPLATE, showlegend=False, yaxis=dict(gridcolor='#374151'))
                st.plotly_chart(fig_risk, use_container_width=True)

            elif "best day" in q or "top day" in q or "highest day" in q:
                best_row = df_filtered.loc[df_filtered['Total Sales'].idxmax()]
                st.success(f"Peak Day: **{str(best_row['Date'].date())}** | Total: **${best_row['Total Sales']:,.0f}**")
                best_detail = best_row[active_cols].sort_values(ascending=False).reset_index()
                best_detail.columns = ['Product', 'Revenue']
                fig_bd = px.bar(best_detail, x='Product', y='Revenue', title=f"Revenue Distribution on {str(best_row['Date'].date())}", color='Product', color_discrete_map=COLOR_MAP)
                fig_bd.update_traces(marker_line=dict(color='#111827', width=1.5), texttemplate=' $%{y:,.0f}', textposition='outside', textfont=dict(size=13, weight='bold', color='white'))
                fig_bd.update_layout(**DARK_TEMPLATE, hovermode="x unified", yaxis=dict(gridcolor='#374151'))
                st.plotly_chart(fig_bd, use_container_width=True)

            elif any(p.lower() in q for p in active_cols):
                found = [p for p in active_cols if p.lower() in q][0]
                fig_sp = px.area(df_filtered, x='Date', y=found, title=f"{found} Daily Revenue", color_discrete_sequence=[COLOR_MAP[found]])
                fig_sp.update_traces(line=dict(width=3, shape='spline'))
                fig_sp.update_layout(**DARK_TEMPLATE, hovermode="x unified", yaxis=dict(gridcolor='#374151'))
                st.plotly_chart(fig_sp, use_container_width=True)

            else:
                st.write("I'm not sure how to answer that yet. Try: **'compare X and Y'**, **'overall product trends'**, **'monthly sales'**, **'trend'**, **'risk'**, **'best day'**.")

    # ==========================================
    # TAB 6 — RAW DATA
    # ==========================================
    with tab6:
        st.markdown('<div class="section-header">Raw Data Export</div>', unsafe_allow_html=True)
        search_term = st.text_input("Search records:", placeholder="e.g. 2024-01")
        display_df  = df_filtered.copy()
        if search_term:
            mask = display_df.astype(str).apply(lambda col: col.str.contains(search_term, case=False)).any(axis=1)
            display_df = display_df[mask]

        st.caption(f"Displaying {len(display_df):,} records")
        st.dataframe(display_df, use_container_width=True)
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download CSV Export", data=csv, file_name="sales_export.csv", mime="text/csv")

if __name__ == '__main__':
    if runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())

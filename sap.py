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
# IMPROVEMENT 1: Super Vibrant / Neon Color Palette
COLOR_MAP = {
    'Crossword': '#00F0FF', 'Bingo': '#FF00AA', 'Spin the Wheel': '#FFE600',
    'Race 6': '#00FF66', 'Spin Roulette': '#00E676', 'Crossword Paradise': '#FF3366',
    'Terdrup': '#B366FF', 'Pick 3': '#3399FF', 'Lotto': '#CC33FF',
    'Pick 4': '#FF6600', 'Free Roll': '#00CCFF'
}

DARK_TEMPLATE = dict(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#ffffff', size=13, family="Sora, sans-serif") 
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
    st.set_page_config(page_title="AI Sales Dashboard", layout="wide", page_icon="📈")

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
    .metric-card {
        background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 100%);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 8px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.6);
    }
    .metric-card .label { font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 4px; font-weight: 600;}
    .metric-card .value { font-size: 28px; font-weight: 700; color: #ffffff; }
    .metric-card .delta-pos { font-size: 13px; font-weight: 600; color: #10b981; margin-top: 2px; }
    .metric-card .delta-neg { font-size: 13px; font-weight: 600; color: #ef4444; margin-top: 2px; }
    .metric-card .delta-neu { font-size: 13px; font-weight: 600; color: #94a3b8; margin-top: 2px; }
    .product-scorecard {
        background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 100%);
        border-radius: 12px;
        padding: 16px;
        border-left: 5px solid var(--accent);
        margin-bottom: 10px;
        border-top: 1px solid rgba(255,255,255,0.08);
        border-right: 1px solid rgba(255,255,255,0.08);
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    .scorecard-name  { font-size: 14px; font-weight: 700; color: #ffffff; margin-bottom: 6px; }
    .scorecard-total { font-size: 22px; font-weight: 700; color: #ffffff; }
    .scorecard-sub   { font-size: 12px; color: #94a3b8; margin-top: 2px; }
    .scorecard-share { font-size: 13px; font-weight: 600; color: #cbd5e1; margin-top: 4px; }
    .rank-badge      { display:inline-block; background: rgba(255,255,255,0.15); border-radius:6px; padding: 2px 8px; font-size:11px; color:#ffffff; margin-right:6px; font-weight: bold;}
    .section-header {
        font-size: 22px; font-weight: 700; color: #ffffff;
        margin: 24px 0 16px 0; padding-bottom: 8px;
        border-bottom: 1px solid rgba(255,255,255,0.15);
    }
    </style>
    """, unsafe_allow_html=True)

    df, product_cols = load_data()

    if df.empty:
        st.error("Data file not found. Please ensure 'Online sales1.csv' is available.")
        return

    with st.sidebar:
        st.markdown("### Filters")
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        date_range = st.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        if len(date_range) == 2:
            start_date, end_date = date_range
            df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]

        selected_products = st.multiselect("Focus Products", options=product_cols, default=product_cols)
        if not selected_products:
            selected_products = product_cols

    df_filtered = df.copy()
    active_cols  = [p for p in product_cols if p in selected_products]

    st.markdown("""
    <div style='padding: 10px 0 20px 0;'>
        <div style='font-size:32px; font-weight:700; color:#ffffff; letter-spacing:-0.5px;'>
            AI Sales Intelligence Dashboard
        </div>
        <div style='font-size:14px; color:#94a3b8; margin-top:4px;'>
            Real-time product performance analysis across all game categories
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
        (c3, "Avg Daily Revenue", f"${avg_daily:,.0f}",     None,      f"Over {active_days} days"),
        (c4, "Needs Attention", worst_product,              None,      f"${df_filtered[worst_product].sum():,.0f} total"),
        (c5, "Products Tracked", str(len(active_cols)),     None,      f"{active_cols[0]} leading"),
    ]
    for col, label, value, delta, sub in kpi_data:
        delta_html = f'<div class="delta-{"pos" if delta and delta>0 else ("neg" if delta and delta<0 else "neu")}">{("&#8593;" if delta and delta>0 else "&#8595;") if delta else "-"} {growth_badge(delta) if delta else ""} {sub}</div>'
        col.markdown(f'<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div>{delta_html}</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "All Products Performance", "Performance Overview", "Trends & Time Analysis",
        "Market Intelligence", "AI Search & Analyze", "Raw Data"
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
                
                with cols_row[idx]:
                    st.markdown(f"""
                    <div class="product-scorecard" style="--accent: {color};">
                        <div class="scorecard-name"><span class="rank-badge">#{rank_val}</span>{product}</div>
                        <div class="scorecard-total">${p_stats['Total Revenue']:,.0f}</div>
                        <div class="scorecard-sub">Avg/Day: ${p_stats['Avg Daily']:,.0f}</div>
                        <div class="scorecard-sub">Peak: ${p_stats['Peak Day']:,.0f} on {p_stats['Peak Date']}</div>
                        <div class="scorecard-share">Market Share: {p_stats['Market Share %']:.1f}% &nbsp;|&nbsp; <span style='color:{"#10b981" if mom_val >= 0 else "#ef4444"};'>MoM {growth_badge(mom_val)}</span></div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Revenue Ranking & Market Share</div>', unsafe_allow_html=True)
        col_rank, col_pie = st.columns([3, 2])

        with col_rank:
            totals = df_filtered[active_cols].sum().sort_values(ascending=True).reset_index()
            totals.columns = ['Product', 'Sales']
            # IMPROVEMENT 2: Add bright outlines to bars for contrast
            fig_rank = go.Figure(go.Bar(
                x=totals['Sales'], y=totals['Product'], orientation='h',
                marker=dict(color=[COLOR_MAP.get(p) for p in totals['Product']], line=dict(color='white', width=1.5)),
                text=totals['Sales'], texttemplate='$%{text:,.0f}', textposition='outside',
                textfont=dict(size=14, weight='bold', color='white')
            ))
            fig_rank.update_layout(
                title=dict(text="Total Revenue Ranking", font=dict(size=18, color='white')), 
                **DARK_TEMPLATE, height=420,
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'), yaxis=dict(tickfont=dict(size=13, weight='bold')),
                margin=dict(l=10, r=100, t=40, b=10)
            )
            st.plotly_chart(fig_rank, use_container_width=True)

        with col_pie:
            pie_data = df_filtered[active_cols].sum().reset_index()
            pie_data.columns = ['Product', 'Sales']
            # IMPROVEMENT 3: Add explosive pull and white outlines to the pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=pie_data['Product'], values=pie_data['Sales'], hole=0.4,
                marker=dict(colors=[COLOR_MAP.get(p) for p in pie_data['Product']], line=dict(color='#000000', width=2)),
                pull=[0.05] * len(pie_data) # Gives the pie an separated, cool look
            )])
            fig_pie.update_traces(textposition='inside', textinfo='percent+label', textfont_size=13, textfont_color='white')
            fig_pie.update_layout(title=dict(text="Revenue Market Share", font=dict(size=18, color='white')), **DARK_TEMPLATE, height=420, showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown('<div class="section-header">Individual Product Deep Dive</div>', unsafe_allow_html=True)
        selected_deep = st.selectbox("Select a product to deep dive", options=active_cols)
        if selected_deep:
            deep_df = df_filtered[['Date', selected_deep]].copy()
            deep_df['7DMA'] = deep_df[selected_deep].rolling(7).mean()
            deep_df['30DMA'] = deep_df[selected_deep].rolling(30).mean()

            fig_deep = go.Figure()
            # IMPROVEMENT 4: Add beautiful splines (curved lines) and outlines
            fig_deep.add_trace(go.Bar(
                x=deep_df['Date'], y=deep_df[selected_deep], name='Daily Revenue', 
                marker_color=COLOR_MAP[selected_deep], marker_line=dict(color='white', width=0.5), opacity=0.8
            ))
            fig_deep.add_trace(go.Scatter(
                x=deep_df['Date'], y=deep_df['7DMA'], name='7-Day MA', 
                line=dict(color='#ffffff', width=3, shape='spline')
            ))
            fig_deep.update_layout(
                title=dict(text=f"{selected_deep} — Daily Revenue + Moving Averages", font=dict(size=18, color='white')),
                **DARK_TEMPLATE, height=400, hovermode="x unified", # IMPROVEMENT 5: Unified Hover
                xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                legend=dict(orientation='h', y=1.1)
            )
            st.plotly_chart(fig_deep, use_container_width=True)

    # ==========================================
    # TAB 2 — PERFORMANCE OVERVIEW
    # ==========================================
    with tab2:
        st.markdown('<div class="section-header">Overall Performance Snapshot</div>', unsafe_allow_html=True)
        fig_area = go.Figure()
        for p in active_cols:
            fig_area.add_trace(go.Scatter(
                x=df_filtered['Date'], y=df_filtered[p], name=p, stackgroup='one',
                line=dict(width=2, color=COLOR_MAP[p], shape='spline'), # Added splines to area chart
                fillcolor=COLOR_MAP[p], opacity=0.9
            ))
        fig_area.update_layout(
            title=dict(text="Cumulative Daily Revenue (Stacked Area)", font=dict(size=18, color='white')),
            **DARK_TEMPLATE, height=450, hovermode="x unified",
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            legend=dict(orientation='h', y=-0.15)
        )
        st.plotly_chart(fig_area, use_container_width=True)

        col_bar, col_tbl = st.columns([2, 1])
        with col_bar:
            monthly = df_filtered.groupby('Month')[active_cols].sum().reset_index()
            melted  = monthly.melt(id_vars='Month', var_name='Product', value_name='Sales')
            fig_monthly = px.bar(
                melted, x='Month', y='Sales', color='Product', barmode='group', 
                title="Monthly Revenue per Product", color_discrete_map=COLOR_MAP
            )
            fig_monthly.update_traces(marker_line=dict(color='white', width=1))
            fig_monthly.update_layout(**DARK_TEMPLATE, height=380, hovermode="x unified", legend=dict(orientation='h', y=-0.2))
            st.plotly_chart(fig_monthly, use_container_width=True)
        with col_tbl:
            st.markdown("**Product Revenue Summary**")
            summary_tbl = stats_df[['Product','Total Revenue','Market Share %','Avg Daily']].copy()
            summary_tbl['Total Revenue'] = summary_tbl['Total Revenue'].apply(lambda x: f"${x:,.0f}")
            summary_tbl['Market Share %'] = summary_tbl['Market Share %'].apply(lambda x: f"{x:.1f}%")
            summary_tbl['Avg Daily'] = summary_tbl['Avg Daily'].apply(lambda x: f"${x:,.0f}")
            st.dataframe(summary_tbl, use_container_width=True, hide_index=True)

    # ==========================================
    # TAB 3 — TRENDS & TIME ANALYSIS
    # ==========================================
    with tab3:
        st.markdown('<div class="section-header">Trends & Time-Series Analysis</div>', unsafe_allow_html=True)
        t_col1, t_col2 = st.columns(2)

        with t_col1:
            df_filtered['7DMA_Total']  = df_filtered['Total Sales'].rolling(7).mean()
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=df_filtered['Date'], y=df_filtered['Total Sales'], name='Daily Total', 
                mode='lines+markers', marker=dict(size=4), line=dict(color='rgba(255,255,255,0.3)', width=1)
            ))
            fig_trend.add_trace(go.Scatter(
                x=df_filtered['Date'], y=df_filtered['7DMA_Total'], name='7-Day MA', 
                line=dict(color='#00FF66', width=4, shape='spline') # Thick neon trend line
            ))
            fig_trend.update_layout(title=dict(text="Total Revenue Trend", font=dict(size=18, color='white')), **DARK_TEMPLATE, height=350, hovermode="x unified", legend=dict(orientation='h', y=1.1))
            st.plotly_chart(fig_trend, use_container_width=True)

        with t_col2:
            dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
            dow_data  = df_filtered.groupby('DayOfWeek')[active_cols].sum().reindex(dow_order).reset_index()
            dow_melted= dow_data.melt(id_vars='DayOfWeek', var_name='Product', value_name='Sales')
            fig_dow = px.bar(
                dow_melted, x='DayOfWeek', y='Sales', color='Product', barmode='stack', 
                title="Revenue by Day of Week", color_discrete_map=COLOR_MAP
            )
            fig_dow.update_traces(marker_line=dict(color='rgba(0,0,0,0.5)', width=1))
            fig_dow.update_layout(**DARK_TEMPLATE, height=350, hovermode="x unified", legend=dict(orientation='h', y=-0.25))
            st.plotly_chart(fig_dow, use_container_width=True)

        st.markdown('<div class="section-header">Weekly Rolling Revenue per Product</div>', unsafe_allow_html=True)
        weekly = df_filtered.groupby(['Year','Week'])[active_cols].sum().reset_index()
        weekly['PeriodLabel'] = weekly['Year'].astype(str) + '-W' + weekly['Week'].astype(str).str.zfill(2)
        melted_w = weekly.melt(id_vars='PeriodLabel', value_vars=active_cols, var_name='Product', value_name='Sales')
        fig_weekly = px.line(
            melted_w, x='PeriodLabel', y='Sales', color='Product', title="Weekly Revenue Breakdown", color_discrete_map=COLOR_MAP
        )
        fig_weekly.update_traces(line=dict(width=3, shape='spline'), mode="lines+markers", marker=dict(size=6, line=dict(color='white', width=1)))
        fig_weekly.update_layout(**DARK_TEMPLATE, height=450, hovermode="x unified", legend=dict(orientation='h', y=-0.3))
        st.plotly_chart(fig_weekly, use_container_width=True)

    # ==========================================
    # TAB 4 — MARKET INTELLIGENCE
    # ==========================================
    with tab4:
        st.markdown('<div class="section-header">Competitive Market Intelligence</div>', unsafe_allow_html=True)
        mi_col1, mi_col2 = st.columns(2)

        with mi_col1:
            monthly_by_product = df_filtered.groupby('Month')[active_cols].sum()
            monthly_winner      = monthly_by_product.idxmax(axis=1).reset_index()
            monthly_winner.columns = ['Month','Top Product']
            monthly_winner['Revenue'] = [monthly_by_product.loc[r['Month'], r['Top Product']] for _, r in monthly_winner.iterrows()]
            fig_winners = px.bar(
                monthly_winner, x='Month', y='Revenue', color='Top Product', text='Top Product',
                title="Monthly Champion Product", color_discrete_map=COLOR_MAP
            )
            fig_winners.update_traces(textposition='inside', textfont=dict(size=14, color='white', weight='bold'), marker_line=dict(color='white', width=2))
            fig_winners.update_layout(**DARK_TEMPLATE, height=370, showlegend=False)
            st.plotly_chart(fig_winners, use_container_width=True)

        with mi_col2:
            sorted_rev = df_filtered[active_cols].sum().sort_values(ascending=False)
            cumulative = (sorted_rev.cumsum() / sorted_rev.sum() * 100).reset_index()
            cumulative.columns = ['Product', 'Cumulative %']
            cumulative['Revenue'] = sorted_rev.values
            fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
            fig_pareto.add_trace(go.Bar(
                x=cumulative['Product'], y=cumulative['Revenue'], name='Revenue', 
                marker_color=[COLOR_MAP.get(p,'#fff') for p in cumulative['Product']], marker_line=dict(color='white', width=1)), secondary_y=False)
            fig_pareto.add_trace(go.Scatter(
                x=cumulative['Product'], y=cumulative['Cumulative %'], name='Cumulative %', 
                line=dict(color='#ffffff', width=3, shape='spline'), mode='lines+markers', marker=dict(size=8, color='#00FF66')), secondary_y=True)
            fig_pareto.update_layout(title=dict(text="Pareto Revenue Analysis", font=dict(size=18, color='white')), **DARK_TEMPLATE, height=370, hovermode="x unified", legend=dict(orientation='h', y=1.12))
            st.plotly_chart(fig_pareto, use_container_width=True)

        st.markdown('<div class="section-header">Efficiency Matrix — Avg Daily vs Total Revenue</div>', unsafe_allow_html=True)
        fig_scatter = px.scatter(
            stats_df, x='Avg Daily', y='Total Revenue', size='Market Share %', color='Product', text='Product',
            title="Product Efficiency Matrix (Bubble Size = Market Share)", color_discrete_map={r['Product']: r['Color'] for _, r in stats_df.iterrows()}
        )
        # Add white outlines to the bubbles so they pop out
        fig_scatter.update_traces(textposition='top center', textfont=dict(size=13, weight='bold', color='white'), marker=dict(line=dict(width=2, color='white')))
        fig_scatter.update_layout(**DARK_TEMPLATE, height=500, showlegend=False, xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'))
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ==========================================
    # TAB 5 — AI SEARCH & ANALYZE
    # ==========================================
    with tab5:
        st.markdown('<div class="section-header">AI Search & Analyze</div>', unsafe_allow_html=True)
        st.markdown("Ask me anything about your sales data. Try: *'Compare Lotto and Bingo'*, *'Overall product trends'*, *'Show monthly'*, *'Trend analysis'*, *'Risk'*, *'Best day'*")

        user_query = st.text_input("Your question:", placeholder="e.g. Overall product trends | Compare Lotto and Bingo | Trend analysis | Worst performers")

        if user_query:
            q = user_query.lower()
            st.write("### Insight Result")

            if "compare" in q or " vs " in q:
                found_products = [p for p in active_cols if p.lower() in q]
                if len(found_products) >= 2:
                    p1, p2 = found_products[0], found_products[1]
                    diff   = df_filtered[p1].sum() - df_filtered[p2].sum()
                    winner = p1 if diff > 0 else p2
                    st.info(f"Comparison: {p1} vs {p2}. **{winner}** leads by **${abs(diff):,.0f}**.")
                    fig_c = px.line(df_filtered, x='Date', y=[p1, p2], title=f"Timeline: {p1} vs {p2}", color_discrete_map={p1: COLOR_MAP[p1], p2: COLOR_MAP[p2]})
                    fig_c.update_traces(line=dict(width=4, shape='spline'), mode='lines+markers')
                    fig_c.update_layout(**DARK_TEMPLATE, hovermode="x unified")
                    st.plotly_chart(fig_c, use_container_width=True)
                else:
                    st.warning("Mention at least two valid product names (e.g., 'Compare Lotto and Bingo').")

            elif "overall" in q or "all product" in q or "product trend" in q or "every product" in q:
                st.info("Showing individual trend lines for all active products over the selected period.")
                df_cum = df_filtered[['Date'] + active_cols].copy()
                for col in active_cols: df_cum[col] = df_cum[col].cumsum()
                fig_cum = px.area(df_cum, x='Date', y=active_cols, title="Cumulative Revenue – All Products", color_discrete_map=COLOR_MAP)
                fig_cum.update_traces(line=dict(width=2, shape='spline'))
                fig_cum.update_layout(**DARK_TEMPLATE, hovermode="x unified")
                st.plotly_chart(fig_cum, use_container_width=True)

                df_ma = df_filtered[['Date'] + active_cols].copy()
                for col in active_cols: df_ma[col] = df_ma[col].rolling(7, min_periods=1).mean()
                fig_ma = px.line(df_ma, x='Date', y=active_cols, title="7-Day Moving Average – All Products", color_discrete_map=COLOR_MAP)
                fig_ma.update_traces(line=dict(width=3, shape='spline'))
                fig_ma.update_layout(**DARK_TEMPLATE, hovermode="x unified")
                st.plotly_chart(fig_ma, use_container_width=True)

            elif "trend" in q or "moving average" in q:
                df_filtered['7DMA'] = df_filtered['Total Sales'].rolling(7).mean()
                st.info("Showing 7-day moving average to smooth out daily fluctuations.")
                fig_t = px.scatter(df_filtered, x='Date', y='Total Sales', title="Sales Trend & 7-Day Moving Average", opacity=0.4)
                fig_t.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['7DMA'], name='7-Day Avg', line=dict(color='#00FF66', width=4, shape='spline')))
                fig_t.update_layout(**DARK_TEMPLATE, hovermode="x unified")
                st.plotly_chart(fig_t, use_container_width=True)

            elif "risk" in q or "worst" in q or "low" in q or "weak" in q:
                bottom_3 = df_filtered[active_cols].sum().sort_values().head(3)
                st.warning(f"Low performers: **{', '.join(bottom_3.index)}**")
                fig_risk = go.Figure(data=[go.Pie(labels=bottom_3.index, values=bottom_3.values, hole=0.4, marker=dict(colors=['#ef4444','#f87171','#fca5a5'], line=dict(color='white', width=2)), pull=[0.1, 0, 0])])
                fig_risk.update_traces(textposition='inside', textinfo='percent+label', textfont=dict(size=14, color='white', weight='bold'))
                fig_risk.update_layout(title="Revenue Share of Lowest Performers", **DARK_TEMPLATE)
                st.plotly_chart(fig_risk, use_container_width=True)

            elif "best day" in q or "top day" in q or "highest day" in q:
                best_row = df_filtered.loc[df_filtered['Total Sales'].idxmax()]
                st.success(f"Best single day: **{str(best_row['Date'].date())}** with **${best_row['Total Sales']:,.0f}** in total sales.")
                best_detail = best_row[active_cols].sort_values(ascending=False).reset_index()
                best_detail.columns = ['Product', 'Revenue']
                fig_bd = px.bar(best_detail, x='Product', y='Revenue', title=f"Revenue Breakdown on {str(best_row['Date'].date())}", color='Product', color_discrete_map=COLOR_MAP)
                fig_bd.update_traces(marker_line=dict(color='white', width=1.5), texttemplate='$%{y:,.0f}', textposition='outside', textfont=dict(size=13, weight='bold', color='white'))
                fig_bd.update_layout(**DARK_TEMPLATE, hovermode="x unified")
                st.plotly_chart(fig_bd, use_container_width=True)

            else:
                st.write("I'm not sure how to answer that yet. Try: **'compare X and Y'**, **'overall product trends'**, **'monthly sales'**, **'trend'**, **'risk'**, **'best day'**.")

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
        st.download_button(label="Download Filtered Data as CSV", data=csv, file_name="sales_export.csv", mime="text/csv")

if __name__ == '__main__':
    if runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())

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
    'Crossword':          '#00e5ff',
    'Bingo':              '#df20df',
    'Spin the Wheel':     '#f1c40f',
    'Race 6':             '#10d25f',
    'Spin Roulette':      '#38b284',
    'Crossword Paradise': '#f06277',
    'Terdrup':            '#8b5cf6',
    'Pick 3':             '#3b82f6',
    'Lotto':              '#a855f7',
    'Pick 4':             '#f97316',
    'Free Roll':          '#0ea5e9'
}

# Richer dark template with subtle grid
DARK_TEMPLATE = dict(
    template="plotly_dark",
    paper_bgcolor='rgba(10,14,26,0)',
    plot_bgcolor='rgba(10,14,26,0)',
    font=dict(color='#cbd5e1', family='Sora, sans-serif', size=12),
    title_font=dict(size=15, color='#f1f5f9', family='Sora, sans-serif'),
)

GRID_STYLE = dict(
    xaxis=dict(showgrid=False, zeroline=False, showline=False, tickfont=dict(size=11, color='#64748b')),
    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)', zeroline=False,
               tickfont=dict(size=11, color='#64748b')),
    margin=dict(l=10, r=10, t=50, b=10),
)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Online sales1.csv')
    except FileNotFoundError:
        return pd.DataFrame(), []

    product_cols = list(COLOR_MAP.keys())
    cols_to_fix  = product_cols + ['Wagers/sales']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(',', '').str.replace('"', ''),
                errors='coerce'
            )

    df['Date']      = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
    df              = df.dropna(subset=['Date']).sort_values('Date')
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
    if val > 0:  return f"+{val:.1f}%"
    elif val < 0: return f"{val:.1f}%"
    return "0.0%"

def compute_mom_growth(df, product_cols):
    monthly = df.groupby(['Year','MonthNum'])[product_cols + ['Total Sales']].sum().reset_index()
    monthly = monthly.sort_values(['Year','MonthNum'])
    if len(monthly) < 2: return None, None
    last, prev = monthly.iloc[-1], monthly.iloc[-2]
    growth = {}
    for col in product_cols + ['Total Sales']:
        growth[col] = ((last[col] - prev[col]) / prev[col] * 100) if prev[col] != 0 else 0.0
    return growth, (f"{int(prev['Year'])} M{int(prev['MonthNum'])}", f"{int(last['Year'])} M{int(last['MonthNum'])}")

def get_product_stats(df, product_cols):
    stats = []
    total_all = df[product_cols].sum().sum()
    for p in product_cols:
        s = df[p].dropna()
        total = s.sum(); avg = s.mean(); peak = s.max()
        peak_date = df.loc[df[p] == peak, 'Date'].values
        peak_str  = str(pd.to_datetime(peak_date[0]).date()) if len(peak_date) > 0 else "N/A"
        share = (total / total_all * 100) if total_all > 0 else 0
        stats.append({'Product': p, 'Total Revenue': total, 'Avg Daily': avg,
                      'Peak Day': peak, 'Peak Date': peak_str,
                      'Market Share %': share, 'Color': COLOR_MAP[p]})
    return pd.DataFrame(stats).sort_values('Total Revenue', ascending=False).reset_index(drop=True)

def hex_to_rgba(hex_color, alpha=0.15):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

# ==========================================
# 3. CHART BUILDERS
# ==========================================
def build_sparkline(df, col, color):
    """Mini sparkline trace."""
    return go.Scatter(x=df['Date'], y=df[col].rolling(7, min_periods=1).mean(),
                      mode='lines', line=dict(color=color, width=1.5),
                      showlegend=False, hoverinfo='skip')

def stacked_area_chart(df_filtered, active_cols):
    fig = go.Figure()
    for p in active_cols:
        color = COLOR_MAP[p]
        fig.add_trace(go.Scatter(
            x=df_filtered['Date'], y=df_filtered[p],
            name=p, stackgroup='one',
            line=dict(width=0, color=color),
            fillcolor=hex_to_rgba(color, 0.65),
            hovertemplate=f"<b>{p}</b><br>Date: %{{x|%b %d}}<br>Revenue: $%{{y:,.0f}}<extra></extra>"
        ))
    fig.update_layout(
        title="Daily Revenue Stack — All Products",
        **DARK_TEMPLATE, **GRID_STYLE, height=420,
        legend=dict(orientation='h', y=-0.18, font=dict(size=11)),
        margin=dict(l=10, r=10, t=50, b=80),
        hovermode='x unified'
    )
    return fig

def ribbon_trend_chart(df_filtered):
    """Total sales with ribbon-style MA band."""
    df = df_filtered.copy()
    df['7DMA']  = df['Total Sales'].rolling(7,  min_periods=1).mean()
    df['30DMA'] = df['Total Sales'].rolling(30, min_periods=1).mean()
    df['Upper'] = df['7DMA'] * 1.05
    df['Lower'] = df['7DMA'] * 0.95

    fig = go.Figure()
    # Confidence ribbon
    fig.add_trace(go.Scatter(x=pd.concat([df['Date'], df['Date'][::-1]]),
                             y=pd.concat([df['Upper'], df['Lower'][::-1]]),
                             fill='toself', fillcolor='rgba(0,229,255,0.07)',
                             line=dict(color='rgba(0,0,0,0)'), showlegend=False, hoverinfo='skip'))
    # Raw
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Total Sales'], name='Daily',
                             mode='lines', line=dict(color='rgba(255,255,255,0.15)', width=1),
                             hovertemplate="$%{y:,.0f}<extra>Daily</extra>"))
    # 7DMA
    fig.add_trace(go.Scatter(x=df['Date'], y=df['7DMA'], name='7-Day MA',
                             line=dict(color='#00e5ff', width=2.5),
                             hovertemplate="$%{y:,.0f}<extra>7-Day MA</extra>"))
    # 30DMA
    fig.add_trace(go.Scatter(x=df['Date'], y=df['30DMA'], name='30-Day MA',
                             line=dict(color='#f1c40f', width=2, dash='dot'),
                             hovertemplate="$%{y:,.0f}<extra>30-Day MA</extra>"))
    fig.update_layout(title="Total Revenue Trend with Confidence Band",
                      **DARK_TEMPLATE, **GRID_STYLE, height=370,
                      legend=dict(orientation='h', y=1.12),
                      hovermode='x unified')
    return fig

def radial_market_share(stats_df):
    """Donut with custom pull on leader."""
    pulls = [0.08 if i == 0 else 0 for i in range(len(stats_df))]
    fig = go.Figure(go.Pie(
        labels=stats_df['Product'], values=stats_df['Total Revenue'],
        hole=0.55, pull=pulls,
        marker=dict(colors=[COLOR_MAP[p] for p in stats_df['Product']],
                    line=dict(color='rgba(10,14,26,1)', width=2)),
        textinfo='percent', textfont=dict(size=11),
        hovertemplate="<b>%{label}</b><br>Revenue: $%{value:,.0f}<br>Share: %{percent}<extra></extra>"
    ))
    fig.update_layout(title="Revenue Market Share", **DARK_TEMPLATE, height=420,
                      showlegend=True,
                      legend=dict(orientation='v', x=1.02, y=0.5, font=dict(size=11)),
                      margin=dict(l=10, r=120, t=50, b=10),
                      annotations=[dict(text='<b>Market<br>Share</b>', x=0.5, y=0.5,
                                        font_size=13, font_color='#94a3b8', showarrow=False)])
    return fig

def horizontal_rank_bar(df_filtered, active_cols):
    totals = df_filtered[active_cols].sum().sort_values(ascending=True).reset_index()
    totals.columns = ['Product', 'Sales']
    max_val = totals['Sales'].max()

    fig = go.Figure()
    for _, row in totals.iterrows():
        color = COLOR_MAP.get(row['Product'], '#fff')
        pct   = row['Sales'] / max_val
        fig.add_trace(go.Bar(
            x=[row['Sales']], y=[row['Product']], orientation='h',
            marker=dict(
                color=color,
                opacity=0.85,
                line=dict(width=0)
            ),
            text=f"${row['Sales']:,.0f}",
            textposition='outside',
            textfont=dict(size=11, color='#cbd5e1'),
            name=row['Product'], showlegend=False,
            hovertemplate=f"<b>{row['Product']}</b><br>${row['Sales']:,.0f}<extra></extra>"
        ))

    fig.update_layout(
        title="Revenue Ranking",
        **DARK_TEMPLATE,
        height=420,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=12, color='#e2e8f0')),
        barmode='overlay',
        margin=dict(l=10, r=130, t=50, b=10)
    )
    return fig

def deep_dive_chart(df_filtered, selected_deep):
    deep_df = df_filtered[['Date', selected_deep]].copy()
    deep_df['7DMA']  = deep_df[selected_deep].rolling(7,  min_periods=1).mean()
    deep_df['30DMA'] = deep_df[selected_deep].rolling(30, min_periods=1).mean()
    p_color = COLOR_MAP[selected_deep]

    fig = go.Figure()
    # Area fill under bar
    fig.add_trace(go.Scatter(
        x=deep_df['Date'], y=deep_df[selected_deep],
        fill='tozeroy', fillcolor=hex_to_rgba(p_color, 0.12),
        line=dict(color='rgba(0,0,0,0)'), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Bar(
        x=deep_df['Date'], y=deep_df[selected_deep],
        name='Daily Revenue',
        marker=dict(color=p_color, opacity=0.55, line=dict(width=0)),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>$%{y:,.0f}<extra>Daily</extra>"
    ))
    fig.add_trace(go.Scatter(
        x=deep_df['Date'], y=deep_df['7DMA'], name='7-Day MA',
        line=dict(color='#f1c40f', width=2.5),
        hovertemplate="$%{y:,.0f}<extra>7-Day MA</extra>"
    ))
    fig.add_trace(go.Scatter(
        x=deep_df['Date'], y=deep_df['30DMA'], name='30-Day MA',
        line=dict(color='#ef4444', width=2, dash='dot'),
        hovertemplate="$%{y:,.0f}<extra>30-Day MA</extra>"
    ))
    fig.update_layout(
        title=f"{selected_deep} — Daily Revenue + Moving Averages",
        **DARK_TEMPLATE, **GRID_STYLE, height=400,
        legend=dict(orientation='h', y=1.1),
        hovermode='x unified',
        bargap=0.15
    )
    return fig

def dow_heatmap(df_filtered, active_cols):
    """Heatmap of revenue by day × product."""
    dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    dow_data  = df_filtered.groupby('DayOfWeek')[active_cols].sum().reindex(dow_order)

    fig = px.imshow(
        dow_data.T,
        color_continuous_scale='Turbo',
        aspect='auto',
        title='Revenue Heatmap — Day of Week × Product',
        labels=dict(x='Day', y='Product', color='Revenue')
    )
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>%{x}<br>Revenue: $%{z:,.0f}<extra></extra>"
    )
    fig.update_layout(
        **DARK_TEMPLATE, height=380,
        coloraxis_colorbar=dict(title='Revenue', tickformat='$,.0f', len=0.8),
        margin=dict(l=10, r=80, t=50, b=10),
        xaxis=dict(tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11))
    )
    return fig

def pareto_chart(stats_df):
    sorted_rev = stats_df.set_index('Product')['Total Revenue'].sort_values(ascending=False)
    cumulative = (sorted_rev.cumsum() / sorted_rev.sum() * 100).reset_index()
    cumulative.columns = ['Product','Cumulative %']
    cumulative['Revenue'] = sorted_rev.values

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=cumulative['Product'], y=cumulative['Revenue'],
        name='Revenue',
        marker=dict(color=[COLOR_MAP.get(p,'#fff') for p in cumulative['Product']],
                    opacity=0.85, line=dict(width=0)),
        hovertemplate="<b>%{x}</b><br>$%{y:,.0f}<extra></extra>"
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=cumulative['Product'], y=cumulative['Cumulative %'],
        name='Cumulative %', mode='lines+markers',
        line=dict(color='#f1c40f', width=2.5),
        marker=dict(size=7, color='#f1c40f', line=dict(color='#0a0e1a', width=2)),
        hovertemplate="%{y:.1f}%<extra>Cumulative</extra>"
    ), secondary_y=True)
    # 80% rule line
    fig.add_hline(y=80, line_dash='dash', line_color='rgba(239,68,68,0.5)',
                  secondary_y=True, annotation_text="80%",
                  annotation_font_color='#ef4444', annotation_position='top right')
    fig.update_layout(title="Pareto Revenue Analysis", **DARK_TEMPLATE, height=380,
                      legend=dict(orientation='h', y=1.12),
                      margin=dict(l=10, r=10, t=50, b=60))
    fig.update_yaxes(title_text="Revenue ($)", secondary_y=False,
                     tickformat='$,.0f', showgrid=True, gridcolor='rgba(255,255,255,0.04)')
    fig.update_yaxes(title_text="Cumulative %", secondary_y=True, range=[0, 110],
                     showgrid=False)
    return fig

def efficiency_scatter(stats_df):
    fig = px.scatter(
        stats_df, x='Avg Daily', y='Total Revenue',
        size='Market Share %', color='Product', text='Product',
        title="Product Efficiency Matrix",
        color_discrete_map={r['Product']: r['Color'] for _, r in stats_df.iterrows()},
        size_max=60
    )
    # Add quadrant lines
    mid_x = stats_df['Avg Daily'].median()
    mid_y = stats_df['Total Revenue'].median()
    fig.add_vline(x=mid_x, line_dash='dot', line_color='rgba(255,255,255,0.1)')
    fig.add_hline(y=mid_y, line_dash='dot', line_color='rgba(255,255,255,0.1)')
    fig.add_annotation(x=stats_df['Avg Daily'].max()*0.95, y=stats_df['Total Revenue'].max()*0.97,
                        text="⭐ Stars", showarrow=False, font=dict(color='rgba(255,255,255,0.2)', size=11))
    fig.add_annotation(x=stats_df['Avg Daily'].min()*1.05, y=stats_df['Total Revenue'].min()*1.1,
                        text="⚠ Laggards", showarrow=False, font=dict(color='rgba(255,255,255,0.2)', size=11))
    fig.update_traces(textposition='top center', textfont_size=10,
                      marker=dict(line=dict(width=1.5, color='rgba(10,14,26,0.8)')))
    fig.update_layout(
        **DARK_TEMPLATE, height=430, showlegend=False,
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)', tickformat='$,.0f'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)', tickformat='$,.0f'),
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig

def correlation_heatmap(df_filtered, active_cols):
    corr = df_filtered[active_cols].corr()
    fig  = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale=['#ef4444','#1e293b','#00e5ff'],
        zmin=-1, zmax=1,
        title="Product Sales Correlation Matrix", aspect='auto'
    )
    fig.update_traces(textfont_size=10,
                      hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>Correlation: %{z:.2f}<extra></extra>")
    fig.update_layout(
        **DARK_TEMPLATE, height=460,
        coloraxis_colorbar=dict(title='r', len=0.8),
        margin=dict(l=10, r=80, t=50, b=10)
    )
    return fig

def waterfall_growth(df_filtered, active_cols, mom_growth):
    if not mom_growth: return None
    products = active_cols
    values   = [mom_growth.get(p, 0) for p in products]
    colors   = ['#10d25f' if v >= 0 else '#ef4444' for v in values]

    fig = go.Figure(go.Bar(
        x=products, y=values,
        marker=dict(color=colors, opacity=0.85, line=dict(width=0)),
        text=[f"{v:+.1f}%" for v in values],
        textposition='outside',
        textfont=dict(size=11, color='#cbd5e1'),
        hovertemplate="<b>%{x}</b><br>MoM Growth: %{y:.1f}%<extra></extra>"
    ))
    fig.add_hline(y=0, line_color='rgba(255,255,255,0.15)', line_width=1)
    fig.update_layout(
        title="Month-over-Month Growth by Product",
        **DARK_TEMPLATE, **GRID_STYLE, height=360,
        yaxis=dict(tickformat='+.0f', ticksuffix='%',
                   showgrid=True, gridcolor='rgba(255,255,255,0.04)'),
        margin=dict(l=10, r=10, t=50, b=60)
    )
    return fig

def weekly_line_chart(df_filtered, active_cols):
    weekly = df_filtered.groupby(['Year','Week'])[active_cols].sum().reset_index()
    weekly['PeriodLabel'] = weekly['Year'].astype(str) + '-W' + weekly['Week'].astype(str).str.zfill(2)
    melted = weekly.melt(id_vars='PeriodLabel', value_vars=active_cols,
                         var_name='Product', value_name='Sales')
    fig = px.line(melted, x='PeriodLabel', y='Sales', color='Product',
                  title="Weekly Revenue per Product",
                  color_discrete_map=COLOR_MAP,
                  line_shape='spline')
    fig.update_traces(line=dict(width=2), mode='lines',
                      hovertemplate="<b>%{fullData.name}</b><br>Week: %{x}<br>$%{y:,.0f}<extra></extra>")
    fig.update_layout(
        **DARK_TEMPLATE, **GRID_STYLE, height=400,
        xaxis=dict(showgrid=False, tickangle=-45, tickfont=dict(size=10)),
        legend=dict(orientation='h', y=-0.3, font=dict(size=11)),
        hovermode='x unified',
        margin=dict(l=10, r=10, t=50, b=110)
    )
    return fig

def monthly_winner_chart(df_filtered, active_cols):
    month_order = ['January','February','March','April','May','June',
                   'July','August','September','October','November','December']
    monthly_by_product = df_filtered.groupby('Month')[active_cols].sum()
    monthly_winner     = monthly_by_product.idxmax(axis=1).reset_index()
    monthly_winner.columns = ['Month','Top Product']
    monthly_winner['Revenue'] = [
        monthly_by_product.loc[r['Month'], r['Top Product']]
        for _, r in monthly_winner.iterrows()
    ]
    monthly_winner['SortKey'] = monthly_winner['Month'].apply(
        lambda x: month_order.index(x) if x in month_order else 99
    )
    monthly_winner = monthly_winner.sort_values('SortKey')

    fig = px.bar(
        monthly_winner, x='Month', y='Revenue',
        color='Top Product', text='Top Product',
        title="Monthly Champion Product",
        color_discrete_map=COLOR_MAP
    )
    fig.update_traces(textposition='inside', textfont=dict(size=11, color='white'),
                      marker=dict(line=dict(width=0)),
                      hovertemplate="<b>%{x}</b><br>Champion: %{text}<br>$%{y:,.0f}<extra></extra>")
    fig.update_layout(**DARK_TEMPLATE, **GRID_STYLE, height=370,
                      showlegend=False, margin=dict(l=10, r=10, t=50, b=60))
    return fig

# ==========================================
# 4. MAIN DASHBOARD
# ==========================================
def main():
    st.set_page_config(page_title="AI Sales Dashboard", layout="wide", page_icon="📈")

    # ---- Custom CSS ----
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Sora', sans-serif;
        background-color: #0a0e1a;
    }

    /* Subtle animated mesh background */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1421 0%, #111827 100%);
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(13,27,42,0.9) 0%, rgba(17,24,39,0.95) 100%);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 20px 22px;
        margin-bottom: 10px;
        box-shadow: 0 4px 32px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        backdrop-filter: blur(10px);
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 40px rgba(0,0,0,0.6);
    }
    .metric-card .label  { font-size: 10px; color: #475569; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 6px; }
    .metric-card .value  { font-size: 24px; font-weight: 800; color: #f1f5f9; letter-spacing: -0.5px; line-height: 1.1; }
    .metric-card .delta-pos { font-size: 12px; color: #10b981; margin-top: 4px; font-weight: 600; }
    .metric-card .delta-neg { font-size: 12px; color: #ef4444; margin-top: 4px; font-weight: 600; }
    .metric-card .delta-neu { font-size: 12px; color: #475569; margin-top: 4px; }
    .metric-card .accent-bar {
        height: 2px; border-radius: 1px;
        background: linear-gradient(90deg, var(--accent-color, #00e5ff), transparent);
        margin-bottom: 14px;
    }

    .product-scorecard {
        background: linear-gradient(135deg, rgba(13,27,42,0.95) 0%, rgba(17,24,39,0.95) 100%);
        border-radius: 14px;
        padding: 18px 18px 14px;
        border-left: 3px solid var(--accent);
        margin-bottom: 12px;
        box-shadow: 0 2px 20px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.03);
        transition: transform 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    .product-scorecard::after {
        content: '';
        position: absolute;
        top: 0; right: 0;
        width: 80px; height: 80px;
        background: radial-gradient(circle at top right, var(--accent-glow, rgba(0,229,255,0.06)), transparent 70%);
        pointer-events: none;
    }
    .product-scorecard:hover { transform: translateY(-2px); }
    .scorecard-name  { font-size: 12px; font-weight: 700; color: #f1f5f9; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }
    .scorecard-total { font-size: 22px; font-weight: 800; color: #f1f5f9; letter-spacing: -0.5px; }
    .scorecard-sub   { font-size: 11px; color: #475569; margin-top: 3px; font-family: 'JetBrains Mono', monospace; }
    .scorecard-share { font-size: 11px; font-weight: 600; color: #64748b; margin-top: 8px; border-top: 1px solid rgba(255,255,255,0.04); padding-top: 8px; }
    .rank-badge      { display:inline-block; background: rgba(255,255,255,0.06); border-radius:5px; padding: 1px 7px; font-size:10px; color:#64748b; margin-right:7px; font-family: 'JetBrains Mono', monospace; }

    .section-header {
        font-size: 18px; font-weight: 700; color: #f1f5f9;
        margin: 28px 0 18px 0; padding-bottom: 10px;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        letter-spacing: -0.3px;
    }
    .section-header::before { content: ''; display: inline-block; width: 4px; height: 18px;
        background: #00e5ff; border-radius: 2px; margin-right: 10px; vertical-align: middle; }

    div[data-testid="stTabs"] button {
        font-family: 'Sora', sans-serif !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }

    /* Chart container glass effect */
    div[data-testid="stPlotlyChart"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.04);
        background: rgba(13,27,42,0.4);
    }

    /* Streamlit metric override */
    [data-testid="stMetric"] {
        background: rgba(13,27,42,0.6);
        border-radius: 10px;
        padding: 12px 16px;
        border: 1px solid rgba(255,255,255,0.05);
    }

    .insight-chip {
        display: inline-block;
        background: rgba(0,229,255,0.08);
        border: 1px solid rgba(0,229,255,0.2);
        color: #00e5ff;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 11px;
        font-weight: 600;
        margin: 3px;
        letter-spacing: 0.3px;
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
        st.markdown("### 🎛 Filters")
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        date_range = st.date_input("Date Range", value=(min_date, max_date),
                                   min_value=min_date, max_value=max_date)
        if len(date_range) == 2:
            start_date, end_date = date_range
            df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]

        selected_products = st.multiselect("Focus Products", options=product_cols,
                                           default=product_cols)
        if not selected_products:
            selected_products = product_cols

        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        st.caption(f"**Range:** {df['Date'].min().date()} → {df['Date'].max().date()}")
        st.caption(f"**Days:** {df['Date'].nunique()}")
        st.caption(f"**Products:** {len(selected_products)}")

    df_filtered = df.copy()
    active_cols  = [p for p in product_cols if p in selected_products]
    stats_df     = get_product_stats(df_filtered, active_cols)
    mom_growth, mom_labels = compute_mom_growth(df_filtered, active_cols)

    # ==========================================
    # HEADER
    # ==========================================
    st.markdown("""
    <div style='padding: 12px 0 24px 0;'>
        <div style='font-size:30px; font-weight:800; color:#f1f5f9; letter-spacing:-1px; line-height:1.1;'>
            AI Sales Intelligence
            <span style='color:#00e5ff;'>Dashboard</span>
        </div>
        <div style='font-size:13px; color:#475569; margin-top:6px; letter-spacing:0.2px;'>
            Real-time product performance analytics · Multi-game revenue intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ==========================================
    # TOP KPI ROW
    # ==========================================
    total_rev     = df_filtered[active_cols].sum().sum()
    best_product  = df_filtered[active_cols].sum().idxmax()
    worst_product = df_filtered[active_cols].sum().idxmin()
    avg_daily     = df_filtered['Total Sales'].mean()
    active_days   = df_filtered['Date'].nunique()
    total_mom     = mom_growth['Total Sales'] if mom_growth else 0

    def delta_class(val):
        if val > 0: return "delta-pos"
        if val < 0: return "delta-neg"
        return "delta-neu"

    def delta_arrow(val):
        if val > 0: return "▲"
        if val < 0: return "▼"
        return "–"

    kpi_configs = [
        ("Total Revenue",     f"${total_rev:,.0f}",         total_mom,  "MoM",          "#00e5ff"),
        ("Top Performer",     best_product,                  None,       f"${df_filtered[best_product].sum():,.0f} total",  "#10d25f"),
        ("Avg Daily Revenue", f"${avg_daily:,.0f}",          None,       f"Over {active_days} active days",                 "#f1c40f"),
        ("Needs Attention",   worst_product,                 None,       f"${df_filtered[worst_product].sum():,.0f} total", "#ef4444"),
        ("Products Tracked",  str(len(active_cols)),         None,       "Active game categories",                          "#a855f7"),
    ]

    kpi_cols = st.columns(5)
    for col, (label, value, delta, sub, accent) in zip(kpi_cols, kpi_configs):
        if delta is not None:
            dc = delta_class(delta)
            delta_html = f'<div class="{dc}">{delta_arrow(delta)} {growth_badge(delta)} vs {sub}</div>'
        else:
            delta_html = f'<div class="delta-neu">{sub}</div>'
        col.markdown(f"""
        <div class="metric-card">
            <div class="accent-bar" style="--accent-color:{accent};"></div>
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
        "🏆 Product Scorecards",
        "📊 Performance Overview",
        "📈 Trends & Time Analysis",
        "🔬 Market Intelligence",
        "🤖 AI Search & Analyze",
        "📋 Raw Data"
    ])

    # ==========================================
    # TAB 1 — PRODUCT SCORECARDS
    # ==========================================
    with tab1:
        st.markdown('<div class="section-header">Individual Product Scorecards</div>', unsafe_allow_html=True)

        rows = [active_cols[i:i+3] for i in range(0, len(active_cols), 3)]
        for row in rows:
            cols_row = st.columns(3)
            for idx, product in enumerate(row):
                p_stats  = stats_df[stats_df['Product'] == product].iloc[0]
                rank_val = stats_df.index[stats_df['Product'] == product].tolist()[0] + 1
                color    = COLOR_MAP[product]
                mom_val  = mom_growth[product] if mom_growth and product in mom_growth else 0
                mom_str  = growth_badge(mom_val)
                mom_color = "#10b981" if mom_val >= 0 else "#ef4444"
                glow      = hex_to_rgba(color, 0.08)

                with cols_row[idx]:
                    st.markdown(f"""
                    <div class="product-scorecard" style="--accent:{color}; --accent-glow:{glow};">
                        <div class="scorecard-name">
                            <span class="rank-badge">#{rank_val}</span>{product}
                        </div>
                        <div class="scorecard-total">${p_stats['Total Revenue']:,.0f}</div>
                        <div class="scorecard-sub">Avg/Day &nbsp;${p_stats['Avg Daily']:,.0f}</div>
                        <div class="scorecard-sub">Peak &nbsp;&nbsp;&nbsp;&nbsp;${p_stats['Peak Day']:,.0f} &nbsp;{p_stats['Peak Date']}</div>
                        <div class="scorecard-share">
                            Share {p_stats['Market Share %']:.1f}%
                            &nbsp;&nbsp;·&nbsp;&nbsp;
                            <span style='color:{mom_color}; font-weight:700;'>MoM {mom_str}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Revenue Ranking & Market Share</div>', unsafe_allow_html=True)
        col_rank, col_pie = st.columns([3, 2])
        with col_rank:
            st.plotly_chart(horizontal_rank_bar(df_filtered, active_cols), use_container_width=True)
        with col_pie:
            st.plotly_chart(radial_market_share(stats_df), use_container_width=True)

        st.markdown('<div class="section-header">Individual Product Deep Dive</div>', unsafe_allow_html=True)
        selected_deep = st.selectbox("Select a product", options=active_cols)
        if selected_deep:
            st.plotly_chart(deep_dive_chart(df_filtered, selected_deep), use_container_width=True)
            d_stats = stats_df[stats_df['Product'] == selected_deep].iloc[0]
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Total Revenue",   f"${d_stats['Total Revenue']:,.0f}")
            s2.metric("Avg Daily",       f"${d_stats['Avg Daily']:,.0f}")
            s3.metric("Peak Single Day", f"${d_stats['Peak Day']:,.0f}")
            s4.metric("Market Share",    f"{d_stats['Market Share %']:.2f}%")

    # ==========================================
    # TAB 2 — PERFORMANCE OVERVIEW
    # ==========================================
    with tab2:
        st.markdown('<div class="section-header">Overall Performance Snapshot</div>', unsafe_allow_html=True)
        st.plotly_chart(stacked_area_chart(df_filtered, active_cols), use_container_width=True)

        col_bar, col_tbl = st.columns([2, 1])
        with col_bar:
            month_order = ['January','February','March','April','May','June',
                           'July','August','September','October','November','December']
            monthly = df_filtered.groupby('Month')[active_cols].sum().reset_index()
            monthly['SortKey'] = monthly['Month'].apply(lambda x: month_order.index(x) if x in month_order else 99)
            monthly = monthly.sort_values('SortKey').drop(columns='SortKey')
            melted  = monthly.melt(id_vars='Month', var_name='Product', value_name='Sales')
            fig_monthly = px.bar(melted, x='Month', y='Sales', color='Product',
                                 barmode='group', title="Monthly Revenue per Product",
                                 color_discrete_map=COLOR_MAP)
            fig_monthly.update_traces(marker=dict(line=dict(width=0)),
                                      hovertemplate="<b>%{fullData.name}</b><br>%{x}: $%{y:,.0f}<extra></extra>")
            fig_monthly.update_layout(**DARK_TEMPLATE, **GRID_STYLE, height=390,
                                      legend=dict(orientation='h', y=-0.25, font=dict(size=10)),
                                      margin=dict(l=10, r=10, t=50, b=90))
            st.plotly_chart(fig_monthly, use_container_width=True)

        with col_tbl:
            st.markdown("**Product Revenue Summary**")
            summary_tbl = stats_df[['Product','Total Revenue','Market Share %','Avg Daily']].copy()
            summary_tbl['Total Revenue']  = summary_tbl['Total Revenue'].apply(lambda x: f"${x:,.0f}")
            summary_tbl['Market Share %'] = summary_tbl['Market Share %'].apply(lambda x: f"{x:.1f}%")
            summary_tbl['Avg Daily']      = summary_tbl['Avg Daily'].apply(lambda x: f"${x:,.0f}")
            st.dataframe(summary_tbl, use_container_width=True, hide_index=True, height=380)

    # ==========================================
    # TAB 3 — TRENDS & TIME ANALYSIS
    # ==========================================
    with tab3:
        st.markdown('<div class="section-header">Revenue Trend Analysis</div>', unsafe_allow_html=True)

        t_col1, t_col2 = st.columns(2)
        with t_col1:
            st.plotly_chart(ribbon_trend_chart(df_filtered), use_container_width=True)
        with t_col2:
            st.plotly_chart(dow_heatmap(df_filtered, active_cols), use_container_width=True)

        st.markdown('<div class="section-header">Product Correlation Matrix</div>', unsafe_allow_html=True)
        st.plotly_chart(correlation_heatmap(df_filtered, active_cols), use_container_width=True)

        st.markdown('<div class="section-header">Weekly Revenue Breakdown</div>', unsafe_allow_html=True)
        st.plotly_chart(weekly_line_chart(df_filtered, active_cols), use_container_width=True)

    # ==========================================
    # TAB 4 — MARKET INTELLIGENCE
    # ==========================================
    with tab4:
        st.markdown('<div class="section-header">Competitive Market Intelligence</div>', unsafe_allow_html=True)

        mi1, mi2 = st.columns(2)
        with mi1:
            st.plotly_chart(monthly_winner_chart(df_filtered, active_cols), use_container_width=True)
        with mi2:
            st.plotly_chart(pareto_chart(stats_df), use_container_width=True)

        st.markdown('<div class="section-header">Month-over-Month Growth</div>', unsafe_allow_html=True)
        wf = waterfall_growth(df_filtered, active_cols, mom_growth)
        if wf:
            st.plotly_chart(wf, use_container_width=True)

        if mom_growth and mom_labels:
            growth_rows = []
            for p in active_cols:
                g = mom_growth.get(p, 0)
                m_prev = df_filtered[df_filtered['MonthNum'] == int(mom_labels[0].split('M')[1])][p].sum()
                m_last = df_filtered[df_filtered['MonthNum'] == int(mom_labels[1].split('M')[1])][p].sum()
                growth_rows.append({
                    'Product': p,
                    f'Revenue ({mom_labels[0]})': f"${m_prev:,.0f}",
                    f'Revenue ({mom_labels[1]})': f"${m_last:,.0f}",
                    'MoM Growth': growth_badge(g),
                    'Direction': "▲ UP" if g > 0 else ("▼ DOWN" if g < 0 else "– FLAT")
                })
            st.dataframe(pd.DataFrame(growth_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Not enough monthly data to compute MoM growth.")

        st.markdown('<div class="section-header">Efficiency Matrix</div>', unsafe_allow_html=True)
        st.plotly_chart(efficiency_scatter(stats_df), use_container_width=True)

    # ==========================================
    # TAB 5 — AI SEARCH & ANALYZE
    # ==========================================
    with tab5:
        st.markdown('<div class="section-header">AI Search & Analyze</div>', unsafe_allow_html=True)

        # Chips for quick prompts
        st.markdown("""
        <div style='margin-bottom:14px;'>
            <span style='font-size:12px; color:#475569; margin-right:8px;'>Quick prompts:</span>
            <span class='insight-chip'>Compare Lotto and Bingo</span>
            <span class='insight-chip'>Monthly breakdown</span>
            <span class='insight-chip'>Trend analysis</span>
            <span class='insight-chip'>Risk</span>
            <span class='insight-chip'>Best day</span>
            <span class='insight-chip'>Overall product trend</span>
        </div>
        """, unsafe_allow_html=True)

        user_query = st.text_input("Your question:",
                                   placeholder="e.g., Compare Lotto and Bingo | Overall product trend | Best day | Risk")

        if user_query:
            q = user_query.lower()
            st.markdown("---")

            # CASE 1: COMPARISON
            if "compare" in q or " vs " in q:
                found_products = [p for p in active_cols if p.lower() in q]
                if len(found_products) >= 2:
                    p1, p2 = found_products[0], found_products[1]
                    diff   = df_filtered[p1].sum() - df_filtered[p2].sum()
                    winner = p1 if diff > 0 else p2
                    st.info(f"**{winner}** leads by **${abs(diff):,.0f}**")

                    fig_c = go.Figure()
                    for px_col, color in [(p1, COLOR_MAP[p1]), (p2, COLOR_MAP[p2])]:
                        fig_c.add_trace(go.Scatter(
                            x=df_filtered['Date'], y=df_filtered[px_col].rolling(7, min_periods=1).mean(),
                            name=px_col, fill='tozeroy', fillcolor=hex_to_rgba(color, 0.1),
                            line=dict(color=color, width=2.5),
                            hovertemplate=f"<b>{px_col}</b><br>$%{{y:,.0f}}<extra></extra>"
                        ))
                    fig_c.update_layout(title=f"7-Day MA: {p1} vs {p2}",
                                        **DARK_TEMPLATE, **GRID_STYLE, height=380,
                                        legend=dict(orientation='h', y=1.1),
                                        hovermode='x unified')
                    st.plotly_chart(fig_c, use_container_width=True)
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
                month_data['Total']   = month_data[active_cols].sum(axis=1)
                month_data['SortKey'] = month_data['Month'].apply(lambda x: month_order.index(x) if x in month_order else 99)
                month_data = month_data.sort_values('SortKey').drop(columns='SortKey')
                best_month = month_data.loc[month_data['Total'].idxmax(), 'Month']
                st.success(f"Strongest month overall: **{best_month}**")
                melted = month_data.drop(columns='Total').melt(id_vars='Month', var_name='Product', value_name='Sales')
                fig_m = px.bar(melted, x='Month', y='Sales', color='Product',
                               barmode='stack', title="Monthly Product Breakdown (Stacked)",
                               color_discrete_map=COLOR_MAP)
                fig_m.update_traces(marker=dict(line=dict(width=0)))
                fig_m.update_layout(**DARK_TEMPLATE, **GRID_STYLE, height=400,
                                    legend=dict(orientation='h', y=-0.2))
                st.plotly_chart(fig_m, use_container_width=True)

            # CASE 3: TREND / MOVING AVERAGE
            elif "trend" in q or "moving average" in q:
                if not ("overall" in q or "all product" in q or "product trend" in q or "every product" in q):
                    st.info("Showing total revenue trend with 7-day and 30-day moving averages.")
                    st.plotly_chart(ribbon_trend_chart(df_filtered), use_container_width=True)

            # CASE 4: RISK / WORST
            elif "risk" in q or "worst" in q or "low" in q or "weak" in q:
                bottom_3 = df_filtered[active_cols].sum().sort_values().head(3)
                st.warning(f"⚠ Low performers: **{', '.join(bottom_3.index)}**")

                fig_risk = make_subplots(rows=1, cols=2, specs=[[{'type':'pie'}, {'type':'bar'}]])
                fig_risk.add_trace(go.Pie(
                    labels=bottom_3.index, values=bottom_3.values, hole=0.45,
                    marker=dict(colors=['#ef4444','#f97316','#f59e0b'], line=dict(color='#0a0e1a', width=2)),
                    textinfo='percent+label', textfont=dict(size=11)
                ), row=1, col=1)
                # Trend lines for bottom 3
                for i, prod in enumerate(bottom_3.index):
                    clr = ['#ef4444','#f97316','#f59e0b'][i]
                    fig_risk.add_trace(go.Scatter(
                        x=df_filtered['Date'],
                        y=df_filtered[prod].rolling(7, min_periods=1).mean(),
                        name=prod, line=dict(color=clr, width=2)
                    ), row=1, col=2)
                fig_risk.update_layout(title="Low Performers — Share & Trend",
                                       **DARK_TEMPLATE, height=380,
                                       legend=dict(orientation='h', y=-0.15))
                st.plotly_chart(fig_risk, use_container_width=True)

            # CASE 5: BEST DAY
            elif "best day" in q or "top day" in q or "highest day" in q:
                best_row = df_filtered.loc[df_filtered['Total Sales'].idxmax()]
                st.success(f"Best day: **{str(best_row['Date'].date())}** — **${best_row['Total Sales']:,.0f}** total")
                best_detail = best_row[active_cols].sort_values(ascending=False).reset_index()
                best_detail.columns = ['Product','Revenue']
                fig_bd = go.Figure(go.Bar(
                    x=best_detail['Product'], y=best_detail['Revenue'],
                    marker=dict(color=[COLOR_MAP.get(p,'#fff') for p in best_detail['Product']],
                                opacity=0.85, line=dict(width=0)),
                    text=best_detail['Revenue'].apply(lambda v: f"${v:,.0f}"),
                    textposition='outside', textfont=dict(size=11, color='#cbd5e1'),
                    hovertemplate="<b>%{x}</b><br>$%{y:,.0f}<extra></extra>"
                ))
                fig_bd.update_layout(
                    title=f"Revenue Breakdown — {str(best_row['Date'].date())}",
                    **DARK_TEMPLATE, **GRID_STYLE, height=380,
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)',
                               tickformat='$,.0f')
                )
                st.plotly_chart(fig_bd, use_container_width=True)

            # CASE 6: OVERALL PRODUCT TRENDS
            elif "overall" in q or "all product" in q or "product trend" in q or "every product" in q:
                st.info("Showing comprehensive trend analysis for all active products.")

                # 1. Cumulative area
                df_cum = df_filtered[['Date'] + active_cols].copy()
                for col in active_cols:
                    df_cum[col] = df_cum[col].cumsum()
                fig_cum = px.area(df_cum, x='Date', y=active_cols,
                                  title="Cumulative Revenue — All Products",
                                  color_discrete_map=COLOR_MAP)
                fig_cum.update_traces(line=dict(width=0.5))
                fig_cum.update_layout(**DARK_TEMPLATE, **GRID_STYLE, height=400,
                                      legend=dict(orientation='h', y=-0.2),
                                      hovermode='x unified',
                                      margin=dict(l=10, r=10, t=50, b=80))
                st.plotly_chart(fig_cum, use_container_width=True)

                # 2. 7DMA per product (spline)
                df_ma = df_filtered[['Date'] + active_cols].copy()
                for col in active_cols:
                    df_ma[col] = df_ma[col].rolling(7, min_periods=1).mean()
                fig_ma = px.line(df_ma, x='Date', y=active_cols,
                                 title="7-Day Moving Average — All Products",
                                 color_discrete_map=COLOR_MAP, line_shape='spline')
                fig_ma.update_traces(line=dict(width=2))
                fig_ma.update_layout(**DARK_TEMPLATE, **GRID_STYLE, height=400,
                                     legend=dict(orientation='h', y=-0.2),
                                     hovermode='x unified',
                                     margin=dict(l=10, r=10, t=50, b=80))
                st.plotly_chart(fig_ma, use_container_width=True)

                # 3. Growth % bar
                growth_data = []
                for col in active_cols:
                    series = df_filtered[col]
                    early  = series.head(7).mean()
                    recent = series.tail(7).mean()
                    pct    = ((recent - early) / early * 100) if early else 0
                    growth_data.append({'Product': col, 'Growth %': round(pct, 1)})
                growth_df = pd.DataFrame(growth_data).sort_values('Growth %', ascending=False)
                colors_g  = ['#10d25f' if v >= 0 else '#ef4444' for v in growth_df['Growth %']]
                fig_growth = go.Figure(go.Bar(
                    x=growth_df['Product'], y=growth_df['Growth %'],
                    marker=dict(color=colors_g, opacity=0.85, line=dict(width=0)),
                    text=[f"{v:+.1f}%" for v in growth_df['Growth %']],
                    textposition='outside', textfont=dict(size=11, color='#cbd5e1'),
                    hovertemplate="<b>%{x}</b><br>Growth: %{y:+.1f}%<extra></extra>"
                ))
                fig_growth.add_hline(y=0, line_color='rgba(255,255,255,0.15)', line_width=1)
                fig_growth.update_layout(
                    title="Period Growth % (First 7 Days vs Last 7 Days)",
                    **DARK_TEMPLATE, **GRID_STYLE, height=360,
                    yaxis=dict(tickformat='+.0f', ticksuffix='%',
                               showgrid=True, gridcolor='rgba(255,255,255,0.04)')
                )
                st.plotly_chart(fig_growth, use_container_width=True)

                # 4. Summary metrics
                st.markdown("#### Product-Level Summary")
                cols_ui = st.columns(len(active_cols))
                for i, col in enumerate(active_cols):
                    series = df_filtered[col]
                    early  = series.head(7).mean()
                    recent = series.tail(7).mean()
                    delta  = ((recent - early) / early * 100) if early else 0
                    cols_ui[i].metric(label=col, value=f"${series.sum():,.0f}",
                                      delta=f"{delta:+.1f}% trend")

            # CASE 7: SPECIFIC PRODUCT
            elif any(p.lower() in q for p in active_cols):
                found   = [p for p in active_cols if p.lower() in q][0]
                p_stats = stats_df[stats_df['Product'] == found].iloc[0]
                st.info(f"Full analysis for **{found}**")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Revenue", f"${p_stats['Total Revenue']:,.0f}")
                m2.metric("Avg Daily",     f"${p_stats['Avg Daily']:,.0f}")
                m3.metric("Peak Day",      f"${p_stats['Peak Day']:,.0f}")
                m4.metric("Market Share",  f"{p_stats['Market Share %']:.2f}%")
                st.plotly_chart(deep_dive_chart(df_filtered, found), use_container_width=True)

            else:
                st.markdown("""
                <div style='color:#64748b; font-size:13px; padding:16px; background:rgba(255,255,255,0.02);
                            border-radius:10px; border:1px solid rgba(255,255,255,0.05);'>
                    💡 Try: <b>compare X and Y</b> · <b>monthly sales</b> · <b>trend</b> · <b>overall product trend</b>
                    · <b>risk</b> · <b>best day</b> · or any product name
                </div>
                """, unsafe_allow_html=True)

    # ==========================================
    # TAB 6 — RAW DATA
    # ==========================================
    with tab6:
        st.markdown('<div class="section-header">Raw Data Explorer</div>', unsafe_allow_html=True)
        search_term = st.text_input("Filter by date or value:", placeholder="e.g. 2024-01")
        display_df  = df_filtered.copy()
        if search_term:
            mask       = display_df.astype(str).apply(lambda c: c.str.contains(search_term, case=False)).any(axis=1)
            display_df = display_df[mask]
        st.caption(f"Showing **{len(display_df):,}** rows")
        st.dataframe(display_df, use_container_width=True)
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇ Download Filtered CSV", data=csv,
                           file_name="sales_export.csv", mime="text/csv")

# ==========================================
# 5. EXECUTION
# ==========================================
if __name__ == '__main__':
    if runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import numpy as np
import altair as alt
from st_aggrid import AgGrid, GridOptionsBuilder
import toml

# -----------------------------
# CONFIGURATION & CONSTANTS
# -----------------------------
# Load configuration
config = {
    "theme": {
        "primary_color": "#4a6fa5",
        "secondary_color": "#ffc107",
        "success_color": "#4caf50",
        "warning_color": "#ff9800",
        "danger_color": "#f44336",
        "font": "Arial"
    },
    "thresholds": {
        "nom_compliance": 90,
        "lean_adoption": 80,
        "wellness_index": 85,
        "efficiency": 75
    }
}

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="NOM-035 & LEAN 2.0 Analytics Platform",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# -----------------------------
# DATA LOADING & PROCESSING
# -----------------------------
@st.cache_data(ttl=3600, show_spinner="Loading organizational data...")
def load_data():
    """Load and process all dashboard data with realistic distributions"""
    depts = ['Producción', 'Calidad', 'Logística', 'Administración', 'Ventas', 'RH', 'TI']
    
    # NOM-035 Data with realistic correlations
    base_scores = np.clip(np.random.normal(loc=85, scale=10, size=len(depts)), 50, 100)
    nom_df = pd.DataFrame({
        'Departamento': depts,
        'Evaluaciones': np.round(base_scores * (1 + np.random.normal(0, 0.05, len(depts))), 1),
        'Capacitaciones': np.round(base_scores * 0.9 * (1 + np.random.normal(0, 0.08, len(depts))), 1),
        'Incidentes': np.round(10 * (1 - base_scores/100) + np.random.poisson(1, len(depts)), 1),
        'Tendencia': np.round(np.random.normal(0.5, 1.5, len(depts)), 2),
        'Encuestas_Pendientes': np.random.randint(0, 15, len(depts))
    })
    
    # LEAN 2.0 Data with realistic patterns
    efficiency_base = np.clip(np.random.normal(loc=75, scale=10, size=len(depts)), 50, 95)
    lean_df = pd.DataFrame({
        'Departamento': depts,
        'Eficiencia': np.round(efficiency_base, 1),
        'Reducción_Desperdicio': np.round(efficiency_base * 0.3 + np.random.normal(5, 3, len(depts)), 1),
        'Proyectos_Activos': np.random.poisson(3, len(depts)),
        'Kaizen_Events': np.random.binomial(10, 0.4, len(depts)),
        'Lead_Time_Reduction': np.round(np.random.uniform(5, 25, len(depts)), 1)
    })
    
    # Wellness time series with seasonality
    dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
    seasonality = 5 * np.sin(2 * np.pi * (dates.month - 1)/11)
    wellness_df = pd.DataFrame({
        'Mes': dates,
        'Índice_Bienestar': np.round(75 + seasonality + np.random.normal(0, 3, 12), 1),
        'Ausentismo': np.round(8 - 0.7 * seasonality + np.random.normal(0, 1.5, 12), 1),
        'Rotación': np.round(12 - 0.5 * seasonality + np.random.normal(0, 2, 12), 1),
        'Participación_Eventos': np.round(np.random.uniform(60, 95, 12), 1)
    })
    
    return nom_df, lean_df, wellness_df

nom_data, lean_data, wellness_data = load_data()

# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------
def styled_kpi(value, title, target, icon="📊", unit="%"):
    """Create a dynamic KPI card with trend analysis"""
    delta = value - target
    delta_pct = (delta / target) * 100 if target != 0 else 0
    status = "success" if value >= target else "warning" if value >= target * 0.9 else "danger"
    
    color_map = {
        "success": config["theme"]["success_color"],
        "warning": config["theme"]["warning_color"],
        "danger": config["theme"]["danger_color"]
    }
    
    arrow = "↑" if delta >= 0 else "↓"
    delta_color = color_map[status]
    
    return f"""
    <div style="
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-left: 5px solid {color_map[status]};
        height: 100%;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <p style="font-size: 0.9rem; color: #666; margin: 0 0 5px 0;">{title}</p>
                <h2 style="margin: 0; color: #333; font-size: 1.8rem;">
                    {value}{unit} 
                    <span style="font-size: 1rem; color: {delta_color}">
                        {arrow}{abs(delta_pct):.1f}%
                    </span>
                </h2>
            </div>
            <span style="font-size: 1.5rem;">{icon}</span>
        </div>
        <div style="margin-top: 1rem;">
            <div style="height: 6px; background: #eee; border-radius: 3px;">
                <div style="
                    width: {min(100, (value/target)*100 if target > 0 else 100)}%;
                    height: 6px; 
                    background: {color_map[status]};
                    border-radius: 3px;
                "></div>
            </div>
            <p style="font-size: 0.8rem; color: #888; margin: 5px 0 0 0;">
                Target: {target}{unit}
            </p>
        </div>
    </div>
    """

def create_priority_matrix(df, x_col, y_col, size_col, color_col, title):
    """Create an interactive priority matrix visualization"""
    base = alt.Chart(df).encode(
        tooltip=list(df.columns)
    ).properties(
        width=600,
        height=400,
        title=title
    )
    
    scatter = base.mark_circle(size=100).encode(
        x=alt.X(x_col, scale=alt.Scale(zero=False)),
        y=alt.Y(y_col, scale=alt.Scale(zero=False)),
        size=alt.Size(size_col, legend=None),
        color=alt.Color(color_col, scale=alt.Scale(scheme='redyellowgreen', reverse=True)),
        opacity=alt.value(0.8)
    )
    
    text = base.mark_text(
        align='left',
        baseline='middle',
        dx=15
    ).encode(
        x=x_col,
        y=y_col,
        text='Departamento'
    )
    
    return (scatter + text).interactive()

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
with st.sidebar:
    st.image("https://via.placeholder.com/200x50?text=Company+Logo", use_column_width=True)
    st.title("🔍 Data Explorer")
    
    with st.expander("⏰ Time Period", expanded=True):
        date_range = st.date_input(
            "Select date range",
            value=[date(2025, 1, 1), date(2025, 4, 1)],
            min_value=date(2024, 1, 1),
            max_value=date(2025, 12, 31)
        )
    
    with st.expander("🏢 Departments", expanded=True):
        dept_filter = st.multiselect(
            "Filter departments",
            options=nom_data['Departamento'].unique(),
            default=nom_data['Departamento'].unique()[0:3],
            key="dept_filter"
        )
    
    with st.expander("📊 Metrics", expanded=False):
        metric_groups = st.multiselect(
            "Metric groups to display",
            options=['NOM-035 Compliance', 'LEAN Metrics', 'Wellness Indicators'],
            default=['NOM-035 Compliance', 'LEAN Metrics']
        )
    
    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("### Quick Actions")
    if st.button("🚨 Create Alert", use_container_width=True):
        st.session_state.show_alert_modal = True
    
    if st.button("📝 New Action Plan", use_container_width=True):
        st.session_state.show_action_plan = True

# -----------------------------
# MAIN DASHBOARD LAYOUT
# -----------------------------
# Header
header_col1, header_col2 = st.columns([1, 4])
with header_col1:
    st.image("https://via.placeholder.com/120x60?text=LOGO", width=120)

with header_col2:
    st.title("NOM-035 & LEAN 2.0 Analytics Platform")
    st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <p style="color: #666; margin: 0;">Strategic monitoring of psychosocial wellbeing and operational efficiency</p>
            <p style="color: #888; font-size: 0.8rem; margin: 0;">Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# KPI Cards
st.subheader("📊 Organizational Health Metrics")
kpi_cols = st.columns(4)
with kpi_cols[0]:
    st.markdown(styled_kpi(
        nom_data['Evaluaciones'].mean(),
        "NOM-035 Compliance",
        config["thresholds"]["nom_compliance"],
        "🧠"
    ), unsafe_allow_html=True)

with kpi_cols[1]:
    st.markdown(styled_kpi(
        lean_data['Eficiencia'].mean(),
        "Operational Efficiency",
        config["thresholds"]["efficiency"],
        "⚙️"
    ), unsafe_allow_html=True)

with kpi_cols[2]:
    st.markdown(styled_kpi(
        wellness_data['Índice_Bienestar'].iloc[-1],
        "Wellness Index",
        config["thresholds"]["wellness_index"],
        "❤️"
    ), unsafe_allow_html=True)

with kpi_cols[3]:
    st.markdown(styled_kpi(
        lean_data['Proyectos_Activos'].sum(),
        "Active LEAN Projects",
        15,
        "🔄",
        ""
    ), unsafe_allow_html=True)

# Main Tabs
tab_nom, tab_lean, tab_wellness, tab_actions = st.tabs([
    "🧠 NOM-035", 
    "🔄 LEAN 2.0", 
    "❤️ Wellness", 
    "🚀 Action Center"
])

# NOM-035 Tab
with tab_nom:
    st.subheader("Psychosocial Risk Management")
    
    # Compliance Overview
    fig_nom = px.bar(
        nom_data[nom_data['Departamento'].isin(dept_filter)],
        x='Departamento',
        y=['Evaluaciones', 'Capacitaciones'],
        barmode='group',
        color_discrete_map={
            'Evaluaciones': config["theme"]["primary_color"],
            'Capacitaciones': config["theme"]["secondary_color"]
        },
        height=400
    )
    fig_nom.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title="",
        yaxis_title="Compliance (%)",
        legend_title="Metric"
    )
    st.plotly_chart(fig_nom, use_container_width=True)
    
    # Priority Matrix
    st.subheader("Risk Priority Matrix")
    priority_chart = create_priority_matrix(
        nom_data,
        'Evaluaciones',
        'Capacitaciones',
        'Incidentes',
        'Tendencia',
        'Psychosocial Risk Assessment'
    )
    st.altair_chart(priority_chart, use_container_width=True)

# LEAN 2.0 Tab
with tab_lean:
    st.subheader("Operational Excellence Metrics")
    
    col_lean1, col_lean2 = st.columns(2)
    
    with col_lean1:
        # Efficiency Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=lean_data['Eficiencia'].mean(),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Average Efficiency"},
            gauge={
                'axis': {'range': [None, 100]},
                'steps': [
                    {'range': [0, 70], 'color': "lightgray"},
                    {'range': [70, 90], 'color': "lightyellow"},
                    {'range': [90, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': lean_data['Eficiencia'].mean()
                }
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col_lean2:
        # Waste Reduction
        fig_waste = px.bar(
            lean_data[lean_data['Departamento'].isin(dept_filter)],
            x='Departamento',
            y='Reducción_Desperdicio',
            color='Reducción_Desperdicio',
            color_continuous_scale='Greens',
            title="Waste Reduction (%)"
        )
        st.plotly_chart(fig_waste, use_container_width=True)
    
    # Kaizen Events
    st.subheader("Continuous Improvement Activities")
    fig_kaizen = px.scatter(
        lean_data,
        x='Proyectos_Activos',
        y='Lead_Time_Reduction',
        size='Kaizen_Events',
        color='Departamento',
        hover_name='Departamento',
        title="Improvement Projects vs Lead Time Reduction"
    )
    st.plotly_chart(fig_kaizen, use_container_width=True)

# Wellness Tab
with tab_wellness:
    st.subheader("Employee Wellness Trends")
    
    # Wellness Time Series
    fig_wellness = px.line(
        wellness_data,
        x='Mes',
        y=['Índice_Bienestar', 'Ausentismo', 'Rotación'],
        markers=True,
        color_discrete_map={
            'Índice_Bienestar': config["theme"]["success_color"],
            'Ausentismo': config["theme"]["danger_color"],
            'Rotación': config["theme"]["warning_color"]
        }
    )
    fig_wellness.update_layout(
        yaxis_title="Percentage",
        xaxis_title="",
        legend_title="Metric"
    )
    st.plotly_chart(fig_wellness, use_container_width=True)
    
    # Participation Heatmap
    st.subheader("Event Participation Rates")
    fig_heat = px.imshow(
        wellness_data.set_index('Mes')[['Participación_Eventos']].T,
        labels=dict(x="Month", y="", color="Participation"),
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# Action Center Tab
with tab_actions:
    st.subheader("Action Planning Center")
    
    # Critical Areas
    st.markdown("### 🔴 Critical Areas Requiring Attention")
    critical_df = nom_data[nom_data['Evaluaciones'] < config["thresholds"]["nom_compliance"]]
    
    if not critical_df.empty:
        gb = GridOptionsBuilder.from_dataframe(critical_df)
        gb.configure_selection('multiple', use_checkbox=True)
        gb.configure_column("Evaluaciones", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=1)
        grid_options = gb.build()
        
        grid_response = AgGrid(
            critical_df,
            gridOptions=grid_options,
            height=200,
            width='100%',
            theme='streamlit',
            enable_enterprise_modules=False
        )
        
        selected = grid_response['selected_rows']
        if selected:
            with st.expander("📝 Create Action Plan for Selected", expanded=True):
                action_col1, action_col2 = st.columns(2)
                with action_col1:
                    action_type = st.selectbox("Action Type", [
                        "Training Program",
                        "Process Redesign",
                        "Policy Update",
                        "Team Workshop"
                    ])
                    owner = st.text_input("Action Owner")
                with action_col2:
                    due_date = st.date_input("Due Date")
                    priority = st.select_slider("Priority", options=["Low", "Medium", "High", "Critical"])
                
                if st.button("💾 Save Action Plan"):
                    st.success("Action plan created successfully!")
    else:
        st.success("🎉 No critical areas identified - all departments meet compliance targets!")
    
    # Improvement Opportunities
    st.markdown("### 🟡 Improvement Opportunities")
    opp_df = lean_data[lean_data['Eficiencia'] < config["thresholds"]["efficiency"]]
    
    if not opp_df.empty:
        st.dataframe(
            opp_df.style
            .background_gradient(subset=['Eficiencia'], cmap='Oranges')
            .format({'Eficiencia': '{:.1f}%'}),
            use_container_width=True
        )
    else:
        st.info("All departments meet efficiency targets")

# -----------------------------
# FOOTER & EXPORT
# -----------------------------
st.markdown("---")
export_cols = st.columns(5)
with export_cols[0]:
    if st.button("📄 PDF Report", help="Generate executive PDF report"):
        st.success("Report generation started - will be available shortly")
with export_cols[1]:
    if st.button("📊 Export Data", help="Export current view to Excel"):
        st.success("Data export initiated")
with export_cols[2]:
    if st.button("📧 Email Summary", help="Send summary to stakeholders"):
        st.success("Email queued for delivery")
with export_cols[3]:
    if st.button("🔄 Sync Systems", help="Update connected systems"):
        st.success("Synchronization in progress")
with export_cols[4]:
    if st.button("⚙️ Settings", help="Configure dashboard settings"):
        st.session_state.show_settings = True

st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>© 2025 Organizational Analytics Platform | v2.1.0</p>
        <p style="font-size: 0.8rem;">For support contact: analytics@company.com | +52 664 123 4567</p>
    </div>
""", unsafe_allow_html=True)

# -----------------------------
# MODALS & SESSION STATE
# -----------------------------
if st.session_state.get('show_alert_modal'):
    with st.expander("🚨 Create New Alert", expanded=True):
        alert_col1, alert_col2 = st.columns(2)
        with alert_col1:
            alert_type = st.selectbox("Alert Type", [
                "Compliance Issue",
                "Wellness Concern",
                "Process Deviation",
                "Other"
            ])
            alert_priority = st.select_slider("Priority", options=["Low", "Medium", "High", "Critical"])
        with alert_col2:
            alert_dept = st.multiselect("Departments", nom_data['Departamento'].unique())
            alert_due = st.date_input("Resolution Due By")
        
        alert_desc = st.text_area("Alert Description")
        
        if st.button("Create Alert"):
            st.success("Alert created and notifications sent")
            st.session_state.show_alert_modal = False
        if st.button("Cancel"):
            st.session_state.show_alert_modal = False
st.markdown("""
<div style="text-align:center; color:#666; font-size:0.9rem;">
    <p>© 2025 Sistema NOM-035 & LEAN 2.0</p>
    <p>📞 Soporte: (663) 558 2452 | 📧 contacto@lean2institute.org</p>
</div>
""", unsafe_allow_html=True)

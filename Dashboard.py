import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import warnings
import io
import base64
import re
warnings.filterwarnings('ignore')

# ========== PAGE CONFIGURATION ==========
st.set_page_config(
    page_title="Sistema Integral NOM-035 & LEAN 2.0",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# ========== CONSTANTS AND CONFIGURATION ==========
DEPARTMENTS = ['Producción', 'Calidad', 'Logística', 'Administración', 'Ventas', 'RH', 'TI', 'Mantenimiento', 'R&D', 'Ingeniería']

COLOR_PALETTE = {
    'primary': '#1e3a8a',       # High-contrast navy blue
    'secondary': '#3b82f6',     # Vibrant blue for interactivity
    'accent': '#60a5fa',        # Light blue for highlights
    'success': '#15803d',       # Accessible green
    'warning': '#b45309',       # Accessible amber
    'danger': '#b91c1c',        # Accessible red
    'background': '#f8fafc',    # Soft gray background
    'text': '#1f2937',          # Dark gray for readability
    'muted': '#6b7280',         # Subtle gray for secondary text
    'card': '#ffffff',          # White for cards
    'border': '#e2e8f0'         # Light border color
}

# Custom CSS for modern, accessible, and responsive design
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --primary: #1e3a8a;
    --secondary: #3b82f6;
    --accent: #60a5fa;
    --success: #15803d;
    --warning: #b45309;
    --danger: #b91c1c;
    --background: #f8fafc;
    --text: #1f2937;
    --muted: #6b7280;
    --card: #ffffff;
    --border: #e2e8f0;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text);
    background-color: var(--background);
    line-height: 1.6;
}

.main {
    padding: 2rem;
    max-width: 1400px;
    margin: 0 auto;
}

[data-testid="stSidebar"] {
    background-color: var(--primary) !important;
    color: white !important;
    padding: 1.5rem;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

h1, h2, h3, h4, h5, h6 {
    color: var(--primary);
    font-weight: 600;
    margin-bottom: 1rem;
}

.card {
    background-color: var(--card);
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    border: 1px solid var(--border);
}

.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
}

.stButton > button {
    background-color: var(--secondary);
    color: white;
    border-radius: 8px;
    padding: 0.75rem 1.5rem;
    font-weight: 500;
    transition: background-color 0.2s ease, transform 0.1s ease;
}

.stButton > button:hover {
    background-color: var(--accent);
    transform: translateY(-2px);
}

.stButton > button:focus {
    outline: 3px solid var(--accent);
    outline-offset: 2px;
}

[data-baseweb="tab-list"] {
    gap: 0.75rem;
    border-bottom: 2px solid var(--border);
    margin-bottom: 1.5rem;
}

[data-baseweb="tab"] {
    background-color: var(--card) !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 0.75rem 1.5rem !important;
    color: var(--text) !important;
    font-weight: 500;
    transition: all 0.2s ease;
}

[data-baseweb="tab"][aria-selected="true"] {
    background-color: var(--primary) !important;
    color: white !important;
    border-bottom: 2px solid var(--secondary);
}

.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--border);
}

.stTextInput > div > input,
.stSelectbox > div > select,
.stDateInput > div > input {
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.75rem;
    background-color: var(--card);
}

.stTextInput > div > input:focus,
.stSelectbox > div > select:focus,
.stDateInput > div > input:focus {
    border-color: var(--secondary);
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
}

.error-message {
    color: var(--danger);
    font-size: 0.875rem;
    margin-top: 0.5rem;
    font-weight: 500;
}

.progress-bar {
    height: 8px;
    background-color: var(--border);
    border-radius: 4px;
    overflow: hidden;
}

.progress-bar-fill {
    height: 100%;
    transition: width 0.3s ease;
}

@media (max-width: 768px) {
    .main {
        padding: 1rem;
    }
    [data-testid="stSidebar"] {
        padding: 1rem;
    }
    .card {
        padding: 1rem;
    }
    [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
        font-size: 0.875rem;
    }
    .stButton > button {
        width: 100%;
        padding: 0.5rem;
    }
}
</style>
""", unsafe_allow_html=True)

# ========== DATA LOADING AND PROCESSING ==========
@st.cache_data(ttl=600)
def load_data():
    try:
        np.random.seed(42)
        # Realistic data with correlations
        n_depts = len(DEPARTMENTS)
        base_evals = np.random.normal(85, 8, n_depts)
        nom = pd.DataFrame({
            'Departamento': DEPARTMENTS,
            'Evaluaciones': np.clip(base_evals, 70, 100).round(1),
            'Capacitaciones': np.clip(base_evals + np.random.normal(0, 5, n_depts), 60, 100).round(1),
            'Incidentes': np.clip(np.round(10 - base_evals / 10 + np.random.normal(0, 1, n_depts)), 0, 10),
            'Tendencia': np.round(np.random.normal(0.5, 1.5, n_depts), 2)
        })
        
        base_eff = np.random.normal(80, 10, n_depts)
        lean = pd.DataFrame({
            'Departamento': DEPARTMENTS,
            'Eficiencia': np.clip(base_eff, 60, 95).round(1),
            'Reducción MURI/MURA/MUDA': np.clip(base_eff / 4 + np.random.normal(0, 3, n_depts), 5, 25).round(1),
            'Proyectos Activos': np.clip(np.round(base_eff / 20 + np.random.normal(0, 1, n_depts)), 1, 6),
            '5S+2_Score': np.clip(base_eff + np.random.normal(0, 5, n_depts), 60, 100).round(1),
            'Kaizen Colectivo': np.clip(base_eff - np.random.normal(5, 5, n_depts), 50, 90).round(1)
        })
        
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
        n_months = len(dates)
        base_well = np.linspace(70, 85, n_months)
        bienestar = pd.DataFrame({
            'Mes': dates,
            'Índice Bienestar': np.clip(base_well + np.random.normal(0, 2, n_months), 60, 90).round(1),
            'Ausentismo': np.clip(10 - base_well / 10 + np.random.normal(0, 0.5, n_months), 5, 15).round(1),
            'Rotación': np.clip(15 - base_well / 15 + np.random.normal(0, 0.7, n_months), 5, 20).round(1),
            'Encuestas': np.clip(np.round(80 + np.random.normal(0, 5, n_months)), 75, 100)
        })
        
        action_plans = pd.DataFrame({
            'ID': range(1, 11),
            'Departamento': np.random.choice(DEPARTMENTS, 10),
            'Problema': [
                'Bajo cumplimiento en evaluaciones psicosociales',
                'Ineficiencias en la línea de ensamblaje',
                'Alta rotación en el turno nocturno',
                'Exceso de desperdicio en materiales',
                'Falta de estandarización en procesos',
                'Baja participación en capacitaciones',
                'Retrasos en la cadena de suministro',
                'Fallas recurrentes en maquinaria',
                'Deficiencias en la documentación de procesos',
                'Bajo índice de bienestar reportado'
            ],
            'Acción': [
                'Implementar evaluaciones mensuales',
                'Aplicar estudio de tiempos y movimientos',
                'Mejorar incentivos para turno nocturno',
                'Introducir programa 5R para materiales',
                'Desarrollar manual de procedimientos',
                'Programar sesiones de capacitación obligatorias',
                'Optimizar logística con proveedores',
                'Implementar mantenimiento predictivo',
                'Capacitar equipo en documentación',
                'Lanzar programa de bienestar integral'
            ],
            'Responsable': [
                'Ana Gómez', 'Pedro Sánchez', 'Lucía Fernández', 'Carlos Ruiz', 
                'María López', 'Juan Martínez', 'Sofía Pérez', 'Diego García', 
                'Elena Torres', 'Miguel Ángel'
            ],
            'Plazo': [date(2024, i, 15) for i in range(6, 11)] + [date(2024, i, 30) for i in range(6, 11)],
            'Estado': np.random.choice(['Pendiente', 'En progreso', 'Completado'], 10, p=[0.3, 0.5, 0.2]),
            'Prioridad': np.random.choice(['Alta', 'Media', 'Baja'], 10, p=[0.4, 0.4, 0.2]),
            '% Avance': np.random.choice([0, 25, 50, 75, 100], 10)
        })
        return nom, lean, bienestar, action_plans
    except Exception as e:
        st.error(f"Error al cargar datos: {e}", icon="🚨")
        return None, None, None, None

# Initialize session state for action plans
if 'action_plans_df' not in st.session_state:
    _, _, _, action_plans = load_data()
    st.session_state.action_plans_df = action_plans

# Load data
nom_df, lean_df, bienestar_df, _ = load_data()
if any(df is None for df in (nom_df, lean_df, bienestar_df)):
    st.stop()

# ========== SIDEBAR ==========
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 2rem;">
            <span style="font-size: 2rem;">📊</span>
            <h2 style="margin: 0; color: white;">NOM-035 & LEAN</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        with st.expander("🔍 Filtros", expanded=True):
            st.markdown("**Período**", help="Seleccione el rango de fechas para el análisis")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Inicio",
                    value=date(2024, 1, 1),
                    min_value=date(2024, 1, 1),
                    max_value=date(2024, 12, 31),
                    key="date_start",
                    help="Fecha de inicio del período de análisis",
                    format="DD/MM/YYYY"
                )
            with col2:
                end_date = st.date_input(
                    "Fin",
                    value=date(2024, 12, 31),
                    min_value=start_date,
                    max_value=date(2024, 12, 31),
                    key="date_end",
                    help="Fecha de fin del período de análisis",
                    format="DD/MM/YYYY"
                )
            
            if start_date > end_date:
                st.markdown("<p class='error-message'>La fecha de inicio no puede ser posterior a la fecha de fin</p>", unsafe_allow_html=True)
                return None, None, None
            
            st.markdown("**Departamentos**", help="Filtre por departamentos específicos")
            departamentos_filtro = st.multiselect(
                "Seleccionar departamentos",
                options=DEPARTMENTS,
                default=['Producción', 'Calidad', 'Logística'],
                key="dept_filter",
                help="Seleccione uno o más departamentos para filtrar los datos"
            )
        
        with st.expander("⚙️ Metas", expanded=False):
            st.markdown("**Establecer Metas**", help="Defina los objetivos para cada métrica")
            nom_target = st.slider("Meta NOM-035 (%)", 50, 100, 90, help="Porcentaje objetivo de cumplimiento NOM-035")
            lean_target = st.slider("Meta LEAN (%)", 50, 100, 80, help="Porcentaje objetivo de adopción LEAN")
            wellbeing_target = st.slider("Meta Bienestar (%)", 50, 100, 85, help="Índice objetivo de bienestar organizacional")
            efficiency_target = st.slider("Meta Eficiencia (%)", 50, 100, 75, help="Porcentaje objetivo de eficiencia operativa")
        
        st.markdown("---")
        if st.button("🔄 Actualizar", use_container_width=True, help="Refresca los datos y visualizaciones"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #a0aec0; font-size: 0.75rem;">
            v3.1.0<br>
            © 2025 RH Analytics
        </div>
        """, unsafe_allow_html=True)
    
    return start_date, end_date, departamentos_filtro, (nom_target, lean_target, wellbeing_target, efficiency_target)

# ========== HEADER ==========
def render_header(start_date, end_date):
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1.5rem; margin-bottom: 2rem;">
        <div>
            <h1 style="margin: 0;">Sistema Integral NOM-035 & LEAN 2.0</h1>
            <p style="color: var(--muted); font-size: 1.125rem; margin: 0;">
                Monitoreo Estratégico de Bienestar y Eficiencia
            </p>
        </div>
        <div class="card" style="padding: 0.75rem 1.5rem;">
            <div style="font-size: 0.875rem; color: var(--primary); font-weight: 500;">
                {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}
            </div>
            <div style="font-size: 0.75rem; color: var(--muted);">
                Actualizado: {datetime.now().strftime('%d/%m/%Y %H:%M')}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ========== KPI CARDS ==========
def kpi_card(value, title, target, icon, help_text):
    delta = value - target
    percentage = min(100, (value / target * 100)) if target != 0 else 0
    status = "✅" if value >= target else "⚠️" if value >= target - 10 else "❌"
    color = COLOR_PALETTE['success'] if value >= target else COLOR_PALETTE['warning'] if value >= target - 10 else COLOR_PALETTE['danger']
    delta_text = f"+{delta:.1f}% sobre meta" if delta >= 0 else f"{delta:.1f}% bajo meta"
    
    st.markdown(f"""
    <div class="card" role="region" aria-label="{title}">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
            <div style="font-size: 1rem; color: var(--text); font-weight: 600;">
                {icon} {title}
            </div>
            <div style="font-size: 1.25rem; color: {color};" aria-label="Estado">
                {status}
            </div>
        </div>
        <div style="font-size: 2rem; font-weight: 700; color: {color}; margin-bottom: 0.5rem;">
            {value:.1f}%
        </div>
        <div style="font-size: 0.875rem; color: var(--muted); margin-bottom: 0.75rem;">
            Meta: {target}% • {delta_text}
        </div>
        <div class="progress-bar">
            <div class="progress-bar-fill" style="width: {percentage}%; background: {color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"<span title='{help_text}'></span>", unsafe_allow_html=True)

# ========== TABS ==========
def render_nom_tab(nom_df, departamentos_filtro, nom_target):
    st.markdown("#### 📋 Cumplimiento NOM-035", help="Monitorea el cumplimiento de la NOM-035 en factores psicosociales")
    filtered_nom = nom_df[nom_df['Departamento'].isin(departamentos_filtro)]
    
    if filtered_nom.empty:
        st.warning("⚠️ No hay datos disponibles para los departamentos seleccionados", icon="⚠️")
        return
    
    nom_view1, nom_view2, nom_view3 = st.tabs(["📊 Métricas", "🔍 Mapa de Riesgo", "📈 Tendencias"])
    
    with nom_view1:
        col1, col2 = st.columns([3, 2], gap="medium")
        with col1:
            with st.spinner("Cargando gráfico..."):
                fig = px.bar(
                    filtered_nom,
                    x="Departamento",
                    y=["Evaluaciones", "Capacitaciones"],
                    barmode="group",
                    color_discrete_sequence=[COLOR_PALETTE['primary'], COLOR_PALETTE['secondary']],
                    labels={'value': 'Porcentaje (%)', 'variable': 'Métrica'},
                    height=450,
                    hover_data={'Evaluaciones': ':.1f', 'Capacitaciones': ':.1f'}
                )
                fig.add_hline(
                    y=nom_target,
                    line_dash="dash",
                    line_color=COLOR_PALETTE['warning'],
                    annotation_text="Meta",
                    annotation_position="top right"
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    yaxis_range=[0, 100],
                    legend_title_text='Métrica',
                    xaxis_title="",
                    margin=dict(l=30, r=30, t=50, b=30),
                    font=dict(family="Inter", size=13, color=COLOR_PALETTE['text']),
                    showlegend=True,
                    hoverlabel=dict(bgcolor=COLOR_PALETTE['card'], font_size=12)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**📌 Resumen**", help="Resumen detallado de métricas NOM-035 por departamento")
            st.dataframe(
                filtered_nom.set_index('Departamento').style.format({
                    'Evaluaciones': '{:.1f}',
                    'Capacitaciones': '{:.1f}',
                    'Incidentes': '{:.0f}',
                    'Tendencia': '{:.2f}'
                }).background_gradient(cmap='RdYlGn', subset=['Evaluaciones', 'Capacitaciones']),
                use_container_width=True,
                height=450
            )
    
    with nom_view2:
        with st.spinner("Cargando mapa de riesgo..."):
            scaler = MinMaxScaler()
            z_values = scaler.fit_transform(filtered_nom[['Evaluaciones', 'Capacitaciones', 'Incidentes']])
            fig_heat = go.Figure(data=go.Heatmap(
                z=z_values.T,
                x=filtered_nom['Departamento'],
                y=['Evaluaciones', 'Capacitaciones', 'Incidentes'],
                colorscale=[[0, COLOR_PALETTE['danger']], [0.5, COLOR_PALETTE['warning']], [1, COLOR_PALETTE['success']]],
                hoverongaps=False,
                text=filtered_nom[['Evaluaciones', 'Capacitaciones', 'Incidentes']].values.T,
                texttemplate="%{text:.1f}",
                colorbar=dict(title="Nivel", tickmode="array", tickvals=[0, 0.5, 1], ticktext=["Bajo", "Medio", "Alto"])
            ))
            fig_heat.update_layout(
                title="Mapa de Riesgo Psicosocial",
                xaxis_title="",
                yaxis_title="",
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=50, r=50, t=60, b=50),
                font=dict(family="Inter", size=13, color=COLOR_PALETTE['text']),
                hoverlabel=dict(bgcolor=COLOR_PALETTE['card'], font_size=12)
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            st.markdown("""
            <div class="card" style="margin-top: 1rem;">
                <p style="font-size: 0.875rem; margin: 0.5rem 0;">
                    <strong>Interpretación:</strong> Valores altos en Evaluaciones y Capacitaciones indican buen cumplimiento, mientras que Incidentes altos señalan riesgos psicosociales.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with nom_view3:
        col1, col2 = st.columns([3, 1], gap="medium")
        with col1:
            with st.spinner("Cargando tendencias..."):
                fig_trend = px.bar(
                    filtered_nom,
                    x='Departamento',
                    y='Tendencia',
                    color='Tendencia',
                    color_continuous_scale=[[0, COLOR_PALETTE['danger']], [0.5, COLOR_PALETTE['warning']], [1, COLOR_PALETTE['success']]],
                    range_color=[-3, 3],
                    labels={'Tendencia': 'Cambio mensual (%)'},
                    height=450,
                    hover_data={'Tendencia': ':.2f'}
                )
                fig_trend.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color=COLOR_PALETTE['muted'],
                    annotation_text="Neutral",
                    annotation_position="top right"
                )
                fig_trend.update_layout(
                    title="Tendencia de Cumplimiento",
                    yaxis_title="Cambio (%)",
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=30, r=30, t=50, b=30),
                    font=dict(family="Inter", size=13, color=COLOR_PALETTE['text']),
                    hoverlabel=dict(bgcolor=COLOR_PALETTE['card'], font_size=12)
                )
                st.plotly_chart(fig_trend, use_container_width=True)
        
        with col2:
            st.markdown("**📊 Interpretación**", help="Guía para interpretar las tendencias de cumplimiento")
            st.markdown("""
            <div class="card">
                <p style="font-size: 0.875rem; margin: 0.5rem 0;">
                    <span style="color: var(--success); font-weight: 600;">↑ Positivo:</span> Mejora continua
                </p>
                <p style="font-size: 0.875rem; margin: 0.5rem 0;">
                    <span style="color: var(--danger); font-weight: 600;">↓ Negativo:</span> Requiere atención
                </p>
                <p style="font-size: 0.875rem; margin: 0.5rem 0;">
                    <span style="color: var(--warning); font-weight: 600;">→ Neutral:</span> Mantener monitoreo
                </p>
            </div>
            """, unsafe_allow_html=True)

def render_lean_tab(lean_df, departamentos_filtro, lean_target):
    st.markdown("#### 🔄 Progreso LEAN 2.0", help="Seguimiento de métricas LEAN para optimización de procesos")
    filtered_lean = lean_df[lean_df['Departamento'].isin(departamentos_filtro)]
    
    if filtered_lean.empty:
        st.warning("⚠️ No hay datos disponibles para los departamentos seleccionados", icon="⚠️")
        return
    
    col1, col2 = st.columns([3, 2], gap="medium")
    with col1:
        with st.spinner("Cargando gráfico de eficiencia..."):
            fig_lean = px.bar(
                filtered_lean,
                x='Departamento',
                y='Eficiencia',
                color='Eficiencia',
                color_continuous_scale=[[0, COLOR_PALETTE['danger']], [0.5, COLOR_PALETTE['warning']], [1, COLOR_PALETTE['success']]],
                range_color=[50, 100],
                labels={'Eficiencia': 'Eficiencia (%)'},
                height=450,
                hover_data={'Eficiencia': ':.1f'}
            )
            fig_lean.add_hline(
                y=lean_target,
                line_dash="dash",
                line_color=COLOR_PALETTE['warning'],
                annotation_text="Meta",
                annotation_position="top right"
            )
            fig_lean.update_layout(
                title="Eficiencia por Departamento",
                yaxis_range=[0, 100],
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=30, r=30, t=50, b=30),
                font=dict(family="Inter", size=13, color=COLOR_PALETTE['text']),
                hoverlabel=dict(bgcolor=COLOR_PALETTE['card'], font_size=12)
            )
            st.plotly_chart(fig_lean, use_container_width=True)
        
        with st.spinner("Cargando análisis de desperdicio..."):
            fig_scatter = px.scatter(
                filtered_lean,
                x='Reducción MURI/MURA/MUDA',
                y='Eficiencia',
                size='Proyectos Activos',
                color='Departamento',
                hover_name='Departamento',
                labels={
                    'Reducción MURI/MURA/MUDA': 'Reducción de Desperdicio (%)',
                    'Eficiencia': 'Eficiencia (%)',
                    'Proyectos Activos': 'Proyectos Activos'
                },
                height=450,
                hover_data={'Proyectos Activos': ':.0f'}
            )
            fig_scatter.update_layout(
                title="Eficiencia vs Reducción de Desperdicio",
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=30, r=30, t=50, b=30),
                font=dict(family="Inter", size=13, color=COLOR_PALETTE['text']),
                hoverlabel=dict(bgcolor=COLOR_PALETTE['card'], font_size=12)
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        st.markdown("**📊 Comparación de Métricas**", help="Radar comparativo de métricas LEAN por departamento")
        with st.spinner("Cargando radar..."):
            scaler = MinMaxScaler()
            lean_radar = filtered_lean.copy()
            metrics = ['Eficiencia', 'Reducción MURI/MURA/MUDA', '5S+2_Score', 'Kaizen Colectivo']
            lean_radar[metrics] = scaler.fit_transform(lean_radar[metrics])
            
            fig_radar = go.Figure()
            for dept in filtered_lean['Departamento']:
                row = lean_radar[lean_radar['Departamento'] == dept].iloc[0]
                fig_radar.add_trace(go.Scatterpolar(
                    r=[row[m] for m in metrics],
                    theta=['Eficiencia', 'Reducción', '5S+2', 'Kaizen'],
                    fill='toself',
                    name=dept,
                    line=dict(width=2),
                    hovertemplate=f"<b>{dept}</b><br>%{{theta}}: %{{r:.2f}}<extra></extra>"
                ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
                    angularaxis=dict(showticklabels=True, tickfont_size=12)
                ),
                height=450,
                showlegend=True,
                margin=dict(l=50, r=50, t=30, b=50),
                font=dict(family="Inter", size=13, color=COLOR_PALETTE['text']),
                hoverlabel=dict(bgcolor=COLOR_PALETTE['card'], font_size=12)
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with st.expander("📌 Detalle de Proyectos", expanded=True):
            st.dataframe(
                filtered_lean[['Departamento', 'Proyectos Activos', '5S+2_Score', 'Kaizen Colectivo']]
                .set_index('Departamento')
                .style.background_gradient(cmap='Greens')
                .format({'Proyectos Activos': '{:.0f}', '5S+2_Score': '{:.1f}', 'Kaizen Colectivo': '{:.1f}'}),
                use_container_width=True
            )

def render_wellbeing_tab(bienestar_df, start_date, end_date, wellbeing_target):
    st.markdown("#### 😊 Bienestar Organizacional", help="Indicadores de bienestar y clima laboral")
    filtered_bienestar = bienestar_df[
        (bienestar_df['Mes'].dt.date >= start_date) &
        (bienestar_df['Mes'].dt.date <= end_date)
    ]
    
    if filtered_bienestar.empty:
        st.warning("⚠️ No hay datos disponibles para el período seleccionado", icon="⚠️")
        return
    
    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        delta = filtered_bienestar['Encuestas'].iloc[-1] - filtered_bienestar['Encuestas'].iloc[0] if len(filtered_bienestar) > 1 else 0
        st.metric(
            label="Encuestas Completadas",
            value=f"{filtered_bienestar['Encuestas'].mean():.0f}%",
            delta=f"{delta:+.0f}%",
            help="Porcentaje promedio de encuestas completadas en el período"
        )
    with col2:
        delta = filtered_bienestar['Ausentismo'].iloc[-1] - filtered_bienestar['Ausentismo'].iloc[0] if len(filtered_bienestar) > 1 else 0
        st.metric(
            label="Ausentismo",
            value=f"{filtered_bienestar['Ausentismo'].iloc[-1]:.1f}%",
            delta=f"{delta:+.1f}%",
            delta_color="inverse",
            help="Tasa de ausentismo laboral en el último mes"
        )
    with col3:
        delta = filtered_bienestar['Rotación'].iloc[-1] - filtered_bienestar['Rotación'].iloc[0] if len(filtered_bienestar) > 1 else 0
        st.metric(
            label="Rotación",
            value=f"{filtered_bienestar['Rotación'].iloc[-1]:.1f}%",
            delta=f"{delta:+.1f}%",
            delta_color="inverse",
            help="Tasa de rotación de personal en el último mes"
        )
    
    wellbeing_view1, wellbeing_view2 = st.tabs(["📈 Tendencias", "🔍 Correlaciones"])
    
    with wellbeing_view1:
        with st.spinner("Cargando tendencias..."):
            fig_bienestar = px.line(
                filtered_bienestar,
                x='Mes',
                y=['Índice Bienestar', 'Ausentismo', 'Rotación'],
                markers=True,
                color_discrete_sequence=[
                    COLOR_PALETTE['success'],
                    COLOR_PALETTE['danger'],
                    COLOR_PALETTE['warning']
                ],
                labels={'value': 'Porcentaje (%)', 'variable': 'Métrica'},
                height=450,
                hover_data={'Índice Bienestar': ':.1f', 'Ausentismo': ':.1f', 'Rotación': ':.1f'}
            )
            fig_bienestar.add_hline(
                y=wellbeing_target,
                line_dash="dash",
                line_color=COLOR_PALETTE['warning'],
                annotation_text="Meta Bienestar",
                annotation_position="top right"
            )
            fig_bienestar.update_layout(
                title="Evolución Mensual",
                yaxis_range=[0, 100],
                plot_bgcolor='rgba(0,0,0,0)',
                legend_title="Métrica",
                margin=dict(l=30, r=30, t=50, b=30),
                font=dict(family="Inter", size=13, color=COLOR_PALETTE['text']),
                hoverlabel=dict(bgcolor=COLOR_PALETTE['card'], font_size=12)
            )
            st.plotly_chart(fig_bienestar, use_container_width=True)
    
    with wellbeing_view2:
        with st.spinner("Cargando correlaciones..."):
            corr_matrix = filtered_bienestar[['Índice Bienestar', 'Ausentismo', 'Rotación', 'Encuestas']].corr()
            fig_corr = px.imshow(
                corr_matrix,
                text_auto='.2f',
                color_continuous_scale=[[0, COLOR_PALETTE['danger']], [0.5, COLOR_PALETTE['warning']], [1, COLOR_PALETTE['success']]],
                range_color=[-1, 1],
                labels=dict(x="", y="", color="Correlación"),
                height=450
            )
            fig_corr.update_layout(
                title="Matriz de Correlación",
                margin=dict(l=50, r=50, t=60, b=50),
                font=dict(family="Inter", size=13, color=COLOR_PALETTE['text']),
                hoverlabel=dict(bgcolor=COLOR_PALETTE['card'], font_size=12)
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            st.markdown("""
            <div class="card" style="margin-top: 1rem;">
                <p style="font-size: 0.875rem; margin: 0.5rem 0;">
                    <strong>Interpretación:</strong> Valores cercanos a 1 indican correlación positiva fuerte, -1 indica correlación negativa, y 0 indica poca o ninguna correlación.
                </p>
            </div>
            """, unsafe_allow_html=True)

def render_action_plans_tab(departamentos_filtro, start_date, end_date):
    st.markdown("#### 📝 Planes de Acción", help="Gestión de planes de acción para abordar problemas identificados")
    filtered_plans = st.session_state.action_plans_df[
        (st.session_state.action_plans_df['Departamento'].isin(departamentos_filtro)) &
        (st.session_state.action_plans_df['Plazo'] >= start_date) &
        (st.session_state.action_plans_df['Plazo'] <= end_date)
    ]
    
    col1, col2 = st.columns([3, 1], gap="medium")
    with col1:
        st.markdown("**📌 Planes Registrados**", help="Lista de planes de acción activos")
        if filtered_plans.empty:
            st.info("ℹ️ No hay planes de acción para los filtros seleccionados", icon="ℹ️")
        else:
            def progress_bar(row):
                color = COLOR_PALETTE['success'] if row['% Avance'] == 100 else COLOR_PALETTE['warning'] if row['% Avance'] > 0 else COLOR_PALETTE['danger']
                return f'<div class="progress-bar"><div class="progress-bar-fill" style="width: {row["% Avance"]}%; background: {color};"></div></div>'
            
            styled_plans = filtered_plans.copy()
            styled_plans['Progreso'] = styled_plans.apply(progress_bar, axis=1)
            st.dataframe(
                styled_plans.style.apply(
                    lambda x: [
                        f"background-color: {COLOR_PALETTE['success']}; color: white" if v == 'Completado'
                        else f"background-color: {COLOR_PALETTE['warning']}" if v == 'En progreso'
                        else f"background-color: {COLOR_PALETTE['danger']}; color: white"
                        for v in x
                    ], subset=['Estado']
                ).format({
                    'Plazo': lambda x: x.strftime('%d/%m/%Y'),
                    '% Avance': '{:.0f}%'
                }).set_properties(**{'Progreso': 'width: 100px'}),
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Progreso': st.column_config.TextColumn(
                        'Progreso',
                        help="Barra de progreso del plan de acción",
                        width="small"
                    )
                },
                height=450
            )
    
    with col2:
        st.markdown("**📊 Resumen por Estado**", help="Distribución de planes por estado")
        if not filtered_plans.empty:
            status_summary = filtered_plans['Estado'].value_counts().reset_index()
            fig_status = px.pie(
                status_summary,
                values='count',
                names='Estado',
                color='Estado',
                color_discrete_map={
                    'Completado': COLOR_PALETTE['success'],
                    'En progreso': COLOR_PALETTE['warning'],
                    'Pendiente': COLOR_PALETTE['danger']
                },
                height=300,
                hover_data={'count': True}
            )
            fig_status.update_layout(
                showlegend=True,
                margin=dict(l=20, r=20, t=30, b=20),
                font=dict(family="Inter", size=13, color=COLOR_PALETTE['text']),
                hoverlabel=dict(bgcolor=COLOR_PALETTE['card'], font_size=12)
            )
            st.plotly_chart(fig_status, use_container_width=True)
        
        st.markdown("**📅 Vencimientos Próximos**", help="Planes con plazos cercanos o vencidos")
        today = date.today()
        upcoming = filtered_plans[filtered_plans['Plazo'] <= today + timedelta(days=30)]
        if not upcoming.empty:
            for _, row in upcoming.iterrows():
                days_left = (row['Plazo'] - today).days
                color = COLOR_PALETTE['danger'] if days_left < 7 else COLOR_PALETTE['warning']
                text = f"Vence en {days_left} días" if days_left > 0 else f"Vencido hace {-days_left} días"
                st.markdown(f"""
                <div class="card">
                    <div style="font-weight: 600;">{row['Departamento']}</div>
                    <div style="font-size: 0.875rem; color: var(--muted);">{row['Problema'][:30]}...</div>
                    <div style="font-size: 0.875rem; color: {color}; font-weight: 500;">{text}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("ℹ️ No hay vencimientos próximos", icon="ℹ️")
    
    with st.expander("➕ Nuevo Plan de Acción", expanded=False):
        with st.form("nuevo_plan_form", clear_on_submit=True):
            st.markdown("**Registrar Nuevo Plan**", help="Complete los campos para añadir un nuevo plan de acción")
            col1, col2 = st.columns(2, gap="medium")
            with col1:
                dept = st.selectbox("Departamento", DEPARTMENTS, help="Departamento responsable del plan")
                problema = st.text_area("Problema", max_chars=200, help="Descripción del problema identificado (máx. 200 caracteres)")
                prioridad = st.selectbox("Prioridad", ["Alta", "Media", "Baja"], help="Nivel de prioridad del plan")
            with col2:
                accion = st.text_area("Acción", max_chars=200, help="Acción propuesta para resolver el problema (máx. 200 caracteres)")
                responsable = st.text_input("Responsable", help="Nombre de la persona responsable (solo letras y espacios)")
                plazo = st.date_input(
                    "Plazo",
                    min_value=today,
                    value=today + timedelta(days=30),
                    help="Fecha límite para completar el plan",
                    format="DD/MM/YYYY"
                )
                avance = st.slider("% Avance", 0, 100, 0, help="Porcentaje de avance actual del plan")
            
            submitted = st.form_submit_button("💾 Guardar", use_container_width=True)
            
            if submitted:
                errors = []
                if not problema:
                    errors.append("El campo Problema es obligatorio")
                if len(problema) > 200:
                    errors.append("El Problema no puede exceder 200 caracteres")
                if not accion:
                    errors.append("El campo Acción es obligatorio")
                if len(accion) > 200:
                    errors.append("La Acción no puede exceder 200 caracteres")
                if not responsable:
                    errors.append("El campo Responsable es obligatorio")
                if not re.match(r'^[A-Za-z\s]+$', responsable):
                    errors.append("El Responsable debe contener solo letras y espacios")
                if plazo < today:
                    errors.append("El Plazo no puede ser anterior a hoy")
                
                if errors:
                    for error in errors:
                        st.markdown(f"<p class='error-message'>{error}</p>", unsafe_allow_html=True)
                else:
                    new_plan = pd.DataFrame([{
                        'ID': len(st.session_state.action_plans_df) + 1,
                        'Departamento': dept,
                        'Problema': problema,
                        'Acción': accion,
                        'Responsable': responsable,
                        'Plazo': plazo,
                        'Estado': 'Pendiente' if avance == 0 else 'En progreso' if avance < 100 else 'Completado',
                        'Prioridad': prioridad,
                        '% Avance': avance
                    }])
                    st.session_state.action_plans_df = pd.concat([st.session_state.action_plans_df, new_plan], ignore_index=True)
                    st.success("✅ Plan registrado correctamente", icon="✅")
                    st.rerun()

# ========== EXPORT AND REPORTING ==========
def render_export_section(nom_df, lean_df, bienestar_df):
    st.markdown("---")
    st.markdown("#### 📤 Exportar y Reportes", help="Opciones para exportar datos y generar reportes")
    
    col1, col2, col3 = st.columns(3, gap="medium")
    
    with col1:
        with st.expander("📄 Generar Reporte PDF", expanded=False):
            st.markdown("**Configurar Reporte**", help="Personalice el contenido del reporte PDF")
            report_type = st.selectbox(
                "Tipo de reporte",
                ["Completo", "Resumido", "Solo NOM-035", "Solo LEAN", "Solo Bienestar"],
                help="Seleccione el tipo de reporte a generar"
            )
            include_charts = st.checkbox("Incluir gráficos", value=True, help="Incluye visualizaciones en el reporte")
            if st.button("🖨️ Generar", use_container_width=True):
                with st.spinner("Generando reporte..."):
                    # Simulate PDF generation
                    st.success("✅ Reporte generado", icon="✅")
                    st.download_button(
                        label="📥 Descargar PDF",
                        data=io.BytesIO(f"Reporte {report_type} con {'gráficos' if include_charts else 'sin gráficos'}".encode()),
                        file_name=f"Reporte_{report_type}_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
    
    with col2:
        with st.expander("📧 Enviar por Correo", expanded=False):
            st.markdown("**Enviar Reporte**", help="Envíe el reporte a un correo electrónico")
            email = st.text_input("Correo", placeholder="usuario@empresa.com", help="Correo electrónico del destinatario")
            subject = st.text_input("Asunto", value="Reporte NOM-035 & LEAN", help="Asunto del correo")
            if st.button("📤 Enviar", use_container_width=True):
                if not email:
                    st.markdown("<p class='error-message'>El campo Correo es obligatorio</p>", unsafe_allow_html=True)
                elif not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email):
                    st.markdown("<p class='error-message'>Ingrese un correo válido</p>", unsafe_allow_html=True)
                elif not subject:
                    st.markdown("<p class='error-message'>El campo Asunto es obligatorio</p>", unsafe_allow_html=True)
                else:
                    with st.spinner("Enviando correo..."):
                        # Simulate email sending
                        st.success("✅ Correo enviado", icon="✅")
                        st.download_button(
                            label="📥 Descargar Contenido",
                            data=f"To: {email}\nSubject: {subject}\n\nReporte enviado el {datetime.now().strftime('%d/%m/%Y %H:%M')}",
                            file_name=f"Correo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
    
    with col3:
        with st.expander("📊 Exportar Datos", expanded=False):
            st.markdown("**Exportar Datos**", help="Descargue los datos en el formato deseado")
            export_format = st.radio("Formato", ["CSV", "Excel", "JSON"], horizontal=True, help="Formato de exportación")
            data_options = st.multiselect(
                "Datos",
                ["NOM-035", "LEAN", "Bienestar", "Planes de Acción"],
                default=["NOM-035", "LEAN", "Bienestar"],
                help="Seleccione los conjuntos de datos a exportar"
            )
            if st.button("💾 Descargar", use_container_width=True):
                if not data_options:
                    st.markdown("<p class='error-message'>Seleccione al menos un tipo de datos</p>", unsafe_allow_html=True)
                else:
                    with st.spinner("Preparando datos..."):
                        export_data = []
                        if "NOM-035" in data_options:
                            export_data.append(nom_df.assign(Tipo="NOM-035"))
                        if "LEAN" in data_options:
                            export_data.append(lean_df.assign(Tipo="LEAN"))
                        if "Bienestar" in data_options:
                            export_data.append(bienestar_df.assign(Tipo="Bienestar"))
                        if "Planes de Acción" in data_options:
                            export_data.append(st.session_state.action_plans_df.assign(Tipo="Planes_Accion"))
                        
                        if export_data:
                            combined_data = pd.concat(export_data, ignore_index=True)
                        else:
                            combined_data = pd.DataFrame()
                        
                        try:
                            if export_format == "CSV":
                                data = combined_data.to_csv(index=False).encode('utf-8')
                                mime = "text/csv"
                                ext = "csv"
                            elif export_format == "Excel":
                                output = io.BytesIO()
                                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                    combined_data.to_excel(writer, index=False, sheet_name='Data')
                                data = output.getvalue()
                                mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                ext = "xlsx"
                            else:
                                data = combined_data.to_json(orient='records', date_format='iso').encode('utf-8')
                                mime = "application/json"
                                ext = "json"
                            
                            st.success(f"✅ Datos exportados como {export_format}", icon="✅")
                            st.download_button(
                                label=f"📥 Descargar .{ext}",
                                data=data,
                                file_name=f"nom_lean_data_{datetime.now().strftime('%Y%m%d')}.{ext}",
                                mime=mime,
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"Error al exportar datos: {e}", icon="🚨")

# ========== MAIN FUNCTION ==========
def main():
    try:
        sidebar_data = render_sidebar()
        if sidebar_data is None or sidebar_data[0] is None or sidebar_data[1] is None or not sidebar_data[2]:
            st.warning("⚠️ Por favor, configure los filtros en la barra lateral para continuar", icon="⚠️")
            return
        
        start_date, end_date, departamentos_filtro, targets = sidebar_data
        nom_target, lean_target, wellbeing_target, efficiency_target = targets
        
        render_header(start_date, end_date)
        
        st.markdown("### Indicadores Clave", help="Resumen de métricas clave para NOM-035, LEAN y bienestar")
        cols = st.columns(4, gap="medium")
        kpis = [
            (
                nom_df[nom_df['Departamento'].isin(departamentos_filtro)]['Evaluaciones'].mean(),
                "Cumplimiento NOM-035",
                nom_target,
                "📋",
                "Porcentaje promedio de cumplimiento con NOM-035 en los departamentos seleccionados"
            ),
            (
                lean_df[lean_df['Departamento'].isin(departamentos_filtro)]['Eficiencia'].mean(),
                "Adopción LEAN",
                lean_target,
                "🔄",
                "Nivel promedio de implementación de prácticas LEAN"
            ),
            (
                bienestar_df[(bienestar_df['Mes'].dt.date >= start_date) & (bienestar_df['Mes'].dt.date <= end_date)]['Índice Bienestar'].mean(),
                "Índice Bienestar",
                wellbeing_target,
                "😊",
                "Índice promedio de bienestar organizacional en el período seleccionado"
            ),
            (
                lean_df[lean_df['Departamento'].isin(departamentos_filtro)]['Eficiencia'].mean(),
                "Eficiencia Operativa",
                efficiency_target,
                "⚙️",
                "Eficiencia promedio de procesos operativos"
            )
        ]
        for i, (value, title, target, icon, help_text) in enumerate(kpis):
            with cols[i]:
                kpi_card(value, title, target, icon, help_text)
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 NOM-035",
            "🔄 LEAN",
            "😊 Bienestar",
            "📝 Planes de Acción"
        ])
        
        with tab1:
            render_nom_tab(nom_df, departamentos_filtro, nom_target)
        with tab2:
            render_lean_tab(lean_df, departamentos_filtro, lean_target)
        with tab3:
            render_wellbeing_tab(bienestar_df, start_date, end_date, wellbeing_target)
        with tab4:
            render_action_plans_tab(departamentos_filtro, start_date, end_date)
        
        render_export_section(nom_df, lean_df, bienestar_df)
        
        st.markdown("""
        <hr style="border-color: var(--border);">
        <div style="text-align: center; color: var(--muted); font-size: 0.875rem; padding: 1.5rem 0;">
            Sistema Integral NOM-035 & LEAN 2.0 • Versión 3.1.0<br>
            © 2025 Departamento de RH
        </div>
        """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error en el dashboard: {e}", icon="🚨")

if __name__ == "__main__":
    main()

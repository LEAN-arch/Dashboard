import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import warnings
import io
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    'primary': '#1e3a8a',
    'secondary': '#3b82f6',
    'accent': '#60a5fa',
    'success': '#15803d',
    'warning': '#b45309',
    'danger': '#b91c1c',
    'background': '#f8fafc',
    'text': '#1f2937',
    'muted': '#6b7280',
    'card': '#ffffff',
    'border': '#e2e8f0'
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

.stDateInput [data-baseweb="calendar"] {
    z-index: 9999;
    background-color: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
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
        logger.info("Loading data...")
        np.random.seed(42)
        n_depts = len(DEPARTMENTS)

        # NOM-035 Data (2022-2025, monthly)
        dates = pd.date_range(start='2022-01-01', end='2025-12-31', freq='M')
        nom_data = []
        logger.info(f"Type of nom_data before append: {type(nom_data)}")
        for dept in DEPARTMENTS:
            base_evals = np.linspace(80, 90, len(dates)) + np.random.normal(0, 3, len(dates))
            for i, date in enumerate(dates):
                if not isinstance(nom_data, list):
                    logger.error(f"nom_data is not a list: {type(nom_data)}")
                    nom_data = []
                nom_data.append({
                    'Departamento': dept,
                    'Mes': date,
                    'Evaluaciones': np.clip(base_evals[i], 70, 100).round(1),
                    'Capacitaciones': np.clip(base_evals[i] + np.random.normal(0, 5), 60, 100).round(1),
                    'Incidentes': np.clip(np.round(10 - base_evals[i] / 10 + np.random.normal(0, 1)), 0, 10),
                    'Satisfacción Laboral': np.clip(base_evals[i] + np.random.normal(0, 4), 65, 95).round(1)
                })
        nom_df = pd.DataFrame(nom_data)
        logger.info(f"NOM-035 DataFrame shape: {nom_df.shape}")

        # LEAN Data (2022-2025, monthly)
        lean_data = []
        logger.info(f"Type of lean_data before append: {type(lean_data)}")
        for dept in DEPARTMENTS:
            base_eff = np.linspace(75, 85, len(dates)) + np.random.normal(0, 4, len(dates))
            for i, date in enumerate(dates):
                if not isinstance(lean_data, list):
                    logger.error(f"lean_data is not a list: {type(lean_data)}")
                    lean_data = []
                lean_data.append({
                    'Departamento': dept,
                    'Mes': date,
                    'Eficiencia': np.clip(base_eff[i], 60, 95).round(1),
                    'Reducción MURI/MURA/MUDA': np.clip(base_eff[i] / 4 + np.random.normal(0, 3), 5, 25).round(1),
                    'Proyectos Activos': np.clip(np.round(base_eff[i] / 20 + np.random.normal(0, 1)), 1, 6),
                    '5S+2_Score': np.clip(base_eff[i] + np.random.normal(0, 5), 60, 100).round(1),
                    'Kaizen Colectivo': np.clip(base_eff[i] - np.random.normal(5, 5), 50, 90).round(1),
                    'Tiempo Ciclo': np.clip(100 - base_eff[i] + np.random.normal(0, 5), 10, 50).round(1)
                })
        lean_df = pd.DataFrame(lean_data)
        logger.info(f"LEAN DataFrame shape: {lean_df.shape}")

        # Bienestar Data (2022-2025, monthly)
        base_well = np.linspace(70, 85, len(dates))
        bienestar_df = pd.DataFrame({
            'Mes': dates,
            'Índice Bienestar': np.clip(base_well + np.random.normal(0, 2, len(dates)), 60, 90).round(1),
            'Ausentismo': np.clip(10 - base_well / 10 + np.random.normal(0, 0.5, len(dates)), 5, 15).round(1),
            'Rotación': np.clip(15 - base_well / 15 + np.random.normal(0, 0.7, len(dates)), 5, 20).round(1),
            'Encuestas': np.clip(np.round(80 + np.random.normal(0, 5, len(dates))), 75, 100),
            'Engagement': np.clip(base_well + np.random.normal(0, 3, len(dates)), 60, 90).round(1)
        })
        logger.info(f"Bienestar DataFrame shape: {bienestar_df.shape}")

        # Action Plans
        action_plans = pd.DataFrame({
            'ID': range(1, 21),
            'Departamento': np.random.choice(DEPARTMENTS, 20),
            'Problema': [
                'Bajo cumplimiento en evaluaciones psicosociales', 'Ineficiencias en la línea de ensamblaje',
                'Alta rotación en el turno nocturno', 'Exceso de desperdicio en materiales',
                'Falta de estandarización en procesos', 'Baja participación en capacitaciones',
                'Retrasos en la cadena de suministro', 'Fallas recurrentes en maquinaria',
                'Deficiencias en la documentación de procesos', 'Bajo índice de bienestar reportado',
                'Altos tiempos de ciclo en producción', 'Falta de adopción de 5S+2',
                'Baja colaboración interdepartamental', 'Errores frecuentes en inventario',
                'Falta de capacitación en herramientas LEAN', 'Bajo engagement en encuestas',
                'Exceso de MURA en procesos', 'Problemas de ergonomía en puestos',
                'Retrasos en proyectos de mejora', 'Falta de comunicación en equipos'
            ],
            'Acción': [
                'Implementar evaluaciones mensuales', 'Aplicar estudio de tiempos y movimientos',
                'Mejorar incentivos para turno nocturno', 'Introducir programa 5R para materiales',
                'Desarrollar manual de procedimientos', 'Programar sesiones de capacitación obligatorias',
                'Optimizar logística con proveedores', 'Implementar mantenimiento predictivo',
                'Capacitar equipo en documentación', 'Lanzar programa de bienestar integral',
                'Rediseñar flujo de producción', 'Auditorías mensuales de 5S+2',
                'Crear equipos interdepartamentales', 'Implementar sistema de gestión de inventarios',
                'Capacitar en metodologías LEAN', 'Rediseñar encuestas de engagement',
                'Estandarizar procesos para reducir MURA', 'Realizar estudios ergonómicos',
                'Establecer cronogramas estrictos', 'Implementar reuniones diarias de equipo'
            ],
            'Responsable': [
                'Ana Gómez', 'Pedro Sánchez', 'Lucía Fernández', 'Carlos Ruiz', 'María López',
                'Juan Martínez', 'Sofía Pérez', 'Diego García', 'Elena Torres', 'Miguel Ángel',
                'Laura Ramírez', 'Jorge Díaz', 'Clara Morales', 'Andrés Vega', 'Patricia Soto',
                'Felipe Castro', 'Marina Ortiz', 'Raúl Méndez', 'Isabel Cruz', 'Héctor Luna'
            ],
            'Plazo': pd.date_range(start='2025-01-15', end='2025-10-30', periods=20),
            'Estado': np.random.choice(['Pendiente', 'En progreso', 'Completado'], 20, p=[0.3, 0.5, 0.2]),
            'Prioridad': np.random.choice(['Alta', 'Media', 'Baja'], 20, p=[0.4, 0.4, 0.2]),
            '% Avance': np.random.choice([0, 25, 50, 75, 100], 20),
            'Costo Estimado': np.random.randint(5000, 50000, 20)
        })
        logger.info(f"Action Plans DataFrame shape: {action_plans.shape}")

        return nom_df, lean_df, bienestar_df, action_plans
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        st.error(f"Error al cargar datos: {e}", icon="🚨")
        return None, None, None, None

# Initialize session state
if 'action_plans_df' not in st.session_state:
    logger.info("Initializing session state for action plans")
    nom_df, lean_df, bienestar_df, action_plans = load_data()
    if action_plans is None:
        st.error("No se pudieron cargar los planes de acción. Intente de nuevo.", icon="🚨")
        st.session_state.action_plans_df = pd.DataFrame(columns=[
            'ID', 'Departamento', 'Problema', 'Acción', 'Responsable', 'Plazo',
            'Estado', 'Prioridad', '% Avance', 'Costo Estimado'
        ])
    else:
        st.session_state.action_plans_df = action_plans

# Load data
nom_df, lean_df, bienestar_df, _ = load_data()
if any(df is None for df in (nom_df, lean_df, bienestar_df)):
    logger.error("One or more DataFrames are None")
    st.error("No se pudieron cargar los datos. Intente de nuevo.", icon="🚨")
    st.stop()

# ========== HELPER FUNCTIONS ==========
def filter_dataframe(df, departamentos_filtro, start_date, end_date, date_column='Mes'):
    """Filter DataFrame by departments and date range."""
    try:
        logger.info(f"Filtering DataFrame with date_column={date_column}, departamentos={departamentos_filtro}")
        if date_column not in df.columns:
            logger.warning(f"Date column {date_column} not in DataFrame")
            return df.copy()
        filtered_df = df[
            (df['Departamento'].isin(departamentos_filtro) if 'Departamento' in df.columns else True) &
            (df[date_column].dt.date >= start_date) &
            (df[date_column].dt.date <= end_date)
        ]
        if filtered_df.empty:
            logger.warning("Filtered DataFrame is empty")
            return pd.DataFrame(columns=df.columns)
        logger.info(f"Filtered DataFrame shape: {filtered_df.shape}")
        return filtered_df
    except Exception as e:
        logger.error(f"Error filtering DataFrame: {e}")
        st.warning(f"Error al filtrar datos: {e}", icon="⚠️")
        return pd.DataFrame(columns=df.columns)

# ========== SIDEBAR ==========
def render_sidebar():
    with st.sidebar:
        logger.info("Rendering sidebar")
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
                    value=date(2022, 1, 1),
                    min_value=date(2022, 1, 1),
                    max_value=date(2025, 12, 31),
                    key="date_start",
                    help="Fecha de inicio del período de análisis",
                    format="DD/MM/YYYY"
                )
            with col2:
                min_end_date = start_date if start_date else date(2022, 1, 1)
                end_date = st.date_input(
                    "Fin",
                    value=date(2025, 12, 31),
                    min_value=min_end_date,
                    max_value=date(2025, 12, 31),
                    key="date_end",
                    help="Fecha de fin del período de análisis",
                    format="DD/MM/YYYY"
                )
            
            if start_date > end_date:
                logger.warning("Start date is after end date")
                st.markdown("<p class='error-message'>La fecha de inicio no puede ser posterior a la fecha de fin</p>", unsafe_allow_html=True)
                return None, None, None, None, None, None
            
            st.markdown("**Departamentos**", help="Filtre por departamentos específicos")
            departamentos_filtro = st.multiselect(
                "Seleccionar departamentos",
                options=DEPARTMENTS,
                default=['Producción', 'Calidad', 'Logística'],
                key="dept_filter",
                help="Seleccione uno o más departamentos para filtrar los datos"
            )
            
            st.markdown("**Métricas**", help="Seleccione métricas para visualizar")
            nom_metrics = st.multiselect(
                "Métricas NOM-035",
                ['Evaluaciones', 'Capacitaciones', 'Incidentes', 'Satisfacción Laboral'],
                default=['Evaluaciones', 'Capacitaciones'],
                key="nom_metrics"
            )
            lean_metrics = st.multiselect(
                "Métricas LEAN",
                ['Eficiencia', 'Reducción MURI/MURA/MUDA', 'Proyectos Activos', '5S+2_Score', 'Kaizen Colectivo', 'Tiempo Ciclo'],
                default=['Eficiencia', '5S+2_Score'],
                key="lean_metrics"
            )
        
        with st.expander("⚙️ Metas", expanded=False):
            st.markdown("**Establecer Metas**", help="Defina los objetivos para cada métrica")
            nom_target = st.slider("Meta NOM-035 (%)", 50, 100, 90, help="Porcentaje objetivo de cumplimiento NOM-035")
            lean_target = st.slider("Meta LEAN (%)", 50, 100, 80, help="Porcentaje objetivo de adopción LEAN")
            wellbeing_target = st.slider("Meta Bienestar (%)", 50, 100, 85, help="Índice objetivo de bienestar organizacional")
            efficiency_target = st.slider("Meta Eficiencia (%)", 50, 100, 75, help="Porcentaje objetivo de eficiencia operativa")
        
        st.markdown("---")
        if st.button("🔄 Actualizar", use_container_width=True, help="Refresca los datos y visualizaciones"):
            logger.info("Clearing cache and rerunning app")
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #a0aec0; font-size: 0.75rem;">
            v3.2.0<br>
            © 2025 RH Analytics
        </div>
        """, unsafe_allow_html=True)
    
    logger.info("Sidebar rendered successfully")
    return start_date, end_date, departamentos_filtro, (nom_target, lean_target, wellbeing_target, efficiency_target), nom_metrics, lean_metrics

# ========== HEADER ==========
def render_header(start_date, end_date):
    logger.info("Rendering header")
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1.5rem; margin-bottom: 2rem;">
        <div>
            <h1 style="margin: 0;">Sistema Integral NOM-035 & LEAN 2.0</h1>
            <p style="color: var(--muted); font-size: 1.125rem; margin: 0;">
                Monitoreo Estratégico de Bienestar y Eficiencia (2022-2025)
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
def kpi_card(value, title, target, icon, help_text, delta=None):
    logger.info(f"Rendering KPI card: {title}")
    delta_value = delta if delta is not None else value - target
    percentage = min(100, (value / target * 100)) if target != 0 else 0
    status = "✅" if value >= target else "⚠️" if value >= target - 10 else "❌"
    color = COLOR_PALETTE['success'] if value >= target else COLOR_PALETTE['warning'] if value >= target - 10 else COLOR_PALETTE['danger']
    delta_text = f"+{delta_value:.1f}% sobre meta" if delta_value >= 0 else f"{delta_value:.1f}% bajo meta"
    
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
def render_nom_tab(nom_df, departamentos_filtro, nom_target, start_date, end_date, nom_metrics):
    logger.info("Rendering NOM-035 tab")
    st.markdown("#### 📋 Cumplimiento NOM-035", help="Monitorea el cumplimiento de la NOM-035 en factores psicosociales")
    
    if not nom_metrics:
        logger.warning("No NOM-035 metrics selected")
        st.warning("⚠️ Por favor, seleccione al menos una métrica NOM-035 en los filtros.", icon="⚠️")
        return
    
    filtered_nom = filter_dataframe(nom_df, departamentos_filtro, start_date, end_date)
    
    if filtered_nom.empty or not all(col in filtered_nom.columns for col in nom_metrics):
        logger.warning("Filtered NOM-035 DataFrame is empty or missing metrics")
        st.warning("⚠️ No hay datos disponibles para los filtros seleccionados o métricas faltantes", icon="⚠️")
        return
    
    nom_view1, nom_view2, nom_view3 = st.tabs(["📊 Métricas", "🔍 Mapa de Riesgo", "📈 Tendencias"])
    
    with nom_view1:
        col1, col2 = st.columns([3, 2], gap="medium")
        with col1:
            with st.spinner("Cargando gráfico..."):
                try:
                    logger.info("Rendering NOM-035 line chart")
                    grouped_data = filtered_nom.groupby(['Mes', 'Departamento'])[nom_metrics].mean().reset_index()
                    if grouped_data.empty:
                        logger.warning("Grouped NOM-035 data is empty")
                        st.warning("⚠️ No hay datos suficientes para renderizar el gráfico", icon="⚠️")
                        return
                    fig = px.line(
                        grouped_data,
                        x="Mes",
                        y=nom_metrics[0],  # Simplify to one metric for robustness
                        color="Departamento",
                        labels={'value': 'Porcentaje (%)', 'variable': 'Métrica'},
                        height=450
                    )
                    fig.add_hline(y=nom_target, line_dash="dash", line_color=COLOR_PALETTE['warning'])
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        yaxis_range=[0, 100],
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    logger.error(f"Error rendering NOM-035 line chart: {e}")
                    st.warning(f"Error al renderizar gráfico: {e}", icon="⚠️")
        
        with col2:
            st.markdown("**📌 Resumen**", help="Resumen detallado de métricas NOM-035 por departamento")
            try:
                summary = filtered_nom.groupby('Departamento')[nom_metrics].mean().round(1)
                st.dataframe(summary.style.format('{:.1f}'), use_container_width=True)
            except Exception as e:
                logger.error(f"Error rendering NOM-035 summary: {e}")
                st.warning(f"Error al renderizar resumen: {e}", icon="⚠️")
    
    with nom_view2:
        with st.spinner("Cargando mapa de riesgo..."):
            try:
                logger.info("Rendering NOM-035 risk heatmap")
                metrics = nom_metrics
                z_values = filtered_nom.groupby('Departamento')[metrics].mean().values
                if z_values.size == 0:
                    logger.warning("NOM-035 heatmap data is empty")
                    st.warning("⚠️ No hay datos suficientes para renderizar el mapa de riesgo", icon="⚠️")
                    return
                fig_heat = go.Figure(data=go.Heatmap(
                    z=z_values.T,
                    x=filtered_nom['Departamento'].unique(),
                    y=metrics,
                    colorscale='RdYlGn'
                ))
                fig_heat.update_layout(title="Mapa de Riesgo Psicosocial", height=400)
                st.plotly_chart(fig_heat, use_container_width=True)
            except Exception as e:
                logger.error(f"Error rendering NOM-035 heatmap: {e}")
                st.warning(f"Error al renderizar mapa de riesgo: {e}", icon="⚠️")
    
    with nom_view3:
        try:
            logger.info("Rendering NOM-035 trends bar chart")
            trend_data = filtered_nom.groupby(['Departamento', pd.Grouper(key='Mes', freq='Y')])[nom_metrics].mean().reset_index()
            if trend_data.empty:
                logger.warning("NOM-035 trend data is empty")
                st.warning("⚠️ No hay datos suficientes para renderizar tendencias", icon="⚠️")
                return
            fig_trend = px.bar(
                trend_data,
                x='Departamento',
                y=nom_metrics[0],
                color='Departamento'
            )
            fig_trend.update_layout(title="Tendencia Anual de Cumplimiento", height=400)
            st.plotly_chart(fig_trend, use_container_width=True)
        except Exception as e:
            logger.error(f"Error rendering NOM-035 trends: {e}")
            st.warning(f"Error al renderizar tendencias: {e}", icon="⚠️")

def render_lean_tab(lean_df, departamentos_filtro, lean_target, start_date, end_date, lean_metrics):
    logger.info("Rendering LEAN tab")
    st.markdown("#### 🔄 Progreso LEAN 2.0", help="Seguimiento de métricas LEAN para optimización de procesos")
    
    if not lean_metrics:
        logger.warning("No LEAN metrics selected")
        st.warning("⚠️ Por favor, seleccione al menos una métrica LEAN en los filtros.", icon="⚠️")
        return
    
    filtered_lean = filter_dataframe(lean_df, departamentos_filtro, start_date, end_date)
    
    if filtered_lean.empty or not all(col in filtered_lean.columns for col in lean_metrics):
        logger.warning("Filtered LEAN DataFrame is empty or missing metrics")
        st.warning("⚠️ No hay datos disponibles para los filtros seleccionados o métricas faltantes", icon="⚠️")
        return
    
    col1, col2 = st.columns([3, 2], gap="medium")
    with col1:
        with st.spinner("Cargando gráfico de eficiencia..."):
            try:
                logger.info("Rendering LEAN area chart")
                grouped_data = filtered_lean.groupby(['Mes', 'Departamento'])[lean_metrics].mean().reset_index()
                if grouped_data.empty:
                    logger.warning("Grouped LEAN data is empty")
                    st.warning("⚠️ No hay datos suficientes para renderizar el gráfico de eficiencia", icon="⚠️")
                    return
                fig_lean = px.area(
                    grouped_data,
                    x='Mes',
                    y=lean_metrics[0],
                    color="Departamento",
                    labels={'value': 'Valor', 'variable': 'Métrica'},
                    height=450
                )
                fig_lean.add_hline(y=lean_target, line_dash="dash", line_color=COLOR_PALETTE['warning'])
                fig_lean.update_layout(
                    title="Evolución de Métricas LEAN",
                    yaxis_range=[0, 100],
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_lean, use_container_width=True)
            except Exception as e:
                logger.error(f"Error rendering LEAN area chart: {e}")
                st.warning(f"Error al renderizar gráfico de eficiencia: {e}", icon="⚠️")
    
    with col2:
        try:
            summary_cols = lean_metrics
            summary = filtered_lean.groupby('Departamento')[summary_cols].mean().round(1)
            st.dataframe(summary.style.format('{:.1f}'), use_container_width=True)
        except Exception as e:
            logger.error(f"Error rendering LEAN summary: {e}")
            st.warning(f"Error al renderizar resumen: {e}", icon="⚠️")

def render_wellbeing_tab(bienestar_df, start_date, end_date, wellbeing_target):
    logger.info("Rendering Wellbeing tab")
    st.markdown("#### 😊 Bienestar Organizacional", help="Indicadores de bienestar y clima laboral")
    filtered_bienestar = filter_dataframe(bienestar_df, [], start_date, end_date)
    
    if filtered_bienestar.empty:
        logger.warning("Filtered Wellbeing DataFrame is empty")
        st.warning("⚠️ No hay datos disponibles para el período seleccionado", icon="⚠️")
        return
    
    try:
        st.metric(
            label="Índice Bienestar",
            value=f"{filtered_bienestar['Índice Bienestar'].mean():.1f}%",
            help="Índice promedio de bienestar organizacional"
        )
    except Exception as e:
        logger.error(f"Error rendering Wellbeing metric: {e}")
        st.warning(f"Error al renderizar métrica: {e}", icon="⚠️")
    
    with st.spinner("Cargando tendencias..."):
        try:
            logger.info("Rendering Wellbeing line chart")
            fig_bienestar = px.line(
                filtered_bienestar,
                x='Mes',
                y='Índice Bienestar',
                markers=True,
                labels={'value': 'Porcentaje (%)'},
                height=400
            )
            fig_bienestar.add_hline(y=wellbeing_target, line_dash="dash", line_color=COLOR_PALETTE['warning'])
            fig_bienestar.update_layout(title="Evolución Mensual", yaxis_range=[0, 100])
            st.plotly_chart(fig_bienestar, use_container_width=True)
        except Exception as e:
            logger.error(f"Error rendering Wellbeing line chart: {e}")
            st.warning(f"Error al renderizar tendencias: {e}", icon="⚠️")

def render_action_plans_tab(departamentos_filtro, start_date, end_date):
    logger.info("Rendering Action Plans tab")
    st.markdown("#### 📝 Planes de Acción", help="Gestión de planes de acción para abordar problemas identificados")
    filtered_plans = filter_dataframe(st.session_state.action_plans_df, departamentos_filtro, start_date, end_date, date_column='Plazo')
    
    if filtered_plans.empty:
        logger.warning("Filtered Action Plans DataFrame is empty")
        st.info("ℹ️ No hay planes de acción para los filtros seleccionados", icon="ℹ️")
        return
    
    try:
        st.dataframe(filtered_plans[['Departamento', 'Problema', 'Acción']], use_container_width=True)
    except Exception as e:
        logger.error(f"Error rendering Action Plans table: {e}")
        st.warning(f"Error al renderizar planes: {e}", icon="⚠️")

# ========== EXPORT AND REPORTING ==========
def render_export_section(nom_df, lean_df, bienestar_df):
    logger.info("Rendering export section")
    st.markdown("---")
    st.markdown("#### 📤 Exportar y Reportes", help="Opciones para exportar datos y generar reportes")
    
    with st.expander("📊 Exportar Datos", expanded=True):
        export_format = st.radio("Formato", ["CSV", "Excel"], horizontal=True)
        data_options = st.multiselect("Datos", ["NOM-035", "LEAN", "Bienestar"], default=["NOM-035"])
        if st.button("💾 Descargar", use_container_width=True):
            if not data_options:
                st.markdown("<p class='error-message'>Seleccione al menos un tipo de datos</p>", unsafe_allow_html=True)
                return
            try:
                logger.info(f"Exporting data in {export_format} format with options: {data_options}")
                export_data = []
                logger.info(f"Type of export_data before append: {type(export_data)}")
                if "NOM-035" in data_options and not nom_df.empty:
                    if not isinstance(export_data, list):
                        logger.error(f"export_data is not a list: {type(export_data)}")
                        export_data = []
                    export_data.append(nom_df.assign(Tipo="NOM-035"))
                if "LEAN" in data_options and not lean_df.empty:
                    if not isinstance(export_data, list):
                        logger.error(f"export_data is not a list: {type(export_data)}")
                        export_data = []
                    export_data.append(lean_df.assign(Tipo="LEAN"))
                if "Bienestar" in data_options and not bienestar_df.empty:
                    if not isinstance(export_data, list):
                        logger.error(f"export_data is not a list: {type(export_data)}")
                        export_data = []
                    export_data.append(bienestar_df.assign(Tipo="Bienestar"))
                
                if not export_data:
                    logger.warning("No valid data to export")
                    st.warning("⚠️ No hay datos válidos para exportar", icon="⚠️")
                    return
                
                combined_data = pd.concat([df for df in export_data], ignore_index=True, sort=False)
                
                if export_format == "CSV":
                    data = combined_data.to_csv(index=False).encode('utf-8')
                    mime = "text/csv"
                    ext = "csv"
                else:
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        combined_data.to_excel(writer, index=False, sheet_name='Data')
                    data = output.getvalue()
                    mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    ext = "xlsx"
                
                st.download_button(
                    label=f"📥 Descargar .{ext}",
                    data=data,
                    file_name=f"nom_lean_data_{datetime.now().strftime('%Y%m%d')}.{ext}",
                    mime=mime,
                    use_container_width=True
                )
            except Exception as e:
                logger.error(f"Error exporting data: {e}")
                st.error(f"Error al exportar datos: {e}", icon="🚨")

# ========== MAIN FUNCTION ==========
def main():
    logger.info("Starting main function")
    try:
        sidebar_data = render_sidebar()
        if sidebar_data is None or sidebar_data[0] is None or sidebar_data[1] is None or not sidebar_data[2]:
            logger.warning("Invalid sidebar data")
            st.warning("⚠️ Por favor, configure los filtros en la barra lateral para continuar", icon="⚠️")
            return
        
        start_date, end_date, departamentos_filtro, targets, nom_metrics, lean_metrics = sidebar_data
        nom_target, lean_target, wellbeing_target, efficiency_target = targets
        
        render_header(start_date, end_date)
        
        st.markdown("### Indicadores Clave", help="Resumen de métricas clave para NOM-035, LEAN y bienestar")
        cols = st.columns(4, gap="medium")
        filtered_nom = filter_dataframe(nom_df, departamentos_filtro, start_date, end_date)
        filtered_lean = filter_dataframe(lean_df, departamentos_filtro, start_date, end_date)
        filtered_bienestar = filter_dataframe(bienestar_df, [], start_date, end_date)
        kpis = [
            (
                filtered_nom['Evaluaciones'].mean() if not filtered_nom.empty and 'Evaluaciones' in filtered_nom.columns else 0,
                "Cumplimiento NOM-035",
                nom_target,
                "📋",
                "Porcentaje promedio de cumplimiento con NOM-035",
                0
            ),
            (
                filtered_lean['Eficiencia'].mean() if not filtered_lean.empty and 'Eficiencia' in filtered_lean.columns else 0,
                "Adopción LEAN",
                lean_target,
                "🔄",
                "Nivel promedio de implementación de prácticas LEAN",
                0
            ),
            (
                filtered_bienestar['Índice Bienestar'].mean() if not filtered_bienestar.empty and 'Índice Bienestar' in filtered_bienestar.columns else 0,
                "Índice Bienestar",
                wellbeing_target,
                "😊",
                "Índice promedio de bienestar organizacional",
                0
            ),
            (
                filtered_lean['Eficiencia'].mean() if not filtered_lean.empty and 'Eficiencia' in filtered_lean.columns else 0,
                "Eficiencia Operativa",
                efficiency_target,
                "⚙️",
                "Eficiencia promedio de procesos operativos",
                0
            )
        ]
        for i, (value, title, target, icon, help_text, delta) in enumerate(kpis):
            with cols[i]:
                kpi_card(value, title, target, icon, help_text, delta)
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 NOM-035",
            "🔄 LEAN",
            "😊 Bienestar",
            "📝 Planes de Acción"
        ])
        
        with tab1:
            render_nom_tab(nom_df, departamentos_filtro, nom_target, start_date, end_date, nom_metrics)
        with tab2:
            render_lean_tab(lean_df, departamentos_filtro, lean_target, start_date, end_date, lean_metrics)
        with tab3:
            render_wellbeing_tab(bienestar_df, start_date, end_date, wellbeing_target)
        with tab4:
            render_action_plans_tab(departamentos_filtro, start_date, end_date)
        
        render_export_section(nom_df, lean_df, bienestar_df)
    
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        st.error(f"Error en el dashboard: {e}", icon="🚨")

if __name__ == "__main__":
    logger.info("Running Streamlit app")
    main()

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
    padding: 1.5rem;
    max-width: 1400px;
    margin: 0 auto;
}

[data-testid="stSidebar"] {
    background-color: var(--primary);
    color: white;
    padding: 1rem;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

h1, h2, h3, h4, h5, h6 {
    color: var(--primary);
    font-weight: 600;
    margin-bottom: 0.75rem;
}

.card {
    background-color: var(--card);
    border-radius: 8px;
    padding: 1rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    border: 1px solid var(--border);
}

.stButton > button {
    background-color: var(--secondary);
    color: white;
    border-radius: 6px;
    padding: 0.5rem 1rem;
    font-weight: 500;
}

.stButton > button:hover {
    background-color: var(--accent);
}

[data-baseweb="tab-list"] {
    gap: 0.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1rem;
}

[data-baseweb="tab"] {
    background-color: var(--card);
    border-radius: 6px 6px 0 0;
    padding: 0.5rem 1rem;
    color: var(--text);
    font-weight: 500;
}

[data-baseweb="tab"][aria-selected="true"] {
    background-color: var(--primary);
    color: white;
}

.stDataFrame {
    border-radius: 8px;
    border: 1px solid var(--border);
}

.stTextInput > div > input,
.stSelectbox > div > select,
.stDateInput > div > input {
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.5rem;
    background-color: var(--card);
    color: var(--text);
    visibility: visible;
    font-size: 0.9rem;
}

.stDateInput > div > input {
    color: var(--text);
    background-color: white;
}

[data-testid="stSidebar"] .stDateInput > div > input {
    color: white;
    background-color: rgba(255, 255, 255, 0.1);
}

.stTextInput > div > input:focus,
.stSelectbox > div > select:focus,
.stDateInput > div > input:focus {
    border-color: var(--secondary);
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
}

.error-message {
    color: var(--danger);
    font-size: 0.8rem;
    margin-top: 0.25rem;
}

.progress-bar {
    height: 6px;
    background-color: var(--border);
    border-radius: 3px;
    overflow: hidden;
}

.progress-bar-fill {
    height: 100%;
    transition: width 0.3s ease;
}

@media (max-width: 768px) {
    .main {
        padding: 0.75rem;
    }
    .card {
        padding: 0.75rem;
    }
    [data-baseweb="tab"] {
        padding: 0.4rem 0.8rem;
        font-size: 0.8rem;
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
        for dept in DEPARTMENTS:
            base_evals = np.linspace(80, 90, len(dates)) + np.random.normal(0, 3, len(dates))
            for i, date_val in enumerate(dates):
                nom_data.append({
                    'Departamento': dept,
                    'Mes': date_val,
                    'Evaluaciones': np.clip(base_evals[i], 70, 100).round(1),
                    'Capacitaciones': np.clip(base_evals[i] + np.random.normal(0, 5), 60, 100).round(1),
                    'Incidentes': np.clip(np.round(10 - base_evals[i] / 10 + np.random.normal(0, 1)), 0, 10),
                    'Satisfacción Laboral': np.clip(base_evals[i] + np.random.normal(0, 4), 65, 95).round(1)
                })
        nom_df = pd.DataFrame(nom_data)
        nom_df['Mes'] = pd.to_datetime(nom_df['Mes'])  # Ensure datetime64
        nom_df = nom_df.drop_duplicates(subset=['Departamento', 'Mes'])
        logger.info(f"NOM-035 DataFrame shape: {nom_df.shape}, Mes dtype: {nom_df['Mes'].dtype}, Duplicates: {nom_df.duplicated(subset=['Departamento', 'Mes']).sum()}")

        # LEAN Data (2022-2025, monthly)
        lean_data = []
        for dept in DEPARTMENTS:
            base_eff = np.linspace(75, 85, len(dates)) + np.random.normal(0, 4, len(dates))
            for i, date_val in enumerate(dates):
                lean_data.append({
                    'Departamento': dept,
                    'Mes': date_val,
                    'Eficiencia': np.clip(base_eff[i], 60, 95).round(1),
                    'Reducción MURI/MURA/MUDA': np.clip(base_eff[i] / 4 + np.random.normal(0, 3), 5, 25).round(1),
                    'Proyectos Activos': np.clip(np.round(base_eff[i] / 20 + np.random.normal(0, 1)), 1, 6),
                    '5S+2_Score': np.clip(base_eff[i] + np.random.normal(0, 5), 60, 100).round(1),
                    'Kaizen Colectivo': np.clip(base_eff[i] - np.random.normal(5, 5), 50, 90).round(1),
                    'Tiempo Ciclo': np.clip(100 - base_eff[i] + np.random.normal(0, 5), 10, 50).round(1)
                })
        lean_df = pd.DataFrame(lean_data)
        lean_df['Mes'] = pd.to_datetime(lean_df['Mes'])  # Ensure datetime64
        lean_df = lean_df.drop_duplicates(subset=['Departamento', 'Mes'])
        logger.info(f"LEAN DataFrame shape: {lean_df.shape}, Mes dtype: {lean_df['Mes'].dtype}, Duplicates: {lean_df.duplicated(subset=['Departamento', 'Mes']).sum()}")

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
        bienestar_df['Mes'] = pd.to_datetime(bienestar_df['Mes'])  # Ensure datetime64
        bienestar_df = bienestar_df.drop_duplicates(subset=['Mes'])
        logger.info(f"Bienestar DataFrame shape: {bienestar_df.shape}, Mes dtype: {bienestar_df['Mes'].dtype}, Duplicates: {bienestar_df.duplicated(subset=['Mes']).sum()}")

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
        action_plans['Plazo'] = pd.to_datetime(action_plans['Plazo'])  # Ensure datetime64
        action_plans = action_plans.drop_duplicates(subset=['ID', 'Departamento', 'Plazo'])
        logger.info(f"Action Plans DataFrame shape: {action_plans.shape}, Plazo dtype: {action_plans['Plazo'].dtype}, Duplicates: {action_plans.duplicated(subset=['ID', 'Departamento', 'Plazo']).sum()}")

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
        st.error("No se pudieron cargar los planes de acción.", icon="🚨")
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
    st.error("No se pudieron cargar los datos.", icon="🚨")
    st.stop()

# ========== HELPER FUNCTIONS ==========
def filter_dataframe(df, departamentos_filtro, start_date, end_date, date_column='Mes'):
    """Filter DataFrame by departments and date range, preserving datetime type and ensuring unique rows."""
    try:
        logger.info(f"Filtering DataFrame with date_column={date_column}")
        if df.empty or date_column not in df.columns:
            logger.warning(f"DataFrame is empty or {date_column} not in columns")
            return pd.DataFrame(columns=df.columns)
        
        # Ensure dates are in correct format
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
        
        # Convert date_column to datetime64
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        logger.info(f"{date_column} dtype after conversion: {df[date_column].dtype}")
        
        # Remove duplicates based on Departamento and date_column
        if 'Departamento' in df.columns:
            df = df.drop_duplicates(subset=['Departamento', date_column], keep='last')
        else:
            df = df.drop_duplicates(subset=[date_column], keep='last')
        logger.info(f"Duplicates removed, DataFrame shape: {df.shape}")
        
        # Filter
        mask = (
            (df[date_column] >= start_date) & 
            (df[date_column] <= end_date)
        )
        if 'Departamento' in df.columns and departamentos_filtro:
            mask &= df['Departamento'].isin(departamentos_filtro)
        
        filtered_df = df[mask]
        
        if filtered_df.empty:
            logger.warning("Filtered DataFrame is empty")
            return pd.DataFrame(columns=df.columns)
        
        logger.info(f"Filtered DataFrame shape: {filtered_df.shape}")
        return filtered_df
    except Exception as e:
        logger.error(f"Error filtering DataFrame: {e}")
        st.warning(f"Error al filtrar datos: {e}", icon="🚨")
        return pd.DataFrame(columns=df.columns)

# ========== SIDEBAR ==========
def render_sidebar():
    with st.sidebar:
        logger.info("Rendering sidebar")
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
            <span style="font-size: 1.5rem;">📊</span>
            <h2 style="margin: 0; color: white;">NOM-035 & LEAN 2.0</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        with st.expander("🔍 Filtros", expanded=True):
            st.markdown("**Período**")
            col1, col2 = st.columns(2)
            with col1:
                default_start = date(2022, 1, 1)
                start_date = st.date_input(
                    "Inicio",
                    value=default_start,
                    min_value=date(2022, 1, 1),
                    max_value=date(2025, 12, 31),
                    key="sidebar_date_start",
                    format="DD/MM/YYYY"
                )
                logger.info(f"Start date selected: {start_date}")
            with col2:
                default_end = date(2025, 12, 31)
                min_end_date = start_date if start_date >= date(2022, 1, 1) else default_start
                end_date = st.date_input(
                    "Fin",
                    value=default_end,
                    min_value=min_end_date,
                    max_value=date(2025, 12, 31),
                    key="sidebar_date_end",
                    format="DD/MM/YYYY"
                )
                logger.info(f"End date selected: {end_date}")
            
            if start_date > end_date:
                logger.warning("Start date is after end date")
                st.markdown("<p class='error-message'>La fecha de inicio no puede ser posterior a la fecha de fin</p>", unsafe_allow_html=True)
                return None, None, None, None, None, None
            
            st.markdown("**Departamentos**")
            departamentos_filtro = st.multiselect(
                "Seleccionar departamentos",
                options=DEPARTMENTS,
                default=['Producción', 'Calidad', 'Logística'],
                key="sidebar_dept_filter"
            )
            
            st.markdown("**Métricas**")
            nom_metrics = st.multiselect(
                "Métricas NOM-035",
                ['Evaluaciones', 'Capacitaciones', 'Incidentes', 'Satisfacción Laboral'],
                default=['Evaluaciones', 'Capacitaciones'],
                key="sidebar_nom_metrics"
            )
            lean_metrics = st.multiselect(
                "Métricas LEAN",
                ['Eficiencia', 'Reducción MURI/MURA/MUDA', 'Proyectos Activos', '5S+2_Score', 'Kaizen Colectivo', 'Tiempo Ciclo'],
                default=['Eficiencia', '5S+2_Score'],
                key="sidebar_lean_metrics"
            )
        
        with st.expander("⚙️ Metas", expanded=False):
            st.markdown("**Establecer Metas**")
            nom_target = st.slider("Meta NOM-035 (%)", 50, 100, 90)
            lean_target = st.slider("Meta LEAN (%)", 50, 100, 80)
            wellbeing_target = st.slider("Meta Bienestar (%)", 50, 100, 85)
            efficiency_target = st.slider("Meta Eficiencia (%)", 50, 100, 75)
        
        st.markdown("---")
        if st.button("🔄 Actualizar", use_container_width=True):
            logger.info("Clearing cache and rerunning app")
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #a0aec0; font-size: 0.7rem;">
            v3.2.0<br>
            © 2025 RH Analytics
        </div>
        """, unsafe_allow_html=True)
    
    logger.info("Sidebar rendered successfully")
    return start_date, end_date, departamentos_filtro, (nom_target, lean_target, wellbeing_target, efficiency_target), nom_metrics, lean_metrics

# ========== HEADER ==========
def render_header(start_date, end_date):
    logger.info("Rendering header")
    st.image("assets/FOBO2.png", width=100)  # Added image
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem; margin-bottom: 1.5rem;">
        <div>
            <h1 style="margin: 0;">Sistema Integral NOM-035-STPS-2018 & LEAN 2.0</h1>
            <p style="color: var(--muted); font-size: 1rem; margin: 0;">
                Monitoreo Estratégico (2022-2025)
            </p>
        </div>
        <div class="card" style="padding: 0.5rem 1rem;">
            <div style="font-size: 0.8rem; color: var(--primary);">
                {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}
            </div>
            <div style="font-size: 0.7rem; color: var(--muted);">
                Actualizado: {datetime.now().strftime('%d/%m/%Y %H:%M')}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ========== KPI CARDS ==========
def kpi_card(value, title, target, icon, delta=None):
    logger.info(f"Rendering KPI card: {title}")
    delta_value = delta if delta is not None else value - target
    percentage = min(100, (value / target * 100)) if target != 0 else 0
    status = "✅" if value >= target else "⚠" if value >= target - 10 else "❌"
    color = COLOR_PALETTE['success'] if value >= target else COLOR_PALETTE['warning'] if value >= target - 10 else COLOR_PALETTE['danger']
    delta_text = f"+{delta_value:.1f}%" if delta_value >= 0 else f"{delta_value:.1f}%"
    
    st.markdown(f"""
    <div class="card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="font-size: 0.9rem; color: var(--text);">{icon} {title}</div>
            <div style="font-size: 1rem; color: {color};">{status}</div>
        </div>
        <div style="font-size: 1.5rem; font-weight: 600; color: {color}; margin: 0.5rem 0;">
            {value:.1f}%
        </div>
        <div style="font-size: 0.8rem; color: var(--muted);">
            Meta: {target}% • {delta_text}
        </div>
        <div class="progress-bar">
            <div class="progress-bar-fill" style="width: {percentage}%; background: {color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ========== TABS ==========
def render_nom_tab(nom_df, departamentos_filtro, nom_target, start_date, end_date, nom_metrics):
    logger.info("Rendering NOM-035 tab")
    st.markdown("#### 📋 Cumplimiento NOM-035")
    
    if not nom_metrics:
        logger.warning("No NOM-035 metrics selected")
        st.warning("🚨 Seleccione al menos una métrica NOM-035.", icon="🚨")
        return
    
    filtered_nom = filter_dataframe(nom_df, departamentos_filtro, start_date, end_date)
    
    if filtered_nom.empty:
        logger.warning("Filtered NOM-035 DataFrame is empty")
        st.warning("🚨 No hay datos para los filtros seleccionados.", icon="🚨")
        return
    
    nom_view1, nom_view2, nom_view3 = st.tabs(["📊 Métricas", "🔍 Mapa de Riesgo", "📈 Tendencias"])
    
    with nom_view1:
        col1, col2 = st.columns([3, 2])
        with col1:
            with st.spinner("Cargando gráfico..."):
                try:
                    logger.info("Rendering NOM-035 line chart")
                    grouped_data = filtered_nom.groupby(['Mes', 'Departamento'])[nom_metrics].mean().reset_index()
                    melted_data = pd.melt(
                        grouped_data, 
                        id_vars=['Mes', 'Departamento'],
                        value_vars=nom_metrics,
                        var_name='Métrica',
                        value_name='Valor'
                    )
                    fig = px.line(
                        melted_data,
                        x="Mes",
                        y="Valor",
                        color="Departamento",
                        facet_col="Métrica",
                        facet_col_wrap=2,
                        color_discrete_sequence=[COLOR_PALETTE['primary'], COLOR_PALETTE['secondary'], COLOR_PALETTE['accent']],
                        labels={'Valor': '%'},
                        height=400
                    )
                    for i, metric in enumerate(nom_metrics):
                        fig.add_hline(
                            y=nom_target,
                            line_dash="dash",
                            line_color=COLOR_PALETTE['warning'],
                            annotation_text="Meta",
                            row=(i // 2) + 1,
                            col=(i % 2) + 1
                        )
                    fig.update_layout(
                        yaxis_range=[0, 100],
                        legend_title_text='Departamento',
                        margin=dict(l=20, r=20, t=40, b=20),
                        font=dict(family="Inter", size=12),
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    logger.error(f"Error rendering NOM-035 line chart: {e}")
                    st.warning(f"Error al renderizar gráfico: {e}", icon="🚨")
        
        with col2:
            st.markdown("**📌 Resumen**")
            try:
                summary_cols = [col for col in nom_metrics + ['Incidentes'] if col in filtered_nom.columns]
                if summary_cols:
                    summary = filtered_nom.groupby('Departamento')[summary_cols].mean().round(1)
                    format_dict = {col: '{:.1f}' for col in summary_cols}
                    st.dataframe(
                        summary.style.format(format_dict).background_gradient(cmap='RdYlGn'),
                        use_container_width=True,
                        height=400
                    )
                else:
                    st.info("ℹ️ No hay métricas disponibles.", icon="ℹ️")
            except Exception as e:
                logger.error(f"Error rendering NOM-035 summary: {e}")
                st.warning(f"Error al renderizar resumen: {e}", icon="🚨")
    
    with nom_view2:
        with st.spinner("Cargando mapa de riesgo..."):
            try:
                logger.info("Rendering NOM-035 risk heatmap")
                scaler = MinMaxScaler()
                metrics = [col for col in nom_metrics + ['Incidentes'] if col in filtered_nom.columns]
                if not metrics:
                    st.warning("🚨 No hay métricas para el mapa de riesgo.", icon="🚨")
                    return
                risk_data = filtered_nom.groupby('Departamento')[metrics].mean()
                z_values = scaler.fit_transform(risk_data)
                fig_heat = go.Figure(data=go.Heatmap(
                    z=z_values.T,
                    x=risk_data.index,
                    y=metrics,
                    colorscale=[[0, COLOR_PALETTE['danger']], [0.5, COLOR_PALETTE['warning']], [1, COLOR_PALETTE['success']]],
                    text=risk_data.values.T.round(1),
                    texttemplate="%{text:.1f}",
                    colorbar=dict(title="Nivel", tickvals=[0, 0.5, 1], ticktext=["Bajo", "Medio", "Alto"])
                ))
                fig_heat.update_layout(
                    title="Mapa de Riesgo Psicosocial",
                    height=400,
                    margin=dict(l=40, r=40, t=50, b=40),
                    font=dict(family="Inter", size=12)
                )
                st.plotly_chart(fig_heat, use_container_width=True)
                st.markdown("""
                <div class="card">
                    <p style="font-size: 0.8rem;">
                        <strong>Interpretación:</strong> Valores altos en métricas positivas indican buen cumplimiento.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                logger.error(f"Error rendering NOM-035 heatmap: {e}")
                st.warning(f"Error al renderizar mapa de riesgo: {e}", icon="🚨")
    
    with nom_view3:
        col1, col2 = st.columns([3, 1])
        with col1:
            with st.spinner("Cargando tendencias..."):
                try:
                    logger.info("Rendering NOM-035 trends bar chart")
                    trend_data = filtered_nom.copy()
                    logger.info(f"Trend data Mes dtype: {trend_data['Mes'].dtype}")
                    
                    # Ensure Mes is datetime64
                    if not pd.api.types.is_datetime64_any_dtype(trend_data['Mes']):
                        logger.warning("Mes column is not datetime64, attempting to convert")
                        trend_data['Mes'] = pd.to_datetime(trend_data['Mes'], errors='coerce')
                        if trend_data['Mes'].isna().all():
                            raise ValueError("No se pudo convertir la columna Mes a datetime")
                    
                    trend_data['Año'] = trend_data['Mes'].dt.year
                    trend_data = trend_data.groupby(['Departamento', 'Año'])[nom_metrics].mean().groupby('Departamento').pct_change().reset_index()
                    trend_data = trend_data.fillna(0)
                    melted_trend = pd.melt(
                        trend_data,
                        id_vars=['Departamento', 'Año'],
                        value_vars=nom_metrics,
                        var_name='Métrica',
                        value_name='Cambio'
                    )
                    fig_trend = px.bar(
                        melted_trend,
                        x='Departamento',
                        y='Cambio',
                        color='Métrica',
                        facet_col='Año',
                        color_discrete_sequence=[COLOR_PALETTE['primary'], COLOR_PALETTE['secondary']],
                        labels={'Cambio': 'Cambio Anual (%)'},
                        height=400
                    )
                    fig_trend.add_hline(
                        y=0,
                        line_dash="dash",
                        line_color=COLOR_PALETTE['muted'],
                        annotation_text="Neutral"
                    )
                    fig_trend.update_layout(
                        title="Tendencia Anual",
                        margin=dict(l=20, r=20, t=40, b=20),
                        font=dict(family="Inter", size=12)
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
                except Exception as e:
                    logger.error(f"Error rendering NOM-035 trends: {e}")
                    st.warning(f"Error al renderizar tendencias: {e}", icon="🚨")
        
        with col2:
            st.markdown("**📊 Interpretación**")
            st.markdown("""
            <div class="card">
                <p style="font-size: 0.8rem;"><span style="color: var(--success);">↑ Positivo:</span> Mejora</p>
                <p style="font-size: 0.8rem;"><span style="color: var(--danger);">↓ Negativo:</span> Atención</p>
                <p style="font-size: 0.8rem;"><span style="color: var(--warning);">→ Neutral:</span> Monitoreo</p>
            </div>
            """, unsafe_allow_html=True)

def render_lean_tab(lean_df, departamentos_filtro, lean_target, start_date, end_date, lean_metrics):
    logger.info("Rendering LEAN tab")
    st.markdown("#### 🔄 Progreso LEAN 2.0")
    
    if not lean_metrics:
        logger.warning("No LEAN metrics selected")
        st.warning("🚨 Seleccione al menos una métrica LEAN.", icon="🚨")
        return
    
    filtered_lean = filter_dataframe(lean_df, departamentos_filtro, start_date, end_date)
    
    if filtered_lean.empty:
        logger.warning("Filtered LEAN DataFrame is empty")
        st.warning("🚨 No hay datos para los filtros seleccionados.", icon="🚨")
        return
    
    col1, col2 = st.columns([3, 2])
    with col1:
        with st.spinner("Cargando gráfico..."):
            try:
                logger.info("Rendering LEAN line chart")
                grouped_data = filtered_lean.groupby(['Mes', 'Departamento'])[lean_metrics].mean().reset_index()
                melted_data = pd.melt(
                    grouped_data, 
                    id_vars=['Mes', 'Departamento'],
                    value_vars=lean_metrics,
                    var_name='Métrica',
                    value_name='Valor'
                )
                fig_lean = px.line(
                    melted_data,
                    x='Mes',
                    y='Valor',
                    color="Departamento",
                    facet_col="Métrica",
                    facet_col_wrap=2,
                    color_discrete_sequence=[COLOR_PALETTE['primary'], COLOR_PALETTE['secondary'], COLOR_PALETTE['accent']],
                    labels={'Valor': 'Valor'},
                    height=400
                )
                for i, metric in enumerate(lean_metrics):
                    fig_lean.add_hline(
                        y=lean_target,
                        line_dash="dash",
                        line_color=COLOR_PALETTE['warning'],
                        annotation_text="Meta",
                        row=(i // 2) + 1,
                        col=(i % 2) + 1
                    )
                fig_lean.update_layout(
                    yaxis_range=[0, 100],
                    margin=dict(l=20, r=20, t=40, b=20),
                    font=dict(family="Inter", size=12),
                    hovermode="x unified"
                )
                st.plotly_chart(fig_lean, use_container_width=True)
            except Exception as e:
                logger.error(f"Error rendering LEAN line chart: {e}")
                st.warning(f"Error al renderizar gráfico: {e}", icon="🚨")
        
        with st.spinner("Cargando análisis..."):
            try:
                logger.info("Rendering LEAN 3D scatter plot")
                grouped_lean = filtered_lean.groupby('Departamento')[lean_metrics].mean().reset_index()
                metrics_3d = lean_metrics[:3]
                if len(metrics_3d) < 3:
                    metrics_3d += [lean_metrics[0]] * (3 - len(metrics_3d))
                fig_scatter = px.scatter_3d(
                    grouped_lean,
                    x=metrics_3d[0],
                    y=metrics_3d[1],
                    z=metrics_3d[2],
                    color='Departamento',
                    size=np.ones(len(grouped_lean)) * 10,
                    hover_name='Departamento',
                    height=400
                )
                fig_scatter.update_layout(
                    title="Análisis Multidimensional",
                    margin=dict(l=20, r=20, t=40, b=20),
                    font=dict(family="Inter", size=12)
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            except Exception as e:
                logger.error(f"Error rendering LEAN 3D scatter: {e}")
                st.warning(f"Error al renderizar análisis: {e}", icon="🚨")
    
    with col2:
        st.markdown("**📊 Comparación de Métricas**")
        with st.spinner("Cargando radar..."):
            try:
                logger.info("Rendering LEAN radar chart")
                scaler = MinMaxScaler()
                lean_radar = filtered_lean.groupby('Departamento')[lean_metrics].mean().reset_index()
                
                # Ensure numeric data
                lean_radar[lean_metrics] = lean_radar[lean_metrics].apply(pd.to_numeric, errors='coerce').fillna(0)
                
                if lean_radar[lean_metrics].isna().all().all():
                    logger.warning("All LEAN metrics are NaN")
                    st.warning("🚨 No hay datos válidos para el radar.", icon="🚨")
                    return
                
                scaled_data = scaler.fit_transform(lean_radar[lean_metrics])
                lean_radar[lean_metrics] = scaled_data
                
                fig_radar = go.Figure()
                for _, row in lean_radar.iterrows():
                    values = [row[m] for m in lean_metrics]
                    if np.isnan(values).any():
                        logger.warning(f"Skipping {row['Departamento']} due to NaN values")
                        continue
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values,
                        theta=lean_metrics,
                        fill='toself',
                        name=row['Departamento'],
                        line=dict(width=2)
                    ))
                
                if not fig_radar.data:
                    logger.warning("No valid data for radar chart")
                    st.warning("🚨 No hay datos suficientes para el radar.", icon="🚨")
                    return
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
                        angularaxis=dict(tickfont_size=10)
                    ),
                    height=400,
                    showlegend=True,
                    margin=dict(l=40, r=40, t=20, b=40),
                    font=dict(family="Inter", size=12)
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            except Exception as e:
                logger.error(f"Error rendering LEAN radar: {e}")
                st.warning(f"Error al renderizar radar: {e}", icon="🚨")
        
        st.markdown("**📌 Detalle de Proyectos**")
        with st.expander("📌 Detalle", expanded=True):
            try:
                # Ensure unique columns
                summary_cols = list(set(lean_metrics + ['Proyectos Activos'] if 'Proyectos Activos' not in lean_metrics else lean_metrics))
                summary_cols = [col for col in summary_cols if col in filtered_lean.columns]
                logger.info(f"Summary columns: {summary_cols}")
                
                if summary_cols:
                    summary = filtered_lean.groupby('Departamento')[summary_cols].mean().round(1).reset_index()
                    logger.info(f"Summary DataFrame index is unique: {summary.index.is_unique}, columns: {summary.columns.tolist()}")
                    
                    # Apply formatting only to numeric columns
                    format_dict = {col: '{:.1f}' for col in summary_cols if pd.api.types.is_numeric_dtype(summary[col])}
                    format_dict['Departamento'] = '{}'  # Ensure Departamento is not formatted as float
                    try:
                        st.dataframe(
                            summary.style.background_gradient(cmap='Greens', subset=summary_cols).format(format_dict),
                            use_container_width=True
                        )
                    except Exception as e:
                        logger.warning(f"Styling failed: {e}, displaying without styling")
                        st.dataframe(
                            summary,
                            use_container_width=True
                        )
                        st.warning(f"Error al aplicar estilo al resumen: {e}", icon="🚨")
                else:
                    st.info("ℹ️ No hay métricas seleccionadas.", icon="ℹ️")
            except Exception as e:
                logger.error(f"Error rendering LEAN summary: {e}")
                st.warning(f"Error al renderizar detalle: {e}", icon="🚨")

def render_wellbeing_tab(bienestar_df, start_date, end_date, wellbeing_target):
    logger.info("Rendering Wellbeing tab")
    st.markdown("#### 😊 Bienestar Organizacional")
    
    # Validate date range
    min_date = bienestar_df['Mes'].min().date()
    max_date = bienestar_df['Mes'].max().date()
    start_date = max(start_date, min_date)
    end_date = min(end_date, max_date)
    logger.info(f"Adjusted date range: {start_date} to {end_date}")
    
    filtered_bienestar = filter_dataframe(bienestar_df, [], start_date, end_date)
    
    if filtered_bienestar.empty:
        logger.warning("Filtered Wellbeing DataFrame is empty, using full dataset")
        filtered_bienestar = bienestar_df.copy()
        st.warning("🚨 No hay datos para el período seleccionado, mostrando todos los datos.", icon="🚨")
    
    logger.info(f"Filtered Wellbeing DataFrame shape: {filtered_bienestar.shape}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        try:
            delta = filtered_bienestar['Encuestas'].iloc[-1] - filtered_bienestar['Encuestas'].iloc[0] if len(filtered_bienestar) > 1 else 0
            st.metric(
                label="Encuestas Completadas",
                value=f"{filtered_bienestar['Encuestas'].mean():.0f}%",
                delta=f"{delta:+.0f}%"
            )
        except Exception as e:
            logger.error(f"Error rendering Encuestas metric: {e}")
            st.warning(f"Error: {e}", icon="🚨")
    with col2:
        try:
            delta = filtered_bienestar['Ausentismo'].iloc[-1] - filtered_bienestar['Ausentismo'].iloc[0] if len(filtered_bienestar) > 1 else 0
            st.metric(
                label="Ausentismo",
                value=f"{filtered_bienestar['Ausentismo'].iloc[-1]:.1f}%",
                delta=f"{delta:+.1f}%",
                delta_color="inverse"
            )
        except Exception as e:
            logger.error(f"Error rendering Ausentismo metric: {e}")
            st.warning(f"Error: {e}", icon="🚨")
    with col3:
        try:
            delta = filtered_bienestar['Rotación'].iloc[-1] - filtered_bienestar['Rotación'].iloc[0] if len(filtered_bienestar) > 1 else 0
            st.metric(
                label="Rotación",
                value=f"{filtered_bienestar['Rotación'].iloc[-1]:.1f}%",
                delta=f"{delta:+.1f}%",
                delta_color="inverse"
            )
        except Exception as e:
            logger.error(f"Error rendering Rotación metric: {e}")
            st.warning(f"Error: {e}", icon="🚨")
    
    wellbeing_view1, wellbeing_view2 = st.tabs(["📈 Tendencias", "🔍 Correlaciones"])
    
    with wellbeing_view1:
        with st.spinner("Cargando tendencias..."):
            try:
                logger.info("Rendering Wellbeing line chart")
                metrics = [col for col in ['Índice Bienestar', 'Ausentismo', 'Rotación', 'Engagement'] if col in filtered_bienestar.columns]
                if not metrics:
                    logger.warning("No valid metrics for Wellbeing line chart")
                    st.warning("🚨 No hay métricas disponibles.", icon="🚨")
                    return
                
                filtered_bienestar[metrics] = filtered_bienestar[metrics].apply(pd.to_numeric, errors='coerce').fillna(0)
                logger.info(f"Valid metrics: {metrics}")
                
                if len(filtered_bienestar) < 1:
                    logger.warning("No valid data for Wellbeing line chart")
                    st.warning("🚨 No hay datos válidos.", icon="🚨")
                    return
                
                fig_bienestar = px.line(
                    filtered_bienestar,
                    x='Mes',
                    y=metrics,
                    markers=True,
                    color_discrete_sequence=[
                        COLOR_PALETTE['success'],
                        COLOR_PALETTE['danger'],
                        COLOR_PALETTE['warning'],
                        COLOR_PALETTE['accent']
                    ],
                    labels={'value': '%', 'variable': 'Métrica'},
                    height=400
                )
                fig_bienestar.add_hline(
                    y=wellbeing_target,
                    line_dash="dash",
                    line_color=COLOR_PALETTE['warning'],
                    annotation_text="Meta"
                )
                fig_bienestar.update_layout(
                    title="Evolución Mensual",
                    yaxis_range=[0, 100],
                    legend_title="Métrica",
                    margin=dict(l=20, r=20, t=40, b=20),
                    font=dict(family="Inter", size=12),
                    hovermode="x unified"
                )
                st.plotly_chart(fig_bienestar, use_container_width=True)
            except Exception as e:
                logger.error(f"Error rendering Wellbeing line chart: {e}")
                st.warning(f"Error al renderizar tendencias: {e}", icon="🚨")
    
    with wellbeing_view2:
        with st.spinner("Cargando correlaciones..."):
            try:
                logger.info("Rendering Wellbeing correlation matrix")
                metrics = [col for col in ['Índice Bienestar', 'Ausentismo', 'Rotación', 'Encuestas', 'Engagement'] if col in filtered_bienestar.columns]
                if len(metrics) < 2:
                    logger.warning("Not enough metrics for correlation matrix")
                    st.warning("🚨 No hay suficientes métricas.", icon="🚨")
                    return
                
                filtered_bienestar[metrics] = filtered_bienestar[metrics].apply(pd.to_numeric, errors='coerce')
                filtered_bienestar = filtered_bienestar.dropna(subset=metrics)
                
                if filtered_bienestar.empty:
                    logger.warning("No valid data for correlation matrix")
                    st.warning("🚨 No hay datos válidos.", icon="🚨")
                    return
                
                corr_matrix = filtered_bienestar[metrics].corr()
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    color_continuous_scale=[[0, COLOR_PALETTE['danger']], [0.5, COLOR_PALETTE['warning']], [1, COLOR_PALETTE['success']]],
                    range_color=[-1, 1],
                    labels=dict(color="Correlación"),
                    height=400
                )
                fig_corr.update_layout(
                    title="Matriz de Correlación",
                    margin=dict(l=40, r=40, t=50, b=40),
                    font=dict(family="Inter", size=12)
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                st.markdown("""
                <div class="card">
                    <p style="font-size: 0.8rem;">
                        <strong>Interpretación:</strong> Valores cercanos a 1 indican correlación positiva, -1 negativa, 0 ninguna.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                logger.error(f"Error rendering Wellbeing correlation: {e}")
                st.warning(f"Error al renderizar correlaciones: {e}", icon="🚨")

def render_action_plans_tab(departamentos_filtro, start_date, end_date):
    logger.info("Rendering Action Plans tab")
    st.markdown("#### 📝 Planes de Acción")
    filtered_plans = filter_dataframe(st.session_state.action_plans_df, departamentos_filtro, start_date, end_date, date_column='Plazo')
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**📌 Planes Registrados**")
        if filtered_plans.empty:
            logger.warning("Filtered Action Plans DataFrame is empty")
            st.info("ℹ️ No hay planes para los filtros seleccionados.", icon="ℹ️")
        else:
            def progress_bar(row):
                color = COLOR_PALETTE['success'] if row['% Avance'] == 100 else COLOR_PALETTE['warning'] if row['% Avance'] > 0 else COLOR_PALETTE['danger']
                return f'<div class="progress-bar"><div class="progress-bar-fill" style="width: {row["% Avance"]}%; background: {color};"></div></div>'
            
            styled_plans = filtered_plans.copy()
            styled_plans['Progreso'] = styled_plans.apply(progress_bar, axis=1)
            try:
                st.dataframe(
                    styled_plans.style.apply(
                        lambda x: [
                            f"background-color: {COLOR_PALETTE['success']}; color: white" if v == 'Completado'
                            else f"background-color: {COLOR_PALETTE['warning']}" if v == 'En progreso'
                            else f"background-color: {COLOR_PALETTE['danger']}; color: white"
                            for v in x
                        ], subset=['Estado']
                    ).format({
                        'Plazo': lambda x: x.strftime('%d/%m/%Y') if pd.notnull(x) else '',
                        '% Avance': '{:.0f}%',
                        'Costo Estimado': 'MXN {:,.0f}'
                    }),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'Progreso': st.column_config.TextColumn('Progreso', width="small")
                    },
                    height=400
                )
            except Exception as e:
                logger.error(f"Error rendering Action Plans table: {e}")
                st.warning(f"Error al renderizar planes: {e}", icon="🚨")
    
    with col2:
        st.markdown("**📊 Resumen por Estado**")
        if not filtered_plans.empty:
            try:
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
                    height=250
                )
                fig_status.update_layout(
                    margin=dict(l=20, r=20, t=20, b=20),
                    font=dict(family="Inter", size=12)
                )
                st.plotly_chart(fig_status, use_container_width=True)
            except Exception as e:
                logger.error(f"Error rendering Action Plans pie chart: {e}")
                st.warning(f"Error al renderizar resumen: {e}", icon="🚨")
        
        st.markdown("**📅 Vencimientos Próximos**")
        today = date.today()
        upcoming = filtered_plans[filtered_plans['Plazo'].dt.date <= today + timedelta(days=30)]
        if not upcoming.empty:
            for _, row in upcoming.iterrows():
                days_left = (row['Plazo'].date() - today).days
                color = COLOR_PALETTE['danger'] if days_left < 7 else COLOR_PALETTE['warning']
                text = f"Vence en {days_left} días" if days_left > 0 else f"Vencido hace {-days_left} días"
                st.markdown(f"""
                <div class="card">
                    <div style="font-weight: 600;">{row['Departamento']}</div>
                    <div style="font-size: 0.8rem; color: var(--muted);">{row['Problema'][:30]}...</div>
                    <div style="font-size: 0.8rem; color: {color};">{text}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("ℹ️ No hay vencimientos próximos.", icon="ℹ️")
    
    with st.expander("➕ Nuevo Plan", expanded=False):
        with st.form("nuevo_plan_form", clear_on_submit=True):
            st.markdown("**Registrar Nuevo Plan**")
            col1, col2 = st.columns(2)
            with col1:
                dept = st.selectbox("Departamento", DEPARTMENTS)
                problema = st.text_area("Problema", max_chars=200)
                prioridad = st.selectbox("Prioridad", ["Alta", "Media", "Baja"])
                costo = st.number_input("Costo Estimado (MXN)", min_value=0, value=10000, step=1000)
            with col2:
                accion = st.text_area("Acción", max_chars=200)
                responsable = st.text_input("Responsable")
                plazo = st.date_input(
                    "Plazo",
                    min_value=today,
                    value=today + timedelta(days=30),
                    format="DD/MM/YYYY"
                )
                avance = st.slider("% Avance", 0, 100, 0)
            
            submitted = st.form_submit_button("💾 Guardar", use_container_width=True)
            
            if submitted:
                errors = []
                if not problema:
                    errors.append("Problema es obligatorio")
                if len(problema) > 200:
                    errors.append("Problema no puede exceder 200 caracteres")
                if not accion:
                    errors.append("Acción es obligatoria")
                if len(accion) > 200:
                    errors.append("Acción no puede exceder 200 caracteres")
                if not responsable:
                    errors.append("Responsable es obligatorio")
                if not re.match(r'^[A-Za-z\s]+$', responsable):
                    errors.append("Responsable solo letras y espacios")
                if plazo < today:
                    errors.append("Plazo no puede ser anterior a hoy")
                
                if errors:
                    for error in errors:
                        st.markdown(f"<p class='error-message'>{error}</p>", unsafe_allow_html=True)
                else:
                    try:
                        new_plan = pd.DataFrame([{
                            'ID': len(st.session_state.action_plans_df) + 1,
                            'Departamento': dept,
                            'Problema': problema,
                            'Acción': accion,
                            'Responsable': responsable,
                            'Plazo': pd.Timestamp(plazo),
                            'Estado': 'Pendiente' if avance == 0 else 'En progreso' if avance < 100 else 'Completado',
                            'Prioridad': prioridad,
                            '% Avance': avance,
                            'Costo Estimado': costo
                        }])
                        st.session_state.action_plans_df = pd.concat([st.session_state.action_plans_df, new_plan], ignore_index=True)
                        st.success("✅ Plan registrado.", icon="✅")
                        st.rerun()
                    except Exception as e:
                        logger.error(f"Error registering new plan: {e}")
                        st.error(f"Error al registrar plan: {e}", icon="🚨")

# ========== EXPORT AND REPORTING ==========
def render_export_section(nom_df, lean_df, bienestar_df):
    logger.info("Rendering export section")
    st.markdown("---")
    st.markdown("#### 📤 Exportar y Reportes")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.expander("📄 Generar Reporte PDF", expanded=False):
            st.markdown("**Configurar Reporte**")
            report_type = st.selectbox("Tipo", ["Completo", "Resumido", "NOM-035", "LEAN", "Bienestar"])
            include_charts = st.checkbox("Incluir gráficos", value=True)
            if st.button("🖨️ Generar", use_container_width=True):
                st.warning("🚨 Generación de PDF no implementada.", icon="🚨")
    
    with col2:
        with st.expander("📧 Enviar por Correo", expanded=False):
            st.markdown("**Enviar Reporte**")
            email = st.text_input("Correo", placeholder="usuario@empresa.com")
            subject = st.text_input("Asunto", value="Reporte NOM-035 & LEAN")
            if st.button("📤 Enviar", use_container_width=True):
                if not email:
                    st.markdown("<p class='error-message'>Correo obligatorio</p>", unsafe_allow_html=True)
                elif not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email):
                    st.markdown("<p class='error-message'>Correo inválido</p>", unsafe_allow_html=True)
                elif not subject:
                    st.markdown("<p class='error-message'>Asunto obligatorio</p>", unsafe_allow_html=True)
                else:
                    st.warning("🚨 Envío de correo no implementado.", icon="🚨")
    
    with col3:
        with st.expander("📊 Exportar Datos", expanded=False):
            st.markdown("**Exportar Datos**")
            export_format = st.radio("Formato", ["CSV", "Excel", "JSON"], horizontal=True)
            data_options = st.multiselect("Datos", ["NOM-035", "LEAN 2.0", "Bienestar", "Planes de Acción"], default=["NOM-035", "LEAN 2.0", "Bienestar"])
            if st.button("💾 Descargar", use_container_width=True):
                if not data_options:
                    st.markdown("<p class='error-message'>Seleccione al menos un tipo de datos</p>", unsafe_allow_html=True)
                else:
                    with st.spinner("Preparando datos..."):
                        try:
                            export_data = []
                            if "NOM-035" in data_options and not nom_df.empty:
                                export_data.append(nom_df.assign(Tipo="NOM-035"))
                            if "LEAN 2.0" in data_options and not lean_df.empty:
                                export_data.append(lean_df.assign(Tipo="LEAN"))
                            if "Bienestar" in data_options and not bienestar_df.empty:
                                export_data.append(bienestar_df.assign(Tipo="Bienestar"))
                            if "Planes de Acción" in data_options and not st.session_state.action_plans_df.empty:
                                export_data.append(st.session_state.action_plans_df.assign(Tipo="Planes_Accion"))
                            
                            if not export_data:
                                logger.warning("No valid data to export")
                                st.warning("🚨 No hay datos válidos.", icon="🚨")
                                return
                            
                            combined_data = pd.concat([df for df in export_data], ignore_index=True, sort=False)
                            
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
                            
                            st.success(f"✅ Datos exportados como {export_format}.", icon="✅")
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
            st.warning("🚨 Configure los filtros en la barra lateral.", icon="🚨")
            return
        
        start_date, end_date, departamentos_filtro, targets, nom_metrics, lean_metrics = sidebar_data
        nom_target, lean_target, wellbeing_target, efficiency_target = targets
        
        render_header(start_date, end_date)
        
        st.markdown("### Indicadores Clave")
        cols = st.columns(4)
        filtered_nom = filter_dataframe(nom_df, departamentos_filtro, start_date, end_date)
        filtered_lean = filter_dataframe(lean_df, departamentos_filtro, start_date, end_date)
        filtered_bienestar = filter_dataframe(bienestar_df, [], start_date, end_date)
        kpis = [
            (
                filtered_nom['Evaluaciones'].mean() if not filtered_nom.empty and 'Evaluaciones' in filtered_nom.columns else 0,
                "Cumplimiento NOM-035",
                nom_target,
                "📋",
                filtered_nom['Evaluaciones'].mean() - filtered_nom.groupby('Departamento')['Evaluaciones'].mean().shift(1).mean() if not filtered_nom.empty and 'Evaluaciones' in filtered_nom.columns else 0
            ),
            (
                filtered_lean['Eficiencia'].mean() if not filtered_lean.empty and 'Eficiencia' in filtered_lean.columns else 0,
                "Adopción LEAN 2.0",
                lean_target,
                "🔄",
                filtered_lean['Eficiencia'].mean() - filtered_lean.groupby('Departamento')['Eficiencia'].mean().shift(1).mean() if not filtered_lean.empty and 'Eficiencia' in filtered_lean.columns else 0
            ),
            (
                filtered_bienestar['Índice Bienestar'].mean() if not filtered_bienestar.empty and 'Índice Bienestar' in filtered_bienestar.columns else 0,
                "Índice Bienestar",
                wellbeing_target,
                "😊",
                filtered_bienestar['Índice Bienestar'].mean() - filtered_bienestar['Índice Bienestar'].shift(1).mean() if not filtered_bienestar.empty and 'Índice Bienestar' in filtered_bienestar.columns else 0
            ),
            (
                filtered_lean['Eficiencia'].mean() if not filtered_lean.empty and 'Eficiencia' in filtered_lean.columns else 0,
                "Eficiencia Operativa",
                efficiency_target,
                "⚙️",
                filtered_lean['Eficiencia'].mean() - filtered_lean.groupby('Departamento')['Eficiencia'].mean().shift(1).mean() if not filtered_lean.empty and 'Eficiencia' in filtered_lean.columns else 0
            )
        ]
        for i, (value, title, target, icon, delta) in enumerate(kpis):
            with cols[i]:
                kpi_card(value, title, target, icon, delta)
        
        tab1, tab2, tab3, tab4 = st.tabs(["📋 NOM-035", "🔄 LEAN 2.0", "😊 Bienestar", "📝 Planes de Acción"])
        
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
        st.error(f"Error en la aplicación: {e}", icon="🚨")

if __name__ == "__main__":
    main()

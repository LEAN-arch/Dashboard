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

        return nom_df, lean_df, bienestar_df, action_plans
    except Exception as e:
        st.error(f"Error al cargar datos: {e}", icon="🚨")
        return None, None, None, None

# Initialize session state
if 'action_plans_df' not in st.session_state:
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
    st.stop()

# ========== HELPER FUNCTIONS ==========
def filter_dataframe(df, departamentos_filtro, start_date, end_date, date_column='Mes'):
    """Filter DataFrame by departments and date range."""
    try:
        if date_column not in df.columns:
            return df
        filtered_df = df[
            (df['Departamento'].isin(departamentos_filtro) if 'Departamento' in df.columns else True) &
            (df[date_column].dt.date >= start_date) &
            (df[date_column].dt.date <= end_date)
        ]
        return filtered_df
    except Exception as e:
        st.warning(f"Error al filtrar datos: {e}", icon="⚠️")
        return pd.DataFrame()

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
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #a0aec0; font-size: 0.75rem;">
            v3.2.0<br>
            © 2025 RH Analytics
        </div>
        """, unsafe_allow_html=True)
    
    return start_date, end_date, departamentos_filtro, (nom_target, lean_target, wellbeing_target, efficiency_target), nom_metrics, lean_metrics

# ========== HEADER ==========
def render_header(start_date, end_date):
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
    st.markdown("#### 📋 Cumplimiento NOM-035", help="Monitorea el cumplimiento de la NOM-035 en factores psicosociales")
    
    if not nom_metrics:
        st.warning("⚠️ Por favor, seleccione al menos una métrica NOM-035 en los filtros.", icon="⚠️")
        return
    
    filtered_nom = filter_dataframe(nom_df, departamentos_filtro, start_date, end_date)
    
    if filtered_nom.empty:
        st.warning("⚠️ No hay datos disponibles para los filtros seleccionados", icon="⚠️")
        return
    
    nom_view1, nom_view2, nom_view3 = st.tabs(["📊 Métricas", "🔍 Mapa de Riesgo", "📈 Tendencias"])
    
    with nom_view1:
        col1, col2 = st.columns([3, 2], gap="medium")
        with col1:
            with st.spinner("Cargando gráfico..."):
                try:
                    # Group by month and department, then calculate mean for selected metrics
                    grouped_data = filtered_nom.groupby(['Mes', 'Departamento'])[nom_metrics].mean().reset_index()
                    
                    # Melt the dataframe to long format for Plotly
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
                        line_group="Departamento",
                        facet_col="Métrica",
                        facet_col_wrap=2,
                        color_discrete_sequence=[COLOR_PALETTE['primary'], COLOR_PALETTE['secondary'], COLOR_PALETTE['accent']],
                        labels={'Valor': 'Porcentaje (%)'},
                        height=450
                    )
                    
                    # Add target line to each subplot
                    for annotation in fig.layout.annotations:
                        if annotation.text in nom_metrics:
                            fig.add_hline(
                                y=nom_target,
                                line_dash="dash",
                                line_color=COLOR_PALETTE['warning'],
                                annotation_text="Meta",
                                annotation_position="top right",
                                row=1,
                                col=int(annotation.text.split("=")[-1]) ) # <-- This was missing the closing parenthesis
                    
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        yaxis_range=[0, 100],
                        legend_title_text='Departamento',
                        xaxis_title="",
                        margin=dict(l=30, r=30, t=50, b=30),
                        font=dict(family="Inter", size=13, color=COLOR_PALETTE['text']),
                        showlegend=True,
                        hoverlabel=dict(bgcolor=COLOR_PALETTE['card'], font_size=12),
                        hovermode="x unified"
                    )
                    
                    # Update y-axis for each subplot
                    for i in range(len(nom_metrics)):
                        fig.update_yaxes(range=[0, 100], row=1, col=i+1)
                    
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Error al renderizar gráfico: {e}", icon="⚠️")
        
                    with col2:
                        st.markdown("**📌 Resumen**", help="Resumen detallado de métricas NOM-035 por departamento")
                        
                        # Get summary data and ensure proper formatting
                        summary = filtered_nom.groupby('Departamento')[nom_metrics + ['Incidentes']].mean().round(1).reset_index()
                        
                        # Create format dictionary for numeric columns only
                        format_dict = {
                            col: '{:.1f}' for col in summary.columns 
                            if pd.api.types.is_numeric_dtype(summary[col]) and col != 'Departamento'
                        }
                        
                        # Create styled DataFrame
                        styled_df = summary.style.format(format_dict)
                        
                        # Apply background gradient only to numeric columns in nom_metrics
                        gradient_cols = [
                            col for col in nom_metrics 
                            if col in summary.columns and pd.api.types.is_numeric_dtype(summary[col])
                        ]
                        
                        if gradient_cols:
                            styled_df = styled_df.background_gradient(
                                cmap='RdYlGn',
                                subset=gradient_cols
                            )
                        
                        # Display the styled DataFrame
                        st.dataframe(
                            styled_df,
                            use_container_width=True,
                            height=450
                        )
                        
                        # Display the styled DataFrame
                        st.dataframe(
                            styled_summary,
                            use_container_width=True,
                            height=450
                        )
    
    with nom_view2:
        with st.spinner("Cargando mapa de riesgo..."):
            try:
                scaler = MinMaxScaler()
                metrics = nom_metrics + ['Incidentes']
                risk_data = filtered_nom.groupby('Departamento')[metrics].mean()
                z_values = scaler.fit_transform(risk_data)
                
                fig_heat = go.Figure(data=go.Heatmap(
                    z=z_values.T,
                    x=risk_data.index,
                    y=metrics,
                    colorscale=[[0, COLOR_PALETTE['danger']], [0.5, COLOR_PALETTE['warning']], [1, COLOR_PALETTE['success']]],
                    hoverongaps=False,
                    text=risk_data.values.T.round(1),
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
                        <strong>Interpretación:</strong> Valores altos en métricas positivas indican buen cumplimiento, mientras que Incidentes altos señalan riesgos.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Error al renderizar mapa de riesgo: {e}", icon="⚠️")
    
    with nom_view3:
        col1, col2 = st.columns([3, 1], gap="medium")
        with col1:
            with st.spinner("Cargando tendencias..."):
                try:
                    # Group by department and year, then calculate mean and pct_change
                    trend_data = filtered_nom.copy()
                    trend_data['Año'] = trend_data['Mes'].dt.year
                    trend_data = trend_data.groupby(['Departamento', 'Año'])[nom_metrics].mean().groupby('Departamento').pct_change().reset_index()
                    
                    # Melt the dataframe for plotting
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
                        height=450
                    )
                    fig_trend.add_hline(
                        y=0,
                        line_dash="dash",
                        line_color=COLOR_PALETTE['muted'],
                        annotation_text="Neutral",
                        annotation_position="top right"
                    )
                    fig_trend.update_layout(
                        title="Tendencia Anual de Cumplimiento",
                        yaxis_title="Cambio (%)",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=30, r=30, t=50, b=30),
                        font=dict(family="Inter", size=13, color=COLOR_PALETTE['text']),
                        showlegend=True,
                        hoverlabel=dict(bgcolor=COLOR_PALETTE['card'], font_size=12)
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
                except Exception as e:
                    st.warning(f"Error al renderizar tendencias: {e}", icon="⚠️")
        
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

def render_lean_tab(lean_df, departamentos_filtro, lean_target, start_date, end_date, lean_metrics):
    st.markdown("#### 🔄 Progreso LEAN 2.0", help="Seguimiento de métricas LEAN para optimización de procesos")
    
    if not lean_metrics:
        st.warning("⚠️ Por favor, seleccione al menos una métrica LEAN en los filtros.", icon="⚠️")
        return
    
    filtered_lean = filter_dataframe(lean_df, departamentos_filtro, start_date, end_date)
    
    if filtered_lean.empty:
        st.warning("⚠️ No hay datos disponibles para los filtros seleccionados", icon="⚠️")
        return
    
    col1, col2 = st.columns([3, 2], gap="medium")
    with col1:
        with st.spinner("Cargando gráfico de eficiencia..."):
            try:
                # Group by month and department, then calculate mean for selected metrics
                grouped_data = filtered_lean.groupby(['Mes', 'Departamento'])[lean_metrics].mean().reset_index()
                
                # Melt the dataframe to long format for Plotly
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
                    line_group="Departamento",
                    facet_col="Métrica",
                    facet_col_wrap=2,
                    color_discrete_sequence=[COLOR_PALETTE['primary'], COLOR_PALETTE['secondary'], COLOR_PALETTE['accent']],
                    labels={'Valor': 'Valor'},
                    height=450
                )
                
                # Add target line to each subplot
                for annotation in fig_lean.layout.annotations:
                    if annotation.text in lean_metrics:
                        fig_lean.add_hline(
                            y=lean_target,
                            line_dash="dash",
                            line_color=COLOR_PALETTE['warning'],
                            annotation_text="Meta",
                            annotation_position="top right",
                            row=1,
                            col=int(annotation.text.split("=")[-1]))
                
                fig_lean.update_layout(
                    title="Evolución de Métricas LEAN",
                    yaxis_range=[0, 100],
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=30, r=30, t=50, b=30),
                    font=dict(family="Inter", size=13, color=COLOR_PALETTE['text']),
                    hoverlabel=dict(bgcolor=COLOR_PALETTE['card'], font_size=12),
                    hovermode="x unified"
                )
                
                # Update y-axis for each subplot
                for i in range(len(lean_metrics)):
                    fig_lean.update_yaxes(range=[0, 100], row=1, col=i+1)
                
                st.plotly_chart(fig_lean, use_container_width=True)
            except Exception as e:
                st.warning(f"Error al renderizar gráfico de eficiencia: {e}", icon="⚠️")
        
        with st.spinner("Cargando análisis de desperdicio..."):
            try:
                # Prepare data for 3D scatter plot
                grouped_lean = filtered_lean.groupby('Departamento')[lean_metrics].mean().reset_index()
                
                # Select the first 3 metrics for the 3D plot
                metrics_3d = lean_metrics[:3]
                if len(metrics_3d) < 3:
                    metrics_3d += [lean_metrics[0]] * (3 - len(metrics_3d))
                
                fig_scatter = px.scatter_3d(
                    grouped_lean,
                    x=metrics_3d[0],
                    y=metrics_3d[1],
                    z=metrics_3d[2],
                    color='Departamento',
                    size=np.ones(len(grouped_lean)) * 10,  # Uniform size
                    hover_name='Departamento',
                    labels={
                        metrics_3d[0]: metrics_3d[0],
                        metrics_3d[1]: metrics_3d[1],
                        metrics_3d[2]: metrics_3d[2]
                    },
                    height=450
                )
                fig_scatter.update_layout(
                    title="Análisis Multidimensional LEAN",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=30, r=30, t=50, b=30),
                    font=dict(family="Inter", size=13, color=COLOR_PALETTE['text']),
                    hoverlabel=dict(bgcolor=COLOR_PALETTE['card'], font_size=12)
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            except Exception as e:
                st.warning(f"Error al renderizar análisis de desperdicio: {e}", icon="⚠️")
    
    with col2:
        st.markdown("**📊 Comparación de Métricas**", help="Radar comparativo de métricas LEAN por departamento")
        with st.spinner("Cargando radar..."):
            try:
                scaler = MinMaxScaler()
                lean_radar = filtered_lean.groupby('Departamento')[lean_metrics].mean().reset_index()
                lean_radar[lean_metrics] = scaler.fit_transform(lean_radar[lean_metrics])
                
                fig_radar = go.Figure()
                for _, row in lean_radar.iterrows():
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[row[m] for m in lean_metrics],
                        theta=lean_metrics,
                        fill='toself',
                        name=row['Departamento'],
                        line=dict(width=2),
                        hovertemplate=f"<b>{row['Departamento']}</b><br>%{{theta}}: %{{r:.2f}}<extra></extra>"
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
            except Exception as e:
                st.warning(f"Error al renderizar radar: {e}", icon="⚠️")
        
        st.markdown("**📌 Detalle de Proyectos**", help="Resumen de métricas LEAN y proyectos activos por departamento")
        with st.expander("📌 Detalle de Proyectos", expanded=True):
            summary_cols = lean_metrics + (['Proyectos Activos'] if 'Proyectos Activos' in filtered_lean.columns else [])
            if summary_cols:
                summary = filtered_lean.groupby('Departamento')[summary_cols].mean().round(1)
                st.dataframe(
                    summary.style.background_gradient(cmap='Greens').format('{:.1f}'),
                    use_container_width=True
                )
            else:
                st.info("ℹ️ No hay métricas seleccionadas para mostrar.", icon="ℹ️")

def render_wellbeing_tab(bienestar_df, start_date, end_date, wellbeing_target):
    st.markdown("#### 😊 Bienestar Organizacional", help="Indicadores de bienestar y clima laboral")
    filtered_bienestar = filter_dataframe(bienestar_df, [], start_date, end_date)
    
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
            try:
                fig_bienestar = px.line(
                    filtered_bienestar,
                    x='Mes',
                    y=['Índice Bienestar', 'Ausentismo', 'Rotación', 'Engagement'],
                    markers=True,
                    color_discrete_sequence=[
                        COLOR_PALETTE['success'],
                        COLOR_PALETTE['danger'],
                        COLOR_PALETTE['warning'],
                        COLOR_PALETTE['accent']
                    ],
                    labels={'value': 'Porcentaje (%)', 'variable': 'Métrica'},
                    height=450,
                    hover_data={'Índice Bienestar': ':.1f', 'Ausentismo': ':.1f', 'Rotación': ':.1f', 'Engagement': ':.1f'}
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
                    paper_bgcolor='rgba(0,0,0,0)',
                    legend_title="Métrica",
                    margin=dict(l=30, r=30, t=50, b=30),
                    font=dict(family="Inter", size=13, color=COLOR_PALETTE['text']),
                    hoverlabel=dict(bgcolor=COLOR_PALETTE['card'], font_size=12),
                    hovermode="x unified"
                )
                st.plotly_chart(fig_bienestar, use_container_width=True)
            except Exception as e:
                st.warning(f"Error al renderizar tendencias: {e}", icon="⚠️")
    
    with wellbeing_view2:
        with st.spinner("Cargando correlaciones..."):
            try:
                corr_matrix = filtered_bienestar[['Índice Bienestar', 'Ausentismo', 'Rotación', 'Encuestas', 'Engagement']].corr()
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
            except Exception as e:
                st.warning(f"Error al renderizar correlaciones: {e}", icon="⚠️")

def render_action_plans_tab(departamentos_filtro, start_date, end_date):
    st.markdown("#### 📝 Planes de Acción", help="Gestión de planes de acción para abordar problemas identificados")
    filtered_plans = filter_dataframe(st.session_state.action_plans_df, departamentos_filtro, start_date, end_date, date_column='Plazo')
    
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
                    '% Avance': '{:.0f}%',
                    'Costo Estimado': 'MXN {:,.0f}'
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
        upcoming = filtered_plans[filtered_plans['Plazo'].dt.date <= today + timedelta(days=30)]
        if not upcoming.empty:
            for _, row in upcoming.iterrows():
                days_left = (row['Plazo'].date() - today).days
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
                costo = st.number_input("Costo Estimado (MXN)", min_value=0, value=10000, step=1000, help="Costo estimado del plan")
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
                        '% Avance': avance,
                        'Costo Estimado': costo
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
                st.warning("⚠️ La generación de PDF no está implementada en esta versión.", icon="⚠️")
    
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
                    st.warning("⚠️ La funcionalidad de envío de correo no está implementada en esta versión.", icon="⚠️")
    
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
                            combined_data = pd.concat([df for df in export_data], ignore_index=True, sort=False)
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
                filtered_nom['Evaluaciones'].mean() if not filtered_nom.empty else 0,
                "Cumplimiento NOM-035",
                nom_target,
                "📋",
                "Porcentaje promedio de cumplimiento con NOM-035",
                filtered_nom['Evaluaciones'].mean() - filtered_nom.groupby('Departamento')['Evaluaciones'].mean().shift(1).mean() if not filtered_nom.empty else 0
            ),
            (
                filtered_lean['Eficiencia'].mean() if not filtered_lean.empty else 0,
                "Adopción LEAN",
                lean_target,
                "🔄",
                "Nivel promedio de implementación de prácticas LEAN",
                filtered_lean['Eficiencia'].mean() - filtered_lean.groupby('Departamento')['Eficiencia'].mean().shift(1).mean() if not filtered_lean.empty else 0
            ),
            (
                filtered_bienestar['Índice Bienestar'].mean() if not filtered_bienestar.empty else 0,
                "Índice Bienestar",
                wellbeing_target,
                "😊",
                "Índice promedio de bienestar organizacional",
                filtered_bienestar['Índice Bienestar'].mean() - filtered_bienestar['Índice Bienestar'].shift(1).mean() if not filtered_bienestar.empty else 0
            ),
            (
                filtered_lean['Eficiencia'].mean() if not filtered_lean.empty else 0,
                "Eficiencia Operativa",
                efficiency_target,
                "⚙️",
                "Eficiencia promedio de procesos operativos",
                filtered_lean['Eficiencia'].mean() - filtered_lean.groupby('Departamento')['Eficiencia'].mean().shift(1).mean() if not filtered_lean.empty else 0
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
        
        st.markdown("""
        <hr style="border-color: var(--border);">
        <div style="text-align: center; color: var(--muted); font-size: 0.875rem; padding: 1.5rem 0;">
            Sistema Integral NOM-035 & LEAN 2.0 • Versión 3.2.0<br>
            © 2025 Departamento de RH
        </div>
        """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error en el dashboard: {e}", icon="🚨")

if __name__ == "__main__":
    main()

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
    'primary': '#1a3c5e',       # Darker blue for better contrast
    'secondary': '#2b6cb0',     # Softer blue for accents
    'accent': '#4a90e2',        # Bright blue for interactive elements
    'success': '#2f855a',       # Accessible green
    'warning': '#d97706',       # Accessible orange
    'danger': '#c53030',        # Accessible red
    'light': '#f7fafc',         # Light background
    'dark': '#1a202c',          # Dark text/icons
    'background': '#ffffff',     # Clean white
    'text': '#2d3748',          # Readable gray
    'muted': '#718096'          # Subtle text
}

# Custom CSS for professional, accessible, and responsive styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --primary: #1a3c5e;
    --secondary: #2b6cb0;
    --accent: #4a90e2;
    --success: #2f855a;
    --warning: #d97706;
    --danger: #c53030;
    --light: #f7fafc;
    --background: #ffffff;
    --text: #2d3748;
    --muted: #718096;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text);
    line-height: 1.5;
}

.main {
    background-color: var(--background);
    padding: 1rem;
}

[data-testid="stSidebar"] {
    background-color: var(--primary) !important;
    color: white !important;
    padding: 1rem;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

h1, h2, h3, h4, h5, h6 {
    color: var(--primary);
    font-weight: 600;
    margin-bottom: 0.5rem;
}

.card {
    background-color: white;
    border-radius: 8px;
    padding: 1.25rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.stButton > button {
    background-color: var(--secondary);
    color: white;
    border-radius: 6px;
    padding: 0.5rem 1rem;
    font-weight: 500;
    transition: background-color 0.2s ease;
}

.stButton > button:hover {
    background-color: var(--accent);
}

.stButton > button:focus {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
}

[data-baseweb="tab-list"] {
    gap: 0.5rem;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 1rem;
}

[data-baseweb="tab"] {
    background-color: var(--light) !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 0.75rem 1.5rem !important;
    color: var(--text) !important;
    font-weight: 500;
    transition: all 0.2s ease;
}

[data-baseweb="tab"][aria-selected="true"] {
    background-color: var(--primary) !important;
    color: white !important;
}

.stDataFrame {
    border-radius: 8px;
    overflow: hidden;
}

.stTextInput > div > input,
.stSelectbox > div > select,
.stDateInput > div > input {
    border: 1px solid #cbd5e0;
    border-radius: 6px;
    padding: 0.5rem;
}

.stTextInput > div > input:focus,
.stSelectbox > div > select:focus,
.stDateInput > div > input:focus {
    border-color: var(--secondary);
    box-shadow: 0 0 0 2px rgba(43, 108, 176, 0.2);
}

.error-message {
    color: var(--danger);
    font-size: 0.875rem;
    margin-top: 0.25rem;
}

@media (max-width: 768px) {
    .main {
        padding: 0.5rem;
    }
    [data-testid="stSidebar"] {
        padding: 0.5rem;
    }
    .card {
        padding: 1rem;
    }
    [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
    }
    .stButton > button {
        width: 100%;
    }
}
</style>
""", unsafe_allow_html=True)

# ========== DATA LOADING AND PROCESSING ==========
@st.cache_data(ttl=600)
def load_data():
    try:
        np.random.seed(42)
        nom = pd.DataFrame({
            'Departamento': DEPARTMENTS,
            'Evaluaciones': np.random.randint(70, 100, len(DEPARTMENTS)),
            'Capacitaciones': np.random.randint(60, 100, len(DEPARTMENTS)),
            'Incidentes': np.random.randint(0, 10, len(DEPARTMENTS)),
            'Tendencia': np.round(np.random.normal(0.5, 1.5, len(DEPARTMENTS)), 2)
        })
        lean = pd.DataFrame({
            'Departamento': DEPARTMENTS,
            'Eficiencia': np.random.randint(60, 95, len(DEPARTMENTS)),
            'Reducción MURI/MURA/MUDA': np.random.randint(5, 25, len(DEPARTMENTS)),
            'Proyectos Activos': np.random.randint(1, 6, len(DEPARTMENTS)),
            '5S+2_Score': np.random.randint(60, 100, len(DEPARTMENTS)),
            'Kaizen Colectivo': np.random.randint(50, 90, len(DEPARTMENTS))
        })
        dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
        bienestar = pd.DataFrame({
            'Mes': dates,
            'Índice Bienestar': np.round(np.linspace(70, 85, 12) + np.random.normal(0, 3, 12), 1),
            'Ausentismo': np.round(np.linspace(10, 7, 12) + np.random.normal(0, 1, 12), 1),
            'Rotación': np.round(np.linspace(15, 10, 12) + np.random.normal(0, 1.5, 12), 1),
            'Encuestas': np.random.randint(80, 100, 12)
        })
        action_plans = pd.DataFrame({
            'ID': range(1, 6),
            'Departamento': np.random.choice(DEPARTMENTS, 5),
            'Problema': [
                'Falta de capacitación en NOM-035',
                'Baja eficiencia en línea de producción',
                'Alto ausentismo en turno nocturno',
                'Desperdicio de materiales',
                'Falta de estandarización de procesos'
            ],
            'Acción': [
                'Programar capacitación obligatoria',
                'Implementar análisis de tiempos y movimientos',
                'Realizar estudio de clima laboral',
                'Aplicar herramientas de Lean Manufacturing',
                'Documentar procesos críticos'
            ],
            'Responsable': ['Juan Pérez', 'María García', 'Luis Martínez', 'Ana López', 'Carlos Rodríguez'],
            'Plazo': [date(2024, 6, 15), date(2024, 5, 30), date(2024, 7, 1), date(2024, 6, 10), date(2024, 8, 15)],
            'Estado': ['En progreso', 'Pendiente', 'Completado', 'En progreso', 'Pendiente'],
            'Prioridad': ['Alta', 'Media', 'Alta', 'Media', 'Baja'],
            '% Avance': [65, 0, 100, 30, 0]
        })
        return nom, lean, bienestar, action_plans
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return None, None, None, None

# Initialize session state for action plans
if 'action_plans_df' not in st.session_state:
    _, _, _, action_plans = load_data()
    st.session_state.action_plans_df = action_plans

# Load data
nom_df, lean_df, bienestar_df, _ = load_data()
if None in (nom_df, lean_df, bienestar_df):
    st.stop()

# ========== SIDEBAR ==========
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.5rem;">
            <span style="font-size: 1.5rem;">📊</span>
            <h2 style="margin: 0; color: white;">NOM-035 & LEAN</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        with st.expander("🔍 Filtros", expanded=True):
            st.markdown("**Período**")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Inicio",
                    value=date(2024, 1, 1),
                    min_value=date(2024, 1, 1),
                    max_value=date(2024, 12, 31),
                    key="date_start",
                    help="Seleccione la fecha de inicio del período"
                )
            with col2:
                end_date = st.date_input(
                    "Fin",
                    value=date(2024, 12, 31),
                    min_value=start_date,
                    max_value=date(2024, 12, 31),
                    key="date_end",
                    help="Seleccione la fecha de fin del período"
                )
            
            if start_date > end_date:
                st.markdown("<p class='error-message'>La fecha de inicio no puede ser posterior a la fecha de fin</p>", unsafe_allow_html=True)
                return None, None, None
            
            st.markdown("**Departamentos**")
            departamentos_filtro = st.multiselect(
                "Seleccionar departamentos",
                options=DEPARTMENTS,
                default=['Producción', 'Calidad', 'Logística'],
                key="dept_filter",
                help="Seleccione uno o más departamentos para filtrar los datos"
            )
        
        with st.expander("⚙️ Metas", expanded=False):
            nom_target = st.slider("Meta NOM-035 (%)", 50, 100, 90, help="Establezca la meta de cumplimiento NOM-035")
            lean_target = st.slider("Meta LEAN (%)", 50, 100, 80, help="Establezca la meta de adopción LEAN")
            wellbeing_target = st.slider("Meta Bienestar (%)", 50, 100, 85, help="Establezca la meta de índice de bienestar")
            efficiency_target = st.slider("Meta Eficiencia (%)", 50, 100, 75, help="Establezca la meta de eficiencia operativa")
        
        st.markdown("---")
        if st.button("🔄 Actualizar", use_container_width=True, help="Actualiza los datos y visualizaciones"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #a0aec0; font-size: 0.75rem;">
            v3.0.0<br>
            © 2025 RH Analytics
        </div>
        """, unsafe_allow_html=True)
    
    return start_date, end_date, departamentos_filtro, (nom_target, lean_target, wellbeing_target, efficiency_target)

# ========== HEADER ==========
def render_header(start_date, end_date):
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem; margin-bottom: 1rem;">
        <div>
            <h1 style="margin: 0;">Sistema Integral NOM-035 & LEAN 2.0</h1>
            <p style="color: var(--muted); font-size: 1rem; margin: 0;">
                Monitoreo Estratégico de Bienestar y Eficiencia
            </p>
        </div>
        <div style="background-color: var(--light); padding: 0.5rem 1rem; border-radius: 1rem; text-align: center;">
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
    delta_text = f"+{delta}% sobre meta" if delta >= 0 else f"{delta}% bajo meta"
    
    st.markdown(f"""
    <div class="card" role="region" aria-label="{title}">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <div style="font-size: 0.875rem; color: var(--text); font-weight: 500;">
                {icon} {title}
            </div>
            <div style="font-size: 0.875rem; color: {color};" aria-label="Estado">
                {status}
            </div>
        </div>
        <div style="font-size: 1.5rem; font-weight: 600; color: {color}; margin-bottom: 0.25rem;">
            {value}%
        </div>
        <div style="font-size: 0.75rem; color: var(--muted); margin-bottom: 0.5rem;">
            Meta: {target}% • {delta_text}
        </div>
        <div style="height: 0.375rem; background: #edf2f7; border-radius: 0.1875rem;">
            <div style="width: {percentage}%; height: 0.375rem; background: {color}; border-radius: 0.1875rem;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"<span title='{help_text}'></span>", unsafe_allow_html=True)

# ========== TABS ==========
def render_nom_tab(nom_df, departamentos_filtro):
    st.markdown("#### Cumplimiento NOM-035")
    filtered_nom = nom_df[nom_df['Departamento'].isin(departamentos_filtro)]
    
    if filtered_nom.empty:
        st.warning("⚠️ No hay datos disponibles para los departamentos seleccionados", icon="⚠️")
        return
    
    nom_view1, nom_view2, nom_view3 = st.tabs(["📊 Métricas", "🔍 Mapa de Riesgo", "📈 Tendencias"])
    
    with nom_view1:
        col1, col2 = st.columns([3, 2])
        with col1:
            with st.spinner("Cargando gráfico..."):
                fig = px.bar(
                    filtered_nom,
                    x="Departamento",
                    y=["Evaluaciones", "Capacitaciones"],
                    barmode="group",
                    color_discrete_sequence=[COLOR_PALETTE['primary'], COLOR_PALETTE['secondary']],
                    labels={'value': 'Porcentaje (%)', 'variable': 'Métrica'},
                    height=400
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    yaxis_range=[0, 100],
                    legend_title_text='Métrica',
                    xaxis_title="",
                    margin=dict(l=20, r=20, t=40, b=20),
                    font=dict(family="Inter", size=12, color=COLOR_PALETTE['text'])
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**📌 Resumen**")
            st.dataframe(
                filtered_nom.set_index('Departamento').style.format({
                    'Evaluaciones': '{:.1f}',
                    'Capacitaciones': '{:.1f}',
                    'Incidentes': '{:.0f}',
                    'Tendencia': '{:.2f}'
                }),
                use_container_width=True,
                height=400
            )
    
    with nom_view2:
        with st.spinner("Cargando mapa de riesgo..."):
            scaler = MinMaxScaler()
            z_values = scaler.fit_transform(filtered_nom[['Evaluaciones', 'Capacitaciones', 'Incidentes']])
            fig_heat = go.Figure(data=go.Heatmap(
                z=z_values.T,
                x=filtered_nom['Departamento'],
                y=['Evaluaciones', 'Capacitaciones', 'Incidentes'],
                colorscale='RdYlGn',
                reversescale=True,
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
                margin=dict(l=50, r=50, t=50, b=50),
                font=dict(family="Inter", size=12, color=COLOR_PALETTE['text'])
            )
            st.plotly_chart(fig_heat, use_container_width=True)
    
    with nom_view3:
        col1, col2 = st.columns([3, 1])
        with col1:
            with st.spinner("Cargando tendencias..."):
                fig_trend = px.bar(
                    filtered_nom,
                    x='Departamento',
                    y='Tendencia',
                    color='Tendencia',
                    color_continuous_scale='RdYlGn',
                    range_color=[-3, 3],
                    labels={'Tendencia': 'Cambio mensual (%)'},
                    height=400
                )
                fig_trend.update_layout(
                    title="Tendencia de Cumplimiento",
                    yaxis_title="Cambio (%)",
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=40, b=20),
                    font=dict(family="Inter", size=12, color=COLOR_PALETTE['text'])
                )
                st.plotly_chart(fig_trend, use_container_width=True)
        
        with col2:
            st.markdown("**📊 Interpretación**")
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

def render_lean_tab(lean_df, departamentos_filtro):
    st.markdown("#### Progreso LEAN 2.0")
    filtered_lean = lean_df[lean_df['Departamento'].isin(departamentos_filtro)]
    
    if filtered_lean.empty:
        st.warning("⚠️ No hay datos disponibles para los departamentos seleccionados", icon="⚠️")
        return
    
    col1, col2 = st.columns([3, 2])
    with col1:
        with st.spinner("Cargando gráfico de eficiencia..."):
            fig_lean = px.bar(
                filtered_lean,
                x='Departamento',
                y='Eficiencia',
                color='Eficiencia',
                color_continuous_scale='Greens',
                range_color=[50, 100],
                labels={'Eficiencia': 'Eficiencia (%)'},
                height=400
            )
            fig_lean.update_layout(
                title="Eficiencia por Departamento",
                yaxis_range=[0, 100],
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=40, b=20),
                font=dict(family="Inter", size=12, color=COLOR_PALETTE['text'])
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
                height=400
            )
            fig_scatter.update_layout(
                title="Eficiencia vs Reducción de Desperdicio",
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=40, b=20),
                font=dict(family="Inter", size=12, color=COLOR_PALETTE['text'])
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        st.markdown("**📊 Comparación de Métricas**")
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
                    theta=['Eficiencia', 'Reducción', '5S', 'Kaizen'],
                    fill='toself',
                    name=dept,
                    line=dict(width=2)
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                height=400,
                showlegend=True,
                margin=dict(l=50, r=50, t=30, b=50),
                font=dict(family="Inter", size=12, color=COLOR_PALETTE['text'])
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

def render_wellbeing_tab(bienestar_df, start_date, end_date):
    st.markdown("#### Bienestar Organizacional")
    filtered_bienestar = bienestar_df[
        (bienestar_df['Mes'].dt.date >= start_date) &
        (bienestar_df['Mes'].dt.date <= end_date)
    ]
    
    if filtered_bienestar.empty:
        st.warning("⚠️ No hay datos disponibles para el período seleccionado", icon="⚠️")
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        delta = filtered_bienestar['Encuestas'].iloc[-1] - filtered_bienestar['Encuestas'].iloc[0] if len(filtered_bienestar) > 1 else 0
        st.metric(
            label="Encuestas Completadas",
            value=f"{filtered_bienestar['Encuestas'].mean():.0f}%",
            delta=f"{delta:+.0f}%",
            help="Porcentaje de encuestas completadas"
        )
    with col2:
        delta = filtered_bienestar['Ausentismo'].iloc[-1] - filtered_bienestar['Ausentismo'].iloc[0] if len(filtered_bienestar) > 1 else 0
        st.metric(
            label="Ausentismo",
            value=f"{filtered_bienestar['Ausentismo'].iloc[-1]:.1f}%",
            delta=f"{delta:+.1f}%",
            delta_color="inverse",
            help="Tasa de ausentismo laboral"
        )
    with col3:
        delta = filtered_bienestar['Rotación'].iloc[-1] - filtered_bienestar['Rotación'].iloc[0] if len(filtered_bienestar) > 1 else 0
        st.metric(
            label="Rotación",
            value=f"{filtered_bienestar['Rotación'].iloc[-1]:.1f}%",
            delta=f"{delta:+.1f}%",
            delta_color="inverse",
            help="Tasa de rotación de personal"
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
                height=400
            )
            fig_bienestar.update_layout(
                title="Evolución Mensual",
                yaxis_range=[0, 100],
                plot_bgcolor='rgba(0,0,0,0)',
                legend_title="Métrica",
                margin=dict(l=20, r=20, t=40, b=20),
                font=dict(family="Inter", size=12, color=COLOR_PALETTE['text'])
            )
            st.plotly_chart(fig_bienestar, use_container_width=True)
    
    with wellbeing_view2:
        with st.spinner("Cargando correlaciones..."):
            corr_matrix = filtered_bienestar[['Índice Bienestar', 'Ausentismo', 'Rotación', 'Encuestas']].corr()
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                color_continuous_scale='RdYlGn',
                range_color=[-1, 1],
                labels=dict(x="", y="", color="Correlación"),
                height=400
            )
            fig_corr.update_layout(
                title="Matriz de Correlación",
                margin=dict(l=50, r=50, t=50, b=50),
                font=dict(family="Inter", size=12, color=COLOR_PALETTE['text'])
            )
            st.plotly_chart(fig_corr, use_container_width=True)

def render_action_plans_tab(departamentos_filtro, start_date, end_date):
    st.markdown("#### Planes de Acción")
    filtered_plans = st.session_state.action_plans_df[
        (st.session_state.action_plans_df['Departamento'].isin(departamentos_filtro)) &
        (st.session_state.action_plans_df['Plazo'] >= start_date) &
        (st.session_state.action_plans_df['Plazo'] <= end_date)
    ]
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**📌 Planes Registrados**")
        if filtered_plans.empty:
            st.info("No hay planes de acción para los filtros seleccionados", icon="ℹ️")
        else:
            st.dataframe(
                filtered_plans.style.apply(
                    lambda x: [
                        f"background-color: {COLOR_PALETTE['success']}; color: white" if v == 'Completado'
                        else f"background-color: {COLOR_PALETTE['warning']}" if v == 'En progreso'
                        else f"background-color: {COLOR_PALETTE['danger']}; color: white"
                        for v in x
                    ], subset=['Estado']
                ).bar(
                    subset=['% Avance'],
                    color=COLOR_PALETTE['secondary'],
                    vmin=0,
                    vmax=100
                ).format({
                    'Plazo': lambda x: x.strftime('%d/%m/%Y'),
                    '% Avance': '{:.0f}%'
                }),
                use_container_width=True,
                hide_index=True,
                height=400
            )
    
    with col2:
        st.markdown("**📊 Resumen por Estado**")
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
                height=300
            )
            fig_status.update_layout(
                showlegend=True,
                margin=dict(l=20, r=20, t=20, b=20),
                font=dict(family="Inter", size=12, color=COLOR_PALETTE['text'])
            )
            st.plotly_chart(fig_status, use_container_width=True)
        
        st.markdown("**📅 Vencimientos Próximos**")
        today = date.today()
        upcoming = filtered_plans[filtered_plans['Plazo'] <= today + timedelta(days=30)]
        if not upcoming.empty:
            for _, row in upcoming.iterrows():
                days_left = (row['Plazo'] - today).days
                color = COLOR_PALETTE['danger'] if days_left < 7 else COLOR_PALETTE['warning']
                text = f"Vence en {days_left} días" if days_left > 0 else f"Vencido hace {-days_left} días"
                st.markdown(f"""
                <div class="card">
                    <div style="font-weight: 500;">{row['Departamento']}</div>
                    <div style="font-size: 0.75rem; color: var(--muted);">{row['Problema'][:30]}...</div>
                    <div style="font-size: 0.75rem; color: {color}; font-weight: 500;">{text}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No hay vencimientos próximos", icon="ℹ️")
    
    with st.expander("➕ Nuevo Plan de Acción", expanded=False):
        with st.form("nuevo_plan_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                dept = st.selectbox("Departamento", DEPARTMENTS, help="Seleccione el departamento")
                problema = st.text_area("Problema", max_chars=200, help="Describa el problema identificado")
                prioridad = st.selectbox("Prioridad", ["Alta", "Media", "Baja"], help="Seleccione la prioridad")
            with col2:
                accion = st.text_area("Acción", max_chars=200, help="Describa la acción propuesta")
                responsable = st.text_input("Responsable", help="Nombre del responsable")
                plazo = st.date_input(
                    "Plazo",
                    min_value=today,
                    value=today + timedelta(days=30),
                    help="Fecha límite para el plan"
                )
                avance = st.slider("% Avance", 0, 100, 0, help="Porcentaje de avance actual")
            
            submitted = st.form_submit_button("💾 Guardar", use_container_width=True)
            
            if submitted:
                errors = []
                if not problema:
                    errors.append("El campo Problema es obligatorio")
                if not accion:
                    errors.append("El campo Acción es obligatorio")
                if not responsable:
                    errors.append("El campo Responsable es obligatorio")
                if not re.match(r'^[A-Za-z\s]+$', responsable):
                    errors.append("El Responsable debe contener solo letras y espacios")
                
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
                    st.success("✅ Plan registrado correctamente")
                    st.rerun()

# ========== EXPORT AND REPORTING ==========
def render_export_section(nom_df, lean_df, bienestar_df):
    st.markdown("---")
    st.markdown("#### Exportar y Reportes")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.expander("📄 Generar Reporte PDF", expanded=False):
            report_type = st.selectbox(
                "Tipo de reporte",
                ["Completo", "Resumido", "Solo NOM-035", "Solo LEAN"],
                help="Seleccione el tipo de reporte"
            )
            if st.button("🖨️ Generar", use_container_width=True):
                with st.spinner("Generando reporte..."):
                    # Simulate PDF generation
                    st.success("✅ Reporte generado")
                    st.download_button(
                        label="📥 Descargar PDF",
                        data=io.BytesIO(b"Simulated PDF content"),
                        file_name=f"Reporte_{report_type}_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
    
    with col2:
        with st.expander("📧 Enviar por Correo", expanded=False):
            email = st.text_input("Correo", placeholder="usuario@empresa.com", help="Ingrese el correo del destinatario")
            if st.button("📤 Enviar", use_container_width=True):
                if not email:
                    st.markdown("<p class='error-message'>El campo Correo es obligatorio</p>", unsafe_allow_html=True)
                elif not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email):
                    st.markdown("<p class='error-message'>Ingrese un correo válido</p>", unsafe_allow_html=True)
                else:
                    with st.spinner("Enviando correo..."):
                        # Simulate email sending
                        st.success("✅ Correo enviado")
                        st.download_button(
                            label="📥 Descargar Contenido",
                            data=f"To: {email}\nSubject: Reporte NOM-035 & LEAN\n\nReporte enviado el {datetime.now().strftime('%d/%m/%Y %H:%M')}",
                            file_name=f"Correo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
    
    with col3:
        with st.expander("📊 Exportar Datos", expanded=False):
            export_format = st.radio("Formato", ["CSV", "Excel", "JSON"], horizontal=True, help="Seleccione el formato de exportación")
            data_options = st.multiselect(
                "Datos",
                ["NOM-035", "LEAN", "Bienestar", "Planes de Acción"],
                default=["NOM-035", "LEAN", "Bienestar"],
                help="Seleccione los datos a exportar"
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
                        
                        combined_data = pd.concat(export_data, ignore_index=True)
                        
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
                            
                            st.success(f"✅ Datos exportados como {export_format}")
                            st.download_button(
                                label=f"📥 Descargar .{ext}",
                                data=data,
                                file_name=f"nom_lean_data.{ext}",
                                mime=mime,
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"Error al exportar datos: {e}")

# ========== MAIN FUNCTION ==========
def main():
    try:
        sidebar_data = render_sidebar()
        # Check if sidebar_data is None or contains None values for critical components
        if sidebar_data is None or sidebar_data[0] is None or sidebar_data[1] is None or len(sidebar_data[2]) == 0:
            return
        
        start_date, end_date, departamentos_filtro, targets = sidebar_data
        nom_target, lean_target, wellbeing_target, efficiency_target = targets
        
        render_header(start_date, end_date)
        
        st.markdown("### Indicadores Clave")
        cols = st.columns(4)
        kpis = [
            (round(nom_df[nom_df['Departamento'].isin(departamentos_filtro)]['Evaluaciones'].mean()), "Cumplimiento NOM-035", nom_target, "📋", "Porcentaje de cumplimiento con NOM-035"),
            (round(lean_df[lean_df['Departamento'].isin(departamentos_filtro)]['Eficiencia'].mean()), "Adopción LEAN", lean_target, "🔄", "Nivel de implementación de LEAN"),
            (round(bienestar_df[(bienestar_df['Mes'].dt.date >= start_date) & (bienestar_df['Mes'].dt.date <= end_date)]['Índice Bienestar'].mean()), "Índice Bienestar", wellbeing_target, "😊", "Indicador de bienestar organizacional"),
            (round(lean_df[lean_df['Departamento'].isin(departamentos_filtro)]['Eficiencia'].mean()), "Eficiencia Operativa", efficiency_target, "⚙️", "Eficiencia general de procesos")
        ]
        for i, (value, title, target, icon, help_text) in enumerate(kpis):
            with cols[i]:
                kpi_card(value, title, target, icon, help_text)
        
        tab1, tab2, tab3, tab4 = st.tabs(["📋 NOM-035", "🔄 LEAN", "😊 Bienestar", "📝 Planes"])
        
        with tab1:
            render_nom_tab(nom_df, departamentos_filtro)
        with tab2:
            render_lean_tab(lean_df, departamentos_filtro)
        with tab3:
            render_wellbeing_tab(bienestar_df, start_date, end_date)
        with tab4:
            render_action_plans_tab(departamentos_filtro, start_date, end_date)
        
        render_export_section(nom_df, lean_df, bienestar_df)
        
        st.markdown("""
        <hr>
        <div style="text-align: center; color: var(--muted); font-size: 0.75rem; padding: 1rem 0;">
            Sistema Integral NOM-035 & LEAN 2.0 • Versión 3.0.0<br>
            © 2025 Departamento de RH
        </div>
        """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error en el dashboard: {e}")

if __name__ == "__main__":
    main()

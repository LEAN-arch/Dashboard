import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import warnings
import io  # FIX: Added for Excel export
warnings.filterwarnings('ignore')

# ========== PAGE CONFIGURATION ==========
st.set_page_config(
    page_title="Sistema Integral NOM-035 & LEAN 2.0",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.example.com/help',
        'Report a bug': "https://www.example.com/bug",
        'About': "# Sistema de Monitoreo Estratégico"
    }
)

# ========== CONSTANTS AND CONFIGURATION ==========
DEPARTMENTS = ['Producción', 'Calidad', 'Logística', 'Administración', 'Ventas', 'RH', 'TI']

# Modern color scheme with accessibility in mind
COLOR_PALETTE = {
    'primary': '#2c3e50',       # Dark blue
    'secondary': '#3498db',     # Bright blue
    'accent': '#2980b9',        # Medium blue
    'success': '#27ae60',       # Green
    'warning': '#f39c12',       # Orange
    'danger': '#e74c3c',        # Red
    'light': '#ecf0f1',         # Light gray
    'dark': '#2c3e50',          # Dark blue
    'background': '#ffffff',    # White
    'text': '#333333',          # Dark gray
    'text_light': '#7f8c8d',    # Light gray text
    'grid': '#e0e0e0'           # Grid lines
}

# Font settings with better typography
FONT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
h1 {
    font-size: 2rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.5rem !important;
}
h2 {
    font-size: 1.75rem !important;
    font-weight: 600 !important;
}
h3 {
    font-size: 1.5rem !important;
}
.subheader {
    font-size: 1.1rem;
    color: #7f8c8d;
    margin-bottom: 1.5rem !important;
}
</style>
"""
st.markdown(FONT_CSS, unsafe_allow_html=True)

# Custom CSS for modern styling
st.markdown(f"""
<style>
    /* Main container */
    .main {{
        background-color: {COLOR_PALETTE['background']};
        padding: 0 2rem;
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {COLOR_PALETTE['primary']} !important;
        color: white !important;
    }}
    
    /* Titles */
    h1, h2, h3, h4, h5, h6 {{
        color: {COLOR_PALETTE['primary']} !important;
        font-weight: 700 !important;
    }}
    
    /* Cards */
    .card {{
        background-color: white;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        border: 1px solid #f0f0f0;
    }}
    
    /* KPI Cards */
    .kpi-card {{
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .kpi-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        margin: 0 !important;
        border: 1px solid {COLOR_PALETTE['light']} !important;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {COLOR_PALETTE['primary']} !important;
        color: white !important;
        border-color: {COLOR_PALETTE['primary']} !important;
    }}
    
    /* Dataframes */
    .stDataFrame {{
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid {COLOR_PALETTE['light']};
    }}
    
    /* Buttons */
    .stButton button {{
        transition: all 0.2s ease;
    }}
    .stButton button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }}
    
    /* Inputs */
    [data-baseweb="input"], [data-baseweb="select"] {{
        border-radius: 8px !important;
    }}
    
    /* Progress bars */
    [role="progressbar"] {{
        border-radius: 4px !important;
    }}
    
    /* Tooltips */
    .stTooltip {{
        border-radius: 8px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    ::-webkit-scrollbar-track {{
        background: {COLOR_PALETTE['light']};
    }}
    ::-webkit-scrollbar-thumb {{
        background: {COLOR_PALETTE['secondary']};
        border-radius: 4px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: {COLOR_PALETTE['accent']};
    }}
</style>
""", unsafe_allow_html=True)  # SECURITY: Ensure dynamic inputs are sanitized if added

# ========== DATA LOADING AND PROCESSING ==========
@st.cache_data(ttl=600)
def load_data():
    """Load and generate synthetic data for the dashboard"""
    np.random.seed(42)
    
    # NOM-035 Data with more realistic distributions
    nom = pd.DataFrame({
        'Departamento': DEPARTMENTS,
        'Evaluaciones': np.clip(np.random.normal(85, 10, len(DEPARTMENTS)), 50, 100).astype(int),
        'Capacitaciones': np.clip(np.random.normal(75, 15, len(DEPARTMENTS)), 50, 100).astype(int),
        'Incidentes': np.random.poisson(3, len(DEPARTMENTS)),
        'Tendencia': np.round(np.random.normal(0.5, 1.5, len(DEPARTMENTS)), 2),
        'Riesgo_Psicosocial': np.random.choice(['Bajo', 'Medio', 'Alto'], len(DEPARTMENTS), p=[0.6, 0.3, 0.1])
    })
    
    # LEAN Data with correlations between metrics
    efficiency = np.clip(np.random.normal(75, 10, len(DEPARTMENTS)), 50, 95)
    lean = pd.DataFrame({
        'Departamento': DEPARTMENTS,
        'Eficiencia': efficiency.astype(int),
        'Reducción_Desperdicio': np.clip(efficiency * 0.3 + np.random.normal(5, 3, len(DEPARTMENTS)), 5, 30).astype(int),
        'Proyectos_Activos': np.random.poisson(3, len(DEPARTMENTS)) + 1,
        '5S_Score': np.clip(efficiency + np.random.normal(0, 5, len(DEPARTMENTS)), 60, 100).astype(int),
        'SMED': np.clip(efficiency * 0.8 + np.random.normal(0, 5, len(DEPARTMENTS)), 50, 90).astype(int),
        'Nivel_LEAN': np.random.choice(['Inicial', 'Intermedio', 'Avanzado'], len(DEPARTMENTS), p=[0.3, 0.5, 0.2])
    })
    
    # Wellbeing Data with seasonality
    dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
    base_wellbeing = np.linspace(70, 85, 12)
    seasonality = 5 * np.sin(np.linspace(0, 2*np.pi, 12))
    bienestar = pd.DataFrame({
        'Mes': dates,
        'Índice_Bienestar': np.round(base_wellbeing + seasonality + np.random.normal(0, 2, 12), 1),
        'Ausentismo': np.round(10 - 0.25*np.arange(12) + 0.5*np.sin(np.linspace(0, 2*np.pi, 12)) + np.random.normal(0, 0.5, 12), 1),
        'Rotación': np.round(15 - 0.4*np.arange(12) + np.random.normal(0, 1, 12), 1),
        'Encuestas': np.random.randint(80, 100, 12),
        'Satisfacción': np.round(base_wellbeing * 0.9 + np.random.normal(0, 3, 12), 1),
        'Compromiso': np.round(base_wellbeing * 0.85 + np.random.normal(0, 3, 12), 1)
    })
    
    # Action Plans Data with more realistic distribution
    action_plans = pd.DataFrame({
        'ID': range(1, 11),
        'Departamento': np.random.choice(DEPARTMENTS, 10),
        'Problema': [
            'Falta de capacitación en NOM-035',
            'Baja eficiencia en línea de producción',
            'Alto ausentismo en turno nocturno',
            'Desperdicio de materiales en empaque',
            'Falta de estandarización de procesos',
            'Espacios de trabajo no ergonómicos',
            'Tiempos muertos en cambio de turno',
            'Falta de señalización de seguridad',
            'Problemas de comunicación interdepartamental',
            'Exceso de inventario en proceso'
        ],
        'Acción': [
            'Programar capacitación obligatoria',
            'Implementar análisis de tiempos y movimientos',
            'Realizar estudio de clima laboral',
            'Aplicar herramientas de Lean Manufacturing',
            'Documentar procesos críticos',
            'Rediseñar estaciones de trabajo',
            'Implementar procedimiento SMED',
            'Instalar señalización adecuada',
            'Establecer reuniones interdepartamentales',
            'Aplicar sistema Kanban'
        ],
        'Responsable': ['Juan Pérez', 'María García', 'Luis Martínez', 'Ana López', 
                       'Carlos Rodríguez', 'Patricia Sánchez', 'Jorge Ramírez', 
                       'Lucía Fernández', 'Roberto Jiménez', 'Sofía Castro'],
        'Plazo': [date(2024, 6, 15), date(2024, 5, 30), date(2024, 7, 1), 
                 date(2024, 6, 10), date(2024, 8, 15), date(2024, 6, 5),
                 date(2024, 7, 20), date(2024, 5, 25), date(2024, 8, 1),
                 date(2024, 9, 15)],
        'Estado': ['En progreso', 'Pendiente', 'Completado', 'En progreso', 
                  'Pendiente', 'Completado', 'En progreso', 'Pendiente',
                  'En progreso', 'Pendiente'],
        'Prioridad': ['Alta', 'Media', 'Alta', 'Media', 'Baja', 'Alta', 
                     'Media', 'Alta', 'Media', 'Baja'],
        '%_Avance': [65, 0, 100, 30, 0, 100, 45, 0, 25, 0],
        'Impacto_Esperado': ['Alto', 'Medio', 'Alto', 'Medio', 'Bajo', 'Alto',
                            'Medio', 'Alto', 'Medio', 'Bajo']
    })
    
    return nom, lean, bienestar, action_plans

# Load data with error handling
try:
    nom_df, lean_df, bienestar_df, action_plans_df = load_data()
except Exception as e:
    st.error(f"Error al cargar los datos: {e}")
    st.stop()

# ========== SIDEBAR ==========
with st.sidebar:
    # Logo and title with better spacing
    st.markdown(f"""
    <div style="display: flex; flex-direction: column; align-items: center; margin-bottom: 1.5rem;">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📊</div>
        <h2 style="color: white; margin-bottom: 0; text-align: center;">NOM-035 & LEAN</h2>
        <p style="color: #bdc3c7; font-size: 0.9rem; margin-top: 0.25rem; text-align: center;">
            Panel de Control Estratégico
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Date range filter with improved UX
    st.markdown("**🔍 Filtros de Periodo**")
    start_date, end_date = st.columns(2)
    with start_date:
        start_date = st.date_input(
            "Inicio", 
            value=date(2024, 1, 1),
            min_value=date(2024, 1, 1),
            max_value=date(2024, 12, 31),
            key="date_start",
            format="DD/MM/YYYY"
        )
    with end_date:
        end_date = st.date_input(
            "Fin", 
            value=date(2024, 4, 1),
            min_value=date(2024, 1, 1),
            max_value=date(2024, 12, 31),
            key="date_end",
            format="DD/MM/YYYY"
        )
    
    # Department filter with search
    st.markdown("**🏢 Departamentos**")
    departamentos_filtro = st.multiselect(
        "Seleccionar departamentos",
        options=DEPARTMENTS,
        default=['Producción', 'Calidad', 'Logística'],
        label_visibility="collapsed",
        placeholder="Selecciona departamentos..."
    )
    
    st.markdown("---")
    
    # KPI targets configuration with better organization
    with st.expander("⚙️ Configuración de Metas", expanded=False):
        st.markdown("**📌 Indicadores Clave**")
        nom_target = st.slider("Meta NOM-035 (%)", 50, 100, 90, help="Meta de cumplimiento de la norma NOM-035")
        lean_target = st.slider("Meta LEAN (%)", 50, 100, 80, help="Meta de adopción de metodologías LEAN")
        
        st.markdown("**😊 Bienestar**")
        wellbeing_target = st.slider("Meta Bienestar (%)", 50, 100, 85, help="Meta del índice de bienestar organizacional")
        
        st.markdown("**⚙️ Operaciones**")
        efficiency_target = st.slider("Meta Eficiencia (%)", 50, 100, 75, help="Meta de eficiencia operativa")
    
    st.markdown("---")
    
    # Refresh button with better styling
    if st.button(
        "🔄 Actualizar Datos", 
        use5377_container_width=True,
        help="Actualiza todos los datos y visualizaciones con los filtros actuales"
    ):
        st.rerun()  # TODO: Consider optimizing with session state for performance
    
    # Download data button
    # FIX: Improved data concatenation for export
    export_data = []
    if nom_df is not None:
        export_data.append(nom_df.assign(Tipo="NOM-035"))
    if lean_df is not None:
        export_data.append(lean_df.assign(Tipo="LEAN"))
    if bienestar_df is not None:
        export_data.append(bienestar_df.assign(Tipo="Bienestar"))
    if action_plans_df is not None:
        export_data.append(action_plans_df.assign(Tipo="Planes_Accion"))
    
    combined_export = pd.concat(export_data, ignore_index=True) if export_data else pd.DataFrame()
    
    st.download_button(
        label="📥 Exportar Datos",
        data=combined_export.to_csv(index=False).encode('utf-8'),
        file_name="nom_lean_data.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    # Version info with better spacing
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #bdc3c7; font-size: 0.8rem; padding: 0.5rem;">
        <p style="margin-bottom: 0.25rem;">v2.2.0</p>
        <p style="margin: 0;">© 2024 RH Analytics</p>
        <p style="margin-top: 0.25rem; font-size: 0.7rem;">Última actualización: {}</p>
    </div>
    """.format(datetime.now().strftime("%d/%m/%Y")), unsafe_allow_html=True)

# ========== HEADER ==========
st.markdown(f"""
<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.5rem;">
    <div>
        <h1 style="margin-bottom: 0.25rem;">Sistema Integral NOM-035 & LEAN 2.0</h1>
        <p class="subheader">
            Monitoreo Estratégico de Bienestar Psicosocial y Eficiencia Operacional
        </p>
    </div>
    <div style="background-color: {COLOR_PALETTE['light']}; padding: 0.75rem 1.25rem; border-radius: 12px; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.05); min-width: 220px;">
        <div style="display: flex; align-items: center; margin-bottom: 0.25rem;">
            <span style="font-size: 1rem; margin-right: 0.5rem;">📅</span>
            <div>
                <div style="font-size: 0.85rem; color: {COLOR_PALETTE['primary']}; font-weight: 600;">
                    {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}
                </div>
                <div style="font-size: 0.75rem; color: {COLOR_PALETTE['text_light']};">
                    Actualizado: {datetime.now().strftime('%d/%m/%Y %H:%M')}
                </div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ========== KPI CARDS ==========
def kpi_card(value, title, target, icon="📊", help_text=None, unit="%"):
    """Create a professional KPI card with trend indicator"""
    delta = value - target
    percentage = min(100, (value / target * 100)) if target != 0 else 0
    
    # Determine status and colors
    if value >= target:
        status = "✅"
        color = COLOR_PALETTE['success']
        delta_text = f"+{delta}{unit} sobre meta"
        trend_icon = "↑"
    elif value >= target * 0.9:  # Within 10% of target
        status = "⚠️"
        color = COLOR_PALETTE['warning']
        delta_text = f"{delta}{unit} bajo meta"
        trend_icon = "→"
    else:
        status = "❌"
        color = COLOR_PALETTE['danger']
        delta_text = f"{delta}{unit} bajo meta"
        trend_icon = "↓"
    
    # Create card HTML
    card_html = f"""
    <div class="kpi-card" style='background-color: white; padding: 1.25rem; border-radius: 12px; 
                box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #f0f0f0; height: 100%;'>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;'>
            <div style='font-size: 1rem; color: {COLOR_PALETTE['text']}; font-weight: 600;'>
                {icon} {title}
            </div>
            <div style='font-size: 0.9rem; color: {color}; font-weight: 600;'>
                {trend_icon} {status}
            </div>
        </div>
        <div style='font-size: 2rem; font-weight: 700; color: {color}; margin-bottom: 8px;'>
            {value}{unit}
        </div>
        <div style='font-size: 0.85rem; color: {COLOR_PALETTE['text_light']}; margin-bottom: 12px;'>
            Meta: {target}{unit} • {delta_text}
        </div>
        <div style='height: 8px; background: #f0f0f0; border-radius: 4px;'>
            <div style='width: {percentage}%; height: 8px; background: {color}; border-radius: 4px; 
                        transition: width 0.5s ease;'></div>
        </div>
    </div>
    """
    
    if help_text:
        # Create a container with tooltip
        with st.container():
            st.markdown(card_html, unsafe_allow_html=True)
            st.markdown(f"<span title='{help_text}'></span>", unsafe_allow_html=True)
    else:
        st.markdown(card_html, unsafe_allow_html=True)

# Calculate KPIs from data
nom_compliance = nom_df['Evaluaciones'].mean()
lean_adoption = lean_df['Eficiencia'].mean()
wellbeing_index = bienestar_df['Índice_Bienestar'].mean()
operational_efficiency = lean_df['Eficiencia'].mean()
incident_rate = nom_df['Incidentes'].sum()
avg_projects = lean_df['Proyectos_Activos'].mean()

# Display KPIs in columns
cols = st.columns(4)
with cols[0]: 
    kpi_card(
        round(nom_compliance), 
        "Cumplimiento NOM-035", 
        nom_target, 
        "📋",
        "Porcentaje de cumplimiento con la norma NOM-035 en evaluaciones realizadas"
    )
with cols[1]: 
    kpi_card(
        round(lean_adoption), 
        "Adopción LEAN", 
        lean_target, 
        "🔄",
        "Nivel de implementación de metodologías LEAN en los departamentos"
    )
with cols[2]: 
    kpi_card(
        round(wellbeing_index), 
        "Índice Bienestar", 
        wellbeing_target, 
        "😊",
        "Indicador general de bienestar organizacional basado en encuestas"
    )
with cols[3]: 
    kpi_card(
        round(operational_efficiency), 
        "Eficiencia Operativa", 
        efficiency_target, 
        "⚙️",
        "Eficiencia general de los procesos operativos medidos"
    )

# Secondary KPIs row
cols_secondary = st.columns(4)
with cols_secondary[0]:
    kpi_card(
        incident_rate, 
        "Incidentes Psicosociales", 
        10, 
        "⚠️",
        "Total de incidentes reportados relacionados con factores de riesgo psicosocial",
        unit=""
    )
with cols_secondary[1]:
    kpi_card(
        round(avg_projects, 1), 
        "Proyectos Activos", 
        3, 
        "📌",
        "Promedio de proyectos LEAN activos por departamento",
        unit=""
    )
with cols_secondary[2]:
    kpi_card(
        round(bienestar_df['Ausentismo'].iloc[-1], 1), 
        "Tasa Ausentismo", 
        8, 
        "🏥",
        "Porcentaje de ausentismo laboral en el último mes",
        unit="%"
    )
with cols_secondary[3]:
    kpi_card(
        action_plans_df[action_plans_df['Estado'] == 'Completado'].shape[0], 
        "Planes Completados", 
        5, 
        "✅",
        "Número de planes de acción completados este año",
        unit=""
    )

# ========== MAIN CONTENT TABS ==========
tab1, tab2, tab3, tab4 = st.tabs(["📋 NOM-035", "🔄 LEAN 2.0", "😊 Bienestar", "📝 Planes de Acción"])

with tab1:
    st.markdown("#### Cumplimiento NOM-035 por Departamento")
    
    # Filter data
    filtered_nom = nom_df[nom_df['Departamento'].isin(departamentos_filtro)]
    
    # Validate filtered data
    if filtered_nom.empty:
        st.warning("⚠️ No hay datos disponibles para los departamentos seleccionados")
    else:
        # Create tabs for different views
        nom_view1, nom_view2, nom_view3 = st.tabs(["📈 Métricas Principales", "🗺️ Mapa de Riesgo", "📊 Análisis de Tendencia"])
        
        with nom_view1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Bar chart for evaluations and trainings with better interactivity
                fig = px.bar(
                    filtered_nom, 
                    x="Departamento", 
                    y=["Evaluaciones", "Capacitaciones"], 
                    barmode="group",
                    color_discrete_sequence=[COLOR_PALETTE['primary'], COLOR_PALETTE['secondary']],
                    labels={'value': 'Porcentaje', 'variable': 'Métrica'},
                    height=450,
                    text='auto'  # FIX: Replaced deprecated text_auto
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    yaxis_range=[0, 100],
                    legend_title_text='Métrica',
                    xaxis_title="",
                    margin=dict(l=20, r=20, t=20, b=20),
                    hovermode="x unified",
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=12,
                        font_family="Inter"
                    )
                )
                fig.update_traces(
                    textfont_size=12,
                    textangle=0,
                    textposition="outside",
                    cliponaxis=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Incident analysis
                st.markdown("**📌 Análisis de Incidentes**")
                fig_incidents = px.bar(
                    filtered_nom,
                    x='Departamento',
                    y='Incidentes',
                    color='Riesgo_Psicosocial',
                    color_discrete_map={
                        'Bajo': COLOR_PALETTE['success'],
                        'Medio': COLOR_PALETTE['warning'],
                        'Alto': COLOR_PALETTE['danger']
                    },
                    labels={'Incidentes': 'Número de Incidentes'},
                    height=300
                )
                fig_incidents.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis_title="",
                    showlegend=True,
                    legend_title_text='Nivel de Riesgo'
                )
                st.plotly_chart(fig_incidents, use_container_width=True)
            
            with col2:
                st.markdown("**📊 Resumen de Indicadores**")
                
                # Create styled dataframe
                summary_df = filtered_nom.set_index('Departamento')[['Evaluaciones', 'Capacitaciones', 'Incidentes', 'Tendencia']]
                summary_df['Riesgo'] = filtered_nom['Riesgo_Psicosocial']
                
                # Apply conditional formatting
                def color_trend(val):
                    if val > 0:
                        color = COLOR_PALETTE['success']
                    elif val < 0:
                        color = COLOR_PALETTE['danger']
                    else:
                        color = COLOR_PALETTE['text']
                    return f'color: {color}; font-weight: bold'
                
                styled_df = summary_df.style \
                    .background_gradient(subset=['Evaluaciones', 'Capacitaciones'], cmap='Blues') \
                    .applymap(color_trend, subset=['Tendencia']) \
                    .format({'Tendencia': "{:.2f}%", 'Evaluaciones': "{:.0f}%", 'Capacitaciones': "{:.0f}%"})
                
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    height=450
                )
                
                # Risk distribution pie chart
                st.markdown("**📌 Distribución de Riesgo**")
                risk_counts = filtered_nom['Riesgo_Psicosocial'].value_counts().reset_index()
                fig_risk = px.pie(
                    risk_counts,
                    values='count',
                    names='Riesgo_Psicosocial',
                    color='Riesgo_Psicosocial',
                    color_discrete_map={
                        'Bajo': COLOR_PALETTE['success'],
                        'Medio': COLOR_PALETTE['warning'],
                        'Alto': COLOR_PALETTE['danger']
                    },
                    hole=0.4,
                    height=250
                )
                fig_risk.update_layout(
                    showlegend=False,
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                fig_risk.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    hovertemplate="<b>%{label}</b><br>%{value} departamentos"
                )
                st.plotly_chart(fig_risk, use_container_width=True)
        
        with nom_view2:
            # Risk heatmap with better visualization
            st.markdown("**🔍 Mapa de Riesgo Psicosocial**")
            
            # Prepare data for heatmap
            heatmap_data = filtered_nom.set_index('Departamento')[['Evaluaciones', 'Capacitaciones', 'Incidentes']]
            heatmap_data.columns = ['Evaluaciones (%)', 'Capacitaciones (%)', 'Incidentes (n)']
            
            # Normalize data for better color scaling
            scaler = MinMaxScaler()
            heatmap_values = scaler.fit_transform(heatmap_data)
            
            fig_heat = go.Figure(data=go.Heatmap(
                z=heatmap_values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                text=heatmap_data.values,
                texttemplate="%{text}",
                colorscale='RdYlGn',
                reversescale=True,
                hoverongaps=False,
                colorbar=dict(
                    title="Riesgo",
                    titleside="right",
                    tickvals=[0, 0.5, 1],
                    ticktext=["Alto", "Medio", "Bajo"]
                )
            ))
            
            fig_heat.update_layout(
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=50, r=50, t=50, b=50),
                xaxis_title="",
                yaxis_title="Departamento",
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_family="Inter"
                )
            )
            
            st.plotly_chart(fig_heat, use_container_width=True)
            
            # Risk interpretation guide
            st.markdown("""
            <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                <h4 style="margin-top: 0; color: #2c3e50;">Guía de Interpretación</h4>
                <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 1rem;">
                    <div style="flex: 1; min-width: 200px;">
                        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                            <div style="width: 20px; height: 20px; background-color: #e74c3c; margin-right: 0.5rem; border-radius: 4px;"></div>
                            <span style="font-weight: 600;">Alto Riesgo</span>
                        </div>
                        <p style="margin: 0; font-size: 0.9rem; color: #7f8c8d;">
                            Evaluaciones < 70%, Capacitaciones < 60%, o Incidentes > 5
                        </p>
                    </div>
                    <div style="flex: 1; min-width: 200px;">
                        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                            <div style="width: 20px; height: 20px; background-color: #f39c12; margin-right: 0.5rem; border-radius: 4px;"></div>
                            <span style="font-weight: 600;">Riesgo Medio</span>
                        </div>
                        <p style="margin: 0; font-size: 0.9rem; color: #7f8c8d;">
                            Evaluaciones 70-80%, Capacitaciones 60-75%, o Incidentes 2-5
                        </p>
                    </div>
                    <div style="flex: 1; min-width: 200px;">
                        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                            <div style="width: 20px; height: 20px; background-color: #27ae60; margin-right: 0.5rem; border-radius: 4px;"></div>
                            <span style="font-weight: 600;">Bajo Riesgo</span>
                        </div>
                        <p style="margin: 0; font-size: 0.9rem; color: #7f8c8d;">
                            Evaluaciones > 80%, Capacitaciones > 75%, y Incidentes < 2
                        </p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with nom_view3:
            # Trend analysis with more insights
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Trend bar chart with reference line
                fig_trend = px.bar(
                    filtered_nom, 
                    x='Departamento', 
                    y='Tendencia',
                    color='Tendencia',
                    color_continuous_scale='RdYlGn',
                    range_color=[-3, 3],
                    labels={'Tendencia': 'Cambio mensual (%)'},
                    height=450
                )
                
                # Add reference lines
                fig_trend.add_hline(
                    y=0, 
                    line_dash="dot", 
                    line_color=COLOR_PALETTE['text_light'],
                    annotation_text="Línea Base", 
                    annotation_position="bottom right"
                )
                
                fig_trend.update_layout(
                    title="Tendencia de Cumplimiento (Último Mes)",
                    yaxis_title="Cambio en puntos porcentuales",
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=40, b=20),
                    coloraxis_colorbar=dict(
                        title="Cambio",
                        tickvals=[-3, 0, 3],
                        ticktext=["↓ Baja", "Neutral", "↑ Alta"]
                    )
                )
                
                st.plotly_chart(fig_trend, use_container_width=True)
                
                # Trend vs current score scatter plot
                st.markdown("**📈 Relación entre Puntuación y Tendencia**")
                fig_scatter = px.scatter(
                    filtered_nom,
                    x='Evaluaciones',
                    y='Tendencia',
                    color='Riesgo_Psicosocial',
                    size='Incidentes',
                    hover_name='Departamento',
                    color_discrete_map={
                        'Bajo': COLOR_PALETTE['success'],
                        'Medio': COLOR_PALETTE['warning'],
                        'Alto': COLOR_PALETTE['danger']
                    },
                    labels={
                        'Evaluaciones': 'Evaluaciones (%)',
                        'Tendencia': 'Cambio Mensual (%)',
                        'Incidentes': 'Número de Incidentes'
                    },
                    height=350
                )
                
                # Add reference lines and zones
                fig_scatter.add_vline(
                    x=nom_target,
                    line_dash="dot",
                    line_color=COLOR_PALETTE['primary'],
                    annotation_text=f"Meta: {nom_target}%", 
                    annotation_position="top right"
                )
                
                fig_scatter.add_hline(
                    y=0,
                    line_dash="dot",
                    line_color=COLOR_PALETTE['text_light']
                )
                
                fig_scatter.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    legend_title_text='Nivel de Riesgo',
                    xaxis_range=[50, 100],
                    yaxis_range=[-3, 3]
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with col2:
                st.markdown("**📊 Interpretación de Tendencias**")
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                        <span style="color: #27ae60; font-weight: bold; margin-right: 0.5rem;">↑ Positivo</span>
                        <span>Mejora continua en el indicador</span>
                    </div>
                    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                        <span style="color: #e74c3c; font-weight: bold; margin-right: 0.5rem;">↓ Negativo</span>
                        <span>Requiere atención inmediata</span>
                    </div>
                    <div style="display: flex; align-items: center;">
                        <span style="color: #f39c12; font-weight: bold; margin-right: 0.5rem;">→ Neutral</span>
                        <span>Mantener monitoreo</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("**📌 Recomendaciones**")
                st.markdown("""
                <div style="background-color: #eaf2f8; padding: 1rem; border-radius: 10px;">
                    <p style="margin-top: 0; margin-bottom: 0.5rem; font-weight: 600;">Para tendencias negativas:</p>
                    <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.9rem;">
                        <li>Revisar carga de trabajo</li>
                        <li>Evaluar clima laboral</li>
                        <li>Reforzar capacitaciones</li>
                    </ul>
                    
                    <p style="margin-top: 0.75rem; margin-bottom: 0.5rem; font-weight: 600;">Para tendencias positivas:</p>
                    <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.9rem;">
                        <li>Identificar mejores prácticas</li>
                        <li>Replicar en otros áreas</li>
                        <li>Reconocer logros</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Department comparison
                st.markdown("**📊 Comparación Departamental**")
                dept_comparison = st.selectbox(
                    "Seleccionar métrica para comparar",
                    options=['Evaluaciones', 'Capacitaciones', 'Incidentes', 'Tendencia'],
                    label_visibility="collapsed"
                )
                
                fig_comparison = px.bar(
                    filtered_nom.sort_values(dept_comparison),
                    y='Departamento',
                    x=dept_comparison,
                    orientation='h',
                    color=dept_comparison,
                    color_continuous_scale='Blues',
                    height=300
                )
                fig_comparison.update_layout(
                    showlegend=False,
                    xaxis_title="",
                    yaxis_title="",
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                st.plotly_chart(fig_comparison, use_container_width=True)

with tab2:
    st.markdown("#### Progreso LEAN 2.0")
    filtered_lean = lean_df[lean_df['Departamento'].isin(departamentos_filtro)]
    
    if filtered_lean.empty:
        st.warning("⚠️ No hay datos disponibles para los departamentos seleccionados")
    else:
        # Create columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Efficiency bar chart with target line
            fig_lean = px.bar(
                filtered_lean, 
                x='Departamento', 
                y='Eficiencia', 
                color='Nivel_LEAN', 
                color_discrete_map={
                    'Inicial': COLOR_PALETTE['warning'],
                    'Intermedio': COLOR_PALETTE['secondary'],
                    'Avanzado': COLOR_PALETTE['success']
                },
                labels={'Eficiencia': 'Eficiencia (%)'},
                height=400,
                text='auto'  # FIX: Replaced deprecated text_auto
            )
            
            # Add target line
            fig_lean.add_hline(
                y=lean_target,
                line_dash="dot",
                line_color=COLOR_PALETTE['danger'],
                annotation_text=f"Meta: {lean_target}%",
                annotation_position="top right"
            )
            
            fig_lean.update_layout(
                title="Eficiencia por Departamento",
                yaxis_range=[0, 100],
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=40, b=20),
                legend_title_text='Nivel LEAN',
                xaxis_title="",
                hovermode="x unified"
            )
            st.plotly_chart(fig_lean, use_container_width=True)
            
            # Waste reduction vs efficiency scatter plot with regression
            fig_scatter = px.scatter(
                filtered_lean,
                x='Reducción_Desperdicio',
                y='Eficiencia',
                size='Proyectos_Activos',
                color='Departamento',
                hover_name='Departamento',
                trendline="lowess",
                labels={
                    'Reducción_Desperdicio': 'Reducción de Desperdicio (%)',
                    'Eficiencia': 'Eficiencia (%)',
                    'Proyectos_Activos': 'Proyectos Activos'
                },
                height=400
            )
            
            # Add target zones
            fig_scatter.add_shape(
                type="rect",
                x0=15, y0=80, x1=30, y1=100,
                line=dict(color=COLOR_PALETTE['success'], width=1),
                fillcolor=COLOR_PALETTE['success'] + "20",
                opacity=0.2,
                label=dict(text="Zona Óptima")
            )
            
            fig_scatter.update_layout(
                title="Relación Eficiencia vs Reducción de Desperdicio",
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=40, b=20),
                legend_title_text='Departamento'
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Radar chart for multiple metrics
            st.markdown("**📊 Comparación de Métricas**")
            
            # Normalize data for radar chart
            metrics = ['Eficiencia', 'Reducción_Desperdicio', '5S_Score', 'SMED']
            scaler = MinMaxScaler()
            lean_radar = filtered_lean.copy()
            lean_radar[metrics] = scaler.fit_transform(lean_radar[metrics])
            
            fig_radar = go.Figure()
            
            for dept in filtered_lean['Departamento']:
                row = lean_radar[lean_radar['Departamento'] == dept].iloc[0]
                fig_radar.add_trace(go.Scatterpolar(
                    r=row[metrics].values,
                    theta=metrics,
                    fill='toself',
                    name=dept,
                    line=dict(width=2),
                    hoverinfo='name+r+theta'
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1]),
                    angularaxis=dict(
                        tickfont=dict(size=10)
                    )
                ),
                height=400,
                showlegend=True,
                margin=dict(l=50, r=50, t=30, b=50),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                )
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Projects summary with interactive table
            with st.expander("📌 Detalle de Proyectos", expanded=True):
                projects_df = filtered_lean[['Departamento', 'Proyectos_Activos', 'Nivel_LEAN']] \
                    .sort_values('Proyectos_Activos', ascending=False)
                
                st.dataframe(
                    projects_df.style \
                        .background_gradient(subset=['Proyectos_Activos'], cmap='Greens') \
                        .applymap(lambda x: f"color: {COLOR_PALETTE['success']}" if x == 'Avanzado' 
                                 else f"color: {COLOR_PALETTE['warning']}" if x == 'Inicial'
                                 else f"color: {COLOR_PALETTE['secondary']}",
                                 subset=['Nivel_LEAN']),
                    use_container_width=True,
                    height=250,
                    hide_index=True
                )
            
            # LEAN maturity distribution
            st.markdown("**📈 Madurez LEAN**")
            maturity = filtered_lean['Nivel_LEAN'].value_counts().reset_index()
            fig_maturity = px.pie(
                maturity,
                values='count',
                names='Nivel_LEAN',
                color='Nivel_LEAN',
                color_discrete_map={
                    'Inicial': COLOR_PALETTE['warning'],
                    'Intermedio': COLOR_PALETTE['secondary'],
                    'Avanzado': COLOR_PALETTE['success']
                },
                hole=0.4,
                height=250
            )
            fig_maturity.update_layout(
                showlegend=False,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            fig_maturity.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate="<b>%{label}</b><br>%{value} departamentos"
            )
            st.plotly_chart(fig_maturity, use_container_width=True)

with tab3:
    st.markdown("#### Tendencias de Bienestar Organizacional")
    
    # FIX: Consistent date handling
    filtered_bienestar = bienestar_df[
        (bienestar_df['Mes'].dt.date >= start_date) & 
        (bienestar_df['Mes'].dt.date <= end_date)
    ]
    
    if filtered_bienestar.empty:
        st.warning("⚠️ No hay datos disponibles para el período seleccionado")
    else:
        # Main metrics with better visualization
        st.markdown("**📊 Indicadores Clave**")
        cols = st.columns(4)
        
        # Calculate changes
        encuestas_change = filtered_bienestar['Encuestas'].iloc[-1] - filtered_bienestar['Encuestas'].iloc[0] if len(filtered_bienestar) > 1 else 0
        ausentismo_change = filtered_bienestar['Ausentismo'].iloc[-1] - filtered_bienestar['Ausentismo'].iloc[0] if len(filtered_bienestar) > 1 else 0
        rotacion_change = filtered_bienestar['Rotación'].iloc[-1] - filtered_bienestar['Rotación'].iloc[0] if len(filtered_bienestar) > 1 else 0
        bienestar_change = filtered_bienestar['Índice_Bienestar'].iloc[-1] - filtered_bienestar['Índice_Bienestar'].iloc[0] if len(filtered_bienestar) > 1 else 0
        
        with cols[0]:
            st.metric(
                label="Encuestas Completadas", 
                value=f"{filtered_bienestar['Encuestas'].mean():.0f}%",
                delta=f"{encuestas_change:+.0f}%",
                delta_color="normal",
                help="Porcentaje de encuestas de bienestar completadas"
            )
        with cols[1]:
            st.metric(
                label="Índice de Bienestar", 
                value=f"{filtered_bienestar['Índice_Bienestar'].iloc[-1]:.1f}",
                delta=f"{bienestar_change:+.1f} puntos",
                delta_color="inverse" if bienestar_change < 0 else "normal",
                help="Índice general de bienestar organizacional"
            )
        with cols[2]:
            st.metric(
                label="Tasa de Ausentismo", 
                value=f"{filtered_bienestar['Ausentismo'].iloc[-1]:.1f}%",
                delta=f"{ausentismo_change:+.1f}%",
                delta_color="normal" if ausentismo_change < 0 else "inverse",
                help="Porcentaje de ausentismo laboral"
            )
        with cols[3]:
            st.metric(
                label="Tasa de Rotación", 
                value=f"{filtered_bienestar['Rotación'].iloc[-1]:.1f}%",
                delta=f"{rotacion_change:+.1f}%",
                delta_color="normal" if rotacion_change < 0 else "inverse",
                help="Porcentaje de rotación de personal"
            )
        
        # Create tabs for different views
        wellbeing_view1, wellbeing_view2, wellbeing_view3 = st.tabs(["📈 Tendencias Mensuales", "🔍 Análisis de Correlación", "📊 Desglose por Componente"])
        
        with wellbeing_view1:
            # Line chart for wellbeing metrics with better interactivity
            fig_bienestar = px.line(
                filtered_bienestar, 
                x='Mes', 
                y=['Índice_Bienestar', 'Ausentismo', 'Rotación'], 
                markers=True,
                color_discrete_sequence=[
                    COLOR_PALETTE['success'], 
                    COLOR_PALETTE['danger'], 
                    COLOR_PALETTE['warning']
                ],
                labels={'value': 'Porcentaje', 'variable': 'Métrica'},
                height=450
            )
            
            # Add target lines
            fig_bienestar.add_hline(
                y=wellbeing_target,
                line_dash="dot",
                line_color=COLOR_PALETTE['success'],
                annotation_text=f"Meta Bienestar: {wellbeing_target}%",
                annotation_position="bottom right"
            )
            
            fig_bienestar.add_hline(
                y=8,
                line_dash="dot",
                line_color=COLOR_PALETTE['danger'],
                annotation_text="Meta Ausentismo: 8%",
                annotation_position="top right"
            )
            
            fig_bienestar.update_layout(
                title="Evolución Mensual de Bienestar",
                yaxis_range=[0, 100],
                plot_bgcolor='rgba(0,0,0,0)',
                legend_title="Métrica",
                margin=dict(l=20, r=20, t=40, b=20),
                hovermode="x unified",
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_family="Inter"
                )
            )
            
            # Customize legend names
            for i, name in enumerate(['Índice Bienestar', 'Ausentismo', 'Rotación']):
                fig_bienestar.data[i].name = name
            
            st.plotly_chart(fig_bienestar, use_container_width=True)
            
            # Survey completion rate
            st.markdown("**📝 Encuestas de Bienestar**")
            fig_surveys = px.bar(
                filtered_bienestar,
                x='Mes',
                y='Encuestas',
                color='Encuestas',
                color_continuous_scale='Blues',
                range_color=[80, 100],
                labels={'Encuestas': 'Encuestas Completadas (%)'},
                height=300
            )
            fig_surveys.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="",
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_surveys, use_container_width=True)
        
        with wellbeing_view2:
            # Correlation analysis with more metrics
            st.markdown("**🔍 Relación entre Indicadores**")
            
            corr_matrix = filtered_bienestar[['Índice_Bienestar', 'Ausentismo', 'Rotación', 'Encuestas', 'Satisfacción', 'Compromiso']].corr()
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                color_continuous_scale='RdYlGn',
                range_color=[-1, 1],
                labels=dict(x="Métrica", y="Métrica", color="Correlación"),
                height=500,
                aspect="auto"
            )
            
            fig_corr.update_layout(
                title="Matriz de Correlación",
                xaxis_title="",
                yaxis_title="",
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Correlation insights
            st.markdown("""
            <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                <h4 style="margin-top: 0; color: #2c3e50;">Interpretación de Correlaciones</h4>
                <ul style="margin: 0; padding-left: 1.2rem;">
                    <li><b>Correlación positiva fuerte (0.7-1.0):</b> Las métricas se mueven en la misma dirección</li>
                    <li><b>Correlación negativa fuerte (-1.0 - -0.7):</b> Las métricas se mueven en direcciones opuestas</li>
                    <li><b>Correlación débil (-0.3 - 0.3):</b> Poca o ninguna relación aparente</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with wellbeing_view3:
            # Component breakdown
            st.markdown("**📊 Componentes del Bienestar**")
            
            fig_components = px.line(
                filtered_bienestar,
                x='Mes',
                y=['Índice_Bienestar', 'Satisfacción', 'Compromiso'],
                markers=True,
                color_discrete_sequence=[
                    COLOR_PALETTE['primary'],
                    COLOR_PALETTE['success'],
                    COLOR_PALETTE['secondary']
                ],
                labels={'value': 'Puntuación', 'variable': 'Componente'},
                height=450
            )
            
            fig_components.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                legend_title="Componente",
                margin=dict(l=20, r=20, t=40, b=20),
                hovermode="x unified"
            )
            
            # Customize legend names
            for i, name in enumerate(['Índice General', 'Satisfacción', 'Compromiso']):
                fig_components.data[i].name = name
            
            st.plotly_chart(fig_components, use_container_width=True)
            
            # Component contribution
            st.markdown("**📌 Contribución al Índice**")
            cols = st.columns(3)
            
            with cols[0]:
                st.markdown("""
                <div style="background-color: #eaf2f8; padding: 1rem; border-radius: 10px; height: 100%;">
                    <h4 style="margin-top: 0; color: #2c3e50;">Satisfacción</h4>
                    <p style="margin-bottom: 0; font-size: 0.9rem;">
                        Mide el nivel de satisfacción general de los empleados con su trabajo, condiciones laborales y ambiente de trabajo.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[1]:
                st.markdown("""
                <div style="background-color: #e8f8f5; padding: 1rem; border-radius: 10px; height: 100%;">
                    <h4 style="margin-top: 0; color: #2c3e50;">Compromiso</h4>
                    <p style="margin-bottom: 0; font-size: 0.9rem;">
                        Evalúa el nivel de involucramiento y conexión emocional de los empleados con la organización.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[2]:
                st.markdown("""
                <div style="background-color: #fef9e7; padding: 1rem; border-radius: 10px; height: 100%;">
                    <h4 style="margin-top: 0; color: #2c3e50;">Índice General</h4>
                    <p style="margin-bottom: 0; font-size: 0.9rem;">
                        Combinación ponderada de todos los componentes que miden el bienestar organizacional.
                    </p>
                </div>
                """, unsafe_allow_html=True)

with tab4:
    st.markdown("#### Planes de Acción y Seguimiento")
    
    # Filter action plans by selected departments
    filtered_plans = action_plans_df[
        action_plans_df['Departamento'].isin(departamentos_filtro if departamentos_filtro else DEPARTMENTS)
    ]
    
    # FIX: Validate filtered plans
    if filtered_plans.empty:
        st.warning("⚠️ No hay planes de acción para los departamentos seleccionados")
    else:
        # Display existing action plans with better visualization
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("**📌 Planes Registrados**")
            
            # Create a styled dataframe with progress bars
            def style_action_plans(df):
                styled = df.copy()
                
                # Format dates
                styled['Plazo'] = styled['Plazo'].apply(lambda x: x.strftime('%d/%m/%Y'))
                
                # Add color based on status
                status_colors = {
                    'Completado': COLOR_PALETTE['success'],
                    'En progreso': COLOR_PALETTE['warning'],
                    'Pendiente': COLOR_PALETTE['danger']
                }
                
                priority_colors = {
                    'Alta': COLOR_PALETTE['danger'],
                    'Media': COLOR_PALETTE['warning'],
                    'Baja': COLOR_PALETTE['success']
                }
                
                # Apply styling
                styled['Estado'] = styled['Estado'].apply(
                    lambda x: f"<span style='color: {status_colors[x]}; font-weight: bold;'>{x}</span>"
                )
                
                styled['Prioridad'] = styled['Prioridad'].apply(
                    lambda x: f"<span style='color: {priority_colors[x]}; font-weight: bold;'>{x}</span>"
                )
                
                # Add progress bars
                styled['Avance'] = styled['%_Avance'].apply(
                    lambda x: f"""
                    <div style="position: relative; height: 20px; background: #f0f0f0; border-radius: 4px;">
                        <div style="position: absolute; height: 20px; width: {x}%; background: {COLOR_PALETTE['secondary']}; border-radius: 4px; 
                                    display: flex; align-items: center; justify-content: center; color: white; font-size: 0.7rem; font-weight: bold;">
                            {x}%
                        </div>
                    </div>
                    """
                )
                
                return styled[['Departamento', 'Problema', 'Acción', 'Responsable', 'Plazo', 'Estado', 'Prioridad', 'Avance']]
            
            styled_plans = style_action_plans(filtered_plans)
            
            # Display with custom HTML for better rendering
            st.markdown(
                styled_plans.to_html(escape=False, index=False),
                unsafe_allow_html=True
            )
            
            # Add spacing
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            
            # Status distribution chart
            st.markdown("**📊 Distribución de Estados**")
            status_cols = st.columns(3)
            
            with status_cols[0]:
                completed = filtered_plans[filtered_plans['Estado'] == 'Completado'].shape[0]
                st.metric("Completados", completed, delta=f"{completed/len(filtered_plans)*100:.1f}%")
            
            with status_cols[1]:
                in_progress = filtered_plans[filtered_plans['Estado'] == 'En progreso'].shape[0]
                st.metric("En Progreso", in_progress, delta=f"{in_progress/len(filtered_plans)*100:.1f}%")
            
            with status_cols[2]:
                pending = filtered_plans[filtered_plans['Estado'] == 'Pendiente'].shape[0]
                st.metric("Pendientes", pending, delta=f"{pending/len(filtered_plans)*100:.1f}%")
        
        with col2:
            st.markdown("**📅 Vencimientos Próximos**")
            
            # Calculate days until deadline and filter
            today = date.today()
            upcoming = filtered_plans.copy()
            upcoming['Días Restantes'] = (upcoming['Plazo'] - today).apply(lambda x: x.days)
            upcoming = upcoming[upcoming['Días Restantes'] <= 30].sort_values('Días Restantes')
            
            if not upcoming.empty:
                for _, row in upcoming.iterrows():
                    days_left = row['Días Restantes']
                    
                    # Determine color based on days left
                    if days_left < 0:
                        status_color = COLOR_PALETTE['danger']
                        status_text = f"Vencido hace {-days_left} días"
                    elif days_left < 7:
                        status_color = COLOR_PALETTE['danger']
                        status_text = f"Vence en {days_left} días"
                    elif days_left < 14:
                        status_color = COLOR_PALETTE['warning']
                        status_text = f"Vence en {days_left} días"
                    else:
                        status_color = COLOR_PALETTE['success']
                        status_text = f"Vence en {days_left} días"
                    
                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; padding: 0.75rem; border-radius: 8px; margin-bottom: 0.75rem; border-left: 4px solid {status_color};">
                        <div style="font-weight: 600; margin-bottom: 0.25rem;">{row['Departamento']}</div>
                        <div style="font-size: 0.8rem; margin-bottom: 0.25rem; color: {COLOR_PALETTE['text_light']};">
                            {row['Problema'][:30]}...
                        </div>
                        <div style="font-size: 0.8rem; color: {status_color}; font-weight: 500;">
                            {status_text}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("🎉 No hay planes con vencimiento próximo", icon="ℹ️")
            
            # Priority distribution pie chart
            st.markdown("**📌 Distribución por Prioridad**")
            priority_counts = filtered_plans['Prioridad'].value_counts().reset_index()
            fig_priority = px.pie(
                priority_counts,
                values='count',
                names='Prioridad',
                color='Prioridad',
                color_discrete_map={
                    'Alta': COLOR_PALETTE['danger'],
                    'Media': COLOR_PALETTE['warning'],
                    'Baja': COLOR_PALETTE['success']
                },
                hole=0.5,
                height=250
            )
            fig_priority.update_layout(
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            fig_priority.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate="<b>%{label}</b><br>%{value} planes"
            )
            st.plotly_chart(fig_priority, use_container_width=True)
    
    # Add new action plan form with better UX
    with st.expander("➕ Registrar Nuevo Plan de Acción", expanded=False):
        with st.form("nuevo_plan_form", clear_on_submit=True):
            st.markdown("**📝 Detalles del Plan**")
            col1, col2 = st.columns(2)
            
            with col1:
                dept = st.selectbox("Departamento", DEPARTMENTS, index=0)
                problema = st.text_area(
                    "Problema identificado", 
                    max_chars=200,
                    placeholder="Describa el problema identificado...",
                    help="Sea específico sobre el problema que necesita atención"
                )
                prioridad = st.selectbox(
                    "Prioridad", 
                    ["Alta", "Media", "Baja"],
                    help="Seleccione la urgencia e importancia de este plan"
                )
            
            with col2:
                accion = st.text_area(
                    "Acción propuesta", 
                    max_chars=200,
                    placeholder="Describa la acción a implementar...",
                    help="Sea claro sobre los pasos a seguir para resolver el problema"
                )
                responsable = st.text_input(
                    "Responsable",
                    placeholder="Nombre del responsable",
                    help="Persona encargada de implementar el plan"
                )
                plazo = st.date_input(
                    "Plazo estimado",
                    min_value=date.today(),
                    value=date.today() + timedelta(days=30),
                    format="DD/MM/YYYY",
                    help="Fecha límite para completar el plan"
                )
                avance = st.slider(
                    "% Avance", 
                    0, 100, 0,
                    help="Porcentaje de completitud actual del plan"
                )
            
            # Form submission
            submitted = st.form_submit_button(
                "💾 Guardar Plan de Acción",
                use_container_width=True
            )
            
            if submitted:
                # Validate required fields
                if not problema or not accion or not responsable:
                    st.error("Por favor complete todos los campos requeridos")
                else:
                    # FIX: Validate unique ID
                    new_id = max(action_plans_df['ID']) + 1 if not action_plans_df.empty else 1
                    # In a real app, this would save to a database
                    new_plan = pd.DataFrame([{
                        'ID': new_id,
                        'Departamento': dept,
                        'Problema': problema,
                        'Acción': accion,
                        'Responsable': responsable,
                        'Plazo': plazo,
                        'Estado': 'Pendiente' if avance == 0 else 'En progreso' if avance < 100 else 'Completado',
                        'Prioridad': prioridad,
                        '%_Avance': avance,
                        'Impacto_Esperado': 'Alto' if prioridad == 'Alta' else 'Medio' if prioridad == 'Media' else 'Bajo'
                    }])
                    
                    action_plans_df = pd.concat([action_plans_df, new_plan], ignore_index=True)
                    st.success("✅ Plan de acción registrado correctamente")
                    st.balloons()
                    st.rerun()  # TODO: Optimize with session state if possible

# ========== EXPORT AND REPORTING ==========
st.markdown("---")
st.markdown("#### Exportación de Datos y Reportes")

export_col1, export_col2, export_col3 = st.columns(3)

with export_col1:
    with st.expander("📄 Generar Reporte PDF", expanded=False):
        report_type = st.selectbox(
            "Tipo de reporte", 
            ["Completo", "Resumido", "Solo NOM-035", "Solo LEAN"],
            label_visibility="collapsed"
        )
        
        report_options = st.multiselect(
            "Incluir secciones",
            ["KPIs", "Gráficos", "Planes de Acción", "Recomendaciones"],
            default=["KPIs", "Gráficos", "Planes de Acción"]
        )
        
       ---------------------------------------------------------------------------if st.button(
            "🖨️ Generar Reporte", 
            use_container_width=True,
            help="Genera un reporte PDF con los datos actuales"
        ):
            with st.spinner("Generando reporte..."):
                # Simulate report generation
                import time
                time.sleep(2)
                st.toast("✅ Reporte generado exitosamente", icon="📄")

with export_col2:
    with st.expander("📧 Enviar por Correo", expanded=False):
        email = st.text_input(
            "Dirección de correo", 
            placeholder="usuario@empresa.com",
            help="Ingrese la dirección de correo del destinatario"
        )
        
        email_message = st.text_area(
            "Mensaje adicional",
            placeholder="Opcional: agregue un mensaje personalizado...",
            height=100
        )
        
        if st.button(
            "📤 Enviar Reporte", 
            use_container_width=True,
            help="Envía el reporte por correo electrónico"
        ):
            if email and "@" in email:
                with st.spinner("Enviando reporte..."):
                    # Simulate email sending
                    import time
                    time.sleep(2)
                    st.toast(f"✉️ Reporte enviado a {email}", icon="✅")
            else:
                st.warning("Por favor ingrese una dirección de correo válida")

with export_col3:
    with st.expander("📊 Exportar Datos", expanded=False):
        export_format = st.radio(
            "Formato", 
            ["CSV", "Excel", "JSON"],
            label_visibility="collapsed",
            horizontal=True
        )
        
        data_options = st.multiselect(
            "Seleccionar datos a exportar",
            ["NOM-035", "LEAN", "Bienestar", "Planes de Acción"],
            default=["NOM-035", "LEAN", "Bienestar", "Planes de Acción"]
        )
        
        # Prepare data based on selection
        export_data = []
        if "NOM-035" in data_options:
            export_data.append(nom_df.assign(Tipo="NOM-035"))
        if "LEAN" in data_options:
            export_data.append(lean_df.assign(Tipo="LEAN"))
        if "Bienestar" in data_options:
            export_data.append(bienestar_df.assign(Tipo="Bienestar"))
        if "Planes de Acción" in data_options:
                        export_data.append(action_plans_df.assign(Tipo="Planes_Accion"))
        
        if export_data:
            combined_data = pd.concat(export_data, ignore_index=True)
            
            try:
                if export_format == "CSV":
                    data = combined_data.to_csv(index=False).encode('utf-8')
                    mime = "text/csv"
                    ext = "csv"
                elif export_format == "Excel":
                    # FIX: Use BytesIO for proper Excel export
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        combined_data.to_excel(writer, index=False, sheet_name='NOM_LEAN_Data')
                    data = output.getvalue()
                    mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    ext = "xlsx"
                else:  # JSON
                    data = combined_data.to_json(orient='records', date_format='iso').encode('utf-8')
                    mime = "application/json"
                    ext = "json"
                
                st.download_button(
                    label=f"💾 Descargar (.{ext})",
                    data=data,
                    file_name=f"nom_lean_data.{ext}",
                    mime=mime,
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error al generar el archivo de exportación: {e}")
        else:
            st.warning("⚠️ Seleccione al menos un tipo de datos para exportar")

# ========== FOOTER ==========
st.markdown("""
<hr>
<div style="text-align: center; color: #7f8c8d; font-size: 0.8rem; padding: 1rem 0;">
    <p style="margin: 0.25rem 0;">Sistema Integral NOM-035 & LEAN 2.0 • Versión 2.2.0</p>
    <p style="margin: 0.25rem 0;">© 2024 Departamento de RH • Todos los derechos reservados</p>
    <p style="margin: 0.25rem 0; font-size: 0.7rem;">
        Para soporte técnico contacte a: <a href="mailto:soporte@empresa.com" style="color: #3498db;">soporte@empresa.com</a>
    </p>
</div>
""", unsafe_allow_html=True)  # SECURITY: Safe for static content; sanitize if dynamic inputs are added
           

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ========== PAGE CONFIGURATION - MUST BE FIRST STREAMLIT COMMAND ==========
st.set_page_config(
    page_title="NOM-035 & LEAN Dashboard",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# ========== CONSTANTS AND CONFIGURATION ==========
DEPARTMENTS = ['Producción', 'Calidad', 'Logística', 'Administración', 'Ventas', 'RH', 'TI']

class DesignSystem:
    # Color palette
    COLORS_PALETTE = {
        'primary': '#2563eb',    # More vibrant blue
        'secondary': '#4f46e5',  # Purple-blue
        'accent': '#7c3aed',     # Vibrant purple
        'success': '#10b981',    # Emerald green
        'warning': '#f59e0b',    # Amber
        'danger': '#ef4444',     # Red
        'light': '#f3f4f6',      # Light gray
        'dark': '#1f2937',       # Dark gray
        'background': '#ffffff', # Pure white
        'text': '#374151'        # Dark gray
    }
    
    # Typography
    FONT = "Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
    
    # Spacing
    SPACING = {
        'xs': '0.25rem',
        'sm': '0.5rem',
        'md': '1rem',
        'lg': '1.5rem',
        'xl': '2rem'
    }
    
    # Shadows
    SHADOWS = {
        'sm': '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
        'md': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
        'lg': '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)'
    }
    
    # Border radius
    RADIUS = {
        'sm': '0.375rem',
        'md': '0.5rem',
        'lg': '0.75rem',
        'full': '9999px'
    }


# Custom CSS for professional styling
st.markdown(f"""
<style>
    /* Main container */
    .main {{
        background-color: {COLOR_PALETTE['background']};
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
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }}
    
    /* Tabs */
    [data-baseweb="tab-list"] {{
        gap: 10px;
    }}
    
    [data-baseweb="tab"] {{
        background-color: {COLOR_PALETTE['light']} !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        margin: 0 5px !important;
    }}
    
    [aria-selected="true"] {{
        background-color: {COLOR_PALETTE['primary']} !important;
        color: white !important;
    }}
    
    /* Dataframes */
    .stDataFrame {{
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
</style>
""", unsafe_allow_html=True)

# [Rest of your code remains exactly the same...]

# ========== DATA LOADING AND PROCESSING ==========
@st.cache_data(ttl=600)
def load_data():
    """Load and generate synthetic data for the dashboard"""
    np.random.seed(42)
    
    # NOM-035 Data
    nom = pd.DataFrame({
        'Departamento': DEPARTMENTS,
        'Evaluaciones': np.random.randint(70, 100, len(DEPARTMENTS)),
        'Capacitaciones': np.random.randint(60, 100, len(DEPARTMENTS)),
        'Incidentes': np.random.randint(0, 10, len(DEPARTMENTS)),
        'Tendencia': np.round(np.random.normal(0.5, 1.5, len(DEPARTMENTS)), 2)
    })
    
    # LEAN Data
    lean = pd.DataFrame({
        'Departamento': DEPARTMENTS,
        'Eficiencia': np.random.randint(60, 95, len(DEPARTMENTS)),
        'Reducción Desperdicio': np.random.randint(5, 25, len(DEPARTMENTS)),
        'Proyectos Activos': np.random.randint(1, 6, len(DEPARTMENTS)),
        '5S_Score': np.random.randint(60, 100, len(DEPARTMENTS)),
        'SMED': np.random.randint(50, 90, len(DEPARTMENTS))
    })
    
    # Wellbeing Data
    dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
    bienestar = pd.DataFrame({
        'Mes': dates,
        'Índice Bienestar': np.round(np.linspace(70, 85, 12) + np.random.normal(0, 3, 12), 1),
        'Ausentismo': np.round(np.linspace(10, 7, 12) + np.random.normal(0, 1, 12), 1),
        'Rotación': np.round(np.linspace(15, 10, 12) + np.random.normal(0, 1.5, 12), 1),
        'Encuestas': np.random.randint(80, 100, 12)
    })
    
    # Action Plans Data
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

nom_df, lean_df, bienestar_df, action_plans_df = load_data()

# ========== SIDEBAR ==========
with st.sidebar:
    # Logo and title
    st.markdown(f"""
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <h1 style="color: white; margin-bottom: 0;">📊</h1>
        <h2 style="color: white; margin-left: 10px; margin-bottom: 0;">NOM-035 & LEAN</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Date range filter
    st.markdown("**🔍 Filtros de Periodo**")
    fecha_inicio, fecha_fin = st.columns(2)
    with fecha_inicio:
        start_date = st.date_input(
            "Inicio", 
            value=date(2024, 1, 1),
            min_value=date(2024, 1, 1),
            max_value=date(2024, 12, 31),
            key="date_start"
        )
    with fecha_fin:
        end_date = st.date_input(
            "Fin", 
            value=date(2024, 4, 1),
            min_value=date(2024, 1, 1),
            max_value=date(2024, 12, 31),
            key="date_end"
        )
    
    # Department filter
    st.markdown("**🏢 Departamentos**")
    departamentos_filtro = st.multiselect(
        "Seleccionar departamentos",
        options=DEPARTMENTS,
        default=['Producción', 'Calidad', 'Logística'],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # KPI targets configuration
    with st.expander("⚙️ Configuración de Metas", expanded=False):
        nom_target = st.slider("Meta NOM-035 (%)", 50, 100, 90)
        lean_target = st.slider("Meta LEAN (%)", 50, 100, 80)
        wellbeing_target = st.slider("Meta Bienestar (%)", 50, 100, 85)
        efficiency_target = st.slider("Meta Eficiencia (%)", 50, 100, 75)
    
    st.markdown("---")
    
    # Refresh button
    if st.button("🔄 Actualizar Datos", use_container_width=True, 
                help="Actualiza todos los datos y visualizaciones"):
        st.rerun()
    
    # Version info
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #bdc3c7; font-size: 0.8rem;">
        v2.1.0<br>
        © 2024 RH Analytics
    </div>
    """, unsafe_allow_html=True)

# ========== HEADER ==========
st.markdown(f"""
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
    <div>
        <h1 style="margin-bottom: 5px; color: {COLOR_PALETTE['primary']};">Sistema Integral NOM-035 & LEAN 2.0</h1>
        <p style="margin-top: 0; color: {COLOR_PALETTE['text']}; font-size: 1.1rem;">
            Monitoreo Estratégico de Bienestar Psicosocial y Eficiencia Operacional
        </p>
    </div>
    <div style="background-color: {COLOR_PALETTE['light']}; padding: 8px 15px; border-radius: 20px; 
                text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <div style="font-size: 0.9rem; color: {COLOR_PALETTE['primary']}; font-weight: 600;">
            {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}
        </div>
        <div style="font-size: 0.8rem; color: {COLOR_PALETTE['text']};">
            Actualizado: {datetime.now().strftime('%d/%m/%Y %H:%M')}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ========== KPI CARDS ==========
def kpi_card(value, title, target, icon="📊", help_text=None):
    """Create a professional KPI card with trend indicator"""
    delta = value - target
    percentage = min(100, (value / target * 100)) if target != 0 else 0
    
    if value >= target:
        status = "✅"
        color = COLOR_PALETTE['success']
        delta_text = f"+{delta}% sobre meta"
    elif value >= target - 10:
        status = "⚠️"
        color = COLOR_PALETTE['warning']
        delta_text = f"{delta}% bajo meta"
    else:
        status = "❌"
        color = COLOR_PALETTE['danger']
        delta_text = f"{delta}% bajo meta"
    
    card_html = f"""
    <div style='background-color: white; padding: 1rem; border-radius: 10px; 
                box-shadow: 0 4px 6px rgba(0,0,0,0.05); height: 100%;'>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
            <div style='font-size: 1rem; color: {COLOR_PALETTE['text']}; font-weight: 600;'>
                {icon} {title}
            </div>
            <div style='font-size: 0.9rem; color: {COLOR_PALETTE['text']};'>
                {status}
            </div>
        </div>
        <div style='font-size: 1.8rem; font-weight: 700; color: {color}; margin-bottom: 5px;'>
            {value}%
        </div>
        <div style='font-size: 0.85rem; color: {COLOR_PALETTE['text']}; margin-bottom: 8px;'>
            Meta: {target}% • {delta_text}
        </div>
        <div style='height: 6px; background: #f0f0f0; border-radius: 3px;'>
            <div style='width: {percentage}%; height: 6px; background: {color}; border-radius: 3px;'></div>
        </div>
    </div>
    """
    
    if help_text:
        # Create a container with tooltip
        container = st.container()
        with container:
            st.markdown(card_html, unsafe_allow_html=True)
        # Add tooltip to the container
        st.markdown(f"<span title='{help_text}'> </span>", unsafe_allow_html=True)
    else:
        st.markdown(card_html, unsafe_allow_html=True)

# Calculate KPIs from data
nom_compliance = nom_df['Evaluaciones'].mean()
lean_adoption = lean_df['Eficiencia'].mean()
wellbeing_index = bienestar_df['Índice Bienestar'].mean()
operational_efficiency = lean_df['Eficiencia'].mean()

# Display KPIs in columns
cols = st.columns(4)
with cols[0]: 
    kpi_card(
        round(nom_compliance), 
        "Cumplimiento NOM-035", 
        nom_target, 
        "📋",
        "Porcentaje de cumplimiento con la norma NOM-035"
    )
with cols[1]: 
    kpi_card(
        round(lean_adoption), 
        "Adopción LEAN", 
        lean_target, 
        "🔄",
        "Nivel de implementación de metodologías LEAN"
    )
with cols[2]: 
    kpi_card(
        round(wellbeing_index), 
        "Índice Bienestar", 
        wellbeing_target, 
        "😊",
        "Indicador general de bienestar organizacional"
    )
with cols[3]: 
    kpi_card(
        round(operational_efficiency), 
        "Eficiencia Operativa", 
        efficiency_target, 
        "⚙️",
        "Eficiencia general de los procesos operativos"
    )

# ========== MAIN CONTENT TABS ==========
tab1, tab2, tab3, tab4 = st.tabs(["📋 NOM-035", "🔄 LEAN 2.0", "😊 Bienestar", "📝 Planes de Acción"])

with tab1:
    st.markdown("#### Cumplimiento NOM-035 por Departamento")
    
    # Filter data
    filtered_nom = nom_df[nom_df['Departamento'].isin(departamentos_filtro)]
    
    # Validate filtered data
    if filtered_nom.empty:
        st.warning("No hay datos disponibles para los departamentos seleccionados")
    else:
        # Create tabs for different views
        nom_view1, nom_view2, nom_view3 = st.tabs(["Métricas Principales", "Mapa de Riesgo", "Análisis de Tendencia"])
        
        with nom_view1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Bar chart for evaluations and trainings
                fig = px.bar(
                    filtered_nom, 
                    x="Departamento", 
                    y=["Evaluaciones", "Capacitaciones"], 
                    barmode="group",
                    color_discrete_sequence=[COLOR_PALETTE['primary'], COLOR_PALETTE['secondary']],
                    labels={'value': 'Porcentaje', 'variable': 'Métrica'},
                    height=400
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    yaxis_range=[0, 100],
                    legend_title_text='Métrica',
                    xaxis_title="",
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**📌 Resumen de Indicadores**")
                st.dataframe(
                    filtered_nom.set_index('Departamento'),
                    use_container_width=True,
                    height=400
                )
        
        with nom_view2:
            # Risk heatmap
            fig_heat = go.Figure(data=go.Heatmap(
                z=filtered_nom[['Evaluaciones', 'Capacitaciones', 'Incidentes']].values.T,
                x=filtered_nom['Departamento'],
                y=['Evaluaciones', 'Capacitaciones', 'Incidentes'],
                colorscale='RdYlGn',
                reversescale=True,
                zmin=0,
                zmax=100,
                hoverongaps=False
            ))
            fig_heat.update_layout(
                title="Mapa de Riesgo Psicosocial",
                xaxis_title="Departamento",
                yaxis_title="Métrica",
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=50, r=50, t=50, b=50)
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        
        with nom_view3:
            # Trend analysis
            col1, col2 = st.columns([3, 1])
            
            with col1:
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
                    title="Tendencia de Cumplimiento (Último Mes)",
                    yaxis_title="Cambio en puntos porcentuales",
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            
            with col2:
                st.markdown("**📊 Interpretación**")
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
                    <p style="font-size: 0.9rem; margin-bottom: 10px;">
                        <span style="color: #27ae60; font-weight: bold;">↑ Positivo:</span> Mejora continua
                    </p>
                    <p style="font-size: 0.9rem; margin-bottom: 10px;">
                        <span style="color: #e74c3c; font-weight: bold;">↓ Negativo:</span> Requiere atención
                    </p>
                    <p style="font-size: 0.9rem;">
                        <span style="color: #f39c12; font-weight: bold;">→ Neutral:</span> Mantener monitoreo
                    </p>
                </div>
                """, unsafe_allow_html=True)

with tab2:
    st.markdown("#### Progreso LEAN 2.0")
    filtered_lean = lean_df[lean_df['Departamento'].isin(departamentos_filtro)]
    
    if filtered_lean.empty:
        st.warning("No hay datos disponibles para los departamentos seleccionados")
    else:
        # Create columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Efficiency bar chart
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
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_lean, use_container_width=True)
            
            # Waste reduction vs efficiency scatter plot
            fig_scatter = px.scatter(
                filtered_lean,
                x='Reducción Desperdicio',
                y='Eficiencia',
                size='Proyectos Activos',
                color='Departamento',
                hover_name='Departamento',
                labels={
                    'Reducción Desperdicio': 'Reducción de Desperdicio (%)',
                    'Eficiencia': 'Eficiencia (%)',
                    'Proyectos Activos': 'Proyectos Activos'
                },
                height=400
            )
            fig_scatter.update_layout(
                title="Relación Eficiencia vs Reducción de Desperdicio",
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Radar chart for multiple metrics
            st.markdown("**📊 Comparación de Métricas**")
            fig_radar = go.Figure()
            
            # Normalize data for radar chart
            scaler = MinMaxScaler()
            lean_radar = filtered_lean.copy()
            lean_radar[['Eficiencia', 'Reducción Desperdicio', '5S_Score', 'SMED']] = scaler.fit_transform(
                lean_radar[['Eficiencia', 'Reducción Desperdicio', '5S_Score', 'SMED']]
            )
            
            for dept in filtered_lean['Departamento']:
                row = lean_radar[lean_radar['Departamento'] == dept].iloc[0]
                fig_radar.add_trace(go.Scatterpolar(
                    r=[row['Eficiencia'], row['Reducción Desperdicio'], row['5S_Score'], row['SMED']],
                    theta=['Eficiencia', 'Reducción', '5S', 'SMED'],
                    fill='toself',
                    name=dept,
                    line=dict(width=2)
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1])
                ),
                height=400,
                showlegend=True,
                margin=dict(l=50, r=50, t=30, b=50)
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Projects summary
            with st.expander("📌 Detalle de Proyectos", expanded=True):
                st.dataframe(
                    filtered_lean[['Departamento', 'Proyectos Activos', '5S_Score', 'SMED']]
                    .set_index('Departamento')
                    .style.background_gradient(cmap='Greens'),
                    use_container_width=True
                )

with tab3:
    st.markdown("#### Tendencias de Bienestar Organizacional")
    
    # Filter wellbeing data by date range
    filtered_bienestar = bienestar_df[
        (bienestar_df['Mes'].dt.date >= start_date) & 
        (bienestar_df['Mes'].dt.date <= end_date)
    ]
    
    if filtered_bienestar.empty:
        st.warning("No hay datos disponibles para el período seleccionado")
    else:
        # Main metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="Encuestas Completadas", 
                value=f"{filtered_bienestar['Encuestas'].mean():.0f}%",
                delta=f"{filtered_bienestar['Encuestas'].iloc[-1] - filtered_bienestar['Encuestas'].iloc[0]:+.0f}%"
            )
        with col2:
            st.metric(
                label="Reducción de Ausentismo", 
                value=f"{filtered_bienestar['Ausentismo'].iloc[-1]:.1f}%",
                delta=f"{filtered_bienestar['Ausentismo'].iloc[-1] - filtered_bienestar['Ausentismo'].iloc[0]:+.1f}%"
            )
        with col3:
            st.metric(
                label="Reducción de Rotación", 
                value=f"{filtered_bienestar['Rotación'].iloc[-1]:.1f}%",
                delta=f"{filtered_bienestar['Rotación'].iloc[-1] - filtered_bienestar['Rotación'].iloc[0]:+.1f}%"
            )
        
        # Create tabs for different views
        wellbeing_view1, wellbeing_view2 = st.tabs(["Tendencias Mensuales", "Análisis de Correlación"])
        
        with wellbeing_view1:
            # Line chart for wellbeing metrics
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
                labels={'value': 'Porcentaje', 'variable': 'Métrica'},
                height=400
            )
            fig_bienestar.update_layout(
                title="Evolución Mensual de Bienestar",
                yaxis_range=[0, 100],
                plot_bgcolor='rgba(0,0,0,0)',
                legend_title="Métrica",
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_bienestar, use_container_width=True)
        
        with wellbeing_view2:
            # Correlation analysis
            corr_matrix = filtered_bienestar[['Índice Bienestar', 'Ausentismo', 'Rotación', 'Encuestas']].corr()
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                color_continuous_scale='RdYlGn',
                range_color=[-1, 1],
                labels=dict(x="Métrica", y="Métrica", color="Correlación"),
                height=400
            )
            fig_corr.update_layout(
                title="Matriz de Correlación",
                xaxis_title="",
                yaxis_title="",
                margin=dict(l=50, r=50, t=50, b=50)
            )
            st.plotly_chart(fig_corr, use_container_width=True)

with tab4:
    st.markdown("#### Planes de Acción y Seguimiento")
    
    # Display existing action plans
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("**📌 Planes Registrados**")
        st.dataframe(
            action_plans_df.style.applymap(
                lambda x: f"background-color: {COLOR_PALETTE['success']}; color: white" if x == 'Completado' 
                else f"background-color: {COLOR_PALETTE['warning']}" if x == 'En progreso' 
                else f"background-color: {COLOR_PALETTE['danger']}; color: white",
                subset=['Estado']
            ).bar(
                subset=['% Avance'], 
                color=COLOR_PALETTE['secondary'],
                vmin=0,
                vmax=100
            ),
            use_container_width=True,
            hide_index=True,
            height=400
        )
    
    with col2:
        st.markdown("**📊 Resumen por Estado**")
        status_summary = action_plans_df['Estado'].value_counts().reset_index()
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
        fig_status.update_layout(showlegend=False)
        st.plotly_chart(fig_status, use_container_width=True)
        
        st.markdown("**📅 Vencimientos Próximos**")
        upcoming = action_plans_df[action_plans_df['Plazo'] <= date.today() + pd.Timedelta(days=30)]
        if not upcoming.empty:
            for _, row in upcoming.iterrows():
                days_left = (row['Plazo'] - date.today()).days
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 10px; border-radius: 8px; margin-bottom: 10px;">
                    <div style="font-weight: 600;">{row['Departamento']}</div>
                    <div style="font-size: 0.8rem;">{row['Problema'][:30]}...</div>
                    <div style="font-size: 0.8rem; color: {'#e74c3c' if days_left < 7 else '#f39c12'}">
                        {f"Vence en {days_left} días" if days_left > 0 else f"Vencido hace {-days_left} días"}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No hay planes con vencimiento próximo", icon="ℹ️")
    
    # Add new action plan form
    with st.expander("➕ Registrar Nuevo Plan de Acción", expanded=False):
        with st.form("nuevo_plan_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            
            with col1:
                dept = st.selectbox("Departamento", DEPARTMENTS)
                problema = st.text_area("Problema identificado", max_chars=200)
                prioridad = st.selectbox("Prioridad", ["Alta", "Media", "Baja"])
            
            with col2:
                accion = st.text_area("Acción propuesta", max_chars=200)
                responsable = st.text_input("Responsable")
                plazo = st.date_input(
                    "Plazo estimado",
                    min_value=date.today(),
                    value=date.today() + pd.Timedelta(days=30)
                )
                avance = st.slider("% Avance", 0, 100, 0)
            
            submitted = st.form_submit_button("💾 Guardar Plan de Acción")
            
            if submitted:
                # In a real app, this would save to a database
                new_plan = pd.DataFrame([{
                    'ID': len(action_plans_df) + 1,
                    'Departamento': dept,
                    'Problema': problema,
                    'Acción': accion,
                    'Responsable': responsable,
                    'Plazo': plazo,
                    'Estado': 'Pendiente' if avance == 0 else 'En progreso' if avance < 100 else 'Completado',
                    'Prioridad': prioridad,
                    '% Avance': avance
                }])
                
                action_plans_df = pd.concat([action_plans_df, new_plan], ignore_index=True)
                st.success("✅ Plan de acción registrado correctamente")
                st.rerun()

# ========== EXPORT AND REPORTING ==========
st.markdown("---")
st.markdown("#### Exportación de Datos y Reportes")

export_col1, export_col2, export_col3 = st.columns(3)

with export_col1:
    with st.expander("📄 Generar Reporte PDF", expanded=False):
        report_type = st.selectbox("Tipo de reporte", 
                                 ["Completo", "Resumido", "Solo NOM-035", "Solo LEAN"],
                                 label_visibility="collapsed")
        if st.button("🖨️ Generar Reporte", use_container_width=True):
            st.toast(f"Generando reporte {report_type}...", icon="⏳")
            st.success("✅ Reporte generado exitosamente")

with export_col2:
    with st.expander("📧 Enviar por Correo", expanded=False):
        email = st.text_input("Dirección de correo", placeholder="usuario@empresa.com")
        if st.button("📤 Enviar Reporte", use_container_width=True):
            if email and "@" in email:
                st.toast(f"Enviando reporte a {email}", icon="✉️")
                st.success("✅ Reporte enviado correctamente")
            else:
                st.warning("Por favor ingrese una dirección de correo válida")

with export_col3:
    with st.expander("📊 Exportar Datos", expanded=False):
        export_format = st.radio("Formato", 
                                ["CSV", "Excel", "JSON"],
                                label_visibility="collapsed")
        if st.button("💾 Descargar Datos", use_container_width=True):
            st.toast(f"Preparando datos en formato {export_format}", icon="⏳")
            st.success(f"✅ Datos exportados como {export_format}")

# ========== FOOTER ==========
st.markdown("""
<hr>
<div style="text-align: center; color: #7f8c8d; font-size: 0.8rem; padding: 10px;">
    Sistema Integral NOM-035 & LEAN 2.0 • Versión 2.1.0<br>
    © 2024 Departamento de RH • Todos los derechos reservados
</div>
""", unsafe_allow_html=True)

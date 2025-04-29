import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ---- CONSTANTS AND CONFIGURATION ----
DEPARTMENTS = ['Producción', 'Calidad', 'Logística', 'Administración', 'Ventas', 'RH', 'TI']
COLOR_SCHEME = {
    'primary': '#2c3e50',
    'secondary': '#3498db',
    'success': '#27ae60',
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'light': '#ecf0f1',
    'dark': '#2c3e50'
}

# ---- GENERAL CONFIGURATION ----
st.set_page_config(
    page_title="Sistema Integral NOM-035 & LEAN 2.0",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# ---- DATA LOADING AND PROCESSING ----
@st.cache_data(ttl=600)
def load_data():
    """Load and generate synthetic data for the dashboard"""
    # Set random seed for reproducibility
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
        '5S_Score': np.random.randint(60, 100, len(DEPARTMENTS))
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
        'Prioridad': ['Alta', 'Media', 'Alta', 'Media', 'Baja']
    })
    
    return nom, lean, bienestar, action_plans

nom_df, lean_df, bienestar_df, action_plans_df = load_data()

# ---- SIDEBAR ----
with st.sidebar:
    st.title("📊 Filtros Avanzados")
    st.markdown("---")
    
    # Date range filter
    fecha_inicio = st.date_input(
        "Fecha de inicio", 
        value=date(2024, 1, 1),
        min_value=date(2024, 1, 1),
        max_value=date(2024, 12, 31)
    )
    
    fecha_fin = st.date_input(
        "Fecha de fin", 
        value=date(2024, 4, 1),
        min_value=date(2024, 1, 1),
        max_value=date(2024, 12, 31)
    )
    
    # Department filter
    departamentos_filtro = st.multiselect(
        "Seleccionar Departamentos",
        options=DEPARTMENTS,
        default=['Producción', 'Calidad', 'Logística']
    )
    
    # Metrics filter
    metricas = st.multiselect(
        "Seleccionar Métricas",
        ['NOM-035', 'LEAN', 'Bienestar', 'Acciones'],
        default=['NOM-035', 'LEAN']
    )
    
    st.markdown("---")
    
    # KPI targets configuration
    with st.expander("⚙️ Configurar Metas"):
        nom_target = st.slider("Meta NOM-035 (%)", 50, 100, 90)
        lean_target = st.slider("Meta LEAN (%)", 50, 100, 80)
        wellbeing_target = st.slider("Meta Bienestar (%)", 50, 100, 85)
        efficiency_target = st.slider("Meta Eficiencia (%)", 50, 100, 75)
    
    st.markdown("---")
    
    # Refresh button
    if st.button("🔄 Actualizar Datos", use_container_width=True):
        st.rerun()
    
    # Version info
    st.markdown("---")
    st.caption("v2.1.0 | © 2024 RH Analytics")

# ---- HEADER ----
st.markdown(f"""
    <div style='display: flex; align-items: center; justify-content: space-between;'>
        <div>
            <h1 style='margin-bottom: 0; color: {COLOR_SCHEME['primary']};'>📈 Sistema Integral NOM-035 & LEAN 2.0</h1>
            <p style='margin-top: 0; color: {COLOR_SCHEME['dark']};'>Monitoreo Estratégico de Bienestar Psicosocial y Eficiencia Operacional</p>
        </div>
        <div style='text-align: right; color: {COLOR_SCHEME['secondary']}; font-size:0.85rem;'>
            Período: {fecha_inicio.strftime('%d/%m/%Y')} - {fecha_fin.strftime('%d/%m/%Y')}<br>
            Última actualización: {datetime.now().strftime('%d/%m/%Y %H:%M')}
        </div>
    </div>
    <hr style='margin-top: 0.5rem;'>
""", unsafe_allow_html=True)

# ---- KPI CARDS ----
def kpi_card(value, title, target, icon="📊"):
    """Create a styled KPI card with trend indicator"""
    delta = value - target
    percentage = (value / target * 100) if target != 0 else 0
    
    if value >= target:
        status = "✅"
        color = COLOR_SCHEME['success']
        delta_text = f"+{delta}% sobre meta"
    elif value >= target - 10:
        status = "⚠️"
        color = COLOR_SCHEME['warning']
        delta_text = f"{delta}% bajo meta"
    else:
        status = "❌"
        color = COLOR_SCHEME['danger']
        delta_text = f"{delta}% bajo meta"
    
    st.markdown(f"""
        <div style='background-color: {color}; padding: 1rem; border-radius: 12px; 
                    color: white; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1);'>
            <div style='font-size: 1.5rem; margin-bottom: 0.5rem;'>{icon} {status} {title}</div>
            <div style='font-size: 2rem; font-weight: bold;'>{value}%</div>
            <div style='font-size: 0.9rem;'>Meta: {target}% • {delta_text}</div>
            <div style='height: 6px; background: rgba(255,255,255,0.3); margin-top: 8px; border-radius: 3px;'>
                <div style='width: {min(100, percentage)}%; height: 6px; background: white; border-radius: 3px;'></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Calculate KPIs from data
nom_compliance = nom_df['Evaluaciones'].mean()
lean_adoption = lean_df['Eficiencia'].mean()
wellbeing_index = bienestar_df['Índice Bienestar'].mean()
operational_efficiency = lean_df['Eficiencia'].mean()

# Display KPIs
cols = st.columns(4)
with cols[0]: kpi_card(round(nom_compliance), "Cumplimiento NOM-035", nom_target, "📋")
with cols[1]: kpi_card(round(lean_adoption), "Adopción LEAN", lean_target, "🔄")
with cols[2]: kpi_card(round(wellbeing_index), "Índice Bienestar", wellbeing_target, "😊")
with cols[3]: kpi_card(round(operational_efficiency), "Eficiencia Operativa", efficiency_target, "⚙️")

# ---- MAIN TABS ----
tab1, tab2, tab3, tab4 = st.tabs(["📋 NOM-035", "🔄 LEAN 2.0", "😊 Bienestar", "📝 Planes de Acción"])

with tab1:
    st.subheader("Cumplimiento NOM-035 por Departamento")
    
    # Filter data
    filtered_nom = nom_df[nom_df['Departamento'].isin(departamentos_filtro)]
    
    # Create tabs for different views
    nom_view1, nom_view2, nom_view3 = st.tabs(["📊 Métricas", "🔥 Mapa de Riesgo", "📈 Tendencia"])
    
    with nom_view1:
        # Bar chart for evaluations and trainings
        fig = px.bar(
            filtered_nom, 
            x="Departamento", 
            y=["Evaluaciones", "Capacitaciones"], 
            barmode="group",
            color_discrete_sequence=[COLOR_SCHEME['primary'], COLOR_SCHEME['secondary']],
            labels={'value': 'Porcentaje', 'variable': 'Métrica'}
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis_range=[0, 100]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with nom_view2:
        # Risk heatmap
        fig_heat = go.Figure(data=go.Heatmap(
            z=filtered_nom[['Evaluaciones', 'Capacitaciones', 'Incidentes']].values.T,
            x=filtered_nom['Departamento'],
            y=['Evaluaciones', 'Capacitaciones', 'Incidentes'],
            colorscale='RdYlGn',
            reversescale=True,
            zmin=0,
            zmax=100
        ))
        fig_heat.update_layout(
            title="Mapa de Riesgo Psicosocial",
            xaxis_title="Departamento",
            yaxis_title="Métrica",
            height=500
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    
    with nom_view3:
        # Trend analysis
        fig_trend = px.bar(
            filtered_nom, 
            x='Departamento', 
            y='Tendencia',
            color='Tendencia',
            color_continuous_scale='RdYlGn',
            range_color=[-3, 3],
            labels={'Tendencia': 'Cambio mensual (%)'}
        )
        fig_trend.update_layout(
            title="Tendencia de Cumplimiento (Último Mes)",
            yaxis_title="Cambio en puntos porcentuales",
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_trend, use_container_width=True)

with tab2:
    st.subheader("Progreso LEAN 2.0")
    filtered_lean = lean_df[lean_df['Departamento'].isin(departamentos_filtro)]
    
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
            labels={'Eficiencia': 'Eficiencia (%)'}
        )
        fig_lean.update_layout(
            title="Eficiencia por Departamento",
            yaxis_range=[0, 100],
            plot_bgcolor='rgba(0,0,0,0)'
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
            }
        )
        fig_scatter.update_layout(
            title="Relación Eficiencia vs Reducción de Desperdicio",
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Radar chart for multiple metrics
        fig_radar = go.Figure()
        
        # Normalize data for radar chart
        scaler = MinMaxScaler()
        lean_radar = filtered_lean.copy()
        lean_radar[['Eficiencia', 'Reducción Desperdicio', '5S_Score']] = scaler.fit_transform(
            lean_radar[['Eficiencia', 'Reducción Desperdicio', '5S_Score']]
        )
        
        for dept in filtered_lean['Departamento']:
            row = lean_radar[lean_radar['Departamento'] == dept].iloc[0]
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['Eficiencia'], row['Reducción Desperdicio'], row['5S_Score']],
                theta=['Eficiencia', 'Reducción Desperdicio', '5S Score'],
                fill='toself',
                name=dept
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            title="Comparación de Métricas LEAN",
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Projects summary
        st.markdown("**📌 Proyectos Activos por Departamento**")
        st.dataframe(
            filtered_lean[['Departamento', 'Proyectos Activos']].set_index('Departamento'),
            use_container_width=True
        )

with tab3:
    st.subheader("Tendencias de Bienestar Organizacional")
    
    # Filter wellbeing data by date range
    filtered_bienestar = bienestar_df[
        (bienestar_df['Mes'].dt.date >= fecha_inicio) & 
        (bienestar_df['Mes'].dt.date <= fecha_fin)
    ]
    
    # Create tabs for different views
    wellbeing_view1, wellbeing_view2 = st.tabs(["📈 Tendencias", "📊 Análisis"])
    
    with wellbeing_view1:
        # Line chart for wellbeing metrics
        fig_bienestar = px.line(
            filtered_bienestar, 
            x='Mes', 
            y=['Índice Bienestar', 'Ausentismo', 'Rotación'], 
            markers=True,
            color_discrete_sequence=[
                COLOR_SCHEME['success'], 
                COLOR_SCHEME['danger'], 
                COLOR_SCHEME['warning']
            ],
            labels={'value': 'Porcentaje', 'variable': 'Métrica'}
        )
        fig_bienestar.update_layout(
            title="Evolución Mensual de Bienestar",
            yaxis_range=[0, 100],
            plot_bgcolor='rgba(0,0,0,0)',
            legend_title="Métrica"
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
            labels=dict(x="Métrica", y="Métrica", color="Correlación")
        )
        fig_corr.update_layout(
            title="Matriz de Correlación",
            xaxis_title="",
            yaxis_title=""
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Trend analysis
        X = np.arange(len(filtered_bienestar)).reshape(-1, 1)
        y = filtered_bienestar['Índice Bienestar'].values
        modelo = LinearRegression().fit(X, y)
        tendencia = modelo.coef_[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Tendencia de Bienestar",
                value=f"{'↑' if tendencia > 0 else '↓'} {abs(tendencia):.2f} pts/mes",
                delta=f"{'Mejorando' if tendencia > 0 else 'Empeorando'}",
                delta_color="normal"
            )
        with col2:
            st.metric(
                label="Encuestas Completadas",
                value=f"{filtered_bienestar['Encuestas'].mean():.0f}%",
                delta=f"{(filtered_bienestar['Encuestas'].iloc[-1] - filtered_bienestar['Encuestas'].iloc[0]):+.0f}% vs inicio",
                delta_color="normal"
            )

with tab4:
    st.subheader("Planes de Acción y Seguimiento")
    
    # Display existing action plans
    st.markdown("### 📌 Planes Registrados")
    st.dataframe(
        action_plans_df.style.applymap(
            lambda x: f"background-color: {COLOR_SCHEME['success']}" if x == 'Completado' 
            else f"background-color: {COLOR_SCHEME['warning']}" if x == 'En progreso' 
            else f"background-color: {COLOR_SCHEME['danger']}",
            subset=['Estado']
        ),
        use_container_width=True,
        hide_index=True
    )
    
    # Add new action plan form
    with st.expander("➕ Registrar Nuevo Plan de Acción", expanded=False):
        with st.form("nuevo_plan_form"):
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
                    'Estado': 'Pendiente',
                    'Prioridad': prioridad
                }])
                
                action_plans_df = pd.concat([action_plans_df, new_plan], ignore_index=True)
                st.success("✅ Plan de acción registrado correctamente")
                st.rerun()

# ---- EXPORT AND REPORTING ----
st.markdown("---")
st.subheader("📤 Exportación de Datos y Reportes")

export_col1, export_col2, export_col3 = st.columns(3)

with export_col1:
    with st.expander("📄 Generar Reporte PDF"):
        st.selectbox("Formato del reporte", ["Completo", "Resumido", "Solo NOM-035", "Solo LEAN"])
        if st.button("🖨️ Generar Reporte"):
            st.toast("Reporte generado exitosamente", icon="✅")

with export_col2:
    with st.expander("📧 Enviar por Correo"):
        email = st.text_input("Dirección de correo")
        if st.button("📤 Enviar Reporte"):
            if email:
                st.toast(f"Reporte enviado a {email}", icon="✉️")
            else:
                st.warning("Por favor ingrese una dirección de correo")

with export_col3:
    with st.expander("📊 Exportar Datos"):
        format = st.radio("Formato", ["CSV", "Excel", "JSON"])
        if st.button("💾 Descargar Datos"):
            st.toast(f"Preparando datos en formato {format}", icon="⏳")

# ---- FOOTER ----
st.markdown("""
    <hr>
    <div style='text-align: center; color: gray; font-size: 0.85rem;'>
        Sistema Integral NOM-035 & LEAN 2.0 • Versión 2.1.0<br>
        © 2024 Departamento de RH • Todos los derechos reservados
    </div>
""", unsafe_allow_html=True)

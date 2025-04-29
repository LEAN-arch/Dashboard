import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Configuración de página
st.set_page_config(
    page_title="Sistema Integral NOM-035 & LEAN 2.0",
    layout="wide",
    page_icon="📊"
)

# --- ESTILOS Y DISEÑO ---
st.markdown("""
<style>
    /* Estilos generales */
    .main { background-color: #f9f9f9; }
    .header { border-bottom: 1px solid #e1e1e1; padding-bottom: 1rem; }
    .kpi-card { 
        border-radius: 10px; 
        padding: 1.5rem; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s;
    }
    .kpi-card:hover { transform: translateY(-5px); }
    .kpi-good { background: linear-gradient(135deg, #4CAF50 0%, #81C784 100%); color: white; }
    .kpi-warning { background: linear-gradient(135deg, #FFC107 0%, #FFD54F 100%); color: #333; }
    .kpi-danger { background: linear-gradient(135deg, #F44336 0%, #E57373 100%); color: white; }
    .alert-critical { border-left: 5px solid #F44336; }
    .alert-warning { border-left: 5px solid #FFC107; }
    .stDataFrame { border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)

# --- CABECERA ---
col1, col2 = st.columns([1, 5])
with col1:
    st.image("logo_empresa.png", width=120)
with col2:
    st.markdown("<h1 style='margin-bottom:0; color: #2d3e50;'>📈 Sistema Integral NOM-035 & LEAN 2.0</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #4a6fa5; margin-top:0;'>Monitoreo Estratégico de Bienestar Psicosocial y Eficiencia Operacional</p>", unsafe_allow_html=True)

st.markdown(f"<div style='text-align: right; color: #666; font-size:0.8rem;'>Última actualización: {datetime.now().strftime('%d/%m/%Y %H:%M')}</div>", unsafe_allow_html=True)
st.markdown("---")

# --- FILTROS ---
with st.sidebar:
    st.markdown("### 🔍 Filtros Avanzados")
    
    with st.expander("📅 Rango Temporal", expanded=True):
        fecha_inicio = st.date_input("Fecha inicio", value=datetime(2025, 1, 1))
        fecha_fin = st.date_input("Fecha fin", value=datetime(2025, 4, 1))
    
    with st.expander("🏢 Unidades Organizacionales", expanded=True):
        departamentos = st.multiselect(
            "Seleccionar departamentos",
            ['Producción', 'Calidad', 'Logística', 'Administración', 'Ventas', 'RH', 'TI'],
            default=['Producción', 'Calidad', 'Logística']
        )
    
    with st.expander("📊 Métricas Clave"):
        metricas = st.multiselect(
            "Indicadores a visualizar",
            ['NOM-035', 'Calidad', 'Productividad', 'Bienestar', 'LEAN'],
            default=['NOM-035', 'Calidad']
        )
    
    st.markdown("---")
    st.markdown("### 🛠 Acciones Rápidas")
    if st.button("🔄 Actualizar Datos"):
        st.experimental_rerun()
    if st.button("📤 Exportar Configuración"):
        st.success("Configuración exportada")

# --- DATOS SIMULADOS ---
@st.cache_data
def cargar_datos():
    # Datos de cumplimiento NOM-035
    nom_data = pd.DataFrame({
        'Departamento': ['Producción', 'Calidad', 'Logística', 'Administración', 'Ventas', 'RH', 'TI'],
        'Evaluaciones': [92, 95, 70, 88, 85, 97, 90],
        'Capacitaciones': [85, 90, 65, 82, 78, 95, 88],
        'Incidentes': [3, 1, 8, 2, 4, 0, 1],
        'Tendencia': [1.2, 0.8, -2.5, 0.5, -1.2, 0.3, 0.7]
    })
    
    # Datos de LEAN 2.0
    lean_data = pd.DataFrame({
        'Departamento': ['Producción', 'Calidad', 'Logística', 'Administración', 'Ventas', 'RH', 'TI'],
        'Eficiencia': [82, 88, 65, 75, 78, 85, 90],
        'Reducción Desperdicio': [15, 20, 5, 12, 8, 18, 25],
        'Proyectos Activos': [3, 2, 1, 4, 2, 3, 5]
    })
    
    # Datos de bienestar
    bienestar_data = pd.DataFrame({
        'Mes': pd.date_range(start='2024-01-01', periods=12, freq='M'),
        'Índice Bienestar': np.round(np.random.normal(75, 5, 12), 1),
        'Ausentismo': np.round(np.random.normal(8, 2, 12), 1),
        'Rotación': np.round(np.random.normal(12, 3, 12), 1)
    })
    
    return nom_data, lean_data, bienestar_data

nom_data, lean_data, bienestar_data = cargar_datos()

# --- KPIs PRINCIPALES ---
st.markdown("## 📊 Tablero de Control Integral")

def crear_kpi(valor, titulo, meta=90, icono="📊"):
    delta = valor - meta
    status = "good" if valor >= meta else "warning" if valor >= meta-15 else "danger"
    
    card = f"""
    <div class="kpi-card kpi-{status}">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <h3 style="margin:0;">{icono} {titulo}</h3>
            <span style="font-size:1.2rem; font-weight:bold; {'color:white;' if status in ['good','danger'] else ''}">
                {valor}% {'(+'+str(delta)+')' if delta >=0 else '('+str(delta)+')'}
            </span>
        </div>
        <div style="margin-top:10px; height:6px; background:rgba(255,255,255,0.3); border-radius:3px;">
            <div style="width:{valor}%; height:6px; background:white; border-radius:3px;"></div>
        </div>
        <p style="margin:5px 0 0 0; font-size:0.8rem;">Meta: {meta}%</p>
    </div>
    """
    return card

cols = st.columns(4)
with cols[0]:
    st.markdown(crear_kpi(92, "Cumplimiento NOM-035", 90, "🧠"), unsafe_allow_html=True)
with cols[1]:
    st.markdown(crear_kpi(85, "Adopción LEAN 2.0", 80, "🔄"), unsafe_allow_html=True)
with cols[2]:
    st.markdown(crear_kpi(78, "Índice Bienestar", 85, "❤️"), unsafe_allow_html=True)
with cols[3]:
    st.markdown(crear_kpi(65, "Eficiencia Operativa", 75, "⚙️"), unsafe_allow_html=True)

# --- VISUALIZACIONES ---
tab1, tab2, tab3 = st.tabs(["NOM-035", "LEAN 2.0", "Bienestar"])

with tab1:
    st.markdown("### 🧠 Cumplimiento NOM-035 por Departamento")
    
    fig_nom = px.bar(
        nom_data[nom_data['Departamento'].isin(departamentos)],
        x='Departamento', y=['Evaluaciones', 'Capacitaciones'],
        barmode='group',
        color_discrete_map={
            'Evaluaciones': '#4a6fa5',
            'Capacitaciones': '#FFC107'
        },
        height=400
    )
    fig_nom.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title="",
        yaxis_title="Porcentaje de Cumplimiento",
        legend_title=""
    )
    st.plotly_chart(fig_nom, use_container_width=True)
    
    # Heatmap de riesgo psicosocial
    st.markdown("### 🗺 Mapa de Riesgo Psicosocial")
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=nom_data[nom_data['Departamento'].isin(departamentos)][['Evaluaciones', 'Capacitaciones', 'Incidentes']].values.T,
        x=nom_data[nom_data['Departamento'].isin(departamentos)]['Departamento'],
        y=['Evaluaciones', 'Capacitaciones', 'Incidentes'],
        colorscale='RdYlGn',
        reversescale=True
    ))
    fig_heatmap.update_layout(height=300)
    st.plotly_chart(fig_heatmap, use_container_width=True)

with tab2:
    st.markdown("### 🔄 Progreso LEAN 2.0")
    
    col1, col2 = st.columns(2)
    with col1:
        fig_lean = px.bar(
            lean_data[lean_data['Departamento'].isin(departamentos)],
            x='Departamento', y='Eficiencia',
            color='Eficiencia',
            color_continuous_scale='Greens',
            height=400
        )
        st.plotly_chart(fig_lean, use_container_width=True)
    
    with col2:
        fig_radar = go.Figure()
        for dept in departamentos:
            data = lean_data[lean_data['Departamento'] == dept].iloc[0]
            fig_radar.add_trace(go.Scatterpolar(
                r=[data['Eficiencia'], data['Reducción Desperdicio'], data['Proyectos Activos']],
                theta=['Eficiencia', 'Red. Desperdicio', 'Proy. Activos'],
                fill='toself',
                name=dept
            ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=400)
        st.plotly_chart(fig_radar, use_container_width=True)

with tab3:
    st.markdown("### ❤️ Tendencias de Bienestar Organizacional")
    
    fig_bienestar = px.line(
        bienestar_data,
        x='Mes', y=['Índice Bienestar', 'Ausentismo', 'Rotación'],
        markers=True,
        color_discrete_map={
            'Índice Bienestar': '#4CAF50',
            'Ausentismo': '#F44336',
            'Rotación': '#FFC107'
        },
        height=400
    )
    fig_bienestar.update_layout(
        yaxis_title="Porcentaje",
        xaxis_title="",
        legend_title="Indicador"
    )
    st.plotly_chart(fig_bienestar, use_container_width=True)

# --- ALERTAS Y ACCIONES ---
st.markdown("## 🚨 Alertas y Planes de Acción")

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 🔴 Áreas Críticas")
    st.dataframe(
        nom_data[nom_data['Evaluaciones'] < 80][['Departamento', 'Evaluaciones', 'Incidentes']]
        .sort_values('Evaluaciones')
        .style.background_gradient(cmap='Reds'),
        use_container_width=True
    )

with col2:
    st.markdown("### 🟡 Oportunidades de Mejora")
    st.dataframe(
        lean_data[lean_data['Eficiencia'] < 75][['Departamento', 'Eficiencia', 'Reducción Desperdicio']]
        .sort_values('Eficiencia')
        .style.background_gradient(cmap='Oranges'),
        use_container_width=True
    )

# --- PLAN DE ACCIÓN ---
with st.expander("📝 Generar Plan de Acción", expanded=False):
    departamento_accion = st.selectbox("Departamento", nom_data['Departamento'].unique())
    problema = st.text_input("Problema identificado")
    accion = st.text_area("Acción propuesta")
    responsable = st.text_input("Responsable")
    plazo = st.date_input("Plazo de implementación")
    
    if st.button("💾 Guardar Plan de Acción"):
        st.success(f"Plan de acción para {departamento_accion} guardado exitosamente")

# --- EXPORTACIÓN ---
st.markdown("---")
st.markdown("## 📤 Exportación de Reportes")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📄 Generar Reporte Ejecutivo"):
        st.success("Reporte generado (simulación)")
with col2:
    if st.button("📊 Exportar Datos a Excel"):
        st.success("Datos exportados (simulación)")
with col3:
    if st.button("📧 Enviar a Gerencia"):
        st.success("Envío programado (simulación)")

# --- PIE DE PÁGINA ---
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#666; font-size:0.9rem; padding:1rem;">
    <p>© 2025 Sistema de Gestión Integral - Cumplimiento NOM-035 & LEAN 2.0</p>
    <p>📞 Soporte técnico: (664) 123-4567 | 📧 contacto@empresa.com.mx</p>
</div>
""", unsafe_allow_html=True)

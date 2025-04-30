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

# ========== INITIALIZATION ==========
# Page config must be first Streamlit command
st.set_page_config(
    page_title="NOM-035 & LEAN Dashboard",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# ========== DESIGN SYSTEM ==========
class DesignSystem:
    # Color palette
    COLORS = {
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

# Initialize design system
ds = DesignSystem()

# Load custom font and styles
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* {{
    font-family: {ds.FONT};
}}

/* Main layout */
.main {{
    background-color: {ds.COLORS['background']};
    padding: {ds.SPACING['md']};
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background-color: {ds.COLORS['dark']} !important;
    color: white !important;
}}

/* Cards */
.card {{
    background-color: white;
    border-radius: {ds.RADIUS['md']};
    padding: {ds.SPACING['md']};
    box-shadow: {ds.SHADOWS['sm']};
    transition: all 0.2s ease;
}}

.card:hover {{
    box-shadow: {ds.SHADOWS['md']};
    transform: translateY(-2px);
}}

/* Interactive elements */
button {{
    transition: all 0.2s ease !important;
}}

button:hover {{
    transform: translateY(-1px);
}}

/* Data tables */
.stDataFrame {{
    border-radius: {ds.RADIUS['sm']};
    box-shadow: {ds.SHADOWS['sm']};
}}

/* Tabs */
[data-baseweb="tab-list"] {{
    gap: {ds.SPACING['sm']};
}}

[data-baseweb="tab"] {{
    border-radius: {ds.RADIUS['full']} !important;
    padding: {ds.SPACING['sm']} {ds.SPACING['md']} !important;
    transition: all 0.2s ease !important;
}}

[aria-selected="true"] {{
    background-color: {ds.COLORS['primary']} !important;
    color: white !important;
    font-weight: 600 !important;
}}
</style>
""", unsafe_allow_html=True)

# ========== DATA MODEL ==========
class DataService:
    DEPARTMENTS = ['Producción', 'Calidad', 'Logística', 'Administración', 'Ventas', 'RH', 'TI']
    
    @staticmethod
    @st.cache_data(ttl=600, show_spinner="Cargando datos...")
    def load_data():
        """Generate synthetic data with realistic patterns"""
        np.random.seed(42)
        
        # Helper function for trended data
        def generate_trend(base, trend, noise=0.1):
            return base * (1 + trend) + np.random.normal(0, noise * base)
        
        # NOM-035 Data with department-specific patterns
        nom_data = {
            'Producción': {'base': 85, 'trend': 0.02, 'volatility': 0.08},
            'Calidad': {'base': 90, 'trend': 0.01, 'volatility': 0.05},
            'Logística': {'base': 80, 'trend': 0.03, 'volatility': 0.1},
            'Administración': {'base': 88, 'trend': 0.015, 'volatility': 0.06},
            'Ventas': {'base': 82, 'trend': 0.025, 'volatility': 0.09},
            'RH': {'base': 92, 'trend': 0.005, 'volatility': 0.04},
            'TI': {'base': 87, 'trend': 0.02, 'volatility': 0.07}
        }
        
        nom = pd.DataFrame({
            'Departamento': DataService.DEPARTMENTS,
            'Evaluaciones': [generate_trend(nom_data[d]['base'], nom_data[d]['trend'], nom_data[d]['volatility']) for d in DataService.DEPARTMENTS],
            'Capacitaciones': np.random.randint(60, 100, len(DataService.DEPARTMENTS)),
            'Incidentes': [max(0, int(30 - x*0.25 + np.random.normal(0, 3))) for x in range(len(DataService.DEPARTMENTS))],
            'Tendencia': [nom_data[d]['trend'] * 100 + np.random.normal(0, 0.5) for d in DataService.DEPARTMENTS]
        })
        
        # LEAN Data with realistic correlations
        lean = pd.DataFrame({
            'Departamento': DataService.DEPARTMENTS,
            'Eficiencia': [x + np.random.randint(-5, 5) for x in np.linspace(65, 90, len(DataService.DEPARTMENTS))],
            'Reducción Desperdicio': np.random.randint(5, 25, len(DataService.DEPARTMENTS)),
            'Proyectos Activos': np.random.randint(1, 6, len(DataService.DEPARTMENTS)),
            '5S_Score': [x + np.random.randint(-10, 10) for x in np.linspace(70, 90, len(DataService.DEPARTMENTS))],
            'SMED': np.random.randint(50, 90, len(DataService.DEPARTMENTS))
        })
        
        # Time-series wellbeing data
        dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
        bienestar = pd.DataFrame({
            'Mes': dates,
            'Índice Bienestar': np.round(np.linspace(70, 85, 12) + np.random.normal(0, 3, 12), 1),
            'Ausentismo': np.round(np.linspace(10, 7, 12) + np.random.normal(0, 1, 12), 1),
            'Rotación': np.round(np.linspace(15, 10, 12) + np.random.normal(0, 1.5, 12), 1),
            'Encuestas': np.random.randint(80, 100, 12)
        })
        
        # Action plans with realistic data
        action_plans = pd.DataFrame({
            'ID': range(1, 6),
            'Departamento': np.random.choice(DataService.DEPARTMENTS, 5),
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
            'Plazo': [date.today() + timedelta(days=np.random.randint(10, 90)) for _ in range(5)],
            'Estado': ['En progreso', 'Pendiente', 'Completado', 'En progreso', 'Pendiente'],
            'Prioridad': ['Alta', 'Media', 'Alta', 'Media', 'Baja'],
            '% Avance': [65, 0, 100, 30, 0]
        })
        
        return nom, lean, bienestar, action_plans

# ========== COMPONENTS ==========
class KPICard:
    @staticmethod
    def render(value, title, target, icon="📊", help_text=None):
        """Create an interactive KPI card with visual indicators"""
        delta = value - target
        percentage = min(100, (value / target * 100)) if target != 0 else 0
        
        if value >= target:
            status = "✅"
            color = ds.COLORS['success']
            delta_text = f"+{delta}% sobre meta"
            icon_bg = "rgba(16, 185, 129, 0.1)"
        elif value >= target - 10:
            status = "⚠️"
            color = ds.COLORS['warning']
            delta_text = f"{delta}% bajo meta"
            icon_bg = "rgba(245, 158, 11, 0.1)"
        else:
            status = "❌"
            color = ds.COLORS['danger']
            delta_text = f"{delta}% bajo meta"
            icon_bg = "rgba(239, 68, 68, 0.1)"
        
        with st.container():
            st.markdown(f"""
            <div class="card" style="border-left: 4px solid {color};">
                <div style="display: flex; align-items: center; gap: {ds.SPACING['sm']}; margin-bottom: {ds.SPACING['sm']};">
                    <div style="background: {icon_bg}; width: 36px; height: 36px; border-radius: {ds.RADIUS['full']}; 
                                display: flex; align-items: center; justify-content: center; color: {color};">
                        {icon}
                    </div>
                    <div style="font-weight: 600; color: {ds.COLORS['text']}; flex-grow: 1;">
                        {title}
                    </div>
                    <div style="font-size: 0.875rem; color: {color};">
                        {status}
                    </div>
                </div>
                <div style="font-size: 1.75rem; font-weight: 700; color: {color}; margin-bottom: {ds.SPACING['xs']};">
                    {value}%
                </div>
                <div style="font-size: 0.875rem; color: {ds.COLORS['text']}; margin-bottom: {ds.SPACING['sm']};">
                    Meta: {target}% • {delta_text}
                </div>
                <div style="height: 6px; background: #f0f0f0; border-radius: {ds.RADIUS['full']}; overflow: hidden;">
                    <div style="width: {percentage}%; height: 100%; background: {color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if help_text:
                st.markdown(f"<span title='{help_text}' style='font-size: 0;'>?</span>", unsafe_allow_html=True)

class DataVisualizer:
    @staticmethod
    def create_bar_chart(data, x, y, title, color=None, barmode='group'):
        """Create a responsive bar chart with consistent styling"""
        fig = px.bar(
            data,
            x=x,
            y=y,
            color=color,
            barmode=barmode,
            title=title,
            template='plotly_white',
            color_discrete_sequence=[ds.COLORS['primary'], ds.COLORS['secondary']]
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=60, b=20),
            hovermode='x unified',
            xaxis_title="",
            yaxis_title="",
            legend_title="",
            font=dict(family=ds.FONT)
        )
        
        return fig
    
    @staticmethod
    def create_line_chart(data, x, y, title):
        """Create a responsive line chart with consistent styling"""
        fig = px.line(
            data,
            x=x,
            y=y,
            title=title,
            template='plotly_white',
            color_discrete_sequence=[ds.COLORS['primary'], ds.COLORS['success'], ds.COLORS['danger']]
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=60, b=20),
            hovermode='x unified',
            xaxis_title="",
            yaxis_title="",
            legend_title="",
            font=dict(family=ds.FONT)
        )
        
        return fig
    
    @staticmethod
    def create_radar_chart(data, categories, values, title):
        """Create a responsive radar chart with consistent styling"""
        fig = go.Figure()
        
        for i, row in data.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=row[values].tolist(),
                theta=categories,
                fill='toself',
                name=row['Departamento'],
                line=dict(color=ds.COLORS['primary'], width=2)
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1]),
                bgcolor='rgba(0,0,0,0)'
            ),
            title=title,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family=ds.FONT),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.1,
                xanchor="center",
                x=0.5
            )
        )
        
        return fig

# ========== APPLICATION ==========
class NOMLEANDashboard:
    def __init__(self):
        self.nom_df, self.lean_df, self.bienestar_df, self.action_plans_df = DataService.load_data()
        self.start_date = date(2024, 1, 1)
        self.end_date = date(2024, 4, 1)
        self.departments_filter = ['Producción', 'Calidad', 'Logística']
        self.nom_target = 90
        self.lean_target = 80
        self.wellbeing_target = 85
        self.efficiency_target = 75
        
        self.initialize_sidebar()
        self.render_header()
        self.render_kpis()
        self.render_main_content()
    
    def initialize_sidebar(self):
        """Configure the sidebar filters and controls"""
        with st.sidebar:
            # Branding
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: {ds.SPACING['lg']};">
                <svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M16 32C24.8366 32 32 24.8366 32 16C32 7.16344 24.8366 0 16 0C7.16344 0 0 7.16344 0 16C0 24.8366 7.16344 32 16 32Z" fill="{ds.COLORS['primary']}"/>
                    <path d="M12 22L18 16L12 10" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <h2 style="color: white; margin-left: {ds.SPACING['sm']}; margin-bottom: 0;">NOM-035 & LEAN</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Date range filter
            st.markdown(f"**🔍 Filtros de Periodo**", help="Seleccione el rango de fechas para analizar")
            fecha_inicio, fecha_fin = st.columns(2)
            with fecha_inicio:
                self.start_date = st.date_input(
                    "Inicio",
                    value=self.start_date,
                    min_value=date(2024, 1, 1),
                    max_value=date(2024, 12, 31),
                    key="date_start"
                )
            with fecha_fin:
                self.end_date = st.date_input(
                    "Fin",
                    value=self.end_date,
                    min_value=date(2024, 1, 1),
                    max_value=date(2024, 12, 31),
                    key="date_end"
                )
            
            # Department filter
            st.markdown(f"**🏢 Departamentos**", help="Seleccione los departamentos a visualizar")
            self.departments_filter = st.multiselect(
                "Seleccionar departamentos",
                options=DataService.DEPARTMENTS,
                default=self.departments_filter,
                label_visibility="collapsed"
            )
            
            st.markdown("---")
            
            # KPI targets configuration
            with st.expander("⚙️ Configuración de Metas", expanded=False):
                self.nom_target = st.slider("Meta NOM-035 (%)", 50, 100, self.nom_target,
                                           help="Establezca la meta de cumplimiento para NOM-035")
                self.lean_target = st.slider("Meta LEAN (%)", 50, 100, self.lean_target,
                                           help="Establezca la meta de adopción para metodologías LEAN")
                self.wellbeing_target = st.slider("Meta Bienestar (%)", 50, 100, self.wellbeing_target,
                                                help="Establezca la meta para el índice de bienestar")
                self.efficiency_target = st.slider("Meta Eficiencia (%)", 50, 100, self.efficiency_target,
                                                 help="Establezca la meta para eficiencia operativa")
            
            st.markdown("---")
            
            # Refresh button
            if st.button("🔄 Actualizar Datos", use_container_width=True,
                        help="Actualiza todos los datos y visualizaciones con los filtros actuales"):
                st.rerun()
            
            # Version info
            st.markdown("---")
            st.markdown(f"""
            <div style="text-align: center; color: #9ca3af; font-size: 0.75rem;">
                v2.2.0<br>
                © {datetime.now().year} RH Analytics
            </div>
            """, unsafe_allow_html=True)
    
    def render_header(self):
        """Render the main header section"""
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: {ds.SPACING['xl']};">
            <div>
                <h1 style="margin-bottom: {ds.SPACING['xs']}; color: {ds.COLORS['dark']};">Sistema Integral NOM-035 & LEAN</h1>
                <p style="margin-top: 0; color: {ds.COLORS['text']}; font-size: 1rem;">
                    Monitoreo Estratégico de Bienestar Psicosocial y Eficiencia Operacional
                </p>
            </div>
            <div style="background-color: {ds.COLORS['light']}; padding: {ds.SPACING['sm']} {ds.SPACING['md']}; 
                        border-radius: {ds.RADIUS['md']}; text-align: center; box-shadow: {ds.SHADOWS['sm']};">
                <div style="font-size: 0.875rem; color: {ds.COLORS['primary']}; font-weight: 600;">
                    {self.start_date.strftime('%d/%m/%Y')} - {self.end_date.strftime('%d/%m/%Y')}
                </div>
                <div style="font-size: 0.75rem; color: {ds.COLORS['text']};">
                    Actualizado: {datetime.now().strftime('%d/%m/%Y %H:%M')}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_kpis(self):
        """Render the KPI cards section"""
        # Calculate KPIs from data
        nom_compliance = self.nom_df['Evaluaciones'].mean()
        lean_adoption = self.lean_df['Eficiencia'].mean()
        wellbeing_index = self.bienestar_df['Índice Bienestar'].mean()
        operational_efficiency = self.lean_df['Eficiencia'].mean()
        
        # Display KPIs in columns
        cols = st.columns(4)
        with cols[0]: 
            KPICard.render(
                round(nom_compliance), 
                "Cumplimiento NOM-035", 
                self.nom_target, 
                "📋",
                "Porcentaje de cumplimiento con la norma NOM-035"
            )
        with cols[1]: 
            KPICard.render(
                round(lean_adoption), 
                "Adopción LEAN", 
                self.lean_target, 
                "🔄",
                "Nivel de implementación de metodologías LEAN"
            )
        with cols[2]: 
            KPICard.render(
                round(wellbeing_index), 
                "Índice Bienestar", 
                self.wellbeing_target, 
                "😊",
                "Indicador general de bienestar organizacional"
            )
        with cols[3]: 
            KPICard.render(
                round(operational_efficiency), 
                "Eficiencia Operativa", 
                self.efficiency_target, 
                "⚙️",
                "Eficiencia general de los procesos operativos"
            )
    
    def render_main_content(self):
        """Render the main content tabs"""
        tab1, tab2, tab3, tab4 = st.tabs(["📋 NOM-035", "🔄 LEAN 2.0", "😊 Bienestar", "📝 Planes de Acción"])
        
        with tab1:
            self.render_nom035_tab()
        
        with tab2:
            self.render_lean_tab()
        
        with tab3:
            self.render_wellbeing_tab()
        
        with tab4:
            self.render_action_plans_tab()
    
    def render_nom035_tab(self):
        """Render the NOM-035 compliance tab"""
        st.markdown("#### Cumplimiento NOM-035 por Departamento")
        
        # Filter data
        filtered_nom = self.nom_df[self.nom_df['Departamento'].isin(self.departments_filter)]
        
        # Validate filtered data
        if filtered_nom.empty:
            st.warning("No hay datos disponibles para los departamentos seleccionados")
            return

# ========== MAIN CONTENT TABS ==========
tab1, tab2, tab3, tab4 = st.tabs(["📋 NOM-035", "🔄 LEAN 2.0", "😊 Bienestar", "📝 Planes de Acción"])

with tab1:
    st.markdown("#### Cumplimiento NOM-035 por Departamento")
    
    # Filter data
 
    
   
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

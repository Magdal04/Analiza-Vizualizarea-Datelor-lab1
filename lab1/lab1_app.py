# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import tempfile
import os

# Professional configuration
st.set_page_config(
    page_title="Energy Data Analytics Dashboard", 
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# Enhanced custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .section-header {
        font-size: 1.6rem;
        color: #2e86ab;
        border-bottom: 3px solid #2e86ab;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(uploaded_file=None):
    """Load and process energy data"""
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, parse_dates=["date"])
            st.success(f"✓ File uploaded successfully: {len(df)} records")
        else:
            df = pd.read_csv("lab1/raw_energy_data.csv", parse_dates=["date"])
            st.info("✓ Using default dataset")
        
        # Enhanced feature engineering
        df['an'] = df["date"].dt.year
        df['luna'] = df["date"].dt.month
        df['ziua'] = df['date'].dt.day
        df['ora'] = df["date"].dt.hour
        df['ziua_saptamanii'] = df["date"].dt.day_name()
        df['weekend'] = df["date"].dt.day_of_week >= 5
        df['trimestru'] = df['date'].dt.quarter
        
        # Energy calculations
        df['total_regenerabila'] = df[['hidro', 'fotovolt', 'eolian', 'biomasa']].sum(axis=1)
        df['total_neregenerabila'] = df[['carbune', 'hidrocarburi', 'nuclear']].sum(axis=1)
        df['procent_regenerabila'] = (df['total_regenerabila'] / df['productie'] * 100).fillna(0)
        df['sold_energetic'] = df['productie'] - df['consum']
        df['eficienta_retea'] = (df['productie'] / df['consum'] * 100).clip(0, 200)
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

def create_professional_metrics(df):
    """Create enhanced metric cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_production = df['productie'].mean()
        st.metric(
            "Average Production", 
            f"{avg_production:,.0f} MW",
            delta=f"{df['productie'].std():.0f} MW std"
        )
    
    with col2:
        renewable_pct = df['procent_regenerabila'].mean()
        st.metric(
            "Renewable Energy %", 
            f"{renewable_pct:.1f}%",
            delta=f"{(df['procent_regenerabila'].max() - renewable_pct):.1f}% max"
        )
    
    with col3:
        energy_balance = df['sold_energetic'].mean()
        st.metric(
            "Net Energy Balance", 
            f"{energy_balance:+.0f} MW",
            delta_color="inverse"
        )
    
    with col4:
        date_range = f"{df['date'].min().strftime('%b %Y')} - {df['date'].max().strftime('%b %Y')}"
        st.metric("Analysis Period", date_range)

def create_production_overview(df):
    """Create production overview visualizations"""
    st.markdown('<h2 class="section-header">Production Overview</h2>', unsafe_allow_html=True)
    
    # Main production timeline
    fig = px.line(df, x='date', y='productie', 
                  title='Energy Production Timeline',
                  labels={'productie': 'Production (MW)', 'date': 'Date'},
                  color_discrete_sequence=['#1f77b4'])
    fig.update_layout(
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Energy composition
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Stacked area chart of energy sources
        energy_sources = ['carbune', 'hidro', 'hidrocarburi', 'nuclear', 'eolian', 'fotovolt', 'biomasa']
        monthly_production = df.groupby(['an', 'luna'])[energy_sources].mean().reset_index()
        monthly_production['period'] = monthly_production['an'].astype(str) + '-' + monthly_production['luna'].astype(str).str.zfill(2)
        
        fig_area = px.area(monthly_production, x='period', y=energy_sources,
                          title='Energy Production Composition Over Time',
                          color_discrete_sequence=px.colors.qualitative.Set3)
        fig_area.update_layout(showlegend=True, xaxis_title="Period", yaxis_title="Production (MW)")
        st.plotly_chart(fig_area, use_container_width=True)
    
    with col2:
        # Current energy mix
        current_totals = df[energy_sources].sum()
        fig_pie = px.pie(values=current_totals.values, names=current_totals.index,
                        title='Current Energy Mix Distribution',
                        color_discrete_sequence=px.colors.qualitative.Set3)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

def create_source_analysis(df, selected_sources):
    """Detailed analysis of energy sources"""
    st.markdown('<h2 class="section-header">Source Performance Analysis</h2>', unsafe_allow_html=True)
    
    if not selected_sources:
        st.warning("Please select at least one energy source from the sidebar")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Source performance over time
        source_data = df.groupby('date')[selected_sources].mean().reset_index()
        fig_sources = px.line(source_data, x='date', y=selected_sources,
                             title='Energy Source Performance Timeline',
                             labels={'value': 'Production (MW)', 'variable': 'Energy Source'})
        st.plotly_chart(fig_sources, use_container_width=True)
    
    with col2:
        # Hourly production patterns
        hourly_patterns = df.groupby('ora')[selected_sources].mean()
        fig_hourly = px.line(hourly_patterns, 
                            title='Average Hourly Production Patterns',
                            labels={'value': 'Production (MW)', 'ora': 'Hour of Day'})
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Source Correlation Analysis")
    correlation_matrix = df[selected_sources].corr()
    fig_corr = px.imshow(correlation_matrix,
                        title='Energy Source Production Correlations',
                        color_continuous_scale='RdBu_r',
                        aspect="auto")
    st.plotly_chart(fig_corr, use_container_width=True)

def create_temporal_analysis(df):
    """Analyze temporal patterns"""
    st.markdown('<h2 class="section-header">Temporal Patterns</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Daily consumption vs production
        daily_patterns = df.groupby('ora')[['consum', 'productie']].mean().reset_index()
        fig_daily = px.line(daily_patterns, x='ora', y=['consum', 'productie'],
                           title='Daily Consumption vs Production Patterns',
                           labels={'value': 'Power (MW)', 'ora': 'Hour of Day'},
                           color_discrete_sequence=['#ff7f0e', '#2ca02c'])
        st.plotly_chart(fig_daily, use_container_width=True)
    
    with col2:
        # Weekly patterns
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_labels = ['Luni', 'Marți', 'Miercuri', 'Joi', 'Vineri', 'Sâmbătă', 'Duminică']
        
        weekly_data = df.groupby('ziua_saptamanii')['consum'].mean().reindex(weekday_order)
        fig_weekly = px.bar(x=weekday_labels, y=weekly_data.values,
                           title='Average Consumption by Weekday',
                           labels={'y': 'Consumption (MW)', 'x': 'Weekday'},
                           color=weekly_data.values,
                           color_continuous_scale='Viridis')
        st.plotly_chart(fig_weekly, use_container_width=True)
    
    # Seasonal analysis
    st.subheader("Seasonal Production Patterns")
    seasonal_data = df.groupby('luna').agg({
        'productie': 'mean',
        'consum': 'mean',
        'procent_regenerabila': 'mean'
    }).reset_index()
    
    fig_seasonal = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig_seasonal.add_trace(
        go.Scatter(x=seasonal_data['luna'], y=seasonal_data['productie'],
                  name="Production", line=dict(color='blue')),
        secondary_y=False,
    )
    
    fig_seasonal.add_trace(
        go.Scatter(x=seasonal_data['luna'], y=seasonal_data['procent_regenerabila'],
                  name="Renewable %", line=dict(color='green', dash='dot')),
        secondary_y=True,
    )
    
    fig_seasonal.update_layout(title_text="Seasonal Production and Renewable Energy Trends")
    fig_seasonal.update_xaxes(title_text="Month")
    fig_seasonal.update_yaxes(title_text="Production (MW)", secondary_y=False)
    fig_seasonal.update_yaxes(title_text="Renewable %", secondary_y=True)
    
    st.plotly_chart(fig_seasonal, use_container_width=True)

def create_comparative_analysis(df):
    """Comparative analysis between years and metrics"""
    st.markdown('<h2 class="section-header">Comparative Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Year-over-year comparison
        yearly_stats = df.groupby('an').agg({
            'productie': 'mean',
            'consum': 'mean',
            'procent_regenerabila': 'mean',
            'sold_energetic': 'mean'
        }).reset_index()
        
        fig_yearly = px.bar(yearly_stats, x='an', y=['productie', 'consum'],
                           title='Yearly Production vs Consumption',
                           barmode='group',
                           labels={'value': 'Power (MW)', 'variable': 'Metric'})
        st.plotly_chart(fig_yearly, use_container_width=True)
    
    with col2:
        # Renewable energy growth
        renewable_growth = df.groupby(['an', 'luna'])['procent_regenerabila'].mean().reset_index()
        renewable_growth['period'] = renewable_growth['an'].astype(str) + '-' + renewable_growth['luna'].astype(str).str.zfill(2)
        
        fig_growth = px.line(renewable_growth, x='period', y='procent_regenerabila',
                            title='Renewable Energy Percentage Trend',
                            labels={'procent_regenerabila': 'Renewable %'})
        st.plotly_chart(fig_growth, use_container_width=True)

def create_data_explorer(df):
    """Interactive data exploration section with enhanced range selection"""
    st.markdown('<h2 class="section-header">Data Explorer</h2>', unsafe_allow_html=True)
    
    # Create columns for main controls
    col1, col2 = st.columns(2)
    
    with col1:
        # Year range selection
        available_years = sorted(df['an'].unique())
        year_range = st.select_slider(
            "Select Year Range",
            options=available_years,
            value=(available_years[0], available_years[-1])
        )
    
    with col2:
        # Month selection with checkboxes
        st.write("Select Months:")
        all_months = list(range(1, 13))
        month_names = [datetime(2024, x, 1).strftime('%B') for x in all_months]

        # Select all checkbox
        select_all = st.checkbox("Select All Months", value=True)

        if select_all:
            selected_months = all_months
        else:
            selected_months = []
            cols = st.columns(4)  # 3 columns for 12 months
            for i, (month_num, month_name) in enumerate(zip(all_months, month_names)):
                col_idx = i % 4
                with cols[col_idx]:
                    if st.checkbox(month_name, value=False, key=f"month_{month_num}"):
                        selected_months.append(month_num)
      

    
    # Filter data based on selections
    start_year, end_year = year_range
    filtered_data = df[
        (df['an'] >= start_year) & 
        (df['an'] <= end_year) & 
        (df['luna'].isin(selected_months))
    ]

    
    if not filtered_data.empty:
        # Display statistics
        st.subheader("Data Summary")
        st.dataframe(
            filtered_data.describe(),
            use_container_width=True,
            height=300
        )
        
        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(
            filtered_data.head(100),
            use_container_width=True,
            height=400
        )
        
        # Key metrics
        st.subheader("Key Metrics")
        col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
        
        with col_metrics1:
            st.metric("Total Records", len(filtered_data))
        
        with col_metrics2:
            st.metric("Date Range", f"{start_year}-{end_year}")
        
        with col_metrics3:
            st.metric("Months Selected", len(selected_months))
        
        with col_metrics4:
            if 'value' in df.columns:
                st.metric("Average Value", f"{filtered_data['value'].mean():.2f}")
        
        # Download section
        st.markdown("---")
        st.subheader("Download Data")
        
        col_dl1, col_dl2 = st.columns([1, 2])
        
        with col_dl1:
            # Download as CSV
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"energy_data_{start_year}_to_{end_year}_months_{'-'.join(map(str, selected_months))}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_dl2:
            # Download as Excel
            @st.cache_data
            def convert_to_excel(df):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='EnergyData')
                return output.getvalue()
            
            excel_data = convert_to_excel(filtered_data)
            st.download_button(
                label="📊 Download as Excel",
                data=excel_data,
                file_name=f"energy_data_{start_year}_to_{end_year}.xlsx",
                mime="application/vnd.ms-excel",
                use_container_width=True
            )
        
        # Show download size info
        csv_size = len(csv) / 1024  # Size in KB
        st.info(f"Dataset contains {len(filtered_data):,} records. Download size: {csv_size:.1f} KB")
        
    else:
        st.warning("⚠️ No data available for the selected period and filters")
        st.info("Try adjusting your date range or filter criteria to see more data.")

# Alternative simpler version if you prefer the original layout but with range:
def create_data_explorer_simple(df):
    """Simplified version with range selection"""
    st.markdown('<h2 class="section-header">Data Explorer</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        available_years = sorted(df['an'].unique())
        year_range = st.select_slider(
            "Select Year Range",
            options=available_years,
            value=(available_years[0], available_years[-1])
        )
    
    with col2:
        st.write("Select Months:")
        all_months = list(range(1, 13))
        month_names = [datetime(2024, x, 1).strftime('%B') for x in all_months]

        # Select all checkbox
        select_all = st.checkbox("Select All Months", value=True)

        if select_all:
            selected_months = all_months
        else:
            selected_months = []
            cols = st.columns(4)  # 3 columns for 12 months
            for i, (month_num, month_name) in enumerate(zip(all_months, month_names)):
                col_idx = i % 4
                with cols[col_idx]:
                    if st.checkbox(month_name, value=False, key=f"month_{month_num}"):
                        selected_months.append(month_num)

    start_year, end_year = year_range
    filtered_data = df[
        (df['an'] >= start_year) & 
        (df['an'] <= end_year) & 
        (df['luna'].isin(selected_months))
    ]
    
    if not filtered_data.empty:
        st.dataframe(
            filtered_data.describe(),
            use_container_width=True,
            height=300
        )
        
        # Download filtered data
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name=f"energy_data_{start_year}_to_{end_year}.csv",
            mime="text/csv"
        )
        
        st.success(f"✅ Showing {len(filtered_data):,} records from {start_year} to {end_year}")
    else:
        st.warning("No data available for selected period")

def main():
    """Main application function"""
    
    # Header section
    st.markdown('<h1 class="main-header">⚡ Energy Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Academic Research Platform | Data Visualization & Analysis**")
    
    # Sidebar configuration
    st.sidebar.header("🔧 Control Panel")
    
    # File upload section
    st.sidebar.subheader("📁 Data Source")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Energy Data CSV",
        type=['csv'],
        help="Upload your energy data CSV file with columns similar to the default dataset"
    )
    
    # Load data
    df = load_data(uploaded_file)
    
    if df is None:
        st.error("Unable to load data. Please check your file format or use the default dataset.")
        return
    
    # Date range filter
    st.sidebar.subheader("📅 Date Range")
    min_date, max_date = df['date'].min().date(), df['date'].max().date()
    date_range = st.sidebar.date_input(
        "Select Analysis Period",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        df = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]
    
    # Energy source selection
    st.sidebar.subheader("🔋 Energy Sources")
    energy_sources = {
        'Renewable': ['hidro', 'eolian', 'fotovolt', 'biomasa'],
        'Conventional': ['carbune', 'hidrocarburi', 'nuclear']
    }
    
    selected_sources = []
    for category, sources in energy_sources.items():
        st.sidebar.markdown(f"**{category}**")
        for source in sources:
            if st.sidebar.checkbox(
                f"{source.title()}",
                value=True,
                key=f"source_{source}"
            ):
                selected_sources.append(source)
    
    # Display professional metrics
    st.markdown("---")
    create_professional_metrics(df)
    st.markdown("---")
    
    # Tab-based navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🔍 Source Analysis", 
        "🕒 Time Patterns", 
        "📈 Comparisons", 
        "💾 Data Explorer"
    ])
    
    with tab1:
        create_production_overview(df)
    
    with tab2:
        create_source_analysis(df, selected_sources)
    
    with tab3:
        create_temporal_analysis(df)
    
    with tab4:
        create_comparative_analysis(df)
    
    with tab5:
        create_data_explorer(df)
    
    # Insights section
    with st.expander("🎓 Academic Insights & Methodology", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Key Findings")
            st.markdown("""
            - **Renewable Integration**: Analysis of sustainable energy source penetration
            - **Demand Patterns**: Identification of consumption peaks and seasonal variations  
            - **Grid Efficiency**: Evaluation of production-consumption balance
            - **Source Correlations**: Understanding energy source complementarity
            """)
        
        with col2:
            st.subheader("Methodology")
            st.markdown("""
            - **Data Preprocessing**: Cleaning, validation, and feature engineering
            - **Time Series Analysis**: Trend decomposition and pattern recognition
            - **Statistical Modeling**: Correlation and comparative analysis
            - **Interactive Visualization**: Multi-dimensional data exploration
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p><strong>Academic Research Project | Master in Data Science</strong></p>
        <p>Advanced Data Visualization Techniques | Energy Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

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
import datetime

# Professional styling
st.set_page_config(
    page_title="Energy Data Analysis Dashboard", 
    layout="wide",
    page_icon="⚡"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        border-bottom: 2px solid #2e86ab;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("Data Analysis/lab1/raw_energy_data.csv", parse_dates=["date"])
    
    # Enhanced feature engineering
    df['an'] = df["date"].dt.year
    df['luna'] = df["date"].dt.month
    df['ziua'] = df['date'].dt.day
    df['ora'] = df["date"].dt.hour
    df['ziua_saptamanii'] = df["date"].dt.day_name()
    df['weekend'] = df["date"].dt.day_of_week >= 5
    
    # Energy calculations
    df['total_regenerabila'] = df['hidro'] + df['fotovolt'] + df['eolian'] + df['biomasa']
    df['total_neregenerabila'] = df['carbune'] + df['hidrocarburi'] + df['nuclear']
    df['procent_regenerabila'] = (df['total_regenerabila'] / df['productie'] * 100).fillna(0)
    df['sold_energetic'] = df['productie'] - df['consum']
    
    return df

df = load_data()

# Header with professional presentation
st.markdown('<h1 class="main-header">⚡ Energy Production Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown("**Academic Presentation - Data Visualization Course**")

# Key metrics at the top
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_production = df['productie'].sum()
    st.metric("Total Production", f"{total_production:,.0f} MW")
with col2:
    avg_renewable = df['procent_regenerabila'].mean()
    st.metric("Avg Renewable %", f"{avg_renewable:.1f}%")
with col3:
    date_range = f"{df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}"
    st.metric("Analysis Period", date_range)
with col4:
    energy_balance = df['sold_energetic'].mean()
    st.metric("Avg Energy Balance", f"{energy_balance:,.0f} MW")

# Enhanced sidebar with better organization
st.sidebar.header("🎛️ Analysis Parameters")

# Date range with more options
date_col1, date_col2 = st.sidebar.columns(2)
with date_col1:
    start_date = st.date_input("Start Date", 
                              datetime.date(2024, 1, 1), 
                              min_value=datetime.date(2024, 1, 1), 
                              max_value=datetime.date(2025, 10, 31))
with date_col2:
    end_date = st.date_input("End Date", 
                            datetime.date(2025, 10, 31), 
                            min_value=datetime.date(2024, 1, 1), 
                            max_value=datetime.date(2025, 10, 31))

# Energy type selection with categories
st.sidebar.subheader("Energy Source Selection")
energy_categories = {
    "Renewable": ['hidro', 'eolian', 'fotovolt', 'biomasa'],
    "Non-Renewable": ['carbune', 'hidrocarburi', 'nuclear']
}

selected_sources = []
for category, sources in energy_categories.items():
    with st.sidebar.expander(f"{category} Sources"):
        for source in sources:
            if st.checkbox(source, value=True, key=source):
                selected_sources.append(source)

# Analysis granularity
granularity = st.sidebar.selectbox(
    "Time Granularity",
    ["Hourly", "Daily", "Weekly", "Monthly"],
    index=1
)

# Visualization theme
theme = st.sidebar.selectbox(
    "Chart Theme",
    ["Seaborn", "Plotly", "Matplotlib", "Dark"]
)


# Apply filters
filtered_df = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]

# Tab-based organization for different analyses
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Overview", "🔍 Source Analysis", "🕒 Time Patterns", 
    "📊 Comparisons", "📋 Data Explorer"
])

with tab1:
    st.markdown('<h2 class="section-header">Energy Production Overview</h2>', unsafe_allow_html=True)
    
    # Interactive production timeline
    fig = px.line(filtered_df, x='date', y='productie', 
                  title='Energy Production Timeline',
                  labels={'productie': 'Production (MW)', 'date': 'Date'})
    fig.update_layout(hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    
    # Energy mix pie chart
    col1, col2 = st.columns(2)
    with col1:
        energy_totals = filtered_df[selected_sources].sum()
        fig_pie = px.pie(values=energy_totals.values, names=energy_totals.index,
                        title='Energy Source Distribution')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Renewable vs Non-renewable
        renewable_avg = filtered_df['procent_regenerabila'].mean()
        non_renewable_avg = 100 - renewable_avg
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = renewable_avg,
            title = {'text': "Renewable Energy Percentage"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "green"},
                'steps': [
                    {'range': [0, 33], 'color': "lightgray"},
                    {'range': [33, 66], 'color': "gray"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

with tab2:
    st.markdown('<h2 class="section-header">Detailed Source Analysis</h2>', unsafe_allow_html=True)
    
    # Multi-line chart for selected sources
    source_data = filtered_df.groupby('date')[selected_sources].mean().reset_index()
    fig_sources = px.line(source_data, x='date', y=selected_sources,
                         title='Energy Production by Source Over Time')
    st.plotly_chart(fig_sources, use_container_width=True)
    
    # Heatmap of production by hour and source
    heatmap_data = filtered_df.groupby('ora')[selected_sources].mean()
    fig_heatmap = px.imshow(heatmap_data.T, 
                           title='Average Production by Hour and Source',
                           labels=dict(x="Hour of Day", y="Energy Source", color="Production (MW)"))
    st.plotly_chart(fig_heatmap, use_container_width=True)

with tab3:
    st.markdown('<h2 class="section-header">Temporal Patterns</h2>', unsafe_allow_html=True)
    
    # Daily patterns
    daily_patterns = filtered_df.groupby('ora')[['consum', 'productie']].mean()
    fig_daily = px.line(daily_patterns, title='Daily Consumption vs Production Patterns')
    st.plotly_chart(fig_daily, use_container_width=True)
    
    # Weekly patterns
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_data = filtered_df.groupby('ziua_saptamanii')['consum'].mean().reindex(weekday_order)
    fig_weekly = px.bar(weekly_data, title='Average Consumption by Weekday')
    st.plotly_chart(fig_weekly, use_container_width=True)

with tab4:
    st.markdown('<h2 class="section-header">Comparative Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        # Year-over-year comparison
        yearly_comparison = filtered_df.groupby('an')[selected_sources].mean()
        fig_yearly = px.bar(yearly_comparison, barmode='group',
                           title='Year-over-Year Energy Production Comparison')
        st.plotly_chart(fig_yearly, use_container_width=True)
    
    with col2:
        # Seasonal analysis
        seasonal_data = filtered_df.groupby('luna')['productie'].mean()
        fig_seasonal = px.line(seasonal_data, title='Seasonal Production Patterns',
                              labels={'value': 'Production (MW)', 'luna': 'Month'})
        st.plotly_chart(fig_seasonal, use_container_width=True)


def create_comprehensive_pdf():
    """Enhanced PDF generation with professional layout"""
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    import tempfile
    import os
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1,
        textColor=colors.HexColor('#1f77b4')
    )
    story.append(Paragraph("Energy Data Analysis Report", title_style))
    
    # Period information
    story.append(Paragraph(f"Analysis Period: {start_date} to {end_date}", styles['Normal']))
    story.append(Paragraph(f"Selected Energy Sources: {', '.join(selected_sources)}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Key statistics table
    stats_data = [['Metric', 'Value']]
    stats_data.extend([
        ['Total Production', f"{filtered_df['productie'].sum():,.0f} MW"],
        ['Average Renewable %', f"{filtered_df['procent_regenerabila'].mean():.1f}%"],
        ['Peak Production', f"{filtered_df['productie'].max():,.0f} MW"],
        ['Energy Balance', f"{filtered_df['sold_energetic'].mean():.1f} MW"]
    ])
    
    stats_table = Table(stats_data)
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(stats_table)
    story.append(Spacer(1, 30))
    
    # Create and save charts with proper file handling
    temp_files = []  # Keep track of temp files
    
    try:
        # Chart 1: Monthly production pattern
        fig, ax = plt.subplots(figsize=(8, 4))
        monthly_data = filtered_df.groupby('luna')['productie'].mean()
        monthly_data.plot(kind='bar', ax=ax, color='skyblue', alpha=0.7)
        ax.set_title('Monthly Production Pattern')
        ax.set_xlabel('Month')
        ax.set_ylabel('Average Production (MW)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save to temporary file
        temp_img1 = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        plt.savefig(temp_img1.name, dpi=150, bbox_inches='tight')
        temp_files.append(temp_img1.name)
        plt.close(fig)  # Explicitly close the figure to release the file
        
        story.append(Paragraph("Monthly Production Pattern", styles['Heading2']))
        story.append(Image(temp_img1.name, width=6*inch, height=3*inch))
        story.append(Spacer(1, 20))
        
    except Exception as e:
        story.append(Paragraph(f"Chart 1 generation error: {str(e)}", styles['Normal']))
    
    try:
        # Chart 2: Energy source distribution
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        energy_totals = filtered_df[selected_sources].sum()
        ax2.pie(energy_totals.values, labels=energy_totals.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Energy Source Distribution')
        
        temp_img2 = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        plt.savefig(temp_img2.name, dpi=150, bbox_inches='tight')
        temp_files.append(temp_img2.name)
        plt.close(fig2)  # Explicitly close the figure
        
        story.append(Paragraph("Energy Source Distribution", styles['Heading2']))
        story.append(Image(temp_img2.name, width=6*inch, height=3*inch))
        
    except Exception as e:
        story.append(Paragraph(f"Chart 2 generation error: {str(e)}", styles['Normal']))
    
    # Build the PDF
    doc.build(story)
    buffer.seek(0)
    
    # Clean up temporary files after PDF is built
    for temp_file in temp_files:
        try:
            os.unlink(temp_file)
        except Exception as e:
            print(f"Warning: Could not delete temporary file {temp_file}: {e}")
    
    return buffer.getvalue()

# PDF generation section
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Report Generation")

if st.sidebar.button("🔄 Generate Comprehensive Report"):
    with st.spinner("Generating professional report..."):
        pdf_bytes = create_comprehensive_pdf()
        st.sidebar.download_button( 
            "📥 Download Full Report",
            data=pdf_bytes,
            file_name=f"energy_analysis_report_{datetime.datetime.now()}.pdf",
            mime="application/pdf"
        )


# Add presentation mode
st.sidebar.markdown("---")
st.sidebar.subheader("🎓 Presentation Mode")

if st.sidebar.checkbox("Enable Presentation Mode"):
    st.markdown("""
    <style>
    .main > div {
        max-width: 100%;
        padding-left: 5%;
        padding-right: 5%;
    }
    </style>
    """, unsafe_allow_html=True)

# Key insights section
with st.expander("🔍 Key Academic Insights", expanded=True):
    st.write("""
    **Data Analysis Highlights:**
    - **Renewable Energy Trends**: Observe the growth patterns of sustainable energy sources
    - **Consumption Patterns**: Identify peak usage hours and seasonal variations
    - **Energy Balance**: Analyze the relationship between production and consumption
    - **Source Correlation**: Understand how different energy sources complement each other
    """)

# Methodology section
with st.expander("📚 Methodology"):
    st.write("""
    **Analysis Approach:**
    - Data preprocessing and cleaning
    - Time series decomposition
    - Comparative statistical analysis
    - Interactive visualization techniques
    - Pattern recognition and trend analysis
    """)

# Footer with academic context
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>Academic Project - Data Visualization Course</strong></p>
    <p>Demonstrating advanced data analysis techniques and interactive visualization capabilities</p>
</div>
""", unsafe_allow_html=True)



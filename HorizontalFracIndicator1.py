import numpy as np
import plotly.graph_objects as go
from scipy.stats import pearsonr
import lasio
import streamlit as st
import io

# Set page configuration
st.set_page_config(
    page_title="Horizontal Fracture Analysis",
    page_icon="⛰️",
    layout="wide"
)

# App title and description
st.title("Horizontal Fracture (HF) Analysis")
st.markdown("""
This app performs Horizontal Fracture (HF) analysis on well log data to find the optimal rotation angle 
that correlates seismic attributes with fracture density.
""")

# Sidebar for file upload and parameters
with st.sidebar:
    st.header("Input Parameters")
    
    # File upload
    uploaded_file = st.file_uploader("Upload LAS file", type=["las", "LAS"])
    
    if uploaded_file is not None:
        # Read the file content
        bytes_data = uploaded_file.read()
        
        # Convert to a file-like object for lasio
        try:
            las = lasio.read(io.StringIO(bytes_data.decode('utf-8')))
        except:
            las = lasio.read(io.BytesIO(bytes_data))
        
        # Get available curve names
        curve_names = list(las.curves.keys())
        
        # Curve selection
        st.subheader("Curve Selection")
        depth_curve = st.selectbox("Depth Curve", options=curve_names, index=0)
        ip_curve = st.selectbox("P-Impedance Curve", options=curve_names, index=min(1, len(curve_names)-1))
        vp_curve = st.selectbox("P-Wave Velocity Curve", options=curve_names, index=min(2, len(curve_names)-1))
        vs_curve = st.selectbox("S-Wave Velocity Curve", options=curve_names, index=min(3, len(curve_names)-1))
        
        # Fracture density curve selection (critical addition)
        if 'FRAC_DEN' in curve_names or 'FRAC_DENS' in curve_names:
            # Try to find fracture density curve automatically
            frac_options = [name for name in curve_names if 'FRAC' in name.upper() or 'DEN' in name.upper()]
            default_frac_index = 0 if frac_options else 0
        else:
            frac_options = curve_names
            default_frac_index = min(4, len(curve_names)-1) if len(curve_names) > 4 else 0
            
        frac_curve = st.selectbox("Fracture Density Curve (εf)", 
                                 options=frac_options, 
                                 index=default_frac_index,
                                 help="Select the curve containing fracture density measurements")
        
        # Get curves
        depth = las[depth_curve]
        IP = las[ip_curve]
        VP = las[vp_curve]
        VS = las[vs_curve]
        FRAC_DEN = las[frac_curve]  # Fracture density data
        
        # Calculate VP/VS ratio
        VP_VS = VP / VS
        
        # Clean data - ensure all required curves have valid values
        valid_idx = ~np.isnan(IP) & ~np.isnan(VP_VS) & ~np.isnan(FRAC_DEN)
        depth = depth[valid_idx]
        IP = IP[valid_idx]
        VP_VS = VP_VS[valid_idx]
        FRAC_DEN = FRAC_DEN[valid_idx]
        
        # Depth range selection
        st.subheader("Depth Range")
        depth_min = st.number_input("Min Depth", value=float(depth.min()))
        depth_max = st.number_input("Max Depth", value=float(depth.max()))
        
        # Calculate button
        calculate_btn = st.button("Calculate HF Analysis")
    else:
        st.info("Please upload a LAS file to begin analysis")
        st.stop()

# HF calculation function (corrected according to theory)
def calculate_HF(theta_deg, IP, VP_VS):
    """Calculate Horizontal Fracture indicator for a given angle"""
    theta_rad = np.deg2rad(theta_deg)
    return IP * np.cos(theta_rad) - VP_VS * np.sin(theta_rad)

# Perform calculation when button is clicked
if calculate_btn:
    # Show spinner while calculating
    with st.spinner("Calculating HF Analysis..."):
        # Get selected depth range
        depth_range = (depth_min, depth_max)
        mask = (depth >= depth_range[0]) & (depth <= depth_range[1])
        
        if not np.any(mask):
            st.error("⚠️ Error: No data in selected depth range!")
            st.stop()
        
        # Extract data for the selected interval
        depth_interval = depth[mask]
        IP_interval = IP[mask]
        VP_VS_interval = VP_VS[mask]
        FRAC_DEN_interval = FRAC_DEN[mask]
        
        # Calculate correlation for different theta values
        theta_values = np.linspace(0, 180, 181)  # 0 to 180 degrees in 1-degree increments
        correlations = []
        
        for theta in theta_values:
            # Calculate HF for this theta across all depth points
            hf_values = calculate_HF(theta, IP_interval, VP_VS_interval)
            
            # Calculate correlation with fracture density
            if len(hf_values) > 1:  # Ensure we have enough data points
                r, _ = pearsonr(hf_values, FRAC_DEN_interval)
                correlations.append(r)
            else:
                correlations.append(0)
        
        # Find optimal angle with maximum correlation
        theta_max = theta_values[np.argmax(correlations)]
        max_correlation = np.max(correlations)
        
        # Calculate HF values at optimal angle for the entire interval
        hf_optimal = calculate_HF(theta_max, IP_interval, VP_VS_interval)
        
        # Create the correlation vs angle plot
        fig_corr = go.Figure()
        
        # Add correlation curve
        fig_corr.add_trace(go.Scatter(
            x=theta_values, y=correlations,
            mode='lines',
            name='Correlation (R)',
            line=dict(color='blue', width=2),
            hovertemplate="θ: %{x}°<br>Correlation: %{y:.3f}<extra></extra>"
        ))
        
        # Add θ_max marker
        fig_corr.add_vline(
            x=theta_max,
            line=dict(color='red', dash='dash', width=1.5),
            annotation_text=f"θ_max = {theta_max:.1f}°",
            annotation_position="top right"
        )
        
        # Update layout
        fig_corr.update_layout(
            title=f'<b>Correlation between HF(θ) and Fracture Density</b><br>'
                  f'<i>Depth: {depth_range[0]:.1f}m - {depth_range[1]:.1f}m</i>',
            xaxis_title='Rotation Angle θ (degrees)',
            yaxis_title='Correlation Coefficient (R)',
            hovermode="x unified",
            template="plotly_white",
            height=500
        )
        
        # Create the HF vs Fracture Density crossplot at optimal angle
        fig_crossplot = go.Figure()
        
        fig_crossplot.add_trace(go.Scatter(
            x=hf_optimal, y=FRAC_DEN_interval,
            mode='markers',
            name='Data points',
            marker=dict(size=4, opacity=0.6),
            hovertemplate="HF: %{x:.2f}<br>Fracture Density: %{y:.3f}<extra></extra>"
        ))
        
        # Add trend line
        if len(hf_optimal) > 1:
            z = np.polyfit(hf_optimal, FRAC_DEN_interval, 1)
            p = np.poly1d(z)
            trend_x = np.linspace(min(hf_optimal), max(hf_optimal), 100)
            trend_y = p(trend_x)
            
            fig_crossplot.add_trace(go.Scatter(
                x=trend_x, y=trend_y,
                mode='lines',
                name='Trend line',
                line=dict(color='red', width=2)
            ))
        
        fig_crossplot.update_layout(
            title=f'<b>HF(θ_max) vs Fracture Density</b><br>'
                  f'<i>θ_max = {theta_max:.1f}°, R = {max_correlation:.3f}</i>',
            xaxis_title=f'HF(θ_max)',
            yaxis_title='Fracture Density (εf)',
            template="plotly_white",
            height=500
        )
        
        # Display the plots
        st.plotly_chart(fig_corr, use_container_width=True)
        st.plotly_chart(fig_crossplot, use_container_width=True)
        
        # Display numerical results
        st.subheader("Analysis Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Optimal angle θ_max", f"{theta_max:.1f}°")
        with col2:
            st.metric("Maximum Correlation", f"{max_correlation:.3f}")
        with col3:
            st.metric("Data Points", f"{len(FRAC_DEN_interval)}")
        
        # Additional details in expander
        with st.expander("Detailed Calculations"):
            st.write(f"**Formula at θ_max:** HF({theta_max:.1f}°) = IP·cos({theta_max:.1f}°) - VP/VS·sin({theta_max:.1f}°)")
            st.write(f"Mean IP: {np.mean(IP_interval):.2f}")
            st.write(f"Mean VP/VS: {np.mean(VP_VS_interval):.2f}")
            st.write(f"Mean Fracture Density: {np.mean(FRAC_DEN_interval):.4f}")
            
            # Show data statistics
            st.write("**Interval Statistics:**")
            st.write(f"Depth range: {depth_range[0]:.1f}m - {depth_range[1]:.1f}m")
            
        # Display the actual HF values at optimal angle alongside depth
        st.subheader("HF Values at Optimal Angle")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Depth vs HF(θ_max)")
            fig_depth_hf = go.Figure()
            fig_depth_hf.add_trace(go.Scatter(
                x=hf_optimal, y=depth_interval,
                mode='lines',
                name='HF(θ_max)',
                line=dict(color='purple', width=1)
            ))
            fig_depth_hf.update_layout(
                xaxis_title='HF(θ_max)',
                yaxis_title='Depth',
                yaxis=dict(autorange='reversed'),
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig_depth_hf, use_container_width=True)
            
        with col2:
            st.write("Depth vs Fracture Density")
            fig_depth_frac = go.Figure()
            fig_depth_frac.add_trace(go.Scatter(
                x=FRAC_DEN_interval, y=depth_interval,
                mode='lines',
                name='Fracture Density',
                line=dict(color='green', width=1)
            ))
            fig_depth_frac.update_layout(
                xaxis_title='Fracture Density (εf)',
                yaxis_title='Depth',
                yaxis=dict(autorange='reversed'),
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig_depth_frac, use_container_width=True)

# Display instructions if no calculation performed yet
else:
    st.info("""
    ### Instructions:
    1. Upload a LAS file using the sidebar
    2. Select the appropriate curves for analysis:
       - Depth curve
       - P-Impedance (IP) curve
       - P-wave velocity (VP) curve
       - S-wave velocity (VS) curve
       - Fracture density (εf) curve (most important!)
    3. Adjust the depth range if needed
    4. Click 'Calculate HF Analysis' to see results
    
    ### Note:
    This analysis finds the optimal rotation angle θ that maximizes correlation between
    the HF attribute (IP·cosθ - VP/VS·sinθ) and measured fracture density.
    """)

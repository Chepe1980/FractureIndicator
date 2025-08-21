import numpy as np
import plotly.graph_objects as go
from scipy.stats import pearsonr
from scipy.signal import savgol_filter
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
that correlates seismic attributes with fracture indicators.
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
        
        # Method selection for fracture proxy
        st.subheader("Fracture Density Estimation Method")
        method = st.radio(
            "Select method to estimate fracture density:",
            ["S-Wave Velocity Anomaly", "P-Wave Velocity Anomaly", "Impedance Contrast", 
             "Sonic Log DTCO", "Manual Curve Selection"]
        )
        
        # Additional parameters based on method
        if method == "Manual Curve Selection":
            frac_curve = st.selectbox("Select Fracture Proxy Curve", options=curve_names, 
                                     index=min(4, len(curve_names)-1))
        elif method == "Sonic Log DTCO":
            if any('DTCO' in name.upper() or 'DTC' in name.upper() for name in curve_names):
                dtco_options = [name for name in curve_names if 'DTCO' in name.upper() or 'DTC' in name.upper()]
                dtco_curve = st.selectbox("Select Sonic Curve (DTCO)", options=dtco_options, index=0)
            else:
                st.warning("No DTCO or DTC curve found. Using manual selection.")
                method = "Manual Curve Selection"
                frac_curve = st.selectbox("Select Fracture Proxy Curve", options=curve_names, 
                                         index=min(4, len(curve_names)-1))
        
        # Get curves
        depth = las[depth_curve]
        IP = las[ip_curve]
        VP = las[vp_curve]
        VS = las[vs_curve]
        
        # Calculate VP/VS ratio
        VP_VS = VP / VS
        
        # Depth range selection
        st.subheader("Depth Range")
        depth_min = st.number_input("Min Depth", value=float(depth.min()))
        depth_max = st.number_input("Max Depth", value=float(depth.max()))
        
        # Calculate button
        calculate_btn = st.button("Calculate HF Analysis")
    else:
        st.info("Please upload a LAS file to begin analysis")
        st.stop()

# Functions to estimate fracture density
def estimate_fracture_density(method, depth, VP, VS, IP, las, curve_name=None):
    """
    Estimate fracture density using various proxy methods
    """
    if method == "S-Wave Velocity Anomaly":
        # Calculate moving average of VS
        window_size = min(21, len(VS) // 10)  # Adaptive window size
        if window_size % 2 == 0:  # Ensure window size is odd for Savitzky-Golay
            window_size += 1
            
        VS_smooth = savgol_filter(VS, window_size, 2)  # Smooth VS
        # Fractures typically reduce S-wave velocity
        frac_estimate = (VS_smooth - VS) / VS_smooth
        frac_estimate = np.clip(frac_estimate, 0, None)  # Keep only positive anomalies
        
    elif method == "P-Wave Velocity Anomaly":
        # Calculate moving average of VP
        window_size = min(21, len(VP) // 10)
        if window_size % 2 == 0:
            window_size += 1
            
        VP_smooth = savgol_filter(VP, window_size, 2)  # Smooth VP
        # Fractures reduce P-wave velocity but less than S-wave
        frac_estimate = (VP_smooth - VP) / VP_smooth
        frac_estimate = np.clip(frac_estimate, 0, None)
        
    elif method == "Impedance Contrast":
        # Calculate moving average of IP
        window_size = min(21, len(IP) // 10)
        if window_size % 2 == 0:
            window_size += 1
            
        IP_smooth = savgol_filter(IP, window_size, 2)  # Smooth IP
        # Fractures typically reduce impedance
        frac_estimate = (IP_smooth - IP) / IP_smooth
        frac_estimate = np.clip(frac_estimate, 0, None)
        
    elif method == "Sonic Log DTCO":
        # Use sonic log (DTCO) as fracture indicator
        DTCO = las[curve_name]
        # Higher DTCO (slower velocity) often indicates fractures
        # Normalize and invert so higher values = more fractures
        dtco_norm = (DTCO - np.nanmin(DTCO)) / (np.nanmax(DTCO) - np.nanmin(DTCO))
        frac_estimate = dtco_norm
        
    elif method == "Manual Curve Selection":
        # Use selected curve as fracture proxy
        frac_estimate = las[curve_name]
        # Normalize
        frac_estimate = (frac_estimate - np.nanmin(frac_estimate)) / \
                       (np.nanmax(frac_estimate) - np.nanmin(frac_estimate))
    
    return frac_estimate

# HF calculation function
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
        
        # Estimate fracture density
        if method == "Sonic Log DTCO":
            FRAC_ESTIMATE = estimate_fracture_density(method, depth_interval, VP[mask], VS[mask], 
                                                     IP_interval, las, dtco_curve)
        elif method == "Manual Curve Selection":
            FRAC_ESTIMATE = estimate_fracture_density(method, depth_interval, VP[mask], VS[mask], 
                                                     IP_interval, las, frac_curve)
        else:
            FRAC_ESTIMATE = estimate_fracture_density(method, depth_interval, VP[mask], VS[mask], 
                                                     IP_interval, las)
        
        # Clean data - ensure all required curves have valid values
        valid_idx = ~np.isnan(IP_interval) & ~np.isnan(VP_VS_interval) & ~np.isnan(FRAC_ESTIMATE)
        depth_interval = depth_interval[valid_idx]
        IP_interval = IP_interval[valid_idx]
        VP_VS_interval = VP_VS_interval[valid_idx]
        FRAC_ESTIMATE = FRAC_ESTIMATE[valid_idx]
        
        # Calculate correlation for different theta values
        theta_values = np.linspace(0, 180, 181)  # 0 to 180 degrees in 1-degree increments
        correlations = []
        
        for theta in theta_values:
            # Calculate HF for this theta across all depth points
            hf_values = calculate_HF(theta, IP_interval, VP_VS_interval)
            
            # Calculate correlation with fracture estimate
            if len(hf_values) > 1:  # Ensure we have enough data points
                r, _ = pearsonr(hf_values, FRAC_ESTIMATE)
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
            title=f'<b>Correlation between HF(θ) and Fracture Estimate ({method})</b><br>'
                  f'<i>Depth: {depth_range[0]:.1f}m - {depth_range[1]:.1f}m</i>',
            xaxis_title='Rotation Angle θ (degrees)',
            yaxis_title='Correlation Coefficient (R)',
            hovermode="x unified",
            template="plotly_white",
            height=500
        )
        
        # Create the HF vs Fracture Estimate crossplot at optimal angle
        fig_crossplot = go.Figure()
        
        fig_crossplot.add_trace(go.Scatter(
            x=hf_optimal, y=FRAC_ESTIMATE,
            mode='markers',
            name='Data points',
            marker=dict(size=4, opacity=0.6),
            hovertemplate="HF: %{x:.2f}<br>Fracture Estimate: %{y:.3f}<extra></extra>"
        ))
        
        # Add trend line
        if len(hf_optimal) > 1:
            z = np.polyfit(hf_optimal, FRAC_ESTIMATE, 1)
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
            title=f'<b>HF(θ_max) vs Fracture Estimate</b><br>'
                  f'<i>θ_max = {theta_max:.1f}°, R = {max_correlation:.3f}</i>',
            xaxis_title=f'HF(θ_max)',
            yaxis_title='Fracture Estimate',
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
            st.metric("Data Points", f"{len(FRAC_ESTIMATE)}")
        
        # Additional details in expander
        with st.expander("Detailed Calculations"):
            st.write(f"**Formula at θ_max:** HF({theta_max:.1f}°) = IP·cos({theta_max:.1f}°) - VP/VS·sin({theta_max:.1f}°)")
            st.write(f"Mean IP: {np.mean(IP_interval):.2f}")
            st.write(f"Mean VP/VS: {np.mean(VP_VS_interval):.2f}")
            st.write(f"Mean Fracture Estimate: {np.mean(FRAC_ESTIMATE):.4f}")
            
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
            st.write("Depth vs Fracture Estimate")
            fig_depth_frac = go.Figure()
            fig_depth_frac.add_trace(go.Scatter(
                x=FRAC_ESTIMATE, y=depth_interval,
                mode='lines',
                name='Fracture Estimate',
                line=dict(color='green', width=1)
            ))
            fig_depth_frac.update_layout(
                xaxis_title='Fracture Estimate',
                yaxis_title='Depth',
                yaxis=dict(autorange='reversed'),
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig_depth_frac, use_container_width=True)
            
        # Explanation of the selected method
        st.subheader("Method Explanation")
        if method == "S-Wave Velocity Anomaly":
            st.info("""
            **S-Wave Velocity Anomaly Method**: 
            This method detects fractures by identifying intervals where S-wave velocity is lower than expected.
            Fractures significantly reduce S-wave velocity more than P-wave velocity, creating detectable anomalies
            when compared to a smoothed background trend.
            """)
        elif method == "P-Wave Velocity Anomaly":
            st.info("""
            **P-Wave Velocity Anomaly Method**: 
            This method detects fractures by identifying intervals where P-wave velocity is lower than expected.
            While P-wave is less sensitive to fractures than S-wave, it still shows measurable reductions in fractured zones.
            """)
        elif method == "Impedance Contrast":
            st.info("""
            **Impedance Contrast Method**: 
            This method uses P-impedance (IP) anomalies to detect fractures. Fractured zones typically have 
            lower impedance than intact rock, creating detectable contrasts against the background trend.
            """)
        elif method == "Sonic Log DTCO":
            st.info("""
            **Sonic Log (DTCO) Method**: 
            This method uses compressional wave slowness (DTCO) as a fracture indicator. Higher DTCO values
            (slower velocities) often correlate with fractured intervals. The curve is normalized to create
            a fracture probability estimate.
            """)
        elif method == "Manual Curve Selection":
            st.info(f"""
            **Manual Curve Selection Method**: 
            Using the {frac_curve} curve as a fracture indicator. This curve has been normalized to create
            a fracture probability estimate. Ensure this curve has a physical relationship with fracturing
            (e.g., resistivity anomalies, density changes, or other fracture-sensitive measurements).
            """)

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
    3. Choose a method to estimate fracture density
    4. Adjust the depth range if needed
    5. Click 'Calculate HF Analysis' to see results
    
    ### Note:
    This analysis finds the optimal rotation angle θ that maximizes correlation between
    the HF attribute (IP·cosθ - VP/VS·sinθ) and estimated fracture density.
    
    ### No Fracture Density Data?
    If you don't have direct fracture measurements, the app provides several methods to estimate
    fracture density from commonly available logs:
    - **S-Wave Velocity Anomaly**: Detects fractures as negative anomalies in S-wave velocity
    - **P-Wave Velocity Anomaly**: Detects fractures as negative anomalies in P-wave velocity  
    - **Impedance Contrast**: Uses P-impedance anomalies to identify fractured zones
    - **Sonic Log DTCO**: Uses sonic slowness as a fracture indicator
    - **Manual Selection**: Choose any curve that might correlate with fractures
    """)

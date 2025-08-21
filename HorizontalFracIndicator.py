import numpy as np
import plotly.graph_objects as go
from scipy.stats import pearsonr
import lasio
import streamlit as st
import io

# Set page configuration
st.set_page_config(
    page_title="Well Log HF Analysis",
    page_icon="⛰️",
    layout="wide"
)

# App title and description
st.title("Well Log HF Analysis")
st.markdown("""
This app performs Horizontal Fluid (HF) analysis on well log data to find the optimal rotation angle 
for fluid detection by correlating impedance properties with seismic response.
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
        
        # Get curves
        depth = las[depth_curve]
        IP = las[ip_curve]
        VP = las[vp_curve]
        VS = las[vs_curve]
        
        # Calculate VP/VS ratio
        VP_VS = VP / VS
        
        # Clean data
        valid_idx = ~np.isnan(IP) & ~np.isnan(VP_VS)
        depth = depth[valid_idx]
        IP = IP[valid_idx]
        VP_VS = VP_VS[valid_idx]
        
        # Depth range selection
        st.subheader("Depth Range")
        depth_min = st.number_input("Min Depth", value=float(depth.min()))
        depth_max = st.number_input("Max Depth", value=float(depth.max()))
        
        # Analysis parameters
        st.subheader("Analysis Parameters")
        noise_level = st.slider("Synthetic Seismic Noise Level", 0.01, 0.5, 0.1, 0.01)
        
        # Calculate button
        calculate_btn = st.button("Calculate HF Analysis")
    else:
        st.info("Please upload a LAS file to begin analysis")
        st.stop()

# HF calculation function
def HF(theta_deg, IP, VP_VS):
    theta_rad = np.deg2rad(theta_deg)
    return IP * np.cos(theta_rad) - VP_VS * np.sin(theta_rad)

def synthetic_seismic_response(theta_deg, noise_level=0.1):
    theta_rad = np.deg2rad(theta_deg)
    signal = 0.8 * np.cos(theta_rad) + 0.5 * np.sin(theta_rad)
    noise = noise_level * np.random.randn(len(theta_deg))
    return signal + noise

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
        
        # Calculate averages for the interval
        IP_avg = np.mean(IP[mask])
        VP_VS_avg = np.mean(VP_VS[mask])
        
        # Core HF calculation
        theta = np.linspace(0, 360, 360)
        hf_values = HF(theta, IP_avg, VP_VS_avg)
        seismic_response = synthetic_seismic_response(theta, noise_level)
        
        # Rolling correlation
        window_size = 30
        correlations = []
        for i in range(len(theta) - window_size):
            r, _ = pearsonr(hf_values[i:i+window_size], seismic_response[i:i+window_size])
            correlations.append(r)
        correlations = np.pad(correlations, (0, window_size), 'edge')
        
        # Find peak correlation angle
        theta_max = theta[np.argmax(correlations)]
        hf_theta_max = HF(theta_max, IP_avg, VP_VS_avg)
        
        # Create the figure
        fig = go.Figure()
        
        # Add HF curve (left axis)
        fig.add_trace(go.Scatter(
            x=theta, y=hf_values,
            mode='lines',
            name='HF(θ)',
            line=dict(color='blue', width=2),
            hovertemplate="θ: %{x}°<br>HF: %{y:.2f}<extra></extra>"
        ))
        
        # Add correlation curve (right axis)
        fig.add_trace(go.Scatter(
            x=theta, y=correlations,
            mode='lines',
            name='Correlation',
            line=dict(color='green', width=1.5),
            yaxis='y2',
            hovertemplate="θ: %{x}°<br>R: %{y:.2f}<extra></extra>"
        ))
        
        # Add θ_max marker
        fig.add_vline(
            x=theta_max,
            line=dict(color='black', dash='dot', width=1),
            annotation_text=f"θ_max = {theta_max:.1f}°",
            annotation_position="top right"
        )
        
        # Update layout
        fig.update_layout(
            title=f'<b>HF Analysis: {depth_range[0]:.1f}m - {depth_range[1]:.1f}m</b><br>'
                  f'<i>IP = {IP_avg:.0f}, VP/VS = {VP_VS_avg:.2f}</i>',
            xaxis_title='Rotation Angle θ (degrees)',
            yaxis=dict(title='HF(θ)', side='left'),
            yaxis2=dict(title='Correlation (R)', overlaying='y', side='right'),
            hovermode="x unified",
            template="plotly_white",
            height=500
        )
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Display numerical results
        st.subheader("Analysis Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Optimal angle θ_max", f"{theta_max:.1f}°")
        with col2:
            st.metric("HF(θ_max)", f"{hf_theta_max:.2f}")
        with col3:
            st.metric("Correlation at θ_max", f"{np.max(correlations):.3f}")
        
        # Additional details in expander
        with st.expander("Detailed Calculations"):
            st.write(f"IP·cos(θ_max) = {IP_avg * np.cos(np.deg2rad(theta_max)):.2f}")
            st.write(f"VP/VS·sin(θ_max) = {VP_VS_avg * np.sin(np.deg2rad(theta_max)):.2f}")
            
            # Show data statistics
            st.write("**Interval Statistics:**")
            st.write(f"Depth range: {depth_range[0]:.1f}m - {depth_range[1]:.1f}m")
            st.write(f"Number of data points: {np.sum(mask)}")
            st.write(f"Mean IP: {IP_avg:.2f}")
            st.write(f"Mean VP/VS: {VP_VS_avg:.2f}")

# Display instructions if no calculation performed yet
else:
    st.info("""
    ### Instructions:
    1. Upload a LAS file using the sidebar
    2. Select the appropriate curves for analysis
    3. Adjust the depth range if needed
    4. Set the synthetic seismic noise level
    5. Click 'Calculate HF Analysis' to see results
    """)
    
    # Placeholder image or text
    st.image("https://via.placeholder.com/800x400?text=HF+Analysis+Results+Will+Appear+Here", 
             caption="HF analysis results will appear here after calculation")

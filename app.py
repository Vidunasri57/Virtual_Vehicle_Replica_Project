import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ai_engine import ECUCommunicationAI
from src.telemetry import (
    DEFAULT_NODERED_URL,
    fetch_nodered_telemetry,
    generate_simulated_telemetry,
)

st.set_page_config(
    page_title="Virtual Vehicle Replica | ECU Diagnostics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Force high-contrast Light Automotive Engineering Theme
st.markdown(
    """
    <style>
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #F1F5F9 !important;
        color: #0F172A !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #CBD5E1 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #0F172A !important;
    }

    /* Make Sidebar Collapse & Unhide (Expand) Buttons Always Visible */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    button[data-testid="stExpandSidebarButton"],
    button[data-testid="stBaseButton-headerNoPadding"] {
        background-color: #EFF6FF !important;
        color: #1E3A8A !important;
        border: 1.5px solid #2563EB !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 5px rgba(15, 23, 42, 0.12) !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    [data-testid="stSidebarCollapsedControl"] *,
    [data-testid="collapsedControl"] *,
    button[data-testid="stExpandSidebarButton"] *,
    button[data-testid="stBaseButton-headerNoPadding"] * {
        color: #1E3A8A !important;
        fill: #1E3A8A !important;
        opacity: 1 !important;
    }

    /* Navigation Bar Buttons (Top Row) */
    div[data-testid="stHorizontalBlock"] button {
        background-color: #FFFFFF !important;
        border: 1px solid #94A3B8 !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        width: 100% !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06) !important;
    }
    div[data-testid="stHorizontalBlock"] button p,
    div[data-testid="stHorizontalBlock"] button span,
    div[data-testid="stHorizontalBlock"] button div {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 14px !important;
    }
    div[data-testid="stHorizontalBlock"] button[kind="primary"] {
        background-color: #2563EB !important;
        border-color: #1D4ED8 !important;
    }
    div[data-testid="stHorizontalBlock"] button[kind="primary"] p,
    div[data-testid="stHorizontalBlock"] button[kind="primary"] span,
    div[data-testid="stHorizontalBlock"] button[kind="primary"] div {
        color: #FFFFFF !important;
    }

    /* Sidebar Clear Button */
    section[data-testid="stSidebar"] button {
        background-color: #2563EB !important;
        border: none !important;
    }
    section[data-testid="stSidebar"] button * {
        color: #FFFFFF !important;
    }

    /* Primary & Action Buttons (e.g., Update Endpoint) */
    div[data-testid="stButton"] button[kind="primary"] {
        background-color: #2563EB !important;
        border: 1px solid #1D4ED8 !important;
        border-radius: 8px !important;
        padding: 8px 20px !important;
    }
    div[data-testid="stButton"] button[kind="primary"] * {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* Text Input Field Contrast */
    div[data-testid="stTextInput"] input {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #94A3B8 !important;
        border-radius: 8px !important;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-top: 4px solid #2563EB !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        box-shadow: 0 4px 10px rgba(15, 23, 42, 0.06) !important;
    }
    div[data-testid="stMetricLabel"] * {
        color: #475569 !important;
        font-weight: 700 !important;
        font-size: 13px !important;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }
    div[data-testid="stMetricValue"] * {
        color: #0F172A !important;
        font-weight: 800 !important;
    }

    /* Alert Contrast */
    div[data-testid="stAlert"] * {
        color: #0F172A !important;
        font-weight: 600 !important;
    }

    /* AI Status Banner */
    .status-banner {
        padding: 16px 22px;
        border-radius: 12px;
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 16px;
        border-left: 8px solid;
        box-shadow: 0 3px 8px rgba(15, 23, 42, 0.06);
    }
    .status-normal {
        background-color: #DCFCE7;
        color: #064E3B;
        border-color: #16A34A;
    }
    .status-degraded {
        background-color: #FEF3C7;
        color: #78350F;
        border-color: #D97706;
    }
    .status-fault {
        background-color: #FEE2E2;
        color: #7F1D1D;
        border-color: #DC2626;
    }

    /* Uniform Architecture Pipeline Cards */
    .pipeline-box {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 12px;
        padding: 14px 20px;
        margin-bottom: 16px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
        flex-wrap: wrap;
        gap: 10px;
    }
    .pipe-step {
        flex: 1;
        min-width: 170px;
        background: #EFF6FF;
        border: 1px solid #93C5FD;
        color: #1E3A8A;
        padding: 10px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 13px;
        text-align: center;
    }
    .pipe-arrow {
        color: #64748B;
        font-weight: 900;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def style_plotly(fig: go.Figure, height: int = 320) -> go.Figure:
    """Forces crisp dark markings, ticks, titles, and gridlines on every Plotly chart."""
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        height=height,
        font=dict(color="#0F172A", size=13),
        title_font=dict(color="#0F172A", size=16),
        legend=dict(
            font=dict(color="#0F172A", size=12),
            bgcolor="rgba(255,255,255,0.8)",
        ),
        xaxis=dict(
            tickfont=dict(color="#1E293B", size=11),
            title_font=dict(color="#0F172A", size=13),
            gridcolor="#E2E8F0",
            linecolor="#94A3B8",
            zerolinecolor="#CBD5E1",
        ),
        yaxis=dict(
            tickfont=dict(color="#1E293B", size=11),
            title_font=dict(color="#0F172A", size=13),
            gridcolor="#E2E8F0",
            linecolor="#94A3B8",
            zerolinecolor="#CBD5E1",
        ),
    )
    return fig


def make_gauge(title: str, value: float, min_val: float, max_val: float, unit: str, bar_color: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": f" {unit}", "font": {"size": 24, "color": "#0F172A"}},
            title={"text": title, "font": {"size": 15, "color": "#0F172A"}},
            gauge={
                "axis": {
                    "range": [min_val, max_val],
                    "tickcolor": "#0F172A",
                    "tickfont": {"color": "#0F172A", "size": 11},
                },
                "bar": {"color": bar_color},
                "bgcolor": "#F1F5F9",
                "borderwidth": 1,
                "bordercolor": "#94A3B8",
            },
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=220,
        margin=dict(l=25, r=25, t=50, b=15),
        paper_bgcolor="#FFFFFF",
        font={"color": "#0F172A"},
    )
    return fig


@st.cache_resource
def load_ai_engine():
    return ECUCommunicationAI()


def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "events" not in st.session_state:
        st.session_state.events = []
    if "last_state" not in st.session_state:
        st.session_state.last_state = None
    if "nodered_url" not in st.session_state:
        st.session_state.nodered_url = DEFAULT_NODERED_URL
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Overview"


init_session_state()

# Sidebar Controls
with st.sidebar:
    st.title("Control Panel")
    data_source = st.radio(
        "Telemetry Source",
        options=["Built-in Simulator", "Node-RED Live API"],
        index=0,
    )
    st.divider()
    sim_profile = st.radio(
        "Simulator Condition",
        options=["Normal", "Degraded", "Critical Fault"],
        index=0,
        disabled=(data_source != "Built-in Simulator"),
    )
    st.divider()
    auto_refresh = st.toggle("Auto-Refresh (2s)", value=True)
    if st.button("Clear History & Logs", use_container_width=True):
        st.session_state.history = []
        st.session_state.events = []
        st.session_state.last_state = None

st.title("Virtual Vehicle Replica — ECU Communication Monitor")

# Always-Visible Top Navigation Bar
NAV_TABS = ["Overview", "Live Telemetry", "AI Diagnostics", "Trends", "Event Log", "Settings"]
nav_cols = st.columns(len(NAV_TABS))
for idx, tab_name in enumerate(NAV_TABS):
    is_active = st.session_state.active_tab == tab_name
    if nav_cols[idx].button(
        tab_name,
        key=f"nav_{tab_name}",
        type="primary" if is_active else "secondary",
        use_container_width=True,
    ):
        st.session_state.active_tab = tab_name
        st.rerun()

try:
    ai_engine = load_ai_engine()
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()


@st.fragment(run_every="2s" if auto_refresh else None)
def render_live_dashboard():
    # 1. Fetch Telemetry
    if data_source == "Node-RED Live API":
        ok, telemetry, status_msg = fetch_nodered_telemetry(st.session_state.nodered_url)
        if not ok:
            st.warning(f"{status_msg} — Showing fallback Normal test telemetry.")
            telemetry = generate_simulated_telemetry("Normal")
            source_badge = "Fallback Local Generator"
        else:
            source_badge = "Node-RED Live API (Port 1880)"
    else:
        telemetry = generate_simulated_telemetry(sim_profile)
        source_badge = f"Built-in Simulator ({sim_profile})"

    # 2. Classify Current Communication State
    ai_result = ai_engine.evaluate(telemetry)

    # 3. Update Rolling History & Event Log
    record = {**telemetry, **ai_result}
    st.session_state.history.append(record)
    st.session_state.history = st.session_state.history[-60:]

    current_state = ai_result["state_label"]
    if current_state != st.session_state.last_state:
        st.session_state.events.insert(
            0,
            {
                "Time": telemetry["timestamp"],
                "Detected State": current_state,
                "Confidence (%)": ai_result["confidence_pct"],
                "Severity Index": ai_result["severity_score"],
                "Diagnostic Summary": "; ".join(ai_result["diagnostics"]),
            },
        )
        st.session_state.last_state = current_state

    # 4. Uniform Visual System Data-Flow Pipeline
    st.markdown(
        f"""
        <div class="pipeline-box">
            <div class="pipe-step">1. ECU Nodes<br><span style="font-weight:400;font-size:11px;">Engine & Battery Sensors</span></div>
            <div class="pipe-arrow">➔</div>
            <div class="pipe-step">2. Telemetry Source<br><span style="font-weight:400;font-size:11px;">{source_badge}</span></div>
            <div class="pipe-arrow">➔</div>
            <div class="pipe-step">3. Random Forest AI<br><span style="font-weight:400;font-size:11px;">State: {current_state}</span></div>
            <div class="pipe-arrow">➔</div>
            <div class="pipe-step">4. Live Dashboard<br><span style="font-weight:400;font-size:11px;">Updated {telemetry['timestamp']}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 5. Top AI Status Banner
    banner_cls = {
        "Normal": "status-normal",
        "Degraded": "status-degraded",
        "Critical Fault": "status-fault",
    }[current_state]

    st.markdown(
        f"""
        <div class="status-banner {banner_cls}">
            Current ECU Communication State: <strong>{current_state.upper()}</strong>
            &nbsp;|&nbsp; Classifier Confidence: <strong>{ai_result['confidence_pct']}%</strong>
            &nbsp;|&nbsp; Severity Index: <strong>{ai_result['severity_score']} / 100</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 6. Render Active View
    active = st.session_state.active_tab

    if active == "Overview":
        g1, g2, g3 = st.columns(3)
        sev_color = "#16A34A" if current_state == "Normal" else ("#D97706" if current_state == "Degraded" else "#DC2626")
        with g1:
            st.plotly_chart(
                make_gauge("AI Degradation Severity", ai_result["severity_score"], 0, 100, "/ 100", sev_color),
                use_container_width=True,
                theme=None,
            )
        with g2:
            st.plotly_chart(
                make_gauge("Vehicle Speed", telemetry["vehicle_speed"], 0, 160, "km/h", "#2563EB"),
                use_container_width=True,
                theme=None,
            )
        with g3:
            st.plotly_chart(
                make_gauge("Battery State of Charge", telemetry["soc"], 0, 100, "%", "#0EA5E9"),
                use_container_width=True,
                theme=None,
            )

        st.subheader("ECU Communication Health Metrics")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Latency", f"{telemetry['latency_ms']} ms")
        c2.metric("Jitter", f"{telemetry['jitter_ms']} ms")
        c3.metric("Packet Loss", f"{telemetry['packet_loss_pct']} %")
        c4.metric("Missed Deadlines", telemetry["missed_deadlines"])
        c5.metric("Timeouts", telemetry["timeouts"])

        st.subheader("Vehicle Subsystem Snapshot")
        v1, v2, v3, v4, v5 = st.columns(5)
        v1.metric("Battery Voltage", f"{telemetry['battery_voltage']} V")
        v2.metric("Battery Current", f"{telemetry['battery_current']} A")
        v3.metric("Engine Temp", f"{telemetry['temperature']} °C")
        v4.metric("Vehicle Speed", f"{telemetry['vehicle_speed']} km/h")
        v5.metric("State of Charge", f"{telemetry['soc']} %")

        hist_df = pd.DataFrame(st.session_state.history)
        if len(hist_df) > 1:
            st.subheader("Real-Time Latency & Jitter Pulse")
            fig_pulse = px.line(
                hist_df,
                x="timestamp",
                y=["latency_ms", "jitter_ms"],
                markers=True,
                labels={"value": "Milliseconds (ms)", "timestamp": "Time", "variable": "Signal"},
                color_discrete_sequence=["#2563EB", "#0284C7"],
            )
            style_plotly(fig_pulse, height=270)
            st.plotly_chart(fig_pulse, use_container_width=True, theme=None)

    elif active == "Live Telemetry":
        st.subheader("Recent Telemetry Stream (Last 60 Samples)")
        hist_df = pd.DataFrame(st.session_state.history)
        display_cols = [
            "timestamp",
            "state_label",
            "latency_ms",
            "jitter_ms",
            "packet_loss_pct",
            "missed_deadlines",
            "timeouts",
            "battery_voltage",
            "battery_current",
            "temperature",
            "vehicle_speed",
            "soc",
        ]
        st.dataframe(
            hist_df[display_cols].iloc[::-1],
            use_container_width=True,
            hide_index=True,
        )

    elif active == "AI Diagnostics":
        col_left, col_right = st.columns([1.2, 1])
        with col_left:
            st.subheader("Current State Class Probabilities")
            prob_df = pd.DataFrame(
                {
                    "Communication State": list(ai_result["class_probabilities"].keys()),
                    "Confidence (%)": list(ai_result["class_probabilities"].values()),
                }
            )
            fig_prob = px.bar(
                prob_df,
                x="Communication State",
                y="Confidence (%)",
                text="Confidence (%)",
                color="Communication State",
                color_discrete_map={
                    "Normal": "#16A34A",
                    "Degraded": "#D97706",
                    "Critical Fault": "#DC2626",
                },
                range_y=[0, 110],
            )
            fig_prob.update_traces(textposition="outside", textfont=dict(color="#0F172A"))
            style_plotly(fig_prob, height=340)
            fig_prob.update_layout(showlegend=False)
            st.plotly_chart(fig_prob, use_container_width=True, theme=None)

            st.subheader("Diagnostic Findings")
            for item in ai_result["diagnostics"]:
                if current_state == "Critical Fault":
                    st.error(item)
                elif current_state == "Degraded":
                    st.warning(item)
                else:
                    st.success(item)

        with col_right:
            st.subheader("Random Forest Feature Importance")
            importances = ai_engine.metadata.get("feature_importances", {})
            if importances:
                imp_df = pd.DataFrame(
                    {
                        "Feature": list(importances.keys()),
                        "Importance": list(importances.values()),
                    }
                ).sort_values("Importance", ascending=True)
                fig_imp = px.bar(
                    imp_df,
                    x="Importance",
                    y="Feature",
                    text="Importance",
                    orientation="h",
                    color_discrete_sequence=["#2563EB"],
                )
                fig_imp.update_traces(textposition="outside", textfont=dict(color="#0F172A"))
                style_plotly(fig_imp, height=340)
                st.plotly_chart(fig_imp, use_container_width=True, theme=None)
            st.caption(
                f"Classifier: Random Forest | Validation Accuracy: "
                f"{ai_engine.metadata.get('validation_accuracy', 'N/A')}"
            )

    elif active == "Trends":
        hist_df = pd.DataFrame(st.session_state.history)
        if len(hist_df) > 1:
            fig_lat = px.line(
                hist_df,
                x="timestamp",
                y=["latency_ms", "jitter_ms"],
                markers=True,
                labels={"value": "Milliseconds (ms)", "timestamp": "Time", "variable": "Metric"},
                title="Latency & Jitter Over Time",
                color_discrete_sequence=["#2563EB", "#0284C7"],
            )
            style_plotly(fig_lat, height=330)
            st.plotly_chart(fig_lat, use_container_width=True, theme=None)

            fig_sev = px.area(
                hist_df,
                x="timestamp",
                y="severity_score",
                markers=True,
                labels={"severity_score": "Current Severity Index (0–100)", "timestamp": "Time"},
                title="Communication Degradation Severity Trend",
                color_discrete_sequence=["#2563EB"],
            )
            fig_sev.update_yaxes(range=[0, 105])
            style_plotly(fig_sev, height=300)
            st.plotly_chart(fig_sev, use_container_width=True, theme=None)
        else:
            st.info("Collecting samples... trend charts will render after 2 refresh cycles.")

    elif active == "Event Log":
        st.subheader("Communication State Transitions")
        if st.session_state.events:
            st.dataframe(
                pd.DataFrame(st.session_state.events),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.write("No state transitions logged yet.")

    elif active == "Settings":
        st.subheader("System & Endpoint Configuration")
        new_url = st.text_input("Node-RED API Endpoint URL", value=st.session_state.nodered_url)
        if st.button("Update Endpoint", type="primary"):
            st.session_state.nodered_url = new_url
            st.success(f"Endpoint updated to {new_url}")

        st.markdown("**Active Model Metadata**")
        st.json(ai_engine.metadata)


render_live_dashboard()
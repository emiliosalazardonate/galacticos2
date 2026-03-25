import streamlit as st
import pandas as pd

st.set_page_config(page_title="Hybrid Games Simulator", layout="wide")

# --- CUSTOM CSS FOR BETTER VISUALS ---
st.markdown("""
    <style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏆 Hybrid Games: Performance Simulator")

# --- SIDEBAR: ATHLETE INFO ---
with st.sidebar:
    st.header("User Profile")
    name = st.text_input("Athlete Name", "Athlete 1")
    gender = st.radio("Gender", ["Male", "Female", "Other"])
    st.divider()
    submit = st.button("Simulate Total Score", type="primary")

# --- MAIN CONTENT: INPUT GROUPS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏋️ Strength Blocks")

    with st.container():
        st.markdown("**💪 Deadlift**")
        dl_weight = st.number_input("Peso (kg)", min_value=0, max_value=500, key="dl_w")
        dl_reps = st.number_input("Reps", min_value=0, max_value=50, key="dl_r")
        st.caption(f"Total Volume: {dl_weight * dl_reps} kg")

    st.divider()

    with st.container():
        st.markdown("**🦵 Back Squat**")
        bs_weight = st.number_input("Peso (kg)", min_value=0, max_value=500, key="bs_w")
        bs_reps = st.number_input("Reps", min_value=0, max_value=50, key="bs_r")
        st.caption(f"Total Volume: {bs_weight * bs_reps} kg")

    st.divider()

    with st.container():
        st.markdown("**🏋️ Shoulder Press**")
        sp_weight = st.number_input("Peso (kg)", min_value=0, max_value=300, key="sp_w")
        sp_reps = st.number_input("Reps", min_value=0, max_value=50, key="sp_r")
        st.caption(f"Total Volume: {sp_weight * sp_reps} kg")

with col2:
    st.subheader("🏃 Cardio Blocks")

    with st.container():
        st.markdown("**🎿 SkiErg**")
        ski_cal = st.number_input("Calories", min_value=0, max_value=1000, key="ski")

    st.divider()

    with st.container():
        st.markdown("**🚴 BikeErg**")
        bike_cal = st.number_input("Calories", min_value=0, max_value=2000, key="bike")

    st.divider()

    with st.container():
        st.markdown("**🚣 Rowerg**")
        r_col_a, r_col_b = st.columns(2)
        row_min = r_col_a.number_input("Min", min_value=0, max_value=60, key="row_m")
        row_seg = r_col_b.number_input("Seg", min_value=0, max_value=59, key="row_s")

# --- CALCULATION LOGIC ---
total_strength = (dl_weight * dl_reps) + (bs_weight * bs_reps) + (sp_weight * sp_reps)
total_cardio_cals = ski_cal + bike_cal
row_total_seconds = (row_min * 60) + row_seg

# --- DISPLAY RESULTS ---
if submit:
    st.divider()
    res_col1, res_col2, res_col3 = st.columns(3)

    res_col1.metric("Total Lifted", f"{total_strength} kg")
    res_col2.metric("Cardio Output", f"{total_cardio_cals} Cal")

    # Simple score algorithm (Example: 1 point per 10kg + 5 points per cal)
    final_score = (total_strength / 10) + (total_cardio_cals * 2) - (row_total_seconds / 10)
    res_col3.metric("Simulated Points", round(final_score, 2))

    st.balloons()
import streamlit as st
import pandas as pd

# --- PAGE CONFIG ---
st.set_page_config(page_title="Hybrid Games Simulator", layout="wide")

# --- MOCK DATABASE (In a real app, use a CSV or SQL) ---
if 'leaderboard' not in st.session_state:
    st.session_state.leaderboard = pd.DataFrame([
        {"Athlete": "Alex Rivers", "Reps": 150, "Time": "08:30", "Points": 95},
        {"Athlete": "Jordan Smith", "Reps": 142, "Time": "09:15", "Points": 88},
        {"Athlete": "Casey V.", "Reps": 130, "Time": "10:00", "Points": 80},
    ])

# --- HEADER ---
st.title("🏆 Workout Simulator & Ranking")
st.markdown("Enter your results below to see where you would place in the current leaderboard.")

# --- SIDEBAR: INPUT DATA ---
with st.sidebar:
    st.header("Your Stats")
    name = st.text_input("Athlete Name", placeholder="e.g. John Doe")
    reps = st.number_input("Total Reps Completed", min_value=0, max_value=500, value=100)
    time_min = st.slider("Time (Minutes)", 0, 20, 10)

    # Simple Logic for Points (Replicating the "Simulador" logic)
    # Higher reps = higher points.
    simulated_points = reps * 0.5 + (20 - time_min) * 2

    if st.button("Simulate My Rank"):
        new_entry = {"Athlete": f"{name} (YOU)", "Reps": reps, "Time": f"{time_min}:00", "Points": simulated_points}
        # Temporary add for simulation
        st.session_state.temp_board = pd.concat([st.session_state.leaderboard, pd.DataFrame([new_entry])])
        st.success(f"Simulated Score: {simulated_points} pts")

# --- MAIN CONTENT: RANKING TABLE ---
display_board = st.session_state.get('temp_board', st.session_state.leaderboard)

# Sort by Points Descending
display_board = display_board.sort_values(by="Points", ascending=False).reset_index(drop=True)
display_board.index += 1  # Make index start at 1 for "Rank"

st.subheader("Current Standings")

# Styling the DataFrame
st.dataframe(
    display_board,
    use_container_width=True,
    column_config={
        "Points": st.column_config.ProgressColumn("Score Progress", min_value=0, max_value=150),
        "Athlete": "Competitor"
    }
)

# --- VISUALIZATION ---
st.divider()
st.subheader("Performance Comparison")
st.bar_chart(data=display_board, x="Athlete", y="Points")
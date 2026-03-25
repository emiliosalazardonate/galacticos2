import streamlit as st
import pandas as pd
import os

# --- FILE SETUP ---
DB_FILE = "athlete_results.csv"


def load_data():
    if os.path.exists(DB_FILE):
        return pd.read_csv(DB_FILE)
    return pd.DataFrame(
        columns=["Name", "Gender", "Category", "Total Lifted (kg)", "Cardio Cal", "Row Time (s)", "Final Score"])


def save_data(new_row):
    df = load_data()
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(DB_FILE, index=False)


# --- PAGE CONFIG ---
st.set_page_config(page_title="Hybrid Games Leaderboard", layout="wide")

# --- SIDEBAR: ATHLETE & CATEGORY ---
with st.sidebar:
    st.header("Registration")
    name = st.text_input("Athlete Name")
    gender = st.selectbox("Gender", ["Male", "Female"])
    category = st.selectbox("Category", ["RX", "Scaled", "Masters", "Elite"])
    st.divider()

# --- INPUT COLUMNS (Same as before) ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("🏋️ Strength")
    dl = st.number_input("Deadlift (kg)", 0) * st.number_input("DL Reps", 0)
    bs = st.number_input("Back Squat (kg)", 0) * st.number_input("BS Reps", 0)
    sp = st.number_input("Shoulder Press (kg)", 0) * st.number_input("SP Reps", 0)
    total_strength = dl + bs + sp

with col2:
    st.subheader("🏃 Cardio")
    ski = st.number_input("SkiErg Cal", 0)
    bike = st.number_input("BikeErg Cal", 0)
    row_m = st.number_input("Row Min", 0)
    row_s = st.number_input("Row Seg", 0)
    total_cardio = ski + bike
    row_total_sec = (row_m * 60) + row_s

# --- CALCULATE & SAVE ---
final_score = (total_strength / 10) + (total_cardio * 2) - (row_total_sec / 10)

if st.button("💾 Save Result to Leaderboard"):
    if name:
        new_entry = {
            "Name": name, "Gender": gender, "Category": category,
            "Total Lifted (kg)": total_strength, "Cardio Cal": total_cardio,
            "Row Time (s)": row_total_sec, "Final Score": round(final_score, 2)
        }
        save_data(new_entry)
        st.success(f"Result saved for {name} in {category}!")
    else:
        st.error("Please enter an Athlete Name.")

# --- DISPLAY LEADERBOARD BY CATEGORY ---
st.divider()
st.header("📊 Official Leaderboard")

data = load_data()

if not data.empty:
    # Filter by Category
    selected_cat = st.tabs(["All", "RX", "Scaled", "Masters", "Elite"])

    categories = ["All", "RX", "Scaled", "Masters", "Elite"]

    for i, tab in enumerate(selected_cat):
        with tab:
            if categories[i] == "All":
                filtered_df = data
            else:
                filtered_df = data[data["Category"] == categories[i]]

            # Sort by Score
            sorted_df = filtered_df.sort_values(by="Final Score", ascending=False).reset_index(drop=True)
            sorted_df.index += 1  # Rank column
            st.table(sorted_df)
else:
    st.info("No results recorded yet.")
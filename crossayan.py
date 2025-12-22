import streamlit as st
import pandas as pd
import os

def calculate_reps(rounds, last_step):
    total_reps = 0
    # Calculate reps for completed rounds
    for i in range(1, rounds + 1):
        total_reps += sum(range(1, i + 1))

    # Add reps for the current, partially completed round
    total_reps += sum(range(1, last_step + 1))
    return total_reps

st.title("Crossaiyan NaviWOD Rep Calculator")

st.write("""
This app calculates the total repetitions for a specific CrossFit workout.
The workout is a ladder, adding an exercise in each round with repetitions equal to the round number.
- Round 1: 1 rep of exercise 1
- Round 2: 1 rep of exercise 1, 2 reps of exercise 2
- Round 3: 1 rep of exercise 1, 2 reps of exercise 2, 3 reps of exercise 3
...and so on, up to 12 rounds.
""")

# --- User Info ---
name = st.text_input("Enter your name:")
level = st.selectbox("Select your level:", ("RX", "SC", "Rookie"))

# --- WOD Progress ---
rounds = st.number_input("Enter the number of completed rounds:", min_value=0, max_value=12, value=0, step=1)
last_step = st.number_input("Enter the last step you finished in the current round:", min_value=0, max_value=12, value=0, step=1)

if st.button("Calculate and Save Reps"):
    if name:
        total_reps = calculate_reps(rounds, last_step)
        st.write(f"Total repetitions: {total_reps}")

        # --- Save to CSV ---
        csv_file = 'crossfit_results.csv'
        
        new_data = {
            'Name': name,
            'Level': level,
            'Completed Rounds': rounds,
            'Last Step': last_step,
            'Total Reps': total_reps
        }
        df_new = pd.DataFrame([new_data])

        # Check if file exists to decide on writing headers
        file_exists = os.path.exists(csv_file)
        
        # Append data to the CSV file
        df_new.to_csv(csv_file, mode='a', header=not file_exists, index=False)

        st.success(f"Well done, {name}! Your result has been saved.")

        # --- Display Results ---
        st.write("### All Results")
        all_results_df = pd.read_csv(csv_file)
        st.dataframe(all_results_df)
    else:
        st.warning("Please enter your name before saving.")

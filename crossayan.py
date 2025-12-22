import streamlit as st

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

rounds = st.number_input("Enter the number of completed rounds:", min_value=0, max_value=12, value=0, step=1)
last_step = st.number_input("Enter the last step you finished in the current round:", min_value=0, max_value=12, value=0, step=1)

if st.button("Calculate Total Reps"):
    total_reps = calculate_reps(rounds, last_step)
    st.write(f"Total repetitions: {total_reps}")

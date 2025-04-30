# File: app.py
import streamlit as st
import pandas as pd
import io
from predict_local import preprocess_raw_data, predict_all

def main():
    # Title
    st.title("Food-Item Classifier")

    # Sidebar inputs
    st.sidebar.header("Survey Responses")
    q1 = st.sidebar.slider(
        "1. Complexity (1=very simple, 5=very complex)", 1, 5, 3
    )
    q2 = st.sidebar.text_input("2. Expected number of ingredients", "5")
    q3 = st.sidebar.multiselect(
        "3. Setting(s) of serving",
        [
            "At a party", "Late night snack", "Week day lunch",
            "Week day dinner", "Weekend lunch", "Weekend dinner"
        ]
    )
    q4 = st.sidebar.text_input("4. Expected price (numeric)", "10")
    q5 = st.sidebar.text_input("5. Movie association", "")
    q6 = st.sidebar.text_input("6. Drink pairing", "")
    q7 = st.sidebar.multiselect(
        "7. Reminds you of",
        ["Parents", "Siblings", "Friends", "Teachers", "Strangers"]
    )
    q8 = st.sidebar.selectbox(
        "8. Hot-sauce level",
        [
            "None", "A little (mild)",
            "A moderate amount (medium)",
            "A lot (hot)",
            "I will have some of this food item with my hot sauce"
        ]
    )

    # Build input DataFrame
    input_df = pd.DataFrame([{
        'Q1_complexity': q1,
        'Q2_ingredients': q2,
        'Q3_setting': ", ".join(q3),
        'Q4_price': q4,
        'Q5_movie': q5,
        'Q6_drink': q6,
        'Q7_reminds': ", ".join(q7),
        'Q8_hot_sauce': q8
    }])

    # Serialize to CSV buffer for preprocessing
    csv_buffer = io.StringIO()
    input_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    # Directly call predict_all on raw CSV buffer
    prediction = predict_all(csv_buffer)

    # Display result
    st.markdown("## 🥡 Predicted food item:")
    st.success(prediction[0])

if __name__ == "__main__":
    main()

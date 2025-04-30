import streamlit as st
import pandas as pd
from predict_local import predict_all, preprocess_raw_data  # your refactored module

st.title("Food-Item Classifier")

# 1. Gather inputs
st.sidebar.header("Survey Responses")
q1 = st.sidebar.slider("1. Complexity (1=very simple, 5=very complex)", 1, 5, 3)
q2 = st.sidebar.text_input("2. Expected number of ingredients", "5")
q3 = st.sidebar.multiselect("3. Setting(s) of serving",
                            ["At a party","Late night snack","Week day lunch","Week day dinner","Weekend lunch","Weekend dinner"])
q4 = st.sidebar.text_input("4. Expected price (numeric)", "10")
q5 = st.sidebar.text_input("5. Movie association", "")
q6 = st.sidebar.text_input("6. Drink pairing", "")
q7 = st.sidebar.multiselect("7. Reminds you of",
                            ["Parents","Siblings","Friends","Teachers","Strangers"])
q8 = st.sidebar.selectbox("8. Hot-sauce level",
                          ["None","A little (mild)","A moderate amount (medium)","A lot (hot)","I will have some of this food item with my hot sauce"])

# 2. Build a one-row DataFrame matching your training format
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

# 3. Run through your preprocessing pipeline
X_raw, _ = preprocess_raw_data(input_df)
# (if your preprocess expects a file, you can refactor it to accept DataFrames)

# 4. Predict
pred = predict_all_from_df(X_raw)  # you may need to expose a helper that skips CSV reads

st.write("## 🥡 Predicted food item:")
st.success(pred[0])

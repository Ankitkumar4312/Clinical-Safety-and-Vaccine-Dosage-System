
import streamlit as st
import pickle
import pandas as pd
import time



# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Vaccine Dosage - Dashboard",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #e8f4f8, #f8fbff);
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: red;
    margin-bottom: 5px;
}

.stButton > button:hover {
    background-color: grey;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

with open(
    r"D:\COMPUTER SCIENCE\PRIME BATCH\PROJECTS\6.Vaccine safety & Dosage\notebook\Logistic_Regression.pkl",
    "rb"
) as file:

    LOG_R = pickle.load(file)


# --------------------------------------------------
# LOAD SCALER
# --------------------------------------------------

with open(r"D:\COMPUTER SCIENCE\PRIME BATCH\PROJECTS\6.Vaccine safety & Dosage\notebook\scaler.pkl","rb")as file:
        scaler=pickle.load(file)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("Clinical Safety And Vaccine Dosage System")
# st.write(type(scaler))


# --------------------------------------------------
# INPUT FEATURES
# --------------------------------------------------

columns = [
    "Age",
    "Biomarker_Level",
    "Pre_Existing_Conditions",
    "Vaccine_Dose_mcg",
    "Prior_Reaction_History"
]


Age = st.number_input(
    "Enter the age:",
    min_value=18.0,
    max_value=100.0,
    step=1.0
)


Biomarker_Level = st.number_input(
    "Enter the Biomarker level of patient:",
    min_value=0.0,
    step=0.01,
    format="%.2f"
)


Pre_Existing_Conditions = st.selectbox(
    "Any Pre-Existing Conditions:",
    [0, 1, 2, 3]
)


Vaccine_Dose_mcg = st.number_input(
    "Vaccine Dose (in mcg):",
    min_value=0.0,
    step=0.1,
    format="%.2f"
)


Prior_Reaction_History = st.selectbox(
    "How was your prior reaction?",
    ["None", "Mild", "Moderate"]
)


# --------------------------------------------------
# CONVERT PRIOR REACTION TO NUMERIC
# --------------------------------------------------

Prior_Reaction_History_mapping = {
    "None": 0,
    "Mild": 1,
    "Moderate": 2
}


Prior_Reaction_History = (
    Prior_Reaction_History_mapping[Prior_Reaction_History]
)


# --------------------------------------------------
# CREATE INPUT DATAFRAME
# --------------------------------------------------

input_data = pd.DataFrame(
    [[
        Age,
        Biomarker_Level,
        Pre_Existing_Conditions,
        Vaccine_Dose_mcg,
        Prior_Reaction_History
    ]],
    columns=columns
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("🔍 Predict Risk"):

    with st.spinner("Analyzing Patient Data..."):

        time.sleep(5)

        # Scale using the SAME scaler used during training
        # input_scaled = scaler.transform(input_data)

        # Prediction
        prediction = LOG_R.predict(input_data)[0]

        # Probability
        probability = LOG_R.predict_proba(input_data)[0][1]


    # --------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------

    if prediction == 0:

        st.success("✅ Low Risk")

    else:

        st.error("⚠️ High Risk")


    st.write("Prediction:", prediction)

    st.write(
        f"High Risk Probability: {probability * 100:.2f}%"
    )


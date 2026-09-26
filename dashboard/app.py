
import streamlit as st
import pandas as pd
import time
import joblib



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
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: pink;
    text-align: center;
    margin-bottom: 5px;
}
.stApp {
    background-color: white;
    background-image:url("https://wallpapercave.com/wp/wp2595600.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
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
    background-color: pink;
    color: black;
}

</style>
""", unsafe_allow_html=True)

if "reset" not in st.session_state:
    st.session_state.reset = 0
# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------


model=joblib.load(r"D:\COMPUTER SCIENCE\PRIME BATCH\PROJECTS\6.Vaccine safety & Dosage\notebook\Logistic_regression.pkl")

# --------------------------------------------------
# LOAD SCALER
# --------------------------------------------------


scaler=joblib.load(r"D:\COMPUTER SCIENCE\PRIME BATCH\PROJECTS\6.Vaccine safety & Dosage\notebook\Scaler.pkl")

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("Clinical Safety And Vaccine Dosage System")
# st.write(type(scaler))
st.subheader("Patient Information")


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
    placeholder="Min_age=18",
    max_value=100.0,
    step=1.0,value=None,
    key=f"Age{st.session_state.reset}"
)


Biomarker_Level = st.number_input(
    "Enter the Biomarker level of patient:",
    min_value=0.0,
    step=0.01,
    format="%.2f",
    value=None,placeholder="Range 0-15",
    key=f"Biomarker_level{st.session_state.reset}"
)


Pre_Existing_Conditions = st.selectbox(
    "Any Pre-Existing Conditions:",
    [0, 1, 2, 3],
    # index="None"
    # placeholder="select an option",
    key=f"Pre_Existing_Conditions{st.session_state.reset}"
)


Vaccine_Dose_mcg = st.number_input(
    "Vaccine Dose (in mcg):",
    min_value=0.0,
    step=0.1,
    format="%.2f",
    value=None,placeholder="Range:25mg - 150mg",
    key=f"Vaccine_Dose_mcg{st.session_state.reset}"
)


Prior_Reaction_History = st.selectbox(
    "How was your prior reaction?",
    ["None", "Mild", "Moderate"],
    placeholder="Select an option",
    key=f"Prior_Reaction_History{st.session_state.reset}"
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

st.divider()
# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("🔍 Predict Risk",width="stretch"):

    with st.spinner("Analyzing Patient Data..."):

        time.sleep(5)

        # Scale using the SAME scaler used during training
        input_scaled = scaler.transform(input_data)

        # Prediction
        prediction = model.predict(input_scaled)[0]

        # Probability
        probability = model.predict_proba(input_scaled)[0][1]


    # --------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------

    if prediction == 0:
        st.balloons()
        st.success("✅ Low Risk")

    else:
        st.error("⚠️ High Risk")


    st.write("Prediction:", prediction)

    st.write(
        f" Risk Probability: {probability * 100:.2f}%"
    )

if st.button("🔄 Enter New Patient", width="stretch"):

    st.session_state.reset += 1
    st.rerun()

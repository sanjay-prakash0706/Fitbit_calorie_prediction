from pathlib import Path
import base64
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent / "notebooks"
BACKGROUND_PATH = Path(__file__).resolve().parent / "background.png"

st.set_page_config(page_title="Fitbit Analytics Dashboard", page_icon="🔥", layout="wide")

# Black + yellow fitness theme
if BACKGROUND_PATH.exists():
    encoded = base64.b64encode(BACKGROUND_PATH.read_bytes()).decode()
    bg = f'''background-image: linear-gradient(rgba(0,0,0,.70), rgba(0,0,0,.82)), url("data:image/png;base64,{encoded}");'''
else:
    bg = "background: #080808;"

st.markdown(f"""
<style>
.stApp {{ {bg} background-size: cover; background-position: center; background-attachment: fixed; }}
.block-container {{ background: rgba(0,0,0,.48); padding-top: 2rem; }}
h1, h2, h3 {{ color: #FFD21F !important; font-weight: 800 !important; }}
p, label, .stMarkdown, .stCaption {{ color: #F5F5F5 !important; }}
section[data-testid="stSidebar"] {{ background: rgba(0,0,0,.94); border-right: 1px solid #FFD21F; }}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] label {{ color: #FFD21F !important; }}
[data-baseweb="select"] > div, [data-baseweb="input"] > div {{ background: #FFFFFF !important; border-radius: 8px !important; }}
input {{ color: #111111 !important; }}
[data-baseweb="select"] * {{ color: #111111 !important; }}
.stButton > button {{ background: #FFD21F !important; color: #000000 !important; border: 0 !important; border-radius: 8px !important; font-weight: 800 !important; }}
.stButton > button:hover {{ background: #F5B800 !important; }}
[data-testid="stMetric"] {{ background: rgba(15,15,15,.92); border: 1px solid #FFD21F; border-radius: 12px; padding: 15px; }}
[data-testid="stMetricLabel"] {{ color: #FFFFFF !important; }}
[data-testid="stMetricValue"] {{ color: #FFD21F !important; }}
button[data-baseweb="tab"] {{ color: #FFFFFF !important; font-weight: 700; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color: #FFD21F !important; }}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    return (
        joblib.load(BASE_DIR / "tuned_xgboost_calorie_model.pkl"),
        joblib.load(BASE_DIR / "cluster_scaler.pkl"),
        joblib.load(BASE_DIR / "workout_kmeans_model.pkl"),
    )

@st.cache_data
def load_data():
    return pd.read_csv(BASE_DIR / "workout_clustered_data.csv")

try:
    calorie_model, cluster_scaler, cluster_model = load_models()
    df = load_data()
except Exception as error:
    st.error("Check that all model files and the CSV are inside the notebooks folder.")
    st.exception(error)
    st.stop()

st.title("🔥 Fitbit Calorie & Workout Analytics Dashboard")
st.caption("Calorie prediction • Workout clustering • Interactive analytics")

def values(column, default):
    if column in df.columns and df[column].dropna().nunique() > 0:
        return df[column].dropna().unique().tolist()
    return default

def median(column, default):
    return float(df[column].median()) if column in df.columns else default


def styled_dataframe(data, height=None):
    """Display a dataframe with black cells and yellow headers."""
    styled_data = (
        data.style
        .set_properties(
            **{
                "background-color": "#111111",
                "color": "#FFFFFF",
                "border-color": "#FFD21F",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#FFD21F"),
                        ("color", "#000000"),
                        ("font-weight", "bold"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [("border", "1px solid #FFD21F")],
                },
            ]
        )
    )

    kwargs = {"use_container_width": True}
    if height is not None:
        kwargs["height"] = height
    st.dataframe(styled_data, **kwargs)

def text_number(label, default, cast=float):
    raw = st.sidebar.text_input(label, value=str(default), placeholder=f"Enter {label.lower()}").strip()
    if not raw:
        st.sidebar.error(f"Please enter {label.lower()}.")
        st.stop()
    try:
        return cast(raw)
    except ValueError:
        st.sidebar.error(f"Please enter a valid value for {label.lower()}.")
        st.stop()

# Sidebar
st.sidebar.header("🏋️ Workout Inputs")
age = text_number("Age", int(median("Age", 25)), int)
gender = st.sidebar.selectbox("Gender", values("Gender", ["Male", "Female"]))
weight = text_number("Weight (kg)", median("Weight_kg", 70.0))
height = st.sidebar.number_input("Height (m)", 1.0, 2.5, median("Height_m", 1.7), step=0.01)
max_bpm = st.sidebar.number_input("Max BPM", 100, 220, int(median("Max_BPM", 180)))
avg_bpm = st.sidebar.number_input("Average BPM", 60, 200, int(median("Avg_BPM", 140)))
resting_bpm = st.sidebar.number_input("Resting BPM", 40, 120, int(median("Resting_BPM", 70)))
duration = st.sidebar.number_input("Session Duration (hours)", 0.1, 10.0, median("Session_Duration_hours", 1.0), step=0.01)
workout_type = st.sidebar.selectbox("Workout Type", values("Workout_Type", ["Cardio", "Strength", "Yoga", "HIIT"]))
fat = st.sidebar.number_input("Fat Percentage", 1.0, 60.0, median("Fat_Percentage", 20.0), step=0.1)
water = st.sidebar.number_input("Water Intake (liters)", 0.1, 10.0, median("Water_Intake_liters", 3.0), step=0.1)
frequency = st.sidebar.number_input("Workout Frequency (days/week)", 1, 7, int(median("Workout_Frequency_days_per_week", 3)))
experience = st.sidebar.selectbox("Experience Level", values("Experience_Level", ["Beginner", "Intermediate", "Advanced"]))
bmi = st.sidebar.number_input("BMI", 10.0, 60.0, median("BMI", 24.0), step=0.1)
weekly_hours = st.sidebar.number_input("Weekly Workout Hours", 0.0, 50.0, median("Weekly_Workout_Hours", 5.0), step=0.1)
bmi_category = st.sidebar.selectbox("BMI Category", values("BMI_Category", ["Normal", "Underweight", "Overweight", "Obese"]))
fat_category = st.sidebar.selectbox("Fat Category", values("Fat_Category", ["Fit", "Average", "High", "Very High"]))

model_features = ["Age", "Gender", "Weight_kg", "Height_m", "Max_BPM", "Avg_BPM", "Resting_BPM", "Session_Duration_hours", "Workout_Type", "Fat_Percentage", "Water_Intake_liters", "Workout_Frequency_days_per_week", "Experience_Level", "BMI", "Weekly_Workout_Hours", "BMI_Category", "Fat_Category"]

input_data = pd.DataFrame([{
    "Age": age, "Gender": gender, "Weight_kg": weight, "Height_m": height,
    "Max_BPM": max_bpm, "Avg_BPM": avg_bpm, "Resting_BPM": resting_bpm,
    "Session_Duration_hours": duration, "Workout_Type": workout_type,
    "Fat_Percentage": fat, "Water_Intake_liters": water,
    "Workout_Frequency_days_per_week": frequency, "Experience_Level": experience,
    "BMI": bmi, "Weekly_Workout_Hours": weekly_hours,
    "BMI_Category": bmi_category, "Fat_Category": fat_category,
}])[model_features]

# KPI cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records", f"{len(df):,}")
c2.metric("Average Calories", f"{df['Calories_Burned_kcal'].mean():.2f} kcal")
c3.metric("Clusters", str(df["Workout_Cluster"].nunique()))
c4.metric("Average Session", f"{df['Session_Duration_hours'].mean():.2f} hrs")

tab1, tab2, tab3, tab4 = st.tabs(["🔥 Calorie Prediction", "🧩 Workout Pattern", "📊 Analytics", "📄 Dataset"])

with tab1:
    st.subheader("Calorie Burn Prediction")
    if st.button("Predict Calories", key="calorie_button"):
        prediction = float(calorie_model.predict(input_data)[0])
        st.success(f"Estimated Calories Burned: {prediction:.2f} kcal")
        fig = px.bar(pd.DataFrame({"Metric": ["Prediction"], "Calories": [prediction]}), x="Metric", y="Calories", title="Estimated Calorie Burn")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Workout Pattern Prediction")
    cluster_features = ["Age", "Weight_kg", "Height_m", "Max_BPM", "Avg_BPM", "Resting_BPM", "Session_Duration_hours", "Fat_Percentage", "Water_Intake_liters", "Workout_Frequency_days_per_week", "BMI", "Weekly_Workout_Hours"]
    if st.button("Predict Workout Cluster", key="cluster_button"):
        cluster_id = int(cluster_model.predict(cluster_scaler.transform(input_data[cluster_features]))[0])
        st.success(f"Predicted Workout Cluster: {cluster_id}")
        summary = df[df["Workout_Cluster"] == cluster_id][cluster_features + ["Calories_Burned_kcal"]].mean(numeric_only=True).to_frame("Average").round(2)
        styled_dataframe(summary)

with tab3:
    st.subheader("Interactive Analytics")
    selected = st.multiselect("Select Clusters", sorted(df["Workout_Cluster"].unique()), default=sorted(df["Workout_Cluster"].unique()))
    filtered = df[df["Workout_Cluster"].isin(selected)]
    a, b = st.columns(2)
    with a:
        counts = filtered["Workout_Cluster"].value_counts().sort_index().reset_index()
        counts.columns = ["Workout_Cluster", "Count"]
        st.plotly_chart(px.bar(counts, x="Workout_Cluster", y="Count", text="Count", title="Cluster Distribution"), use_container_width=True)
    with b:
        st.plotly_chart(px.box(filtered, x="Workout_Cluster", y="Calories_Burned_kcal", title="Calories by Cluster"), use_container_width=True)
    a, b = st.columns(2)
    with a:
        st.plotly_chart(px.scatter(filtered, x="Session_Duration_hours", y="Calories_Burned_kcal", color="Workout_Cluster", hover_data=["Age", "Workout_Type", "Avg_BPM"], title="Duration vs Calories"), use_container_width=True)
    with b:
        workout_counts = filtered["Workout_Type"].value_counts().reset_index()
        workout_counts.columns = ["Workout_Type", "Count"]
        st.plotly_chart(px.pie(workout_counts, names="Workout_Type", values="Count", title="Workout Type Distribution"), use_container_width=True)
    styled_dataframe(filtered.groupby("Workout_Cluster").agg(Average_Age=("Age", "mean"), Average_Duration=("Session_Duration_hours", "mean"), Average_BPM=("Avg_BPM", "mean"), Average_Frequency=("Workout_Frequency_days_per_week", "mean"), Average_Calories=("Calories_Burned_kcal", "mean"), Record_Count=("Workout_Cluster", "count")).round(2))

with tab4:
    st.subheader("Dataset Preview")
    styled_dataframe(df.head(100), height=450)
    st.download_button("Download Dataset", df.to_csv(index=False), "workout_data.csv", "text/csv")

st.caption("Fitbit Calories Prediction & Workout Pattern Clustering")

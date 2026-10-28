
import numpy as np
import pandas as pd
import joblib
import sklearn

from sklearn.ensemble import RandomForestRegressor

print("Scikit-learn version:", sklearn.__version__)

# Generate the same type of sample data
np.random.seed(42)
n = 1000

data = pd.DataFrame({
    "study_hours": np.random.uniform(0, 10, n),
    "attendance": np.random.uniform(50, 100, n),
    "previous_score": np.random.uniform(30, 100, n),
    "sleep_hours": np.random.uniform(4, 10, n),
    "assignments_completed": np.random.randint(0, 11, n)
})

noise = np.random.normal(0, 7, n)

data["exam_score"] = (
    15
    + 3.2 * data["study_hours"]
    + 0.28 * data["attendance"]
    + 0.32 * data["previous_score"]
    + 1.7 * data["sleep_hours"]
    + 1.4 * data["assignments_completed"]
    + noise
).clip(0, 100)

X = data.drop("exam_score", axis=1)
y = data["exam_score"]

# Train using the laptop's installed version
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X, y)

# Replace the old model files
joblib.dump(model, "student_score_model.joblib")
joblib.dump(list(X.columns), "feature_names.joblib")

print("Model retrained and saved successfully!")

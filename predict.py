"""Predict heart disease for one patient with the saved model.

Usage:
    python predict.py                      # runs a built-in example patient
    python predict.py --age 62 --sex 0 --cp 3 --trestbps 160 --chol 164 \
        --fbs 0 --restecg 2 --thalach 145 --exang 0 --oldpeak 6.2 \
        --slope 2 --ca 3 --thal 2

Educational project only - not a medical device.
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).parent / "models" / "heart_disease_model.joblib"
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
EXAMPLE = dict(age=62, sex=0, cp=3, trestbps=160, chol=164, fbs=0, restecg=2,
               thalach=145, exang=0, oldpeak=6.2, slope=2, ca=3, thal=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in FEATURES:
        parser.add_argument(f"--{name}", type=float, default=EXAMPLE[name])
    values = vars(parser.parse_args())

    model = joblib.load(MODEL_PATH)
    patient = pd.DataFrame([values], columns=FEATURES)
    probability = model.predict_proba(patient)[0, 1]
    label = "Heart disease likely" if probability >= 0.5 else "Heart disease unlikely"
    print(f"{label} (estimated probability {probability:.1%})")


if __name__ == "__main__":
    main()

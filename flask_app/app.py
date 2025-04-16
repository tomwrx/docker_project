import sys
import os
import warnings
from flask import Flask, request, render_template

# Ensure the /app folder (where helper_functions is) is in the path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))

from helper_functions.ml_functions import (
    data_preparation,
    clf_model_prediction,
    reg_model_prediction,
    load_model_and_pipe,
)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")

app = Flask(__name__, template_folder="templates")

PIPE_PATH = os.path.join(BASE_DIR, "pipelines")
MODEL_PATH = os.path.join(BASE_DIR, "models")


@app.route("/", methods=["GET", "POST"])
def index():
    """Renders the main index page."""
    return render_template("index.html")


@app.route("/stroke_prediction/", methods=["GET", "POST"])
def stroke_prediction():
    """Renders the stroke prediction form."""
    return render_template("stroke_prediction.html")


@app.route("/hypertension_prediction/", methods=["GET", "POST"])
def hypertension_prediction():
    """Renders the hypertension prediction form."""
    return render_template("hypertension_prediction.html")


@app.route("/glucose_prediction/", methods=["GET", "POST"])
def glucose_prediction():
    """Renders the glucose prediction form."""
    return render_template("glucose_prediction.html")


@app.route("/bmi_prediction/", methods=["GET", "POST"])
def bmi_prediction():
    """Renders the BMI prediction form."""
    return render_template("bmi_prediction.html")


@app.route("/predict/stroke", methods=["POST"])
def predict_stroke() -> str:
    """Returns stroke classification prediction as percentage string."""
    user_input = request.form.to_dict()
    df = data_preparation(user_input)
    pipe, model = load_model_and_pipe(
        PIPE_PATH, "stroke_final_pipeline.pkl", MODEL_PATH, "stroke_catboost.pkl"
    )
    pred = clf_model_prediction(model, pipe, df)
    return f"{(pred[0] * 100):.2f}%"


@app.route("/predict/hypertension", methods=["POST"])
def predict_hypertension() -> str:
    """Returns hypertension classification prediction as percentage string."""
    user_input = request.form.to_dict()
    df = data_preparation(user_input)
    pipe, model = load_model_and_pipe(
        PIPE_PATH,
        "hypertension_final_pipeline.pkl",
        MODEL_PATH,
        "hypertension_xgboost.pkl",
    )
    pred = clf_model_prediction(model, pipe, df)
    return f"{(pred[0] * 100):.2f}%"


@app.route("/predict/glucose", methods=["POST"])
def predict_glucose() -> str:
    """Returns glucose regression prediction as float string."""
    user_input = request.form.to_dict()
    df = data_preparation(user_input)
    pipe, model = load_model_and_pipe(
        PIPE_PATH, "avg_glucose_final_pipeline.pkl", MODEL_PATH, "glucose_lightgbm.pkl"
    )
    pred = reg_model_prediction(model, pipe, df)
    return f"{pred[0]:.2f}"


@app.route("/predict/bmi", methods=["POST"])
def predict_bmi() -> str:
    """Returns BMI regression prediction as float string."""
    user_input = request.form.to_dict()
    df = data_preparation(user_input)
    pipe, model = load_model_and_pipe(
        PIPE_PATH, "bmi_final_pipeline.pkl", MODEL_PATH, "bmi_lightgbm.pkl"
    )
    pred = reg_model_prediction(model, pipe, df)
    return f"{pred[0]:.2f}"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)


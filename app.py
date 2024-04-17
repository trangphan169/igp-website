import os
import flask
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Initialize Flask app
app = flask.Flask(__name__, template_folder="templates")

def root_dir():  # pragma: no cover
    return os.path.abspath(os.path.dirname(__file__))

# Load models and data
MODELS = {}
DATA = {}


def load_models_and_data():
    global MODELS, DATA
    
    root_dir = root_dir()

    with open(root_dir + "Models/xgb_cv_final.pkl", "rb") as f:
        MODELS["clf_individual"] = pickle.load(f)

    with open(root_dir + "Models/gb_cv_compact_joint.pkl", "rb") as f:
        MODELS["clf_joint"] = pickle.load(f)

    with open(root_dir + "Models/knn_regression.pkl", "rb") as f:
        MODELS["knn"] = pickle.load(f)

    DATA["df_macro_mean"] = pd.read_csv(
        root_dir + "Data/df_macro_mean.csv", index_col=0, dtype=np.float64
    )
    DATA["df_macro_std"] = pd.read_csv(
        root_dir + "Data/df_macro_std.csv", index_col=0, dtype=np.float64
    )


load_models_and_data()


# Utility functions
# Function to preprocess individual input
def preprocess_individual_input(input_data: pd.DataFrame):
    # A deep copy of the DataFrame
    numerical_df = input_data.copy()

    # List to keep track of columns to drop at the end
    drop_columns = ["emp_title"]

    # Encode purpose column
    encoder = LabelEncoder()
    numerical_df["purpose"] = encoder.fit_transform(numerical_df["purpose"])

    # Convert 'pub_rec_bankruptcies' and 'pub_rec' features to binary
    numerical_df["pub_rec_bankruptcies"] = np.where(
        numerical_df["pub_rec_bankruptcies"] > 0, 1, 0
    )
    numerical_df["pub_rec"] = np.where(numerical_df["pub_rec"] > 0, 1, 0)

    # Dictionaries for mapping categorical values to numerical ones
    term_map = {"36 months": 1, "60 months": 2}
    home_map = {"MORTGAGE": 4, "RENT": 3, "OWN": 5, "ANY": 2, "OTHER": 1, "NONE": 0}
    ver_status_map = {"Source Verified": 2, "Verified": 1, "Not Verified": 0}

    # Replaced categorical values with numerical values
    numerical_df.replace(
        {
            "term": term_map,
            "home_ownership": home_map,
            "verification_status": ver_status_map,
        },
        inplace=True,
    )

    # Extracting first 2 digits of zip code
    numerical_df["zip_2"] = numerical_df["zip_code"].str[:2]
    drop_columns.append("zip_code")

    # Drop rows with missing values
    numerical_df = numerical_df.dropna(axis=0)

    # Convert date columns to datetime & create new feature 'cred_history'(RFE)
    numerical_df["earliest_cr_line"] = pd.to_datetime(
        numerical_df["earliest_cr_line"], infer_datetime_format=True
    )
    credit_history = pd.Timestamp.today().normalize() - numerical_df["earliest_cr_line"]
    numerical_df["credit_history"] = credit_history.dt.days
    drop_columns.extend(["earliest_cr_line"])

    # RFE 'credit_line_ratio' & 'balance_annual_income'
    numerical_df["credit_line_ratio"] = (
        numerical_df["open_acc"] / numerical_df["total_acc"]
    )
    numerical_df["balance_annual_income"] = (
        numerical_df["loan_amnt"] / numerical_df["annual_inc"]
    )
    drop_columns.extend(["open_acc", "total_acc"])

    # Log transform 'annual_inc'
    numerical_df["annual_inc"] += 1
    numerical_df["annual_inc"] = np.log(numerical_df["annual_inc"])

    # RFE 'installment_amnt_ratio'
    numerical_df["installment_amnt_ratio"] = (
        numerical_df["installment"] / numerical_df["loan_amnt"]
    )
    drop_columns.extend(["installment", "loan_amnt"])

    # Drop features listed in col_drop_list
    numerical_df = numerical_df.drop(columns=drop_columns)

    return numerical_df


# Function to process joint input
def preprocess_joint_input(input_data: pd.DataFrame):
    # A deep copy of the DataFrame
    numerical_df = input_data.copy()

    # List to keep track of columns to drop at the end
    drop_columns = ["emp_title"]

    # Encode purpose column
    encoder = LabelEncoder()
    numerical_df["purpose"] = encoder.fit_transform(numerical_df["purpose"])


def scale_individual_data(df_unscaled: pd.DataFrame):
    scaled_df = []

    zip = int(df_unscaled.loc[0, "zip_2"])

    # Scaling each feature for the current zip code
    for col in DATA["df_macro_mean"].columns:
        mean = DATA["df_macro_mean"].loc[zip, col]
        std = DATA["df_macro_std"].loc[zip, col]
        df_unscaled[col] = (float(df_unscaled[col]) - float(mean)) / float(std)

    scaled_df.append(df_unscaled)

    # Concatenate scaled df
    df_scaled = pd.concat(scaled_df)

    drop_features = [
        "pub_rec",
        "pub_rec_bankruptcies",
        "emp_length",
        "purpose",
        "revol_bal",
        "grade",
        "int_rate",
        "zip_2",
    ]

    df_scaled = df_scaled.drop(columns=drop_features)

    return df_scaled


def get_sub_grade(fico_score):
    return MODELS["knn"].predict(np.array(fico_score).reshape(1, -1))[0]


# Route definitions
@app.route("/")
def main():
    return flask.render_template("index.html")


@app.route("/individual-loan")
def report():
    return flask.render_template("individual-loan.html")


@app.route("/joint-loan")
def jointreport():
    return flask.render_template("joint-loan.html")


@app.route("/more-information")
def more_information():
    return flask.render_template("EDA.html")


@app.route("/Individual", methods=["POST"])
def individual():
    input_data = flask.request.form.to_dict()

    # Extract features from input
    zip_code = input_data["zip_code"]
    emp_length = float(input_data["emp_length"])
    emp_title = input_data["emp_title"]
    annual_inc = float(input_data["annual_inc"])
    fico_avg_score = float(input_data["fico_avg_score"])
    dti = float(input_data["dti"])
    earliest_cr_line = input_data["earliest_cr_line"]
    open_acc = float(input_data["open_acc"])
    total_acc = float(input_data["total_acc"])
    revol_util = float(input_data["revol_util"])
    revol_bal = float(input_data["revol_bal"])
    mort_acc = float(input_data["mort_acc"])
    home_ownership = input_data["home_ownership"]
    purpose = input_data["purpose"]
    pub_rec = float(input_data["pub_rec"])
    pub_rec_bankruptcies = float(input_data["pub_rec_bankruptcies"])
    loan_amnt = float(input_data["loan_amnt"])
    int_rate = float(input_data["int_rate"])
    term = input_data["term"]
    installment = float(input_data["installment"])
    verification_status = input_data["verification_status"]

    # Predict sub-grade
    sub_grade = get_sub_grade([[fico_avg_score]])
    grade = round(sub_grade / 5) + 1

    # Define input
    df_input = pd.DataFrame(
        {
            "loan_amnt": [loan_amnt],
            "term": [term],
            "int_rate": [int_rate],
            "installment": [installment],
            "grade": [grade],
            "sub_grade": [sub_grade],
            "emp_title": [emp_title],
            "emp_length": [emp_length],
            "home_ownership": [home_ownership],
            "annual_inc": [annual_inc],
            "verification_status": [verification_status],
            "purpose": [purpose],
            "dti": [dti],
            "earliest_cr_line": [earliest_cr_line],
            "open_acc": [open_acc],
            "total_acc": [total_acc],
            "revol_bal": [revol_bal],
            "revol_util": [revol_util],
            "pub_rec": [pub_rec],
            "pub_rec_bankruptcies": [pub_rec_bankruptcies],
            "mort_acc": [mort_acc],
            "zip_code": [zip_code],
            "fico_avg_score": [fico_avg_score],
        }
    )

    # Preprocess input
    df_processed = preprocess_individual_input(df_input)

    # # Scale input
    df_scaled = scale_individual_data(df_processed).values

    # Check debt-to-income and revolving utilization conditions
    if (
        dti > 50
        or fico_avg_score < 400
        or revol_util >= 75
        or pub_rec_bankruptcies > 2
        or pub_rec > 5
        or verification_status == "Not Verified"
    ):
        res = "Loan Denied"
    else:
        # Make prediction
        pred = MODELS["clf_individual"].predict(df_scaled.reshape(1, -1))[0]
        res = "Congratulations! Approved!" if pred == 0 else "Loan Denied"

    return flask.render_template("result.html", result=res)


@app.route("/Joint", methods=["GET", "POST"])
def joint():
    # input_data = flask.request.form.to_dict()

    # # Extract features from input
    # zip_code = input_data["zip_code"]
    # emp_length = float(input_data["emp_length"])
    # emp_title = input_data["emp_title"]
    # annual_inc = float(input_data["annual_inc"])
    # fico_avg_score = float(input_data["fico_avg_score"])
    # dti = float(input_data["dti"])
    # earliest_cr_line = input_data["earliest_cr_line"]
    # open_acc = float(input_data["open_acc"])
    # total_acc = float(input_data["total_acc"])
    # revol_util = float(input_data["revol_util"])
    # revol_bal = float(input_data["revol_bal"])
    # mort_acc = float(input_data["mort_acc"])
    # home_ownership = input_data["home_ownership"]
    # purpose = input_data["purpose"]
    # pub_rec = float(input_data["pub_rec"])
    # pub_rec_bankruptcies = float(input_data["pub_rec_bankruptcies"])
    # loan_amnt = float(input_data["loan_amnt"])
    # int_rate = float(input_data["int_rate"])
    # term = input_data["term"]
    # installment = float(input_data["installment"])
    # verification_status = input_data["verification_status"]

    # # Predict sub-grade
    # sub_grade = get_sub_grade([[fico_avg_score]])
    # grade = round(sub_grade / 5) + 1

    # # Define input
    # df_input = pd.DataFrame(
    #     {
    #         "loan_amnt": [loan_amnt],
    #         "term": [term],
    #         "int_rate": [int_rate],
    #         "installment": [installment],
    #         "grade": [grade],
    #         "sub_grade": [sub_grade],
    #         "emp_title": [emp_title],
    #         "emp_length": [emp_length],
    #         "home_ownership": [home_ownership],
    #         "annual_inc": [annual_inc],
    #         "verification_status": [verification_status],
    #         "purpose": [purpose],
    #         "dti": [dti],
    #         "earliest_cr_line": [earliest_cr_line],
    #         "open_acc": [open_acc],
    #         "total_acc": [total_acc],
    #         "revol_bal": [revol_bal],
    #         "revol_util": [revol_util],
    #         "pub_rec": [pub_rec],
    #         "pub_rec_bankruptcies": [pub_rec_bankruptcies],
    #         "mort_acc": [mort_acc],
    #         "zip_code": [zip_code],
    #         "fico_avg_score": [fico_avg_score],
    #     }
    # )

    # # Preprocess input
    # df_processed = preprocess_input(df_input)

    # # # Scale input
    # df_scaled = cust_scaler(df_processed).values

    # # Check debt-to-income and revolving utilization conditions
    # # Make prediction
    # pred = MODELS["clf_joint"].predict(df_scaled.reshape(1, -1))[0]
    # res = "Congratulations! Approved!" if pred == 0 else "Loan Denied"

    # return flask.render_template("result.html", result=res)
    pass


# if __name__ == "__main__":
#     app.run()

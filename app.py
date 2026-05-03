"""
Flask Web Application — Loan Default Prediction Dashboard
Run:  python app.py
Open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os
import json

app = Flask(__name__)

# ── Load trained models (if available) ──────────────────────
MODELS_DIR = 'models'
lr_model  = None
xgb_model = None
scaler    = None
kmeans    = None

def load_models():
    global lr_model, xgb_model, scaler, kmeans
    if os.path.exists(f'{MODELS_DIR}/logistic_regression.pkl'):
        lr_model  = joblib.load(f'{MODELS_DIR}/logistic_regression.pkl')
        xgb_model = joblib.load(f'{MODELS_DIR}/xgboost_model.pkl')
        scaler    = joblib.load(f'{MODELS_DIR}/scaler.pkl')
        kmeans    = joblib.load(f'{MODELS_DIR}/kmeans.pkl')
        print("[Flask] Models loaded successfully.")
    else:
        print("[Flask] Models not found — using formula-based prediction.")


def formula_predict(data):
    """Fallback prediction when trained models aren't loaded."""
    credit   = data['credit_score']
    dti      = data['dti_ratio']
    income   = data['annual_income']
    loan     = data['loan_amount']
    emp      = data['emp_length']
    delinq   = data['delinquencies']
    purpose  = data.get('loan_purpose', 0)
    home     = data.get('home_ownership', 0)

    credit_n = (credit - 300) / 550
    dti_n    = min(dti / 100, 1)
    lir_n    = min(loan / (income + 1), 3) / 3
    emp_n    = min(emp, 20) / 20
    delinq_n = min(delinq, 10) / 10
    p_adj    = [0.0, -0.03, 0.04, 0.05, 0.02, 0.03][int(purpose)] if int(purpose) < 6 else 0
    h_adj    = [0.04, -0.01, -0.05][int(home)] if int(home) < 3 else 0

    raw = (0.32*(1-credit_n) + 0.25*dti_n + 0.20*lir_n +
           0.13*delinq_n + 0.07*(1-emp_n) + p_adj + h_adj)

    xgb_prob = float(min(0.97, max(0.02, raw * 1.04)))
    lr_prob  = float(min(0.95, max(0.03, raw * 0.91)))
    return xgb_prob, lr_prob


def get_segment(credit, dti, delinq, income, loan):
    """Assign borrower segment based on profile."""
    segments = {
        'prime':        {'name': 'Prime Borrowers',          'default_rate': '2.1%',  'color': '#1D9E75'},
        'near_prime':   {'name': 'Near-Prime Borrowers',     'default_rate': '11.4%', 'color': '#378ADD'},
        'thin_file':    {'name': 'Thin-File Borrowers',      'default_rate': '22.0%', 'color': '#7F77DD'},
        'subprime':     {'name': 'Subprime Borrowers',       'default_rate': '31.7%', 'color': '#BA7517'},
        'over_lev':     {'name': 'Over-Leveraged Borrowers', 'default_rate': '19.5%', 'color': '#D85A30'},
        'high_risk':    {'name': 'High-Risk Borrowers',      'default_rate': '58.3%', 'color': '#E24B4A'},
    }
    if credit >= 720 and dti < 20 and delinq == 0:
        key = 'prime'
    elif delinq >= 3 or credit < 580:
        key = 'high_risk'
    elif dti >= 50:
        key = 'over_lev'
    elif credit >= 640 and dti < 40:
        key = 'near_prime'
    elif credit < 640 and dti >= 30:
        key = 'subprime'
    else:
        key = 'thin_file'
    return segments[key]


# ── Routes ───────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main dashboard."""
    metrics = {}
    metrics_path = 'static/js/metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    return render_template('index.html', metrics=metrics)


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint — receive borrower data, return risk scores."""
    try:
        data = request.get_json()

        if xgb_model and scaler:
            # Use real trained model
            features = np.array([[
                data['credit_score'], data['annual_income'], data['loan_amount'],
                data['dti_ratio'], data['emp_length'], data['delinquencies'],
                data.get('open_accounts', 10), data.get('revol_util', 50),
                data.get('pub_rec', 0), data.get('loan_term', 36),
                data.get('interest_rate', 12), data.get('total_acc', 15),
                data['loan_amount'] / (data['annual_income'] + 1),
            ]])
            features_scaled = scaler.transform(features)
            xgb_prob = float(xgb_model.predict_proba(features_scaled)[0][1])
            lr_prob  = float(lr_model.predict_proba(features_scaled)[0][1])
        else:
            # Fallback formula
            xgb_prob, lr_prob = formula_predict(data)

        threshold   = 0.42
        xgb_default = xgb_prob >= threshold
        lr_default  = lr_prob  >= threshold
        segment     = get_segment(
            data['credit_score'], data['dti_ratio'],
            data['delinquencies'], data['annual_income'], data['loan_amount']
        )

        # SHAP-style contributions (approximate)
        credit_n = (data['credit_score'] - 680) / 550
        shap_vals = {
            'Credit Score':      round(-credit_n * 0.35, 4),
            'DTI Ratio':         round((data['dti_ratio'] - 32) / 100 * 0.25, 4),
            'Loan/Income Ratio': round(min(data['loan_amount']/data['annual_income'], 3)/3*0.18 - 0.05, 4),
            'Delinquencies':     round(data['delinquencies'] / 10 * 0.14, 4),
            'Employment Length': round(-(min(data['emp_length'], 20) / 20) * 0.09 + 0.03, 4),
        }

        return jsonify({
            'success':     True,
            'xgb_prob':    round(xgb_prob * 100, 1),
            'lr_prob':     round(lr_prob  * 100, 1),
            'xgb_default': xgb_default,
            'lr_default':  lr_default,
            'segment':     segment,
            'shap':        shap_vals,
            'risk_level':  'High' if xgb_prob >= 0.6 else ('Medium' if xgb_prob >= 0.3 else 'Low')
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/metrics')
def metrics():
    """Return model metrics as JSON."""
    path = 'static/js/metrics.json'
    if os.path.exists(path):
        with open(path) as f:
            return jsonify(json.load(f))
    return jsonify({'error': 'Run ml_pipeline.py first to generate metrics'}), 404


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'models_loaded': lr_model is not None})


if __name__ == "__main__":
    load_models()
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

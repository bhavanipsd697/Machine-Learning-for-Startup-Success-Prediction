# app.py
from flask import Flask, render_template, request
import joblib
import json

app = Flask(__name__)
model=joblib.load('random_forest_model.pkl')

# Build template features at startup (recreate preprocessing from training)
import pandas as pd
template_row = None
feature_columns = None
try:
    data = pd.read_csv('startup data.csv') 

    # State mapping
    data['State'] = 'other'
    data.loc[(data['state_code'] == 'CA'), 'State'] = 'CA'
    data.loc[(data['state_code'] == 'NY'), 'State'] = 'NY'
    data.loc[(data['state_code'] == 'MA'), 'State'] = 'MA'
    data.loc[(data['state_code'] == 'TX'), 'State'] = 'TX'
    data.loc[(data['state_code'] == 'WA'), 'State'] = 'WA'

    # Category mapping
    data['category'] = 'other'
    cats = ['software','web','mobile','enterprise','advertising','games_video','semiconductor','network_hosting','biotech','hardware']
    for cat in cats:
        if 'category_code' in data.columns:
            data.loc[(data['category_code'] == cat), 'category'] = cat

    # Drop identifier/leaky columns (same as training)
    drop_cols = ['id','name','status','Unnamed: 0','state_code.1','founded_at','closed_at','category_code','object_id']
    drop_cols = [c for c in drop_cols if c in data.columns]
    X = data.drop(columns=drop_cols).copy()

    # One-hot encode low-cardinality categoricals
    cat_cols = [c for c in ['State','category'] if c in X.columns]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, dummy_na=False)

    # Normalize boolean-like columns
    bool_cols = ['has_VC','has_angel','has_roundA','has_roundB','has_roundC','has_roundD','is_CA','is_NY','is_MA','is_TX','is_otherstate']
    for c in bool_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0).astype(int)

    # Impute numeric columns with median (coerce to float to allow float medians)
    num_cols = X.select_dtypes(include=['number']).columns
    X[num_cols] = X[num_cols].astype(float).fillna(X[num_cols].median())

    # Convert remaining low-cardinality non-numeric to dummies, drop very high-cardinality
    non_num = X.select_dtypes(exclude=['number']).columns.tolist()
    for c in non_num:
        if X[c].nunique() <= 50:
            X = pd.get_dummies(X, columns=[c], dummy_na=False)
        else:
            X = X.drop(columns=[c])

    numeric_medians = X[num_cols].median().to_dict()

    # Align feature columns to exactly what the trained model expects (if available)
    expected = list(getattr(model, 'feature_names_in_', []))
    if expected:
        feature_columns = expected
        # build template row from expected columns, filling medians where available
        template_row = pd.DataFrame([{c: numeric_medians.get(c, 0) for c in feature_columns}])
    else:
        feature_columns = X.columns.tolist()
        import numpy as np
        template_row = pd.DataFrame([{c: np.nan for c in feature_columns}])
        for k,v in numeric_medians.items():
            if k in template_row.columns:
                template_row.at[0, k] = v
        template_row = template_row.fillna(0)

except Exception as e:
    import traceback
    print("Warning: failed to build template features:", e)
    traceback.print_exc()
    feature_columns = None
    template_row = None

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    age_first_funding_year = float(request.form['age_first_funding_year'])
    age_last_funding_year = float(request.form['age_last_funding_year'])
    age_first_milestone_year = float(request.form['age_first_milestone_year'])
    age_last_milestone_year = float(request.form['age_last_milestone_year'])
    relationships = float(request.form['relationships'])
    funding_rounds = float(request.form['funding_rounds'])
    funding_total_usd = float(request.form['funding_total_usd'])
    milestones = float(request.form['milestones'])
    avg_participants = float(request.form['avg_participants'])

    # Prepare an input row using the template features built at app startup
    if template_row is None or feature_columns is None:
        return render_template('result.html', result="Server error: model metadata not available")

    input_df = template_row.copy()

    # Fill known numeric fields if they exist in the trained feature set
    field_map = {
        'age_first_funding_year': age_first_funding_year,
        'age_last_funding_year': age_last_funding_year,
        'age_first_milestone_year': age_first_milestone_year,
        'age_last_milestone_year': age_last_milestone_year,
        'relationships': relationships,
        'funding_rounds': funding_rounds,
        'funding_total_usd': funding_total_usd,
        'milestones': milestones,
        'avg_participants': avg_participants
    }

    for fname, val in field_map.items():
        if fname in input_df.columns:
            input_df.loc[0, fname] = val
        else:
            # If the model doesn't have the raw field (it may have been renamed/encoded), ignore it
            pass

    # Ensure columns are ordered exactly as the model was trained on
    input_df = input_df[feature_columns]

    # Get probabilities and prediction
    proba = model.predict_proba(input_df.values)[0]
    prediction = int(model.predict(input_df.values)[0])

    # Map the predicted label to a meaningful output
    result = 'Acquired' if prediction == 1 else 'Closed'

    # Build processed features dump (JSON formatted)
    processed_features = json.dumps(input_df.iloc[0].to_dict(), indent=2)

    # Render the prediction result with additional debug info
    return render_template('result.html', result=result, proba=proba, processed_features=processed_features)

if __name__ == '__main__':
    app.run(debug=True)
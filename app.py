from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import plotly
import plotly.express as px
import json
import io
import base64
import logging
from scipy.sparse import issparse

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

logging.basicConfig(filename='logs/app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Component definitions with default parameters
IMPUTERS = {
    'MeanImputer': {'imputer': SimpleImputer, 'params': {'strategy': 'mean'}},
    'MedianImputer': {'imputer': SimpleImputer, 'params': {'strategy': 'median'}},
    'KNNImputer': {'imputer': KNNImputer, 'params': {'n_neighbors': 5}}
}

ENCODERS = {
    'LabelEncoder': {'encoder': LabelEncoder, 'params': {}},
    'OneHotEncoder': {'encoder': OneHotEncoder, 'params': {'sparse_output': False, 'drop': 'first'}}
}

PREPROCESSORS = {
    'StandardScaler': {'preprocessor': StandardScaler, 'params': {}},
    'MinMaxScaler': {'preprocessor': MinMaxScaler, 'params': {'feature_range': (0, 1)}}
}

FEATURE_SELECTORS = {
    'SelectKBest': {'selector': SelectKBest, 'params': {'k': 10, 'score_func': f_classif}}
}

MODELS = {
    'classification': {
        'LogisticRegression': {'type': 'sklearn', 'model': LogisticRegression, 'params': {'C': 1.0, 'max_iter': 1000}},
        'DecisionTree': {'type': 'sklearn', 'model': DecisionTreeClassifier, 'params': {'max_depth': 5}},
        'RandomForest': {'type': 'sklearn', 'model': RandomForestClassifier, 'params': {'n_estimators': 100}},
        'NeuralNetwork': {'type': 'keras', 'model': lambda params: create_keras_model(params, 'classification'), 
                         'params': {'layers': [64, 32], 'dropout': 0.2, 'epochs': 20}}
    },
    'regression': {
        'LinearRegression': {'type': 'sklearn', 'model': LinearRegression, 'params': {}},
        'DecisionTree': {'type': 'sklearn', 'model': DecisionTreeRegressor, 'params': {'max_depth': 5}},
        'RandomForest': {'type': 'sklearn', 'model': RandomForestRegressor, 'params': {'n_estimators': 100}},
        'NeuralNetwork': {'type': 'keras', 'model': lambda params: create_keras_model(params, 'regression'), 
                         'params': {'layers': [64, 32], 'dropout': 0.2, 'epochs': 20}}
    }
}

def create_keras_model(params, task_type, n_outputs=1):
    model = Sequential()
    for i, units in enumerate(params['layers']):
        if i == 0:
            model.add(Dense(units, activation='relu', input_shape=(session['data_shape'][1] - n_outputs,)))
        else:
            model.add(Dense(units, activation='relu'))
        if params.get('dropout', 0) > 0:
            model.add(Dropout(params['dropout']))
    model.add(Dense(n_outputs, activation='softmax' if task_type == 'classification' and n_outputs > 1 else 'sigmoid' if task_type == 'classification' else 'linear'))
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy' if task_type == 'classification' and n_outputs > 1 else 'binary_crossentropy' if task_type == 'classification' else 'mse',
                  metrics=['accuracy'] if task_type == 'classification' else ['mse'])
    return model

@app.route('/')
def index():
    session.clear()
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_data():
    if 'file' not in request.files:
        logging.error("No file uploaded")
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    try:
        df = pd.read_csv(file)
        session['data'] = df.to_json()
        session['data_shape'] = df.shape
        
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        summary = {
            'head': df.head().to_dict(),
            'dtypes': dtypes,
            'numeric': df.select_dtypes(include=np.number).columns.tolist(),
            'categorical': df.select_dtypes(exclude=np.number).columns.tolist(),
            'missing': df.isnull().sum().to_dict(),
            'stats': df.describe().to_dict(),
            'shape': df.shape
        }
        session['summary'] = summary
        
        numeric_df = df.select_dtypes(include=np.number)
        heatmap = None
        if not numeric_df.empty:
            plt.figure(figsize=(10, 8))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            heatmap = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
        
        logging.info(f"Data uploaded: {df.shape}")
        return jsonify({'summary': summary, 'heatmap': heatmap})
    except Exception as e:
        logging.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/get_columns', methods=['GET'])
def get_columns():
    if 'summary' not in session:
        return jsonify({'error': 'No data uploaded'}), 400
    return jsonify({'columns': list(session['summary']['head'].keys())})

@app.route('/set_task', methods=['POST'])
def set_task():
    task_type = request.json.get('task_type')
    targets = request.json.get('targets')
    if not targets or not all(target in session['summary']['head'] for target in targets):
        logging.error("Invalid target(s) specified")
        return jsonify({'error': 'Invalid target(s) specified'}), 400
    session['task_type'] = task_type
    session['targets'] = targets
    logging.info(f"Task set: {task_type}, Targets: {targets}")
    return jsonify({'message': f'Task set to {task_type} with targets {targets}'})

@app.route('/train', methods=['POST'])
def train_model():
    if 'data' not in session or 'task_type' not in session or 'targets' not in session:
        return jsonify({'error': 'Data, task type, or targets not set'}), 400
    
    pipeline = request.json.get('pipeline', [])
    df = pd.read_json(session['data'])
    X = df.drop(columns=session['targets'])
    y = df[session['targets']].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_indices = [X.columns.get_loc(col) for col in categorical_cols if col not in session['targets']]
    n_targets = len(session['targets'])
    
    try:
        for step in pipeline:
            step_type = step['type']
            params = step.get('params', {})
            logging.info(f"Processing step: {step_type} with params {params}")

            if step_type in IMPUTERS:
                imputer_info = IMPUTERS[step_type]
                imputer = imputer_info['imputer'](**{**imputer_info['params'], **params})
                X_train.iloc[:, [i for i in range(X_train.shape[1]) if i not in cat_indices]] = \
                    imputer.fit_transform(X_train.iloc[:, [i for i in range(X_train.shape[1]) if i not in cat_indices]])
                X_test.iloc[:, [i for i in range(X_test.shape[1]) if i not in cat_indices]] = \
                    imputer.transform(X_test.iloc[:, [i for i in range(X_test.shape[1]) if i not in cat_indices]])

            elif step_type in ENCODERS:
                encoder_info = ENCODERS[step_type]
                encoder_params = {k: v for k, v in {**encoder_info['params'], **params}.items() if k != 'target'}
                encoder = encoder_info['encoder'](**encoder_params)
                if step_type == 'LabelEncoder' and session['task_type'] == 'classification' and params.get('target', False):
                    if n_targets == 1:
                        y_train = encoder.fit_transform(y_train)
                        y_test = encoder.transform(y_test)
                    else:
                        y_train = np.array([encoder.fit_transform(y_train[:, i]) for i in range(n_targets)]).T
                        y_test = np.array([encoder.transform(y_test[:, i]) for i in range(n_targets)]).T
                elif cat_indices:
                    if step_type == 'OneHotEncoder':
                        X_train_encoded_raw = encoder.fit_transform(X_train[categorical_cols])
                        X_test_encoded_raw = encoder.transform(X_test[categorical_cols])
                        X_train_encoded = X_train_encoded_raw.toarray() if issparse(X_train_encoded_raw) else X_train_encoded_raw
                        X_test_encoded = X_test_encoded_raw.toarray() if issparse(X_test_encoded_raw) else X_test_encoded_raw
                        X_train_encoded_df = pd.DataFrame(X_train_encoded,
                                                         columns=encoder.get_feature_names_out(categorical_cols),
                                                         index=X_train.index)
                        X_test_encoded_df = pd.DataFrame(X_test_encoded,
                                                        columns=encoder.get_feature_names_out(categorical_cols),
                                                        index=X_test.index)
                        X_train = X_train.drop(columns=categorical_cols).join(X_train_encoded_df)
                        X_test = X_test.drop(columns=categorical_cols).join(X_test_encoded_df)
                    else:
                        for col in categorical_cols:
                            X_train[col] = encoder.fit_transform(X_train[col])
                            X_test[col] = encoder.transform(X_test[col])

            elif step_type in PREPROCESSORS:
                preprocessor_info = PREPROCESSORS[step_type]
                preprocessor = preprocessor_info['preprocessor'](**{**preprocessor_info['params'], **params})
                X_train = pd.DataFrame(preprocessor.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
                X_test = pd.DataFrame(preprocessor.transform(X_test), columns=X_test.columns, index=X_test.index)

            elif step_type in FEATURE_SELECTORS:
                selector_info = FEATURE_SELECTORS[step_type]
                selector = selector_info['selector'](**{**selector_info['params'], 
                                                       'score_func': f_classif if session['task_type'] == 'classification' else f_regression, 
                                                       **params})
                X_train = pd.DataFrame(selector.fit_transform(X_train, y_train[:, 0] if n_targets > 1 else y_train), 
                                      index=X_train.index)
                X_test = pd.DataFrame(selector.transform(X_test), index=X_test.index)

            elif step_type in MODELS[session['task_type']]:
                model_info = MODELS[session['task_type']][step_type]
                model_params = {**model_info['params'], **params}
                if n_targets > 1 and model_info['type'] == 'sklearn':
                    base_model = model_info['model'](**model_params)
                    model = MultiOutputClassifier(base_model) if session['task_type'] == 'classification' else MultiOutputRegressor(base_model)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    cv_scores = None
                elif model_info['type'] == 'sklearn':
                    model = model_info['model'](**model_params)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                else:  # Keras
                    model = create_keras_model(model_params, session['task_type'], n_targets)
                    if session['task_type'] == 'classification' and n_targets == 1:
                        model.fit(X_train, y_train, epochs=model_params['epochs'], batch_size=32, verbose=0, validation_split=0.2)
                        y_pred = model.predict(X_test).flatten()
                        y_pred = (y_pred > 0.5).astype(int)
                    else:
                        model.fit(X_train, y_train, epochs=model_params['epochs'], batch_size=32, verbose=0, validation_split=0.2)
                        y_pred = model.predict(X_test)
                        if session['task_type'] == 'classification':
                            y_pred = np.argmax(y_pred, axis=1) if n_targets == 1 else np.argmax(y_pred, axis=1).reshape(-1, n_targets)
                    cv_scores = None

        metrics = {}
        if session['task_type'] == 'classification':
            if n_targets == 1:
                n_classes = len(np.unique(y_test))
                average = 'binary' if n_classes == 2 else 'weighted'
                metrics.update({
                    'accuracy': float(accuracy_score(y_test, y_pred)),
                    'precision': float(precision_score(y_test, y_pred, average=average)),
                    'recall': float(recall_score(y_test, y_pred, average=average))
                })
            else:
                metrics.update({
                    f'accuracy_target_{i}': float(accuracy_score(y_test[:, i], y_pred[:, i])) for i in range(n_targets)
                })
                for i in range(n_targets):
                    n_classes = len(np.unique(y_test[:, i]))
                    average = 'binary' if n_classes == 2 else 'weighted'
                    metrics[f'precision_target_{i}'] = float(precision_score(y_test[:, i], y_pred[:, i], average=average))
                    metrics[f'recall_target_{i}'] = float(recall_score(y_test[:, i], y_pred[:, i], average=average))
        else:
            if n_targets == 1:
                metrics.update({
                    'mse': float(mean_squared_error(y_test, y_pred)),
                    'r2': float(r2_score(y_test, y_pred))
                })
            else:
                metrics.update({
                    f'mse_target_{i}': float(mean_squared_error(y_test[:, i], y_pred[:, i])) for i in range(n_targets)
                })
                metrics.update({
                    f'r2_target_{i}': float(r2_score(y_test[:, i], y_pred[:, i])) for i in range(n_targets)
                })

        if cv_scores is not None and n_targets == 1:
            metrics['cv_mean'] = float(cv_scores.mean())
            metrics['cv_std'] = float(cv_scores.std())

        # Ensure consistent lengths for plotting
        y_test_plot = y_test[:, 0] if n_targets > 1 else y_test
        y_pred_plot = y_pred[:, 0] if n_targets > 1 and y_pred.ndim > 1 else y_pred
        fig = px.scatter(x=y_test_plot.flatten(), 
                        y=y_pred_plot.flatten(), 
                        labels={'x': 'Actual', 'y': 'Predicted'},
                        title=f'Actual vs Predicted (Target 0)' if n_targets > 1 else 'Actual vs Predicted', 
                        trendline='ols')
        plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        logging.info(f"Training completed: {metrics}")
        return jsonify({'metrics': metrics, 'plot': plot_json})

    except Exception as e:
        logging.error(f"Training error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/save_pipeline', methods=['POST'])
def save_pipeline():
    pipeline = request.json.get('pipeline')
    session['pipeline'] = pipeline
    logging.info("Pipeline saved")
    return jsonify({'message': 'Pipeline saved'})

@app.route('/load_pipeline', methods=['GET'])
def load_pipeline():
    if 'pipeline' not in session:
        return jsonify({'error': 'No pipeline saved'}), 400
    return jsonify({'pipeline': session['pipeline']})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless server mode without GUI
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

app = Flask(__name__)

# Load 600-record delivery logistics dataset
dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'delivery_dataset.csv')
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

df = pd.read_csv(dataset_path)

# Variables: X = Distance (km), y = Delivery Time (min)
X = df[['distance_km']].values
y = df['delivery_time_min'].values

# Train Simple Linear Regression model using Scikit-Learn
model = LinearRegression()
model.fit(X, y)

slope = float(model.coef_[0])
intercept = float(model.intercept_)
r_squared = float(model.score(X, y))

def generate_plot(predicted_distance=None, predicted_time=None):
    """Generates scatter plot with fitted regression line and highlighted prediction point."""
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=130)

    # Scatter plot of historical data
    ax.scatter(df['distance_km'], df['delivery_time_min'],
               alpha=0.35, color='#4cc9f0', edgecolors='none', s=35,
               label=f'Historical Deliveries ({len(df)} records)')

    # Linear regression line fitted by Scikit-Learn
    x_line = np.linspace(df['distance_km'].min(), df['distance_km'].max(), 100).reshape(-1, 1)
    y_line = model.predict(x_line)
    ax.plot(x_line, y_line, color='#e94560', linewidth=2.5,
            label=f'Regression Line: y = {slope:.4f}x + {intercept:.2f} (R² = {r_squared:.4f})')

    # Highlight prediction point if active
    if predicted_distance is not None and predicted_time is not None:
        ax.scatter([predicted_distance], [predicted_time], color='#10b981', s=160, zorder=6,
                   edgecolors='#ffffff', linewidths=2.5,
                   label=f'Current Prediction: {predicted_distance} km → {predicted_time} min')
        ax.axvline(x=predicted_distance, color='#10b981', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.axhline(y=predicted_time, color='#10b981', linestyle=':', linewidth=1.5, alpha=0.7)

    # Visual styling with dark theme
    ax.set_xlabel('Travel Distance (km) [Independent Variable X]', fontsize=11, color='#e2e8f0', labelpad=10)
    ax.set_ylabel('Delivery Time (min) [Dependent Variable Y]', fontsize=11, color='#e2e8f0', labelpad=10)
    ax.set_title('Linear Regression Model: Travel Distance vs Delivery Time', fontsize=13, fontweight='bold', color='#f1f5f9', pad=14)
    ax.legend(fontsize=9.5, loc='upper left', framealpha=0.85, facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0')
    ax.set_facecolor('#131b2e')
    fig.patch.set_facecolor('#0b0f19')
    ax.tick_params(colors='#94a3b8')
    ax.spines['bottom'].set_color('#334155')
    ax.spines['left'].set_color('#334155')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.15, color='#94a3b8', linestyle='--')

    # Stream to in-memory Base64 image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return plot_base64

# ================= NAVIGATION ROUTES =================

# 1. Home
@app.route('/')
def home():
    return render_template('home.html')

# 2. Machine Learning -> Concepts
@app.route('/ml/concepts')
def ml_concepts():
    return render_template('ml_concepts.html')

# 3. Machine Learning -> Types of ML
@app.route('/ml/types')
def ml_types():
    return render_template('ml_types.html')

# 4. Machine Learning -> Use Cases (4 independent pages)
@app.route('/ml/use-cases/1')
@app.route('/ml/use-cases/use-case-1')
def use_case_1():
    return render_template('use_case_1.html')

@app.route('/ml/use-cases/2')
@app.route('/ml/use-cases/use-case-2')
def use_case_2():
    return render_template('use_case_2.html')

@app.route('/ml/use-cases/3')
@app.route('/ml/use-cases/use-case-3')
def use_case_3():
    return render_template('use_case_3.html')

@app.route('/ml/use-cases/4')
@app.route('/ml/use-cases/use-case-4')
def use_case_4():
    return render_template('use_case_4.html')

# 5. Supervised -> Linear Regression -> Concepts
@app.route('/supervised/linear-regression/concepts')
@app.route('/ml/supervised/lr/concepts')
def lr_concepts():
    return render_template('lr_concepts.html')

# 6. Supervised -> Linear Regression -> Application
@app.route('/supervised/linear-regression/application', methods=['GET', 'POST'])
@app.route('/ml/supervised/lr/application', methods=['GET', 'POST'])
def lr_application():
    prediction = None
    prediction_hours = None
    distance_input = None
    error = None

    if request.method == 'POST':
        distance_input = request.form.get('distance', '').strip()
        try:
            distance_val = float(distance_input)
            if distance_val <= 0:
                error = 'Please enter a positive distance value greater than 0 km.'
                plot_url = generate_plot()
            else:
                pred = model.predict([[distance_val]])[0]
                prediction = round(float(pred), 2)
                hrs = int(prediction // 60)
                mins = int(prediction % 60)
                prediction_hours = f"{hrs} h {mins} min" if hrs > 0 else f"{mins} min"
                plot_url = generate_plot(predicted_distance=distance_val, predicted_time=prediction)
        except ValueError:
            error = 'Please enter a valid numeric value (e.g., 12.5 or 45.0).'
            plot_url = generate_plot()
    else:
        plot_url = generate_plot()

    stats = {
        'total_records': len(df),
        'slope': round(slope, 4),
        'intercept': round(intercept, 2),
        'r_squared': round(r_squared, 4),
        'r_squared_pct': round(r_squared * 100, 2),
        'mean_distance': round(float(df['distance_km'].mean()), 2),
        'mean_time': round(float(df['delivery_time_min'].mean()), 2),
        'min_distance': round(float(df['distance_km'].min()), 2),
        'max_distance': round(float(df['distance_km'].max()), 2),
    }

    sample_data = df.head(10).to_dict('records')

    return render_template('lr_application.html',
                           prediction=prediction,
                           prediction_hours=prediction_hours,
                           distance_input=distance_input,
                           error=error,
                           plot_url=plot_url,
                           stats=stats,
                           sample_data=sample_data)

# ================= ACTIVITY 2 - CLASSIFICATION MODELS =================
# ------------- Logistic Regression: Customer Churn (1 independent var) -------------
churn_path = os.path.join(os.path.dirname(__file__), 'data', 'logistic_churn.csv')
churn_df = pd.read_csv(churn_path)
X_churn = churn_df[['tenure_months']].values
y_churn = churn_df['churn'].values
Xch_tr, Xch_te, ych_tr, ych_te = train_test_split(X_churn, y_churn, test_size=0.2, random_state=42)
logistic_model = LogisticRegression(max_iter=2000)
logistic_model.fit(Xch_tr, ych_tr)
churn_pred = logistic_model.predict(Xch_te)
cm_churn = confusion_matrix(ych_te, churn_pred)
churn_metrics = {
    'accuracy': round(accuracy_score(ych_te, churn_pred), 4),
    'precision': round(precision_score(ych_te, churn_pred, zero_division=0), 4),
    'recall': round(recall_score(ych_te, churn_pred, zero_division=0), 4),
    'f1': round(f1_score(ych_te, churn_pred, zero_division=0), 4),
}
CHURN_FEATURE = 'tenure_months'
CHURN_TARGET = 'churn'

# ------------- SGD Classifier (assigned model): Credit Risk (4 independent vars) -------------
credit_path = os.path.join(os.path.dirname(__file__), 'data', 'sgd_credit_risk.csv')
credit_df = pd.read_csv(credit_path)
CREDIT_FEATURES = ['income_usd', 'loan_amount_usd', 'credit_score', 'months_employed']
X_credit = credit_df[CREDIT_FEATURES].values
y_credit = credit_df['high_risk'].values
Xcr_tr, Xcr_te, ycr_tr, ycr_te = train_test_split(X_credit, y_credit, test_size=0.2, random_state=42)
sgd_model = make_pipeline(StandardScaler(), SGDClassifier(loss='log_loss', max_iter=5000, random_state=42))
sgd_model.fit(Xcr_tr, ycr_tr)
credit_pred = sgd_model.predict(Xcr_te)
cm_credit = confusion_matrix(ycr_te, credit_pred)
credit_metrics = {
    'accuracy': round(accuracy_score(ycr_te, credit_pred), 4),
    'precision': round(precision_score(ycr_te, credit_pred, zero_division=0), 4),
    'recall': round(recall_score(ycr_te, credit_pred, zero_division=0), 4),
    'f1': round(f1_score(ycr_te, credit_pred, zero_division=0), 4),
}
CREDIT_TARGET = 'high_risk'


def plot_churn(pred_tenure=None, pred_class=None):
    """Scatter of the churn dataset coloured by class + logistic decision threshold."""
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=130)
    m0 = churn_df[CHURN_TARGET] == 0
    m1 = ~m0
    j0 = np.random.default_rng(1).uniform(-0.06, 0.06, size=int(m0.sum()))
    j1 = np.random.default_rng(2).uniform(-0.06, 0.06, size=int(m1.sum()))
    ax.scatter(churn_df.loc[m0, CHURN_FEATURE], churn_df.loc[m0, CHURN_TARGET] + j0,
               s=30, alpha=0.45, color='#4cc9f0', edgecolors='none',
               label='Class 0 = Se queda (Stay)')
    ax.scatter(churn_df.loc[m1, CHURN_FEATURE], churn_df.loc[m1, CHURN_TARGET] + j1,
               s=30, alpha=0.45, color='#e94560', edgecolors='none',
               label='Class 1 = Se va (Churn)')
    threshold = -logistic_model.intercept_[0] / logistic_model.coef_[0][0]
    ax.axvline(x=threshold, color='#facc15', linestyle='--', linewidth=2,
               label=f'Classification threshold ≈ {threshold:.1f} months')
    if pred_tenure is not None and pred_class is not None:
        ax.scatter([pred_tenure], [pred_class], color='#10b981', s=170, zorder=6,
                   edgecolors='#ffffff', linewidths=2.5,
                   label=f'Current prediction: tenure {pred_tenure} months -> class {pred_class}')
        ax.axvline(x=pred_tenure, color='#10b981', linestyle=':', linewidth=1.4, alpha=0.7)
    ax.set_xlabel('Independent Variable: Time as Client (tenure_months)', fontsize=11, color='#e2e8f0', labelpad=10)
    ax.set_ylabel('Target Variable: churn (0 = Stay, 1 = Churn)', fontsize=11, color='#e2e8f0', labelpad=10)
    ax.set_title('Logistic Regression: Customer Churn vs Time as Client', fontsize=13, fontweight='bold', color='#f1f5f9', pad=14)
    ax.legend(fontsize=9.5, loc='lower right', framealpha=0.85, facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0')
    ax.set_facecolor('#131b2e')
    fig.patch.set_facecolor('#0b0f19')
    ax.set_xlim(0, 65)
    ax.set_ylim(-0.35, 1.45)
    ax.set_yticks([0, 1])
    ax.tick_params(colors='#94a3b8')
    ax.spines['bottom'].set_color('#334155')
    ax.spines['left'].set_color('#334155')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.15, color='#94a3b8', linestyle='--')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    plot = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return plot


def plot_credit():
    """Scatter of credit dataset (2 most relevant features) coloured by class."""
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=130)
    m0 = credit_df[CREDIT_TARGET] == 0
    m1 = ~m0
    ax.scatter(credit_df.loc[m0, 'income_usd'], credit_df.loc[m0, 'credit_score'],
               s=30, alpha=0.5, color='#4cc9f0', edgecolors='none',
               label='Class 0 = Bajo riesgo (Low Risk)')
    ax.scatter(credit_df.loc[m1, 'income_usd'], credit_df.loc[m1, 'credit_score'],
               s=30, alpha=0.5, color='#e94560', edgecolors='none',
               label='Class 1 = Alto riesgo (High Risk)')
    ax.set_xlabel('Monthly Income (income_usd)', fontsize=11, color='#e2e8f0', labelpad=10)
    ax.set_ylabel('Credit Score (credit_score)', fontsize=11, color='#e2e8f0', labelpad=10)
    ax.set_title('SGD Classifier: Credit Risk (income vs credit score, by class)', fontsize=13, fontweight='bold', color='#f1f5f9', pad=14)
    ax.legend(fontsize=9.5, loc='upper left', framealpha=0.85, facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0')
    ax.set_facecolor('#131b2e')
    fig.patch.set_facecolor('#0b0f19')
    ax.tick_params(colors='#94a3b8')
    ax.spines['bottom'].set_color('#334155')
    ax.spines['left'].set_color('#334155')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.15, color='#94a3b8', linestyle='--')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    plot = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return plot


def plot_confusion_matrix(cm, title):
    """Confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(4.6, 3.9), dpi=130)
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred 0', 'Pred 1'], color='#e2e8f0')
    ax.set_yticklabels(['Real 0', 'Real 1'], color='#e2e8f0')
    ax.set_xlabel('Predicted Class', color='#e2e8f0', labelpad=8)
    ax.set_ylabel('Actual Class', color='#e2e8f0', labelpad=8)
    ax.set_title(title, fontsize=12, fontweight='bold', color='#f1f5f9', pad=12)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                    fontsize=16, fontweight='bold',
                    color='white' if cm[i, j] > cm.max() / 2 else '#0b0f19')
    fig.patch.set_facecolor('#0b0f19')
    ax.tick_params(colors='#94a3b8')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    plot = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return plot


def cm_cards(cm):
    return {
        'tn': int(cm[0][0]), 'fp': int(cm[0][1]),
        'fn': int(cm[1][0]), 'tp': int(cm[1][1]),
    }


# --------------------- Logistic Regression routes ---------------------
@app.route('/supervised/logistic-regression/concepts')
@app.route('/ml/supervised/logistic-regression/concepts')
def logistic_concepts():
    return render_template('logistic_concepts.html')


@app.route('/supervised/logistic-regression/application', methods=['GET', 'POST'])
@app.route('/ml/supervised/logistic-regression/application', methods=['GET', 'POST'])
def logistic_application():
    error = None
    prediction = None
    tenure_input = None
    prob = None
    class_label = None
    class_meaning = None
    if request.method == 'POST':
        tenure_input = request.form.get('tenure', '').strip()
        try:
            value = float(tenure_input)
            if value <= 0 or value > 60:
                error = 'Please enter a tenure greater than 0 and up to 60 months.'
            else:
                p = float(logistic_model.predict_proba([[value]])[0][1])
                cls = int(p >= 0.5)
                prob = round(p, 4)
                prediction = cls
                class_label = 'Churn (Cliente se va)' if cls == 1 else 'Stay (Cliente se queda)'
                class_meaning = ('El modelo clasifica al cliente como PROPENSO A ABANDONAR el servicio, '
                                 'recomendando aplicar una estrategia de retenci\u00f3n.') if cls == 1 else \
                                ('El modelo clasifica al cliente como ESTABLE (poco riesgo de abandono), '
                                 'sin necesidad de acciones de retenci\u00f3n urgentes.')
        except ValueError:
            error = 'Please enter a valid numeric value (e.g., 12.0 or 30.5).'
    plot = plot_churn(pred_tenure=tenure_input if prediction is not None else None,
                      pred_class=prediction)
    stats = {
        'total_records': len(churn_df),
        'class0': int((churn_df[CHURN_TARGET] == 0).sum()),
        'class1': int((churn_df[CHURN_TARGET] == 1).sum()),
        'feature': 'tenure_months',
        'target': 'churn',
        'test_size': int(len(Xch_te)),
        'train_size': int(len(Xch_tr)),
    }
    return render_template('logistic_application.html', stats=stats, plot_url=plot,
                           error=error, tenure_input=tenure_input, prediction=prediction,
                           prob=prob, class_label=class_label, class_meaning=class_meaning)


@app.route('/supervised/logistic-regression/metrics')
@app.route('/ml/supervised/logistic-regression/metrics')
def logistic_metrics():
    cm = cm_cards(cm_churn)
    plot = plot_confusion_matrix(cm_churn, 'Confusion Matrix - Logistic Regression (Churn)')
    return render_template('logistic_metrics.html', cm=cm, metrics=churn_metrics,
                           cm_plot=plot, test_size=int(len(Xch_te)))


# --------------------- SGD Classifier (assigned model) routes ---------------------
@app.route('/supervised/sgd-classifier/concepts')
@app.route('/ml/supervised/sgd-classifier/concepts')
def sgd_concepts():
    return render_template('sgd_concepts.html')


@app.route('/supervised/sgd-classifier/application', methods=['GET', 'POST'])
@app.route('/ml/supervised/sgd-classifier/application', methods=['GET', 'POST'])
def sgd_application():
    error = None
    inputs = {}
    prediction = None
    prob = None
    class_label = None
    class_meaning = None
    if request.method == 'POST':
        try:
            income = float(request.form.get('income', '').strip())
            loan = float(request.form.get('loan', '').strip())
            credit = float(request.form.get('credit', '').strip())
            months = float(request.form.get('months', '').strip())
            inputs = {'income': income, 'loan': loan, 'credit': credit, 'months': months}
            if income <= 0 or loan <= 0 or not (300 <= credit <= 850) or months < 0:
                error = ('Please enter valid values: income > 0, loan amount > 0, '
                         'credit score between 300 and 850, and employment months >= 0.')
            else:
                p = float(sgd_model.predict_proba([[income, loan, credit, months]])[0][1])
                cls = int(p >= 0.5)
                prob = round(p, 4)
                prediction = cls
                class_label = 'High Risk (Alto riesgo)' if cls == 1 else 'Low Risk (Bajo riesgo)'
                class_meaning = ('El solicitante fue clasificado como ALTO RIESGO de impago; '
                                 'el pr\u00e9stamo deber\u00eda revisarse o rechazarse.') if cls == 1 else \
                                ('El solicitante fue clasificado como BAJO RIESGO de impago; '
                                 'el pr\u00e9stamo puede aprobarse.')
        except ValueError:
            error = 'Please enter valid numeric values in all four fields.'
    plot = plot_credit()
    stats = {
        'total_records': len(credit_df),
        'class0': int((credit_df[CREDIT_TARGET] == 0).sum()),
        'class1': int((credit_df[CREDIT_TARGET] == 1).sum()),
        'features': CREDIT_FEATURES,
        'target': CREDIT_TARGET,
        'test_size': int(len(Xcr_te)),
        'train_size': int(len(Xcr_tr)),
    }
    return render_template('sgd_application.html', stats=stats, plot_url=plot,
                           error=error, inputs=inputs, prediction=prediction,
                           prob=prob, class_label=class_label, class_meaning=class_meaning)


@app.route('/supervised/sgd-classifier/metrics')
@app.route('/ml/supervised/sgd-classifier/metrics')
def sgd_metrics():
    cm = cm_cards(cm_credit)
    plot = plot_confusion_matrix(cm_credit, 'Confusion Matrix - SGD Classifier (Credit Risk)')
    return render_template('sgd_metrics.html', cm=cm, metrics=credit_metrics,
                           cm_plot=plot, test_size=int(len(Xcr_te)))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

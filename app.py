import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless server mode without GUI
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression

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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

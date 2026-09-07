# Data with Roots

Interactive **Machine Learning** web application built with **Python**, **Flask**, and **Bootstrap** to estimate **delivery time based on distance traveled** using a **Simple Linear Regression** model.

<p align="center">
  <strong>Universidad de Cundinamarca · Semester 6 · Machine Learning</strong>
</p>

---

## Project Topica

**Estimate delivery time based on distance traveled.**

**600 historical shipment records** (distance in km and actual delivery time in minutes) are used to train a Simple Linear Regression model that predicts delivery time for new distances.

## Features

- **Machine Learning concepts** explained visually
- **Types of ML**: Supervised, Unsupervised, and Reinforcement Learning
- **4 Use Cases** in different contexts (healthcare, finance, retail, automotive)
- **Linear Regression** - fundamental concepts
- **Scatter plot** with regression line generated using matplotlib
- **Real-time prediction form** using the scikit-learn model

## Technologies

| Technology | Version | Function |
|-----------|---------|---------|
| Python | 3.x | Main language |
| Flask | 3.0.0 | Web microframework |
| scikit-learn | 1.3.2 | Linear Regression model |
| numpy | 1.26.2 | Numerical calculations |
| pandas | 2.1.4 | Data manipulation |
| matplotlib | 3.8.2 | Visualization (charts) |
| Bootstrap | 5.3.2 | Responsive design |

## Project Structure

```text
ML/
|-- app.py                    # Flask Application + ML Model
|-- generate_dataset.py       # Generator for 600 records
|-- requirements.txt          # Python dependencies
|-- Procfile                  # Render configuration
|-- data/
|   +-- delivery_dataset.csv  # 600 dataset records
|-- static/css/
|   +-- style.css             # Styles (dark theme)
+-- templates/
    |-- base.html             # Base template (navbar + footer)
    |-- home.html             # Home page
    |-- ml_concepts.html      # ML Concepts
    |-- ml_types.html         # ML Types
    |-- use_case_1.html       # Use Case 1
    |-- use_case_2.html       # Use Case 2
    |-- use_case_3.html       # Use Case 3
    |-- use_case_4.html       # Use Case 4
    |-- lr_concepts.html      # Linear Regression Concepts
    +-- lr_application.html   # Linear Regression Application (chart + form)

## Linear Regression Model

- **Independent Variable (X):** Distance traveled (km)
- **Dependent Variable (Y):** Delivery time (min)
- **Records:** 600
- **Equation:** `Time = 0.4513 × Distance + 4.87`
- **R²:** 0.9961 (99.6% of variance explained)

## Local Execution

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
python app.py

# 3. Open in browser
# http://127.0.0.1:5000
```

## Deployment on Render

1. Create a Web Service connected to the repository
2. **Build Command:** `pip install -r requirements.txt`
3. **Start Command:** `gunicorn app:app`
4. Create the service

## Links

- Repository: `https://github.com/machinelearning-source/data-with-roots-1`
- Application (Render): `https://data-with-roots-1.onrender.com`

## Academic Context

This project is part of the university curriculum in **Machine Learning** and demonstrates how predictive models can be applied in real-world logistics scenarios, combining data analysis, visualization, and web development in a practical and educational way.

---

**Universidad de Cundinamarca · Faculty of Engineering · Semester 6 · Machine Learning**

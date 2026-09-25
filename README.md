# Data with Roots

Data with Roots is a Flask web application for the Machine Learning course at Universidad de Cundinamarca. The repository contains the existing supervised learning modules and the R2A2 unsupervised learning module.

## R2A2 topic

**Selected topic:** Flight segmentation by distance and duration

The R2A2 module is available in English and includes:

- Unsupervised learning, clustering, K-Means, centroids, iterations, and Euclidean distance.
- A manual exercise with 100 records, two numerical variables, three initial centroids, three iterations, complete distance and assignment tables, plots, and WCSS comparison.
- A Scikit-Learn application with 1,000 flight records, standardized preprocessing, three clusters, centroids, cluster sizes, a complete assignment table, and silhouette evaluation.
- The assigned Logistic Regression model for the binary target `delayed`: `0 = on-time` and `1 = delayed` using a 15-minute threshold.
- An interactive prediction form for a new flight.

The rubric mentions `study_hours` as an independent variable, but the selected flight dataset does not contain that field. The implementation uses the available flight predictors `distance_km` and `duration_min` rather than inventing an invalid variable.

## Public routes

| Section | Route |
| --- | --- |
| Home | `/` |
| K-Means concepts | `/unsupervised/kmeans/concepts` |
| Manual K-Means exercise | `/unsupervised/kmeans/manual` |
| K-Means application | `/unsupervised/kmeans/application` |
| Logistic Regression flight-delay application | `/supervised/logistic-regression/flight-delay` |
| Health check | `/health` |

The application uses relative routes and does not require authentication. All module pages are linked from the main navigation.

## Project structure

```text
.
├── app.py
├── kmeans_ml.py
├── requirements.txt
├── Procfile
├── render.yaml
├── data/
│   ├── flights_manual_100.csv
│   └── flights_dataset.csv
├── templates/
│   ├── base.html
│   ├── home.html
│   ├── kmeans_concepts.html
│   ├── kmeans_manual.html
│   ├── kmeans_application.html
│   └── kmeans_classification.html
├── static/
│   └── css/style.css
└── generate_flights_dataset.py
```

## Local execution

```bash
py -m pip install -r requirements.txt
py app.py
```

Open `http://127.0.0.1:5000` in a browser. The application uses Matplotlib's `Agg` backend, so it does not require a graphical desktop on the server.

The R2A2 computations and Base64 plots are cached after their first request. This avoids repeating K-Means, CSV loading, and chart generation on every visit and keeps navigation responsive.

## Git workflow

The R2A2 work is developed on the `R2A2` branch and is based on the repository's existing structure. The branch is intended to contain the progressive implementation evidence required by the assignment.

```bash
git switch R2A2
git status
git log --oneline --decorate -10
```

The repository is configured for the public GitHub project:

<https://github.com/machinelearning-source/data-with-roots-1>

## Render deployment

The root `render.yaml` and `Procfile` use:

```text
Build command: pip install -r requirements.txt
Start command: gunicorn app:app
```

The public service URL recorded for the existing deployment is:

<https://data-with-roots-1.onrender.com>

Render's free plan may sleep after inactivity, so the first request can take longer while the service wakes up. All application routes are public once the service is connected to the public repository.

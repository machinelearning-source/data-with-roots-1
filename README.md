# Data with Roots - Machine Learning Web Application

> **Universidad de Cundinamarca** - Facultad de Ingeniería  
> **Asignatura:** Machine Learning - Semestre 6  
> **Proyecto:** Data with Roots  
> **Tema Asignado:** Estimar el tiempo de entrega según la distancia recorrida (Regresión Lineal Simple)  
> **Desarrollador:** Luis Revolledo  

---

## 📋 Descripción del Proyecto

**Data with Roots** es una plataforma web interactiva desarrollada con **Python**, **Flask** y **Bootstrap 5** que cumple dos propósitos fundamentales:
1. **Módulo Educativo:** Presentar de manera didáctica y visual los conceptos esenciales de Machine Learning, la taxonomía de modelos (Supervisado, No Supervisado y por Refuerzo) y 4 casos de uso reales en distintas industrias.
2. **Módulo Práctico y Predictivo:** Implementar un modelo de **Regresión Lineal Simple** entrenado con **Scikit-Learn** sobre un conjunto de **600 registros** de logística de envíos para predecir en tiempo real el tiempo de entrega ($Y$ en minutos) a partir de la distancia recorrida ($X$ en kilómetros).

---

## 🗂️ Estructura Obligatoria del Menú y Rutas

| Menú Principal | Submenú / Sección | Ruta en la App (URL) | Descripción |
| :--- | :--- | :--- | :--- |
| **Home** | Inicio | `/` | Página principal y presentación del proyecto |
| **Machine Learning** | Concepts | `/ml/concepts` | ¿Qué es ML?, propósito, datos, modelos y ejemplos |
| **Machine Learning** | Types of ML | `/ml/types` | Supervisado, No Supervisado y Refuerzo |
| **Machine Learning** | Caso de Uso 1 | `/ml/use-cases/1` | Diagnóstico Médico por Imágenes (Salud / CNN) |
| **Machine Learning** | Caso de Uso 2 | `/ml/use-cases/2` | Detección de Fraude Financiero (Fintech) |
| **Machine Learning** | Caso de Uso 3 | `/ml/use-cases/3` | Predicción de Demanda en Retail (Series de Tiempo) |
| **Machine Learning** | Caso de Uso 4 | `/ml/use-cases/4` | Navegación de Vehículos Autónomos (Robótica / RL) |
| **Supervised** | LR - Concepts | `/supervised/linear-regression/concepts` | Fundamentos teóricos de Regresión Lineal Simple |
| **Supervised** | LR - Application | `/supervised/linear-regression/application` | Modelo Scikit-Learn, 600 datos, gráfica y formulario interactivo |

---

## 🔬 Especificaciones del Modelo de Machine Learning

- **Problema:** Estimación de tiempos de entrega en logística de última milla.
- **Variable Independiente ($X$):** Distancia recorrida en carretera (`distance_km`), medida en **kilómetros ($km$)**.
- **Variable Dependiente ($Y$):** Tiempo total de entrega (`delivery_time_min`), medido en **minutos ($min$)**.
- **Tamaño del Dataset:** **600 registros** (supera el requisito mínimo de 500 registros).
- **Algoritmo:** `sklearn.linear_model.LinearRegression`.
- **Ecuación del Modelo:**
  $$\text{Tiempo (min)} = 0.4482 \times \text{Distancia (km)} + 5.28$$
- **Métricas:**
  - **Coeficiente de Determinación ($R^2$):** $\approx 97.86\%$ (Ajuste lineal excelente).
  - **Pendiente ($\beta_1$):** $\approx 0.4482\text{ min/km}$ (~2.2 minutos por cada 5 km).
  - **Intercepto ($\beta_0$):** $\approx 5.28\text{ min}$ (Tiempo base de preparación y despacho).

---

## 🛠️ Tecnologías y Librerías Utilizadas

- **Lenguaje:** Python 3.14 / 3.11+
- **Backend:** Flask 3.x
- **Machine Learning:** Scikit-Learn 1.x
- **Manipulación de Datos:** Pandas y NumPy
- **Visualización Gráfica:** Matplotlib (Backend `'Agg'`, codificación Base64 en memoria)
- **Frontend:** HTML5, CSS3 moderno con tema oscuro, Bootstrap 5.3.2
- **Servidor de Producción:** Gunicorn
- **Control de Versiones:** Git & GitHub (Flujo `master` $\rightarrow$ `R1A1` $\rightarrow$ PR $\rightarrow$ `master`)
- **PaaS / Despliegue:** Render

---

## 🌿 Flujo Git y Gestión de Ramas (Requisito Obligatorio)

El desarrollo del proyecto se ejecutó rigurosamente siguiendo la especificación docente:
```text
master ───> rama R1A1 ───> desarrollo progresivo ───> commits ───> Pull Request ───> merge ───> master
```

1. Se inició en la rama `master`.
2. Se creó la rama `R1A1` (`git checkout -b R1A1`).
3. Se realizaron commits progresivos y descriptivos que demuestran la evolución del código.
4. Se integró mediante Pull Request hacia la rama `master`.

---

## 🚀 Instrucciones de Ejecución Local

1. Clonar el repositorio o abrir la carpeta en **Visual Studio Code**:
   ```bash
   code .
   ```
2. Instalar dependencias:
   ```bash
   py -m pip install -r requirements.txt
   ```
3. Iniciar el servidor local:
   ```bash
   py app.py
   ```
4. Abrir en el navegador:
   [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## ☁️ Despliegue en Render

1. Conectar tu repositorio público de GitHub en [Render.com](https://render.com).
2. Crear un nuevo **Web Service**.
3. Parámetros de configuración:
   - **Environment:** `Python`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
4. Desplegar y verificar que todas las páginas del menú sean navegables.

### 🌐 URL de Producción

La aplicación desplegada en Render está disponible en:

**https://data-with-roots-1.onrender.com**

> **Nota:** Render asigna el sufijo `-1` automáticamente porque el nombre `data-with-roots` ya estaba registrado en su plataforma. Usa **siempre** esta URL con el `-1`. El dominio sin ese sufijo (`https://data-with-roots.onrender.com`) **no está disponible**.

### 🔁 Auto-Deploy

- El servicio Render está conectado al repositorio **`machinelearning-source/data-with-roots-1`** (rama `main`).
- Cada **push a `main`** dispara automáticamente un nuevo despliegue (*Auto-Deploy*).
- Configuración del servicio:
  - **Build Command:** `pip install -r requirements.txt`
  - **Start Command:** `gunicorn app:app`
  - **Runtime:** `python-3.12.7`

### 🥶 Nota sobre el plan gratuito (spin-down)

El plan **free** de Render **duerme** la instancia tras un período de inactividad. Al volver a visitar el sitio, la **primera carga puede tardar ~50 segundos** mientras "despierta". No es un error.

Consejos:
1. **Esperar** la primera carga (una vez despierto, responde normalmente).
2. **Keep-alive gratuito:** usar [UptimeRobot](https://uptimerobot.com) para sondear la URL cada 5 minutos y evitar que se duerma.
3. **Upgrade (opcional):** el plan con pago elimina el spin-down.

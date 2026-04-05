# Docker Lab 2 — Iris Classifier (Flask + Multi-Model + Docker)

This document summarizes **all changes made for this submission**, explains **what the lab demonstrates**, and gives **step-by-step instructions to build and run** the project. Evidence screenshots live in [`screenshots/`](./screenshots/).

---

## 1. Changes implemented (submission work)

Everything below was added or updated to complete and extend the original Lab 2 starter.

### Training pipeline (`src/model_training.py`)

- Loads the **Iris** dataset, splits train/test (`test_size=0.2`, `random_state=42`).
- Fits a single **`StandardScaler`** on training features; applies it to train and test.
- Saves the scaler as **`scaler.pkl`** (`joblib`) so serving uses **identical preprocessing** as training.
- Trains **`KNeighborsClassifier` (k=5)** on scaled data and saves **`knn_model.pkl`**.
- Trains a small **Keras** classifier (two dense layers, softmax) on the **same scaled** data for 50 epochs and saves **`my_model.keras`**.
- Prints a confirmation line listing all three artifacts.

### Flask API (`src/main.py`)

- Loads **`my_model.keras`**, **`knn_model.pkl`**, and **`scaler.pkl`** at startup.
- **`GET /health`** — returns `{"status":"ok"}` for probes and Docker **`HEALTHCHECK`**.
- **`GET /`** — welcome string for the root route.
- **`/predict` (GET/POST)** — **Keras** predictions after **`scaler.transform`** on the four iris features.
- **`/predict_knn` (GET/POST)** — **KNN** predictions after the same scaling.
- **Input parsing**: supports **query parameters (GET)**, **HTML form (POST)**, and **JSON (POST with `Content-Type: application/json`)**; bare **GET** without all four parameters serves the HTML form.
- Responses include **`predicted_class`** and a **`model`** field (`keras` vs `knn`) for clarity.
- Listens on **`0.0.0.0:8080`** with **`debug=False`** (suitable for containers). Port **8080** avoids binding privileged port **80** as a non-root user.

### Web UI (`src/templates/`)

- **`predict.html`** — form and UI for the **Keras** model; static flower visuals wired via Flask `url_for('static', ...)`.
- **`predict_knn.html`** — parallel UI for the **KNN** model, posting to **`/predict_knn`**.
- Static assets under **`src/statics/`** include **SVG** placeholders and **JPEG** images for the three species (Setosa, Versicolor, Virginica).

### Dependencies (`requirements.txt`)

- **`tensorflow>=2.20,<2.22`** — range chosen so installs work on **current PyPI** (e.g. Apple Silicon / newer Python) while the **Docker image** still uses **Python 3.10**.
- **`scikit-learn`**, **`Flask`**, **`requests`**, **`joblib`**.
- Standalone **`keras`** was removed from requirements; **`tf.keras`** is used with the pinned TensorFlow stack.

### Docker image (`Dockerfile`)

- **Multi-stage build**:
  - **Stage `model_training`**: copies `model_training.py` and `requirements.txt`, installs deps, runs training, produces **`my_model.keras`**, **`scaler.pkl`**, **`knn_model.pkl`**.
  - **Stage `serving`**: copies all three artifacts from the first stage, copies **`main.py`**, templates, and statics, installs deps, runs the Flask app.
- **`python:3.10-slim`** base for a smaller image.
- **`EXPOSE 8080`** aligned with the app.
- **`USER appuser`** (non-root) after `chown` on `/app`.
- **`HEALTHCHECK`** calls **`/health`** via Python’s **`urllib`** (no extra `curl` package).
- File is named **`Dockerfile`** (default for `docker build`).

### Docker Compose (`docker-compose.yml`)

- **`model-training`**: installs requirements, runs **`python src/model_training.py`**, copies **`my_model.keras`**, **`scaler.pkl`**, **`knn_model.pkl`** into a named volume **`model_exchange`**.
- **`serving`**: waits for training to **complete successfully**, copies artifacts from the volume into `/app`, installs deps, runs **`python src/main.py`**.
- Host port **`8080:8080`** published for the browser and `curl`.
- Source, templates, and statics are **bind-mounted** for development-style runs.

### Supporting files

- **`.dockerignore`** — excludes `__pycache__`, virtualenvs, stray local **`.pkl` / `.keras`** files, and common noise so build context stays small and reproducible.
- **`HOWTO`** — quick commands for **`docker build`**, **`docker run`**, **`curl`** checks, optional Compose, and notes about working directory / port **8080**.

### Evidence screenshots (`screenshots/`)

Captured while running the lab: successful build, running container with port mapping, **`/health`** and **`/`** via terminal and browser, both prediction endpoints via GET, and the two HTML forms. See [Section 4](#4-screenshots-evidence).

---

## 2. About the lab (what this work demonstrates)

### Original lab idea

The course Docker lab introduces **containerizing** a small ML workflow: **train a model**, then **serve predictions** with **Flask**, so graders can reproduce your work with **`docker build`** and **`docker run`**.

### What this submission adds beyond a minimal baseline

| Theme | What you implemented |
|--------|----------------------|
| **Reproducible training in the image** | Training runs inside **stage 1** of the `Dockerfile`; the runtime image **does not** depend on ad-hoc local files. |
| **Clear separation of train vs serve** | Multi-stage copy (`COPY --from=model_training`) mirrors “build artifacts once, ship a lean server image.” |
| **Two models, one preprocessor** | Keras + KNN share **`StandardScaler`**; avoids inconsistent train/serve behavior. |
| **Observability** | **`/health`** plus Docker **`HEALTHCHECK`** for “is the process up?” checks. |
| **Multiple interfaces** | JSON API, query strings, and HTML forms for demos and manual testing. |
| **Optional Compose path** | Second way to run: train container → shared volume → serve container. |

Together, this matches common **MLOps** goals: **repeatable builds**, **artifact handoff**, **health endpoints**, and **simple multi-model serving**—all inside **Docker**.

---

## 3. How to run the lab

Run all commands from **this directory** (`Labs/Docker_Labs/Lab2`).

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine) running.
- Host port **8080** free. If something else uses it (e.g. an old `docker run`), stop that container first.

### Path to the lab folder

From the repo root:

```bash
cd Labs/Docker_Labs/Lab2
```

If your shell prompt already shows you are inside **`Lab2`**, do **not** run `cd Labs/Docker_Labs/Lab2` again (that path is relative to the repo root).

### Build the image

```bash
docker build -t iris-app .
```

The first build can take several minutes (TensorFlow wheel + training epochs).

### Run the container

Foreground (logs in the terminal):

```bash
docker run --rm -p 8080:8080 iris-app
```

Background (typical for testing with `curl` / browser):

```bash
docker run --rm -d --name iris -p 8080:8080 iris-app
```

### Verify with `curl`

```bash
curl -s http://localhost:8080/health
curl -s http://localhost:8080/
curl -s "http://localhost:8080/predict?sepal_length=5.1&sepal_width=3.5&petal_length=1.4&petal_width=0.2"
curl -s "http://localhost:8080/predict_knn?sepal_length=5.1&sepal_width=3.5&petal_length=1.4&petal_width=0.2"
```

JSON **POST** example:

```bash
curl -s -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

### Verify in the browser

- [http://127.0.0.1:8080/](http://127.0.0.1:8080/) — welcome message  
- [http://127.0.0.1:8080/health](http://127.0.0.1:8080/health) — health JSON  
- [http://127.0.0.1:8080/predict](http://127.0.0.1:8080/predict) — Keras form  
- [http://127.0.0.1:8080/predict_knn](http://127.0.0.1:8080/predict_knn) — KNN form  

### Optional: Docker Compose

Stop any container already bound to **8080**, then:

```bash
docker compose up --build
```

When finished:

```bash
docker compose down -v
```

### Optional: container health status

If you used the built image with **`HEALTHCHECK`**, after the start period:

```bash
docker inspect --format '{{.State.Health.Status}}' iris
```

Expect **`healthy`** when the app responds on **`/health`**.

### Stop a background container

```bash
docker stop iris
```

---

## 4. Screenshots (evidence)

Below are the screenshots stored in **`screenshots/`**, in a logical order: build → run → API checks → browser UI.

### Docker build succeeded

Shows a completed **`docker build -t iris-app .`** (multi-stage build finishing successfully).

![Docker build succeeded](./screenshots/Docker%20build%20succeeded.png)

### Container running and port mapping

Shows **`docker ps`** with the **`iris`** container and **`8080->8080`** port publish.

![Container running + port mapping](./screenshots/Container%20running%20%2B%20port%20mapping.png)

### Health endpoint (`curl`)

Shows **`curl`** to **`/health`** returning **`{"status":"ok"}`**.

![health JSON](./screenshots/health%20JSON.png)

### Root endpoint (`curl`)

Shows **`curl`** to **`/`** returning the welcome string.

![welcome text](./screenshots/welcome%20text.png)

### Both models via GET

Shows **`curl`** to **`/predict`** and **`/predict_knn`** with the same iris measurements; JSON includes **`model`** and **`predicted_class`**.

![Both models via GET](./screenshots/Both%20models%20via%20GET.png)

### Health endpoint (browser)

Shows **`/health`** rendered in the browser as JSON.

![Health in browser](./screenshots/Health%20in%20browser.png)

### Keras prediction form (browser)

Shows the **NEURAL IRIS** HTML form served at **`/predict`**.

![Keras form](./screenshots/Keras%20form.png)

### KNN prediction form (browser)

Shows the **KNN IRIS** HTML form served at **`/predict_knn`**.

![KNN form](./screenshots/KNN%20form.png)

---

## File map (quick reference)

| Path | Role |
|------|------|
| `Dockerfile` | Multi-stage train + serve image |
| `docker-compose.yml` | Train volume + serving service |
| `requirements.txt` | Python dependencies |
| `HOWTO` | Short command cheat sheet |
| `.dockerignore` | Build context exclusions |
| `src/model_training.py` | Train/save Keras + KNN + scaler |
| `src/main.py` | Flask app, routes, inference |
| `src/templates/predict.html` | Keras UI |
| `src/templates/predict_knn.html` | KNN UI |
| `src/statics/*` | Flower images / SVGs |
| `screenshots/*.png` | Submission evidence |

---

*Lab 2 — Docker: Flask deployment, multi-model serving, and container workflow.*

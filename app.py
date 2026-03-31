from flask import (
    Flask,
    render_template,
    request,
    g,
    redirect,
    url_for,
    session,
    flash,
    send_file,
)
import pickle
import numpy as np
import sqlite3
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import io

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ----------------- CONFIG -----------------
MODEL_FILENAME = "heart-disease-prediction-knn-model.pkl"
DATABASE = "heart_predictions.db"

app = Flask(__name__)
app.secret_key = "change_this_to_a_random_secret_key"  # change in production


# ----------------- MODEL LOADING -----------------
with open(MODEL_FILENAME, "rb") as f:
    model = pickle.load(f)


# ----------------- DB HELPERS -----------------
def get_db():
    """Get a connection for the current request."""
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db


def init_db():
    """Create tables if they don't exist."""
    with sqlite3.connect(DATABASE) as conn:
        cur = conn.cursor()

        # predictions table – stores each user input + prediction
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                age INTEGER,
                sex INTEGER,
                cp INTEGER,
                trestbps INTEGER,
                chol INTEGER,
                fbs INTEGER,
                restecg INTEGER,
                thalach INTEGER,
                exang INTEGER,
                oldpeak REAL,
                slope INTEGER,
                ca INTEGER,
                thal INTEGER,
                prediction INTEGER,
                proba REAL
            )
            """
        )

        # us
        # ers table – simple auth
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT
            )
            """
        )

        conn.commit()


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()


# create tables on startup
init_db()


# ----------------- AUTH DECORATOR -----------------
def login_required(view_func):
    from functools import wraps

    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if not session.get("logged_in"):
            flash("Please login or create an account to continue.", "warning")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapped_view


# ----------------- ROUTES -----------------
@app.route("/")
def home():
    # Landing page (AK Health Prediction)
    return render_template("index.html")


@app.route("/predict_form")
@login_required
def predict_form():
    # Prediction form page
    return render_template("main.html")


# ---------- PREDICTION + SAVE TO DB + STORE IN SESSION ----------
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    try:
        # 1. Read user input from form
        age = int(request.form["age"])
        sex = int(request.form["sex"])
        cp = int(request.form["cp"])
        trestbps = int(request.form["trestbps"])
        chol = int(request.form["chol"])
        fbs = int(request.form["fbs"])
        restecg = int(request.form["restecg"])
        thalach = int(request.form["thalach"])
        exang = int(request.form["exang"])
        oldpeak = float(request.form["oldpeak"])
        slope = int(request.form["slope"])
        ca = int(request.form["ca"])
        thal = int(request.form["thal"])

        features = np.array(
            [
                [
                    age,
                    sex,
                    cp,
                    trestbps,
                    chol,
                    fbs,
                    restecg,
                    thalach,
                    exang,
                    oldpeak,
                    slope,
                    ca,
                    thal,
                ]
            ]
        )

        # 2. Model prediction
        prediction = int(model.predict(features)[0])

        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(features)[0][1])  # probability of class 1

        created_at = datetime.now().isoformat(timespec="seconds")

        # 3. Save everything to DB (for logs) – best effort
        db_id = None
        try:
            conn = get_db()
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO predictions (
                    created_at, age, sex, cp, trestbps, chol, fbs,
                    restecg, thalach, exang, oldpeak, slope, ca, thal,
                    prediction, proba
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    age,
                    sex,
                    cp,
                    trestbps,
                    chol,
                    fbs,
                    restecg,
                    thalach,
                    exang,
                    oldpeak,
                    slope,
                    ca,
                    thal,
                    prediction,
                    proba,
                ),
            )
            conn.commit()
            db_id = cur.lastrowid
        except Exception as db_err:
            # Don't crash the app; just log it
            print("DB error during insert:", db_err)

        # 4. Store latest prediction in session (for PDF generation)
        session["last_prediction"] = {
            "id": db_id,
            "created_at": created_at,
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
            "prediction": prediction,
            "proba": proba,
        }

        # 5. Render result page
        return render_template(
            "result.html",
            prediction=prediction,
            proba=proba,
            error=None,
        )

    except Exception as e:
        return render_template(
            "result.html",
            prediction=None,
            proba=None,
            error=str(e),
        )


# ---------- BUILD PDF FROM SESSION DATA (NOW WITH SYMPTOMS & CAUSES) ----------
def build_pdf_from_data(data, username=None):
    """
    data is the dict stored in session["last_prediction"].
    """
    pid = data.get("id") or "N/A"
    created_at = data.get("created_at", "N/A")
    age = data["age"]
    sex = data["sex"]
    cp = data["cp"]
    trestbps = data["trestbps"]
    chol = data["chol"]
    fbs = data["fbs"]
    restecg = data["restecg"]
    thalach = data["thalach"]
    exang = data["exang"]
    oldpeak = data["oldpeak"]
    slope = data["slope"]
    ca = data["ca"]
    thal = data["thal"]
    prediction = data["prediction"]
    proba = data["proba"]

    # human readable mappings
    sex_map = {0: "Female", 1: "Male"}
    cp_map = {
        0: "Typical Angina",
        1: "Atypical Angina",
        2: "Non-anginal Pain",
        3: "Asymptomatic",
    }
    fbs_map = {0: "≤ 120 mg/dL", 1: "> 120 mg/dL"}
    restecg_map = {
        0: "Normal",
        1: "ST-T Wave Abnormality",
        2: "Left Ventricular Hypertrophy",
    }
    exang_map = {0: "No", 1: "Yes"}
    slope_map = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
    thal_map = {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}

    result_text = (
        "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected"
    )
    confidence_text = f"{proba * 100:.2f}%" if proba is not None else "N/A"

    pdf_buffer = io.BytesIO()
    p = canvas.Canvas(pdf_buffer, pagesize=letter)
    width, height = letter

    y = height - 50
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, y, "Heart Disease Prediction Report")

    y -= 25
    p.setFont("Helvetica", 10)
    p.drawString(50, y, f"Report ID: {pid}")
    y -= 15
    p.drawString(50, y, f"Generated At: {created_at}")

    # include username when available
    if username:
        y -= 15
        p.drawString(50, y, f"User: {username}")

    y -= 25
    p.setFont("Helvetica-Bold", 12)
    p.drawString(50, y, "User Inputs")
    p.setFont("Helvetica", 10)
    y -= 15

    def ensure_space():
        nonlocal y
        if y < 80:
            p.showPage()
            y = height - 50
            p.setFont("Helvetica", 10)

    def line(label, value):
        nonlocal y
        ensure_space()
        p.drawString(60, y, f"{label}: {value}")
        y -= 15

    def text_line(text):
        nonlocal y
        ensure_space()
        p.drawString(60, y, text)
        y -= 15

    # ------------- USER INPUTS -------------
    line("Age", age)
    line("Sex", sex_map.get(sex, sex))
    line("Chest Pain Type (cp)", cp_map.get(cp, cp))
    line("Resting BP (trestbps)", f"{trestbps} mmHg")
    line("Cholesterol (chol)", f"{chol} mg/dL")
    line("Fasting Blood Sugar (fbs)", fbs_map.get(fbs, fbs))
    line("Resting ECG (restecg)", restecg_map.get(restecg, restecg))
    line("Maximum Heart Rate (thalach)", f"{thalach} bpm")
    line("Exercise-Induced Angina (exang)", exang_map.get(exang, exang))
    line("ST Depression (oldpeak)", oldpeak)
    line("Slope of ST Segment", slope_map.get(slope, slope))
    line("Number of Major Vessels (ca)", ca)
    line("Thalassemia (thal)", thal_map.get(thal, thal))

    # ------------- PREDICTION RESULT -------------
    y -= 10
    ensure_space()
    p.setFont("Helvetica-Bold", 12)
    p.drawString(50, y, "Prediction Result")
    y -= 20
    p.setFont("Helvetica", 10)
    line("Prediction", result_text)
    line("Model Confidence", confidence_text)

    # ------------- SYMPTOMS SECTION -------------
    y -= 10
    ensure_space()
    p.setFont("Helvetica-Bold", 12)
    p.drawString(50, y, "Heart Disease – Common Symptoms")
    y -= 18
    p.setFont("Helvetica", 10)

    text_line("• Chest pain, pressure, or discomfort (especially with activity)")
    text_line("• Shortness of breath or difficulty breathing")
    text_line("• Unusual fatigue or weakness")
    text_line("• Pain in arms, neck, jaw, back, or stomach")
    text_line("• Palpitations (feeling of rapid or irregular heartbeat)")
    text_line("• Dizziness, light-headedness, or fainting")
    text_line("• Swelling of legs, ankles, or feet")

    # ------------- CAUSES / RISK FACTORS SECTION -------------
    y -= 10
    ensure_space()
    p.setFont("Helvetica-Bold", 12)
    p.drawString(50, y, "Common Causes / Risk Factors")
    y -= 18
    p.setFont("Helvetica", 10)

    text_line("• High blood pressure (hypertension)")
    text_line("• High cholesterol levels")
    text_line("• Diabetes or high blood sugar")
    text_line("• Smoking or tobacco use")
    text_line("• Obesity and lack of regular physical activity")
    text_line("• Unhealthy diet (high in salt, sugar, saturated fat)")
    text_line("• Family history of heart disease")
    text_line("• Increasing age and long-term stress")

    # ------------- DISCLAIMER -------------
    y -= 10
    ensure_space()
    p.setFont("Helvetica-Oblique", 9)
    p.drawString(
        50,
        y,
        "Disclaimer: This report is generated by a machine learning model and is not a medical diagnosis.",
    )
    y -= 12
    ensure_space()
    p.drawString(
        50,
        y,
        "If you have concerning symptoms, please consult a qualified healthcare professional immediately.",
    )

    p.showPage()
    p.save()
    pdf_buffer.seek(0)
    filename = f"heart_prediction_report_{created_at.replace(':', '-')}.pdf"
    return pdf_buffer, filename


# ---------- PDF DOWNLOAD FROM SESSION ----------
@app.route("/download_pdf")
@login_required
def download_pdf():
    data = session.get("last_prediction")
    if not data:
        flash(
            "No recent prediction found to generate PDF. Please run a new prediction.",
            "warning",
        )
        return redirect(url_for("predict_form"))

    # pass current logged-in username (if any) into PDF
    username = session.get("admin_user") or session.get("username")
    pdf_buffer, filename = build_pdf_from_data(data, username=username)
    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name=filename,
        mimetype="application/pdf",
    )


<<<<<<< HEAD
# ---------- REST API (JSON) ----------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        payload = request.get_json(force=True)
        required = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
        for field in required:
            if field not in payload:
                return {"error": f"Missing field: {field}"}, 400

        features = np.array(
            [[
                int(payload["age"]),
                int(payload["sex"]),
                int(payload["cp"]),
                int(payload["trestbps"]),
                int(payload["chol"]),
                int(payload["fbs"]),
                int(payload["restecg"]),
                int(payload["thalach"]),
                int(payload["exang"]),
                float(payload["oldpeak"]),
                int(payload["slope"]),
                int(payload["ca"]),
                int(payload["thal"]),
            ]]
        )

        prediction = int(model.predict(features)[0])
        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(features)[0][1])

        return {
            "prediction": prediction,
            "probability": proba,
            "message": "Heart disease detected" if prediction == 1 else "No heart disease detected",
        }

    except Exception as err:
        return {"error": str(err)}, 500


=======
>>>>>>> 1359b50 (Initial commit for heart disease prediction)
# ----------------- REGISTER / LOGIN / LOGOUT -----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if not username or not password:
            flash("Username and password are required.", "warning")
            return render_template("register.html")

        if password != confirm:
            flash("Passwords do not match.", "danger")
            return render_template("register.html")

        if len(password) < 4:
            flash("Password must be at least 4 characters long.", "warning")
            return render_template("register.html")

        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE username = ?", (username,))
        existing = cur.fetchone()
        if existing:
            flash("Username already taken. Choose another.", "danger")
            return render_template("register.html")

        password_hash = generate_password_hash(password)
        cur.execute(
            """
            INSERT INTO users (username, password_hash, created_at)
            VALUES (?, ?, ?)
            """,
            (username, password_hash, datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()

        session["logged_in"] = True
        session["admin_user"] = username
        flash("Registration successful. You are now logged in.", "success")
        return redirect(url_for("predict_form"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, password_hash FROM users WHERE username = ?", (username,)
        )
        user = cur.fetchone()

        if user and check_password_hash(user[1], password):
            session["logged_in"] = True
            session["admin_user"] = username
            flash("Logged in successfully.", "success")
            return redirect(url_for("predict_form"))
        else:
            flash("Invalid username or password.", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))


# ----------------- ADMIN LOGS (OPTIONAL) -----------------
@app.route("/logs")
@login_required
def logs():
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, created_at, age, sex, cp, trestbps, chol, fbs,
               restecg, thalach, exang, oldpeak, slope, ca, thal,
               prediction, proba
        FROM predictions
        ORDER BY created_at DESC
        """
    )
    rows = cur.fetchall()
    return render_template("logs.html", rows=rows)


if __name__ == "__main__":
    app.run(debug=True)

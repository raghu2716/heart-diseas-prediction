import sqlite3
import os
import sys

db_path = os.path.join(os.getcwd(), 'heart_predictions.db')
print('DB path:', db_path)

if not os.path.exists(db_path):
    print('Database not found at', db_path)
    sys.exit(1)

conn = sqlite3.connect(db_path)
cur = conn.cursor()

try:
    cur.execute('SELECT count(*) FROM predictions')
    count = cur.fetchone()[0]
    print('Predictions rows:', count)

    cur.execute('SELECT id, created_at, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, prediction, proba FROM predictions ORDER BY created_at DESC LIMIT 5')
    rows = cur.fetchall()
    for r in rows:
        print(r)
except Exception as e:
    print('Error querying DB:', e)
finally:
    conn.close()

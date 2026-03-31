import sqlite3
import os
from datetime import datetime

db_path = os.path.join(os.getcwd(), 'heart_predictions.db')
print('DB path:', db_path)
conn = sqlite3.connect(db_path)
cur = conn.cursor()

created_at = datetime.now().isoformat(timespec='seconds')
try:
    cur.execute('''
        INSERT INTO predictions (
            created_at, age, sex, cp, trestbps, chol, fbs,
            restecg, thalach, exang, oldpeak, slope, ca, thal,
            prediction, proba
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        created_at, 55, 1, 0, 130, 250, 0,
        1, 150, 0, 2.3, 1, 0, 2,
        0, 0.12
    ))
    conn.commit()
    print('Inserted test row id', cur.lastrowid)
except Exception as e:
    print('Insert error:', e)
finally:
    conn.close()

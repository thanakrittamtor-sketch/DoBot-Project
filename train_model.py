# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def train_hand_model(csv_file, model_file, hand_name):
    """เทรนโมเดลสำหรับมือข้างหนึ่ง"""
    if not os.path.exists(csv_file):
        print(f"[SKIP] ไม่พบ {csv_file} -- ข้าม {hand_name}")
        return False

    print(f"\n{'='*50}")
    print(f"  เทรน AI: {hand_name}")
    print(f"{'='*50}")

    print(f"[INFO] อ่านข้อมูลจาก {csv_file} ...")
    df = pd.read_csv(csv_file)
    print(f"[INFO] ข้อมูลทั้งหมด: {len(df)} แถว, ท่าทาง: {df['class'].unique().tolist()}")

    # แยก features (X) และ labels (y)
    X = df.drop('class', axis=1)
    y = df['class']

    # แบ่ง Train 80% / Test 20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"[INFO] กำลังเทรน Random Forest ...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # ทดสอบความแม่นยำ
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"[OK] ความแม่นยำ {hand_name}: {accuracy * 100:.2f}%")

    # บันทึกโมเดล
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"[OK] บันทึกโมเดลลง '{model_file}'")

    return True


if __name__ == "__main__":
    print("=" * 50)
    print("  เทรน AI สำหรับระบบ 2 มือ")
    print("=" * 50)

    data_dir = os.path.dirname(os.path.abspath(__file__))

    # เทรนมือขวา (ทิศทาง)
    train_hand_model(os.path.join(data_dir, "right_hand_dataset.csv"),
                     "right_hand_model.pkl", "มือขวา (ทิศทาง)")

    # เทรนมือซ้าย (หัวดูด)
    train_hand_model(os.path.join(data_dir, "left_hand_dataset.csv"),
                     "left_hand_model.pkl", "มือซ้าย (หัวดูด)")

    print("\n" + "=" * 50)
    print("  เทรนเสร็จทั้งหมด!")
    print("  ขั้นตอนถัดไป: รัน run_model.py")
    print("=" * 50)

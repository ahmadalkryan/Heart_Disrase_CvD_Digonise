# train_local.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("🔄 إعادة تدريب النماذج - حل مشكلة التوافق")
print("=" * 60)

# ================================================
# 1. تحميل البيانات
# ================================================
print("\n📂 تحميل البيانات...")

df = pd.read_csv('heart_disease_cleaned_no_duplicates.csv')
print(f"✅ تم تحميل {df.shape[0]} صف")

# معالجة الأصفار في الكوليسترول
zero_chol = (df['cholesterol'] == 0).sum()
if zero_chol > 0:
    median_chol = df[df['cholesterol'] != 0]['cholesterol'].median()
    df.loc[df['cholesterol'] == 0, 'cholesterol'] = median_chol
    print(f"✅ تم استبدال {zero_chol} قيمة صفرية")

# ================================================
# 2. تعريف الميزات
# ================================================
ALL_FEATURES = ['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol', 
                'fasting blood sugar', 'resting ecg', 'max heart rate', 
                'exercise angina', 'oldpeak', 'ST slope']

FEATURES_TOP3 = ['ST slope', 'exercise angina', 'chest pain type']
FEATURES_TOP5 = ['ST slope', 'exercise angina', 'chest pain type', 'oldpeak', 'max heart rate']

X = df[ALL_FEATURES]
y = df['target']

# ================================================
# 3. تقسيم البيانات
# ================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# تطبيع البيانات
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n📊 تقسيم البيانات: تدريب={len(X_train)}, اختبار={len(X_test)}")

# ================================================
# 4. حفظ Scaler
# ================================================
os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/scaler.pkl')
print("✅ تم حفظ Scaler")

# ================================================
# 5. تدريب نموذج Top 3 (Logistic Regression)
# ================================================
print("\n🔄 تدريب نموذج Top 3 (3 ميزات)...")

indices_top3 = [ALL_FEATURES.index(f) for f in FEATURES_TOP3]
X_train_top3 = X_train_scaled[:, indices_top3]

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_top3, y_train)

model_top3 = LogisticRegression(max_iter=1000, random_state=42)
model_top3.fit(X_train_bal, y_train_bal)

# تقييم
y_pred = model_top3.predict(X_test_scaled[:, indices_top3])
acc = accuracy_score(y_test, y_pred)
print(f"   ✅ الدقة: {acc*100:.2f}%")

joblib.dump(model_top3, 'models/model_top3.pkl')
print("   ✅ حفظ: models/model_top3.pkl")

# ================================================
# 6. تدريب نموذج Top 5 (Gradient Boosting)
# ================================================
print("\n🔄 تدريب نموذج Top 5 (5 ميزات)...")

indices_top5 = [ALL_FEATURES.index(f) for f in FEATURES_TOP5]
X_train_top5 = X_train_scaled[:, indices_top5]

X_train_bal, y_train_bal = smote.fit_resample(X_train_top5, y_train)

model_top5 = GradientBoostingClassifier(n_estimators=100, random_state=42)
model_top5.fit(X_train_bal, y_train_bal)

# تقييم
y_pred = model_top5.predict(X_test_scaled[:, indices_top5])
acc = accuracy_score(y_test, y_pred)
print(f"   ✅ الدقة: {acc*100:.2f}%")

joblib.dump(model_top5, 'models/model_top5.pkl')
print("   ✅ حفظ: models/model_top5.pkl")

# ================================================
# 7. تدريب نموذج All 11 (Gradient Boosting)
# ================================================
print("\n🔄 تدريب نموذج All 11 (11 ميزات)...")

X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

model_all11 = GradientBoostingClassifier(n_estimators=100, random_state=42)
model_all11.fit(X_train_bal, y_train_bal)

# تقييم
y_pred = model_all11.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"   ✅ الدقة: {acc*100:.2f}%")

joblib.dump(model_all11, 'models/model_all11.pkl')
print("   ✅ حفظ: models/model_all11.pkl")

print("\n" + "=" * 60)
print("✅ تم إعادة تدريب جميع النماذج بنجاح!")
print("=" * 60)
print("\n📁 النماذج الجديدة保存在:")
print("   - models/model_top3.pkl")
print("   - models/model_top5.pkl")
print("   - models/model_all11.pkl")
print("   - models/scaler.pkl")
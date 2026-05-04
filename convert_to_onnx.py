# convert_to_onnx.py
# تحويل النماذج المدربة إلى صيغة ONNX

import joblib
import numpy as np
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import os

print("=" * 60)
print("🔄 تحويل النماذج إلى صيغة ONNX")
print("=" * 60)

# ================================================
# 1. تحميل النماذج المدربة
# ================================================
print("\n📂 تحميل النماذج المدربة...")

model_top3 = joblib.load('models/model_top3.pkl')
model_top5 = joblib.load('models/model_top5.pkl')
model_all11 = joblib.load('models/model_all11.pkl')
scaler = joblib.load('models/scaler.pkl')

print("✅ تم تحميل النماذج بنجاح")

# ================================================
# 2. تعريف الميزات لكل نموذج
# ================================================
ALL_FEATURES = ['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol', 
                'fasting blood sugar', 'resting ecg', 'max heart rate', 
                'exercise angina', 'oldpeak', 'ST slope']

FEATURES_TOP3 = ['ST slope', 'exercise angina', 'chest pain type']
FEATURES_TOP5 = ['ST slope', 'exercise angina', 'chest pain type', 'oldpeak', 'max heart rate']

# أبعاد الإدخال لكل نموذج
INPUT_DIMS = {
    'top3': len(FEATURES_TOP3),
    'top5': len(FEATURES_TOP5),
    'all11': len(ALL_FEATURES)
}

print(f"\n📊 أبعاد الإدخال:")
print(f"   نموذج Top 3: {INPUT_DIMS['top3']} ميزات")
print(f"   نموذج Top 5: {INPUT_DIMS['top5']} ميزات")
print(f"   نموذج All 11: {INPUT_DIMS['all11']} ميزات")

# ================================================
# 3. إنشاء مجلد للنماذج المحولة
# ================================================
os.makedirs('onnx_models', exist_ok=True)
print("\n✅ تم إنشاء مجلد onnx_models/")

# ================================================
# 4. تحويل نموذج Top 3
# ================================================
print("\n🔄 تحويل نموذج Top 3 (Logistic Regression)...")

try:
    initial_type = [('float_input', FloatTensorType([None, INPUT_DIMS['top3']]))]
    onnx_model = convert_sklearn(model_top3, initial_types=initial_type)
    
    # حفظ النموذج
    onnx_path = 'onnx_models/model_top3.onnx'
    with open(onnx_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    
    print(f"   ✅ تم حفظ: {onnx_path}")
    
    # التحقق من صحة النموذج
    session = ort.InferenceSession(onnx_path)
    print(f"   ✅ تم التحقق من النموذج (مدخلات: {session.get_inputs()[0].shape})")
    
except Exception as e:
    print(f"   ❌ فشل التحويل: {e}")

# ================================================
# 5. تحويل نموذج Top 5
# ================================================
print("\n🔄 تحويل نموذج Top 5 (Gradient Boosting)...")

try:
    initial_type = [('float_input', FloatTensorType([None, INPUT_DIMS['top5']]))]
    onnx_model = convert_sklearn(model_top5, initial_types=initial_type)
    
    # حفظ النموذج
    onnx_path = 'onnx_models/model_top5.onnx'
    with open(onnx_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    
    print(f"   ✅ تم حفظ: {onnx_path}")
    
    # التحقق من صحة النموذج
    session = ort.InferenceSession(onnx_path)
    print(f"   ✅ تم التحقق من النموذج (مدخلات: {session.get_inputs()[0].shape})")
    
except Exception as e:
    print(f"   ❌ فشل التحويل: {e}")

# ================================================
# 6. تحويل نموذج All 11
# ================================================
print("\n🔄 تحويل نموذج All 11 (Gradient Boosting)...")

try:
    initial_type = [('float_input', FloatTensorType([None, INPUT_DIMS['all11']]))]
    onnx_model = convert_sklearn(model_all11, initial_types=initial_type)
    
    # حفظ النموذج
    onnx_path = 'onnx_models/model_all11.onnx'
    with open(onnx_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    
    print(f"   ✅ تم حفظ: {onnx_path}")
    
    # التحقق من صحة النموذج
    session = ort.InferenceSession(onnx_path)
    print(f"   ✅ تم التحقق من النموذج (مدخلات: {session.get_inputs()[0].shape})")
    
except Exception as e:
    print(f"   ❌ فشل التحويل: {e}")

# ================================================
# 7. حفظ معاملات Scaler كـ JSON (للاستخدام مع ONNX)
# ================================================
print("\n📊 حفظ معاملات Scaler...")

import json
scaler_params = {
    'mean': scaler.mean_.tolist(),
    'scale': scaler.scale_.tolist(),
    'features': ALL_FEATURES
}

with open('onnx_models/scaler_params.json', 'w') as f:
    json.dump(scaler_params, f, indent=2)

print("✅ تم حفظ: onnx_models/scaler_params.json")

# ================================================
# 8. اختبار النماذج المحولة
# ================================================
print("\n" + "=" * 60)
print("🧪 اختبار النماذج المحولة")
print("=" * 60)

# تحميل البيانات للاختبار
import pandas as pd
df = pd.read_csv('heart_disease_cleaned_no_duplicates.csv')

# اختيار عينة للاختبار
test_sample = df.iloc[0].to_dict()
actual = df.iloc[0]['target']

print(f"\n📊 بيانات الاختبار (المريض الأول):")
print(f"   العمر: {test_sample['age']}")
print(f"   الجنس: {'ذكر' if test_sample['sex']==1 else 'أنثى'}")
print(f"   نوع ألم الصدر: {test_sample['chest pain type']}")
print(f"   ضغط الدم: {test_sample['resting bp s']}")
print(f"   الكوليسترول: {test_sample['cholesterol']}")
print(f"   أقصى معدل للقلب: {test_sample['max heart rate']}")
print(f"   ST slope: {test_sample['ST slope']}")
print(f"   الحالة الفعلية: {'مريض' if actual==1 else 'سليم'}")

# دالة التنبؤ باستخدام ONNX
def predict_onnx(model_path, features, scaler_params):
    """تنبؤ باستخدام نموذج ONNX"""
    import onnxruntime as ort
    
    # تحميل النموذج
    session = ort.InferenceSession(model_path)
    
    # تطبيع البيانات
    full_data = np.zeros(11)
    feature_indices = {
        'ST slope': 10, 'exercise angina': 8, 'chest pain type': 2,
        'oldpeak': 9, 'max heart rate': 7, 'age': 0, 'sex': 1,
        'resting bp s': 3, 'cholesterol': 4, 'fasting blood sugar': 5,
        'resting ecg': 6
    }
    
    for f_name, f_value in features.items():
        if f_name in feature_indices:
            full_data[feature_indices[f_name]] = f_value
    
    # تطبيع باستخدام معاملات Scaler
    normalized = (full_data - scaler_params['mean']) / scaler_params['scale']
    
    # استخراج الميزات المطلوبة (حسب النموذج)
    if 'top3' in model_path:
        required_features = ['ST slope', 'exercise angina', 'chest pain type']
        indices = [feature_indices[f] for f in required_features]
        input_data = normalized[indices].reshape(1, -1).astype(np.float32)
    elif 'top5' in model_path:
        required_features = ['ST slope', 'exercise angina', 'chest pain type', 'oldpeak', 'max heart rate']
        indices = [feature_indices[f] for f in required_features]
        input_data = normalized[indices].reshape(1, -1).astype(np.float32)
    else:
        input_data = normalized.reshape(1, -1).astype(np.float32)
    
    # تنبؤ
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})
    
    if len(outputs[0].shape) == 2:
        pred = np.argmax(outputs[0], axis=1)[0]
        prob = outputs[0][0][1]
    else:
        pred = int(outputs[0][0] > 0.5)
        prob = float(outputs[0][0])
    
    return pred, prob

# اختبار النماذج
print("\n📊 نتائج الاختبار:")
print("-" * 50)

# Top 3
try:
    pred, prob = predict_onnx('onnx_models/model_top3.onnx', test_sample, scaler_params)
    print(f"نموذج Top 3:   {'🔴 مريض' if pred==1 else '🟢 سليم'} (ثقة: {prob*100:.1f}%)")
except Exception as e:
    print(f"نموذج Top 3:   ❌ فشل: {e}")

# Top 5
try:
    pred, prob = predict_onnx('onnx_models/model_top5.onnx', test_sample, scaler_params)
    print(f"نموذج Top 5:   {'🔴 مريض' if pred==1 else '🟢 سليم'} (ثقة: {prob*100:.1f}%)")
except Exception as e:
    print(f"نموذج Top 5:   ❌ فشل: {e}")

# All 11
try:
    pred, prob = predict_onnx('onnx_models/model_all11.onnx', test_sample, scaler_params)
    print(f"نموذج All 11:  {'🔴 مريض' if pred==1 else '🟢 سليم'} (ثقة: {prob*100:.1f}%)")
except Exception as e:
    print(f"نموذج All 11:  ❌ فشل: {e}")

print("\n" + "=" * 60)
print("✅ اكتمل التحويل!")
print("=" * 60)
print("\n📁 النماذج المحولة保存在:")
print("   - onnx_models/model_top3.onnx")
print("   - onnx_models/model_top5.onnx")
print("   - onnx_models/model_all11.onnx")
print("   - onnx_models/scaler_params.json")
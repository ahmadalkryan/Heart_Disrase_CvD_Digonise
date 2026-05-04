# app_flask.py
# Heart Disease Prediction API using Flask

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import os

# إنشاء تطبيق Flask
app = Flask(__name__)
CORS(app)  # السماح بالطلبات من أي مصدر

# ================================================
# تحميل النماذج
# ================================================
print("📂 جاري تحميل النماذج...")

model_top3 = joblib.load('models/model_top3.pkl')
model_top5 = joblib.load('models/model_top5.pkl')
model_all11 = joblib.load('models/model_all11.pkl')
scaler = joblib.load('models/scaler.pkl')

print("✅ تم تحميل جميع النماذج بنجاح!")

# ================================================
# تعريف الميزات
# ================================================
ALL_FEATURES = ['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol', 
                'fasting blood sugar', 'resting ecg', 'max heart rate', 
                'exercise angina', 'oldpeak', 'ST slope']

FEATURES_TOP3 = ['ST slope', 'exercise angina', 'chest pain type']
FEATURES_TOP5 = ['ST slope', 'exercise angina', 'chest pain type', 'oldpeak', 'max heart rate']

# معلومات النماذج
MODELS_INFO = {
    'top3': {
        'name': 'النموذج السريع (3 ميزات)',
        'features': FEATURES_TOP3,
        'model_type': 'Logistic Regression',
        'accuracy': 83.15
    },
    'top5': {
        'name': 'النموذج المتوازن (5 ميزات)',
        'features': FEATURES_TOP5,
        'model_type': 'Gradient Boosting',
        'accuracy': 89.32
    },
    'all11': {
        'name': 'النموذج الشامل (11 ميزة)',
        'features': ALL_FEATURES,
        'model_type': 'Gradient Boosting',
        'accuracy': 88.04
    }
}

# ================================================
# دالة التنبؤ العامة
# ================================================
def predict_disease(model, model_features, patient_data, scaler):
    """تنبؤ باستخدام النموذج المحدد"""
    
    # إنشاء متجه كامل (11 ميزة) للتطبيع
    full_X = np.zeros((1, len(ALL_FEATURES)))
    
    for i, f in enumerate(ALL_FEATURES):
        if f in model_features:
            full_X[0, i] = patient_data.get(f, 0)
        elif f == 'resting ecg':
            full_X[0, i] = 0
        elif f == 'fasting blood sugar':
            full_X[0, i] = 0
        elif f == 'cholesterol':
            full_X[0, i] = 200
        elif f == 'age':
            full_X[0, i] = 50
        elif f == 'sex':
            full_X[0, i] = 1
        elif f == 'resting bp s':
            full_X[0, i] = 120
    
    # تطبيع البيانات
    X_scaled = scaler.transform(full_X)
    
    # استخراج الميزات المطلوبة
    indices = [ALL_FEATURES.index(f) for f in model_features]
    X_final = X_scaled[:, indices]
    
    # التنبؤ
    prediction = model.predict(X_final)[0]
    probability = model.predict_proba(X_final)[0][1]
    
    return prediction, probability

# ================================================
# واجهات API
# ================================================

@app.route('/')
def index():
    """الصفحة الرئيسية - واجهة المستخدم"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """فحص صحة الخادم والنماذج"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': True,
        'models_available': list(MODELS_INFO.keys())
    })

@app.route('/models', methods=['GET'])
def get_models_info():
    """الحصول على معلومات جميع النماذج"""
    return jsonify({
        'success': True,
        'models': MODELS_INFO
    })

@app.route('/predict/<model_name>', methods=['POST'])
def predict(model_name):
    """
    التنبؤ باستخدام النموذج المحدد
    
    POST /predict/top3
    POST /predict/top5
    POST /predict/all11
    
    الجسم (JSON):
    {
        "age": 55,
        "sex": 1,
        "chest pain type": 4,
        "resting bp s": 140,
        "cholesterol": 240,
        "fasting blood sugar": 0,
        "resting ecg": 0,
        "max heart rate": 150,
        "exercise angina": 0,
        "oldpeak": 1.5,
        "ST slope": 2
    }
    """
    
    # التحقق من وجود النموذج
    if model_name not in MODELS_INFO:
        return jsonify({
            'success': False,
            'error': f'النموذج {model_name} غير موجود. النماذج المتاحة: {list(MODELS_INFO.keys())}'
        }), 404
    
    # الحصول على البيانات من الطلب
    data = request.get_json()
    
    if not data:
        return jsonify({
            'success': False,
            'error': 'لم يتم إرسال بيانات'
        }), 400
    
    # اختيار النموذج المناسب
    if model_name == 'top3':
        model = model_top3
        model_features = FEATURES_TOP3
    elif model_name == 'top5':
        model = model_top5
        model_features = FEATURES_TOP5
    else:
        model = model_all11
        model_features = ALL_FEATURES
    
    try:
        # التنبؤ
        prediction, probability = predict_disease(model, model_features, data, scaler)
        
        # تحديد مستوى الخطر
        if probability > 0.7:
            risk_level = "HIGH"
            risk_ar = "عالي 🔴"
        elif probability > 0.3:
            risk_level = "MEDIUM"
            risk_ar = "متوسط 🟡"
        else:
            risk_level = "LOW"
            risk_ar = "منخفض 🟢"
        
        # النتيجة
        result = {
            'success': True,
            'model_used': MODELS_INFO[model_name]['name'],
            'model_type': MODELS_INFO[model_name]['model_type'],
            'model_accuracy': MODELS_INFO[model_name]['accuracy'],
            'prediction': int(prediction),
            'result': 'DISEASE' if prediction == 1 else 'HEALTHY',
            'result_ar': 'مريض 🔴' if prediction == 1 else 'سليم 🟢',
            'probability': float(probability),
            'probability_percent': f"{probability*100:.1f}%",
            'risk_level': risk_level,
            'risk_level_ar': risk_ar,
            'recommendation_ar': 'يرجى مراجعة الطبيب المختص' if prediction == 1 else 'حافظ على نمط حياة صحي'
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/predict/auto', methods=['POST'])
def predict_auto():
    """
    التنبؤ التلقائي - يختار النموذج المناسب بناءً على البيانات المتوفرة
    
    POST /predict/auto
    """
    data = request.get_json()
    
    if not data:
        return jsonify({'success': False, 'error': 'لم يتم إرسال بيانات'}), 400
    
    # التحقق من الميزات المتوفرة
    has_top3 = all(f in data for f in FEATURES_TOP3)
    has_top5 = all(f in data for f in FEATURES_TOP5)
    has_all11 = all(f in data for f in ALL_FEATURES)
    
    if has_all11:
        model_name = 'all11'
        model = model_all11
        model_features = ALL_FEATURES
    elif has_top5:
        model_name = 'top5'
        model = model_top5
        model_features = FEATURES_TOP5
    elif has_top3:
        model_name = 'top3'
        model = model_top3
        model_features = FEATURES_TOP3
    else:
        return jsonify({
            'success': False,
            'error': 'البيانات غير كافية للتنبؤ. الميزات المطلوبة: ' + ', '.join(FEATURES_TOP3)
        }), 400
    
    try:
        prediction, probability = predict_disease(model, model_features, data, scaler)
        
        return jsonify({
            'success': True,
            'model_used': MODELS_INFO[model_name]['name'],
            'prediction': int(prediction),
            'result': 'DISEASE' if prediction == 1 else 'HEALTHY',
            'probability': float(probability),
            'risk_level': 'HIGH' if probability > 0.7 else ('MEDIUM' if probability > 0.3 else 'LOW')
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ================================================
# تشغيل التطبيق
# ================================================
if __name__ == '__main__':
    print("=" * 60)
    print("❤️ Heart Disease Prediction API")
    print("=" * 60)
    print(f"Models loaded:")
    print(f"  - Top 3 (3 features): Logistic Regression")
    print(f"  - Top 5 (5 features): Gradient Boosting")
    print(f"  - All 11 (11 features): Gradient Boosting")
    print("=" * 60)
    print("\n📍 API Endpoints:")
    print("   GET  /              - Web interface")
    print("   GET  /health        - Health check")
    print("   GET  /models        - List all models")
    print("   POST /predict/top3  - Predict using Top 3 model")
    print("   POST /predict/top5  - Predict using Top 5 model")
    print("   POST /predict/all11 - Predict using All 11 model")
    print("   POST /predict/auto  - Auto-select model")
    print("=" * 60)
    print("\n🚀 Starting Flask server...")
    print("   http://localhost:5000")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
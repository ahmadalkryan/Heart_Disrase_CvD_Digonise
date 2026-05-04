# app.py
# Heart Disease Prediction App - Updated for locally trained models

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px

# إعداد صفحة التطبيق
st.set_page_config(
    page_title="تشخيص أمراض القلب",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================
# تحميل النماذج (باستخدام النماذج المدربة محلياً)
# ================================================
@st.cache_resource
def load_pickle_models():
    """تحميل نماذج Pickle المدربة محلياً"""
    try:
        model_top3 = joblib.load('models/model_top3.pkl')
        model_top5 = joblib.load('models/model_top5.pkl')
        model_all11 = joblib.load('models/model_all11.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model_top3, model_top5, model_all11, scaler
    except FileNotFoundError as e:
        st.error(f"⚠️ ملفات النماذج غير موجودة: {e}")
        st.info("""
        **الملفات المطلوبة في مجلد models/:**
        - model_top3.pkl
        - model_top5.pkl
        - model_all11.pkl
        - scaler.pkl
        
        قم بتشغيل ملف train_local.py لتدريب النماذج
        """)
        return None, None, None, None

@st.cache_resource
def load_onnx_models():
    """تحميل نماذج ONNX (اختياري)"""
    try:
        import onnxruntime as ort
        model_top3 = ort.InferenceSession('onnx_models/model_top3.onnx')
        model_top5 = ort.InferenceSession('onnx_models/model_top5.onnx')
        model_all11 = ort.InferenceSession('onnx_models/model_all11.onnx')
        return model_top3, model_top5, model_all11, True
    except Exception as e:
        return None, None, None, False

# ================================================
# تعريف الميزات
# ================================================
ALL_FEATURES = ['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol', 
                'fasting blood sugar', 'resting ecg', 'max heart rate', 
                'exercise angina', 'oldpeak', 'ST slope']

FEATURES_TOP3 = ['ST slope', 'exercise angina', 'chest pain type']
FEATURES_TOP5 = ['ST slope', 'exercise angina', 'chest pain type', 'oldpeak', 'max heart rate']

# التسميات العربية
FEATURES_AR = {
    'ST slope': 'ميل مقطع ST',
    'exercise angina': 'ذبحة أثناء الجهد',
    'chest pain type': 'نوع ألم الصدر',
    'oldpeak': 'انخفاض ST (oldpeak)',
    'max heart rate': 'أقصى معدل لضربات القلب',
    'age': 'العمر',
    'sex': 'الجنس',
    'resting bp s': 'ضغط الدم الانقباضي',
    'cholesterol': 'الكوليسترول',
    'fasting blood sugar': 'سكر الدم الصائم',
    'resting ecg': 'تخطيط القلب'
}

# أوصاف الميزات
FEATURES_DESC = {
    'ST slope': '1 = مائل للأعلى (طبيعي), 2 = مسطح (مشبوه), 3 = مائل للأسفل (خطير)',
    'exercise angina': 'هل يعاني المريض من ألم في الصدر عند بذل مجهود؟',
    'chest pain type': '1=ذبحة نموذجية, 2=ذبحة غير نموذجية, 3=ألم غير ذبحي, 4=بدون أعراض',
    'oldpeak': 'انخفاض مقطع ST أثناء اختبار الجهد (بالملليمتر)',
    'max heart rate': 'أقصى معدل لضربات القلب أثناء اختبار الجهد (نبضة/دقيقة)'
}

# معلومات النماذج (تم تحديثها بناءً على أداء النماذج الجديدة)
MODELS_INFO = {
    'النموذج السريع (3 ميزات)': {
        'key': 'top3',
        'features': FEATURES_TOP3,
        'n_features': 3,
        'model_type': 'Logistic Regression',
        'accuracy': 83.15,
        'f1_score': 0.8502,
        'description': '⚡ أسرع نموذج - يستخدم 3 ميزات فقط',
        'icon': '⚡'
    },
    'النموذج المتوازن (5 ميزات)': {
        'key': 'top5',
        'features': FEATURES_TOP5,
        'n_features': 5,
        'model_type': 'Gradient Boosting',
        'accuracy': 89.32,
        'f1_score': 0.9039,
        'description': '🎯 أفضل توازن - دقة عالية مع 5 ميزات',
        'icon': '⭐'
    },
    'النموذج الشامل (11 ميزة)': {
        'key': 'all11',
        'features': ALL_FEATURES,
        'n_features': 11,
        'model_type': 'Gradient Boosting',
        'accuracy': 88.04,
        'f1_score': 0.8911,
        'description': '🏆 أعلى دقة - يستخدم جميع الميزات',
        'icon': '🏆'
    }
}

# القيم الافتراضية
DEFAULT_VALUES = {
    'ST slope': 2,
    'exercise angina': 0,
    'chest pain type': 4,
    'oldpeak': 0.6,
    'max heart rate': 150,
    'age': 55,
    'sex': 1,
    'resting bp s': 120,
    'cholesterol': 200,
    'fasting blood sugar': 0,
    'resting ecg': 0
}

# ================================================
# دوال التنبؤ
# ================================================
def predict_pickle(model, model_features, patient_data, scaler):
    """تنبؤ باستخدام نموذج Pickle"""
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
    
    X_scaled = scaler.transform(full_X)
    indices = [ALL_FEATURES.index(f) for f in model_features]
    X_final = X_scaled[:, indices]
    
    pred = model.predict(X_final)[0]
    prob = model.predict_proba(X_final)[0][1]
    return pred, prob

def predict_onnx(ort_session, model_features, patient_data, scaler):
    """تنبؤ باستخدام نموذج ONNX"""
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
    
    X_scaled = scaler.transform(full_X)
    indices = [ALL_FEATURES.index(f) for f in model_features]
    X_final = X_scaled[:, indices].astype(np.float32)
    
    inputs = {ort_session.get_inputs()[0].name: X_final}
    outputs = ort_session.run(None, inputs)
    
    if len(outputs) >= 1:
        if len(outputs[0].shape) == 2:
            pred = np.argmax(outputs[0], axis=1)[0]
            prob = outputs[0][0][1]
        else:
            pred = int(outputs[0][0] > 0.5)
            prob = float(outputs[0][0])
    else:
        pred = 0
        prob = 0.0
    
    return pred, prob

# ================================================
# عرض النتيجة
# ================================================
def display_result(prediction, probability, model_name, model_info):
    """عرض نتيجة التنبؤ بشكل جذاب"""
    
    if prediction == 1:
        st.markdown(f"""
        <div style='background-color: #ffebee; padding: 25px; border-radius: 15px; border-right: 5px solid #e74c3c;'>
            <h2 style='color: #e74c3c; margin: 0;'>⚠️ نتيجة الفحص</h2>
            <h1 style='color: #e74c3c; margin: 10px 0;'>🔴 ارتفاع خطر الإصابة!</h1>
            <p style='font-size: 18px;'>يُنصح بمراجعة الطبيب المختص لإجراء فحوصات إضافية</p>
            <hr>
            <p style='font-size: 14px; color: gray;'>النموذج المستخدم: {model_name} | دقة النموذج: {model_info['accuracy']:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("نسبة الثقة", f"{probability:.1%}", delta="احتمال وجود مرض", delta_color="inverse")
        with col2:
            st.metric("مستوى الخطر", "عالي 🔴")
        
        with st.expander("📋 نصائح وإرشادات", expanded=True):
            st.markdown("""
            **ما يجب فعله:**
            - ✅ راجع طبيب قلب في أقرب وقت
            - ✅ التزم بالأدوية الموصوفة
            - ✅ اتبع نظاماً غذائياً صحياً منخفض الدهون
            - ✅ مارس النشاط البدني بانتظام
            - ✅ أقلع عن التدخين إذا كنت مدخناً
            """)
    else:
        st.markdown(f"""
        <div style='background-color: #e8f5e9; padding: 25px; border-radius: 15px; border-right: 5px solid #2ecc71;'>
            <h2 style='color: #2ecc71; margin: 0;'>✅ نتيجة الفحص</h2>
            <h1 style='color: #2ecc71; margin: 10px 0;'>🟢 خطر منخفض</h1>
            <p style='font-size: 18px;'>النتائج تشير إلى أن خطر الإصابة بأمراض القلب منخفض</p>
            <hr>
            <p style='font-size: 14px; color: gray;'>النموذج المستخدم: {model_name} | دقة النموذج: {model_info['accuracy']:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("نسبة الثقة", f"{1-probability:.1%}", delta="احتمال السلامة")
        with col2:
            st.metric("مستوى الخطر", "منخفض 🟢")
        
        with st.expander("💚 نصائح للحفاظ على الصحة", expanded=True):
            st.markdown("""
            **للحفاظ على قلب سليم:**
            - ✅ حافظ على نشاطك البدني (30 دقيقة يومياً)
            - ✅ تناول طعاماً صحياً متوازناً
            - ✅ تجنب التدخين والكحول
            - ✅ حافظ على وزن صحي
            - ✅ قم بفحوصات دورية
            """)

# ================================================
# رسم نسبة الخطر
# ================================================
def create_risk_chart(probability):
    """إنشاء مخطط دائري لنسبة الخطر"""
    fig = go.Figure(data=[go.Pie(
        labels=['خطر مرتفع', 'خطر منخفض'],
        values=[probability, 1-probability],
        hole=.6,
        marker_colors=['#e74c3c', '#2ecc71'],
        textinfo='percent'
    )])
    
    fig.update_layout(
        title="نسبة خطر الإصابة",
        height=350,
        showlegend=True,
        annotations=[dict(text=f'{probability:.1%}', x=0.5, y=0.5, font_size=24, showarrow=False)]
    )
    return fig

# ================================================
# الواجهة الرئيسية
# ================================================
def main():
    st.title("❤️ نظام تشخيص أمراض القلب")
    st.markdown("#### أداة مساعدة للكشف المبكر عن خطر الإصابة بأمراض القلب")
    st.markdown("---")
    
    # اختيار تنسيق النموذج
    st.sidebar.markdown("## ⚙️ الإعدادات")
    
    model_format = st.sidebar.radio(
        "تنسيق النموذج:",
        options=["Pickle (.pkl) - Python", "ONNX (.onnx) - Cross-platform"],
        index=0,
        help="Pickle أسرع وأفضل توافق مع Python، ONNX للاستخدام عبر المنصات"
    )
    
    # تحميل النماذج حسب التنسيق المختار
    use_onnx = "ONNX" in model_format
    
    if use_onnx:
        model_top3, model_top5, model_all11, onnx_available = load_onnx_models()
        if not onnx_available:
            st.warning("⚠️ ONNX Runtime غير متوفر. جارٍ استخدام نماذج Pickle بدلاً من ذلك.")
            model_top3, model_top5, model_all11, scaler = load_pickle_models()
            use_onnx = False
        else:
            scaler = joblib.load('models/scaler.pkl')
    else:
        model_top3, model_top5, model_all11, scaler = load_pickle_models()
    
    if model_top3 is None or scaler is None:
        st.stop()
    
    # الشريط الجانبي - اختيار النموذج
    with st.sidebar:
        st.markdown("## 🎯 اختيار النموذج")
        
        selected_model_name = st.selectbox(
            "اختر النموذج المناسب:",
            options=list(MODELS_INFO.keys()),
            index=1,
            help="اختر النموذج حسب دقتك والبيانات المتوفرة"
        )
        
        model_info = MODELS_INFO[selected_model_name]
        st.markdown(f"{model_info['icon']} **{model_info['description']}**")
        
        # عرض معلومات النموذج
        st.markdown("---")
        st.markdown("### 📊 معلومات النموذج")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("الدقة", f"{model_info['accuracy']:.1f}%")
            st.metric("F1-Score", f"{model_info['f1_score']:.3f}")
        with col2:
            st.metric("الميزات", model_info['n_features'])
            st.metric("النوع", model_info['model_type'][:15])
        
        st.markdown("---")
        st.markdown("### 📋 الميزات المطلوبة")
        for f in model_info['features']:
            st.markdown(f"- {FEATURES_AR.get(f, f)}")
        
        st.markdown("---")
        st.markdown("### 📊 تفسير النتائج")
        st.markdown("""
        | الاحتمالية | مستوى الخطر |
        |------------|-------------|
        | > 70% | 🔴 عالي |
        | 30-70% | 🟡 متوسط |
        | < 30% | 🟢 منخفض |
        """)
        
        st.markdown("---")
        st.markdown("### ⚠️ تنبيه")
        st.markdown("> هذه الأداة لأغراض توعوية فقط. لا تُغني عن استشارة الطبيب.")
    
    # تحديد النموذج والميزات
    if model_info['key'] == 'top3':
        current_model = model_top3
        model_features = FEATURES_TOP3
    elif model_info['key'] == 'top5':
        current_model = model_top5
        model_features = FEATURES_TOP5
    else:
        current_model = model_all11
        model_features = ALL_FEATURES
    
    # واجهة الإدخال والنتيجة
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📝 أدخل بيانات المريض")
        
        input_data = {}
        
        for feature in model_features:
            if feature == 'ST slope':
                input_data[feature] = st.selectbox(
                    FEATURES_AR[feature],
                    options=[1, 2, 3],
                    format_func=lambda x: {1: "1 - مائل للأعلى (طبيعي)", 
                                           2: "2 - مسطح (مشبوه)", 
                                           3: "3 - مائل للأسفل (خطير)"}[x],
                    index=DEFAULT_VALUES[feature]-1,
                    help=FEATURES_DESC[feature]
                )
            
            elif feature == 'exercise angina':
                input_data[feature] = st.radio(
                    FEATURES_AR[feature],
                    options=[0, 1],
                    format_func=lambda x: "نعم" if x == 1 else "لا",
                    horizontal=True,
                    index=DEFAULT_VALUES[feature]
                )
            
            elif feature == 'chest pain type':
                input_data[feature] = st.selectbox(
                    FEATURES_AR[feature],
                    options=[1, 2, 3, 4],
                    format_func=lambda x: {1: "1 - ذبحة نموذجية", 
                                           2: "2 - ذبحة غير نموذجية", 
                                           3: "3 - ألم غير ذبحي", 
                                           4: "4 - بدون أعراض"}[x],
                    index=DEFAULT_VALUES[feature]-1
                )
            
            elif feature == 'oldpeak':
                input_data[feature] = st.slider(
                    FEATURES_AR[feature],
                    min_value=-2.6, max_value=6.2, 
                    value=DEFAULT_VALUES[feature], 
                    step=0.1,
                    help=FEATURES_DESC[feature]
                )
            
            elif feature == 'max heart rate':
                input_data[feature] = st.number_input(
                    FEATURES_AR[feature],
                    min_value=60, max_value=202, 
                    value=DEFAULT_VALUES[feature],
                    help=FEATURES_DESC[feature]
                )
            
            elif feature == 'age':
                input_data[feature] = st.number_input("العمر (سنوات)", min_value=20, max_value=100, value=55)
            elif feature == 'sex':
                input_data[feature] = st.radio("الجنس", options=[0, 1], format_func=lambda x: "أنثى" if x == 0 else "ذكر", horizontal=True)
            elif feature == 'resting bp s':
                input_data[feature] = st.number_input("ضغط الدم الانقباضي (mmHg)", min_value=80, max_value=200, value=120)
            elif feature == 'cholesterol':
                input_data[feature] = st.number_input("الكوليسترول (mg/dL)", min_value=100, max_value=600, value=200)
            elif feature == 'fasting blood sugar':
                input_data[feature] = st.radio("سكر الدم الصائم", options=[0, 1], format_func=lambda x: "طبيعي" if x == 0 else "مرتفع", horizontal=True)
            elif feature == 'resting ecg':
                input_data[feature] = st.selectbox("نتيجة تخطيط القلب", options=[0, 1, 2], 
                                                   format_func=lambda x: {0: "طبيعي", 1: "اضطراب", 2: "تضخم"}[x])
        
        predict_button = st.button("🔍 تشخيص", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### 🩺 نتيجة التشخيص")
        
        if predict_button:
            with st.spinner("جاري التحليل..."):
                if use_onnx:
                    prediction, probability = predict_onnx(current_model, model_features, input_data, scaler)
                else:
                    prediction, probability = predict_pickle(current_model, model_features, input_data, scaler)
                
                display_result(prediction, probability, selected_model_name, model_info)
                st.plotly_chart(create_risk_chart(probability), use_container_width=True)
                
                with st.expander("📋 ملخص البيانات المدخلة", expanded=False):
                    for f in model_features:
                        st.write(f"**{FEATURES_AR.get(f, f)}:** {input_data[f]}")
        else:
            st.info("👈 قم بإدخال بيانات المريض ثم اضغط على 'تشخيص'")
            
            st.markdown(f"""
            <div style='text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;'>
                <p style='font-size: 48px;'>{model_info['icon']}</p>
                <p style='font-size: 18px; font-weight: bold;'>نظام تشخيص أمراض القلب</p>
                <p style='font-size: 14px;'>✅ النموذج النشط: {selected_model_name}</p>
                <p style='font-size: 12px;'>دقة النموذج: {model_info['accuracy']:.1f}% | F1: {model_info['f1_score']:.3f}</p>
                <hr style='background: white;'>
                <p style='font-size: 12px;'>التنسيق: {model_format}</p>
            </div>
            """, unsafe_allow_html=True)

# ================================================
# تشغيل التطبيق
# ================================================
if __name__ == "__main__":
    main()
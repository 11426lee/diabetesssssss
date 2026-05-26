import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# 1. 페이지 설정 및 사용자 정의 CSS 주입 (의료/헬스케어 테마)
# -----------------------------
st.set_page_config(
    page_title="당뇨 예측 프로그램",
    page_icon="🩺",
    layout="centered"
)

# 병원/클리닉 느낌의 깔끔하고 신뢰감 있는 CSS 디자인 주입
st.markdown("""
    <style>
    /* 전체 배경색 - 아주 연한 파스텔 블루/그레이 (청결한 느낌) */
    .stApp {
        background-color: #F0F4F8;
    }
    
    /* 기본 텍스트 색상 - 짙은 네이비/그레이 */
    p, span, label {
        color: #2C3E50 !important;
    }

    /* 메인 타이틀 스타일 */
    .medical-title {
        color: #154360; /* 짙은 네이비 블루 */
        font-weight: bold;
        text-align: center;
        border-bottom: 3px solid #3498DB; /* 밝은 파란색 밑줄 */
        padding-bottom: 15px;
        margin-bottom: 25px;
    }

    /* 입력/결과 컨테이너(카드) 스타일 */
    .medical-card {
        background-color: #FFFFFF; /* 흰색 배경 */
        border: 1px solid #D6DBDF; /* 연한 회색 테두리 */
        border-top: 4px solid #2980B9; /* 상단 포인트 컬러 (의료 블루) */
        border-radius: 10px;
        padding: 25px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.05); /* 부드러운 그림자 */
        margin-bottom: 25px;
    }

    /* 서브 헤더 스타일 */
    .medical-subheader {
        color: #2980B9;
        font-weight: bold;
        margin-bottom: 15px;
        font-size: 1.2rem;
    }

    /* 예측하기 버튼 스타일 */
    .stButton>button {
        background-color: #2980B9 !important; /* 의료 블루 */
        color: white !important;
        border-radius: 8px;
        border: none;
        width: 100%; /* 버튼을 넓게 */
        font-size: 1.1rem;
        font-weight: bold;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1A5276 !important; /* 호버시 짙은 네이비 */
        transform: translateY(-2px);
        box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
    }
    
    /* 프로그레스 바(게이지) 색상 덮어쓰기 */
    .stProgress > div > div > div {
        background-color: #3498DB;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# 2. 모델 & 스케일러 불러오기 (기존 로직 유지)
# -----------------------------
log_model_eng = joblib.load("diabetes_model.pkl")
scaler = joblib.load("diabetes_scaler.pkl")

# -----------------------------
# 3. Streamlit UI (디자인 적용)
# -----------------------------

st.markdown('<h1 class="medical-title">🩺 당뇨 예측 프로그램</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #7F8C8D;'>생활 정보를 입력하면 당뇨 여부를 예측합니다.</p>", unsafe_allow_html=True)

st.write("") # 간격 띄우기

# 입력창을 카드 디자인으로 감싸기
st.markdown('<div class="medical-card">', unsafe_allow_html=True)
st.markdown('<div class="medical-subheader">📋 환자 데이터 입력</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("임신횟수", min_value=0, step=1)
    glucose = st.number_input("혈당", min_value=0.0)
    bp = st.number_input("혈압", min_value=0.0)
    skin = st.number_input("피부두께", min_value=0.0)

with col2:
    insulin = st.number_input("인슐린", min_value=0.0)
    bmi = st.number_input("체질량지수(BMI)", min_value=0.0)
    dpf = st.number_input("가족력", min_value=0.0)
    age = st.number_input("나이", min_value=0, step=1)

st.markdown('</div>', unsafe_allow_html=True) # 입력 카드 닫기

st.write("") # 간격 띄우기

# 예측 버튼
if st.button("🔍 당뇨 예측하기", use_container_width=True):

    # 사용자 입력 데이터
    input_data = pd.DataFrame(
        [[preg, glucose, bp, skin, insulin, bmi, dpf, age]],
        columns=[
            '임신횟수',
            '혈당',
            '혈압',
            '피부두께',
            '인슐린',
            '체질량지수',
            '가족력',
            '나이'
        ]
    )

    # 파생 변수 생성 (기존 로직 100% 유지)
    input_data['비만위험'] = (input_data['체질량지수'] >= 30).astype(int)
    input_data['고혈당'] = (input_data['혈당'] >= 140).astype(int)
    input_data['고령'] = (input_data['나이'] >= 50).astype(int)
    input_data['대사위험'] = (
        (input_data['체질량지수'] >= 25).astype(int)
        + (input_data['혈당'] >= 130).astype(int)
    )

    # 스케일링
    input_scaled = scaler.transform(input_data)

    # 예측
    predicted = log_model_eng.predict(input_scaled)
    prob = log_model_eng.predict_proba(input_scaled)

    diabetes_prob = prob[0][1] * 100

    # 결과 화면을 새로운 카드로 감싸서 출력
    st.markdown('<div class="medical-card" style="border-top: 4px solid #27AE60;">', unsafe_allow_html=True)
    st.markdown('<div class="medical-subheader">📊 예측 결과</div>', unsafe_allow_html=True)

    if predicted[0] == 1:
        st.error("⚠️ 당뇨 가능성이 높습니다.")
    else:
        st.success("✅ 정상 가능성이 높습니다.")

    st.metric(
        label="당뇨 확률",
        value=f"{diabetes_prob:.1f}%"
    )

    st.progress(int(diabetes_prob))

    with st.expander("입력 데이터 및 파생 변수 보기"):
        st.dataframe(input_data)
        
    st.markdown('</div>', unsafe_allow_html=True) # 결과 카드 닫기
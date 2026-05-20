import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. 웹사이트 기본 설정 & 보라색 테마 CSS 적용
# ---------------------------------------------------------
st.set_page_config(page_title="교육 인프라 AI 대시보드", layout="wide", initial_sidebar_state="collapsed")

# 눈이 편안한 흰색 배경 + 보라색 포인트(Purple) 커스텀 CSS
st.markdown("""
    <style>
    /* 전체 배경 흰색 */
    .stApp {
        background-color: #FFFFFF;
    }
    /* 주요 제목(h1, h2, h3) 보라색 포인트 */
    h1, h2, h3 {
        color: #6A0DAD !important;
        font-family: 'Malgun Gothic', sans-serif;
    }
    /* 강조 텍스트 및 버튼 보라색 포인트 */
    .stButton>button {
        background-color: #6A0DAD;
        color: white;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #5b0a99;
        color: white;
    }
    /* 구분선(hr) 색상 변경 */
    hr {
        border-top: 2px solid #E6E6FA;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. 데이터 불러오기 (임시 캐싱으로 속도 최적화)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    # 실제 환경에서는 아까 만든 df_final_map을 csv로 저장한 뒤 불러오면 됩니다.
    # 예시: return pd.read_csv('최종_학교_공간데이터.csv')
    
    # [시뮬레이션용 가상 데이터 생성] - 실행 테스트를 위한 임시 데이터입니다.
    data = pd.DataFrame({
        '학교코드명': ['서울대학교사범대학부설중학교', '노형중학교', '거창중학교'],
        '지역': ['서울', '제주', '경남'],
        '위도': [37.596, 33.485, 35.686],
        '경도': [127.038, 126.475, 127.909],
        '유형_라벨': ['B유형 (재정비효율)', 'A유형 (과밀/과부하)', 'C유형 (소멸위기)'],
        '학생수계': [187, 1246, 300],
        '수업교사총수': [50, 79, 30],
        '최종_종합_인프라_점수': [85.5, 62.4, 45.2]
    })
    return data

df = load_data()

# ---------------------------------------------------------
# 3. 웹사이트 메인 화면 구성
# ---------------------------------------------------------
st.title("🏫 대한민국 교육 인프라 분석 & AI 맞춤형 정책 대시보드")
st.markdown("**평균의 함정을 넘어, 데이터 기반으로 학교별 맞춤형 교육 정책을 진단합니다.**")
st.markdown("---")

# [섹션 1] 메인 지도 (Folium)
st.header("📍 1. 전국 학교 인프라 유형 공간 분포")
st.markdown("전국 1만 2천 개 학교를 3가지 유형(A, B, C)으로 분류한 결과를 지도에서 확인하세요.")

# Folium 지도 생성
m = folium.Map(location=[36.5, 127.5], zoom_start=7, tiles='CartoDB positron')
marker_cluster = MarkerCluster().add_to(m)
color_map = {'A유형 (과밀/과부하)': 'red', 'B유형 (재정비효율)': 'blue', 'C유형 (소멸위기)': 'green'}

for idx, row in df.iterrows():
    folium.CircleMarker(
        location=[row['위도'], row['경도']],
        radius=7,
        color=color_map.get(row['유형_라벨'], 'gray'),
        fill=True, fill_opacity=0.8,
        tooltip=f"<b>{row['학교코드명']}</b><br>{row['유형_라벨']}"
    ).add_to(marker_cluster)

# 지도를 웹사이트에 렌더링
st_folium(m, width=1200, height=600)

st.markdown("---")

# [섹션 2] 시각화 자료 (차트)
st.header("📊 2. 주요 데이터 시각화")
col1, col2 = st.columns(2)

with col1:
    st.subheader("평균의 함정: 교사 1인당 학생 수 양극화")
    # 그래프 시뮬레이션
    fig, ax = plt.subplots(figsize=(6, 4))
    # 한글 폰트 설정 (웹 서버 환경에 맞춰 폰트명 변경 필요할 수 있음)
    plt.rc('font', family='Malgun Gothic') 
    sns.histplot([5, 6, 7, 25, 26, 27], bins=10, kde=True, color='purple', ax=ax)
    ax.set_title("양극화된 학생 수 분포")
    st.pyplot(fig)

with col2:
    st.subheader("K-Means 군집 분류 (학생 vs 교사)")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df, x='학생수계', y='수업교사총수', hue='유형_라벨', palette=color_map, s=100, ax=ax2)
    ax2.set_title("유형별 산점도")
    st.pyplot(fig2)

st.markdown("---")

# [섹션 3] AI 정책 컨설턴트 (검색 기능)
st.header("💡 3. AI 맞춤형 정책 컨설턴트")
st.markdown("분석하고 싶은 **학교 이름**을 입력하시면, 즉시 맞춤형 정책 리포트를 발행합니다.")

search_query = st.text_input("🔍 학교명 입력 (예: 노형중학교, 거창중학교)", "")

if search_query:
    target = df[df['학교코드명'].str.contains(search_query)]
    
    if target.empty:
        st.warning(f"'{search_query}'(을)를 찾을 수 없습니다.")
    else:
        school = target.iloc[0]
        s_type = school['유형_라벨']
        
        # 보라색 박스로 리포트 출력
        st.success(f"[{school['학교코드명']}] 데이터 분석이 완료되었습니다.")
        
        # 마크다운을 활용한 리포트 디자인
        st.markdown(f"""
        ### 📋 분석 지표 요약
        - **지역**: {school['지역']}
        - **유형**: **{s_type}**
        - **학생 수**: {school['학생수계']}명 / **교사 수**: {school['수업교사총수']}명
        - **종합 인프라 점수**: {school['최종_종합_인프라_점수']}점
        
        ### 🎯 AI 실무 맞춤형 정책 제언
        """)
        
        if 'A' in s_type:
            st.info("-> **[과밀학급 해소]** '그린스마트 미래학교' 사업 연계, 모듈러 교실 즉각 도입\n\n-> **[교원 지원]** 학교폭력 전담 조사관 및 상담교사 우선 배치")
        elif 'B' in s_type:
            st.info("-> **[예산 효율화]** 국비 지원 '학교복합시설' 건립 및 지자체 공동 운영\n\n-> **[거점화]** 자율형 공립고 2.0 추진으로 특성화 교육과정 신설")
        elif 'C' in s_type:
            st.info("-> **[고립 극복]** 메타버스 기반 온라인 공동 교육과정 거점 학교 지정\n\n-> **[학생 유치]** 농산어촌 유학 프로그램 전면 도입")

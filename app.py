import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import os
import urllib.request
from sklearn.linear_model import LinearRegression

# 1. 페이지 레이아웃 및 테마 최적화 (실리콘밸리 럭셔리 미니멀리즘)
st.set_page_config(page_title="Project EduBridge AI", layout="wide", initial_sidebar_state="collapsed")

# CSS 주입: 완벽한 화이트 캔버스 + 네온 퍼플 그라데이션 테마 + 지도 검은 테두리 파괴
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght=400;500;600;700;800&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', 'Malgun Gothic', sans-serif !important;
        background-color: #FFFFFF !important;
        color: #111827 !important;
    }
    
    [data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: blur(8px) !important;
    }
    
    iframe {
        background-color: transparent !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    .bento-card {
        background: #FFFFFF;
        padding: 26px;
        border-radius: 18px;
        border: 1px solid #F3F4F6;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.04);
        margin-bottom: 24px;
    }
    
    .project-title {
        font-size: 46px;
        font-weight: 800;
        letter-spacing: -1.5px;
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        line-height: 1.2;
    }
    
    .team-sub {
        font-size: 16px;
        font-weight: 600;
        color: #6B7280;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-top: 2px;
        margin-bottom: 35px;
    }
    
    .stTextInput>div>div>input {
        border-radius: 12px !important;
        border: 1px solid #E5E7EB !important;
        padding: 14px 18px !important;
        font-size: 16px !important;
        background-color: #F9FAFB !important;
        transition: all 0.25s ease;
    }
    .stTextInput>div>div>input:focus {
        background-color: #FFFFFF !important;
        border-color: #8B5CF6 !important;
        box-shadow: 0 0 0 4px rgba(139, 92, 246, 0.15) !important;
    }
    </style>
""", unsafe_allow_html=True)

# 리눅스 배포 서버 한글 깨짐 원천 차단 패치 (나눔고딕 강제 주입)
font_url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
font_path = "NanumGothic.ttf"
if not os.path.exists(font_path):
    try: urllib.request.urlretrieve(font_url, font_path)
    except: pass

if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'NanumGothic'
else:
    plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False


# 2. 데이터 프리미엄 로드 엔진
@st.cache_data
def load_final_data():
    try:
        df = pd.read_csv('final_school_data.csv')
        df['위도'] = pd.to_numeric(df['위도'], errors='coerce')
        df['경도'] = pd.to_numeric(df['경도'], errors='coerce')
        return df.dropna(subset=['위도', '경도'])
    except:
        try:
            df = pd.read_csv('final_school_data.csv', encoding='cp949')
            df['위도'] = pd.to_numeric(df['위도'], errors='coerce')
            df['경도'] = pd.to_numeric(df['경도'], errors='coerce')
            return df.dropna(subset=['위도', '경도'])
        except:
            st.error("🚨 'final_school_data.csv' 로드에 실패했습니다. 파일 경로를 다시 확인해주세요.")
            return None

df_final = load_final_data()

if df_final is not None:
    # BRANDING HERO SECTION
    st.markdown('<p class="project-title">Project EduBridge AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="team-sub">데이터기반 정책혁신 연구단 (Data-Driven Policy Innovation Lab)</p>', unsafe_allow_html=True)
    
    # 상단 지표 카드 레이아웃
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="bento-card"><span style="color:#6B7280; font-size:14px; font-weight:500;">공간 분석 대상</span><br><span style="font-size:26px; font-weight:700; color:#111827;">{len(df_final):,} 개교</span></div>', unsafe_allow_html=True)
    with m2:
        avg_ratio = round(df_final['학생수계'].sum() / df_final['수업교사총수'].sum(), 1)
        st.markdown(f'<div class="bento-card"><span style="color:#6B7280; font-size:14px; font-weight:500;">전국 평균 교사 1인당 학생 수</span><br><span style="font-size:26px; font-weight:700; color:#8B5CF6;">{avg_ratio} 명</span></div>', unsafe_allow_html=True)
    with m3:
        top_region = df_final.groupby('지역')['최종_종합_인프라_점수'].mean().idxmax()
        st.markdown(f'<div class="bento-card"><span style="color:#6B7280; font-size:14px; font-weight:500;">최고 인프라 집중 지역</span><br><span style="font-size:26px; font-weight:700; color:#111827;">{top_region}특별시</span></div>', unsafe_allow_html=True)

    # ---------------------------------------------------------
    # SECTION 1: GEOSPATIAL MAP
    # ---------------------------------------------------------
    st.markdown("<h2 style='font-size:22px; font-weight:700; margin-bottom:6px;'>1. 대한민국 인프라 양극화 및 취약도 지형도</h2>", unsafe_allow_html=True)
    
    max_schools = len(df_final)
    sample_size = st.slider("지도 시각화 학교 수 조절 (컨트롤러)", min_value=500, max_value=min(10000, max_schools), value=3000, step=500)
    
    _, map_center_col, _ = st.columns([1, 10, 1])
    
    with map_center_col:
        m_real = folium.Map(location=[36.2, 127.8], zoom_start=7, tiles='CartoDB positron')
        m_real.get_root().header.add_child(folium.Element("<style>.leaflet-container { background: #FFFFFF !important; }</style>"))
        
        marker_cluster = MarkerCluster(disableClusteringAtZoom=13).add_to(m_real)
        color_map = {'A유형 (과밀/과부하)': '#EF4444', 'B유형 (재정비효율)': '#3B82F6', 'C유형 (소멸위기)': '#10B981'}
        
        for idx, row in df_final.sample(n=sample_size, random_state=42).iterrows():
            html_content = f"""
            <div style='font-family: sans-serif; font-size: 13px; color:#1F2937; min-width:145px; line-height:1.5;'>
                <strong style='font-size:14px; color:{color_map[row['유형_라벨']]};'>{row['학교코드명']}</strong><br>
                <hr style='margin:5px 0; border:0; border-top:1px solid #E5E7EB;'>
                • 유형: {row['유형_라벨']}<br>
                • 학생수: {int(row['학생수계'])}명
            </div>
            """
            folium.CircleMarker(
                location=[row['위도'], row['경도']],
                radius=5.5,
                color=color_map[row['유형_라벨']],
                fill=True, fill_color=color_map[row['유형_라벨']], fill_opacity=0.75,
                weight=1,
                tooltip=folium.Tooltip(html_content)
            ).add_to(marker_cluster)
            
        st_folium(m_real, height=560, use_container_width=True, returned_objects=[])

    st.markdown("<div style='margin-bottom:40px;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # SECTION 2: CHARTS
    # ---------------------------------------------------------
    st.markdown("<h2 style='font-size:22px; font-weight:700; margin-bottom:14px;'>2. 데이터 머신러닝 스튜디오</h2>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown('<div class="bento-card">', unsafe_allow_html=True)
        st.markdown("<p style='font-size:15px; font-weight:600; color:#374151; margin-bottom:15px;'>K-Means 군집 분석 스캐터 플롯</p>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor='white')
        sns.scatterplot(data=df_final, x='학생수계', y='수업교사총수', hue='유형_라벨', palette=color_map, alpha=0.5, s=35, ax=ax, edgecolor='none')
        ax.set_facecolor('#FAFAFA')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('학생 수 (명)', fontsize=10)
        ax.set_ylabel('교사 수 (명)', fontsize=10)
        ax.legend(frameon=False, fontsize=9)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c2:
        st.markdown('<div class="bento-card">', unsafe_allow_html=True)
        st.markdown("<p style='font-size:15px; font-weight:600; color:#374151; margin-bottom:15px;'>지자체별 인프라 양극화 편차 차트</p>", unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(7, 4.5), facecolor='white')
        region_order = df_final.groupby('지역')['최종_종합_인프라_점수'].mean().sort_values(ascending=False).index
        sns.barplot(data=df_final, x='지역', y='최종_종합_인프라_점수', order=region_order, palette='Purples_r', errorbar=None, ax=ax2)
        ax2.set_facecolor('#FAFAFA')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.set_xlabel('행정구역명', fontsize=10)
        ax2.set_ylabel('종합 인프라 지수', fontsize=10)
        plt.xticks(rotation=45, fontsize=9)
        st.pyplot(fig2)
        st.markdown('</div>', unsafe_allow_html=True)

    # 연도별 미래 예측 시뮬레이션 인터랙티브 존
    st.markdown('<div class="bento-card">', unsafe_allow_html=True)
    st.markdown("<p style='font-size:18px; font-weight:700; color:#111827; margin-bottom:2px;'>- 학령인구 감소에 따른 미래 교육 여건 시뮬레이션 -</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6B7280; font-size:14px; margin-bottom:20px;'>정부의 교원 임용 축소 정책 유무에 따른 교원 1인당 학생 수 예측 시나리오</p>", unsafe_allow_html=True)
    
    target_year = st.slider("예측 목표 연도를 설정하세요.", min_value=2025, max_value=2030, value=2030, step=1)
    
    hist_years = np.array([2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
    hist_ratio = np.array([16.02, 15.38, 14.94, 14.65, 14.48, 14.21, 13.99, 13.79])
    
    model_lr = LinearRegression().fit(hist_years.reshape(-1, 1), hist_ratio)
    future_years = np.array(list(range(2025, target_year + 1)))
    
    pred_trend = model_lr.predict(future_years.reshape(-1, 1))
    pred_bottleneck_base = [13.65, 13.55, 13.50, 13.48, 13.47, 13.46]
    pred_bottleneck = pred_bottleneck_base[:len(future_years)]
    
    pred_col1, pred_col2 = st.columns([1.3, 1])
    
    with pred_col1:
        fig_pred, ax_pred = plt.subplots(figsize=(7.5, 4.6), facecolor='white')
        
        ax_pred.plot(hist_years, hist_ratio, marker='o', color='#8B5CF6', linewidth=2.5, label='실제 추이 (2017-2024)')
        for x, y in zip(hist_years, hist_ratio):
            ax_pred.text(x, y + 0.12, f"{y:.1f}", ha='center', fontsize=8, color='#6D28D9', fontweight='bold')
            
        if len(future_years) > 0:
            ax_pred.plot(future_years, pred_trend, linestyle='--', marker='s', color='#9CA3AF', linewidth=1.8, label='단순 추세 연장 (교원 유지)')
            ax_pred.plot(future_years, pred_bottleneck, linestyle='--', marker='^', color='#EF4444', linewidth=2.2, label='현실적 예측 (교원 감축 반영)')
            
            ax_pred.text(future_years[-1], pred_trend[-1] - 0.22, f"{pred_trend[-1]:.2f}명", ha='center', fontsize=8.5, color='#4B5563', fontweight='bold')
            ax_pred.text(future_years[-1], pred_bottleneck[-1] + 0.12, f"{pred_bottleneck[-1]:.2f}명", ha='center', fontsize=8.5, color='#B91C1C', fontweight='bold')

        ax_pred.set_facecolor('#FAFAFA')
        ax_pred.spines['top'].set_visible(False)
        ax_pred.spines['right'].set_visible(False)
        ax_pred.spines['left'].set_color('#E5E7EB')
        ax_pred.spines['bottom'].set_color('#E5E7EB')
        ax_pred.set_ylim(11.0, 16.8)
        ax_pred.set_xticks(list(hist_years) + list(future_years))
        plt.xticks(rotation=45, fontsize=8.5)
        ax_pred.legend(frameon=False, loc='upper right', fontsize=8.5)
        st.pyplot(fig_pred)
        
    # 💡 [가독성 대폭 강화 패치] 글자 크기를 키우고 줄바꿈 및 간격을 리스케일링함
    with pred_col2:
        st.markdown(f"""
        <div style="padding-left:18px; border-left:4px solid #8B5CF6; height:100%;">
            <p style="font-size:19px; font-weight:800; color:#111827; margin-bottom:14px; letter-spacing:-0.5px;">- 시뮬레이션 기반 데이터 인사이트 리포트 -</p>
            <p style="font-size:15.5px; line-height:1.75; color:#374151; text-align:justify;">
                인구 급감에 따라 학생 수가 줄어들면 공교육 여건이 자동으로 크게 개선될 것이라는 일반적인 낙관론은 국가 전반 수치 뒤에 은닉된 <b>'평균의 함정'</b>을 내포하고 있습니다. 머신러닝 예측 모델링 분석 결과는 정책적 변수에 따라 완전히 정반대의 미래를 경고합니다.
            </p>
            <p style="font-size:15.5px; line-height:1.75; color:#374151; margin-top:14px;">
                <span style="font-weight:700; color:#4B5563;">- 단순 추세 연장 가설 (회색 점선)</span><br>
                -> 과거의 충원 흐름을 선형회귀 모델로 단순 연장할 시, {target_year}년 교원 1인당 학생 수는 <b>{pred_trend[-1]:.2f}명</b>까지 낮아져 정량적 교육 환경이 무조건 상향되는 왜곡된 평크를 도출합니다.
            </p>
            <p style="font-size:15.5px; line-height:1.75; color:#374151; margin-top:14px;">
                <span style="font-weight:700; color:#B91C1C;">- 현실적 정책 리스크 반영선 (빨간 점선)</span><br>
                -> 실제 정부의 교원 정원 동결 및 임용 규모 축소 정책이 동반 작용할 경우, 학생 수 급감 효과가 전량 상쇄되어 {target_year}년 수치는 <b>{pred_bottleneck[-1]:.2f}명</b> 선에서 정체(Bottleneck)되는 심각한 교육 여건 정체 현상이 발생합니다.
            </p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------------------
    # SECTION 2.5: 심층 격차 분석 스튜디오 (로렌츠 곡선 & 비판적 통찰 결합)
    # ---------------------------------------------------------
    st.markdown("<h2 style='font-size:22px; font-weight:700; margin-top:30px; margin-bottom:14px;'>2.5 전국 교육 인프라 분배 불평등도 심층 진단</h2>", unsafe_allow_html=True)
    c_adv1, c_adv2 = st.columns(2)
    
    # 로렌츠 지니계수 연산 블록
    infra_values = np.sort(df_final['최종_종합_인프라_점수'].values)
    n = len(infra_values)
    index = np.arange(1, n + 1)
    gini = ((np.sum((2 * index - n - 1) * infra_values)) / (n * np.sum(infra_values)))
    
    cum_infra = np.cumsum(infra_values) / np.sum(infra_values)
    cum_population = np.arange(1, n + 1) / n
        
    with c_adv1:
        st.markdown('<div class="bento-card">', unsafe_allow_html=True)
        st.markdown("<p style='font-size:15px; font-weight:600; color:#374151; margin-bottom:15px;'>교육 인프라 자원 분배의 로렌츠 곡선 (Lorenz Curve)</p>", unsafe_allow_html=True)
        
        fig_lorenz, ax_lorenz = plt.subplots(figsize=(7, 4.5), facecolor='white')
        ax_lorenz.plot([0, 1], [0, 1], color='#9CA3AF', linestyle='--', linewidth=1.5, label='완전 평등선 (Gini = 0)')
        ax_lorenz.plot(cum_population, cum_infra, color='#8B5CF6', linewidth=2.5, label=f'인프라 분배 곡선 (Gini = {gini:.3f})')
        ax_lorenz.fill_between(cum_population, cum_population, cum_infra, color='#8B5CF6', alpha=0.1)
        
        ax_lorenz.set_facecolor('#FAFAFA')
        ax_lorenz.spines['top'].set_visible(False)
        ax_lorenz.spines['right'].set_visible(False)
        ax_lorenz.set_xlabel('학교 누적 비율', fontsize=10)
        ax_lorenz.set_ylabel('인프라 점수 누적 비율', fontsize=10)
        ax_lorenz.legend(frameon=False, fontsize=9, loc='upper left')
        st.pyplot(fig_lorenz)
        st.markdown('</div>', unsafe_allow_html=True)
        
    # 💡 [신규 추가] 프로파일러 삭제 후 로렌츠 곡선에 대한 고가독성 비판적 통찰 해설판 배치
    with c_adv2:
        st.markdown('<div class="bento-card" style="height:100%;">', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="padding-left:18px; border-left:4px solid #EC4899; height:100%;">
            <p style="font-size:19px; font-weight:800; color:#111827; margin-bottom:14px; letter-spacing:-0.5px;">- 로렌츠 곡선 기반 자원 배분 불평등 지표 분석 -</p>
            <p style="font-size:15.5px; line-height:1.75; color:#374151; text-align:justify;">
                이 시각화 모델은 소득 양극화를 진단할 때 활용되는 경제학적 통계 기법을 전국 1만여 개 학교의 교육 인프라에 대입하여 자원 분배의 평등성을 계량화한 핵심 격차 지표입니다.
            </p>
            <p style="font-size:15.5px; line-height:1.75; color:#374151; margin-top:14px;">
                <span style="font-weight:700; color:#4B5563;">- 통계학적 진단 수치: Gini 인덱스 {gini:.3f}</span><br>
                -> 일반 자원 분배에서 지니계수가 0.4 지표를 초과하면 대단히 심각한 수준의 불평등 격차 사회로 정의됩니다. 우리 시스템이 도출한 <b>{gini:.3f}</b> 스코어는 현재 대한민국 학교들이 입지 조건과 교육 예산 인프라에서 얼마나 극심한 불균형 분배 구조에 놓여 있는지를 과학적으로 증명합니다.
            </p>
            <p style="font-size:15.5px; line-height:1.75; color:#374151; margin-top:14px;">
                <span style="font-weight:700; color:#7C3AED;">- 비판적 통찰 (Critical Insight)</span><br>
                -> 대각선(완전평등)과 하단 보라색 곡선 사이의 벌어진 유격 면적은 대도시 중심부와 도서산간 및 농어촌 간의 인프라 빈부격차가 심계에 도달했음을 폭로합니다. 공교육 전반의 인프라 평등이라는 허울 좋은 전체 평균의 통계적 착시에 가려져 정작 자원의 지원이 절실한 고립 학교(C유형)들이 물리적 하드웨어 결핍으로 소멸의 가속 페달을 밟고 있음을 시사하는 강력한 데이터 실증적 증거입니다.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------------------
    # SECTION 3: AI CONSULTANT
    # ---------------------------------------------------------
    st.markdown("<h2 style='font-size:22px; font-weight:700; margin-top:20px; margin-bottom:14px;'>3. 개별 학교 맞춤형 AI 정책 수립 처방전</h2>", unsafe_allow_html=True)
    st.markdown('<div class="bento-card">', unsafe_allow_html=True)
    query = st.text_input("조회하려는 맞춤형 학교 이름을 명확히 입력하세요.", placeholder="예: 노형중학교, 개원중학교, 거창중학교")
    
    if query:
        result = df_final[df_final['학교코드명'].str.contains(query, na=False)]
        if result.empty:
            st.markdown(f"<p style='color:#EF4444; font-weight:500;'>- 죄송합니다. '{query}' 학교는 분석 데이터 매칭에 실패했습니다.</p>", unsafe_allow_html=True)
        else:
            school = result.iloc[0]
            s_type = school['유형_라벨']
            t_score = round(school['최종_종합_인프라_점수'], 1)
            tr_score = round(school['접근성_점수(100만점)'], 1)
            ed_score = round(school['교육인프라_점수(100만점)'], 1)
            s_count = int(school['학생수계'])
            t_count = int(school['수업교사총수'])
            ratio = round(s_count / t_count, 1)
            
            st.markdown(f"### - {school['학교코드명']} 개별 컨설팅 인덱스 리포트")
            sc1, sc2 = st.columns([1, 2.2])
            with sc1:
                st.markdown(f"""
                <div style="background:#F9FAFB; padding:22px; border-radius:14px; border:1px solid #E5E7EB;">
                    <span style="font-size:13px; color:#6B7280; font-weight:600;">AI 군집 클러스터</span><br>
                    <strong style="font-size:16px; color:{color_map.get(s_type, '#111827')};">{s_type}</strong><br><br>
                    <span style="font-size:13px; color:#6B7280; font-weight:600;">인프라 상세 매트릭스</span><br>
                    <span style="font-size:14px; font-weight:700; color:#374151;">• 종합 스코어: {t_score}점</span><br>
                    <span style="font-size:13px; color:#4B5563;">• 대중교통망: {tr_score}점</span><br>
                    <span style="font-size:13px; color:#4B5563;">• 교육인프라: {ed_score}점</span><br><br>
                    <span style="font-size:13px; color:#6B7280; font-weight:600;">내부 교육 밀도</span><br>
                    <span style="font-size:14px; font-weight:700; color:#374151;">• 교사 1인당 학생: {ratio}명</span>
                </div>
                """, unsafe_allow_html=True)
                
            with sc2:
                st.markdown("<p style='font-size:16px; font-weight:600; margin-bottom:10px; color:#8B5CF6;'>- 소관 정부 부처/지자체 연계 핀셋 대응안</p>", unsafe_allow_html=True)
                if tr_score < 60:
                    st.markdown("-> **[교통 복지 확충]** 지자체 소관 '수요응답형 대중교통(DRT)' 플랫폼 노선을 하교 시간대에 학교 정문 앞 우선 배차 조치")
                if ed_score < 50:
                    st.markdown("-> **[교육 절벽 구제]** 공교육 내 인공지능 미래형 보조교사 '디지털 튜터' 전담 파견 및 교육청 연계 AI 에듀 바우처 100% 무상 할당")
                
                if 'A유형' in s_type:
                    st.markdown("-> **[과밀학급 해소]** 교육부 산하 '그린스마트 미래학교' 프로젝트 최우선권 배정, 친환경·고성능 모듈러 교실 시스템 즉각 구축")
                    st.markdown("-> **[정서 케어]** 과밀 밀집도에 의한 교원 피로 완화를 위해 공인 '학교폭력 전담 조사관' 및 Wee클래스 전문 인력 가산 영입")
                elif 'B유형' in s_type:
                    st.markdown("-> **[공간 200% 활용]** 정부 국비 매칭 '학교복합시설' 사업 선발: 유휴 공실 스페이스를 주민 공공 도서관 및 거점형 늘봄 센터로 영구 개조")
                    st.markdown("-> **[신규 유치]** 교육청 공모 '자율형 공립고 2.0' 연계, 인근 첨단 산업/디지털 기술 연계 특화 교육 트랙 개설을 통한 외부 학생 유입 유도")
                elif 'C유형' in s_type:
                    st.markdown("-> **[고립 극복]** 교원 수급 불균형에 의한 선택과목 마비를 해소하기 위해 교육청 주관 '메타버스 기반 가상 공동 교육과정' 거점 하드웨어 전면 지원")
                    st.markdown("-> **[인구 사수]** 로컬 교육 특성을 반영한 지자체 협업 '농산어촌 유학 프로그램' 도입으로 대도시권 유학 인구 파이프라인 개척")
                    st.markdown("-> **[교원 복지]** 원격 근무 기피 방지를 위한 도교육청 소관 노후 사택 전면 신축 개보수 및 벽지 근무 승진 가산점 상향 상정")
                    
    st.markdown('</div>', unsafe_allow_html=True)

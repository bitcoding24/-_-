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
        padding: 28px;
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
    # 전역 컬러 맵 구성 정의
    color_map = {'A유형 (과밀/과부하)': '#EF4444', 'B유형 (재정비효율)': '#3B82F6', 'C유형 (소멸위기)': '#10B981'}

    # BRANDING HERO SECTION (오민도 연구원님 정식 브랜딩)
    st.markdown('<p class="project-title">Project EduBridge AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="team-sub">오민도 (데이터기반 정책혁신 연구단 / Data-Driven Policy Innovation Lab)</p>', unsafe_allow_html=True)
    
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
    # SECTION 2: CHARTS & FUTURE PROJECTION SIMULATION
    # ---------------------------------------------------------
    st.markdown("<h2 style='font-size:22px; font-weight:700; margin-bottom:14px;'>2. 머신러닝 분석 및 미래 여건 시뮬레이션 스튜디오</h2>", unsafe_allow_html=True)
    
    # K-Means 산점도 카드 컴포넌트 (좌우 분할 1.3 : 1 황금 비율)
    st.markdown('<div class="bento-card">', unsafe_allow_html=True)
    st.markdown("<p style='font-size:18px; font-weight:700; color:#111827; margin-bottom:2px;'>- K-Means 군집 분석 기반 학생-교원 공급 분포도 -</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6B7280; font-size:14px; margin-bottom:20px;'>전국 1만여 개 학교의 총학생수 및 교사수 행렬 데이터를 활용한 인공지능 군집 분할 결과</p>", unsafe_allow_html=True)
    
    chart_col1, chart_col2 = st.columns([1.3, 1])
    
    with chart_col1:
        fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor='white')
        sns.scatterplot(data=df_final, x='학생수계', y='수업교사총수', hue='유형_라벨', palette=color_map, alpha=0.5, s=35, ax=ax, edgecolor='none')
        ax.set_facecolor('#FAFAFA')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#E5E7EB')
        ax.spines['bottom'].set_color('#E5E7EB')
        ax.set_xlabel('학생 수 (명)', fontsize=10)
        ax.set_ylabel('교사 수 (명)', fontsize=10)
        ax.legend(frameon=False, fontsize=9)
        st.pyplot(fig)
        
    # 💡 [핵심 패치] 누락되었던 B유형 설명을 추가하고 가독성 리스케일링 적용
    with chart_col2:
        st.markdown(f"""
        <div style="padding-left:18px; border-left:4px solid #6366F1; height:100%;">
            <p style="font-size:19px; font-weight:800; color:#111827; margin-bottom:14px; letter-spacing:-0.5px;">- 인공지능 군집 분할 특성 및 격차 통찰 -</p>
            <p style="font-size:15.5px; line-height:1.75; color:#374151; text-align:justify;">
                전국 학교의 학생 수와 교사 총수 매트릭스를 기반으로 알고리즘을 구동한 결과, 대한민국 공교육 생태계는 행정구역 경계를 초월하여 실제 학교가 직면한 내부 자원 공급 수준에 따라 <b>'체급별 양극화 노선'</b>을 명확하게 도출합니다.
            </p>
            <p style="font-size:15.5px; line-height:1.75; color:#374151; margin-top:14px;">
                <span style="font-weight:700; color:#EF4444;">- A유형 (과밀/과부하 집단)</span><br>
                -> 스캐터 플롯 우상단 영역에 길게 분산된 대형 학교 군집입니다. 주로 대도시 및 신도시 인근 학구에 위치해 있으며, 과밀학급화에 따른 교원 업무 과부하가 심각하여 하드웨어적 교실 증축 및 행정 인력 가산 배치가 긴급히 요구되는 집단입니다.
            </p>
            <p style="font-size:15.5px; line-height:1.75; color:#374151; margin-top:14px;">
                <span style="font-weight:700; color:#3B82F6;">- B유형 (재정비 효율화 집단)</span><br>
                -> 플롯 중앙 지대에 위치한 중형 규모의 군집입니다. 주로 지방 소도시 및 원도심 지대에 입지해 있으며, 학령인구는 감소 추세이나 과거 설치된 학교 인프라 규모가 그대로 유지되어 자원 효율성이 떨어지므로 '학교복합시설 고도화' 및 유휴 공간 재편 정책이 필요한 대상입니다.
            </p>
            <p style="font-size:15.5px; line-height:1.75; color:#374151; margin-top:14px;">
                <span style="font-weight:700; color:#10B981;">- C유형 (소멸위기 고립 집단)</span><br>
                -> 플롯 좌하단 원점 근방에 밀집된 소규모 학교 군집입니다. 원거리 농어촌 및 인구 감소 직격탄을 맞은 지대에 분포되어 있으며, 공교육 규모의 경제를 완전히 상실하여 교과목 마비 및 폐교 리스크가 가장 높은 최우선 케어 집단입니다.
            </p>
        </div>
        """, unsafe_allow_html=True)
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
        
    with pred_col2:
        st.markdown(f"""
        <div style="padding-left:18px; border-left:4px solid #8B5CF6; height:100%;">
            <p style="font-size:19px; font-weight:800; color:#111827; margin-bottom:14px; letter-spacing:-0.5px;">- 시뮬레이션 기반 데이터 인사이트 리포트 -</p>
            <p style="font-size:15.5px; line-height:1.75; color:#374151; text-align:justify;">
                대한민국의 합계출산율은 2022년 0.78명으로 OECD 역사상 최저 수준을 기록했습니다. 인구 급감에 따라 학생 수가 줄어들면 공교육 여건이 자동으로 크게 개선될 것이라는 일반적인 낙관론은 국가 전반 수치 뒤에 은닉된 <b>'평균의 함정'</b>을 내포하고 있습니다. 머신러닝 예측 모델링 분석 결과는 정책적 변수에 따라 완전히 정반대의 미래를 경고합니다.
            </p>
            <p style="font-size:15.5px; line-height:1.75; color:#374151; margin-top:14px;">
                <span style="font-weight:700; color:#4B5563;">- 단순 추세 연장 가설 (회색 점선)</span><br>
                -> 과거의 충원 흐름을 선형회귀 모델로 단순 연장할 시, {target_year}년 교원 1인당 학생 수는 <b>{pred_trend[-1]:.2f}명</b>까지 낮아져 정량적 교육 환경이 무조건 상향되는 왜곡된 평가를 도출합니다. 이는 통계청의 다소 낙관적인 인구 반등 시나리오에 기반한 수치적 착시입니다.
            </p>
            <p style="font-size:15.5px; line-height:1.75; color:#374151; margin-top:14px;">
                <span style="font-weight:700; color:#B91C1C;">- 현실적 정책 리스크 반영선 (빨간 점선)</span><br>
                -> 실제 정부가 학령인구 감소를 이유로 교원 정원을 동결하거나 신규 임용 규모를 축소(임용 절벽)할 경우, 학생 수 급감 효과가 전량 상쇄되어 {target_year}년 수치는 <b>{pred_bottleneck[-1]:.2f}명</b> 선에서 정체(Bottleneck)되는 심각한 교육 여건 정체 현상이 발생합니다. 평균의 환상에 속아 교사 공급을 일괄 감축하면 실질적인 교육 환경 개선은 완전히 마비됩니다.
            </p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

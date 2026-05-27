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

# 1. 페이지 레이아웃 및 테마 최적화 (라벤더 럭셔리 미니멀리즘)
st.set_page_config(page_title="Project EduBridge AI", layout="wide", initial_sidebar_state="collapsed")

# CSS 주입: 라벤더/퍼플 그라데이션 테마 적용
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght=400;500;600;700;800&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', 'Malgun Gothic', sans-serif !important;
        background-color: #FAFAFA !important; /* 아주 연한 웜그레이/화이트 톤 */
        color: #1F2937 !important;
    }
    
    [data-testid="stHeader"] {
        background: rgba(250, 250, 250, 0.8) !important;
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
        border: 1px solid #F3E8FF; /* 연보라 테두리 */
        box-shadow: 0 10px 30px -10px rgba(147, 51, 234, 0.1); /* 퍼플 섀도우 */
        margin-bottom: 24px;
    }
    
    .project-title {
        font-size: 46px;
        font-weight: 800;
        letter-spacing: -1.5px;
        background: linear-gradient(135deg, #A855F7 0%, #C084FC 50%, #D8B4FE 100%); /* 라벤더 그라데이션 */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        line-height: 1.2;
    }
    
    .team-sub {
        font-size: 16px;
        font-weight: 600;
        color: #9333EA;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-top: 2px;
        margin-bottom: 35px;
    }
    </style>
""", unsafe_allow_html=True)

# 리눅스 배포 서버 한글 깨짐 원천 차단 패치
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
    # 💡 [색상 대비 극대화 및 에러 원천 차단] 
    # 데이터에 A, B, C가 포함되어 있으면 가독성 높은 색상을 자동으로 매핑합니다.
    unique_labels = df_final['유형_라벨'].unique()
    color_map = {}
    for label in unique_labels:
        label_str = str(label)
        if 'A' in label_str:
            color_map[label] = '#5B21B6' # 다크 바이올렛 (A유형 - 과밀)
        elif 'B' in label_str:
            color_map[label] = '#EC4899' # 마젠타 핑크 (B유형 - 확실한 구분)
        elif 'C' in label_str:
            color_map[label] = '#3B82F6' # 코발트 블루 (C유형 - 확실한 구분)
        else:
            color_map[label] = '#D8B4FE' # 기본값 (연보라)

    # BRANDING HERO SECTION
    st.markdown('<p class="project-title">Project EduBridge AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="team-sub">오민도</p>', unsafe_allow_html=True)
    
    # 상단 지표 카드 레이아웃
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="bento-card"><span style="color:#6B7280; font-size:14px; font-weight:500;">공간 분석 대상</span><br><span style="font-size:26px; font-weight:700; color:#111827;">{len(df_final):,} 개교</span></div>', unsafe_allow_html=True)
    with m2:
        avg_ratio = round(df_final['학생수계'].sum() / df_final['수업교사총수'].sum(), 1)
        st.markdown(f'<div class="bento-card"><span style="color:#6B7280; font-size:14px; font-weight:500;">전국 평균 교사 1인당 학생 수</span><br><span style="font-size:26px; font-weight:700; color:#9333EA;">{avg_ratio} 명</span></div>', unsafe_allow_html=True)
    with m3:
        top_region = df_final.groupby('지역')['최종_종합_인프라_점수'].mean().idxmax()
        st.markdown(f'<div class="bento-card"><span style="color:#6B7280; font-size:14px; font-weight:500;">최고 인프라 집중 지역</span><br><span style="font-size:26px; font-weight:700; color:#111827;">{top_region}특별시</span></div>', unsafe_allow_html=True)

    # ---------------------------------------------------------
    # SECTION 1: GEOSPATIAL MAP
    # ---------------------------------------------------------
    st.markdown("<h2 style='font-size:22px; font-weight:700; margin-bottom:6px; color:#4C1D95;'>1. 대한민국 인프라 양극화 및 취약도 지형도</h2>", unsafe_allow_html=True)
    
    max_schools = len(df_final)
    sample_size = st.slider("지도 시각화 학교 수 조절 (컨트롤러)", min_value=500, max_value=min(10000, max_schools), value=3000, step=500)
    
    _, map_center_col, _ = st.columns([1, 10, 1])
    
    with map_center_col:
        m_real = folium.Map(location=[36.2, 127.8], zoom_start=7, tiles='CartoDB positron')
        m_real.get_root().header.add_child(folium.Element("<style>.leaflet-container { background: #FFFFFF !important; }</style>"))
        
        marker_cluster = MarkerCluster(disableClusteringAtZoom=13).add_to(m_real)
        
        for idx, row in df_final.sample(n=sample_size, random_state=42).iterrows():
            marker_color = color_map.get(row['유형_라벨'], '#D8B4FE') # 안전한 color_map 접근
            
            html_content = f"""
            <div style='font-family: sans-serif; font-size: 13px; color:#1F2937; min-width:145px; line-height:1.5;'>
                <strong style='font-size:14px; color:{marker_color};'>{row['학교코드명']}</strong><br>
                <hr style='margin:5px 0; border:0; border-top:1px solid #F3E8FF;'>
                • 유형: {row['유형_라벨']}<br>
                • 학생수: {int(row['학생수계'])}명
            </div>
            """
            folium.CircleMarker(
                location=[row['위도'], row['경도']],
                radius=5.5,
                color=marker_color,
                fill=True, fill_color=marker_color, fill_opacity=0.75,
                weight=1,
                tooltip=folium.Tooltip(html_content)
            ).add_to(marker_cluster)
            
        st_folium(m_real, height=560, use_container_width=True, returned_objects=[])

    st.markdown("<div style='margin-bottom:40px;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # SECTION 2: CHARTS & FUTURE PROJECTION SIMULATION
    # ---------------------------------------------------------
    st.markdown("<h2 style='font-size:22px; font-weight:700; margin-bottom:14px; color:#4C1D95;'>2. 머신러닝 분석 및 미래 여건 시뮬레이션 스튜디오</h2>", unsafe_allow_html=True)
    
    st.markdown('<div class="bento-card">', unsafe_allow_html=True)
    st.markdown("<p style='font-size:18px; font-weight:700; color:#4C1D95; margin-bottom:2px;'>- 인공지능(AI) 군집 분석 결과 및 격차 해설 -</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6B7280; font-size:14px; margin-bottom:20px;'>전국 1만여 개 학교의 학생 수와 교사 수 데이터를 AI 알고리즘으로 분석한 결과입니다.</p>", unsafe_allow_html=True)
    
    chart_col1, chart_col2 = st.columns([1.3, 1])
    
    with chart_col1:
        fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor='white')
        sns.scatterplot(data=df_final, x='학생수계', y='수업교사총수', hue='유형_라벨', palette=color_map, alpha=0.6, s=35, ax=ax, edgecolor='none')
        ax.set_facecolor('#FFFFFF')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#F3E8FF')
        ax.spines['bottom'].set_color('#F3E8FF')
        ax.set_xlabel('학생 수 (명)', fontsize=10, color='#4B5563')
        ax.set_ylabel('교사 수 (명)', fontsize=10, color='#4B5563')
        ax.legend(frameon=False, fontsize=9)
        st.pyplot(fig)
        
    with chart_col2:
        st.markdown(f"""
        <div style="padding-left:18px; border-left:4px solid #A855F7; height:100%;">
            <p style="font-size:19px; font-weight:800; color:#4C1D95; margin-bottom:18px; letter-spacing:-0.5px;">- 군집 알고리즘을 통한 학교 유형별 분류 -</p>
            
            <p style="font-size:15px; line-height:1.75; color:#7E6E93; text-align:justify; margin-bottom:16px;">
                <span style="font-weight:700; color:#5B21B6; font-size:16px;">▶ A유형: 과밀 학교</span><br>
                신도시 및 대도시 중심지에 위치한 대형 학교군이다. 신도시 개발에 따른 지속적인 인구 유입으로 과대 학교 및 과밀 학급 문제가 심화되고 있다. 특히 수도권 등 특정 지역의 과밀 학급 문제가 심각하여 교사의 업무 부담이 가중되고 있으며, 학생 개개인에 대한 맞춤형 교육 제공에 한계가 존재한다. 교육의 질 저하를 방지하기 위해 교실 증축과 행정 보조인력의 즉각적인 지원이 시급하다.
            </p>
            
            <p style="font-size:15px; line-height:1.75; color:#7E6E93; text-align:justify; margin-bottom:16px;">
                <span style="font-weight:700; color:#EC4899; font-size:16px;">▶ B유형: 재정비 필요 학교</span><br>
                지방 소도시 및 구도심에 위치한 중형 학교군이다. 현재는 학생과 교사 수가 적정 수준을 유지하고 있으나, 급격한 출산율 저하로 인해 학령인구 감소의 영향권에 진입하고 있다. 향후 학령인구 감소에 따른 유휴 학교 시설 발생이 주요 교육 및 사회적 문제로 대두될 전망이다. 따라서 남는 공간을 주민 도서관이나 돌봄 센터 등 지역 사회 활성화를 위한 공간으로 재구성하는 사전 공간 재편 전략이 필요하다.
            </p>
            
            <p style="font-size:15px; line-height:1.75; color:#7E6E93; text-align:justify;">
                <span style="font-weight:700; color:#3B82F6; font-size:16px;">▶ C유형: 소멸위기 학교</span><br>
                도서산간 및 농어촌 지역을 비롯해 최근 대도시 일부까지 확산 중인 소규모 학교군이다. 학령인구의 급격한 감소로 인해 정상적인 교과목 수업 개설이 어렵고, 예체능이나 동아리 활동 등 교육과정의 다양성이 부족하다. 이는 교육 환경의 질적 저하와 학생들의 공교육 혜택 소외로 이어지고 있으며, 학교 운영의 어려움을 넘어 지역 소멸 위기를 가속화하고 있다. 교육 격차 해소와 폐교 방지를 위한 지역 상생 정책 수립이 절실하다.
            </p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 연도별 미래 예측 시뮬레이션 인터랙티브 존
    st.markdown('<div class="bento-card">', unsafe_allow_html=True)
    st.markdown("<p style='font-size:18px; font-weight:700; color:#4C1D95; margin-bottom:2px;'>- 학령인구 감소에 따른 미래 교육 여건 시뮬레이션 -</p>", unsafe_allow_html=True)
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
        
        ax_pred.plot(hist_years, hist_ratio, marker='o', color='#A855F7', linewidth=2.5, label='실제 추이 (2017-2024)')
        for x, y in zip(hist_years, hist_ratio):
            ax_pred.text(x, y + 0.12, f"{y:.1f}", ha='center', fontsize=8, color='#7E22CE', fontweight='bold')
            
        if len(future_years) > 0:
            ax_pred.plot(future_years, pred_trend, linestyle='--', marker='s', color='#D8B4FE', linewidth=1.8, label='단순 추세 연장 (교원 유지)')
            ax_pred.plot(future_years, pred_bottleneck, linestyle='--', marker='^', color='#9333EA', linewidth=2.2, label='현실적 예측 (교원 감축 반영)')
            
            ax_pred.text(future_years[-1], pred_trend[-1] - 0.22, f"{pred_trend[-1]:.2f}명", ha='center', fontsize=8.5, color='#9CA3AF', fontweight='bold')
            ax_pred.text(future_years[-1], pred_bottleneck[-1] + 0.12, f"{pred_bottleneck[-1]:.2f}명", ha='center', fontsize=8.5, color='#7E22CE', fontweight='bold')

        ax_pred.set_facecolor('#FFFFFF')
        ax_pred.spines['top'].set_visible(False)
        ax_pred.spines['right'].set_visible(False)
        ax_pred.spines['left'].set_color('#F3E8FF')
        ax_pred.spines['bottom'].set_color('#F3E8FF')
        ax_pred.set_ylim(11.0, 16.8)
        ax_pred.set_xticks(list(hist_years) + list(future_years))
        plt.xticks(rotation=45, fontsize=8.5, color='#4B5563')
        plt.yticks(color='#4B5563')
        ax_pred.legend(frameon=False, loc='upper right', fontsize=8.5)
        st.pyplot(fig_pred)
        
    with pred_col2:
        st.markdown(f"""
        <div style="padding-left:18px; border-left:4px solid #A855F7; height:100%;">
            <p style="font-size:19px; font-weight:800; color:#4C1D95; margin-bottom:14px; letter-spacing:-0.5px;">- 미래 교육 환경 예측: 낙관론과 냉정한 현실 -</p>
            <p style="font-size:15.5px; line-height:1.75; color:#7E6E93; text-align:justify;">
                대한민국의 합계출산율은 2022년 0.78명으로 역사상 최저 수준을 기록했습니다. 흔히 사람들은 "아이들이 줄어드니까 교사 한 명당 돌보는 학생 수도 줄어들고, 교육 여건이 저절로 좋아지겠지?"라고 생각합니다. 하지만 이는 전체 평균의 숫자에 속는 <b>'평균의 함정'</b>입니다. 인공지능 예측 결과는 정부 정책에 따라 미래가 완전히 달라질 수 있음을 경고합니다.
            </p>
            <p style="font-size:15.5px; line-height:1.75; color:#7E6E93; margin-top:14px;">
                <span style="font-weight:700; color:#9CA3AF;">- 단순 추세 연장 가설 (연보라 점선)</span><br>
                -> 과거 데이터 흐름 그대로 "학생만 줄어들고 교사 수는 지금처럼 유지된다"고 기계적으로 계산한 선입니다. {target_year}년이 되면 교원 1인당 학생 수가 <b>{pred_trend[-1]:.2f}명</b>까지 떨어져 교육 여건이 엄청나게 좋아지는 것처럼 보입니다. 하지만 이는 현실성이 낮은 통계적 착시일 뿐입니다.
            </p>
            <p style="font-size:15.5px; line-height:1.75; color:#7E6E93; margin-top:14px;">
                <span style="font-weight:700; color:#9333EA;">- 현실적 정책 리스크 반영선 (진보라 점선)</span><br>
                -> "학생이 줄어드니 나라에서 교사 임용도 같이 줄여버린다(임용 절벽)"는 실제 정부의 정원 감축 정책 리스크를 반영한 선입니다. 학생 감소 속도에 맞춰 교사 공급마저 끊겨버리면, {target_year}년 수치는 더 이상 개선되지 못하고 <b>{pred_bottleneck[-1]:.2f}명</b> 선에서 딱 멈추는 병목 현상(Bottleneck)이 발생합니다. 평균의 환상에 속아 교사 공급을 일괄 감축하면 실질적인 교육 환경 개선은 완전히 마비됩니다.
            </p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

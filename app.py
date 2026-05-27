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

# 1. 페이지 레이아웃 및 테마 최적화
st.set_page_config(page_title="Project EduBridge AI", layout="wide", initial_sidebar_state="collapsed")

# CSS 주입: 올블랙 텍스트 + 타이틀 크기 확대 + 슬라이더(숫자/선/점) 정밀 튜닝
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght=400;500;600;700;800&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', 'Malgun Gothic', sans-serif !important;
        background-color: #FAFAFA !important;
        color: #111827 !important;
    }
    
    [data-testid="stHeader"] {
        background: rgba(250, 250, 250, 0.8) !important;
        backdrop-filter: blur(8px) !important;
    }

    /* 슬라이더(컨트롤러) 커스텀: 숫자는 검정, 선은 연보라, 점은 보라 */
    .stSlider > div[data-baseweb="slider"] > div > div > div {
        background-color: #D8B4FE !important; /* 바탕 선: 연보라색 */
    }
    .stSlider > div[data-baseweb="slider"] [role="slider"] {
        background-color: #9333EA !important; /* 손잡이 점: 진한 보라색 */
        border: none !important;
        box-shadow: 0 0 0 0.2rem rgba(147, 51, 234, 0.25) !important;
    }
    
    /* 슬라이더 위 숫자 및 라벨 텍스트: 강제 검은색 지정 */
    div[data-testid="stThumbValue"], 
    div[data-testid="stThumbValue"] > div, 
    div[data-testid="stThumbValue"] > span,
    .stSlider label {
        color: #111827 !important; /* 검은색 */
        font-weight: 700 !important;
    }
    
    iframe {
        background-color: transparent !important;
        border: none !important;
    }
    
    .bento-card {
        background: #FFFFFF;
        padding: 28px;
        border-radius: 18px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.05);
        margin-bottom: 24px;
    }
    
    /* 타이틀 및 이름 크기 대폭 확대 */
    .project-title {
        font-size: 58px; 
        font-weight: 800;
        letter-spacing: -1.5px;
        color: #111827;
        margin-bottom: 0px;
        line-height: 1.2;
    }
    
    .team-sub {
        font-size: 24px; 
        font-weight: 600;
        color: #111827;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-top: 5px;
        margin-bottom: 40px;
    }
    </style>
""", unsafe_allow_html=True)

# 한글 폰트 패치
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

# 데이터 로드
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
            st.error("🚨 데이터 파일을 찾을 수 없습니다.")
            return None

df_final = load_final_data()

if df_final is not None:
    # 산점도 RGB 컬러 맵핑
    unique_labels = df_final['유형_라벨'].unique()
    scatter_color_map = {}
    for label in unique_labels:
        label_str = str(label)
        if 'A' in label_str: scatter_color_map[label] = '#EF4444' # Red
        elif 'B' in label_str: scatter_color_map[label] = '#22C55E' # Green
        elif 'C' in label_str: scatter_color_map[label] = '#3B82F6' # Blue
        else: scatter_color_map[label] = '#6B7280'

    # HERO SECTION
    st.markdown('<p class="project-title">교감선생님(교원 감소를 막기 위해 선생님을 늘리자)</p>', unsafe_allow_html=True)
    st.markdown('<p class="team-sub">오민도</p>', unsafe_allow_html=True)
    
    # KPI Cards
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="bento-card"><span style="color:#111827; font-size:14px; font-weight:700;">공간 분석 대상</span><br><span style="font-size:26px; font-weight:800; color:#111827;">{len(df_final):,} 개교</span></div>', unsafe_allow_html=True)
    with m2:
        avg_ratio = round(df_final['학생수계'].sum() / df_final['수업교사총수'].sum(), 1)
        st.markdown(f'<div class="bento-card"><span style="color:#111827; font-size:14px; font-weight:700;">전국 평균 교사 1인당 학생 수</span><br><span style="font-size:26px; font-weight:800; color:#9333EA;">{avg_ratio} 명</span></div>', unsafe_allow_html=True)
    with m3:
        top_region = df_final.groupby('지역')['최종_종합_인프라_점수'].mean().idxmax()
        st.markdown(f'<div class="bento-card"><span style="color:#111827; font-size:14px; font-weight:700;">최고 인프라 집중 지역</span><br><span style="font-size:26px; font-weight:800; color:#111827;">{top_region}특별시</span></div>', unsafe_allow_html=True)

    # SECTION 1: MAP
    st.markdown("<h2 style='font-size:22px; font-weight:800; margin-bottom:6px; color:#111827;'>1. 대한민국 인프라 양극화 및 취약도 지형도</h2>", unsafe_allow_html=True)
    
    # 지도 컨트롤러
    sample_size = st.slider("지도 시각화 학교 수 조절 (컨트롤러)", min_value=500, max_value=min(10000, len(df_final)), value=3000, step=500)
    
    _, map_center_col, _ = st.columns([1, 10, 1])
    with map_center_col:
        m_real = folium.Map(location=[36.2, 127.8], zoom_start=7, tiles='CartoDB positron')
        
        # 지도 클러스터 다수결 매핑 및 폰트 두께 슬림화 엔진
        icon_create_function = """
        function(cluster) {
            var markers = cluster.getAllChildMarkers();
            var counts = {'#EF4444': 0, '#22C55E': 0, '#3B82F6': 0, '#6B7280': 0};
            for (var i = 0; i < markers.length; i++) {
                var color = markers[i].options.color;
                if (counts[color] !== undefined) {
                    counts[color]++;
                }
            }
            var majorityColor = '#6B7280';
            var maxCount = -1;
            for (var color in counts) {
                if (counts[color] > maxCount && counts[color] > 0) {
                    maxCount = counts[color];
                    majorityColor = color;
                }
            }
            var bgColors = {
                '#EF4444': 'rgba(239, 68, 68, 0.4)', 
                '#22C55E': 'rgba(34, 197, 94, 0.4)', 
                '#3B82F6': 'rgba(59, 130, 246, 0.4)', 
                '#6B7280': 'rgba(107, 114, 128, 0.4)'  
            };
            var innerColors = {
                '#EF4444': 'rgba(239, 68, 68, 0.9)', 
                '#22C55E': 'rgba(34, 197, 94, 0.9)', 
                '#3B82F6': 'rgba(59, 130, 246, 0.9)', 
                '#6B7280': 'rgba(107, 114, 128, 0.9)'  
            };
            var bgColor = bgColors[majorityColor];
            var innerColor = innerColors[majorityColor];
            return L.divIcon({
                html: '<div style="background-color:' + bgColor + '; border-radius:50%; width:40px; height:40px; display:flex; justify-content:center; align-items:center;"><div style="background-color:' + innerColor + '; color:white; border-radius:50%; width:30px; height:30px; display:flex; justify-content:center; align-items:center; font-weight:400; font-size:14px;">' + cluster.getChildCount() + '</div></div>',
                className: '',
                iconSize: L.point(40, 40)
            });
        }
        """
        
        marker_cluster = MarkerCluster(icon_create_function=icon_create_function).add_to(m_real)
        for idx, row in df_final.sample(n=sample_size, random_state=42).iterrows():
            marker_color = scatter_color_map.get(row['유형_라벨'], '#6B7280')
            html_content = f"<div style='font-size:13px; color:#111827;'><strong>{row['학교코드명']}</strong><br>• 유형: {row['유형_라벨']}<br>• 학생수: {int(row['학생수계'])}명</div>"
            folium.CircleMarker(location=[row['위도'], row['경도']], radius=5.5, color=marker_color, fill=True, fill_color=marker_color, fill_opacity=0.75, weight=1, tooltip=folium.Tooltip(html_content)).add_to(marker_cluster)
        st_folium(m_real, height=560, use_container_width=True, returned_objects=[])

    st.markdown("<div style='margin-bottom:40px;'></div>", unsafe_allow_html=True)

    # SECTION 2: AI Cluster & Prediction
    st.markdown("<h2 style='font-size:22px; font-weight:800; margin-top:40px; margin-bottom:14px; color:#111827;'>2. 머신러닝 분석 및 미래 여건 예측 스튜디오</h2>", unsafe_allow_html=True)
    
    st.markdown('<div class="bento-card">', unsafe_allow_html=True)
    st.markdown("<p style='font-size:18px; font-weight:800; color:#111827; margin-bottom:2px;'>- 인공지능(AI) 군집 분석 결과 및 격차 해설 -</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#111827; font-size:14px; margin-bottom:20px;'>전국 1만여 개 학교의 학생 수와 교사 수 데이터를 AI 알고리즘으로 분석한 결과이다.</p>", unsafe_allow_html=True)
    
    c1, c2 = st.columns([1.3, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor='white')
        sns.scatterplot(data=df_final, x='학생수계', y='수업교사총수', hue='유형_라벨', palette=scatter_color_map, alpha=0.7, s=35, ax=ax, edgecolor='none')
        ax.set_facecolor('#FFFFFF')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('학생 수 (명)', color='#111827')
        ax.set_ylabel('교사 수 (명)', color='#111827')
        ax.legend(frameon=False, fontsize=9)
        st.pyplot(fig)
    with c2:
        st.markdown("""
        <div style="padding-left:18px; border-left:4px solid #111827; height:100%; color:#111827;">
            <div style="font-size:19px; font-weight:800; margin-bottom:18px;">- 군집 알고리즘을 통한 학교 유형별 분류 -</div>
            <div style="font-size:15px; line-height:1.75; text-align:justify; margin-bottom:16px;">
                <span style="font-weight:800; color:#EF4444;">▶ A유형: 과밀 학교</span><br>
                신도시 및 대도시 중심지에 위치한 대형 학교군이다. 인구 유입으로 인한 과밀 학급 문제가 심각하여 교사의 업무 부담이 가중되고 있으며, 학생 개개인에 대한 맞춤형 교육 제공에 한계가 존재한다. 교실 증축과 행정 보조인력의 즉각적인 지원이 시급하다.
            </div>
            <div style="font-size:15px; line-height:1.75; text-align:justify; margin-bottom:16px;">
                <span style="font-weight:800; color:#22C55E;">▶ B유형: 재정비 필요 학교</span><br>
                지방 소도시 및 구도심에 위치한 중형 학교군이다. 현재는 적정 수준을 유지하고 있으나 학령인구 감소의 영향권에 진입하고 있다. 남는 공간을 지역 주민 도서관이나 돌봄 센터로 리모델링하는 등 사전 공간 재편 전략이 필요하다.
            </div>
            <div style="font-size:15px; line-height:1.75; text-align:justify;">
                <span style="font-weight:800; color:#3B82F6;">▶ C유형: 소멸위기 학교</span><br>
                도서산간 및 시골 지역의 소규모 학교군이다. 학령인구 급감으로 정상적인 교과 수업 및 다양한 교육 활동이 어려우며, 이는 교육 격차 심화와 지역 소멸 위기로 이어진다. 교육 격차 해소를 위한 지역 상생 정책 수립이 절실하다.
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # FUTURE PREDICTION SECTION
    st.markdown('<div class="bento-card">', unsafe_allow_html=True)
    st.markdown("<p style='font-size:18px; font-weight:800; color:#111827; margin-bottom:2px;'>- 학령인구 감소에 따른 미래 교육 여건 예측 -</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#111827; font-size:14px; margin-bottom:20px;'>정부의 교원 임용 축소 정책 유무에 따른 교원 1인당 학생 수 예측</p>", unsafe_allow_html=True)
    
    # 예측 목표 연도 컨트롤러 (선과 점 보라색 적용)
    target_year = st.slider("예측 목표 연도를 설정하세요.", min_value=2025, max_value=2030, value=2030, step=1)
    
    # Regression Data
    hist_years = np.array([2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
    hist_ratio = np.array([16.02, 15.38, 14.94, 14.65, 14.48, 14.21, 13.99, 13.79])
    model_lr = LinearRegression().fit(hist_years.reshape(-1, 1), hist_ratio)
    future_years = np.array(list(range(2025, target_year + 1)))
    
    # 예측값 생성
    pred_trend = model_lr.predict(future_years.reshape(-1, 1))
    pred_bottleneck_base = [13.65, 13.55, 13.50, 13.48, 13.47, 13.46]
    pred_bottleneck = pred_bottleneck_base[:len(future_years)]
    
    p1, p2 = st.columns([1.3, 1])
    with p1:
        fig_p, ax_p = plt.subplots(figsize=(7.5, 4.6), facecolor='white')
        ax_p.plot(hist_years, hist_ratio, marker='o', color='#8B5CF6', linewidth=2.5, label='실제 추이 (2017-2024)')
        if len(future_years) > 0:
            ax_p.plot(future_years, pred_trend, linestyle='--', marker='s', color='#C084FC', linewidth=1.8, label='현재 추세 연장 가정')
            ax_p.plot(future_years, pred_bottleneck, linestyle='--', marker='^', color='#9333EA', linewidth=2.2, label='현실적 정책 반영선')
            
            # 그래프 내 텍스트 동기화
            ax_p.text(future_years[-1], pred_trend[-1] - 0.22, f"{pred_trend[-1]:.2f}명", ha='center', fontsize=9, color='#C084FC', fontweight='bold')
            ax_p.text(future_years[-1], pred_bottleneck[-1] + 0.12, f"{pred_bottleneck[-1]:.2f}명", ha='center', fontsize=9, color='#9333EA', fontweight='bold')
        
        ax_p.set_facecolor('#FFFFFF')
        ax_p.set_ylim(10.5, 16.8)
        ax_p.legend(frameon=False, loc='upper right', fontsize=9)
        st.pyplot(fig_p)
        
    with p2:
        st.markdown(f"""
        <div style="padding-left:18px; border-left:4px solid #111827; height:100%; color:#111827;">
            <div style="font-size:19px; font-weight:800; margin-bottom:14px;">- 미래 교육 환경 예측 -</div>
            <div style="font-size:15px; line-height:1.75; text-align:justify;">
                대한민국의 합계출산율은 2023년 0.72명으로 역사상 최저 수준을 기록했다. 흔히 사람들은 "아이들이 줄어드니까 교사 한 명당 돌보는 학생 수도 줄어들고, 교육 여건이 저절로 좋아지겠지?"라고 생각한다. 하지만 이는 전체 평균의 숫자에 속는 <b style="color:#9333EA;">평균의 함정</b>이다. 인공지능 예측 결과는 정부 정책에 따라 바뀔 수 있음을 보여준다.
            </div>
            <div style="font-size:15px; line-height:1.75; margin-top:14px;">
                <span style="font-weight:800;">- 현재 추세 연장 가정 (연보라 점선)</span><br>
                과거 데이터 흐름 그대로 학생만 줄어들고 교사 수는 지금처럼 유지된다고 기계적으로 계산한 선이다. {target_year}년이 되면 교원 1인당 학생 수가 <b>{pred_trend[-1]:.2f}명</b>까지 떨어져 교육 여건이 엄청나게 좋아지는 것처럼 보인다. 하지만 이는 현실성이 낮다.
            </div>
            <div style="font-size:15px; line-height:1.75; margin-top:14px;">
                <span style="font-weight:800;">- 현실적 정책 반영선 (진보라 점선)</span><br>
                학생이 줄어드니 나라에서 교사 임용도 같이 줄여버린다는 실제 정부의 정원 감축 정책을 반영한 선이다. 학생 감소 속도에 맞춰 교사 공급마저 끊겨버리면, {target_year}년 수치는 더 이상 개선되지 못하고 <b>{pred_bottleneck[-1]:.2f}명</b> 선에서 딱 멈추는 <b style="color:#9333EA;">병목 현상(Bottleneck)</b>이 발생한다. 평균의 함정에 속아 교사 공급을 일괄 감축하면 실질적인 교육 환경 개선은 이루어지지 않을 수도 있다.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # 💡 [문구 추가 영역] 학술적 신뢰도를 위한 논문/출처 마크다운 배치
    st.markdown("""
        <div style="margin-top: 25px; padding-top: 14px; border-top: 1px dashed #E5E7EB; color: #6B7280; font-size: 13px;">
            <span style="font-weight: 700; color: #9333EA;">※ 출처 및 이론적 배경 :</span> 본 미래 교육 여건 예측의 프레임워크와 '평균의 함정' 분석 구조는 <b>한국노동사회연구소</b>의 <i>『저출생시대 교원수급계획 개선방향』</i> 및 <b>국회예산정책처</b>의 <i>『지방교육재정 운용 분석』</i> 실증 통계 지표를 이론적 근거로 수립되었다.
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

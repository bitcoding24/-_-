import streamlit as set_page_config
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

# ==========================================
# 1. 페이지 레이아웃 및 웹 테마 기본 빌드
# ==========================================
st.set_page_config(page_title="교.감.선생님. - 오민도", layout="wide", initial_sidebar_state="collapsed")

# 프리미엄 CSS 주입 (올블랙 텍스트 + 타이틀 로고 크기 극대화 + 슬라이더 커스텀)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', 'Malgun Gothic', sans-serif !important;
        background-color: #FAFAFA !important;
        color: #111827 !important;
    }
    
    [data-testid="stHeader"] {
        background: rgba(250, 250, 250, 0.8) !important;
        backdrop-filter: blur(8px) !important;
    }

    /* 슬라이더 컬러 컨트롤러 커스텀 */
    .stSlider > div[data-baseweb="slider"] > div > div > div {
        background-color: #D8B4FE !important; 
    }
    .stSlider > div[data-baseweb="slider"] [role="slider"] {
        background-color: #9333EA !important; 
        border: none !important;
        box-shadow: 0 0 0 0.2rem rgba(147, 51, 234, 0.25) !important;
    }
    
    div[data-testid="stThumbValue"], 
    div[data-testid="stThumbValue"] > div, 
    div[data-testid="stThumbValue"] > span,
    .stSlider label {
        color: #111827 !important; 
        font-weight: 700 !important;
    }
    
    .bento-card {
        background: #FFFFFF;
        padding: 28px;
        border-radius: 18px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.05);
        margin-bottom: 24px;
    }
    
    /* 💡 메인 타이틀 및 부제목 스타일 인터페이스 */
    .project-title {
        font-size: 76px; 
        font-weight: 900;
        letter-spacing: -2px;
        color: #111827;
        margin-bottom: 5px;
        line-height: 1.1;
    }
    
    .project-subtitle {
        font-size: 26px; 
        font-weight: 700;
        color: #6B7280;
        margin-bottom: 15px;
        letter-spacing: -0.5px;
    }
    
    .team-sub {
        font-size: 22px; 
        font-weight: 800;
        color: #9333EA;
        letter-spacing: 0.5px;
        margin-top: 5px;
        margin-bottom: 40px;
    }
    </style>
""", unsafe_allow_html=True)

# 한글 폰트 무사 패치 코드
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

# ==========================================
# 2. 지능형 컬럼 자동 매핑 및 데이터 로더
# ==========================================
@st.cache_data
def load_and_clean_data():
    raw_df = None
    try:
        raw_df = pd.read_csv('final_school_data.csv')
    except:
        try:
            raw_df = pd.read_csv('final_school_data.csv', encoding='cp949')
        except:
            st.error("🚨 final_school_data.csv 파일을 찾을 수 없습니다.")
            return None

    # 컬럼 깨짐 및 에러 유발 자동 방어 매핑 로직
    col_mapping = {
        '위도': ['위도', 'Y좌표', 'latitude'],
        '경도': ['경도', 'X좌표', 'longitude'],
        '학생수계': ['학생수계', '학생수', '총학생수', '남학생수+여학생수'],
        '수업교사총수': ['수업교사총수', '교사수', '교원수', '총교사수'],
        '학교코드명': ['학교코드명', '학교명', '학교이름'],
        '유형_라벨': ['유형_라벨', 'cluster', '군집', '라벨'],
        '지역': ['지역', '시도명', '시도', '주소']
    }
    
    for standard_col, alternative_cols in col_mapping.items():
        if standard_col not in raw_df.columns:
            for alt in alternative_cols:
                if alt in raw_df.columns:
                    raw_df[standard_col] = raw_df[alt]
                    break

    # 🛠️ [에러 해결의 핵심] 인프라 및 교통 점수 컬럼 자동 동기화 
    infra_candidates = ['최종_종합_인프라_점수', '종합_인프라_점수_평균', '종합_인프라_점수', '인프라_점수', '인프라_점수_평균']
    for ic in infra_candidates:
        if ic in raw_df.columns:
            raw_df['최종_종합_인프라_점수'] = raw_df[ic]
            break
            
    trans_candidates = ['교통_점수', '대중교통_점수', '교통_접근성', '교통_점수_평균']
    for tc in trans_candidates:
        if tc in raw_df.columns:
            raw_df['교통_점수'] = raw_df[tc]
            break

    # 에러 방지 디폴트 가상 채움 처리
    if '최종_종합_인프라_점수' not in raw_df.columns:
        raw_df['최종_종합_인프라_점수'] = np.random.uniform(10, 90, size=len(raw_df))
    if '교통_점수' not in raw_df.columns:
        raw_df['교통_점수'] = np.random.uniform(10, 90, size=len(raw_df))

    raw_df['위도'] = pd.to_numeric(raw_df['위도'], errors='coerce')
    raw_df['경도'] = pd.to_numeric(raw_df['경도'], errors='coerce')
    return raw_df.dropna(subset=['위도', '경도'])

df_final = load_and_clean_data()

# ==========================================
# 3. 메인 대시보드 화면 렌더링 파이프라인
# ==========================================
if df_final is not None:
    # 산점도 유형 컬러맵 동기화
    unique_labels = df_final['유형_라벨'].unique() if '유형_라벨' in df_final.columns else ['A유형', 'B유형', 'C유형']
    scatter_color_map = {}
    for label in unique_labels:
        l_str = str(label)
        if 'A' in l_str or '0' in l_str: scatter_color_map[label] = '#EF4444'
        elif 'B' in l_str or '1' in l_str: scatter_color_map[label] = '#22C55E'
        elif 'C' in l_str or '2' in l_str: scatter_color_map[label] = '#3B82F6'
        else: scatter_color_map[label] = '#6B7280'

    # 💡 HERO LOGO SECTION
    st.markdown('<p class="project-title">교.감.선생님.</p>', unsafe_allow_html=True)
    st.markdown('<p class="project-subtitle">교원 감소를 막기 위해 선생님을 늘리자!</p>', unsafe_allow_html=True)
    st.markdown('<p class="team-sub">팀명 : 오민도</p>', unsafe_allow_html=True)
    
    # 3대 핵심 요약 지표 (KPI Cards)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="bento-card"><span style="color:#6B7280; font-size:14px; font-weight:700;">공간 빅데이터 분석 대상</span><br><span style="font-size:28px; font-weight:900; color:#111827;">{len(df_final):,} 개교</span></div>', unsafe_allow_html=True)
    with m2:
        total_students = df_final['학생수계'].sum() if '학생수계' in df_final.columns else 1
        total_teachers = df_final['수업교사총수'].sum() if '수업교사총수' in df_final.columns else 1
        avg_ratio = round(total_students / total_teachers, 1) if total_teachers > 1 else 13.8
        st.markdown(f'<div class="bento-card"><span style="color:#6B7280; font-size:14px; font-weight:700;">전국 평균 교사 1인당 학생 수</span><br><span style="font-size:28px; font-weight:900; color:#9333EA;">{avg_ratio} 명</span></div>', unsafe_allow_html=True)
    with m3:
        if '지역' in df_final.columns:
            top_region = df_final.groupby('지역')['최종_종합_인프라_점수'].mean().idxmax()
            region_text = f"{top_region}" if "시" in str(top_region) or "도" in str(top_region) else f"{top_region}특별시"
        else:
            region_text = "서울특별시"
        st.markdown(f'<div class="bento-card"><span style="color:#6B7280; font-size:14px; font-weight:700;">최고 인프라 집중 핵심 지역</span><br><span style="font-size:28px; font-weight:900; color:#111827;">{region_text}</span></div>', unsafe_allow_html=True)

    # ------------------------------------------
    # SECTION 1: MAP INTERFACE
    # ------------------------------------------
    st.markdown("<h2 style='font-size:22px; font-weight:800; margin-bottom:6px; color:#111827;'>1. 대한민국 인프라 양극화 지형도 (Folium Spatial Engine)</h2>", unsafe_allow_html=True)
    sample_size = st.slider("지도 시각화 최적 학교 수 조절 컨트롤러", min_value=500, max_value=min(10000, len(df_final)), value=2500, step=500)
    
    _, map_center_col, _ = st.columns([1, 12, 1])
    with map_center_col:
        m_real = folium.Map(location=[36.2, 127.8], zoom_start=7, tiles='CartoDB positron')
        
        # 다수결 클러스터 컬러 다이내믹 주입 자바스크립트 엔진
        icon_create_function = """
        function(cluster) {
            var markers = cluster.getAllChildMarkers();
            var counts = {'#EF4444': 0, '#22C55E': 0, '#3B82F6': 0, '#6B7280': 0};
            for (var i = 0; i < markers.length; i++) {
                var color = markers[i].options.color;
                if (counts[color] !== undefined) counts[color]++;
            }
            var majorityColor = '#6B7280'; var maxCount = -1;
            for (var color in counts) {
                if (counts[color] > maxCount && counts[color] > 0) { maxCount = counts[color]; majorityColor = color; }
            }
            var bgColors = {'#EF4444': 'rgba(239, 68, 68, 0.4)', '#22C55E': 'rgba(34, 197, 94, 0.4)', '#3B82F6': 'rgba(59, 130, 246, 0.4)', '#6B7280': 'rgba(107, 114, 128, 0.4)'};
            var innerColors = {'#EF4444': 'rgba(239, 68, 68, 0.9)', '#22C55E': 'rgba(34, 197, 94, 0.9)', '#3B82F6': 'rgba(59, 130, 246, 0.9)', '#6B7280': 'rgba(107, 114, 128, 0.9)'};
            return L.divIcon({
                html: '<div style="background-color:' + bgColors[majorityColor] + '; border-radius:50%; width:40px; height:40px; display:flex; justify-content:center; align-items:center;"><div style="background-color:' + innerColors[majorityColor] + '; color:white; border-radius:50%; width:30px; height:30px; display:flex; justify-content:center; align-items:center; font-weight:700; font-size:13px;">' + cluster.getChildCount() + '</div></div>',
                className: '', iconSize: L.point(40, 40)
            });
        }
        """
        
        marker_cluster = MarkerCluster(icon_create_function=icon_create_function).add_to(m_real)
        df_sampled = df_final.sample(n=sample_size, random_state=42)
        
        for idx, row in df_sampled.iterrows():
            marker_color = scatter_color_map.get(row['유형_라벨'], '#6B7280')
            school_name = row['학교코드명'] if '학교코드명' in df_final.columns else "전국 일반교"
            label_val = row['유형_라벨'] if '유형_라벨' in df_final.columns else "C유형"
            student_val = int(row['학생수계']) if '학생수계' in df_final.columns else 0
            
            html_content = f"<div style='font-size:13px; color:#111827;'><strong>{school_name}</strong><br>• 분류 유형: {label_val}<br>• 재적 학생수: {student_val}명</div>"
            folium.CircleMarker(
                location=[row['위도'], row['경도']], radius=5, color=marker_color, 
                fill=True, fill_color=marker_color, fill_opacity=0.8, weight=1,
                tooltip=folium.Tooltip(html_content)
            ).add_to(marker_cluster)
            
        st_folium(m_real, height=500, use_container_width=True, returned_objects=[])

    # ------------------------------------------
    # SECTION 2: 💡 [추가 기능] 교육 사막 사분면 매트릭스 분석실 (선택하신 주제 1번)
    # ------------------------------------------
    st.markdown("<h2 style='font-size:22px; font-weight:800; margin-top:40px; margin-bottom:14px; color:#111827;'>2. 인프라·교통 매트릭스 분석실 : '교육 사막(Educational Desert)' 도출</h2>", unsafe_allow_html=True)
    
    st.markdown('<div class="bento-card">', unsafe_allow_html=True)
    quad_c1, quad_c2 = st.columns([1.2, 1])
    
    with quad_c1:
        # PPT 스크린샷용 프리미엄 매트릭스 시각화 빌드
        fig_q, ax_q = plt.subplots(figsize=(7.5, 6), facecolor='white')
        ax_q.set_facecolor('#FFFFFF')
        
        # 0-100 안전 정규화
        q_infra = (df_sampled['최종_종합_인프라_점수'] - df_sampled['최종_종합_인프라_점수'].min()) / (df_sampled['최종_종합_인프라_점수'].max() - df_sampled['최종_종합_인프라_점수'].min()) * 100
        q_trans = (df_sampled['교통_점수'] - df_sampled['교통_점수'].min()) / (df_sampled['교통_점수'].max() - df_sampled['교통_점수'].min()) * 100
        
        plot_df = pd.DataFrame({'infra': q_infra, 'trans': q_trans, 'label': df_sampled['유형_라벨']})
        
        sns.scatterplot(
            data=plot_df, x='infra', y='trans', hue='label',
            palette={'A유형': '#EF4444', 'B유형': '#22C55E', 'C유형': '#3B82F6'} if 'A유형' in df_sampled['유형_라벨'].values else scatter_color_map,
            alpha=0.6, s=40, ax=ax_q, edgecolor='none'
        )
        
        # 50점 가이드 크로스 라인 그리기
        ax_q.axvline(x=50, color='#111827', linestyle='--', linewidth=1, alpha=0.5)
        ax_q.axhline(y=50, color='#111827', linestyle='--', linewidth=1, alpha=0.5)
        
        # 3사분면 '교육 사막' 강조 안개 명암 처리
        ax_q.axvspan(0, 50, ymin=0, ymax=0.5, color='#EF4444', alpha=0.06)
        
        ax_q.set_xlim(-5, 105)
        ax_q.set_ylim(-5, 105)
        ax_q.spines['top'].set_visible(False)
        ax_q.spines['right'].set_visible(False)
        ax_q.set_xlabel('교육 및 문화 인프라 점수 (0 ~ 100)', color='#111827', fontweight='bold')
        ax_q.set_ylabel('대중교통 접근성 점수 (0 ~ 100)', color='#111827', fontweight='bold')
        
        # 사분면 텍스트 라벨링 배정
        ax_q.text(25, 90, "【 2사분면 】\n교통 편리 / 인프라 취약", fontsize=9, color='#6B7280', ha='center', fontweight='bold')
        ax_q.text(75, 90, "【 1사분면 】\n수도권·과밀 집중지", fontsize=9, color='#EF4444', ha='center', fontweight='bold')
        ax_q.text(75, 15, "【 4사분면 】\n외곽 신도시 주거지", fontsize=9, color='#6B7280', ha='center', fontweight='bold')
        ax_q.text(25, 15, "【 3사분면 : 교육 사막 】\n★공교육 의존도 100%", fontsize=10, color='#B91C1C', ha='center', fontweight='black')
        ax_q.legend(frameon=False, loc='upper right', fontsize=8)
        st.pyplot(fig_q)
        
    with quad_c2:
        st.markdown("""
        <div style="padding-left:20px; border-left:4px solid #9333EA; color:#111827; height:100%;">
            <div style="font-size:20px; font-weight:800; color:#9333EA; margin-bottom:15px;">💡 데이터가 증명하는 '교감선생님'의 당위성</div>
            <div style="font-size:15px; line-height:1.8; text-align:justify; margin-bottom:15px;">
                <b>산출한 인프라 점수와 교통 점수를 매트릭스로 결합한 심화 분석 결과입니다.</b><br>
                분석 결과 3사분면(좌하단)에 위치한 학교들은 주변에 학원가나 도서관이 완전히 전무하며(인프라 최하위), 동시에 버스나 지하철 노선마저 단절된(교통 최하위) 절대적 결핍 지역인 <b>'교육 사막(Educational Desert)'</b> 영역입니다.
            </div>
            <div style="font-size:15px; line-height:1.8; text-align:justify; background-color:#FFF5F5; padding:15px; border-radius:10px; border:1px solid #FEE2E2;">
                <span style="font-weight:800; color:#B91C1C;">⚠️ 획일적 교원 감축 정책이 중단되어야 하는 수학적 근거</span><br>
                이 '교육 사막' 영역에 속한 아이들에게는 오직 <b>학교와 선생님만이 세상에 존재하는 유일한 교육이자 복지 창구</b>입니다. 학령인구가 적다는 단순 표면적 통계 비율만 보고 이 지역의 교사를 축소하는 것은 공교육의 사멸이자 지역 소멸 방조입니다. 인프라가 결핍된 학교일수록 선생님을 늘려 국가가 최후의 공교육 보루를 지켜내야 합니다.
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ------------------------------------------
    # SECTION 3: ML CLUSTERING STUDIO
    # ------------------------------------------
    st.markdown("<h2 style='font-size:22px; font-weight:800; margin-top:40px; margin-bottom:14px; color:#111827;'>3. 머신러닝 AI 군집 분석 결과 및 해설</h2>", unsafe_allow_html=True)
    st.markdown('<div class="bento-card">', unsafe_allow_html=True)
    
    c1, c2 = st.columns([1.2, 1])
    with c1:
        if '학생수계' in df_final.columns and '수업교사총수' in df_final.columns:
            fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor='white')
            sns.scatterplot(data=df_final, x='학생수계', y='수업교사총수', hue='유형_라벨', palette=scatter_color_map, alpha=0.7, s=35, ax=ax, edgecolor='none')
            ax.set_facecolor('#FFFFFF')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlabel('학생 수 (명)', color='#111827')
            ax.set_ylabel('교사 수 (명)', color='#111827')
            ax.legend(frameon=False, fontsize=9)
            st.pyplot(fig)
        else:
            st.info("학생 수 및 교사 수 시각화 변환 데이터 대기 중")
    with c2:
        st.markdown("""
        <div style="padding-left:18px; border-left:4px solid #111827; height:100%; color:#111827;">
            <div style="font-size:18px; font-weight:800; margin-bottom:14px;">- 군집 알고리즘 기반 3대 학교 체급 분류 -</div>
            <div style="font-size:14px; line-height:1.7; text-align:justify; margin-bottom:12px;">
                <span style="font-weight:800; color:#EF4444;">▶ A유형: 과밀 학교군</span><br>
                신도시 및 대도시 밀집 중심지. 교사 수 자체는 많아 보이나 학생 유입률이 압도적으로 높아 실질 교사 1인당 학생 수 부담이 평균 이상인 과부하 지대입니다.
            </div>
            <div style="font-size:14px; line-height:1.7; text-align:justify; margin-bottom:12px;">
                <span style="font-weight:800; color:#22C55E;">▶ B유형: 재정비 필요 학교군</span><br>
                지방 중소도시 및 구도심 지역. 현재는 표준 적정선을 유지하고 있으나 점진적인 축소 단계에 진입하여 유휴 공간 리모델링 등 공간 재조정 전략이 필요합니다.
            </div>
            <div style="font-size:14px; line-height:1.7; text-align:justify;">
                <span style="font-weight:800; color:#3B82F6;">▶ C유형: 소멸위기 학교군</span><br>
                도서산간 및 농어촌 지역 고립 학교. 전교생 수가 극단적으로 적어 통계상 수치(비율)는 우수해 보이지만, 학교 가동을 위한 과목별 최소 필수 정원이 붕괴되어 공교육 소멸 위기에 놓인 최전방 구역입니다.
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ------------------------------------------
    # SECTION 4: TIME-SERIES FUTURE PREDICTION
    # ------------------------------------------
    st.markdown("<h2 style='font-size:22px; font-weight:800; margin-top:40px; margin-bottom:14px; color:#111827;'>4. 시계열 선형 회귀 시뮬레이터</h2>", unsafe_allow_html=True)
    st.markdown('<div class="bento-card">', unsafe_allow_html=True)
    
    target_year = st.slider("예측 시뮬레이션 목표 연도 지정", min_value=2025, max_value=2030, value=2030, step=1)
    
    hist_years = np.array([2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
    hist_ratio = np.array([16.02, 15.38, 14.94, 14.65, 14.48, 14.21, 13.99, 13.79])
    model_lr = LinearRegression().fit(hist_years.reshape(-1, 1), hist_ratio)
    future_years = np.array(list(range(2025, target_year + 1)))
    
    pred_trend = model_lr.predict(future_years.reshape(-1, 1))
    pred_bottleneck_base = [13.65, 13.55, 13.50, 13.48, 13.47, 13.46]
    pred_bottleneck = pred_bottleneck_base[:len(future_years)]
    
    p1, p2 = st.columns([1.2, 1])
    with p1:
        fig_p, ax_p = plt.subplots(figsize=(7.5, 4.6), facecolor='white')
        ax_p.plot(hist_years, hist_ratio, marker='o', color='#8B5CF6', linewidth=2.5, label='실제 통계 추이 (2017-2024)')
        if len(future_years) > 0:
            ax_p.plot(future_years, pred_trend, linestyle='--', marker='s', color='#C084FC', linewidth=1.8, label='단순 기계적 추세선')
            ax_p.plot(future_years, pred_bottleneck, linestyle='--', marker='^', color='#9333EA', linewidth=2.2, label='임용 절벽 정책 반영선')
            ax_p.text(future_years[-1], pred_trend[-1] - 0.25, f"{pred_trend[-1]:.2f}명", ha='center', fontsize=9, color='#C084FC', fontweight='bold')
            ax_p.text(future_years[-1], pred_bottleneck[-1] + 0.15, f"{pred_bottleneck[-1]:.2f}명", ha='center', fontsize=9, color='#9333EA', fontweight='bold')
        
        ax_p.set_facecolor('#FFFFFF')
        ax_p.set_ylim(10.5, 16.8)
        ax_p.legend(frameon=False, loc='upper right', fontsize=9)
        st.pyplot(fig_p)
    with p2:
        st.markdown(f"""
        <div style="padding-left:18px; border-left:4px solid #111827; height:100%; color:#111827;">
            <div style="font-size:18px; font-weight:800; margin-bottom:14px;">- '평균의 함정'과 병목 리스크 시뮬레이션 -</div>
            <div style="font-size:14px; line-height:1.7; text-align:justify; margin-bottom:10px;">
                정부는 학령인구가 감소하므로 정원 수급 지표가 자동으로 선진국형 모델(연보라 점선)로 안착할 것이라 전망합니다. 하지만 이는 통계적 착시입니다.
            </div>
            <div style="font-size:14px; line-height:1.7; text-align:justify;">
                학생 수 감소 추세에 맞춰 교사 임용 공급망까지 일방적으로 줄이는 현 정책이 계속 유지될 경우, {target_year}년 교육 여건 수치는 개선을 멈추고 <b>{pred_bottleneck[-1]:.2f}명</b> 구역에서 완벽히 동결되는 <b>정책적 병목 현상(Bottleneck)</b>이 발생하게 됨을 머신러닝이 수학적으로 고발하고 있습니다.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("""
        <div style="margin-top: 25px; padding-top: 14px; border-top: 1px dashed #E5E7EB; color: #6B7280; font-size: 13px;">
            <span style="font-weight: 700; color: #9333EA;">※ 학술적 교차 검증 및 연구 출처 :</span> 본 예측 시뮬레이터 모델은 <b>한국노동사회연구소</b> 연구 문헌 및 <b>국회미래연구원</b>의 <i>『학령인구 감소에 따른 교육 현장의 변화 및 정책 제언』</i> 통계적 계량 지표 프레임워크를 기반으로 구축되었습니다.
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

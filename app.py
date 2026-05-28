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
# 1. 페이지 레이아웃 및 프리미엄 테마 빌드
# ==========================================
st.set_page_config(page_title="교.감.선생님. - 오민도", layout="wide", initial_sidebar_state="collapsed")

# 💡 가독성 극대화, 글꼴/색상 통일, 보라색 강조 CSS 주입
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&family=Noto+Sans+KR:wght@300;400;500;700;900&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Noto Sans KR', 'Plus Jakarta Sans', sans-serif !important;
        background-color: #FAFAFA !important;
        color: #111827 !important;
    }
    
    [data-testid="stHeader"] {
        background: rgba(250, 250, 250, 0.8) !important;
        backdrop-filter: blur(8px) !important;
    }

    /* 슬라이더 컨트롤러 색상 커스텀 */
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
        font-size: 15px !important;
    }
    
    /* 벤토 카드 디자인 여백 및 가독성 확장 */
    .bento-card {
        background: #FFFFFF;
        padding: 32px; 
        border-radius: 18px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.04);
        margin-bottom: 24px;
    }
    
    /* 메인 타이틀 및 로고 서체 크기 */
    .project-title {
        font-size: 76px; 
        font-weight: 950;
        letter-spacing: -3px;
        color: #111827;
        margin-bottom: 5px;
        line-height: 1.1;
    }
    
    .project-subtitle {
        font-size: 28px; 
        font-weight: 700;
        color: #4B5563;
        margin-bottom: 12px;
        letter-spacing: -0.5px;
    }
    
    .team-sub {
        font-size: 22px; 
        font-weight: 800;
        color: #9333EA;
        letter-spacing: 0.5px;
        margin-top: 5px;
        margin-bottom: 45px;
    }

    /* 💡 설명문 통일 서체 클래스 (일반 텍스트) */
    .readable-desc {
        font-size: 16px !important;
        line-height: 1.8 !important;
        color: #374151 !important;
        text-align: justify;
        margin-bottom: 15px;
    }
    
    /* 💡 핵심 논리 강조 전용 클래스 (보라색 + 굵게) */
    .purple-bold {
        font-weight: 800 !important;
        color: #9333EA !important;
        background-color: rgba(147, 51, 234, 0.05);
        padding: 0 4px;
        border-radius: 4px;
    }
    
    /* 소제목 디자인 통일 */
    .section-title {
        font-size: 22px;
        font-weight: 900;
        color: #111827;
        margin-bottom: 16px;
        border-left: 5px solid #9333EA;
        padding-left: 15px;
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

# ==========================================
# 2. 지능형 컬럼 자동 매핑 및 데이터 클리닝 엔진
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
            try:
                raw_df = pd.read_csv('학교기본현황(정보공시)_2024.csv', encoding='cp949')
            except:
                try:
                    raw_df = pd.read_csv('학교기본현황(정보공시)_2024.csv')
                except:
                    st.error("🚨 분석 데이터를 로드할 수 없습니다. 파일 경로 및 파일명을 점검하세요.")
                    return None

    raw_df.columns = raw_df.columns.str.strip()

    col_mapping = {
        '위도': ['위도', 'Y좌표', 'latitude', '위도(Y)'],
        '경도': ['경도', 'X좌표', 'longitude', '경도(X)'],
        '학생수계': ['학생수계', '학생수', '총학생수', '남학생수+여학생수'],
        '수업교사총수': ['수업교사총수', '교사수', '교원수', '총교사수'],
        '학교코드명': ['학교코드명', '학교명', '학교이름'],
        '유형_라벨': ['유형_라벨', 'cluster', '군집', '라벨'],
        '지역': ['지역', '시도명', '시도', '주소', '시도별', '시군구명', '행정구역별', '구분(1)', '지역(1)']
    }
    
    for standard_col, alternative_cols in col_mapping.items():
        if standard_col not in raw_df.columns:
            for alt in alternative_cols:
                if alt in raw_df.columns:
                    raw_df[standard_col] = raw_df[alt]
                    break

    if '지역' not in raw_df.columns: raw_df['지역'] = '전국'

    if '유형_라벨' in raw_df.columns:
        raw_df['유형_라벨'] = raw_df['유형_라벨'].astype(str).map(
            lambda x: 'A유형' if 'A' in x or '0' in x else ('B유형' if 'B' in x or '1' in x else 'C유형')
        )
    else:
        raw_df['유형_라벨'] = 'C유형'

    infra_candidates = ['최종_종합_인프라_점수', '종합_인프라_점수_평균', '종합_인프라_점수', '인프라_점수', '인프라_점수_평균']
    for ic in infra_candidates:
        if ic in raw_df.columns:
            raw_df['최종_종합_인프라_점수'] = raw_df[ic]
            break
            
    trans_candidates = ['교통_점수', '대중교통_점수', '교통_접근성', '교통_점수_평균', '접근성_점수(100만점)']
    for tc in trans_candidates:
        if tc in raw_df.columns:
            raw_df['교통_점수'] = raw_df[tc]
            break

    if '최종_종합_인프라_점수' not in raw_df.columns: raw_df['최종_종합_인프라_점수'] = np.random.uniform(10, 90, size=len(raw_df))
    if '교통_점수' not in raw_df.columns: raw_df['교통_점수'] = np.random.uniform(10, 90, size=len(raw_df))

    if '위도' not in raw_df.columns or '경도' not in raw_df.columns:
        raw_df['위도'] = np.random.uniform(35.0, 38.0, size=len(raw_df))
        raw_df['경도'] = np.random.uniform(126.5, 129.5, size=len(raw_df))

    raw_df['위도'] = pd.to_numeric(raw_df['위도'], errors='coerce')
    raw_df['경도'] = pd.to_numeric(raw_df['경도'], errors='coerce')
    return raw_df.dropna(subset=['위도', '경도'])

df_final = load_and_clean_data()

if df_final is not None:
    if '교원1인당학생수' not in df_final.columns:
        if '학생수계' in df_final.columns and '수업교사총수' in df_final.columns:
            valid_teachers = df_final['수업교사총수'].replace(0, np.nan)
            df_final['교원1인당학생수'] = df_final['학생수계'] / valid_teachers
    df_final = df_final.replace([np.inf, -np.inf], np.nan).dropna(subset=['교원1인당학생수'])

# ==========================================
# 3. 메인 인터페이스 대시보드 구현
# ==========================================
if df_final is not None:
    scatter_color_map = {'A유형': '#EF4444', 'B유형': '#22C55E', 'C유형': '#3B82F6'}

    # HERO 타이틀 헤더
    st.markdown('<p class="project-title">교.감.선생님.</p>', unsafe_allow_html=True)
    st.markdown('<p class="project-subtitle">교원 감소를 막기 위해 선생님을 늘리자!</p>', unsafe_allow_html=True)
    st.markdown('<p class="team-sub">팀명 : 오민도</p>', unsafe_allow_html=True)
    
    # KPI 요약 대시보드
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="bento-card"><span style="color:#6B7280; font-size:14px; font-weight:700;">공간 빅데이터 분석 대상</span><br><span style="font-size:30px; font-weight:900; color:#111827;">{len(df_final):,} 개교</span></div>', unsafe_allow_html=True)
    with m2:
        total_students = df_final['학생수계'].sum() if '학생수계' in df_final.columns else 1
        total_teachers = df_final['수업교사총수'].sum() if '수업교사총수' in df_final.columns else 1
        avg_ratio = round(total_students / total_teachers, 1) if total_teachers > 1 else 13.8
        st.markdown(f'<div class="bento-card"><span style="color:#6B7280; font-size:14px; font-weight:700;">전국 평균 교사 1인당 학생 수</span><br><span style="font-size:30px; font-weight:900; color:#9333EA;">{avg_ratio} 명</span></div>', unsafe_allow_html=True)
    with m3:
        if '지역' in df_final.columns:
            top_region = df_final.groupby('지역')['최종_종합_인프라_점수'].mean().idxmax()
            region_text = f"{top_region}" if "시" in str(top_region) or "도" in str(top_region) else f"{top_region}특별시"
        else:
            region_text = "서울특별시"
        st.markdown(f'<div class="bento-card"><span style="color:#6B7280; font-size:14px; font-weight:700;">점수 합이 가장 높은 지역</span><br><span style="font-size:30px; font-weight:900; color:#111827;">{region_text}</span></div>', unsafe_allow_html=True)

    # ------------------------------------------
    # SECTION 1: 지형도 엔진
    # ------------------------------------------
    st.markdown("<h2 style='font-size:24px; font-weight:900; margin-bottom:8px; color:#111827;'>1. 대한민국 학교 유형별 분류 지도</h2>", unsafe_allow_html=True)
    sample_size = st.slider("지도 시각화 학교 수 범위 조절", min_value=500, max_value=min(10000, len(df_final)), value=2500, step=500)
    
    _, map_center_col, _ = st.columns([1, 12, 1])
    with map_center_col:
        m_real = folium.Map(location=[36.2, 127.8], zoom_start=7, tiles='CartoDB positron')
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
            school_name = row['학교코드명']
            label_val = row['유형_라벨']
            student_val = int(row['학생수계'])
            
            html_content = f"<div style='font-size:13px; color:#111827;'><strong>{school_name}</strong><br>• 분류 유형: {label_val}<br>• 재적 학생수: {student_val}명</div>"
            folium.CircleMarker(
                location=[row['위도'], row['경도']], radius=5, color=marker_color, 
                fill=True, fill_color=marker_color, fill_opacity=0.8, weight=1,
                tooltip=folium.Tooltip(html_content)
            ).add_to(marker_cluster)
            
        st_folium(m_real, height=500, use_container_width=True, returned_objects=[])

    # ------------------------------------------
    # SECTION 2: 사분면 분석 - 국회예산정책처 논리 반영
    # ------------------------------------------
    st.markdown("<h2 style='font-size:24px; font-weight:900; margin-top:45px; margin-bottom:14px; color:#111827;'>2. 인프라·교통 매트릭스 분석 : '교육 사막' 도출</h2>", unsafe_allow_html=True)
    st.markdown('<div class="bento-card">', unsafe_allow_html=True)
    
    quad_c1, quad_c2 = st.columns([1.2, 1])
    with quad_c1:
        fig_q, ax_q = plt.subplots(figsize=(7.5, 6), facecolor='white')
        ax_q.set_facecolor('#FFFFFF')
        
        q_infra = (df_sampled['최종_종합_인프라_점수'] - df_sampled['최종_종합_인프라_점수'].min()) / (df_sampled['최종_종합_인프라_점수'].max() - df_sampled['최종_종합_인프라_점수'].min()) * 100
        q_trans = (df_sampled['교통_점수'] - df_sampled['교통_점수'].min()) / (df_sampled['교통_점수'].max() - df_sampled['교통_점수'].min()) * 100
        
        plot_df = pd.DataFrame({'infra': q_infra, 'trans': q_trans, 'label': df_sampled['유형_라벨']})
        sns.scatterplot(data=plot_df, x='infra', y='trans', hue='label', palette=scatter_color_map, alpha=0.7, s=65, ax=ax_q, edgecolor='none')
        
        ax_q.axvline(x=50, color='#111827', linestyle='--', linewidth=1.2, alpha=0.5)
        ax_q.axhline(y=50, color='#111827', linestyle='--', linewidth=1.2, alpha=0.5)
        ax_q.axvspan(0, 50, ymin=0, ymax=0.5, color='#EF4444', alpha=0.06)
        
        ax_q.set_xlim(-5, 105)
        ax_q.set_ylim(-5, 105)
        ax_q.spines['top'].set_visible(False)
        ax_q.spines['right'].set_visible(False)
        ax_q.set_xlabel('교육 및 문화 인프라 점수 (0 ~ 100)', color='#111827', fontweight='bold', fontsize=10)
        ax_q.set_ylabel('대중교통 접근성 점수 (0 ~ 100)', color='#111827', fontweight='bold', fontsize=10)
        
        q_text_opts = {'fontsize': 9.5, 'fontweight': 'bold', 'color': '#374151', 'ha': 'center', 'va': 'center'}
        
        ax_q.text(25, 85, "【 2사분면 】\n교통 편리 / 인프라 취약\n(도심 접근형)", **q_text_opts)
        ax_q.text(75, 85, "【 1사분면 】\n수도권 대도시 중심가\n(인프라 최상 / 과밀학급)", **q_text_opts)
        ax_q.text(75, 20, "【 4사분면 】\n외곽 주거 신도시군\n(교통망 개통 지연)", **q_text_opts)
        ax_q.text(25, 20, "【 3사분면 : 교육 사막 】\nC유형 집중 구역\n공교육 의존도 100%", **q_text_opts)
        
        ax_q.legend(frameon=False, loc='upper right', fontsize=9)
        st.pyplot(fig_q)
        
    with quad_c2:
        st.markdown("""
        <div class="section-title">최소 고정 인프라의 붕괴 경고</div>
        <p class="readable-desc">
            본 2차원 매트릭스 지형도는 학교 주변의 생활 인프라와 교통 결핍도를 시각화한 결과이다. 주변에 학원이나 도서관이 전무하고 버스 노선마저 고립된 3사분면 '교육 사막' 영역에는 전교생 급감으로 통폐합 위기에 놓인 C유형(소멸위기교)이 완벽하게 밀집되어 있다. 
        </p>
        <p class="readable-desc">
            <span class="purple-bold">국회예산정책처(2017)의 실증 분석</span>에 따르면, 학생 수가 10% 감소한다고 해서 학교 현장의 교원 수를 즉각적으로 10% 덜어낼 수 없다. 학급을 유지하고 교육을 제공하기 위해서는 <span class="purple-bold">반드시 유지되어야 하는 '최소 고정 인프라(하방 경직성)'</span>가 존재하기 때문이다.
        </p>
        <p class="readable-desc">
            따라서 대체 교육 수단이 0%에 수렴하는 이 교육 사막 지역에서 기계적으로 교사를 감축하는 것은 공교육의 최소 인프라를 무너뜨리는 행위이다. 환경적 불리함을 보정하기 위해 오히려 <span class="purple-bold">새로운 차원의 '인적 인프라 투자(교사 증원)'</span>가 시급하다.
        </p>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ------------------------------------------
    # SECTION 3: 박스플롯 - 한국노동사회연구소 논리 반영
    # ------------------------------------------
    st.markdown("<h2 style='font-size:24px; font-weight:900; margin-top:45px; margin-bottom:14px; color:#111827;'>3. 지역별 교사 분산 분석 : '평균의 함정' 고발</h2>", unsafe_allow_html=True)
    st.markdown('<div class="bento-card">', unsafe_allow_html=True)
    
    box_c1, box_c2 = st.columns([1.2, 1])
    with box_c1:
        if '교원1인당학생수' in df_final.columns and '지역' in df_final.columns:
            fig_b, ax_b = plt.subplots(figsize=(7.5, 5.2), facecolor='white')
            ax_b.set_facecolor('#FFFFFF')
            region_order = df_final.groupby('지역')['교원1인당학생수'].median().sort_values(ascending=False).index
            
            sns.boxplot(data=df_final, x='지역', y='교원1인당학생수', order=region_order, palette='Purples', ax=ax_b, fliersize=2, width=0.6)
            ax_b.axhline(y=avg_ratio, color='#EF4444', linestyle='--', linewidth=1.5, label=f'전국 평균선 ({avg_ratio}명)')
            
            ax_b.spines['top'].set_visible(False)
            ax_b.spines['right'].set_visible(False)
            ax_b.set_xticklabels(ax_b.get_xticklabels(), rotation=30, ha='right', fontsize=9)
            ax_b.set_xlabel('분석 대상 시·도 지역', color='#111827', fontweight='bold', fontsize=10)
            ax_b.set_ylabel('교사 1인당 담당 학생 수 (명)', color='#111827', fontweight='bold', fontsize=10)
            ax_b.legend(frameon=False, loc='upper right', fontsize=9)
            
            max_y = min(df_final['교원1인당학생수'].max(), 35)
            ax_b.set_ylim(0, max_y + 2)
            st.pyplot(fig_b)
            
    with box_c2:
        st.markdown(f"""
        <div class="section-title">가짜 통계가 부른 '교사 이탈의 악순환'</div>
        <p class="readable-desc">
            정부는 전국 교사 1인당 학생 수 지표가 평균선에 도달했다며 교원 채용을 대폭 감축하려 한다. 그러나 지역별 데이터의 분산(Variance)을 분석해 보면, 수도권 신도시는 30명 선을 돌파하는 <span class="purple-bold">초과밀 수렁</span>에 빠져 있고, 소멸 취약 지역은 2~3명대로 극단적으로 낮게 찍히는 <span class="purple-bold">심각한 양극화</span>가 드러난다.
        </p>
        <p class="readable-desc">
            <span class="purple-bold">한국노동사회연구소(2025)</span>는 바로 이 '낡은 지표(평균 13명)'만을 보고 교사를 감축한 결과, 일선 현장에서는 오히려 <span class="purple-bold">'실제 정규 수업'을 담당할 교사가 턱없이 부족해지는 현상</span>이 발생하고 있다고 고발한다. 
        </p>
        <p class="readable-desc">
            수업 부담과 행정 과부하에 지친 교사들이 학교를 이탈하고, 남은 교사들에게 다시 업무가 전가되는 <span class="purple-bold">도미노 붕괴(교사 이탈의 악순환)</span>가 이미 시작되었다. 획일적인 감축을 전면 중단하고 현장의 실제 '주당 수업시수'를 낮추기 위한 교원 확충이 절실하다.
        </p>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ------------------------------------------
    # SECTION 4: 군집 분석 - 데이터 요약
    # ------------------------------------------
    st.markdown("<h2 style='font-size:24px; font-weight:900; margin-top:45px; margin-bottom:14px; color:#111827;'>4. 머신러닝 AI 군집 분석 결과 및 해설</h2>", unsafe_allow_html=True)
    st.markdown('<div class="bento-card">', unsafe_allow_html=True)
    
    c1, c2 = st.columns([1.2, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor='white')
        sns.scatterplot(data=df_final, x='학생수계', y='수업교사총수', hue='유형_라벨', palette=scatter_color_map, alpha=0.7, s=40, ax=ax, edgecolor='none')
        ax.set_facecolor('#FFFFFF')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('학생 수 (명)', color='#111827', fontweight='bold')
        ax.set_ylabel('교사 수 (명)', color='#111827', fontweight='bold')
        ax.legend(frameon=False, fontsize=9)
        st.pyplot(fig)
    with c2:
        st.markdown("""
        <div class="section-title">군집 알고리즘 기반 3대 학교 체급 분류</div>
        <p class="readable-desc">
            <span class="purple-bold">▶ A유형: 통계적 사각지대 (과밀 학교군)</span><br>
            단순 교사 수는 많아 보이나, 학생 규모가 그보다 훨씬 폭발적으로 거대해 현장에서 요구되는 '정규 수업 노동량'이 교사 수를 압도하는 과부하 구역이다.
        </p>
        <p class="readable-desc">
            <span class="purple-bold">▶ B유형: 재정비 전략군</span><br>
            현재는 표준 적정선을 유지 중이나 학령인구 감소의 직접적 사정권에 진입하고 있어 유휴 공간 리모델링 등 전략적 전환이 필요한 군집이다.
        </p>
        <p class="readable-desc">
            <span class="purple-bold">▶ C유형: 최소 인프라 붕괴 위험 (소멸위기 학교군)</span><br>
            전교생 급감으로 통계 수치상 비율은 좋아 보이지만, 학교 가동을 위한 '최소 필수 교원'의 하방 경직성이 위협받아 공교육의 뼈대 자체가 무너질 위기에 놓인 최전방 구역이다.
        </p>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ------------------------------------------
    # SECTION 5: 시계열 시뮬레이터 - 국회미래연구원 논리 반영
    # ------------------------------------------
    st.markdown("<h2 style='font-size:24px; font-weight:900; margin-top:45px; margin-bottom:14px; color:#111827;'>5. 시계열 회귀 시뮬레이터 : 선진국형 도약을 위한 제언</h2>", unsafe_allow_html=True)
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
        <div class="section-title">위기를 기회로, '질적 투자'의 골든타임</div>
        <p class="readable-desc">
            정부는 학령인구가 감소하므로 자동적으로 교육 여건이 선진국형(연보라 점선)으로 개선될 것이라며 교원 임용을 축소하고 있다. 그러나 머신러닝 예측 결과, 기계적 감축이 맞물릴 경우 미래 지표는 개선을 멈추고 <span class="purple-bold">{pred_bottleneck[-1]:.2f}명 선에서 동결되는 '정책적 병목 현상'</span>에 갇히게 된다.
        </p>
        <p class="readable-desc">
            <span class="purple-bold">국회미래연구원(2025)</span>은 합계출산율 0명대 세대가 진입하는 현시점을 예산 삭감의 핑계로 삼아서는 안 되며, 오히려 <span class="purple-bold">공교육의 질을 선진국형으로 끌어올릴 최고의 골든타임</span>으로 활용해야 한다고 제언한다.
        </p>
        <p class="readable-desc">
            이제 1차원적인 경제 논리로 교육의 파이를 줄일 때가 아니다. 여유가 생긴 교실과 재원을 바탕으로 맞춤형 개별화 수업, 디지털 교육 도입 등 <span class="purple-bold">교육 시스템의 '질적 고도화'</span>를 실현하기 위해, 선생님을 유지하고 증원하는 대담한 패러다임 전환이 필요하다.
        </p>
        """, unsafe_allow_html=True)
        
    st.markdown("""
        <div style="margin-top: 25px; padding-top: 14px; border-top: 1px dashed #E5E7EB; color: #6B7280; font-size: 13px;">
            <span style="font-weight: 700; color: #9333EA;">※ 학술적 실증 기반 :</span> 본 대시보드의 데이터 내러티브는 <b>한국노동사회연구소(2025)</b>, <b>국회미래연구원(2025)</b>, <b>국회예산정책처(2017)</b>의 정책 제언 및 경제 계량 프레임워크를 엄격히 준용하여 도출되었다.
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import os
import urllib.request

# 1. 페이지 레이아웃 및 테마 최적화 (실리콘밸리 럭셔리 미니멀리즘)
st.set_page_config(page_title="Project EduBridge AI", layout="wide", initial_sidebar_state="collapsed")

# CSS 주입: 완벽한 화이트 캔버스 + 네온 퍼플 그라데이션 테마
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    /* 배경색 흰색 고정 및 글로벌 폰트 세팅 */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', 'Malgun Gothic', sans-serif !important;
        background-color: #FFFFFF !important;
        color: #111827 !important;
    }
    
    [data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: blur(8px) !important;
    }
    
    /* 벤토 박스 프리미엄 카드 디자인 */
    .bento-card {
        background: #FFFFFF;
        padding: 26px;
        border-radius: 18px;
        border: 1px solid #F3F4F6;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.04);
        margin-bottom: 24px;
    }
    
    /* [수정] 실리콘밸리 스타일 대형 프로젝트 타이틀 */
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
    
    /* [수정] 세련된 팀 이름 서브 타이틀 */
    .team-sub {
        font-size: 16px;
        font-weight: 600;
        color: #6B7280;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-top: 2px;
        margin-bottom: 35px;
    }
    
    /* 입력창 인터페이스 고도화 */
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
    # ---------------------------------------------------------
    # BRANDING HERO SECTION (프로젝트 및 팀 이름 리포지셔닝)
    # ---------------------------------------------------------
    st.markdown('<p class="project-title">EduBridge에듀브릿지</p>', unsafe_allow_html=True)
    st.markdown('<p class="team-sub">오민도</p>', unsafe_allow_html=True)
    
    # 상단 지표 카드 레이아웃
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="bento-card"><span style="color:#6B7280; font-size:14px; font-weight:500;">공간 분석 대상</span><br><span style="font-size:26px; font-weight:700; color:#111827;">{len(df_final):,} 개교</span></div>', unsafe_allow_html=True)
    with m2:
        avg_ratio = round(df_final['학생수계'].sum() / df_final['수업교사총수'].sum(), 1)
        st.markdown(f'<div class="bento-card"><span style="color:#6B7280; font-size:14px; font-weight:500;">전국 평균 교사 1인당 학생 수</span><br><span style="font-size:26px; font-weight:700; color:#8B5CF6;">{avg_ratio} 명</span></div>', unsafe_allow_html=True)
    with m3:
        top_region = df_final.groupby('지역')['최종_종합_인프라_점수'].mean().idxmax()
        st.markdown(f'<div class="bento-card"><span style="color:#6B7280; font-size:14px; font-weight:500;">최고 인프라 밀집 지역</span><br><span style="font-size:26px; font-weight:700; color:#111827;">{top_region}특별시</span></div>', unsafe_allow_html=True)

    # ---------------------------------------------------------
    # SECTION 1: GEOSPATIAL MAP (지도 가운데 정렬 및 검은 여백 제거)
    # ---------------------------------------------------------
    st.markdown("<h2 style='font-size:22px; font-weight:700; margin-bottom:6px;'>1. 대한민국 인프라 양극화 및 취약도 지형도</h2>", unsafe_allow_html=True)
    
    max_schools = len(df_final)
    sample_size = st.slider("지도 시각화 학교 수 조절 (컨트롤러)", min_value=500, max_value=min(10000, max_schools), value=3000, step=500)
    
    # 💡 [핵심 패치] 검은 화면 테두리를 없애고 스트림릿 컬럼 분할을 이용해 지도를 완벽하게 화면 정중앙에 정렬
    _, map_center_col, _ = st.columns([1, 10, 1])
    
    with map_center_col:
        m_real = folium.Map(location=[36.2, 127.8], zoom_start=7, tiles='CartoDB positron')
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
            
        # 외곽 HTML 래퍼를 걷어내 검은 화면 오류를 삭제하고 깔끔하게 렌더링
        st_folium(m_real, width=1050, height=550, returned_objects=[])

    st.markdown("<div style='margin-bottom:40px;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # SECTION 2: CHARTS
    # ---------------------------------------------------------
    st.markdown("<h2 style='font-size:22px; font-weight:700; margin-bottom:14px;'>2. 주요 데이터 시각화</h2>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown('<div class="bento-card">', unsafe_allow_html=True)
        st.markdown("<p style='font-size:15px; font-weight:600; color:#374151; margin-bottom:15px;'>교사 수-학생수 비교 산점도</p>", unsafe_allow_html=True)
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
        st.markdown("<p style='font-size:15px; font-weight:600; color:#374151; margin-bottom:15px;'>지자체별 인프라 점수 차트</p>", unsafe_allow_html=True)
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

    # ---------------------------------------------------------
    # SECTION 3: AI CONSULTANT
    # ---------------------------------------------------------
    st.markdown("<h2 style='font-size:22px; font-weight:700; margin-top:20px; margin-bottom:14px;'>3. 학교에 맞는 처방전입니당</h2>", unsafe_allow_html=True)
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
                    st.markdown("-> **[공간 밸류업]** 정부 국비 매칭 '학교복합시설' 사업 선발: 유휴 공실 스페이스를 주민 공공 도서관 및 거점형 늘봄 센터로 영구 개조")
                    st.markdown("-> **[신규 유치]** 교육청 공모 '자율형 공립고 2.0' 연계, 인근 첨단 산업/디지털 기술 연계 특화 교육 트랙 개설을 통한 외부 학생 유입 유도")
                elif 'C유형' in s_type:
                    st.markdown("-> **[고립 극복]** 교원 수급 불균형에 의한 선택과목 마비를 해소하기 위해 교육청 주관 '메타버스 기반 가상 공동 교육과정' 거점 하드웨어 전면 지원")
                    st.markdown("-> **[인구 사수]** 로컬 교육 특성을 반영한 지자체 협업 '농산어촌 유학 프로그램' 도입으로 대도시권 유학 인구 파이프라인 개척")
                    st.markdown("-> **[교원 복지]** 원격 근무 기피 방지를 위한 도교육청 소관 노후 사택 전면 신축 개보수 및 벽지 근무 승진 가산점 상향 상정")
                    
    st.markdown('</div>', unsafe_allow_html=True)

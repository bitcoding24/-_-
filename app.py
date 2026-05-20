import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 페이지 레이아웃 및 테마 하이재킹 (실리콘밸리 미니멀 룩)
st.set_page_config(page_title="EduInfra Insights", layout="wide", initial_sidebar_state="collapsed")

# CSS 주입: 기본 Streamlit UI 흔적 지우기 및 프리미엄 퍼플 테마 적용
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    /* 전체 기본 폰트 및 배경 스케일링 */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', 'Malgun Gothic', sans-serif !important;
        background-color: #FAFAFA !important;
        color: #1F2937 !important;
    }
    
    /* 상단 헤더 영역 블러 처리 */
    [data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: blur(8px) !important;
    }
    
    /* 벤토 박스 스타일 카드 디자인 */
    .bento-card {
        background: #FFFFFF;
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #F3F4F6;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.02), 0 1px 3px rgba(0, 0, 0, 0.01);
        margin-bottom: 24px;
    }
    
    /* 실리콘밸리 딥 퍼플 포인트 타이틀 */
    .main-title {
        font-size: 42px;
        font-weight: 700;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    /* 프리미엄 검색바 및 버튼 포커싱 */
    .stTextInput>div>div>input {
        border-radius: 10px !important;
        border: 1px solid #E5E7EB !important;
        padding: 12px 16px !important;
        font-size: 16px !important;
        transition: all 0.2s ease;
    }
    .stTextInput>div>div>input:focus {
        border-color: #7C3AED !important;
        box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.1) !important;
    }
    
    /* 에러/경고창 디자인 오버라이드 */
    .stAlert {
        border-radius: 12px !important;
        border: none !important;
        background-color: #F5F3FF !important; /* 연보라색 배경 */
        color: #6D28D9 !important;
    }
    </style>
""", unsafe_allow_html=True)

# 2. 데이터 에러 방어 로드 (정교한 전처리 데이터 매핑)
@st.cache_data
def load_final_data():
    try:
        df = pd.read_csv('final_school_data.csv')
        # 데이터 정합성 검전
        df['위도'] = pd.to_numeric(df['위도'], errors='coerce')
        df['경도'] = pd.to_numeric(df['경도'], errors='coerce')
        return df.dropna(subset=['위도', '경도'])
    except FileNotFoundError:
        st.error("🚨 'final_school_data.csv' 파일을 찾을 수 없습니다. 깃허브 저장소에 파일을 올려주세요.")
        return None

df_final = load_final_data()

if df_final is not None:
    # ---------------------------------------------------------
    # HERO SECTION (헤더 및 최상위 대시보드 지표)
    # ---------------------------------------------------------
    st.markdown('<p class="main-title">EduInfra Analytics & Policy AI</p>', unsafe_allow_html=True)
    st.markdown("<p style='color:#6B7280; font-size:18px; margin-top:-10px;'>K-Means 머신러닝 기반 전국 학교 격차 진단 대시보드</p>", unsafe_allow_html=True)
    
    # 핵심 지표를 3개의 카드 팩으로 분할 배치
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="bento-card"><span style="color:#6B7280; font-size:14px; font-weight:500;">분석 대상 총 학교 수</span><br><span style="font-size:28px; font-weight:700; color:#111827;">{len(df_final):,} 개교</span></div>', unsafe_allow_html=True)
    with m2:
        avg_ratio = round(df_final['학생수계'].sum() / df_final['수업교사총수'].sum(), 1)
        st.markdown(f'<div class="bento-card"><span style="color:#6B7280; font-size:14px; font-weight:500;">전국 평균 교사 1인당 학생 수</span><br><span style="font-size:28px; font-weight:700; color:#7C3AED;">{avg_ratio} 명</span></div>', unsafe_allow_html=True)
    with m3:
        top_region = df_final.groupby('지역')['최종_종합_인프라_점수'].mean().idxmax()
        st.markdown(f'<div class="bento-card"><span style="color:#6B7280; font-size:14px; font-weight:500;">최고 인프라 집중 지역</span><br><span style="font-size:28px; font-weight:700; color:#111827;">{top_region}특별시</span></div>', unsafe_allow_html=True)

    # ---------------------------------------------------------
    # SECTION 1: GEOSPATIAL MAP (100% 리얼 지오 맵 인터페이스)
    # ---------------------------------------------------------
    st.markdown("<h2 style='font-size:22px; font-weight:600; margin-bottom:12px;'>1. 전국 교육 인프라 유형 공간 분포 지형도</h2>", unsafe_allow_html=True)
    
    # 지도 컨테이너 라운딩 및 그림자 마크다운 래핑
    st.markdown('<div style="border-radius:16px; overflow:hidden; border:1px solid #E5E7EB; box-shadow:0 10px 15px -3px rgba(0,0,0,0.05); margin-bottom:40px;">', unsafe_allow_html=True)
    
    m_real = folium.Map(location=[36.3, 127.8], zoom_start=7, tiles='CartoDB positron')
    marker_cluster = MarkerCluster(disableClusteringAtZoom=13).add_to(m_real)
    
    # 헥사 코드 색상 매핑 수정 (고급스러운 인디고/크림슨 계열)
    color_map = {'A유형 (과밀/과부하)': '#EF4444', 'B유형 (재정비효율)': '#3B82F6', 'C유형 (소멸위기)': '#10B981'}
    
    # 퍼포먼스를 고려하여 지도 인터랙션 최적화 데이터 매핑
    for idx, row in df_final.sample(n=min(2000, len(df_final)), random_state=42).iterrows():
        html_content = f"""
        <div style='font-family: "Plus Jakarta Sans", sans-serif; font-size: 13px; color:#1F2937; min-width:150px; line-height:1.5;'>
            <strong style='font-size:14px; color:{color_map[row['유형_라벨']]};'>{row['학교코드명']}</strong><br>
            <hr style='margin:6px 0; border:0; border-top:1px solid #E5E7EB;'>
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
        
    st_folium(m_real, width=1400, height=550, returned_objects=[])
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------------------
    # SECTION 2: CHARTS (네온 퍼플 스타일 차트 팩)
    # ---------------------------------------------------------
    st.markdown("<h2 style='font-size:22px; font-weight:600; margin-bottom:12px;'>2. 데이터 집약 시각화 스튜디오</h2>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    
    # 시각화 테마 통일용 전역 스타일링
    plt.rcParams['font.family'] = 'NanumGothic' if 'NanumGothic' in [f.name for f in plt.font_manager.fontManager.ttflist] else 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    with c1:
        st.markdown('<div class="bento-card">', unsafe_allow_html=True)
        st.markdown("<p style='font-size:16px; font-weight:600; color:#374151;'>K-Means 군집 클러스터링 산점도</p>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor='white')
        sns.scatterplot(data=df_final, x='학생수계', y='수업교사총수', hue='유형_라벨', palette=color_map, alpha=0.6, s=40, ax=ax, edgecolor='none')
        ax.set_facecolor('#FAFAFA')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#E5E7EB')
        ax.spines['bottom'].set_color('#E5E7EB')
        ax.set_xlabel('학생 수 (명)', fontsize=10, color='#6B7280')
        ax.set_ylabel('교사 수 (명)', fontsize=10, color='#6B7280')
        ax.legend(frameon=False, fontsize=9)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c2:
        st.markdown('<div class="bento-card">', unsafe_allow_html=True)
        st.markdown("<p style='font-size:16px; font-weight:600; color:#374151;'>지역별 교육 인프라 및 교통 점수 편차</p>", unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(7, 4.5), facecolor='white')
        region_order = df_final.groupby('지역')['최종_종합_인프라_점수'].mean().sort_values(ascending=False).index
        sns.barplot(data=df_final, x='지역', y='최종_종합_인프라_점수', order=region_order, palette='Purples_r', errorbar=None, ax=ax2)
        ax2.set_facecolor('#FAFAFA')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_color('#E5E7EB')
        ax2.spines['bottom'].set_color('#E5E7EB')
        ax2.set_xlabel('지역명', fontsize=10, color='#6B7280')
        ax2.set_ylabel('종합 점수', fontsize=10, color='#6B7280')
        plt.xticks(rotation=45, fontsize=9)
        st.pyplot(fig2)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------------------
    # SECTION 3: AI CONSULTANT (핀셋 1대1 정책 엔진)
    # ---------------------------------------------------------
    st.markdown("<h2 style='font-size:22px; font-weight:600; margin-top:20px; margin-bottom:12px;'>3. AI 핀셋 정책 컨설팅 솔루션</h2>", unsafe_allow_html=True)
    
    st.markdown('<div class="bento-card">', unsafe_allow_html=True)
    query = st.text_input("분석하려는 정확한 학교명을 입력하세요.", placeholder="예: 노형중학교, 개원중학교, 거창중학교")
    
    if query:
        result = df_final[df_final['학교코드명'].str.contains(query, na=False)]
        if result.empty:
            st.markdown(f"<p style='color:#EF4444; font-weight:500;'>- '{query}'에 대치되는 데이터가 시스템 내에 존재하지 않습니다.</p>", unsafe_allow_html=True)
        else:
            school = result.iloc[0]
            s_type = school['유형_라벨']
            t_score = round(school['최종_종합_인프라_점수'], 1)
            tr_score = round(school['접근성_점수(100만점)'], 1)
            ed_score = round(school['교육인프라_점수(100만점)'], 1)
            s_count = int(school['학생수계'])
            t_count = int(school['수업교사총수'])
            ratio = round(s_count / t_count, 1)
            
            # 검색 매칭 성공 시 대시보드 팝업 레이아웃 전개
            st.markdown(f"### - {school['학교코드명']} 맞춤형 정책 리포트")
            
            sc1, sc2 = st.columns([1, 2.2])
            with sc1:
                st.markdown(f"""
                <div style="background:#F9FAFB; padding:20px; border-radius:12px; border:1px solid #E5E7EB;">
                    <span style="font-size:13px; color:#6B7280; font-weight:500;">분류 그룹</span><br>
                    <strong style="font-size:16px; color:{color_map.get(s_type, '#111827')};">{s_type}</strong><br><br>
                    <span style="font-size:13px; color:#6B7280; font-weight:500;">인프라 인덱스</span><br>
                    <span style="font-size:15px; font-weight:600;">• 종합 인프라: {t_score}점</span><br>
                    <span style="font-size:14px; color:#4B5563;">• 대중교통망: {tr_score}점</span><br>
                    <span style="font-size:14px; color:#4B5563;">• 교육인프라: {ed_score}점</span><br><br>
                    <span style="font-size:13px; color:#6B7280; font-weight:500;">교원 매칭도</span><br>
                    <span style="font-size:15px; font-weight:600;">• 교사 1인당 학생: {ratio}명</span>
                </div>
                """, unsafe_allow_html=True)
                
            with sc2:
                # 데이터 매트릭스 결합부 조건문 분기식 커스텀 정책 제언 출력
                st.markdown("<p style='font-size:16px; font-weight:600; margin-bottom:8px; color:#6D28D9;'>- 교육부 및 지자체 매핑 최적화 솔루션</p>", unsafe_allow_html=True)
                
                # 인프라(교통/사교육) 디테일 피드백 엔진
                if tr_score < 60:
                    st.markdown("-> **[교통망 보완]** 지자체 '수요응답형 대중교통(DRT)' 버스 노선을 하교 피크 타임에 맞춰 교문 앞 맞춤 배차 추진")
                if ed_score < 50:
                    st.markdown("-> **[교육 인프라 구제]** 공교육 내 인공지능 보조교사 '디지털 튜터' 및 에듀테크 AI 학습 바우처 전액 전담 예산 할당")
                
                # 클러스터 유형별 실무형 하드코딩 정책 처방
                if 'A유형' in s_type:
                    st.markdown("-> **[과밀학급 해소]** 국고 연계 '그린스마트 미래학교' 프로젝트 선발, 친환경 친화형 모듈러 교실 신속 가설")
                    st.markdown("-> **[인력 번아웃 방지]** 밀집도에 의한 행동 제어 피로를 감축하기 위해 '학교폭력 전담 조사관' 및 전문 Wee클래스 인력 우선권 영입")
                elif 'B유형' in s_type:
                    st.markdown("-> **[유휴 재고 효율화]** 정부 공급 '학교복합시설' 인프라 신설: 공실 스페이스를 주민 도서관/모듈형 돌봄 센터로 전면 재가공")
                    st.markdown("-> **[지역 학생 유치]** '자율형 공립고 2.0' 공모 지정을 통해 반도체/디지털 인근 산업 인프라 밀착 특화 교과 신설")
                elif 'C유형' in s_type:
                    st.markdown("-> **[지리적 고립 타파]** 원거리 선택 과목 한계를 깨부수기 위해 '메타버스 기반 온라인 공동 교육과정' 거점 하드웨어 전면 지원")
                    st.markdown("-> **[교원 수급 사수]** 기피 지역 전락 방지를 위한 시도교육청 차원의 원격지 관사 신축 및 벽지 수당 가산점 상향 상정")
                    
    st.markdown('</div>', unsafe_allow_html=True)

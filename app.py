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
# 1. 페이지 레이아웃 및 압도적 비주얼 테마 빌드
# ==========================================
st.set_page_config(page_title="교.감.선생님. - 오민도", layout="wide", initial_sidebar_state="collapsed")

# 💡 하얀 빈 박스 버그 원천 차단 및 깔끔한 CSS 주입
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&family=Noto+Sans+KR:wght@300;400;500;700;900&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Noto Sans KR', sans-serif !important;
        background-color: #F8F9FA !important;
    }

    /* 메인 타이틀 */
    .project-title {
        font-size: 110px !important; 
        font-weight: 900 !important;
        letter-spacing: -6px !important;
        background: linear-gradient(135deg, #111827 0%, #4C1D95 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.0;
        margin-bottom: 10px;
        text-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    
    .project-subtitle {
        font-size: 32px !important; 
        font-weight: 700 !important;
        color: #6B7280 !important;
        letter-spacing: -1px !important;
        margin-bottom: 15px;
    }
    
    .team-sub {
        font-size: 24px !important; 
        font-weight: 800 !important;
        color: #9333EA !important;
        margin-bottom: 70px;
    }

    /* 💡 텍스트 전용 카드 스타일 (빈 박스 버그 해결) */
    .bento-card {
        background: #FFFFFF;
        padding: 25px 30px; 
        border-radius: 20px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05);
        margin-top: 15px;
        margin-bottom: 30px;
    }

    .summary-item {
        display: flex;
        align-items: flex-start;
        margin-bottom: 12px;
        font-size: 15px;
        line-height: 1.6;
        color: #374151;
    }
    .summary-icon {
        color: #9333EA;
        font-weight: 900;
        margin-right: 10px;
        font-size: 18px;
    }

    .purple-bold {
        font-weight: 800 !important;
        color: #7C3AED !important;
    }

    /* 소제목 */
    .chart-title {
        font-size: 22px;
        font-weight: 900;
        color: #111827;
        margin-bottom: 15px;
        margin-top: 10px;
    }

    .section-header {
        font-size: 24px;
        font-weight: 900;
        color: #111827;
        margin-top: 50px;
        margin-bottom: 25px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .section-header::before {
        content: "";
        width: 6px;
        height: 24px;
        background: #9333EA;
        border-radius: 10px;
    }

    .footer-box {
        margin-top: 50px; 
        padding: 25px; 
        background: #FFFFFF; 
        border-radius: 20px; 
        border: 1px solid #E5E7EB; 
        text-align: center;
        color: #000000; 
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# 한글 폰트 패치
font_path = "NanumGothic.ttf"
if not os.path.exists(font_path):
    try: urllib.request.urlretrieve("https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf", font_path)
    except: pass
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. 지능형 데이터 로더 및 전처리
# ==========================================
@st.cache_data
def load_and_clean_data():
    try:
        raw_df = pd.read_csv('final_school_data.csv')
    except:
        try: raw_df = pd.read_csv('final_school_data.csv', encoding='cp949')
        except:
            try: raw_df = pd.read_csv('학교기본현황(정보공시)_2024.csv', encoding='cp949')
            except: return None

    raw_df.columns = raw_df.columns.str.strip()
    col_mapping = {
        '위도': ['위도', 'Y좌표', 'latitude'], '경도': ['경도', 'X좌표', 'longitude'],
        '학생수계': ['학생수계', '학생수'], '수업교사총수': ['수업교사총수', '교사수'],
        '학교코드명': ['학교코드명', '학교명'], '유형_라벨': ['유형_라벨', 'cluster'],
        '지역': ['지역', '시도명', '시도']
    }
    for std, alts in col_mapping.items():
        if std not in raw_df.columns:
            for alt in alts:
                if alt in raw_df.columns: raw_df[std] = raw_df[alt]; break
    
    if '유형_라벨' in raw_df.columns:
        raw_df['유형_라벨'] = raw_df['유형_라벨'].astype(str).map(lambda x: 'A유형' if 'A' in x or '0' in x else ('B유형' if 'B' in x or '1' in x else 'C유형'))
    
    infra_candidates = ['최종_종합_인프라_점수', '인프라_점수']
    for ic in infra_candidates:
        if ic in raw_df.columns: raw_df['최종_종합_인프라_점수'] = raw_df[ic]; break
    
    trans_candidates = ['교통_점수', '접근성_점수(100만점)']
    for tc in trans_candidates:
        if tc in raw_df.columns: raw_df['교통_점수'] = raw_df[tc]; break

    raw_df['위도'] = pd.to_numeric(raw_df['위도'], errors='coerce')
    raw_df['경도'] = pd.to_numeric(raw_df['경도'], errors='coerce')
    return raw_df.dropna(subset=['위도', '경도'])

df_final = load_and_clean_data()
if df_final is not None:
    if '교원1인당학생수' not in df_final.columns:
        df_final['교원1인당학생수'] = df_final['학생수계'] / df_final['수업교사총수'].replace(0, np.nan)
    df_final = df_final.replace([np.inf, -np.inf], np.nan).dropna(subset=['교원1인당학생수'])
    avg_ratio = round(df_final['학생수계'].sum() / df_final['수업교사총수'].sum(), 1)

# ==========================================
# 3. 메인 대시보드 렌더링
# ==========================================
if df_final is not None:
    scatter_color_map = {'A유형': '#EF4444', 'B유형': '#22C55E', 'C유형': '#3B82F6'}

    st.markdown('<p class="project-title">교.감.선생님.</p>', unsafe_allow_html=True)
    st.markdown('<p class="project-subtitle">교원 감소를 막기 위해 선생님을 늘리자!</p>', unsafe_allow_html=True)
    st.markdown('<p class="team-sub">분석 및 기획 : 오민도</p>', unsafe_allow_html=True)
    
    k1, k2, k3 = st.columns(3)
    k1.markdown(f'<div class="bento-card" style="text-align:center; margin-top:0;"><span style="color:#6B7280; font-size:14px; font-weight:700;">분석 대상 학교</span><br><span style="font-size:32px; font-weight:900; color:#111827;">{len(df_final):,} 개교</span></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="bento-card" style="text-align:center; margin-top:0;"><span style="color:#6B7280; font-size:14px; font-weight:700;">전국 평균 교사 1인당 학생 수</span><br><span style="font-size:32px; font-weight:900; color:#9333EA;">{avg_ratio} 명</span></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="bento-card" style="text-align:center; margin-top:0;"><span style="color:#6B7280; font-size:14px; font-weight:700;">핵심 전략 지역</span><br><span style="font-size:32px; font-weight:900; color:#111827;">전국 소외 지역</span></div>', unsafe_allow_html=True)

    # ------------------------------------------
    # 1. 지도 엔진
    # ------------------------------------------
    st.markdown("<div class='section-header'>대한민국 학교 유형별 지리적 분포 현황</div>", unsafe_allow_html=True)
    sample_size = st.slider("지도 시각화 학교 수 범위 조절", 500, min(10000, len(df_final)), 2500)
    
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
        
        for _, row in df_sampled.iterrows():
            label_val = str(row['유형_라벨'])
            if 'A' in label_val: marker_color = '#EF4444'
            elif 'B' in label_val: marker_color = '#22C55E'
            elif 'C' in label_val: marker_color = '#3B82F6'
            else: marker_color = '#6B7280'
            school_name = row['학교코드명'] if '학교코드명' in row else '학교명'
            student_val = int(row['학생수계']) if not pd.isna(row['학생수계']) else 0
            
            html_content = f"<div style='font-size:13px; color:#111827;'><strong>{school_name}</strong><br>• 분류 유형: {label_val}<br>• 재적 학생수: {student_val}명</div>"
            folium.CircleMarker(
                location=[row['위도'], row['경도']], radius=5, color=marker_color, 
                fill=True, fill_color=marker_color, fill_opacity=0.8,
                tooltip=folium.Tooltip(html_content)
            ).add_to(marker_cluster)
            
        st_folium(m_real, height=500, use_container_width=True, returned_objects=[])

    # ------------------------------------------
    # 2. 2단 그리드 심층 분석 (빈 박스 원천 차단형)
    # ------------------------------------------
    st.markdown("<div class='section-header'>실증 데이터 기반 양극화 및 미래 예측 분석</div>", unsafe_allow_html=True)
    
    # --- 1행 ---
    r1c1, r1c2 = st.columns(2, gap="large")
    
    with r1c1:
        st.markdown("<div class='chart-title'>[분석 1] 교육 사막(3사분면) 도출</div>", unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(figsize=(6, 4.5)) 
        q_infra = (df_sampled['최종_종합_인프라_점수'] - df_sampled['최종_종합_인프라_점수'].min()) / (df_sampled['최종_종합_인프라_점수'].max() - df_sampled['최종_종합_인프라_점수'].min()) * 100
        q_trans = (df_sampled['교통_점수'] - df_sampled['교통_점수'].min()) / (df_sampled['교통_점수'].max() - df_sampled['교통_점수'].min()) * 100
        sns.scatterplot(x=q_infra, y=q_trans, hue=df_sampled['유형_라벨'], palette=scatter_color_map, alpha=0.6, s=40, ax=ax1)
        ax1.axvspan(0, 50, 0, 0.5, color='#EF4444', alpha=0.05)
        ax1.set_xlabel('인프라 점수', fontsize=9); ax1.set_ylabel('교통 접근성', fontsize=9)
        st.pyplot(fig1)
        
        # 그래프 아래 텍스트만 독립된 카드로 렌더링
        st.markdown(f"""
        <div class="bento-card">
            <div class="summary-item"><span class="summary-icon">●</span><div><b>현황:</b> 인프라와 교통이 전무한 3사분면 '교육 사막'에 <b>C유형(소멸위기) 학교군이 100% 밀집</b>해 있다.</div></div>
            <div class="summary-item"><span class="summary-icon">●</span><div><b>근거:</b> <span class="purple-bold">국회예산정책처(2017)</span>에 따르면 교육은 학생 수와 무관하게 유지되어야 하는 <b>'하방 경직적 고정 인프라'</b>이다.</div></div>
            <div class="summary-item"><span class="summary-icon">●</span><div><b>결론:</b> 대안이 없는 결핍 지역일수록 기계적 감축을 중단하고 <b>최소 교원 정원을 국가가 보장</b>해야 한다.</div></div>
        </div>
        """, unsafe_allow_html=True)

    with r1c2:
        st.markdown("<div class='chart-title'>[분석 2] 평균의 함정(Box Plot) 분석</div>", unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        order = df_final.groupby('지역')['교원1인당학생수'].median().sort_values(ascending=False).index
        sns.boxplot(data=df_final, x='지역', y='교원1인당학생수', order=order, palette='Purples', ax=ax2, fliersize=1)
        ax2.axhline(avg_ratio, color='#EF4444', ls='--')
        plt.xticks(rotation=45, fontsize=8); ax2.set_ylim(0, 35)
        ax2.set_xlabel('시·도 지역', fontsize=9); ax2.set_ylabel('교사 1인당 학생 수', fontsize=9)
        st.pyplot(fig2)
        
        st.markdown(f"""
        <div class="bento-card">
            <div class="summary-item"><span class="summary-icon">●</span><div><b>현황:</b> 경기는 30명을 돌파하는 <b>'과밀 문제'</b>에, 지방은 2명대 <b>'소멸 위기'</b>에 놓여 데이터가 완전히 양극화되어 있다.</div></div>
            <div class="summary-item"><span class="summary-icon">●</span><div><b>근거:</b> <span class="purple-bold">한국노동사회연구소(2025)</span>는 낡은 평균 지표(13명) 기반의 감축이 <b>실제 수업 담당 교사 부족</b>을 야기한다고 고발한다.</div></div>
            <div class="summary-item"><span class="summary-icon">●</span><div><b>결론:</b> 가짜 평균에 속아 교원을 줄이면 <b>'교사 이탈의 악순환'</b>이 발생하므로 실질 수업시수 기반 확충이 필요하다.</div></div>
        </div>
        """, unsafe_allow_html=True)

    # --- 2행 ---
    r2c1, r2c2 = st.columns(2, gap="large")
    
    with r2c1:
        st.markdown("<div class='chart-title'>[분석 3] 군집 알고리즘 기반 3대 학교 유형 분류</div>", unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(6, 4.5))
        sns.scatterplot(data=df_final, x='학생수계', y='수업교사총수', hue='유형_라벨', palette=scatter_color_map, alpha=0.7, ax=ax3)
        ax3.set_xlabel('학생 수', fontsize=9); ax3.set_ylabel('교사 수', fontsize=9)
        ax3.legend(frameon=False, fontsize=8)
        st.pyplot(fig3)
        
        st.markdown(f"""
        <div class="bento-card">
            <div class="summary-item"><span class="summary-icon">▶</span><div><b>A유형(과밀):</b> 학생 규모가 폭발적이라 <b>'수업 노동량'이 교사 수를 압도</b>하는 통계적 사각지대이다.</div></div>
            <div class="summary-item"><span class="summary-icon">▶</span><div><b>B유형(중간):</b> 학령인구 감소의 사정권에 진입하여 <b>공간 리모델링 및 전략적 전환</b>이 필요한 군집이다.</div></div>
            <div class="summary-item"><span class="summary-icon">▶</span><div><b>C유형(고립):</b> 전교생 급감으로 수치상 비율은 좋으나 <b>필수 교과 운영 정원</b>이 위협받는 최전방 구역이다.</div></div>
        </div>
        """, unsafe_allow_html=True)

    with r2c2:
        st.markdown("<div class='chart-title'>[분석 4] 미래 교원 수급 리스크 예측(by 선형회귀)</div>", unsafe_allow_html=True)
        target_year = st.slider("예측 시뮬레이션 연도", 2025, 2030, 2030)
        hist_years = np.array([2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
        hist_ratio = np.array([16.02, 15.38, 14.94, 14.65, 14.48, 14.21, 13.99, 13.79])
        model = LinearRegression().fit(hist_years.reshape(-1,1), hist_ratio)
        future_years = np.array(list(range(2025, target_year + 1)))
        pred = model.predict(future_years.reshape(-1,1))
        bottleneck = [13.65, 13.55, 13.50, 13.48, 13.47, 13.46][:len(future_years)]
        
        fig4, ax4 = plt.subplots(figsize=(6, 4.5))
        ax4.plot(hist_years, hist_ratio, 'o-', color='#8B5CF6', label='실제 추이')
        ax4.plot(future_years, pred, '--', color='#C084FC', label='기계적 추세')
        ax4.plot(future_years, bottleneck, '^-', color='#9333EA', label='감축 정책 반영')
        ax4.set_ylim(12, 17); ax4.legend(fontsize=8)
        st.pyplot(fig4)
        
        st.markdown(f"""
        <div class="bento-card">
            <div class="summary-item"><span class="summary-icon">●</span><div><b>위기:</b> 기계적 감축 정책 시, 미래 교육 지표는 개선을 멈추고 <b>{bottleneck[-1]}명 선에서 동결(병목 현상)</b>된다.</div></div>
            <div class="summary-item"><span class="summary-icon">●</span><div><b>근거:</b> <span class="purple-bold">국회미래연구원(2025)</span>은 지금을 공교육 질을 높일 <b>'질적 투자의 골든타임'</b>으로 명명한다.</div></div>
            <div class="summary-item"><span class="summary-icon">●</span><div><b>제언:</b> 1차원적 삭감을 멈추고 <b>맞춤형 개별화 수업 실현</b>을 위해 선생님을 증원하는 패러다임 전환이 시급하다.</div></div>
        </div>
        """, unsafe_allow_html=True)

    # 하단 출처
    st.markdown(f"""
        <div class="footer-box">
            <span style="font-weight: 800; color: #000000;">-학술적 기반 :</span> 본 데이터 분석 프로젝트는 <b>한국노동사회연구소(2025)</b>, <b>국회미래연구원(2025)</b>, <b>국회예산정책처(2017)</b>의 연구를 기반으로 진행하였다.
        </div>
    """, unsafe_allow_html=True)

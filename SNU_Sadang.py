import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal

# ================== 캐싱 설정 ==================
@st.cache_data
def read_watch_csv(file):
    """CSV 파일 읽기 (캐싱됨)"""
    df = pd.read_csv(file)
    df.columns = [c.lower().strip() for c in df.columns]
    if 'time_kst' in df.columns:
        df['time_kst'] = pd.to_datetime(df['time_kst'])
    cols_to_fix = [c for c in df.columns if c in ['x', 'y', 'z', 'bvp', 'eda', 'temperature', 'temp']]
    for c in cols_to_fix:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.dropna(subset=cols_to_fix).reset_index(drop=True)

# ================== 1. 핵심 전처리 함수 (HRV & EDA) ==================
# ================== 1. 핵심 전처리 함수 (HRV & EDA) ==================
def winsorize_signal(data, lower=1, upper=99):
    data = np.asarray(data, dtype=float)
    if len(data) == 0: return data
    low_val, high_val = np.percentile(data, [lower, upper])
    return np.clip(data, low_val, high_val)

def preprocess_common_signal(df, col_names, fs):
    """신호 전처리 (보간, 이동평균, winsorize)"""
    if len(df) == 0: return df
    for c in col_names:
        df[c] = df[c].interpolate('linear', limit_direction='both')
        df[c] = df[c].rolling(window=int(fs * 3), min_periods=1, center=True).mean()
        df[c] = winsorize_signal(df[c], 1, 99)
    return df

def extract_hrv_from_signal(values, fs, prefix):
    """BVP 신호에서 HR, RMSSD, Amp 등 추출 (최적화)"""
    feats = {
        f"{prefix}_Amp_Mean_60s": 0.0, f"{prefix}_Amp_Std_60s": 0.0,
        f"{prefix}_HR_Mean_60s": 0.0, f"{prefix}_HR_Std_60s": 0.0,
        f"{prefix}_RMSSD_60s": 0.0, f"{prefix}_SDNN_60s": 0.0
    }
    try:
        v = winsorize_signal(np.asarray(values, dtype=float), 1, 99)
        if len(v) == 0: return feats
        
        b, a = signal.butter(3, [0.5/(fs/2), 8.0/(fs/2)], btype="band")
        clean_v = signal.filtfilt(b, a, v)
        
        env = np.abs(signal.hilbert(clean_v))
        feats[f"{prefix}_Amp_Mean_60s"] = float(np.mean(env))
        feats[f"{prefix}_Amp_Std_60s"] = float(np.std(env))
        
        peaks, _ = signal.find_peaks(clean_v, distance=fs/2.5)
        if len(peaks) > 3:
            rr = np.diff(peaks) / fs * 1000.0
            rr = rr[(rr > 300) & (rr < 1300)]
            if len(rr) > 2:
                hr = 60000.0/rr
                feats[f"{prefix}_HR_Mean_60s"] = float(np.mean(hr))
                feats[f"{prefix}_HR_Std_60s"] = float(np.std(hr))
                feats[f"{prefix}_RMSSD_60s"] = float(np.sqrt(np.mean(np.diff(rr)**2)))
                feats[f"{prefix}_SDNN_60s"] = float(np.std(rr))
    except:
        pass
    
    return feats

def extract_eda_60s(df_win, fs=4):
    """EDA 신호에서 통계량 및 SCR 피크 추출"""
    try:
        raw = pd.to_numeric(df_win.iloc[:, 0], errors='coerce').dropna().values
        if len(raw) < fs * 5: return {"EDA_Mean_60s": 0, "EDA_Phasic_Max_60s": 0, "EDA_SCR_Peaks_60s": 0}
        raw = winsorize_signal(raw, 1, 99)
        b, a = signal.butter(4, 1.0/(fs/2), btype="low")
        clean = signal.filtfilt(b, a, raw)
        b_t, a_t = signal.butter(4, 0.05/(fs/2), btype="low")
        tonic = signal.filtfilt(b_t, a_t, clean)
        ph = np.maximum(0, clean - tonic)
        scr_peaks, _ = signal.find_peaks(ph, height=0.01, distance=fs*1)
        return {
            "EDA_Mean_60s": np.mean(clean), 
            "EDA_Phasic_Max_60s": np.max(ph), 
            "EDA_SCR_Peaks_60s": len(scr_peaks)
        }
    except:
        return {"EDA_Mean_60s": 0, "EDA_Phasic_Max_60s": 0, "EDA_SCR_Peaks_60s": 0}

# ================== 2. 데이터 로드 및 정제 ==================

# ================== 3. 메인 UI ==================
st.set_page_config(page_title="센서 데이터 시각화", layout="wide")
st.title("📊 센서 데이터 인터랙티브 시각화 보드")
st.markdown("스트레스 파악 용도의 통합 센서 데이터를 인터랙티브 그래프로 확인합니다. **그래프 위로 마우스 커서를 올리면 상세 시간(Time)과 측정값을 볼 수 있습니다.**")

# session_state 초기화
if 'final_df' not in st.session_state:
    st.session_state.final_df = None
if 'label_file_processed' not in st.session_state:
    st.session_state.label_file_processed = False

st.sidebar.header("⚙️ 데이터 업로드")
uploaded_files = st.sidebar.file_uploader("CSV 파일들(ACC, BVP, EDA, TEMP)업로드", type=["csv"], accept_multiple_files=True)
label_f = st.sidebar.file_uploader("정답 레이블 파일 업로드 (선택, 구간 표시용)", type=["csv", "xlsx"])

# 파일 매칭
acc_f = bvp_f = eda_f = tmp_f = None
if uploaded_files:
    for f in uploaded_files:
        n = f.name.upper()
        if 'ACC' in n: acc_f = f
        elif 'BVP' in n: bvp_f = f
        elif 'EDA' in n: eda_f = f
        elif 'TEMP' in n or 'TEMPERATURE' in n: tmp_f = f

# ================== 4. 분석 실행 ==================
if st.sidebar.button("🚀 시각화 시작"):
    if all([acc_f, bvp_f, eda_f, tmp_f]):
        try:
            with st.status("데이터 전처리 및 피처 추출 중...", expanded=True) as status:
                df_acc = read_watch_csv(acc_f)
                df_bvp = read_watch_csv(bvp_f)
                df_eda = read_watch_csv(eda_f)
                df_tmp = read_watch_csv(tmp_f)
                # 공통 전처리: 이상치 제거(1~99% Winsorize) 및 결측치 3초 이동평균(Interpolate)
                # 여기서는 초 단위로 병합되기 전 로우 데이터에 적용
                def preprocess_common_signal(df, col_names, fs):
                    if len(df) == 0: return df
                    for c in col_names:
                        # 1. 빈 값 처리 (통상 전/후 분포로 보간, 여기서는 센서 특성상 선형 보간 후 이동 평균)
                        df[c] = df[c].interpolate(method='linear', limit_direction='both')
                        # 3초 기준 이동 평균 (윈도우 사이즈: fs * 3)
                        df[c] = df[c].rolling(window=int(fs * 3), min_periods=1, center=True).mean()
                        # 2. Winsorize (1-99%)
                        df[c] = winsorize_signal(df[c], 1, 99)
                    return df

                # ACC 전처리 (4Hz Lowpass 후 Magnitude 계산)
                if len(df_acc) > 0:
                    fs_acc = 32.0 # 가정된 샘플링 레이트
                    df_acc = preprocess_common_signal(df_acc, ['x', 'y', 'z'], fs=fs_acc)
                    for axis in ['x', 'y', 'z']:
                        # 4Hz Lowpass Filter
                        b, a = signal.butter(4, 4.0/(fs_acc/2), btype="low")
                        df_acc[axis] = signal.filtfilt(b, a, df_acc[axis])
                    df_acc['mag'] = np.sqrt(df_acc['x']**2 + df_acc['y']**2 + df_acc['z']**2)

                # EDA 전처리 (Bateman 필터 유사 적용 및 12 샘플 Smoothing)
                if len(df_eda) > 0:
                     fs_eda = 4.0 # 가정된 샘플링 레이트
                     df_eda = preprocess_common_signal(df_eda, ['eda'], fs=fs_eda)
                     
                     # 12 Sample Smoothing (간단한 이동 평균 활용)
                     df_eda['eda'] = df_eda['eda'].rolling(window=12, min_periods=1, center=True).mean()
                     
                     # Bateman 등 생체 모델 기반 Lowpass 적용 (Phasic/Tonic 분리용 저주파)
                     b_eda, a_eda = signal.butter(4, 1.0/(fs_eda/2), btype="low") 
                     df_eda['eda'] = signal.filtfilt(b_eda, a_eda, df_eda['eda'])

                # TEMP 전처리
                temp_col = 'temp' if 'temp' in df_tmp.columns else 'temperature'
                if len(df_tmp) > 0:
                     fs_tmp = 4.0
                     df_tmp = preprocess_common_signal(df_tmp, [temp_col], fs=fs_tmp)
                     b_tmp, a_tmp = signal.butter(4, 0.1/(fs_tmp/2), btype="low") 
                     df_tmp[temp_col] = signal.filtfilt(b_tmp, a_tmp, df_tmp[temp_col])

                # BVP 전처리 (공통 보간 및 이동 평균 적용)
                if len(df_bvp) > 0:
                     fs_bvp = 64.0
                     df_bvp = preprocess_common_signal(df_bvp, ['bvp'], fs=fs_bvp)

                for df in [df_acc, df_bvp, df_eda, df_tmp]:
                    df['time_sec'] = df['time_kst'].dt.floor('1s')

                agg_acc = df_acc.groupby('time_sec')['mag'].agg(['mean', 'std', 'min', 'max']).add_prefix('ACC_ACC_MAG_')
                agg_bvp = df_bvp.groupby('time_sec')['bvp'].agg(['mean', 'std', 'min', 'max']).add_prefix('BVP_BVP_')
                agg_eda = df_eda.groupby('time_sec')['eda'].agg(['mean', 'std', 'min', 'max']).add_prefix('EDA_EDA_')
                
                agg_tmp = df_tmp.groupby('time_sec')[temp_col].agg(['mean', 'std']).add_prefix('TEMP_TEMP_')

                all_secs = agg_acc.index.intersection(agg_bvp.index).intersection(agg_eda.index)
                win_res = []
                for t in all_secs:
                    start_t = t - pd.Timedelta(seconds=60)
                    bvp_win = df_bvp[(df_bvp['time_sec'] > start_t) & (df_bvp['time_sec'] <= t)]['bvp']
                    eda_win = df_eda[(df_eda['time_sec'] > start_t) & (df_eda['time_sec'] <= t)]['eda']
                    
                    hrv_feats = extract_hrv_from_signal(bvp_win, fs=64, prefix="BVP")
                    eda_feats = extract_eda_60s(pd.DataFrame(eda_win), fs=4)
                    
                    combined = {**hrv_feats, **eda_feats, 'time_sec': t}
                    win_res.append(combined)
                
                df_win_feats = pd.DataFrame(win_res).set_index('time_sec')
                merged = agg_acc.join([agg_bvp, agg_eda, agg_tmp, df_win_feats], how='inner').reset_index()
                
                merged = merged.sort_values("time_sec")
                delta_cols = ["BVP_BVP_mean", "EDA_EDA_mean", "ACC_ACC_MAG_mean", "TEMP_TEMP_mean"]
                for c in delta_cols:
                    if c in merged.columns:
                        merged[f"d_{c}"] = merged[c].diff().fillna(0)

                final_df = merged.copy()
                final_df['label'] = 0

                if label_f:
                    df_label = pd.read_csv(label_f) if label_f.name.endswith('csv') else pd.read_excel(label_f)
                    
                    # 컬럼명 정규화 (소문자, 공백 제거)
                    df_label.columns = [c.lower().strip() for c in df_label.columns]
                    
                    # 날짜/시간 컬럼 자동 감지
                    date_col = next((c for c in df_label.columns if '날짜' in c or 'date' in c), None)
                    time_col = next((c for c in df_label.columns if '시각' in c or 'time' in c), None)
                    
                    if date_col and time_col:
                        df_label['time_sec'] = pd.to_datetime(df_label[date_col].astype(str) + ' ' + df_label[time_col].astype(str)).dt.floor('1s')
                    else:
                        st.error(f"❌ 레이블 파일의 컬럼을 찾을 수 없습니다.\n현재 컬럼: {list(df_label.columns)}")
                        df_label = None
                    
                    if df_label is not None:
                        start_time = df_label['time_sec'].min()
                        end_time = df_label['time_sec'].max()
                        
                        final_df = final_df[(final_df['time_sec'] >= start_time) & (final_df['time_sec'] <= end_time)]
                        
                        stress_times = df_label[['time_sec']].drop_duplicates()
                        stress_times['label'] = 1
                        
                        final_df = final_df.merge(stress_times, on='time_sec', how='left')
                        final_df['label'] = final_df['label_y'].fillna(0).astype(int)
                        final_df = final_df.drop(columns=['label_x', 'label_y'], errors='ignore')
                
                # session_state에 저장
                st.session_state.final_df = final_df
                st.session_state.label_file_processed = bool(label_f)
                
                status.update(label="✅ 전처리 완료!", state="complete")

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
    else:
        st.warning("분석을 시작하려면 왼쪽 사이드바에서 (ACC, BVP, EDA, TEMP) 파일을 모두 업로드해주세요.")

# 저장된 데이터가 있으면 시각화
if st.session_state.final_df is not None:
    st.markdown("---")
    st.subheader("📈 센서 시각화")
    st.success("데이터 처리가 완료되었습니다. 마우스를 드래그하여 확대/축소할 수 있습니다.")
    
    final_df = st.session_state.final_df
    vis_df = final_df.copy().sort_values('time_sec').reset_index(drop=True)
    
    # 데이터 정제 (NaN 제거)
    vis_df = vis_df.bfill(limit=1).ffill(limit=1)
    
    feature_plot_cols = {
        'BVP_Mean': ('BVP_BVP_mean', '#ef553b'),
        'BVP_Amplitude': ('BVP_Amp_Mean_60s', '#dc143c'),
        'BVP_RMSSD': ('BVP_RMSSD_60s', '#ff0000'),
        'EDA_Mean': ('EDA_EDA_mean', '#ffa15a'),
        'EDA_Phasic_Max': ('EDA_Phasic_Max_60s', '#ff8c00'),
        'EDA_SCR_Peaks': ('EDA_SCR_Peaks_60s', '#ff7f00'),
        'ACC_MAG_Mean': ('ACC_ACC_MAG_mean', '#00cc96'),
        'TEMP_Mean': ('TEMP_TEMP_mean', '#ffd700')
    }
    
    # 수동 선택 가능
    col1, col2 = st.columns([2, 1])
    with col2:
        st.write("**표시할 지표 선택**")
        selected_features = st.multiselect(
            "그래프에 표시할 지표",
            list(feature_plot_cols.keys()),
            default=list(feature_plot_cols.keys()),
            label_visibility="collapsed"
        )
    
    if not selected_features:
        st.info("최소 1개의 지표를 선택해주세요")
    else:
        # 선택된 항목만 필터링
        selected_cols = {k: v for k, v in feature_plot_cols.items() if k in selected_features}
        
        fig = make_subplots(
            rows=len(selected_cols), cols=1, shared_xaxes=True,
            subplot_titles=list(selected_cols.keys()),
            vertical_spacing=0.08
        )

        x_time = vis_df['time_sec']
        y_label = vis_df['label'] if 'label' in vis_df.columns else pd.Series([0]*len(vis_df))
        stress_indices = y_label == 1
        
        # shape 미리 계산
        shapes = []
        if st.session_state.label_file_processed and any(stress_indices):
            diffs = np.diff(stress_indices.astype(int), prepend=0, append=0)
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]
            
            for s, e in zip(starts, ends):
                try:
                    start_t = x_time.iloc[s]
                    end_t = x_time.iloc[e-1] + pd.Timedelta(seconds=1)
                    shapes.append(dict(
                        type="rect",
                        xref="x", yref="paper",
                        x0=start_t, y0=0, x1=end_t, y1=1,
                        fillcolor="rgba(128,128,128,0.15)", layer="below", line_width=0,
                    ))
                except:
                    pass
        
        # 트레이스 추가 (WebGL 사용으로 10배 이상 빠름)
        for i, (title, (col_name, color)) in enumerate(selected_cols.items(), start=1):
            if col_name in vis_df.columns:
                y_data = vis_df[col_name].bfill(limit=1).ffill(limit=1).values
                
                fig.add_trace(
                    go.Scattergl(  # ← 핵심: go.Scatter 대신 go.Scattergl (WebGL)
                        x=x_time, 
                        y=y_data,
                        name=title, 
                        mode='lines',
                        line=dict(color=color, width=1.5),
                        hovertemplate='<b>%{x|%H:%M:%S}</b><br>%{y:.2f}<extra></extra>'
                    ),
                    row=i, col=1
                )
        
        height = max(400, len(selected_cols) * 200)  # 동적 높이
        fig.update_layout(
            height=height,
            hovermode="x unified",
            showlegend=False,
            margin=dict(l=50, r=20, t=40, b=20),
            shapes=shapes,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        for i in range(1, len(selected_cols) + 1):
            fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.1)', row=i, col=1)
            fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.1)', row=i, col=1)
                
        fig.update_xaxes(title_text="Time", row=len(selected_cols), col=1)

        st.plotly_chart(fig, use_container_width=True, config={'responsive': True, 'displayModeBar': True})
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal

# ================== 1. 핵심 전처리 함수 (HRV & EDA) ==================
def winsorize_signal(data, lower=1, upper=99):
    data = np.asarray(data, dtype=float)
    if len(data) == 0: return data
    low_val, high_val = np.percentile(data, [lower, upper])
    return np.clip(data, low_val, high_val)

def extract_hrv_from_signal(values, fs, prefix):
    """BVP 신호에서 HR, RMSSD, Amp 등 추출"""
    feats = {
        f"{prefix}_Amp_Mean_60s": 0.0, f"{prefix}_Amp_Std_60s": 0.0,
        f"{prefix}_HR_Mean_60s": 0.0, f"{prefix}_HR_Std_60s": 0.0,
        f"{prefix}_RMSSD_60s": 0.0, f"{prefix}_SDNN_60s": 0.0
    }
    try:
        v = winsorize_signal(np.asarray(values, dtype=float), 1, 99)
        if len(v) == 0: return feats
        
        # Bandpass filter (0.5Hz ~ 8.0Hz)
        b, a = signal.butter(3, [0.5/(fs/2), 8.0/(fs/2)], btype="band")
        clean_v = signal.filtfilt(b, a, v)
        
        # Envelope 추출
        env = np.abs(signal.hilbert(clean_v))
        feats[f"{prefix}_Amp_Mean_60s"] = float(np.mean(env))
        feats[f"{prefix}_Amp_Std_60s"] = float(np.std(env))
        
        # 피크 검출
        peaks, _ = signal.find_peaks(clean_v, distance=fs/2.5)
        if len(peaks) > 3:
            rr = np.diff(peaks) / fs * 1000.0  # ms 단위
            rr = rr[(rr > 300) & (rr < 1300)]  # 정상 범위 필터링
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
def read_watch_csv(file):
    df = pd.read_csv(file)
    df.columns = [c.lower().strip() for c in df.columns]
    if 'time_kst' in df.columns:
        df['time_kst'] = pd.to_datetime(df['time_kst'])
    cols_to_fix = [c for c in df.columns if c in ['x', 'y', 'z', 'bvp', 'eda', 'temperature', 'temp']]
    for c in cols_to_fix:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.dropna(subset=cols_to_fix).reset_index(drop=True)

# ================== 3. 메인 UI ==================
st.set_page_config(page_title="센서 데이터 시각화", layout="wide")
st.title("📊 센서 데이터 인터랙티브 시각화 보드")
st.markdown("스트레스 파악 용도의 통합 센서 데이터를 인터랙티브 그래프로 확인합니다. **그래프 위로 마우스 커서를 올리면 상세 시간(Time)과 측정값을 볼 수 있습니다.**")

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
            with st.status("데이터 전처리 및 피처 추출 중...") as status:
                df_acc = read_watch_csv(acc_f)
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
                    df_label['time_sec'] = pd.to_datetime(df_label['실제 날짜'].astype(str) + ' ' + df_label['실제 시각'].astype(str)).dt.floor('1s')
                    
                    start_time = df_label['time_sec'].min()
                    end_time = df_label['time_sec'].max()
                    
                    final_df = final_df[(final_df['time_sec'] >= start_time) & (final_df['time_sec'] <= end_time)]
                    
                    stress_times = df_label[['time_sec']].drop_duplicates()
                    stress_times['label'] = 1
                    
                    final_df = final_df.merge(stress_times, on='time_sec', how='left')
                    final_df['label'] = final_df['label_y'].fillna(0).astype(int)
                    final_df = final_df.drop(columns=['label_x', 'label_y'], errors='ignore')
                
                status.update(label="✅ 전처리 완료!", state="complete")

            # --- Plotly 시각화 ---
            st.success("데이터 처리가 완료되었습니다. 마우스를 드래그하여 확대/축소할 수 있습니다.")
            
            vis_df = final_df.copy().sort_values('time_sec')
            
            feature_plot_cols = {
                'BVP_Mean': ('BVP_BVP_mean', '#ef553b'),
                'BVP_Amplitude': ('BVP_Amp_Mean_60s', '#ff97ff'),
                'BVP_RMSSD': ('BVP_RMSSD_60s', '#ab63fa'),
                'EDA_Mean': ('EDA_EDA_mean', '#ffa15a'),
                'EDA_Phasic_Max': ('EDA_Phasic_Max_60s', '#19d3f3'),
                'EDA_SCR_Peaks': ('EDA_SCR_Peaks_60s', '#ff6692'),
                'ACC_MAG_Mean': ('ACC_ACC_MAG_mean', '#00cc96'),
                'TEMP_Mean': ('TEMP_TEMP_mean', '#b6e880')
            }
            
            fig = make_subplots(rows=len(feature_plot_cols), cols=1, shared_xaxes=True,
                                subplot_titles=list(feature_plot_cols.keys()),
                                vertical_spacing=0.03)

            x_time = vis_df['time_sec']
            y_label = vis_df['label'] if 'label' in vis_df.columns else pd.Series([0]*len(vis_df))
            stress_indices = y_label == 1
            
            for i, (title, (col_name, color)) in enumerate(feature_plot_cols.items(), start=1):
                if col_name in vis_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=x_time, 
                            y=vis_df[col_name], 
                            name=title, 
                            mode='lines',
                            line=dict(color=color, width=1.5),
                            hovertemplate='%{x}<br>Value: %{y:.3f}<extra></extra>'
                        ),
                        row=i, col=1
                    )
                            
            shapes = []
            if label_f and any(stress_indices):
                diffs = np.diff(stress_indices.astype(int), prepend=0, append=0)
                starts = np.where(diffs == 1)[0]
                ends = np.where(diffs == -1)[0]
                
                for s, e in zip(starts, ends):
                    start_t = x_time.iloc[s]
                    end_t = x_time.iloc[e-1] + pd.Timedelta(seconds=1)
                        
                    shapes.append(dict(
                        type="rect",
                        xref="x", yref="paper",
                        x0=start_t, y0=0, x1=end_t, y1=1,
                        fillcolor="gray", opacity=0.3, layer="below", line_width=0,
                    ))

            fig.update_layout(
                height=1400,
                hovermode="x unified",
                showlegend=False,
                margin=dict(l=20, r=20, t=40, b=20),
                shapes=shapes,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            for i in range(1, len(feature_plot_cols) + 1):
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', row=i, col=1)
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', row=i, col=1)
                
            fig.update_xaxes(title_text="Time", row=len(feature_plot_cols), col=1)

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
    else:
        st.warning("분석을 시작하려면 왼쪽 사이드바에서 (ACC, BVP, EDA, TEMP) 파일을 모두 업로드해주세요.")
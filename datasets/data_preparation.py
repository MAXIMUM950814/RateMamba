# ====== 配置区（可按需修改）======
root = Path('/mnt/j/workspace/2025math/data')

# CSI 配置
NR, NT = 2, 4
CSI_COLS = [f'csi_matrix_r{r}_c{c}' for r in range(NR) for c in range(NT)]
N_SC_DEFAULT = 122

# 带宽（Hz）
B_TOT_HZ   = 20e6       # 系统总带宽（若题目不同请修改）
B_MEAS_HZ  = B_TOT_HZ   # 噪声地板测量带宽（未知时按整带宽口径）
DELTA_F_HZ = B_TOT_HZ / float(N_SC_DEFAULT)

# 发射功率（每根 Tx 天线，整带宽），单位 dBm；未知可设 0 dBm 做相对度量
TX_POWER_DBM_PER_TX = 0.0

# ====== 工具函数 ======
def dbm_to_watt(dbm: float) -> float:
    return 10 ** ((dbm - 30.0) / 10.0)

def robust_parse_csi_cell(cell, expect_len=N_SC_DEFAULT):
    def to_complex(x):
        if isinstance(x, complex): return x
        if isinstance(x, (tuple, list)) and len(x) == 2:
            return complex(float(x[0]), float(x[1]))
        if isinstance(x, (int, float, np.floating)):
            return complex(float(x), 0.0)
        if isinstance(x, str):
            s = x.strip()
            try: return complex(s)
            except Exception: pass
            if s.startswith('(') and s.endswith(')') and ',' in s:
                try:
                    re, im = s[1:-1].split(',', 1)
                    return complex(float(re), float(im))
                except Exception: pass
            if 'i' in s and 'j' not in s:
                try: return complex(s.replace('i','j'))
                except Exception: pass
            try: return complex(s.replace(' ', ''))
            except Exception: pass
        raise ValueError(f'Cannot parse complex from: {x!r}')

    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return [0j] * expect_len

    if isinstance(cell, (list, tuple, np.ndarray)):
        arr = list(cell)
    elif isinstance(cell, str):
        s = cell.strip()
        try:
            val = ast.literal_eval(s)
            arr = list(val) if isinstance(val, (list, tuple, np.ndarray)) else s.replace('[','').replace(']','').split()
        except Exception:
            if ',' in s:
                arr = [x for x in s.replace('[','').replace(']','').split(',') if x.strip()!='']
            else:
                try: arr = [to_complex(s)] * expect_len
                except Exception: arr = [0j] * expect_len
    else:
        try: arr = [to_complex(cell)] * expect_len
        except Exception: arr = [0j] * expect_len

    out = []
    for x in arr:
        try: out.append(to_complex(x))
        except Exception: out.append(0j)
    return out[:expect_len] if len(out)>=expect_len else out + [0j]*(expect_len-len(out))

def assemble_H_per_sc(row, n_sc=N_SC_DEFAULT):
    H = np.zeros((n_sc, NR, NT), dtype=complex)
    for r in range(NR):
        for c in range(NT):
            col = f'csi_matrix_r{r}_c{c}'
            vec = robust_parse_csi_cell(row.get(col, None), expect_len=n_sc)
            for k in range(n_sc):
                H[k, r, c] = vec[k]
    return H

# --------- SINR: non-TxBF（关闭）---------
def compute_per_sc_sinr_non_txbf(H, noise_floor_dbm,
                                 tx_power_dbm_per_tx=TX_POWER_DBM_PER_TX,
                                 b_tot_hz=B_TOT_HZ, b_meas_hz=B_MEAS_HZ):
    H = np.asarray(H); n_sc, nr, nt = H.shape
    assert nr == NR and nt == NT, "H 维度与 NR/NT 不匹配"
    p_tx_total = dbm_to_watt(tx_power_dbm_per_tx) * float(nt)        # W
    p_sig_per_sc = p_tx_total / float(n_sc)                           # W
    p_noise_meas = dbm_to_watt(float(noise_floor_dbm))                # W (在 B_meas_hz)
    n0 = p_noise_meas / float(b_meas_hz)                              # W/Hz
    delta_f = float(b_tot_hz) / float(n_sc)
    p_noise_per_sc = max(n0 * delta_f, 1e-18)                         # W
    fro2 = np.sum(np.abs(H)**2, axis=(1,2))                           # (n_sc,)
    sinr = (p_sig_per_sc * fro2) / p_noise_per_sc
    return sinr  # 线性

# --------- SINR: TxBF（SVD 预编码，逐层+汇总）---------
def compute_per_sc_sinr_txbf_svd(H, noise_floor_dbm,
                                 tx_power_dbm_per_tx=TX_POWER_DBM_PER_TX,
                                 b_tot_hz=B_TOT_HZ, b_meas_hz=B_MEAS_HZ,
                                 n_layers=None):
    H = np.asarray(H); n_sc, nr, nt = H.shape
    if n_layers is None: n_layers = min(nr, nt)
    p_tx_total = dbm_to_watt(tx_power_dbm_per_tx) * float(nt)        # W（所有天线合计）
    p_sig_per_sc = p_tx_total / float(n_sc)                           # 每子载波总功率
    p_sig_per_sc_layer = p_sig_per_sc / float(n_layers)               # 均匀分配到每层
    p_noise_meas = dbm_to_watt(float(noise_floor_dbm))
    n0 = p_noise_meas / float(b_meas_hz)
    delta_f = float(b_tot_hz) / float(n_sc)
    p_noise_per_sc = max(n0 * delta_f, 1e-18)

    sinr_layers_list = []   # list of (n_layers,) for each sc
    sinr_sum = np.zeros(n_sc, dtype=float)
    for i in range(n_sc):
        # H_i = U diag(sigma) V^H
        try:
            sigmas = np.linalg.svd(H[i], compute_uv=False)
        except np.linalg.LinAlgError:
            sigmas = np.zeros(min(nr, nt), dtype=float)
        if sigmas.size < n_layers:
            sigmas = np.pad(sigmas, (0, n_layers - sigmas.size))
        else:
            sigmas = sigmas[:n_layers]
        sigma2 = sigmas**2  # 增益

        # per-layer SINR（线性）
        sinr_layers = (p_sig_per_sc_layer * sigma2) / p_noise_per_sc
        sinr_layers_list.append(sinr_layers)

        # 汇总版（把各层信号功率求和 / 噪声功率）
        sinr_sum[i] = (p_sig_per_sc * (sigma2.sum()/float(n_layers))) / p_noise_per_sc

    sinr_layers_arr = np.vstack(sinr_layers_list) if len(sinr_layers_list)>0 else np.empty((0,n_layers))
    return sinr_layers_arr, sinr_sum  # (n_sc, L), (n_sc,)

def stats_from_array(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0 or np.all(~np.isfinite(x)):
        return dict(mean=np.nan, min=np.nan, p05=np.nan, median=np.nan, p95=np.nan, max=np.nan)
    return dict(
        mean=float(np.nanmean(x)),
        min=float(np.nanmin(x)),
        p05=float(np.nanpercentile(x, 5)),
        median=float(np.nanmedian(x)),
        p95=float(np.nanpercentile(x, 95)),
        max=float(np.nanmax(x)),
    )

def serialize_complex_array(arr):
    return json.dumps([f"{z.real:+.10e}{z.imag:+.10e}j" for z in arr], ensure_ascii=False)

def serialize_float_array(arr):
    return json.dumps([float(v) for v in np.asarray(arr).ravel().tolist()], ensure_ascii=False)

def serialize_2d_float(arr2d):
    arr2d = np.asarray(arr2d, dtype=float)
    return json.dumps([[float(v) for v in row] for row in arr2d.tolist()], ensure_ascii=False)

# ====== 主流程 ======
def process_excel_files_to_csv():
    train_rows_non, train_rows_txbf = [], []
    valid_rows_non, valid_rows_txbf = [], []

    print('开始处理数据，根目录:', root)
    source_dirs = [d for d in root.iterdir() if d.is_dir()]
    print('找到', len(source_dirs), '个来源文件夹')

    all_files_to_process = []
    for source_dir in source_dirs:
        terminal_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
        for terminal_dir in terminal_dirs:
            terminal_id = terminal_dir.name
            train_dir = terminal_dir / 'train'
            if train_dir.exists():
                for f in train_dir.glob('*.xlsx'):
                    all_files_to_process.append((f, 'train', terminal_id))
            valid_dir = terminal_dir / 'valid'
            if valid_dir.exists():
                for f in valid_dir.glob('*.xlsx'):
                    all_files_to_process.append((f, 'valid', terminal_id))

    print('总共需要处理', len(all_files_to_process), '个Excel文件')

    for excel_file, split_type, terminal_id in tqdm(all_files_to_process, desc='处理文件进度'):
        try:
            df = pd.read_excel(excel_file)
            missing = [col for col in CSI_COLS if col not in df.columns]
            if missing:
                print(f'[警告] {excel_file} 缺少列: {missing}，跳过该文件')
                continue

            out_records = []
            for _, row in df.iterrows():
                H = assemble_H_per_sc(row, n_sc=N_SC_DEFAULT)  # (122, 2, 4)

                noise_dbm = row.get('noise_floor', np.nan)
                try: noise_dbm = float(noise_dbm)
                except Exception: noise_dbm = np.nan

                if np.isnan(noise_dbm):
                    sinr_non = np.full(N_SC_DEFAULT, np.nan, dtype=float)
                    sinr_txbf_layers = np.full((N_SC_DEFAULT, min(NR,NT)), np.nan, dtype=float)
                    sinr_txbf_sum = np.full(N_SC_DEFAULT, np.nan, dtype=float)
                else:
                    sinr_non = compute_per_sc_sinr_non_txbf(
                        H, noise_dbm,
                        tx_power_dbm_per_tx=TX_POWER_DBM_PER_TX,
                        b_tot_hz=B_TOT_HZ, b_meas_hz=B_MEAS_HZ
                    )
                    sinr_txbf_layers, sinr_txbf_sum = compute_per_sc_sinr_txbf_svd(
                        H, noise_dbm,
                        tx_power_dbm_per_tx=TX_POWER_DBM_PER_TX,
                        b_tot_hz=B_TOT_HZ, b_meas_hz=B_MEAS_HZ,
                        n_layers=min(NR, NT)
                    )

                st_non  = stats_from_array(sinr_non)
                st_tsum = stats_from_array(sinr_txbf_sum)

                H_flat = H.reshape(-1)
                H_real = H.real.reshape(-1)
                H_imag = H.imag.reshape(-1)

                rec = dict(row)
                rec['terminal_id'] = terminal_id
                rec['source_file'] = str(excel_file)

                # 关闭 TxBF 的 per-SC SINR（线性）
                rec['sinr_per_sc_non'] = serialize_float_array(sinr_non)
                rec['sinr_non_mean']   = st_non['mean']
                rec['sinr_non_min']    = st_non['min']
                rec['sinr_non_p05']    = st_non['p05']
                rec['sinr_non_median'] = st_non['median']
                rec['sinr_non_p95']    = st_non['p95']
                rec['sinr_non_max']    = st_non['max']

                # 开启 TxBF 的 per-SC SINR：逐层 与 汇总
                rec['sinr_per_sc_txbf_layers'] = serialize_2d_float(sinr_txbf_layers)  # (n_sc, L)
                rec['sinr_per_sc_txbf_sum']    = serialize_float_array(sinr_txbf_sum)   # (n_sc,)
                rec['sinr_txbf_sum_mean']   = st_tsum['mean']
                rec['sinr_txbf_sum_min']    = st_tsum['min']
                rec['sinr_txbf_sum_p05']    = st_tsum['p05']
                rec['sinr_txbf_sum_median'] = st_tsum['median']
                rec['sinr_txbf_sum_p95']    = st_tsum['p95']
                rec['sinr_txbf_sum_max']    = st_tsum['max']

                # 合并后的 H（复数/实部/虚部）
                rec['H_merged_complex'] = serialize_complex_array(H_flat)
                rec['H_real_flat']      = serialize_float_array(H_real)
                rec['H_imag_flat']      = serialize_float_array(H_imag)

                out_records.append(rec)

            if not out_records:
                continue

            out_df = pd.DataFrame(out_records)
            bf_col = 'beamforming_en'
            if bf_col not in out_df.columns:
                out_df[bf_col] = 0
            out_df[bf_col] = pd.to_numeric(out_df[bf_col], errors='coerce').fillna(0).astype(int)
            is_txbf = (out_df[bf_col] == 1)

            # 按行拆成两份
            df_txbf = out_df[out_df[bf_col] == 1].copy()
            df_non  = out_df[out_df[bf_col] == 0].copy()

            # 列隔离：各自只保留自己的 SINR 相关列（其他原始字段保留）
            all_cols = set(out_df.columns)
            to_drop_in_non  = (TXBF_KEEP & all_cols)
            to_drop_in_txbf = (NON_KEEP  & all_cols)

            if to_drop_in_non:
                df_non.drop(columns=list(to_drop_in_non), inplace=True)
            if to_drop_in_txbf:
                df_txbf.drop(columns=list(to_drop_in_txbf), inplace=True)

            # 训练 / 验证分别追加
            if split_type == 'train':
                if len(df_txbf): train_rows_txbf.append(df_txbf)
                if len(df_non):  train_rows_non.append(df_non)
            else:
                if len(df_txbf): valid_rows_txbf.append(df_txbf)
                if len(df_non):  valid_rows_non.append(df_non)

        except Exception as e:
            print('错误处理文件', excel_file, ':', e)

    def _concat_safe(lst):
        lst = [x for x in lst if isinstance(x, pd.DataFrame) and len(x)]
        return pd.concat(lst, ignore_index=True) if lst else pd.DataFrame()

    train_non = _concat_safe(train_rows_non)
    train_txbf = _concat_safe(train_rows_txbf)
    valid_non = _concat_safe(valid_rows_non)
    valid_txbf = _concat_safe(valid_rows_txbf)

    train_non_csv = root / 'train_all_non_txbf.csv'
    train_txbf_csv = root / 'train_all_txbf.csv'
    valid_non_csv = root / 'valid_all_non_txbf.csv'
    valid_txbf_csv = root / 'valid_all_txbf.csv'
    combined_csv   = root / 'combined_all.csv'

    if len(train_non):  train_non.to_csv(train_non_csv, index=False)
    if len(train_txbf): train_txbf.to_csv(train_txbf_csv, index=False)
    if len(valid_non):  valid_non.to_csv(valid_non_csv, index=False)
    if len(valid_txbf): valid_txbf.to_csv(valid_txbf_csv, index=False)

    combined_all = _concat_safe([train_non, train_txbf, valid_non, valid_txbf])
    if len(combined_all): combined_all.to_csv(combined_csv, index=False)

    print('数据处理完成！')
    print('训练-非TxBF 行数:', len(train_non))
    print('训练- TxBF 行数:', len(train_txbf))
    print('验证-非TxBF 行数:', len(valid_non))
    print('验证- TxBF 行数:', len(valid_txbf))
    print('总数据行数:', len(combined_all))

if __name__ == "__main__":
    process_excel_files_to_csv()

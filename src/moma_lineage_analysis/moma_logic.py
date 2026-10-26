import pandas as pd
import numpy as np
from typing import Optional, Iterable, Tuple, Union
from warnings import warn

# Version 2.3 - Root divisions, exact halving, validated units and cycle fits
LOGIC_VERSION = 2.3

def get_distY_mat(cells_t1: np.ndarray, cells_t2: np.ndarray) -> np.ndarray:
    return np.abs(cells_t1[:, np.newaxis] - cells_t2)

def get_delta_length_mat(cells_t1: np.ndarray, cells_t2: np.ndarray):
    return cells_t2 - cells_t1[:, np.newaxis]

def tracing_lineage(cells_index, cells_ancestor_index, end_cell_index: Union[int, Iterable, None] = None):
    if isinstance(cells_index, pd.Series): cells_index = cells_index.values.astype(int)
    if isinstance(cells_ancestor_index, pd.Series): cells_ancestor_index = cells_ancestor_index.values.astype(int)

    if end_cell_index is None:
        end_cell_index = cells_index[~np.isnan(cells_index)]
    elif isinstance(end_cell_index, int):
        end_cell_index = [end_cell_index]

    tracing_dict = {}
    for cell_index in end_cell_index:
        lineage_length = 0
        division_count = 0
        current_fov_cell_index = cell_index
        tracing_lineage_list = [cell_index]
        division_index_list = []

        while True:
            mask = cells_index == current_fov_cell_index
            if not np.any(mask): break
            ancestor_index = cells_ancestor_index[mask][0]
            # Roots use a self-edge, which is not a biological child. Record
            # their real divisions before stopping the backward walk.
            children = ((cells_ancestor_index == current_fov_cell_index)
                        & (cells_index != current_fov_cell_index))
            if children.sum() > 1:
                division_count += 1
                division_index_list.append(current_fov_cell_index)
            if ancestor_index == current_fov_cell_index: break

            lineage_length += 1
            current_fov_cell_index = ancestor_index
            tracing_lineage_list.append(current_fov_cell_index)

        tracing_dict[cell_index] = (lineage_length, division_count, tracing_lineage_list, division_index_list)
    return tracing_dict

def get_cell_lineage(time_order, time_points, fov_cell_index, cell_index, cells_length, division_threshold=0.4):
    lineage = np.copy(fov_cell_index)
    for time_i in range(len(time_order) - 1):
        t0, t1 = time_order[time_i], time_order[time_i + 1]
        m0, m1 = time_points == t0, time_points == t1
        
        c0_idx, c1_idx = cell_index[m0], cell_index[m1]
        c0_len, c1_len = cells_length[m0], cells_length[m1]
        lin_t1 = lineage[m1]
        c0_fov = fov_cell_index[m0]

        j_start = 0
        j_max, i_max = len(c1_idx)-1, len(c0_idx)-1
        i_sort, j_sort = np.argsort(c0_idx), np.argsort(c1_idx)

        for i in range(min(len(c0_idx), len(c1_idx))):
            j = j_start
            if j + 1 <= j_max:
                l1_0, l1_1 = c1_len[j_sort[j]], c1_len[j_sort[j+1]]
                sum_l1 = l1_0 + l1_1 
                l0 = c0_len[i_sort[i]]
                if (l0 * division_threshold > (sum_l1 - l0) >= 0 and (l0 * 0.9 > l1_0)):
                    lin_t1[j_sort[j]] = c0_fov[i_sort[i]]
                    lin_t1[j_sort[j+1]] = c0_fov[i_sort[i]]
                    j_start = j + 2
                else:
                    lin_t1[j_sort[j]] = c0_fov[i_sort[i]]
                    j_start = j + 1
            elif j <= j_max:
                lin_t1[j_sort[j]] = c0_fov[i_sort[i]]
                j_start = j + 1
            if j_start > j_max or i + 1 > i_max: break
        lineage[m1] = lin_t1
    return lineage

def compute_traj_stats(traj_df, div_indices):
    """Split after division mothers and fit log-length slopes in hours.

    Work on a stable time-sorted copy. Rows without finite times cannot be
    placed in a cycle; other rows remain in segments, but fitting requires
    finite, positive lengths. Invalid observations are reported, not imputed.
    """
    ordered = traj_df.copy()
    ordered['time_h'] = pd.to_numeric(ordered['time_h'], errors='coerce')
    ordered['spine_length'] = pd.to_numeric(ordered['spine_length'], errors='coerce')
    finite_time = np.isfinite(ordered['time_h'].to_numpy(dtype=float))
    lengths = ordered['spine_length'].to_numpy(dtype=float)
    valid_length = np.isfinite(lengths) & (lengths > 0)
    invalid_count = int((~(finite_time & valid_length)).sum())
    if invalid_count:
        warn(f"Ignoring {invalid_count} invalid observation(s) in trajectory fits; "
             "finite time and finite positive spine_length are required.",
             RuntimeWarning, stacklevel=2)
    ordered = ordered.loc[finite_time].sort_values('time_h', kind='stable')
    t_h = ordered['time_h'].to_numpy(dtype=float)
    division_positions = np.flatnonzero(ordered['fov_cell_index'].isin(div_indices))
    div_pts = t_h[division_positions]
    div_time = (np.array([(div_pts[:-1] + div_pts[1:]) / 2, np.diff(div_pts)])
                if len(div_pts) > 1 else np.array([]))

    segs = []
    s_idx = 0
    for position in division_positions:
        e_idx = position + 1
        segs.append((t_h[s_idx:e_idx], ordered.iloc[s_idx:e_idx]))
        s_idx = e_idx
    if s_idx < len(t_h):
        segs.append((t_h[s_idx:], ordered.iloc[s_idx:]))

    gr_data = []
    for segment_index, (ts, df) in enumerate(segs):
        lengths = df['spine_length'].to_numpy(dtype=float)
        valid = np.isfinite(lengths) & (lengths > 0)
        if valid.sum() < 4:
            gr_data.append([np.nan, np.nan, np.nan])
            continue
        fit_time = ts[valid]
        if np.ptp(fit_time) == 0:
            warn(f"Cannot fit trajectory segment {segment_index}: no time variation.",
                 RuntimeWarning, stacklevel=2)
            gr_data.append([np.nan, np.nan, np.nan])
            continue
        log_length = np.log(lengths[valid])
        midpoint = (fit_time[0] + fit_time[-1]) / 2
        # A constant response has a valid zero slope but undefined R^2.
        if np.ptp(log_length) == 0:
            gr_data.append([midpoint, 0.0, np.nan])
            continue
        centered_time = fit_time - fit_time[0]
        try:
            coeffs, _, rank, _, _ = np.polyfit(centered_time, log_length, 1, full=True)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            warn(f"Cannot fit trajectory segment {segment_index}: {exc}",
                 RuntimeWarning, stacklevel=2)
            gr_data.append([np.nan, np.nan, np.nan])
            continue
        if rank < 2 or not np.isfinite(coeffs).all():
            warn(f"Cannot fit trajectory segment {segment_index}: invalid linear fit.",
                 RuntimeWarning, stacklevel=2)
            gr_data.append([np.nan, np.nan, np.nan])
            continue
        residual = log_length - np.polyval(coeffs, centered_time)
        total_ss = np.sum((log_length - log_length.mean()) ** 2)
        r2 = 1 - np.sum(residual ** 2) / total_ss
        gr_data.append([midpoint, coeffs[0], r2])

    return segs, div_time, np.array(gr_data, dtype=float).reshape(-1, 3).T

def get_one_trajectory(cells_data, division_threshold=0.4, time_unit='s', mode='most_division', start_time=None, stop_time=None, min_spine_length=0.0):
    time_scales = {'s': 3600., 'min': 60., 'h': 1., 'd': 1/24}
    if time_unit not in time_scales:
        raise ValueError(f"Unsupported time_unit {time_unit!r}; expected one of {tuple(time_scales)}")
    time_scale = time_scales[time_unit]
    cells_data = cells_data.copy()
    # Pre-division filter: drop abnormal cells (e.g. segmentation debris) whose spine
    # length is below min_spine_length BEFORE lineage linking / division finding.
    # Removing them here keeps debris from shifting the positional pairing in
    # get_cell_lineage. min_spine_length <= 0 disables the filter.
    if min_spine_length and min_spine_length > 0:
        cells_data = cells_data[cells_data['spine_length'] >= min_spine_length]
        if cells_data.empty: return None
    cells_data['time_h'] = (cells_data['time_s'] - cells_data['time_s'].min()) / time_scale
    
    if start_time is not None:
        cells_data = cells_data[cells_data['time_h'] >= start_time]
    if stop_time is not None:
        cells_data = cells_data[cells_data['time_h'] <= stop_time]
    
    if cells_data.empty: return None
    try:
        # fov_{index}_{channel}_t_{timeIndex}.tif
        t_order = sorted(cells_data['time_point'].unique(), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    except:
        # t{timeIndex}.tif
        t_order = sorted(cells_data['time_point'].unique(), key=lambda x: int(x[1:].split('.')[0]))
    lineage = get_cell_lineage(t_order, cells_data['time_point'].values, 
                               cells_data['fov_cell_index'].values, 
                               cells_data['cell_index'].values, 
                               cells_data['spine_length'].values, 
                               division_threshold)
    cells_data['ancestor_index'] = lineage
    
    ends = cells_data[~cells_data['fov_cell_index'].isin(cells_data['ancestor_index'])]['fov_cell_index']
    lin_dict = tracing_lineage(cells_data['fov_cell_index'], cells_data['ancestor_index'], ends)
    if not lin_dict: return None

    key_cell = max(lin_dict, key=lambda x: lin_dict[x][1] if mode=='most_division' else lin_dict[x][0])
    traj = (cells_data[cells_data['fov_cell_index'].isin(lin_dict[key_cell][2])]
            .sort_values('time_h', kind='stable').copy())
    div_indices = lin_dict[key_cell][-1]
    
    segs, div_time, gr_data = compute_traj_stats(traj, div_indices)
    
    return (traj, segs, div_time, gr_data, div_indices), lin_dict

def process_channel_worker(fov_name, chan_name, cells, div_thresh, time_unit, mode, min_duration, start_time=None, stop_time=None, min_spine_length=0.0):
    ret = get_one_trajectory(cells, division_threshold=div_thresh,
                           time_unit=time_unit, mode=mode,
                           start_time=start_time, stop_time=stop_time,
                           min_spine_length=min_spine_length)
    if ret:
        (traj, segs, div_t, gr, div_indices), _ = ret
        if (traj['time_h'].max() - traj['time_h'].min()) >= min_duration:
            key = f"FOV_{fov_name}_CH_{chan_name}"
            return key, {'trajectory': traj, 'segments': segs, 
                        'division_time': div_t, 'growth_rate': gr,
                        'div_indices': list(div_indices),
                        'original_div_indices': list(div_indices)}
    return None

COLNAME_NOT_FLU_CHANNEL = ['time_s', 'time_point', 'channels', 'cell_index', 'spine_length', 'area', 'umppx', 'width',
                       'width_area', 'x_min', 'x_max', 'y_min', 'y_max', 'time_h', 'fov_cell_index', 'ancestor_index']

import pandas as pd
import numpy as np
from typing import Optional, Iterable, Tuple, Union
from warnings import warn

# Version 2.2 - Added min_spine_length pre-division cell filter
LOGIC_VERSION = 2.2

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
            if ancestor_index == current_fov_cell_index: break

            lineage_length += 1
            if (cells_ancestor_index == current_fov_cell_index).sum() > 1:
                division_count += 1
                division_index_list.append(current_fov_cell_index)

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
                if (l0 * division_threshold > (sum_l1 - l0) > 0 and (l0 * 0.9 > l1_0)):
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
    div_pts = traj_df[traj_df['fov_cell_index'].isin(div_indices)]['time_h'].values
    div_pts.sort()
    
    div_time = np.array([(div_pts[:-1] + div_pts[1:]) / 2, np.diff(div_pts)]) if len(div_pts) > 1 else np.array([])
    
    segs = []
    t_h = traj_df['time_h'].values
    s_idx = 0
    for pt in div_pts:
        e_idx = np.searchsorted(t_h, pt) + 1
        segs.append((t_h[s_idx:e_idx], traj_df.iloc[s_idx:e_idx]))
        s_idx = e_idx
    if s_idx < len(t_h): segs.append((t_h[s_idx:], traj_df.iloc[s_idx:]))

    gr_data = []
    for ts, df in segs:
        l = df['spine_length'].values
        m = ~np.isnan(l)
        if m.sum() >= 4:
            try:
                coeffs, RSS, _, _, _ = np.polyfit(ts[m], np.log(l[m]), 1, full=True)
                r2 = 1 - (RSS[0] / np.sum((np.log(l[m]) - np.log(l[m]).mean())**2))
                gr_data.append([(ts[m][0]+ts[m][-1])/2, coeffs[0], r2])
            except: gr_data.append([np.nan, np.nan, np.nan])
        else: gr_data.append([np.nan, np.nan, np.nan])
    
    return segs, div_time, np.array(gr_data).T

def get_one_trajectory(cells_data, division_threshold=0.4, time_unit='s', mode='most_division', start_time=None, stop_time=None, min_spine_length=0.0):
    time_scale = {'s': 3600, 'h': 1., 'min': 60., 'd': 1/24}.get(time_unit, 3600)
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
    traj = cells_data[cells_data['fov_cell_index'].isin(lin_dict[key_cell][2])]
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

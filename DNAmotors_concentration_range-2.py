# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from joblib import Parallel, delayed
# change the path to the folder of DeepSPT
import os
import pickle
import natsort

# have DeepSPT in your directory with this script
try:
    from deepspt_src import *
except:
    os.sys.path.append('../..')
    from SPT.github_final.DeepSPT.deepspt_src import *


def plot_diffusion_simple(track, label_list):
    color_dict = {'0':'blue', '1':'steelblue', '2':'salmon', '3':'darkorange'}
    plt.figure()
    x,y = track[:,0], track[:,1]
    c = [colors.to_rgba(color_dict[str(label)]) for label in label_list]
    
    lines = [((x0,y0), (x1,y1)) for x0, y0, x1, y1 in zip(x[:-1], y[:-1], x[1:], y[1:])]
    
    colored_lines = LineCollection(lines, colors=c, linewidths=(2,))
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in color_dict.values()]
    diff_types = ['Norm', 'Dir', 'Conf', 'Sub']
    segl, cp, val = find_segments(label_list)
    
    # plot data
    fig, ax = plt.subplots()
    ax.add_collection(colored_lines)
    ax.autoscale_view()
    plt.xlabel('x')
    plt.ylabel('y')

    plt.scatter(track[0,0], track[0,1])
    plt.annotate('Class: {}'.format(val), xy=(0.02, 0.03), xycoords='axes fraction')
    plt.annotate('segment lengths: {}'.format(segl), xy=(0.02, 0.1), xycoords='axes fraction')
    #plt.legend(markers, diff_types, numpoints=1, bbox_to_anchor=(1.33, 1.04))

    plt.show()


def translate_label(label):
    if label==0:
        return 'No virus'
    elif label==1:
        return 'Virus'
    elif label==2:
        return 'Persistently restricted'

def plot_diffusion(track, label_list, name='',savename='', changepoint=0,
                   num_change_points=0, type_of_temporal_behavior='Free'):
    color_dict = {'0':'blue', '1':'steelblue', '2':'salmon', '3':'darkorange'}
    plt.figure()
    x,y = track[:,0], track[:,1]
    c = [colors.to_rgba(color_dict[str(label)]) for label in label_list]
    
    lines = [((x0,y0), (x1,y1)) for x0, y0, x1, y1 in zip(x[:-1], y[:-1], x[1:], y[1:])]
    
    colored_lines = LineCollection(lines, colors=c, linewidths=(2,))
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in color_dict.values()]
    diff_types = ['Norm', 'Dir', 'Conf', 'Sub']
    # plot data
    fig, ax = plt.subplots()
    ax.add_collection(colored_lines)
    ax.autoscale_view()
    plt.xlabel('x')
    plt.ylabel('y')
    #plt.legend(markers, diff_types, numpoints=1, bbox_to_anchor=(1.33, 1.04))
    plt.title(name, size=20)
    plt.annotate('Change point: {}'.format(changepoint[1:-1]), xy=(0.02, 0.03), xycoords='axes fraction')
    plt.annotate('Class: {}'.format(type_of_temporal_behavior), xy=(0.02, 0.1), xycoords='axes fraction')
    if len(savename)>0:
        plt.savefig(savename+'.pdf', dpi=300, 
                    pad_inches=.1, bbox_inches='tight')
    plt.show()


def timepoint_confidence_plot(pred, savename=''):
    colors_dict = {'Normal':'blue',
               'Directed':'red',
               'Confined':'green',
               'Sub':'darkorange'}
    fig, ax = plt.subplots(figsize=(6.5,2))
    ax.stackplot(list(range(len(pred[0]))), pred, colors=list(colors_dict.values()), 
                labels=colors_dict.keys())
    plt.xlabel('Timestamp')
    plt.ylabel('Confidence')
    if len(savename)>0:
            plt.savefig(savename+'.pdf', dpi=300, 
                        pad_inches=.1, bbox_inches='tight')
    plt.show()

def timepoint_confidence_plot_2state(pred, savename=''):
    colors_dict = {'0':'steelblue',
                   '1':'salmon'}
    fig, ax = plt.subplots(figsize=(6.5,2))
    ax.stackplot(list(range(len(pred[0]))), pred, colors=list(colors_dict.values()), 
                labels=colors_dict.keys())
    plt.xlabel('Timestamp')
    plt.ylabel('Confidence')
    if len(savename)>0:
            plt.savefig(savename+'.pdf', dpi=300, 
                        pad_inches=.1, bbox_inches='tight')
    plt.show()

def prep_csv_tracks(df, xname='x', yname='y', timename='A', identifiername='particle', center=False):
    df_by_particle = dict(tuple(df.groupby(identifiername)))
    X = [np.vstack(val[[xname, yname]].values).astype(float) for val in df_by_particle.values()]
    T = [np.vstack(val[timename].values).astype(float) for val in df_by_particle.values()]
    if center:
        X = [x-x[0] for x in X]
    return X, T


def create_tracks(df, identifiername='TRACK_ID', timename='FRAME', 
                  xname='POSITION_X', yname='POSITION_Y', center=False):
    X = df.sort_values(by=[identifiername, timename]).reset_index(drop=True)
    X, T = prep_csv_tracks(X, xname=xname, yname=yname, 
                    identifiername=identifiername,
                    timename=timename,
                    center=center)
    X = np.array(X, dtype=object)
    T = np.array(T, dtype=object)
    return X, T


def load_tracks_from_csv(path, f, min_length=20,
                         identifiername='particle',
                         timename='frame',
                         xname='x',
                         yname='y'):
    df_spots = pd.read_csv(path+f).fillna('NaN')
    X, T = create_tracks(df_spots, identifiername=identifiername, 
                         timename=timename,  xname=xname, 
                         yname=yname, center=False)
    filter_length = [len(x) > min_length for x in X]
    tracks = X.copy()[filter_length]
    return tracks, T[filter_length]


def flatten_list_of_lists(l):
    return [item for sublist in l for item in sublist]

def parallel_create_fingerprint_track(track, fp_datapath, 
                                      hmm_filename, dim, dt):
        FP = create_fingerprint_track(track, 
                                      fp_datapath, 
                                      hmm_filename, 
                                      dim, dt, 'Normal')
        return FP


def get_inst_msd(tracks, dim, dt):
    def SquareDist(x0, x1, y0, y1, z0, z1):
        """Computes the squared distance between the two points (x0,y0) and (y1,y1)

        Returns
        -------
        float
            squared distance between the two input points

        """
        return (x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2
    
    inst_msds_all = []
    for i, t in enumerate(tracks):
        try:
            x, y, z = t[:,0], t[:,1], t[:,2]
        except:
            x, y = t[:,0], t[:,1]
            z = np.zeros(len(x))
        lag = 1
        inst_msd = np.mean(
                    [
                        SquareDist(x[j], x[j + lag], y[j], y[j + lag], z[j], z[j + lag])
                        for j in range(len(x) - lag)
                    ]
                )/(2*dim*dt)
        inst_msds_all.append(inst_msd)
    inst_msds_all = np.array(inst_msds_all)
    return inst_msds_all


def net_displacement(tracks):
    net_displacements = []
    for i, t in enumerate(tracks):
        try:
            x, y, z = t[:,0], t[:,1], t[:,2]
        except:
            x, y = t[:,0], t[:,1]
            z = np.zeros(len(x))
        net_displacements.append(np.sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2 + (z[-1]-z[0])**2))
    return np.array(net_displacements)


min_length = 360
data_path = 'DNAmotorsAtlanta/motors with different concentration of virus/'

df_tracks_dict = {}
df_timepoints_dict = {}
exp_type_list = []
exp_type_list_long = []
num_tracks = {}
for path in os.listdir(data_path):
    if '.csv' not in path or 'cond' not in path:
        continue
    exp_type_list.append(path.split('100pPEGV4')[0])
    exp_type_list_long.append(path)
    _tracks, _timepoints = load_tracks_from_csv(
        data_path, path, min_length=min_length,
        identifiername='Trajectory', timename='Frame')
    num_tracks[path] = len(_tracks)
    # assert _timepoints always grows
    for i in range(len(_timepoints)):
        assert np.all(np.diff(_timepoints[i])>0)
    df_tracks_dict[path] = _tracks*0.55
    df_timepoints_dict[path] = _timepoints
exp_type_list = np.array(exp_type_list)
exp_type_list_long = np.array(exp_type_list_long)
exp_type_list, num_tracks


testcsv = data_path + 'test.csv'
df = pd.read_csv(testcsv)
c_columns = ['102', '103', '104', '105', '106', '107', '0']
print([len(df[k].values[~np.isnan(df[k].values)]) for k in c_columns])


keys_all = np.array(list(df_tracks_dict.keys()))
for uniq_paths in np.unique(exp_type_list):
    keys_all_exp = keys_all[exp_type_list==uniq_paths]
    print(keys_all_exp)

    colors_list = ['C2', 'C0']
    plt.figure(figsize=(5,5))
    for bi in range(len(keys_all_exp)):
        tracks_to_plot = df_tracks_dict[keys_all_exp[bi]]
        for i in range(len(tracks_to_plot)):
            if i == 0:
                plt.plot(tracks_to_plot[i][:,0]-tracks_to_plot[i][0,0],
                    tracks_to_plot[i][:,1]-tracks_to_plot[i][0,1],
                    color=colors_list[bi], alpha=0.5, label=uniq_paths+' bio'+str(bi+1))
            else:
                plt.plot(tracks_to_plot[i][:,0]-tracks_to_plot[i][0,0],
                    tracks_to_plot[i][:,1]-tracks_to_plot[i][0,1],
                    color=colors_list[bi], alpha=0.5)
            
    plt.xlim(-200,20)
    plt.ylim(-20,200)
    plt.xlabel('x um')
    plt.ylabel('y um')
    plt.title(uniq_paths+' N: {}'.format(np.sum([num_tracks[k] for k in keys_all_exp])))
    plt.legend()
    plt.savefig(data_path+'figures/Medusa_plot{}_min_length{}.png'.format(uniq_paths,min_length), dpi=300,
                pad_inches=.1, bbox_inches='tight')


# %%

np.random.seed(42)

plt.figure(figsize=(5,5))
keys_all = np.array(list(df_tracks_dict.keys()))
for uniq_paths in np.unique(exp_type_list):
    keys_all_exp = keys_all[exp_type_list==uniq_paths]

    colors_list = ['k', 'k']
    for bi in range(len(keys_all_exp)):
        print(keys_all_exp[bi])
        if keys_all_exp[bi] == '0.004fM BA1 100pPEGV4100sPEGV4 breath cond.csv':
            continue
        print(keys_all_exp[bi])
        tracks_to_plot = df_tracks_dict[keys_all_exp[bi]]
        for i in range(0,len(tracks_to_plot),2):
            # generate point in a sphere of radius 400
            x = np.random.uniform(-350, 400)
            y = np.random.uniform(-350, 400)
            while x**2 + y**2 > 400**2:
                x = np.random.uniform(-350, 400)
                y = np.random.uniform(-350, 400)
            plt.plot(tracks_to_plot[i][:,0]-tracks_to_plot[i][0,0]+x,
                tracks_to_plot[i][:,1]-tracks_to_plot[i][0,1]+y,
                color=colors_list[bi])
            
plt.xlim(-500,500)
plt.ylim(-500,500)
plt.xlabel('x um')
plt.ylabel('y um')
plt.legend()
plt.savefig(data_path+'final_figures/schematic_tracks.pdf', dpi=300,
            pad_inches=.1, bbox_inches='tight')


# %%

max_length = np.max([np.max([len(t) for t in df_tracks_dict[k]]) for k in df_tracks_dict.keys()])
print(max_length)

dim = 2 # dimension of tracks
# define dataset and method that model was trained on to find the model
if dim == 3:
    datasets = ['SimDiff_dim3_ntraces300000_Drandom0.0001-0.5_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16']
    modeldir = '36'
if dim == 2:
    datasets = ['SimDiff_dim2_ntraces300000_Drandom0.0001-0.5_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16']
    modeldir = '3'
methods = ['XYZ_SL_DP']

# find the model
dir_name = ''
modelpath = 'github_final/DeepSPT/mlruns/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

path = modelpath+'{}'.format(modeldir)
best_models_sorted = find_models_for_from_path(path)
print(best_models_sorted) # ordered as found

# model/data params
min_max_len = 601 # min and max length of tracks model used during training
X_padtoken = 0 # pre-pad tracks to get them equal length
y_padtoken = 10 # pad y for same reason
batch_size = 32 # batch size for evaluation
use_temperature = True # use temperature scaling for softmax

# save paths

rerun_segmentaion = False # Set false to load previous results
selected_features = [0, 1, 4, 6,
                     10, 11, 12, 13, 14, 17,
                     18, 19, 21, 23, 24, 25, 26,
                     27, 28, 29, 30, 31, 32,
                     33, 34, 35, 36, 37, 38, 39]
print(len(selected_features))
fp_datapath = 'github_final/DeepSPT/_Data/Simulated_diffusion_tracks/'
hmm_filename = 'simulated2D_HMM.json'
dim = 2
dt = 5
length = []
if not os.path.exists(data_path+'analytics/FP_dict_conc_range_min_length{}.pkl'.format(min_length)):
    FP_dict = {}
    ensemble_pred_dict = {}
    ensemble_score_dict = {}
    for path in os.listdir(data_path):
        if '.csv' not in path or 'cond' not in path:
            continue
        tracks_to_compute = df_tracks_dict[path]
        results = Parallel(n_jobs=10)(
            delayed(parallel_create_fingerprint_track)(
                track, fp_datapath, hmm_filename, dim, dt)
                for track in tracks_to_compute)
        FP_= np.vstack([r for r in results])

        X = [x-x[0] for x in tracks_to_compute]
        features = ['XYZ', 'SL', 'DP']
        X_to_eval = add_features(X, features)
        y_to_eval = [np.ones(len(x))*0.5 for x in X_to_eval]

        savename_score = data_path+'analytics/ensemble_score_{}_min_length{}.pkl'.format(path.split('.csv')[0], min_length)
        savename_pred = data_path+'analytics/ensemble_pred_{}_min_length{}.pkl'.format(path.split('.csv')[0], min_length)
        ensemble_score, ensemble_pred = run_temporalsegmentation(
                                best_models_sorted, 
                                X_to_eval, y_to_eval,
                                use_mlflow=False,  
                                dir_name=dir_name, 
                                device=device, 
                                dim=dim, 
                                min_max_len=min_max_len, 
                                X_padtoken=X_padtoken, 
                                y_padtoken=y_padtoken,
                                batch_size=batch_size,
                                rerun_segmentaion=rerun_segmentaion,
                                savename_score=savename_score,
                                savename_pred=savename_pred,
                                use_temperature=use_temperature)
        
        from copy import deepcopy
        two_state_emsemble_pred = deepcopy(ensemble_pred)
        for i in range(len(two_state_emsemble_pred)):
            two_state_emsemble_pred[i][two_state_emsemble_pred[i]==3] = 2
            two_state_emsemble_pred[i][two_state_emsemble_pred[i]==0] = 1

        # sum row 2 and 3 while keeping 0 and 1
        two_state_ensemble_score = []
        for i in range(len(ensemble_score)):
            tmp = np.zeros((2, ensemble_score[i].shape[1]))
            tmp[0,:] = ensemble_score[i][0,:]+\
                    ensemble_score[i][1,:]
            tmp[1,:] = ensemble_score[i][2,:]+\
                    ensemble_score[i][3,:]
            two_state_ensemble_score.append(tmp)

        new_feature1, new_feature2, new_feature3,\
            new_feature4, new_feature5, new_feature6 = gen_temporal_features(two_state_emsemble_pred)
        perc_ND, perc_DM, perc_CD,\
        perc_SD, num_cp = get_perc_per_diff(two_state_emsemble_pred)
        
        inst_D_ = get_inst_msd(tracks_to_compute, dim, dt)  
        net_displacements_ = net_displacement(tracks_to_compute)
        
        list_features_to_add = [new_feature1, new_feature2, new_feature3,
                                new_feature5, perc_DM, perc_CD, num_cp,
                                inst_D_, net_displacements_]
        for i in range(len(list_features_to_add)):
            FP_ = np.hstack([FP_, list_features_to_add[i].reshape(-1,1)])

        FP_dict[path] = FP_[:,selected_features]
        ensemble_pred_dict[path] = two_state_emsemble_pred
        ensemble_score_dict[path] = two_state_ensemble_score

    pickle.dump(FP_dict, 
                open(data_path+'analytics/FP_dict_conc_range_min_length{}.pkl'.format(min_length), 'wb'))
    pickle.dump(ensemble_pred_dict,
                open(data_path+'analytics/ensemble_pred_dict_conc_range_min_length{}_.pkl'.format(min_length), 'wb'))
    pickle.dump(ensemble_score_dict,
                open(data_path+'analytics/ensemble_score_dict_conc_range_min_length{}_.pkl'.format(min_length), 'wb'))

else:
    print('min_length {} already computed'.format(min_length))
    FP_dict = pickle.load(open(data_path+'analytics/FP_dict_conc_range_min_length{}.pkl'.format(min_length), 'rb'))
    ensemble_pred_dict = pickle.load(open(data_path+'analytics/ensemble_pred_dict_conc_range_min_length{}_.pkl'.format(min_length), 'rb'))
    ensemble_score_dict = pickle.load(open(data_path+'analytics/ensemble_score_dict_conc_range_min_length{}_.pkl'.format(min_length), 'rb'))

# %%
print(natsort.natsorted(np.unique(exp_type_list)))
print(ensemble_pred_dict.keys())

colorlist = ['blue', 'C0', 'green', 'limegreen', 'peru', 'C1', 'darkred']
colorlist_new = {}
keys_all_list_sorted = []
keys_all_list_short_sorted = []
labellist = []
keys_all = np.array(list(df_tracks_dict.keys()))
for i, uniq_paths in enumerate(natsort.natsorted(np.unique(exp_type_list))):
    keys_all_exp = keys_all[exp_type_list==uniq_paths]
    keys_all_list_short_sorted.append(keys_all_exp[0].split('100pPEG')[0])
    for k in keys_all_exp:
        keys_all_list_sorted.append(k)
        colorlist_new[k] = colorlist[i]
        labellist.append(i)


rest_thres = 0

type_of_behavior_per_track_dict = {}
segment_length_restricted_dict = {}
segment_length_restricted_sum_dict = {}
num_times_restricted_dict = {}
go_stop_or_stop_go_all = {}
end_stop_or_go_all = {}
average_time_as_free_all = {}
for keys_all_exp in list(colorlist_new.keys()):
    
    ensemble_pred_exp = ensemble_pred_dict[keys_all_exp]
    type_of_behavior_per_track = []
    segment_length_restricted = []
    segment_length_restricted_sum = []
    num_times_restricted = []
    go_stop_or_stop_go = []
    end_stop_or_go = []
    average_time_as_free = []
    for i, tep in enumerate(ensemble_pred_exp):
        segl, cp, val = find_segments(tep)
        segment_length_restricted.append(segl[val==2]/len(tep))
        segment_length_restricted_sum.append(np.sum(segl[val==2])/len(tep))
        num_times_restricted.append(np.sum(val==2))
        average_time_as_free.append(np.sum(segl[val==1]))

        if len(val)==1:
            if val[0]==1:
                type_of_behavior_per_track.append(0) # just DM
            elif val[0]==2:
                type_of_behavior_per_track.append(1) # just restricted
        elif len(val)==2:
            if val[0]==1 and val[1]==2:
                go_stop_or_stop_go.append(0)
            elif val[0]==2 and val[1]==1:
                go_stop_or_stop_go.append(1)
            type_of_behavior_per_track.append(2) # one free, one restricted
        elif len(val)>=3:
            if val[-1]==1:
                end_stop_or_go.append(0)
            elif val[-1]==2:
                end_stop_or_go.append(1)
            type_of_behavior_per_track.append(3) # multiple switches free restricted
    type_of_behavior_per_track_dict[keys_all_exp] = type_of_behavior_per_track
    segment_length_restricted_dict[keys_all_exp] = segment_length_restricted
    segment_length_restricted_sum_dict[keys_all_exp] = segment_length_restricted_sum
    num_times_restricted_dict[keys_all_exp] = num_times_restricted
    go_stop_or_stop_go_all[keys_all_exp] = go_stop_or_stop_go
    end_stop_or_go_all[keys_all_exp] = end_stop_or_go
    average_time_as_free_all[keys_all_exp] = average_time_as_free

print(end_stop_or_go_all.keys())
# 0 is end stop 1 is stop-go, so mean reflects fraction of tracks that are stop-go
print([np.mean(end_stop_or_go_all[k]) for k in end_stop_or_go_all.keys()])

# 0 is go-stop 1 is stop-go, so mean reflects fraction of tracks that are stop-go
print([np.mean(go_stop_or_stop_go_all[k]) for k in go_stop_or_stop_go_all.keys()])

# %%
xs = labellist

plt.figure()
for i, k in enumerate(list(go_stop_or_stop_go_all.keys())):
    x = xs[i]
    plt.scatter(x, 100*(1-np.mean(go_stop_or_stop_go_all[k])), color=colorlist_new[k])

plt.xlabel('Concentration')
plt.ylabel('% 2-state tracks "go-stop"')
plt.xticks(range(len(keys_all_list_short_sorted)),
           keys_all_list_short_sorted, rotation=45, ha='right')
plt.ylim(0,105)

plt.figure()
for i, k in enumerate(list(end_stop_or_go_all.keys())):
    x = xs[i]
    plt.scatter(x, 100*np.mean(end_stop_or_go_all[k]), color=colorlist_new[k])

plt.xlabel('Concentration')
plt.ylabel('% multi-state tracks end "stop"')
plt.xticks(range(len(keys_all_list_short_sorted)),
           keys_all_list_short_sorted, rotation=45, ha='right')
plt.ylim(0,105)

plt.figure()
for i, k in enumerate(list(average_time_as_free_all.keys())):
    x = xs[i]
    plt.scatter(x, np.mean(average_time_as_free_all[k]), color=colorlist_new[k])


plt.xlabel('Concentration')
plt.ylabel('Frames spent as free')
plt.xticks(range(len(keys_all_list_short_sorted)),
           keys_all_list_short_sorted, rotation=45, ha='right')
plt.ylim(-10,365)

# %%

fig, ax = plt.subplots(1,1,figsize=(10,5))
x_offsets = np.linspace(-0.2, 0.2, len(keys_all))
for i, uniq_paths in enumerate(list(colorlist_new.keys())):
    print(uniq_paths, colorlist_new[uniq_paths])
    type_of_behavior_per_track = type_of_behavior_per_track_dict[uniq_paths]

    
    uniq, counts = np.unique(type_of_behavior_per_track, return_counts=True)
    if '400fM BA1' in uniq_paths:
        print(uniq, counts)

    if len(uniq)<4:
        all_val = np.arange(4)
        # check values that are not in uniq
        # make count 0 for those
        for val in all_val:
            if val not in uniq:
                uniq = np.hstack([uniq, val])
                counts = np.hstack([counts, 0])
    if 'cond.' in uniq_paths:
        ax.plot(uniq+x_offsets[i], 
                counts/np.sum(counts), 
                'o',
                color=colorlist_new[uniq_paths], 
                label=uniq_paths.split('100pP')[0],
                markersize=10)
    else:
        ax.plot(uniq+x_offsets[i], 
                counts/np.sum(counts), 
                'o',
                color=colorlist_new[uniq_paths], 
                markersize=10)
    ax.set_xticks([0,1,2,3], ['Free', 'Restricted', 'Free+\nRestricted', 'Multiple switches'])
    ax.set_ylim(-0.05,1)

plt.tight_layout()
plt.ylabel('Fraction of tracks')
plt.legend(fontsize=11, ncols=7, bbox_to_anchor=(0.97, 1.1),
           frameon=False, handletextpad=0.01,
              columnspacing=0.2,)
plt.savefig(data_path+'figures/stacked_barplot_min_length{}.pdf'.format(min_length), dpi=300,
            pad_inches=.1, bbox_inches='tight')
plt.show()



# %%
column_names =\
    ['alpha',
    'D',
    'extra',
    'pval',
    'Efficiency',
    'logEfficiency',
    'FractalDim',
    'Gaussianity',
    'Kurtosis',
    'MSDratio',
    'Trappedness',
    't0',
    't1',
    't2',
    't3',
    'lifetime',
    'len(x)',
    'Step length',
    'MSD',
    'mean dotproduct',
    'autocorr dotproduct',
    'bias dotproduct',
    'sum SL',
    'min SL',
    'max SL',
    'broadness',
    'speed',
    'CoV',
    'slowfrac',
    'fastfrac',
    'volume',
    'new_feature1', 
    'new_feature2', 
    'new_feature3',
    'new_feature5', 
    'perc_DM', 
    'perc_CD', 
    'num_cp',
    'inst_D_', 
    'net_displacements_',
    'perc_track_restricted',
    ]

print()

column_names = np.array(column_names)[selected_features]

len(column_names)

featnum = -1
bins = 20

all_FP_cond = []
keys_all = np.array(list(df_tracks_dict.keys()))
for uniq_paths in np.unique(exp_type_list):
    keys_all_exp = keys_all[exp_type_list==uniq_paths]
    print(keys_all_exp)

    colors_list = ['C2', 'C0']
    X = np.vstack([FP_dict[k] for k in keys_all_exp])

    all_FP_cond.append(X[:,featnum])

    plt.figure(figsize=(5,5))
    for bi in range(len(keys_all_exp)):
        FP_to_plot = FP_dict[keys_all_exp[bi]]
        tracks_to_plot = df_tracks_dict[keys_all_exp[bi]]
        len_tracks = np.array([len(t) for t in tracks_to_plot])

        plt.hist(FP_to_plot[:,featnum]/len_tracks, bins=bins, 
                alpha=0.5, density=True, 
                label=uniq_paths+' '+str(bi), 
                color=colors_list[bi])
        plt.title(column_names[featnum])
    #plt.vlines(np.median(X[:,featnum]), 0, .005, color='k', linestyle='-', label='Median')
    #plt.vlines(np.mean(X[:,featnum]), 0, .005, color='r', linestyle='--', label='Mean')
    #plt.vlines(np.mean(X[:,featnum])+np.std(X[:,featnum], ddof=1), 0, .005, color='k', linestyle=':', label='Mean+std')
    #plt.vlines(np.mean(X[:,featnum])+1.96*np.std(X[:,featnum], ddof=1), 0, .005, color='dimgrey', linestyle=':', label='95% CI')
    plt.legend()
    plt.show()

# %%

# plt.figure(figsize=(5,5))
# ants = np.repeat(np.arange(len(all_FP_cond)), [len(all_FP_cond[i]) for i in range(len(all_FP_cond))])
# ants_spread = ants + np.random.uniform(-0.1, 0.1, len(ants))
# plt.scatter(ants_spread, np.hstack(all_FP_cond), s=10, color='dimgrey')
# plt.errorbar(list(range(len(all_FP_cond))), 
#              [np.mean(f) for f in all_FP_cond], 
#              yerr=[np.std(f, ddof=1) for f in all_FP_cond],
#              color='k', fmt='o')
# plt.xticks(range(len(all_FP_cond)),
#             np.unique(exp_type_list), rotation=45, ha='center')

# natsort the list
    
plt.figure(figsize=(6,5))
yvalues = all_FP_cond

natsorted_list = natsort.natsorted(np.unique(exp_type_list))

yvalues_natsort = np.array([yvalues[i] for i in np.argsort(natsorted_list)])
ants = np.repeat(np.arange(len(yvalues_natsort)), [len(yvalues_natsort[i]) for i in range(len(yvalues_natsort))])
displ = np.random.uniform(-0.2, 0.2, len(ants))
ants_spread_natsort = ants + displ

plt.scatter(ants_spread_natsort, 
            np.hstack(yvalues_natsort), 
            s=10, color='lightgrey',
            alpha=0.5)
plt.errorbar(list(range(len(yvalues_natsort))), 
             [np.mean(f) for f in yvalues_natsort], 
             yerr=[np.std(f, ddof=1) for f in yvalues_natsort],
             color='k', fmt='.', capsize=5, 
             elinewidth=1, markersize=0,
             capthick=1, ecolor='k')
plt.hlines([np.mean(f) for f in yvalues_natsort],
            xmin=np.array(range(len(yvalues_natsort)))-0.2,
            xmax=np.array(range(len(yvalues_natsort)))+0.2,
            color='r', linestyle='--', linewidth=1)
plt.hlines([np.median(f) for f in yvalues_natsort],
            xmin=np.array(range(len(yvalues_natsort)))-0.2,
            xmax=np.array(range(len(yvalues_natsort)))+0.2,
            color='k', linestyle='--', linewidth=1)

plt.xticks(range(len(yvalues_natsort)),
            natsorted_list, rotation=45, ha='center')

plt.ylabel(column_names[featnum])

plt.savefig(data_path+'figures/scatter_errobar_{}_min_length{}.pdf'.format(column_names[featnum],min_length), dpi=300,
            pad_inches=.1, bbox_inches='tight')

# %%


plt.figure(figsize=(6,5))
yvalues = list(segment_length_restricted_sum_dict.values())
yvalues = [np.hstack(y) for y in yvalues]
ykeys = [a.split('100pPEG')[0] for a in
          list(segment_length_restricted_sum_dict.keys())]

ants = np.repeat(np.arange(1,len(yvalues)+1), [len(yvalues[i]) for i in range(len(yvalues))])
displ = np.random.uniform(-0.2, 0.2, len(ants))
ants_spread_natsort = ants + displ

keys_all = np.array(list(segment_length_restricted_sum_dict.keys()))
plt.scatter(ants_spread_natsort, 
            np.hstack(yvalues), 
            s=10, color='lightgrey',
            alpha=0.5)
plt.boxplot(yvalues, showfliers=False)
plt.hlines([np.mean(f) for f in yvalues],
            xmin=np.array(range(1,1+len(yvalues)))-0.2,
            xmax=np.array(range(1,1+len(yvalues)))+0.2,
            color='r', linestyle='--', linewidth=1)
plt.hlines([np.median(f) for f in yvalues],
            xmin=np.array(range(1,1+len(yvalues)))-0.2,
            xmax=np.array(range(1,1+len(yvalues)))+0.2,
            color='k', linestyle='--', linewidth=1)

plt.xticks(range(1,1+len(yvalues)),
            ykeys, rotation=45, ha='right')
plt.ylabel('Restricted fraction \n(%track)')

# %%

print([len(f) for f in yvalues_natsort])
c_columns = ['102', '103', '104', '105', '106', '107', '0']
print([len(df[k].values[~np.isnan(df[k].values)]) for k in c_columns])

for uniq_paths in np.unique(exp_type_list):
    keys_all_exp = keys_all[exp_type_list==uniq_paths]

    colors_list = ['blue', 'C0']
    X = np.vstack([FP_dict[k] for k in keys_all_exp])
    print(keys_all_exp, [len(FP_dict[k] )for k in keys_all_exp])

# %%

# 10 fold cv on X[selcted_features]

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# bookmark
conc_virus_to_train = '40fM BA1 '
conc_novirus_to_train = 'no virus '

keys_all = np.array(list(df_tracks_dict.keys()))
# find all tracks with virus ('4fM BA1 ') and without virus
keys_all_virus = [k for (i,k) in enumerate(keys_all) 
                  if conc_virus_to_train==exp_type_list[i]]
keys_all_novirus = [k for (i,k) in enumerate(keys_all) if conc_novirus_to_train==exp_type_list[i]]
print(keys_all_virus)

# find all FP with virus ('4fM BA1 ') and without virus
X_virus = np.vstack([FP_dict[k] for k in keys_all_virus])
X_novirus = np.vstack([FP_dict[k] for k in keys_all_novirus])

segment_sum_rest_novirus = np.hstack([segment_length_restricted_sum_dict[k] for k in keys_all_novirus])
segment_sum_rest_virus = np.hstack([segment_length_restricted_sum_dict[k] for k in keys_all_virus])

#X_virus = np.column_stack([X_virus, segment_sum_rest_virus.reshape(-1,1)])
#X_novirus = np.column_stack([X_novirus, segment_sum_rest_novirus.reshape(-1,1)])

print(X_virus.shape, X_novirus.shape)

netdisp_idx = -1
nondead_thres = 15
filter_nondead_motors = X_novirus[:,-1]>nondead_thres
X_novirus = X_novirus[filter_nondead_motors]

print(X_virus.shape, X_novirus.shape)

# create labels
y_virus = np.ones(len(X_virus))
y_novirus = np.zeros(len(X_novirus))

# combine to one dataset
X = np.vstack([X_virus, X_novirus])
y = np.hstack([y_virus, y_novirus])
print(np.unique(y, return_counts=True))

# standardize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
X[np.isnan(X)] = 0

classifier_ = LogisticRegression(max_iter=10000)

kf = KFold(n_splits=10, shuffle=True, random_state=42)
kf.get_n_splits(X)

accs = []
TP = []
TN = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = classifier_.fit(X_train, y_train)
    pred = clf.predict(X_test)
    TP.append(np.mean(pred[y_test==1]==1))
    TN.append(np.mean(pred[y_test==0]==0))
    acc = clf.score(X_test, y_test)
    accs.append(acc)
print(np.sum(y==1), np.sum(y==0))
print(np.mean(accs), np.std(accs, ddof=1))
print(np.mean(TP), np.std(TP, ddof=1))
print(np.mean(TN), np.std(TN, ddof=1))
print('what could have been if "dead motors" werent there', (np.mean(TP)*np.sum(y==1) + 1*np.sum(y==0))/len(y))

clf = classifier_.fit(X, y)
pred_all = clf.predict(X)
pred_infected = clf.predict(X[y==1])
pred_survivor = clf.predict(X[y==0])
np.mean(pred_infected==1), np.mean(pred_survivor==0), np.mean(pred_all==y)


pred_all_dict = {}
total_tracks = {}
suspecious_restricted_time = 50
suspecious_slow = 15
for uniq_paths in np.unique(exp_type_list):
    keys_all_exp = keys_all[exp_type_list==uniq_paths]

    for bi in range(len(keys_all_exp)):
        FP_to_pred = FP_dict[keys_all_exp[bi]]
        #FP_to_pred = np.column_stack([FP_to_pred, np.array(segment_length_restricted_sum_dict[keys_all_exp[bi]]).reshape(-1,1)])
        FP_to_pred = scaler.transform(FP_to_pred)
        FP_to_pred[np.isnan(FP_to_pred)] = 0
        pred = clf.predict(FP_to_pred)

        FP_of_pred = FP_dict[keys_all_exp[bi]]
        ensemble_of_pred = ensemble_pred_dict[keys_all_exp[bi]]
        tracks_of_pred = df_tracks_dict[keys_all_exp[bi]]
        len_tracks = np.array([len(t) for t in tracks_of_pred])
        total_tracks[keys_all_exp[bi]] = len(tracks_of_pred)

        count_sus = True
        count_slow = True
        count_sus_name = ''
        count_slow_name = ''

        # bookmark
        if count_slow:
            count_slow_name = 'counted_slow'
            for i in np.arange(len(pred))[pred==0]:
                net_disp = FP_of_pred[i,-1]
                if net_disp<suspecious_slow:
                    pred[i] = 1

        if count_sus:
            count_sus_name = 'counted_sus'
            for i in np.arange(len(pred))[pred==0]:
                segl, cp, val = find_segments(ensemble_of_pred[i])
                if np.unique(val).shape[0]>1:
                    if np.max(segl[val==2])>suspecious_restricted_time:
                        pred[i] = 2
        

        pred_all_dict[keys_all_exp[bi]] = pred
        print(keys_all_exp[bi], np.unique(pred, return_counts=True)[1]/len(pred))


pred_all_dict.keys()
print()
print('IF restricted more than {} frames, then virus!'.upper().format(suspecious_restricted_time))


type_of_temporal_behavior_dict = {}
change_points_dict = {}
num_change_points_dict = {}

keys_all = np.array(list(df_tracks_dict.keys()))
for uniq_paths in np.unique(exp_type_list):
    keys_all_exp = keys_all[exp_type_list==uniq_paths]

    colors_list = ['blue', 'C0']
    for bi in range(len(keys_all_exp)):
        ensemble_pred_ = ensemble_pred_dict[keys_all_exp[bi]]

        type_of_temporal_behavior = []
        change_points = []
        num_change_points = []
        for i, e in enumerate(ensemble_pred_):
            sl, cp, val = find_segments(e)
            change_points.append(cp)
            num_change_points.append(len(val))
            if len(val)>1:
                if val[0] == 1:
                    type_of_temporal_behavior.append('Switch start free')
                elif val[0] == 2:
                    type_of_temporal_behavior.append('Switch start restricted')
            else:
                if val[0] == 1:
                    type_of_temporal_behavior.append('Free')
                elif val[0] == 2:
                    type_of_temporal_behavior.append('Restricted')
        
        type_of_temporal_behavior_dict[keys_all_exp[bi]] = type_of_temporal_behavior
        change_points_dict[keys_all_exp[bi]] = change_points
        num_change_points_dict[keys_all_exp[bi]] = num_change_points


frac_0s_per_condition = []
frac_1s_per_condition = []
frac_2s_per_condition = []
keys_all_per_condition = []
total_tracks_per_condition = []
for uniq_paths in np.unique(exp_type_list):
    keys_all_exp = keys_all[exp_type_list==uniq_paths]

    print('keys_all_exp', keys_all_exp)
    frac_0s = []
    frac_1s = []
    frac_2s = []
    total_tracks = 0
    for bi in range(len(keys_all_exp)):
        pred_to_plot = pred_all_dict[keys_all_exp[bi]]
        tracks_to_plot = df_tracks_dict[keys_all_exp[bi]]
        total_tracks += len(tracks_to_plot)

        frac_0 = np.round(100*np.sum(pred_to_plot==0)/len(pred_to_plot), 1)
        frac_1 = np.round(100*np.sum(pred_to_plot==1)/len(pred_to_plot), 1)
        frac_2 = np.round(100*np.sum(pred_to_plot==2)/len(pred_to_plot), 1)

        frac_0s.append(100*(np.sum(pred_to_plot==0)/len(pred_to_plot)))
        frac_1s.append(100*(np.sum(pred_to_plot==1)/len(pred_to_plot)))
        frac_2s.append(100*(np.sum(pred_to_plot==2)/len(pred_to_plot)))

        # fig, ax = plt.subplots(2,3,figsize=(10,5),
        #                        gridspec_kw={'height_ratios': [2, 1]})
        # for i in range(len(tracks_to_plot)):
        #     if pred_to_plot[i]==0:
        #         ax[0,0].plot(tracks_to_plot[i][:,0]-tracks_to_plot[i][0,0],
        #             tracks_to_plot[i][:,1]-tracks_to_plot[i][0,1],
        #             color='green', alpha=0.25)
        #         ax[0,0].set_ylim(-20,200)
        #         ax[0,0].set_xlim(-200,20)
        #         ax[0,0].set_aspect('equal')
        #         ax[0,0].set_title('Pred: "No virus": {}%\nN: {}'.format(frac_0, np.sum(pred_to_plot==0)))
                
        #         ax[1,0].plot(tracks_to_plot[i][:,0]-tracks_to_plot[i][0,0],
        #             tracks_to_plot[i][:,1]-tracks_to_plot[i][0,1],
        #             color='green', alpha=0.25)
        #         ax[1,0].set_aspect('equal')

        #     elif pred_to_plot[i]==1:
        #         ax[0,1].plot(tracks_to_plot[i][:,0]-tracks_to_plot[i][0,0],
        #             tracks_to_plot[i][:,1]-tracks_to_plot[i][0,1],
        #             color='red', alpha=0.25)
        #         ax[0,1].set_ylim(-20,200)
        #         ax[0,1].set_xlim(-200,20)
        #         # equal aspect ratio
        #         ax[0,1].set_aspect('equal')
        #         ax[0,1].set_title('Pred: "Virus": {}%\nN: {}'.format(frac_1, np.sum(pred_to_plot==1)))
                
        #         ax[1,1].plot(tracks_to_plot[i][:,0]-tracks_to_plot[i][0,0],
        #             tracks_to_plot[i][:,1]-tracks_to_plot[i][0,1],
        #             color='red', alpha=0.25)
        #         ax[1,1].set_aspect('equal')

        #     elif pred_to_plot[i]==2:
        #         ax[0,2].plot(tracks_to_plot[i][:,0]-tracks_to_plot[i][0,0],
        #             tracks_to_plot[i][:,1]-tracks_to_plot[i][0,1],
        #             color='C1', alpha=0.25)
        #         ax[0,2].set_ylim(-20,200)
        #         ax[0,2].set_xlim(-200,20)
        #         ax[0,2].set_aspect('equal')
        #         ax[0,2].set_title('Suspeciously\n restricted: {}%\nN: {}'.format(frac_2, np.sum(pred_to_plot==2)))
                
        #         ax[1,2].plot(tracks_to_plot[i][:,0]-tracks_to_plot[i][0,0],
        #             tracks_to_plot[i][:,1]-tracks_to_plot[i][0,1],
        #             color='C1', alpha=0.25)
        #         ax[1,2].set_aspect('equal')

        # plt.tight_layout()
        # plt.suptitle(keys_all_exp[bi]+' N: {}'.format(len(pred_to_plot)), fontsize=16,
        #              y=1.05)
        # plt.savefig(data_path+'figures/prediction_medusaplots_{}_min_length.png'.format(keys_all_exp[bi], min_length),
        #             dpi=200, pad_inches=.1, bbox_inches='tight')

    frac_0s_per_condition.append(frac_0s)
    frac_1s_per_condition.append(frac_1s)
    frac_2s_per_condition.append(frac_2s)
    keys_all_per_condition.append(uniq_paths)
    total_tracks_per_condition.append(total_tracks)

import natsort

natsorted_list = natsort.natsorted(np.unique(exp_type_list))
print(natsorted_list)
print(natsorted_list)
print(np.argsort(natsorted_list))
argsort_natsort = np.argsort(natsorted_list)

natsorted_list = np.roll(natsorted_list, 1)
argsort_natsort = np.roll(argsort_natsort, 1)
print(argsort_natsort)

frac_1s_per_condition_natsort = np.array([frac_1s_per_condition[i] for i in argsort_natsort])
frac_2s_per_condition_natsort = np.array([frac_2s_per_condition[i] for i in argsort_natsort])
total_tracks_per_condition_natsort = np.array([total_tracks_per_condition[i] for i in argsort_natsort])

l1_all = frac_1s_per_condition_natsort
l2_all = frac_2s_per_condition_natsort

frac_1s_2s_per_condition_natsort = [np.array(l1)+np.array(l2) for l1, l2 in zip(l1_all, l2_all)]

# remove index 1 from frac_1s_2s_per_condition_natsort
frac_1s_2s_per_condition_natsort = [a for i, a in enumerate(frac_1s_2s_per_condition_natsort) if i!=1]
frac_1s_per_condition_natsort = [a for i, a in enumerate(frac_1s_per_condition_natsort) if i!=1]
frac_2s_per_condition_natsort = [a for i, a in enumerate(frac_2s_per_condition_natsort) if i!=1]
argsort_natsort = [a for i, a in enumerate(argsort_natsort) if i!=1]
natsorted_list = [a for i, a in enumerate(natsorted_list) if i!=1]


x = []
for i in range(len(frac_1s_2s_per_condition_natsort)):
    x.append(np.ones(len(frac_1s_2s_per_condition_natsort[i]))*i)

x = np.hstack(x) 

plt.figure(figsize=(8,4))
plt.plot([np.mean(n) for n in frac_1s_2s_per_condition_natsort], 
        '-', color='lightgrey')
plt.plot([np.mean(n) for n in frac_1s_2s_per_condition_natsort], 
        'o', color='grey', label='All pred.')

plt.plot([np.mean(n) for n in frac_1s_per_condition_natsort], 
        '-', alpha=0.5, color='red')
plt.plot([np.mean(n) for n in frac_1s_per_condition_natsort], 
        'o', alpha=0.75, color='red', label='Pred. Virus')

plt.plot([np.mean(n) for n in frac_2s_per_condition_natsort], 
        '-', alpha=0.5, color='C1')
plt.plot([np.mean(n) for n in frac_2s_per_condition_natsort], 
        'o', alpha=0.75, color='C1', label='Persistently restricted')

x = x + np.random.normal(0, 0.05, len(x))
plt.scatter(x, np.hstack(frac_1s_2s_per_condition_natsort), 
            s=20, color='grey')
x = x + np.random.normal(0, 0.05, len(x))
plt.scatter(x, np.hstack(frac_1s_per_condition_natsort), 
            s=20, color='red')
x = x + np.random.normal(0, 0.05, len(x))
plt.scatter(x, np.hstack(frac_2s_per_condition_natsort), 
            s=20, color='C1')

plt.xticks(range(len(frac_1s_2s_per_condition_natsort)),
         natsorted_list, rotation=45, ha='center')
plt.ylabel('Fraction of tracks')
for ti, tt in enumerate(total_tracks_per_condition_natsort):
    plt.text(ti, 102, tt, ha='center', va='bottom', fontsize=8)
plt.ylim(0,106)
# horizontal legend
plt.legend(bbox_to_anchor=(-.1, 1.2), loc='upper left', ncol=3)
plt.savefig(data_path+'final_figures/frac_pred_virus_{}_{}_min_length{}.pdf'.format(count_slow_name,count_sus_name,min_length),
            dpi=300, pad_inches=.1, bbox_inches='tight')
plt.show()
print(data_path+'final_figures/')
print('min_length', min_length)

# l1_all = frac_1s_per_condition
# l2_all = frac_2s_per_condition

# print(keys_all_per_condition)
# frac_1s_2s_per_condition = [np.array(l1)+np.array(l2) for l1, l2 in zip(l1_all, l2_all)]

# x = []
# for i in range(len(frac_1s_2s_per_condition)):
#     x.append(np.ones(len(frac_1s_2s_per_condition[i]))*i)
# x = np.hstack(x) 
# plt.figure(figsize=(5,5))
# plt.plot([np.mean(n) for n in frac_1s_2s_per_condition], '-', color='grey')
# plt.plot([np.mean(n) for n in frac_1s_2s_per_condition], 'o', color='k')

# for ti, tt in enumerate(total_tracks_per_condition_natsort):
#     plt.text(ti, 101, tt, ha='center', va='bottom', fontsize=8)

# plt.scatter(x, np.hstack(frac_1s_2s_per_condition), s=10, color='dimgrey')

# plt.xticks(range(len(frac_1s_2s_per_condition)),
#          np.unique(exp_type_list), rotation=45, ha='center')
# plt.ylabel('Fraction predicted as virus')
# plt.show()


# %%

pred_all_dict

# %%
def plot_diffusion_and_confidence(
        track, label_list, score, name='',savename='', changepoint=0,
        num_change_points=0, type_of_temporal_behavior='Free'):
    color_dict = {'0':'blue', '1':'steelblue', '2':'salmon', '3':'darkorange'}
    
    fig, ax = plt.subplots(2,1, figsize=(8,8),
                           gridspec_kw={'height_ratios': [3, 1]})
    x,y = track[:,0], track[:,1]
    c = [colors.to_rgba(color_dict[str(label)]) for label in label_list]
    
    lines = [((x0,y0), (x1,y1)) for x0, y0, x1, y1 in zip(x[:-1], y[:-1], x[1:], y[1:])]
    
    colored_lines = LineCollection(lines, colors=c, linewidths=(2,))
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in color_dict.values()]
    diff_types = ['Norm', 'Dir', 'Conf', 'Sub']
    # plot data
    ax[0].add_collection(colored_lines)
    ax[0].autoscale_view()
    ax[0].axis('equal')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    #plt.legend(markers, diff_types, numpoints=1, bbox_to_anchor=(1.33, 1.04))
    ax[0].set_title(name, size=20)
    ax[0].annotate('Change point: {}'.format(changepoint[1:-1]), xy=(0.02, 0.03), xycoords='axes fraction')
    ax[0].annotate('Class: {}'.format(type_of_temporal_behavior), xy=(0.02, 0.1), xycoords='axes fraction')
        
    colors_dict = {'0':'steelblue',
                    '1':'salmon'}
    
    ax[1].stackplot(list(range(len(score[0]))), score, colors=list(colors_dict.values()), 
                labels=colors_dict.keys())
    ax[1].set_xlabel('Timestamp')
    ax[1].set_ylabel('Confidence')

    plt.tight_layout()
    if len(savename)>0:
            plt.savefig(savename+'.pdf', dpi=300, 
                        pad_inches=.1, bbox_inches='tight')
    plt.show()

def plot_diffusion_simple_zoom(track, label_list, xmin, xmax, ymin, ymax, savename=''):
    color_dict = {'0':'blue', '1':'steelblue', '2':'salmon', '3':'darkorange'}
    plt.figure()
    x,y = track[:,0], track[:,1]
    c = [colors.to_rgba(color_dict[str(label)]) for label in label_list]
    
    lines = [((x0,y0), (x1,y1)) for x0, y0, x1, y1 in zip(x[:-1], y[:-1], x[1:], y[1:])]
    
    colored_lines = LineCollection(lines, colors=c, linewidths=(2,))
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in color_dict.values()]
    diff_types = ['Norm', 'Dir', 'Conf', 'Sub']
    segl, cp, val = find_segments(label_list)
    
    # plot data
    fig, ax = plt.subplots()
    ax.add_collection(colored_lines)
    ax.autoscale_view()
    plt.xlabel('x')
    plt.ylabel('y')

    plt.scatter(track[0,0], track[0,1])
    #plt.legend(markers, diff_types, numpoints=1, bbox_to_anchor=(1.33, 1.04))
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xticks(np.arange(xmin+0.1, xmax, .5))
    plt.yticks(np.arange(ymin+0.1, ymax, .5))
    if len(savename)>0:
        plt.savefig(savename+'.pdf', dpi=300, 
                    pad_inches=.1, bbox_inches='tight')
    plt.show()


i = 10
keys_all_to_plot = keys_all[i]
print(keys_all_to_plot)
tracks_to_plot = df_tracks_dict[keys_all[i]]
ensemble_pred_ = ensemble_pred_dict[keys_all[i]]
ensemble_score_ = ensemble_score_dict[keys_all[i]]
type_of_temporal_behavior = type_of_temporal_behavior_dict[keys_all[i]]
change_points = change_points_dict[keys_all[i]]
num_change_points = num_change_points_dict[keys_all[i]]
pred = pred_all_dict[keys_all[i]]

print(len(pred))
i = 150 #np.random.randint(len(pred)) # 38, 87
save = True

xmin, xmax = 605.9, 607.1
ymin, ymax = 563.1, 564.3

print(i, pred[i], translate_label(pred[i]))
if save:
    savename1 = data_path+'final_figures/track_conf{}_{}_min_length{}'.format(i,keys_all_to_plot,min_length)
    plot_diffusion_and_confidence(
                tracks_to_plot[i], 
                ensemble_pred_[i], 
                ensemble_score_[i],
                name=translate_label(pred[i]), changepoint=change_points[i],
                savename=savename1,
                num_change_points=num_change_points[i],
                type_of_temporal_behavior=type_of_temporal_behavior[i])
    plot_diffusion_simple_zoom(
                         tracks_to_plot[i], 
                         ensemble_pred_[i], 
                         xmin, xmax, 
                         ymin, ymax,
                         savename=savename1+'_zoom')

else:
    plot_diffusion_and_confidence(
                tracks_to_plot[i], 
                ensemble_pred_[i], 
                ensemble_score_[i],
                name=translate_label(pred[i]), changepoint=change_points[i],
                savename='',
                num_change_points=num_change_points[i],
                type_of_temporal_behavior=type_of_temporal_behavior[i])
    plot_diffusion_simple_zoom(
                         tracks_to_plot[i], 
                         ensemble_pred_[i], 
                         xmin, xmax, 
                         ymin, ymax,
                         savename='')


# %%


# %%
# pred 0 is no virus
# pred 1 is virus   
# pred 2 is persistent restricted

key_to_plot = '0.004fM BA1 100pPEGV4100sPEGV4 breath cond.csv'

cond_to_plot = pred_all_dict['0.004fM BA1 100pPEGV4100sPEGV4 breath cond.csv']
FP_to_plot = FP_dict['0.004fM BA1 100pPEGV4100sPEGV4 breath cond.csv']
tracks_to_plot = df_tracks_dict['0.004fM BA1 100pPEGV4100sPEGV4 breath cond.csv']
ensemble_to_plot= ensemble_pred_dict['0.004fM BA1 100pPEGV4100sPEGV4 breath cond.csv']
ensemble_score_to_plot= ensemble_score_dict['0.004fM BA1 100pPEGV4100sPEGV4 breath cond.csv']
change_points_to_plot = change_points_dict['0.004fM BA1 100pPEGV4100sPEGV4 breath cond.csv']
num_change_points_to_plot = num_change_points_dict['0.004fM BA1 100pPEGV4100sPEGV4 breath cond.csv']
type_of_temporal_behavior_to_plot = type_of_temporal_behavior_dict['0.004fM BA1 100pPEGV4100sPEGV4 breath cond.csv']

assert len(cond_to_plot) == len(tracks_to_plot) == len(FP_to_plot)
print(len(cond_to_plot), len(tracks_to_plot), len(FP_to_plot))


free = np.arange(len(cond_to_plot))[cond_to_plot==0]
virus = np.arange(len(cond_to_plot))[cond_to_plot==1]
restricted = np.arange(len(cond_to_plot))[cond_to_plot==2]

free_i = 0
virus_i = 17
virus_i2 = 8
restricted_i = 4
restricted_i2 = 6

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(tracks_to_plot[free[free_i]][:,0]-tracks_to_plot[free[free_i]][0,0],
            tracks_to_plot[free[free_i]][:,1]-tracks_to_plot[free[free_i]][0,1],
            color='green')
ax.set_title('Pred: "No virus"')
ax.set_aspect('equal')
ax.plot(tracks_to_plot[virus[virus_i]][:,0]-tracks_to_plot[virus[virus_i]][0,0],
            tracks_to_plot[virus[virus_i]][:,1]-tracks_to_plot[virus[virus_i]][0,1]+100,
            color='red')
ax.set_title('Pred: "Virus"')
ax.set_aspect('equal')
ax.plot(tracks_to_plot[restricted[restricted_i]][:,0]-tracks_to_plot[restricted[restricted_i]][0,0],
            tracks_to_plot[restricted[restricted_i]][:,1]-tracks_to_plot[restricted[restricted_i]][0,1]+200,
            color='blue')
ax.set_title('Pred: "Persistent restricted"')
ax.set_aspect('equal')
ax.plot(tracks_to_plot[restricted[restricted_i2]][:,0]-tracks_to_plot[restricted[restricted_i2]][0,0],
            tracks_to_plot[restricted[restricted_i2]][:,1]-tracks_to_plot[restricted[restricted_i2]][0,1]+300,
            color='blue')
ax.set_title('Pred: "Persistent restricted"')
ax.set_aspect('equal')
plt.show()


fig, ax = plt.subplots(figsize=(10,5))
ax.plot(tracks_to_plot[restricted[restricted_i]][:,0]-tracks_to_plot[restricted[restricted_i]][0,0],
            tracks_to_plot[restricted[restricted_i]][:,1]-tracks_to_plot[restricted[restricted_i]][0,1]+20,
            color='blue')
ax.set_title('Pred: "Persistent restricted"')
ax.set_aspect('equal')
ax.plot(tracks_to_plot[restricted[restricted_i2]][:,0]-tracks_to_plot[restricted[restricted_i2]][0,0],
            tracks_to_plot[restricted[restricted_i2]][:,1]-tracks_to_plot[restricted[restricted_i2]][0,1],
            color='blue')
ax.set_title('Pred: "Persistent restricted"')
ax.set_aspect('equal')
plt.show()


i = free[free_i]
savename1 = data_path+'final_figures/track_schematic{}_{}_min_length{}_free'.format(i,key_to_plot,min_length)
plot_diffusion_and_confidence(
        tracks_to_plot[i], 
        ensemble_to_plot[i],
        ensemble_score_to_plot[i],
        name=translate_label(cond_to_plot[i]), changepoint=change_points_to_plot[i],
        savename=savename1,
        num_change_points=num_change_points_to_plot[i],
        type_of_temporal_behavior=type_of_temporal_behavior_to_plot[i])


i = virus[virus_i]
savename1 = data_path+'final_figures/track_schematic{}_{}_min_length{}_virus'.format(i,key_to_plot,min_length)
plot_diffusion_and_confidence(
        tracks_to_plot[i], 
        ensemble_to_plot[i],
        ensemble_score_to_plot[i],
        name=translate_label(cond_to_plot[i]), changepoint=change_points_to_plot[i],
        savename=savename1,
        num_change_points=num_change_points_to_plot[i],
        type_of_temporal_behavior=type_of_temporal_behavior_to_plot[i])


i = restricted[restricted_i]
savename1 = data_path+'final_figures/track_schematic{}_{}_min_length{}_persistent'.format(i,key_to_plot,min_length)
plot_diffusion_and_confidence(
        tracks_to_plot[i], 
        ensemble_to_plot[i],
        ensemble_score_to_plot[i],
        name=translate_label(cond_to_plot[i]), changepoint=change_points_to_plot[i],
        savename=savename1,
        num_change_points=num_change_points_to_plot[i],
        type_of_temporal_behavior=type_of_temporal_behavior_to_plot[i])


i = restricted[restricted_i2]
savename1 = data_path+'final_figures/track_schematic{}_{}_min_length{}_persistent'.format(i,key_to_plot,min_length)
plot_diffusion_and_confidence(
        tracks_to_plot[i], 
        ensemble_to_plot[i],
        ensemble_score_to_plot[i],
        name=translate_label(cond_to_plot[i]), changepoint=change_points_to_plot[i],
        savename=savename1,
        num_change_points=num_change_points_to_plot[i],
        type_of_temporal_behavior=type_of_temporal_behavior_to_plot[i])



i = virus[virus_i2]
savename1 = data_path+'final_figures/track_schematic{}_{}_min_length{}_virus'.format(i,key_to_plot,min_length)
plot_diffusion_and_confidence(
        tracks_to_plot[i], 
        ensemble_to_plot[i],
        ensemble_score_to_plot[i],
        name=translate_label(cond_to_plot[i]), changepoint=change_points_to_plot[i],
        savename=savename1,
        num_change_points=num_change_points_to_plot[i],
        type_of_temporal_behavior=type_of_temporal_behavior_to_plot[i])


# %%
# 10 fold cv on X[selcted_features]

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# bookmark

keys_all = np.array(list(colorlist_new.keys()))

# combine to one dataset
print(labellist)
print(keys_all)
print(colorlist_new.keys())

perc_track_restricted = list(segment_length_restricted_sum_dict.values())
perc_track_restricted = np.hstack([np.hstack(y) for y in perc_track_restricted]).reshape(-1,1)

num_change_points = list(num_times_restricted_dict.values())
num_change_points = np.hstack([np.hstack(y) for y in num_change_points]).reshape(-1,1)

FP_all = np.vstack([
    FP_dict[k] for k in colorlist_new.keys()])

print(segment_length_restricted_sum_dict.keys())
print(FP_dict.keys())
print(num_times_restricted_dict.keys())

# %%

for k in colorlist_new.keys():
    print(k, np.mean(num_times_restricted_dict[k]))

# %%
FP_all = np.column_stack([FP_all, perc_track_restricted])
FP_all = np.vstack([
    FP_dict[k] for k in list(segment_length_restricted_sum_dict.keys())])
tracks_all = np.array([
    df_tracks_dict[k] for k in list(segment_length_restricted_sum_dict.keys())])
keys_order_FP = [k.split('100pPE')[0] for k in list(segment_length_restricted_sum_dict.keys())]
keys_order_FP_uniq = natsort.natsorted(np.unique(keys_order_FP))
lencond = [len(FP_dict[k]) for k in list(segment_length_restricted_sum_dict.keys())]
# FP_all = np.column_stack([FP_all, num_change_points])
X = FP_all
print(X.shape)

print('FP_dict', FP_dict.keys())
print('segm', segment_length_restricted_sum_dict.keys())
print('ncp', num_times_restricted_dict.keys())
print('c',colorlist_new.keys())
print(keys_order_FP)
print(lencond)
print('c',len(colorlist_new.keys()),len(labellist))
print(tracks_all.shape)

# '0.004fM BA1 ', '0.004fM BA1 ', 
# '0.04fM BA1 ', '0.04fM BA1 ', '
# 0.4fM BA1 ', '0.4fM BA1 ', 
# '4fM BA1 ', '4fM BA1 ', '
# 40fM BA1 ', '40fM BA1 ', 
# '400fM BA1 ', 
# 'no virus ', 'no virus '


labellist_2class_pred = [
                        0,0,
                         1,1,
                         2,2,
                         3,3,
                         4,4,
                         5,
                          6,6]
labellist_2class_pred = [
                        0,0,
                         0,0,
                         0,0,
                         0,0,
                         0,0,
                         0,
                          1,1]
labellist_2class_pred = [
                        
                        -1,-1,
                         -1,-1,
                         -1,-1,
                         -1,-1,
                         -1,-1,
                         0,
                          1,1]
labellist_2class_pred = [
                        0,0,
                         0,0,
                         0,0,
                         0,0,
                         0,0,
                         0,
                          1,1]

labellist_2class_pred_list  = [
                               [
                                -1,-1,
                                -1,-1,
                                -1,-1,
                                -1,-1,
                                -1,-1,
                                0,
                                1,1],
                                [
                                -1,-1,
                                -1,-1,
                                -1,-1,
                                -1,-1,
                                0,0,
                                -1,
                                1,1],
                                [
                                -1,-1,
                                -1,-1,
                                -1,-1,
                                0,0,
                                -1,-1,
                                -1,
                                1,1],
                                [
                                -1,-1,
                                -1,-1,
                                0,0,
                                -1,-1,
                                -1,-1,
                                -1,
                                1,1],
                                [
                                -1,-1,
                                0,0,
                                -1,-1,
                                -1,-1,
                                -1,-1,
                                -1,
                                1,1],
                                [
                                0,0,
                                -1,-1,
                                -1,-1,
                                -1,-1,
                                -1,-1,
                                -1,
                                1,1],
                                [-1,-1,
                                0,0,
                                0,0,
                                0,0,
                                0,0,
                                0,
                                1,1]
]

# 0: 400fM vs no virus, 10^7 check
# 1: 40fM vs no virus, 10^6 check
# 2: 4fM vs no virus, 10^5 check
# 3: 0.4fM vs no virus, 10^4 check
# 4: 0.04fM vs no virus, 10^3 off
# 5: 0.004fM vs no virus, 10^2 check
# 6: all conc vs no virus, all check

# max depth: 
# 400fM: 7
# 40fM: 7
# 4fM: 7
# 0.4fM: 3
# 0.04fM: 9
# 0.004fM: 3
# all conc: 6,7,8

# all conc: 6,7,8 (idx 6)
# 0.4fM 10^4: 3,4,5 (idx 3)
# 0.04fM 10^3: 9,10,11 (idx 4)
# 0.004fM 10^4: 3,4,5 (idx 5)
# rest 7,8,9 (idx: 0,1,2)

idx_virus_vs_novirus = 6
labellist_2class_pred = labellist_2class_pred_list[idx_virus_vs_novirus]

labellist_2class_pred_str = [
                '0.004fM BA1 ', '0.004fM BA1 ', 
                '0.04fM BA1 ', '0.04fM BA1 ',
                '0.4fM BA1 ', '0.4fM BA1 ', 
                '4fM BA1 ', '4fM BA1 ', 
                '40fM BA1 ', '40fM BA1 ', 
                '400fM BA1 ', 
                'no virus ', 'no virus ']
labellist_2class_pred = np.array(labellist_2class_pred)
print(labellist_2class_pred)
print(labellist_2class_pred_str)
included_labellist_2class_pred_str = [s for i, s in enumerate(labellist_2class_pred_str) if labellist_2class_pred[i]!=-1]
print('included_labellist_2class_pred_str', included_labellist_2class_pred_str)


seen_idx = []
included_labellist_2class_pred_str = []
for i, s in enumerate(labellist_2class_pred_str):
    if np.sum(labellist_2class_pred==-1)==2:
        included_labellist_2class_pred_str.append('virus')
        included_labellist_2class_pred_str.append('no virus')
        break
    if np.sum(labellist_2class_pred==-1)==0:
        included_labellist_2class_pred_str.append('virus')
        included_labellist_2class_pred_str.append('no virus')
        break
    if labellist_2class_pred[i] in seen_idx:
        continue
    if labellist_2class_pred[i]!=-1:
        included_labellist_2class_pred_str.append(s)
        seen_idx.append(labellist_2class_pred[i])

# plt.figure()
# for i in range(50):
#     plt.plot(tracks_all[0][i][:,0]-tracks_all[0][i][0,0], 
#              tracks_all[0][i][:,1]-tracks_all[0][i][0,1])
# plt.ylim(-20,130)
# plt.xlim(-130,20)

# plt.figure()
# for i in range(50):
#     plt.plot(tracks_all[-1][i][:,0]-tracks_all[-1][i][0,0], 
#              tracks_all[-1][i][:,1]-tracks_all[-1][i][0,1])
# plt.ylim(-20,130)
# plt.xlim(-130,20)

# use the y that suits what you want to predict either all classes or a given subset not -1
y = np.hstack([np.repeat(labellist[i], len(FP_dict[k])) 
               for i,k in enumerate(list(segment_length_restricted_sum_dict.keys()))])

y = np.hstack([np.repeat(labellist_2class_pred[i], len(FP_dict[k])) 
               for i,k in enumerate(list(segment_length_restricted_sum_dict.keys()))])
print(np.unique(y, return_counts=True))

X = X[y!=-1]
y = y[y!=-1]

# standardize
from sklearn.preprocessing import StandardScaler
# random oversample imblearn
from imblearn.over_sampling import RandomOverSampler

use_ensemble = True
if '0.004fM BA1 ' in included_labellist_2class_pred_str and len(included_labellist_2class_pred_str)==2:
    max_depth = 3
elif '0.04fM BA1 ' in included_labellist_2class_pred_str and len(included_labellist_2class_pred_str)==2:
    max_depth = 9
elif '0.4fM BA1 ' in included_labellist_2class_pred_str and len(included_labellist_2class_pred_str)==2:
    max_depth = 3
elif '4fM BA1 ' in included_labellist_2class_pred_str and len(included_labellist_2class_pred_str)==2:
    max_depth = 7
elif '40fM BA1 ' in included_labellist_2class_pred_str and len(included_labellist_2class_pred_str)==2:
    max_depth = 7
elif '400fM BA1 ' in included_labellist_2class_pred_str and len(included_labellist_2class_pred_str)==2:
    max_depth = 7
else:
    max_depth = 6

print('included_labellist_2class_pred_str', included_labellist_2class_pred_str, 'max_depth', max_depth)
print('max_depth', max_depth, max_depth+1, max_depth+2)

classifier_1 = LogisticRegression(max_iter=10000)
classifier_1 = RandomForestClassifier(n_estimators=100, 
                                     max_depth=max_depth, 
                                     random_state=0)
classifier_2 = RandomForestClassifier(n_estimators=100, 
                                     max_depth=max_depth+1, 
                                     random_state=0)
classifier_3 = RandomForestClassifier(n_estimators=100, 
                                     max_depth=max_depth+2, 
                                     random_state=0)

kf = KFold(n_splits=10, shuffle=True, random_state=42)
kf.get_n_splits(X)

accs = []
pred_all = []
true_all = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train[np.isnan(X_train)] = 0
    X_train[np.isinf(X_train)] = 0
    X_test[np.isnan(X_test)] = 0
    X_test[np.isinf(X_test)] = 0

    ros = RandomOverSampler(random_state=0)
    X_train, y_train = ros.fit_resample(X_train, y_train)

    if use_ensemble:
        pred1 = classifier_1.fit(X_train, y_train).predict(X_test)
        pred2 = classifier_2.fit(X_train, y_train).predict(X_test)
        pred3 = classifier_3.fit(X_train, y_train).predict(X_test)
        pred = scipy.stats.mode(np.column_stack([pred1, pred2, pred3]), axis=1, keepdims=True)[0].reshape(-1)
    else:
        clf = classifier_1.fit(X_train, y_train)
        pred = clf.predict(X_test)

    acc = np.mean(pred==y_test)
    accs.append(acc)
    pred_all.append(pred)
    true_all.append(y_test)

print(np.mean(accs), np.std(accs, ddof=1))

# confusion matrix

from sklearn.metrics import confusion_matrix
import seaborn as sns

fontsize = 25
l = [0,1,3,4,5]
labels_cm = included_labellist_2class_pred_str
labels_cm = ['V', 'NV']
print(labels_cm)
cm = confusion_matrix(np.hstack(true_all),
                      np.hstack(pred_all), 
                      normalize='true')

group_percentages = ['{0:.1%}'.format(value) for value in cm.flatten()]
labels = np.asarray(group_percentages).reshape(len(labels_cm),len(labels_cm))
plt.figure(figsize=(3,3))
if len(labels_cm)==2:
    ax = sns.heatmap(cm*100, annot=labels, 
                    fmt='', cmap='Blues', 
                    annot_kws={"fontsize":fontsize},
                    vmin=0, vmax=100,
                    cbar=False)
    plt.xticks(np.arange(len(labels_cm))+0.5, labels_cm, 
    rotation=0, ha='center', fontsize=16)
    plt.yticks(np.arange(len(labels_cm))+0.5, labels_cm, 
    rotation=90, va='center', fontsize=16)
    plt.xlabel('Predicted', fontsize=16)
    plt.ylabel('True', fontsize=16)
else:
    ax = sns.heatmap(cm*100, annot=labels, 
                    fmt='', cmap='Blues', 
                    annot_kws={"fontsize":12},
                    vmin=0, vmax=100,
                    cbar=False)
    plt.xticks(np.arange(len(labels_cm))+0.5, labels_cm, 
    rotation=45, ha='center', fontsize=12)
    plt.yticks(np.arange(len(labels_cm))+0.5, labels_cm, 
    rotation=45, va='center', fontsize=12)
    plt.xlabel('Predicted')
    plt.ylabel('True')

string_conditions = '_'.join(np.unique(included_labellist_2class_pred_str)).replace(' ', '')
plt.savefig(data_path+'final_figures/confusion_matrix_{}_{}_{}_min_length{}.pdf'.format(string_conditions, count_slow_name,count_sus_name,min_length),
            dpi=300, pad_inches=.1, bbox_inches='tight')
plt.show()
data_path+'final_figures/confusion_matrix_{}_{}_{}_min_length{}.pdf'.format(string_conditions, count_slow_name,count_sus_name,min_length)
# %%

ks = ['0.4fM BA1 100pPEGV4100sPEGV4 in breath cond.csv',
      '0.4fM BA1 100pPEGV4100sPEGV4 breath cond 2.csv']
k = ks[0]

tracks_to_plot = df_tracks_dict[k]
ensemble_pred_ = ensemble_pred_dict[k]
ensemble_score_ = ensemble_score_dict[k]
pred = pred_all_dict[k]
FP_ = FP_dict[k]
print(pred)
plt.figure(figsize=(5,5))
plt.hist(np.array(FP_)[np.array(pred)==0][:,-1], alpha=0.7, 
         bins=50, range=(0,100), density=True)
plt.hist(np.array(FP_)[np.array(pred)==1][:,-1], alpha=0.7, 
         bins=50, range=(0,100), density=True)
plt.hist(np.array(FP_)[np.array(pred)==2][:,-1], alpha=0.7, 
         bins=50, range=(0,100), density=True)

k = ks[1]
tracks_to_plot = df_tracks_dict[k]
ensemble_pred_ = ensemble_pred_dict[k]
ensemble_score_ = ensemble_score_dict[k]
pred = pred_all_dict[k]
FP_ = FP_dict[k]
print(pred)
plt.figure(figsize=(5,5))
plt.hist(np.array(FP_)[np.array(pred)==0][:,-1], alpha=0.7, 
         bins=50, range=(0,100), density=True)
plt.hist(np.array(FP_)[np.array(pred)==1][:,-1], alpha=0.7, 
         bins=50, range=(0,100), density=True)
plt.hist(np.array(FP_)[np.array(pred)==2][:,-1], alpha=0.7, 
         bins=50, range=(0,100), density=True)

ks = ['0.04fM BA1 100pPEGV4100sPEGV4 breath cond.csv',
      '0.04fM BA1 100pPEGV4 100sPEGV4 breath cond 2.csv']
k = ks[0]

tracks_to_plot = df_tracks_dict[k]
ensemble_pred_ = ensemble_pred_dict[k]
ensemble_score_ = ensemble_score_dict[k]
pred = pred_all_dict[k]
FP_ = FP_dict[k]
print(pred)
plt.figure(figsize=(5,5))
plt.hist(np.array(FP_)[np.array(pred)==0][:,-1], alpha=0.7, 
         bins=50, range=(0,100), density=True)
plt.hist(np.array(FP_)[np.array(pred)==1][:,-1], alpha=0.7, 
         bins=50, range=(0,100), density=True)
plt.hist(np.array(FP_)[np.array(pred)==2][:,-1], alpha=0.7, 
         bins=50, range=(0,100), density=True)

k = ks[1]
tracks_to_plot = df_tracks_dict[k]
ensemble_pred_ = ensemble_pred_dict[k]
ensemble_score_ = ensemble_score_dict[k]
pred = pred_all_dict[k]
FP_ = FP_dict[k]
print(pred)
plt.figure(figsize=(5,5))
plt.hist(np.array(FP_)[np.array(pred)==0][:,-1], alpha=0.7, 
         bins=50, range=(0,100), density=True)
plt.hist(np.array(FP_)[np.array(pred)==1][:,-1], alpha=0.7, 
         bins=50, range=(0,100), density=True)
plt.hist(np.array(FP_)[np.array(pred)==2][:,-1], alpha=0.7, 
         bins=50, range=(0,100), density=True)
# %%

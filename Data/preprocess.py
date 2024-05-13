import os, pickle, tqdm
import numpy as np
from skimage.util.shape import view_as_windows
from scipy import stats
import scipy.signal as sci_sig

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sci_sig.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    data = sci_sig.detrend(data)
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sci_sig.lfilter(b, a, data)
    return y

def slidingWindow(sequence, winSize, step=1):
    num_of_chunks = ((len(sequence) - winSize) / step) + 1
    for i in range(0, int(num_of_chunks) * step, step):
        yield sequence[i:i + winSize]

def filtered_resample_sig(signal_data, fs, band_pass_low, band_pass_high, band_pass_order, resample_fs):
    filtered_signal = butter_bandpass_filter(signal_data, band_pass_low, 
                                             band_pass_high, fs, 
                                             order=band_pass_order)

    num_secs_in_signal = len(filtered_signal) / fs
    num_samples_to_resample = round(num_secs_in_signal * resample_fs)
    filtered_signal_resample = sci_sig.resample(filtered_signal, num_samples_to_resample)
    return filtered_signal_resample

def preprocess_ieee(data_dir, band_pass_low, band_pass_high, band_pass_order, resample_fs, 
                    ieee_data: str="IEEE_TRAIN"):
    if not os.path.exists("./Data/IEEE_TRAIN/IEEE_SPC_train.pkl") or not os.path.exists("./Data/IEEE_TEST/IEEE_SPC_test.pkl"):
        data = dict()
        if (ieee_data=="IEEE_TEST"):
            dataset_list = [file for file in os.listdir(data_dir) if file.endswith('Test_Data')]
        elif (ieee_data=="IEEE_TRAIN"):
            dataset_list = [file for file in os.listdir(data_dir) if file.endswith('Train_Data')]
            
        for idx, ieee_dataset in enumerate(dataset_list):
            
            with open(os.path.join(data_dir, ieee_dataset), 'rb') as f:
                dataset = pickle.load(f)

            # get sample rates of signals
            fs_dataset_PPG = dataset[0]["PPG_fs"]
            fs_dataset_ACC = dataset[0]["ACC_fs"]
            
            x_list, y_list, sub_list = [], [], []
            for subject in tqdm.tqdm(range(len(dataset)), 
                                            desc='Sessions: ', total=len(dataset)): 
                ppg1_signal_windowed = []
                ppg2_signal_windowed = []
                accx_signal_windowed = []
                accy_signal_windowed = []
                accz_signal_windowed = []
                
                # preprocess the PPG signal - filter, resample and window
                ppg_signal1 = filtered_resample_sig(dataset[subject]["Raw_PPG_1"], fs_dataset_PPG,
                                         band_pass_low, band_pass_high, band_pass_order, resample_fs)
                for i in slidingWindow(ppg_signal1, 8 * resample_fs, 2 * resample_fs):
                    i = stats.zscore(i)
                    ppg1_signal_windowed.append(i)
                    
                ppg_signal2 = filtered_resample_sig(dataset[subject]["Raw_PPG_2"], fs_dataset_PPG,
                                         band_pass_low, band_pass_high, band_pass_order, resample_fs)
                for i in slidingWindow(ppg_signal2, 8 * resample_fs, 2 * resample_fs):
                    i = stats.zscore(i)
                    ppg2_signal_windowed.append(i)

                accx_signal = filtered_resample_sig(dataset[subject]["Raw ACC_X"], fs_dataset_ACC,
                                         band_pass_low, band_pass_high, band_pass_order, resample_fs)
                for i in slidingWindow(accx_signal, 8 * resample_fs, 2 * resample_fs):
                    i = stats.zscore(i)
                    accx_signal_windowed.append(i)

                accy_signal = filtered_resample_sig(dataset[subject]["Raw ACC_Y"], fs_dataset_ACC,
                                         band_pass_low, band_pass_high, band_pass_order, resample_fs)
                for i in slidingWindow(accy_signal, 8 * resample_fs, 2 * resample_fs):
                    i = stats.zscore(i)
                    accy_signal_windowed.append(i)

                accz_signal = filtered_resample_sig(dataset[subject]["Raw ACC_Z"], fs_dataset_ACC,
                                         band_pass_low, band_pass_high, band_pass_order, resample_fs)
                for i in slidingWindow(accz_signal, 8 * resample_fs, 2 * resample_fs):
                    i = stats.zscore(i)
                    accz_signal_windowed.append(i)

                signals_stacked = np.stack((np.array(ppg1_signal_windowed),
                                            np.array(ppg2_signal_windowed),
                                            np.array(accx_signal_windowed),
                                            np.array(accy_signal_windowed), 
                                            np.array(accz_signal_windowed)), axis=2)
                if idx == 1:
                    pass
                else:
                    sub_list.append(np.full(len(signals_stacked), subject))
                x_list.append(signals_stacked)
                y_list.append(dataset[subject]["truth_values"])
            
            """1,2 | 6,7"""
            if ieee_dataset == 'IEEE_Test_Data':
                # dup_idx = ['02': [1,2], '6': [6,7]]
                x_list[1] = np.concatenate([x_list[1], x_list[2]], axis=0)
                y_list[1] = np.concatenate([y_list[1], y_list[2]], axis=0)
                x_list[6] = np.concatenate([x_list[6], x_list[7]], axis=0)
                y_list[6] = np.concatenate([y_list[6], y_list[7]], axis=0)

                del x_list[2], x_list[7], y_list[2], y_list[7]
                
                sub_list = [np.full(x.shape[0], sub) for sub, x in enumerate(x_list)]

            sub = np.hstack(sub_list)
            X = np.vstack(x_list)
            y = np.reshape(np.hstack(y_list), (-1, 1))

            name = dataset[0]['Dataset'].split()[-1]
            
            data[f'X_{name}'] = X
            data[f'y_{name}'] = y
            data[f'subject_{name}'] = sub+data['subject_Train'][-1]+1 if (len(dataset_list)==2) and (idx==1) else sub
        
        data_final = dict()
        if len(dataset_list) == 2:
            data_final['X'] = np.swapaxes(np.concatenate([data['X_Train'], data['X_Test']], axis=0), 1, 2)
            data_final['y'] = np.concatenate([data['y_Train'], data['y_Test']], axis=0)
            data_final['subject'] = np.concatenate([data['subject_Train'], data['subject_Test']], axis=0)
        
        elif len(dataset_list) == 1:
            data_final['X'] = np.swapaxes(data[f'X_{name}'], 1, 2)
            data_final['y'] = data[f'y_{name}']
            data_final['subject'] = data[f'subject_{name}']
        
        # save ieee dataset as pickle
        if ieee_data=="IEEE_TEST":
            with open(os.path.join(data_dir, f"IEEE_SPC_test.pkl"), 'wb') as f:
                pickle.dump(data_final, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(os.path.join(data_dir, f"IEEE_SPC_train.pkl"), 'wb') as f:
                pickle.dump(data_final, f, pickle.HIGHEST_PROTOCOL)
    
    else:
        if ieee_data=="IEEE_TEST":
            with open(os.path.join(data_dir, f"IEEE_SPC_test.pkl"), 'rb') as f:
                data_final = pickle.load(f)
        else:
            with open(os.path.join(data_dir, f"IEEE_SPC_train.pkl"), 'rb') as f:
                data_final = pickle.load(f)
    data_final['X'] = np.concatenate([np.expand_dims(data_final['X'][:, 0,:], axis=1), np.expand_dims(data_final['X'][:, 1,:], axis=1)], axis=1)
    return data_final

def preprocess_dalia(data_dir):
    fs = 32
    window_size = 8
    
    S = dict()
    acc = dict()
    ppg = dict()
    target = dict()
    
    # Load data
    if not os.path.exists(os.path.join(data_dir, 'PPG_dalia.pkl')):
        subjects = list(range(1, 16))
        for sub in subjects:
            with open(data_dir + 'S' + str(sub) + '/' + 'S' + str(sub) + '.pkl', 'rb') as f:
                S[sub] = pickle.load(f, encoding='latin1')
            ppg[sub] = S[sub]['signal']['wrist']['BVP'][::2]
            acc[sub] = S[sub]['signal']['wrist']['ACC']
            target[sub] = S[sub]['label']
            
        sig = dict()
        subject_list = []
        sig_list = []
        target_list = []
        
        for k in target:
            sig[k] = np.concatenate((ppg[k], acc[k]), axis=1)   # PPG + ACC
            
            # PPG downsampling
            sig[k] = np.moveaxis(view_as_windows(sig[k], (fs * window_size, 4), fs*2)[:, 0, :, :], 1, 2)  # shape 확인
            
            subject_list.append(np.full(sig[k].shape[0], k)) 
            sig_list.append(sig[k])
            target[k] = np.reshape(target[k], (target[k].shape[0], 1))
            target_list.append(target[k])
        
        subject_list = np.hstack([sub-1 for sub in subject_list])
        X = np.vstack(sig_list)
        y = np.reshape(np.vstack(target_list), (-1, 1))
        
        data = dict()
        data['X'] = X
        data['y'] = y
        data['subject'] = subject_list
        
        with open(os.path.join(data_dir, 'PPG_dalia.pkl'), "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        
    else:
        with open(os.path.join(data_dir, 'PPG_dalia.pkl'), "rb") as f:
                data = pickle.load(f)
    
    data['X'] = np.expand_dims(data['X'][:, 0,:], axis=1) # PPG only
    return data
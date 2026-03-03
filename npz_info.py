import numpy as np

early_trials = list(range(1, 8))  # trials 1 to 7

for trial in early_trials:
    npz_path = f"kaggle_data/train/S00{trial}_trial_info.npz"
    npz = np.load(npz_path)
    print(f"Trial {trial} - keys: {npz.files}")
    
    for k in npz.files:
        arr = npz[k]
        print(f"{k}: shape={arr.shape}, dtype={arr.dtype}")
        if arr.ndim == 0:
            print(arr.item())
        else:
            print(arr[:10])

    ends = npz["end_bins"] 
    starts = npz["start_bins"]
    lengths = ends - starts
    print("IMPORTANT:")
    print("min/median/max trial length:", lengths.min(), np.median(lengths), lengths.max())
    print("+++++++++++++++++++++++++++++")
    print("num trials >=201:", (lengths >= 201).sum(), "out of", len(lengths))

print(" ====================================\n")

mid_trials = list(range(150, 159))  # trials 8 to 14

for trial in mid_trials:
    npz_path = f"kaggle_data/train/S{trial}_trial_info.npz"
    npz = np.load(npz_path)
    print(f"Trial {trial} - keys: {npz.files}")
    
    for k in npz.files:
        arr = npz[k]
        print(f"{k}: shape={arr.shape}, dtype={arr.dtype}")
        if arr.ndim == 0:
            print(arr.item())
        else:
            print(arr[:10])

    ends = npz["end_bins"] 
    starts = npz["start_bins"]
    lengths = ends - starts
    print("IMPORTANT:")
    print("min/median/max trial length:", lengths.min(), np.median(lengths), lengths.max())
    print("+++++++++++++++++++++++++++++")
    print("num trials >=201:", (lengths >= 201).sum(), "out of", len(lengths))

print(" ====================================\n")

late_trials = list(range(231, 235))  # trials 15 to 21
   

for trial in late_trials:
    npz_path = f"kaggle_data/train/S{trial}_trial_info.npz"
    npz = np.load(npz_path)
    print(f"Trial {trial} - keys: {npz.files}")
    
    for k in npz.files:
        arr = npz[k]
        print(f"{k}: shape={arr.shape}, dtype={arr.dtype}")
        if arr.ndim == 0:
            print(arr.item())
        else:
            print(arr[:10])

    ends = npz["end_bins"] 
    starts = npz["start_bins"]
    lengths = ends - starts
    print("IMPORTANT:")
    print("min/median/max trial length:", lengths.min(), np.median(lengths), lengths.max())
    print("+++++++++++++++++++++++++++++")
    print("num trials >=201:", (lengths >= 201).sum(), "out of", len(lengths))

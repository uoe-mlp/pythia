'''
Use this script to specify your gridsearch parameters
'''
import json
import os
from datetime import datetime
import subprocess


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

def js_r(filename: str):
    with open(filename) as f_in:
        return json.load(f_in)

default_settings = js_r(os.path.join('pythia', 'utils', 'grid_search', 'settings_default.json'))
grid_search_dir = os.path.join('data', 'experiments', 'grid_search___' + timestamp)

if not os.path.isdir(grid_search_dir):
    os.mkdir(grid_search_dir)


# CHANGE THIS ONE
features_targets = ["SPY", "IWM", "ONEQ"]

# Settings
date_pairs = [("01-01-2004", "20-12-2018"), ("01-01-2005", "20-12-2019"), ("01-01-2006", "20-12-2020")]
learning_rates = [1e-3, 1e-4, 1e-5]
epochs = [1000, 2000]
hidden_sizes = [[32], [64, 64], [128, 128, 128]]
batch_sizes = [100, 500]
window_sizes = [11, 22, 44]

for ft in features_targets:
    for dp in date_pairs:
        for lr in learning_rates:
            for e in epochs:
                for hs in hidden_sizes:
                    for bs in batch_sizes:
                        for ws in window_sizes:
                            # Allocate correct settingss
                            default_settings["market"]["params"]["features"] = ft
                            default_settings["market"]["params"]["targets"] = ft
                            default_settings["market"]["params"]["start_date"] = dp[0]
                            default_settings["market"]["params"]["end_date"] = dp[1]
                            default_settings["agent"]["params"]["predictor"]["params"]["initial_learning_rate"] = lr
                            default_settings["agent"]["params"]["predictor"]["params"]["epochs"] = e
                            default_settings["agent"]["params"]["predictor"]["params"]["hidden_size"] = hs
                            default_settings["agent"]["params"]["predictor"]["params"]["batch_size"] = bs
                            default_settings["agent"]["params"]["predictor"]["params"]["window_size"] = ws



                            # Create subfolder
                            run_dir = os.path.join(grid_search_dir, "ft-%s_dp-%s_lr-%f_e-%i_hs-%i-%i_bs-%i_ws-%i" % (ft, dp[0], lr, e, hs[0], len(hs), bs, ws))
                            if not os.path.isdir(run_dir):
                                os.mkdir(run_dir)
                            
                            # Set the output folder
                            default_settings["analysis"]["folder"] = run_dir


                            with open(os.path.join(run_dir, "settings.json"), 'w') as fp:
                                json.dump(default_settings, fp, indent=4, sort_keys=True)

                            # Now do the run
                            subprocess.run(["python", "pythia", "--run", os.path.join(run_dir, "settings.json")])




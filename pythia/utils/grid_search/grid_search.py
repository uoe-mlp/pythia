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



num_layers = [2, 3]
hidden_sizes = [32, 64, 128]
window_sizes = [11, 22, 44]

for l in num_layers:
    for h in hidden_sizes:
        for w in window_sizes:
            default_settings["agent"]["params"]["predictor"]["params"]["hidden_size"] = [h] * l
            default_settings["agent"]["params"]["predictor"]["params"]["window_size"] = w
            # Create subfolder
            run_dir = os.path.join(grid_search_dir, "l_%i_h_%i_w_%i" % (l, h, w))
            if not os.path.isdir(run_dir):
                os.mkdir(run_dir)

            with open(os.path.join(run_dir, "settings.json"), 'w') as fp:
                json.dump(default_settings, fp, indent=4, sort_keys=True)

            # Now do the run
            subprocess.run(["python", "pythia", "--run", os.path.join(run_dir, "settings.json")])




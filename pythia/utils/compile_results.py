import json
import copy
import re
from pythia.utils import ArgsParser
import csv
import os
'''
This file is not integrated into the main framework. Its only purpose is to compile
results from output data into csv files
'''

# Specify Main and sub-directory here
directory = "chalvatzis_short"
sub_dir = "results__20210308_154939"
epochs_between_validation = 5
final_epochs = 3

timestamp_folder = os.path.join("data", "experiments", directory, sub_dir)

all_metrics = ["Epochs", "Cumulative Return", "Maximum Drawdown", "Mean Directional Accuracy",
                    "Number of Trades", "Sharpe Ratio", "Sortino Ratio", "Volatility"]

# Since the files are processed out of order keep a dictionary
ordered_metrics = {}
final_epochs = final_epochs if final_epochs != 0 else epochs_between_validation

# Get list of all files
folders = [f for f in os.listdir(timestamp_folder) if re.match('^train_', f)]
last_folder_index = len(folders) - 1
for f in folders:
    subdir = os.path.join(timestamp_folder, f, 'data.json')
    with open(subdir, 'r') as fp:
        data = json.load(fp)

    index = int(f[6:])
    epochs = (index + 1) * epochs_between_validation if index != last_folder_index else index * epochs_between_validation + final_epochs
    cumulative_return = ArgsParser.get_or_error(data, 'cumulative_return')
    maximum_drawdown = ArgsParser.get_or_error(data, 'maximum_drawdown')
    mean_directional_accuracy = ArgsParser.get_or_error(data, 'mean_directional_accuracy')
    number_of_trades = ArgsParser.get_or_error(data, 'number_of_trades')
    sharpe_ratio = ArgsParser.get_or_error(data, 'sharpe_ratio')
    sortino_ratio = ArgsParser.get_or_error(data, 'sortino_ratio')
    volatility = ArgsParser.get_or_error(data, 'volatility')

    ordered_metrics[index] = [epochs, cumulative_return, maximum_drawdown, mean_directional_accuracy, number_of_trades,
                        sharpe_ratio, sortino_ratio, volatility]

sorted_dict = dict(sorted(ordered_metrics.items()))

output_csv = os.path.join(timestamp_folder, "train_full")
if not os.path.isdir(output_csv):
    os.mkdir(output_csv)
output_csv = os.path.join(output_csv, "output.csv")
# Write to csv
with open(output_csv, 'w+', newline='\n') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(all_metrics)
        for k, v in sorted_dict.items():
            writer.writerow(v)

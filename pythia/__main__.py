# main.py
from os.path import isfile
import sys
import os

from pythia.experiment import ExperimentParser


HELP_MSG = """
Usage: python pythia [command] [file]
        
The file contains the experiment settings.

-h, --help         print helper
-r, --run          runs the file with specifications
"""

ERROR_MSG = """
Command %s not recognised. 

Valid commands are --help (-h) and --run (-r).
"""

FILE_NOT_FOUND_MSG = """
File %s not found.
"""

FILE_NOT_PROVIDED_MSG = """
File not provided. Usage: python pythia [command] [file]
"""

if __name__ == "__main__":
    args = sys.argv
    if len(args) == 0:
        print(HELP_MSG)
    elif args[1] in ['-h', '--help']:
        print(HELP_MSG)
    elif args[1] in ['-r', '--run']:
        if len(args) > 2:
            file = args[2]
            if isfile(file):
                experiment = ExperimentParser.parse(file)
                experiment.run()
            else:
                print(FILE_NOT_FOUND_MSG % file)
        else:
            print(FILE_NOT_PROVIDED_MSG)
    else:
        print(ERROR_MSG % args[1])
    
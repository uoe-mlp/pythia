# Makefile

conda-update:
	--conda update conda --all
	--conda update anaconda

conda-create:
	--conda env create -f ./environment.yml
	
conda-freeze:
	--conda list -e > requirements.txt

remove-cache:
	--find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf
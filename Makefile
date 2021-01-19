# Makefile

conda-update:
	--conda update conda --all
	--conda update anaconda

conda-create:
	--conda env create -f ./environment.yml
	
conda-freeze:
	--conda list -e > requirements.txt
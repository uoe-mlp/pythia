# Makefile

conda-update:
	--conda update conda --all
	--conda update anaconda

conda-create:
	--conda create --name mlp_lstm python=3.8
	--conda install -n mlp_lstm --file requirements.txt

conda-activate:
ifeq ($(OS),Windows_NT)
	--activate mlp_lstm
else
	# --source activate mlp_lstm
endif
	
conda-freeze:
	--conda list -e > requirements.txt
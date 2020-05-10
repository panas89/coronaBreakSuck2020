.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = coronaBreakSuck2020
PYTHON_INTERPRETER = python3

# url to download data

metadata_DATA_URL = https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-04-10/metadata.csv
medrxiv_DATA_URL = https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-04-03/biorxiv_medrxiv.tar.gz
comm_DATA_URL = https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/latest/comm_use_subset.tar.gz
non_comm_DATA_URL = https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/latest/noncomm_use_subset.tar.gz
cust_DATA_URL = https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/latest/custom_license.tar.gz

forecast_US_conf = https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv
forecast_global_conf = https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv
forecast_US_death = https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv
forecast_global_death = https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv
forecast_global_recovered = https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv


ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Save current requirements
save_requirements: 
	$(PYTHON_INTERPRETER) -m pip freeze > requirements.txt

## Download datasets
download_data:
	@echo ">>> Downloading data from Semantic Scholar"
	curl -o data/raw/metadata.csv $(metadata_DATA_URL)
	@echo ">>> Downloading data from Semantic Scholar"
	@echo ">>> Downloading medarxiv json files"
	curl -o data/raw/biorxiv_medrxiv.tar.gz $(medrxiv_DATA_URL)
	@echo ">>> Unzipping."
	tar xvzf data/raw/biorxiv_medrxiv.tar.gz -C data/raw
	@echo ">>> Downloading commercial data json files"
	curl -o data/raw/comm_use_subset.tar.gz $(comm_DATA_URL)
	@echo ">>> Unzipping."
	tar xvzf data/raw/comm_use_subset.tar.gz -C data/raw
	@echo ">>> Downloading non commercial data json files"
	curl -o data/raw/noncomm_use_subset.tar.gz $(non_comm_DATA_URL)
	@echo ">>> Unzipping."
	tar xvzf data/raw/noncomm_use_subset.tar.gz -C data/raw
	@echo ">>> Downloading custom data json files"
	curl -o data/raw/custom_license.tar.gz $(cust_DATA_URL)
	@echo ">>> Unzipping."
	tar xvzf data/raw/custom_license.tar.gz -C data/raw

download_forecasting_data:
	@echo ">>> Downloading Forecasting data from John Hopkins"
	@echo ">>> Downloading data confirmed cases USA"
	curl -o data/raw/conf_USA.csv $(forecast_US_conf)
	@echo ">>> Downloading data death data USA"
	curl -o data/raw/death_USA.csv $(forecast_US_death)
	@echo ">>> Downloading data confirmed cases Global"
	curl -o data/raw/conf_global.csv $(forecast_global_conf)
	@echo ">>> Downloading data death data global"
	curl -o data/raw/death_global.csv $(forecast_global_death)
	@echo ">>> Downloading data recovered cases global"
	curl -o data/raw/recovered_global.csv $(forecast_global_recovered)


## Make Dataset
#getting raw json files, not for metadata but other arxivs
data: #requirements
	$(PYTHON_INTERPRETER) covid/data/make_dataset.py data/raw/biorxiv_medrxiv/pdf_json/ data/processed/ bioarxiv.csv ["title","abstract"] False
	$(PYTHON_INTERPRETER) covid/data/make_dataset.py data/raw/comm_use_subset/pdf_json/ data/processed/ comm_use_subset.csv ["title","abstract"] False
	$(PYTHON_INTERPRETER) covid/data/make_dataset.py data/raw/noncomm_use_subset/pdf_json/ data/processed/ noncomm_use_subset.csv ["title","abstract"] False
	$(PYTHON_INTERPRETER) covid/data/make_dataset.py data/raw/custom_license/pdf_json/ data/processed/ custom_license.csv ["title","abstract"] False

#joining csv files to metadata csv
join_datasets: 
	$(PYTHON_INTERPRETER) covid/data/join_datasets.py data/raw/ data/raw/merged_raw_data.csv metadata.csv ["bioarxiv.csv","comm_use_subset.csv","noncomm_use_subset.csv","custom_license.csv"]

## Query Datasets
query_data: #requirements
	$(PYTHON_INTERPRETER) covid/data/query_data.py data/raw/ data/filtered/ metadata.csv ["title","abstract"]

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Upload Data to S3
sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/
else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif

## Download Data from S3
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')

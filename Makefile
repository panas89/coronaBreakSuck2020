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
# date_str = $(shell date +'%Y-%m-%d')
date_str = 2022-01-31#$(shell date +%Y-%m-%d -d "2 days ago")

start_date = 2020-01-01 #date to inlcude data entries

#https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases.html
DATA_URL_Sem_Schol = https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_$(date_str).tar.gz

#https://www.dimensions.ai/covid19/
DATA_dimensions = https://dimensions.figshare.com/ndownloader/files/30718538

forecast_US_conf = https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv
forecast_global_conf = https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv
forecast_US_death = https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv
forecast_global_death = https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv
forecast_global_recovered = https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv


#yaml_path = covid/models/paperclassifier/interest.yaml
# yaml_path = covid/models/paperclassifier/Davids_interest.yaml
yaml_path = covid/models/paperclassifier/Davids_interest_meshed.yaml


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


################################# Semantics  scholar topics #################################

sem_scholar_topics: download_data download_forecasting_data data join_datasets mesh_yaml classify_data preproc_dataset make_topics

################################ Semantics  scholar #########################################
#############################################################################################

## Download datasets
download_data:
	@echo ">>> Downloading data from Semantic Scholar"
	@echo ">>> Downloading data files of $(date_str)"
	curl -o data/raw/cord-19_$(date_str).tar.gz $(DATA_URL_Sem_Schol)
	@echo ">>> Unzipping. $(date_str) document parses"
	tar xvzf data/raw/$(date_str)/document_parses.tar.gz -C data/raw/$(date_str)
	@echo ">>> Unzipping."
	tar xvzf data/raw/cord-19_$(date_str).tar.gz -C data/raw
	@echo ">>> Unzipping. $(date_str) embeddings"
	tar xvzf data/raw/$(date_str)/cord_19_embeddings.tar.gz -C data/raw/$(date_str)

download_forecasting_data:
	@echo ">>> Downloading Forecasting data from John Hopkins"
	@echo ">>> Downloading data confirmed cases USA"
	curl -o data/raw/$(date_str)/conf_USA.csv $(forecast_US_conf)
	@echo ">>> Downloading data death data USA"
	curl -o data/raw/$(date_str)/death_USA.csv $(forecast_US_death)
	@echo ">>> Downloading data confirmed cases Global"
	curl -o data/raw/$(date_str)/conf_global.csv $(forecast_global_conf)
	@echo ">>> Downloading data death data global"
	curl -o data/raw/$(date_str)/death_global.csv $(forecast_global_death)
	@echo ">>> Downloading data recovered cases global"
	curl -o data/raw/$(date_str)/recovered_global.csv $(forecast_global_recovered)

semantics_scholar_topics: data join_datasets mesh_yaml classify_data preproc_dataset make_topics

## Make Dataset
#getting raw json files, not for metadata but other arxivs
data: #requirements
	$(PYTHON_INTERPRETER) covid/data/make_dataset.py data/raw/$(date_str)/document_parses/pdf_json/ data/raw/$(date_str)/ merged_raw_data.csv ['title','abstract'] False

#joining csv files to metadata csv to get publish time
join_datasets: 
	$(PYTHON_INTERPRETER) covid/data/join_datasets.py data/raw/$(date_str)/ data/raw/merged_raw_data.csv metadata.csv merged_raw_data.csv

## Using Mesh keyworkds to update the yaml file
mesh_yaml: #requirements
	$(PYTHON_INTERPRETER) covid/models/paperclassifier/meshsearch.py covid/models/paperclassifier/Davids_interest.yaml covid/models/paperclassifier/ Davids_interest_meshed.yaml

## Classify Datasets to find only covid papers reduces file size by 100 fold
cols_sem_scholar = ["sha","title","doi","abstract_x","publish_time","affiliations","location"] #default cols to follow through code 
classify_data: #requirements
	$(PYTHON_INTERPRETER) covid/data/classify_data.py data/raw/merged_raw_data.csv data/paperclassifier/classified_merged_covid.csv $(yaml_path) $(cols_sem_scholar)


## Preprocess Datasets
preproc_dataset: #location and affilliations classification
	###### get location for all papers around 2hrs run time
	# $(PYTHON_INTERPRETER) covid/data/preproc_dataset.py data/raw/merged_raw_data.csv data/processed/merged_raw_data.csv 11
	###### get location for covid papers only
	$(PYTHON_INTERPRETER) covid/data/preproc_dataset.py data/paperclassifier/classified_merged_covid.csv data/processed/classified_merged_covid.csv 11

## Run topic modelling over covid corpus
make_topics:
	$(PYTHON_INTERPRETER) covid/models/topicmodeling/topic_generator.py $(yaml_path) classified_merged_covid sem_scholar_covid_topics $(start_date)

#############################################################################################
#############################################################################################

################################# Dimensions topics #################################

dimensions_publications_topics: download_dimensions dimensions_publications_datasets classify_dimensions_publications_data preproc_dimensions_publications_dataset top_5_perc_altmetric_papers top_5_10_perc_altmetric_papers make_dimensions_publications_topics

################################ Dimensions papers  #########################################
#############################################################################################

sheet_name_pub = Publications
sheet_name_dat = Datasets
sheet_name_clin_trials = "Clinical Trials"
sheet_name_grants = Grants

## Download datasets
download_dimensions:
	wget -O data/raw/$(date_str)/dimensions.zip $(DATA_dimensions)
	unzip data/raw/$(date_str)/dimensions.zip -d data/raw/$(date_str)
	# depending on the format the data in the zip are they have to be moved to the raw date folder
	cp data/raw/$(date_str)/tmp/tmpl41unnri/dimensions-covid19-export-2021-09-01-h15-01-02_publications.csv data/raw/$(date_str)/dims_$(sheet_name_pub)_raw_data.csv
	cp data/raw/$(date_str)/tmp/tmpl41unnri/dimensions-covid19-export-2021-09-01-h15-01-02_datasets.csv data/raw/$(date_str)/dims_$(sheet_name_dat)_raw_data.csv
	cp data/raw/$(date_str)/tmp/tmpl41unnri/dimensions-covid19-export-2021-09-01-h15-01-02_grants.csv data/raw/$(date_str)/dims_$(sheet_name_grants)_raw_data.csv
	cp data/raw/$(date_str)/tmp/tmpl41unnri/dimensions-covid19-export-2021-09-01-h15-01-02_clinical_trials.csv data/raw/$(date_str)/dims_$(sheet_name_clin_trials)_raw_data.csv



#making dataset of dimensions publications into csv
colsA_pub = ["Publication ID","Title","DOI","Abstract","Publication Date","Authors Affiliations","Country of Research organization"] #cols to be renamed
colsB_pub = ["sha","title","doi","abstract_x","publish_time","affiliations","location"] #default cols to follow through code 

dimensions_publications_filepath = /tmp/tmpl41unnri/dimensions-covid19-export-2021-09-01-h15-01-02_publications.csv
dimensions_publications_datasets: 
	$(PYTHON_INTERPRETER) covid/data/make_dimensions_dataset.py data/raw/$(date_str)/ data/raw/$(date_str)/dims_$(sheet_name_pub)_raw_data.csv $(dimensions_publications_filepath) False $(sheet_name_pub) $(colsA_pub) $(colsB_pub) 

## Classify Datasets to find only covid papers reduces file size by 100 fold
classify_dimensions_publications_data: #requirements
	$(PYTHON_INTERPRETER) covid/data/classify_data.py data/raw/$(date_str)/dims_$(sheet_name_pub)_raw_data.csv data/paperclassifier/classified_dims_$(sheet_name_pub)_covid.csv $(yaml_path) $(colsB_pub)

## Preprocess Datasets
preproc_dimensions_publications_dataset: #location and affilliations classification
	###### get location for all papers around 2hrs run time
	# $(PYTHON_INTERPRETER) covid/data/preproc_dataset.py data/raw/merged_raw_data.csv data/processed/merged_raw_data.csv 11
	###### get location for covid papers only
	$(PYTHON_INTERPRETER) covid/data/preproc_dataset.py data/paperclassifier/classified_dims_$(sheet_name_pub)_covid.csv data/processed/classified_dims_$(sheet_name_pub)_covid.csv 11

## Filtering top 5% high impact papers
lb_95 = 0.95
ub_100 = 1.00
top_5_perc_altmetric_papers: 
	$(PYTHON_INTERPRETER) covid/data/make_quantile_dimentions_papers.py data/raw/$(date_str)/ data/processed/classified_dims_$(sheet_name_pub)_hi_95_100_covid.csv $(dimensions_publications_filepath) data/processed/classified_dims_$(sheet_name_pub)_covid.csv $(sheet_name_pub) $(lb_95) $(ub_100) 

## Filtering top 5% high impact papers
lb_90 = 0.90
ub_95 = 0.9499
top_5_10_perc_altmetric_papers: 
	$(PYTHON_INTERPRETER) covid/data/make_quantile_dimentions_papers.py data/raw/$(date_str)/ data/processed/classified_dims_$(sheet_name_pub)_hi_90_95_covid.csv $(dimensions_publications_filepath) data/processed/classified_dims_$(sheet_name_pub)_covid.csv $(sheet_name_pub) $(lb_90) $(ub_95) 

## Run topic modelling over covid corpus
make_dimensions_publications_topics:
	$(PYTHON_INTERPRETER) covid/models/topicmodeling/topic_generator.py $(yaml_path) classified_dims_$(sheet_name_pub)_hi_95_100_covid dims_$(sheet_name_pub)_hi_95_100_covid_topics $(start_date)
	$(PYTHON_INTERPRETER) covid/models/topicmodeling/topic_generator.py $(yaml_path) classified_dims_$(sheet_name_pub)_hi_90_95_covid dims_$(sheet_name_pub)_hi_90_95_covid_topics $(start_date)


#############################################################################################
#############################################################################################


#############################################################################################
#############################################################################################

################################# Dimensions datasets topics ################################

dimensions_datasets_topics: dimensions_datasets classify_dimensions_datasets make_dimensions_datasets_topics

################################ Dimensions datasets  #########################################
#############################################################################################

#making dataset of dimensions publications into csv
colsA_dat = ["Dataset ID","Title","Source Linkout","Description","Date added","Dataset author"] #cols to be renamed
colsB_dat = ["sha","title","doi","abstract_x","publish_time","affiliations"] #default cols to follow through code 

datasets_filename = /tmp/tmpl41unnri/dimensions-covid19-export-2021-09-01-h15-01-02_datasets.csv
dimensions_datasets: 
	$(PYTHON_INTERPRETER) covid/data/make_dimensions_dataset.py data/raw/$(date_str)/ data/raw/$(date_str)/dims_$(sheet_name_dat)_raw_data.csv $(datasets_filename) False $(sheet_name_dat) $(colsA_dat) $(colsB_dat) 

## Classify Datasets to find only covid papers reduces file size by 100 fold
classify_dimensions_datasets: #requirements
	$(PYTHON_INTERPRETER) covid/data/classify_data.py data/raw/$(date_str)/dims_$(sheet_name_dat)_raw_data.csv data/processed/classified_dims_$(sheet_name_dat)_covid.csv $(yaml_path) $(colsB_dat)

## Run topic modelling over covid corpus
make_dimensions_datasets_topics:
	$(PYTHON_INTERPRETER) covid/models/topicmodeling/topic_generator.py $(yaml_path) classified_dims_$(sheet_name_dat)_covid dims_$(sheet_name_dat)_covid_topics $(start_date)

#############################################################################################
#############################################################################################


################################# Dimensions Clinical Trials topics #################################

dimensions_clin_trials_topics: dimensions_clin_trials classify_dimensions_clin_trials make_dimensions_clin_trials_topics

################################ Dimensions clinical trials  #########################################
#############################################################################################

#making dataset of dimensions publications into csv
colsA_clin_trials = ["Trial ID","Title","Source Linkout","Abstract","Publication date","Country of Sponsor/Collaborator","Funder Country"]#cols to be renamed
colsB_clin_trials = ["sha","title","doi","abstract_x","publish_time","affiliations","location"] #default cols to follow through code 

clinical_trials_filename = /tmp/tmpl41unnri/dimensions-covid19-export-2021-09-01-h15-01-02_clinical_trials.csv
dimensions_clin_trials: 
	$(PYTHON_INTERPRETER) covid/data/make_dimensions_dataset.py data/raw/$(date_str)/ data/raw/$(date_str)/dims_$(sheet_name_clin_trials)_raw_data.csv $(clinical_trials_filename) False $(sheet_name_clin_trials) $(colsA_clin_trials) $(colsB_clin_trials)

## Classify Datasets to find only covid papers reduces file size by 100 fold
classify_dimensions_clin_trials: #requirements
	$(PYTHON_INTERPRETER) covid/data/classify_data.py data/raw/$(date_str)/dims_$(sheet_name_clin_trials)_raw_data.csv data/processed/classified_dims_$(sheet_name_clin_trials)_covid.csv $(yaml_path) $(colsB_clin_trials)

## Run topic modelling over covid corpus
make_dimensions_clin_trials_topics:
	$(PYTHON_INTERPRETER) covid/models/topicmodeling/topic_generator.py $(yaml_path) classified_dims_$(sheet_name_clin_trials)_covid dims_$(sheet_name_clin_trials)_covid_topics $(start_date)

#############################################################################################
#############################################################################################

################################# Dimensions Grants topics #################################

dimensions_grants_topics: dimensions_grants classify_dimensions_grants make_dimensions_grants_topics

################################ Dimensions grants  #########################################
#############################################################################################

#making dataset of dimensions publications into csv
colsA_grants = ["Grant ID","Title","Source linkout","Abstract","Start date","Research organizations country","Funders country"]#cols to be renamed
colsB_grants = ["sha","title","doi","abstract_x","publish_time","affiliations","location"] #default cols to follow through code 

grants_filename = /tmp/tmpl41unnri/dimensions-covid19-export-2021-09-01-h15-01-02_grants.csv
dimensions_grants: 
	$(PYTHON_INTERPRETER) covid/data/make_dimensions_dataset.py data/raw/$(date_str)/ data/raw/$(date_str)/dims_$(sheet_name_grants)_raw_data.csv $(grants_filename) False $(sheet_name_grants) $(colsA_grants) $(colsB_grants) 

## Classify Datasets to find only covid papers reduces file size by 100 fold
classify_dimensions_grants: #requirements
	$(PYTHON_INTERPRETER) covid/data/classify_data.py data/raw/$(date_str)/dims_$(sheet_name_grants)_raw_data.csv data/processed/classified_dims_$(sheet_name_grants)_covid.csv $(yaml_path) $(colsB_grants)

## Run topic modelling over covid corpus
make_dimensions_grants_topics:
	$(PYTHON_INTERPRETER) covid/models/topicmodeling/topic_generator.py $(yaml_path) classified_dims_$(sheet_name_grants)_covid dims_$(sheet_name_grants)_covid_topics $(start_date)

#############################################################################################
#############################################################################################


#############################################################################################
#############################################################################################

################################# Relationship extraction   #################################

re_run_all_sources: re_sem_scholar re_dims_pub re_dims_clin_trials re_pre_proc

################################ Semantic scholar relationships #############################

re_sem_scholar:
	$(PYTHON_INTERPRETER) scripts/script_relation_extraction.py data/processed/classified_merged_covid.csv data/paperclassifier/classified_sem_scholar_covid_relation.csv $(yaml_path)

#############################################################################################
################################ Dims Publications 5, and 5_10 perc relationships ###########

re_dims_pub:
	$(PYTHON_INTERPRETER) scripts/script_relation_extraction.py data/processed/classified_dims_$(sheet_name_pub)_hi_95_100_covid.csv data/paperclassifier/classified_dims_$(sheet_name_pub)_hi_95_100_covid_relation.csv $(yaml_path)
	$(PYTHON_INTERPRETER) scripts/script_relation_extraction.py data/processed/classified_dims_$(sheet_name_pub)_hi_90_95_covid.csv data/paperclassifier/classified_dims_$(sheet_name_pub)_hi_90_95_covid_relation.csv $(yaml_path)


#############################################################################################
################################ Dims Clinical trials relationships #########################

re_dims_clin_trials:
	$(PYTHON_INTERPRETER) scripts/script_relation_extraction.py data/processed/classified_dims_$(sheet_name_clin_trials)_covid.csv data/paperclassifier/classified_dims_$(sheet_name_clin_trials)_covid_relation.csv $(yaml_path)

#############################################################################################
################################ pre-processing re relations to common keyword ##############

re_pre_proc:
	$(PYTHON_INTERPRETER) covid/models/relation/relation_preproc.py

#############################################################################################


################################# Dash data requirements ####################################

finalize_data: dash_file_structure final_data_pre_proc_location

################################ creating final folder with dash files ######################

dash_file_structure:
	######## forecasting data
	@echo ">>> creating covid folder"
	mkdir -p data/dashDatasets/covid/
	@echo ">>> Moving data confirmed cases USA"
	cp data/raw/$(date_str)/conf_USA.csv data/dashDatasets/covid/
	@echo ">>> Moving data death data USA"
	cp data/raw/$(date_str)/death_USA.csv data/dashDatasets/covid/
	@echo ">>> Moving data confirmed cases Global"
	cp data/raw/$(date_str)/conf_global.csv data/dashDatasets/covid/
	@echo ">>> Moving data death data global"
	cp data/raw/$(date_str)/death_global.csv data/dashDatasets/covid/
	@echo ">>> Moving data recovered cases global"
	cp data/raw/$(date_str)/recovered_global.csv data/dashDatasets/covid/
	######## nre data
	@echo ">>> creating nre folder"
	mkdir -p data/dashDatasets/nre/
	@echo ">>> Moving NRE Clinical trials"
	cp data/paperclassifier/pre_proc_classified_dims_Clinical\ Trials_covid_relation.csv data/dashDatasets/nre/
	mv data/dashDatasets/nre/pre_proc_classified_dims_Clinical\ Trials_covid_relation.csv data/dashDatasets/nre/Clinical_Trials.csv
	@echo ">>> Moving NRE Publications 90-95% importance"
	cp data/paperclassifier/pre_proc_classified_dims_Publications_hi_90_95_covid_relation.csv data/dashDatasets/nre/
	mv data/dashDatasets/nre/pre_proc_classified_dims_Publications_hi_90_95_covid_relation.csv data/dashDatasets/nre/Publications_hi_90_95.csv
	@echo ">>> Moving NRE Publications 95-100% importance"
	cp data/paperclassifier/pre_proc_classified_dims_Publications_hi_95_100_covid_relation.csv data/dashDatasets/nre/
	mv data/dashDatasets/nre/pre_proc_classified_dims_Publications_hi_95_100_covid_relation.csv data/dashDatasets/nre/Publications_hi_95_100.csv
	@echo ">>> Moving NRE Publications from semantic_scholar"
	cp data/paperclassifier/pre_proc_classified_sem_scholar_covid_relation.csv data/dashDatasets/nre/
	mv data/dashDatasets/nre/pre_proc_classified_sem_scholar_covid_relation.csv data/dashDatasets/nre/semantic_scholar.csv
	######## topic modelling data
	@echo ">>> creating nre folder"
	mkdir -p data/dashDatasets/topicmodeling/
	@echo ">>> Moving topic modelling Clinical trials"
	cp data/topicmodels/dims_Clinical\ Trials_covid_topics/pcf_dims_Clinical\ Trials_covid_topics_topic_data.csv data/dashDatasets/topicmodeling/
	mv data/dashDatasets/topicmodeling/pcf_dims_Clinical\ Trials_covid_topics_topic_data.csv data/dashDatasets/topicmodeling/Clinical_Trials.csv
	@echo ">>> Moving topic modelling Publications 90-95% importance"
	cp data/topicmodels/dims_Publications_hi_90_95_covid_topics/pcf_dims_Publications_hi_90_95_covid_topics_topic_data.csv data/dashDatasets/topicmodeling/
	mv data/dashDatasets/topicmodeling/pcf_dims_Publications_hi_90_95_covid_topics_topic_data.csv data/dashDatasets/topicmodeling/Publications_hi_90_95.csv
	@echo ">>> Moving topic modelling Publications 95-100% importance"
	cp data/topicmodels/dims_Publications_hi_95_100_covid_topics/pcf_dims_Publications_hi_95_100_covid_topics_topic_data.csv data/dashDatasets/topicmodeling/
	mv data/dashDatasets/topicmodeling/pcf_dims_Publications_hi_95_100_covid_topics_topic_data.csv data/dashDatasets/topicmodeling/Publications_hi_95_100.csv
	@echo ">>> Moving topic modelling Publications from semantic_scholar"
	cp data/topicmodels/sem_scholar_covid_topics/pcf_sem_scholar_covid_topics_topic_data.csv data/dashDatasets/topicmodeling/
	mv data/dashDatasets/topicmodeling/pcf_sem_scholar_covid_topics_topic_data.csv data/dashDatasets/topicmodeling/semantic_scholar.csv
	@echo ">>> Moving topic modelling Grants"
	cp data/topicmodels/dims_Grants_covid_topics/pcf_dims_Grants_covid_topics_topic_data.csv data/dashDatasets/topicmodeling/
	mv data/dashDatasets/topicmodeling/pcf_dims_Grants_covid_topics_topic_data.csv data/dashDatasets/topicmodeling/Grants.csv



#############################################################################################
################################ preprocess location of dash files ##########################

final_data_pre_proc_location:
	$(PYTHON_INTERPRETER) covid/data/final_data_pre_proc_location.py

#############################################################################################
#############################################################################################


## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 covid

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

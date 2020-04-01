# coronaBreakSuck2020

# files

- dataPreProcessing.py: reads csv files and preProcesses them using nltk, choose if to run files with text or only the metadata files, default only metadata
- nltkPreProc.py: module that contains text pre-processing methods
- dataReader.py: reads json format citations and converts them to csv format
- .gitignore: python gitignore file, create a Data folder to store data (ignored by git, all relative paths designed for this folder)
- requirements.txt: all dependecies needed to run the project

```
pip install -r requirements.txt
```

# Project Goal 
We want to build a knowledge discovery system for everyone (in particular the healthcare professionals) that can solve some of the questions in the Kaggle COVID-19 Open Research Dataset Challenge (CORD-19): https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge.

# Knowledge discovery system
This system is not to just identify information about COVID-19, but it is about retrieving top relevant information from a pool of literatures and providing ACTIONABLE insights to the viewers (particularly based on the perspective of doctors and scientists). That is to say, when we are deciding what to retrieve and show, we are thinking in term of what could be useful to clinicians. 

This system has two-fold. First, given a question (E.g., what are the risk factors for COVID-19 patients?), we will retrieve a subset of literatures that are most relevant to this question. Then, we will provide time-series plots to show how the ‘papers’ opinion/experimental-results’ change for the subject of interest along time (e.g., is age a risk factor?). Each time point will provide the corresponding papers (as well as the specific section) and timestamp. The data for time-series plots can be stratified by country, age group, etc because we notice that there are inconsistency about the risk factors from different country. 

# Resources


# Acknowledgement
This project is carried out by Panayiotis and Johnny, in hope to help the healthcare providers, scientific researchers, and data scientists in fighting against the Coronavirus outbreak. 

# Copywrite
MIT

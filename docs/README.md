# 12/23/20 Improvement plan 
1.	(Johnny) Write the descriptions for the NRE graphs and send them over to Vasilis
2.  (Later) For the 1st overview NRE distribution graph, it contains too much information
    - Suggestion 1: Aggregate data points into a monthly scale instead of day scale
    - Suggestion 2: shall we show top-3 strongest or top-3 weakest keywords only 
    - Ask David opinion
3.	(Panayiotis) Check the Mesh file; solve the grouping issues
4.	(Vasilis) Miscellaneous clean-up:
    - Styling the app & add datasets
    - Polish graphs: remove “publish time”, make y-axis and x-axis consistent, revise the title. For the 3rd plot, show only the top-5 keywords
    - (optional) Change the way we selected the keywords for second NRE plot


# 08/20/20 Improvement plan
Johnny: 
1. Implement the end-to-end script to extract all the relations from an excel sheet. (Done?)
2. Visualziation upgrade:
    1. Allow these visualization capabilities: view by paper (focus on papers with high impact factor only or by journal name), by region (where the publication is published). 
    2. Incorporate the relation extractor result for the interest keywords; as an indepedent app in DASH (Johnny)
    3. Make it pretty (Johnny)
    
Panayiotis:
1. Incorporate the external datasets (datasets, and clinical trials) from dimension.ai into the dash dashboard
2. Explore funding options for our projects.
3. Implement the "buy us a coffee" buttom. 

Vasili:
1. Clean up the UI
2. Integrate the MESH script to upgrade the interest.yaml file. 


# 08/03/20 Improvement plan

Panayiotis:
1. Explore the external datasets (datasets, and clinical trials) from dimension.ai
2. Explore funding options for our projects.
3. Implement the "buy us a coffee" buttom. 

Johnny: 
1. Implement a new function to extract relations of a list of keywords without respect to the covid-19 (through our papers sources). This is for Panayiotis new ideas for relationship extraction. 
    - 08/09/20 The function is now avaiable in coronaBreakSuck2020/covid/models/relation/extraction.py, `extract(text, e1=None, e2=None)`.
2. Visualziation upgrade:
    1. Allow these visualization capabilities: view by paper (focus on papers with high impact factor only or by journal name), by region (where the publication is published). 
    2. Incorporate the relation extractor result for the interest keywords; as an indepedent app in DASH (Johnny)
    3. Make it pretty (Johnny)

Vasili:
TBD (topic modeling)


# 07/12/20 Improvement plan
1. Focus on keywords that make more senes to doctors in topic modeling result presentation; filter away phrases that are not medical-related and not doctor-understandable in topic modeling
    1. Use abstract only for topic-modeling (Vasili, Panayiotis)
    2. Created a knowledge base that contains a lot of keywords that make senses to doctor
        1. Knowledge base: e.g., MeSH term (Vasili)
    3. The topic modeling keyword results should be presented after filtering by the knowledge base and then by whether the keywords have strong relationship (via the relation extractor) (Vasili)
2. Upgrade the interest.yaml to include more relevant medical keywords (e.g., treatment)
    1. from David (Panayiotis)
3. Visualziation upgrade:
    1. Allow these visualization capabilities: view by paper (focus on papers with high impact factor only or by journal name), by region (where the publication is published). 
    2. Incorporate the relation extractor result for the interest keywords; as an indepedent app in DASH (Johnny)
    3. Make it pretty (Johnny)
   

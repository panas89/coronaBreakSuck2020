# A utility module supporting data retreival
from geotext import GeoText
from difflib import SequenceMatcher
from .CONSTANTS import GEOTEXT_LOC_DICT

import geograpy3
import time



def extract_location(text, is_robust=True):
    """
    Given a string, extraction the location information (i.e., countries) from the string.
    
    - geograpy
        - geograpy1 or 2 did not work because of the errors in 
           installation: https://github.com/Corollarium/geograpy2 has installation issues. DID NOT USE IT
        - geograpy3 works: https://github.com/jmbielec/geograpy3
            - very slow, it can take >1 second or so to run large text
    - can GeoText as well: https://github.com/elyase/geotext
    - look like the geograpy3 is more robust to bad text (although not 100% correct) 
        that can detect more regions, but the precision may be low
    - may be we should combine both the use of geograpy and geotext
    - GeoText country abbreivation to country full name
        - https://github.com/elyase/geotext/blob/master/geotext/data/countryInfo.txt
        - Need to remove the "#" from the header column. Created a countryInfo_revised.txt and 
          save into the data folder
          
    To learn how to use it, please visit this notebook: main_kch_affiliation_extraction.ipynb
    
    :param text (string): string pbject
    :param is_robust (boolean): whether to use robust method or an approximation. Robust method will
                                use the voting by both the geography3 and GeoText method. Approximation
                                will use GeoText only. Generally speaking, we should always use Robust method if possible
    """
    def similar(a, b):
        """
        Compare how similar two strings are. E.g., similar("Apple","Appel")=0.8
        """
        return SequenceMatcher(None, a, b).ratio()
    
    # GeoText extraction
    gt_places = GeoText(text).country_mentions
    gt_countries = [GEOTEXT_LOC_DICT[c] for c in list(gt_places.keys())]
    
    if is_robust:
        # geograpy3 extraction
        gg_places = geograpy3.get_place_context(text = text)
        gg_countries = list(gg_places.country_cities.keys())

        # intersection of both Geography3 and GeoText extraction
        cs = []
        for c in gg_countries:
            for c2 in gt_countries:
                if similar(c,c2)>0.8:
                    cs.append(c)
    else:
        cs = gt_countries
    return ",".join(cs)















import pandas as pd
import numpy as np
from covid.models.paperclassifier.paperclassifier import PaperClassifier

class FrontPaperClassifier(PaperClassifier):
    
    
    def __init__(self, km_path):
        
        """
        This class helps us extract and visualize the information carried out by PaperClassifier.
        
        The corresponding information is a nested dictionary of:
        outer keys: classes (e.g. Risk factors)
        inner keys: subclasses (e.g. gender)
        inner values: list of keywords (e.g. male, female, sex, etc)
        
        ----------
        
        Inputs:
          - param km_path (string): file path for the knowledge map of subject of interests 
          
        Attributes:
          - km: yaml dict
          - classes: classes
          - df: a dataframe with multi-index=(class, subclass) and values=keywords
        """
    
        
        # Must Instantiate Parent class first
        super().__init__(km_path=km_path) # inherits parent's attribute self.km (yaml dict)
        
        self.classes = list(self.km.keys())
        self.df = self._create_df()
             
            
#----------------------- Public Methods ---------------------------
            
            
    def get_class_dict(self, class_name):
                
        return self.km[class_name]
        
    def get_subclasses(self, class_name):
                
        return list(self.km[class_name].keys())
    
    def get_parent_class(self, subclass):
        
        assert isinstance(subclass, str), f'{subclass} is not a string.'
        
        for class_name in self.classes:
            if self.get_class_dict(class_name).get(subclass):
                return class_name
            
        raise KeyError(f"{subclass} is not a valid subclass.")

    def get_keywords(self, subclass):
        
        parent_class = self.get_parent_class(subclass)

        return self.get_class_dict(parent_class)[subclass]
        
    
            
#----------------------- Private Methods ---------------------------

    def _create_df(self):
        
        multi_index = []
        data = []
        for class_name in self.classes:
            for subclass in self.get_subclasses(class_name):
                data.append(" ".join(self.get_class_dict(class_name)[subclass]))
                multi_index.append((class_name, subclass))

        df = pd.DataFrame(data = data,
                          index = pd.MultiIndex.from_tuples(multi_index),
                          columns = ['keywords'] 
                         )
        df.index.set_names(['class', 'subclass'], inplace=True)
        
        return df
    
    
    
     
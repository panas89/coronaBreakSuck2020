import opennre
import spacy

class RelationExtractor(object):
    def __init__(self):
        """
        Extraction class for relations among entities.
        
        - There are two model available: wiki80_bert_softmax or wiki80_cnn_softmax. 
        The first one is better. The model is from the supervised relation 
        extraction: http://opennre.thunlp.ai/#/sent_re. 
        - We use an additiona NER classifier to classify 
        entities if the entities are not provided.
        - Among all the relations, only the "has part (P527)" is useful. 
        """
        self.model = opennre.get_model('wiki80_bert_softmax')
        self.nlp = spacy.load("en_core_web_sm") # package
        
    def extract(self, text, e1=None, e2=None):
        """
        Extract entity relations
        
        :param text (string): the sentence that contains the entities
        :param e1 (string): entity 1
        :param e2 (string): entity 2
        """
        # preprocess
        text = text.lower()
        e1 = e1.lower() if e1 else e1
        e2 = e2.lower() if e2 else e2
        
        # if entities are None, then automatically detect it
        if not e1 or not e2:
            doc = self.nlp(text)
            if len(doc.ents) < 2:
                return None
            
            # assign head and tails
            e1 = doc.ents[0].string.strip()
            e2 = doc.ents[1].string.strip()
            
        # get the index of the entity
        index1 = text.find(e1)
        index2 = text.find(e2)
        
        # relation extraction
        relation = self.model.infer({'text': text, 
                                     'h': {'pos': (index1, index1+len(e1))}, 
                                     't': {'pos': (index2, index2+len(e2))}
                                    })
        return e1, e2, relation
        
      

            







from tqdm import tqdm_notebook as tqdm
import opennre
import spacy
import yaml

class RelationExtractor(object):
    def __init__(self, km_path='interest.yaml'):
        """
        Extraction class for relations among entities.
        
        - There are two model available: wiki80_bert_softmax or wiki80_cnn_softmax. 
        The first one is better. The model is from the supervised relation 
        extraction: http://opennre.thunlp.ai/#/sent_re. 
        - We use an additiona NER classifier to classify 
        entities if the entities are not provided.
        - Among all the relations, only the "has part (P527)" is useful. 
        
        :param km_path (string): file path for the knowledge map of subject of interests 
        """
        self.model = opennre.get_model('wiki80_bert_softmax')
        self.nlp = spacy.load("en_core_web_sm") # package
        
        # Load the basic synonyms that we define
        with open(km_path) as f:
            self.km = yaml.load(f, Loader=yaml.FullLoader)
                       
        
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
        else:
            texts = text.split()
            if e1 not in texts or e2 not in texts:
                return None
            
        # get the index of the entity
        index1 = text.find(e1)
        index2 = text.find(e2)
        
        # relation extraction
        relation = self.model.infer({'text': text, 
                                     'h': {'pos': (index1, index1+len(e1))}, 
                                     't': {'pos': (index2, index2+len(e2))}
                                    })
        return e1, e2, relation
        

    def extract_all(self, df):
        """
        Extract the covid and each keyword from a subclass relationship.
        
        :param df (Dataframe): the original dataframe that contains the info (abstract, 
                                title, and keywords that present) 
        """
        # get the covid common name
        bns = self.km['disease_name']['disease_common_name']['kw']
        
        # extract relationships between the key and the covid
        relations = []
        for i in tqdm(range(0, df.shape[0])):
            df_i = df.iloc[i]
            abstract = df_i['abstract']
            
            # preprocess-simple
            abstract = abstract.lower()
            
            # keywords
            kws = df_i['keywords']
            if kws is None or kws == '0' or kws == 0:
                relations.append(None)
                continue
            
            # identify what covid keyword is in the abstract.
            # We identify the first occurance. Technically, 
            # we should find the words that are closed to the keyword
            # within a distance
            bns_this = [bn for bn in bns if bn in abstract]
            if len(bns_this) < 1:
                relations.append(None)
                continue
            
            # loop each keyword
            relation_this = []
            for kw in kws.split(','):
                # Idententify the covid keyword that has the shortest distance
                # with respect to the kw
                bn_closest = self._get_closest_bn(abstract, kw, bns_this)
                
                # extraction the relation
                # There are couple of relations that are interesting
                relation = self.extract(abstract, bn_closest, kw)
                relation_this.append(relation)
            relations.append(relation_this)
        return relations

           
    def _get_closest_bn(self, abstract, kw, bns):
        """
        Get the closest covid base name with respect to keyword.
        """
        bn_closest = bns[0]
        closest_dist = 99999999
        
        for bn in bns:
            dist = abs(abstract.index(bn) - abstract.index(kw))
            if dist < closest_dist:
                bn_closest = bn
                closest_dist = dist
        return bn_closest

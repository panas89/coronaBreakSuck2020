import yaml
import pickle
import xml.etree.ElementTree as et
from collections import defaultdict

# ####################################################################################################

class MeshSearch:

    def __init__(self, mesh_xml_file=None, id2kws_path=None, kw2id_path=None):

        # Create mesh dicts
        if mesh_xml_file:
            self.id2keywords, self.kw2id = self._create_mesh_dicts(mesh_xml_file)
        elif id2kws_path and kw2id_path:
            self.id2keywords, self.kw2id = self._load_mesh_dicts(id2kws_path, kw2id_path)
        else:
            print("Enter either 'mesh_xml_file' or ('id2kws_path', 'kw2id_path') ")

    # ----------------------------------------------------------------------------------------------------
    def search_mesh(self, kw, verbose=False) -> list:

        try:
            return self.id2keywords[self.kw2id[kw]]
        except:
            if verbose:
                print('Term does not exist in MeSH dict')
            return 

    # ----------------------------------------------------------------------------------------------------
    def _create_mesh_dicts(self, mesh_xml_file) -> (dict, dict):

        # Instantiate xml parser
        self.xtree = et.parse(mesh_xml_file)
        self.root = self.xtree.getroot()

        # Create output dicts
        id2keywords = defaultdict(list)
        kw2id = {}
        # Iterate over all MeSH nodes
        for node in self.root.iter('DescriptorRecord'):
            # Get the id of the MeSH node
            ui = node.find('.//DescriptorUI').text
            # Append all the related keywords to dict
            for term in node.findall('.//ConceptList/Concept/TermList/Term'):
                kw = term.find('String').text.lower()
                id2keywords[ui].append(kw)
                kw2id[kw] = ui

        return id2keywords, kw2id 

    # ----------------------------------------------------------------------------------------------------
    def _load_mesh_dicts(self, id2kws_path, kw2id_path) -> (dict, dict):
        
        with open(id2kws_path, 'rb') as f:
            id2keywords = pickle.load(f)

        with open(kw2id_path, 'rb') as f:
            kw2id = pickle.load(f)

        return id2keywords, kw2id

# ####################################################################################################
 
def mesh_extension(mesh_obj, yaml_path) -> dict:
    
    # Create dict from yaml
    with open(yaml_path) as f:
        yml = yaml.load(f, Loader=yaml.FullLoader)
    
    # Iterate over class/subclass/kws and extend kws
    for cls in yml:
        for subcls in yml[cls]:
            new_kws = []
            for kw in yml[cls][subcls]['kw']:
            
                # if mesh finds related terms to kw
                # add them to new_kws (incl. kw)
                mesh_kws = mesh_obj.search_mesh(kw)
                if mesh_kws:
                    new_kws.extend(mesh_kws)
                else:
                    new_kws.append(kw)
            
            yml[cls][subcls]['kw'] = new_kws
            
    return yml

# ####################################################################################################

if __name__ == '__main__':

    # Define file paths
    id2kws_path = 'mesh_id2keywords.pkl'
    kw2id_path = 'mesh_kw2id.pkl'
    yaml_path = './interest.yaml'

    # Instantiate MeshSearch object
    mesh_obj = MeshSearch(id2kws_path=id2kws_path, kw2id_path=kw2id_path)

    # Create mesh extension of kws in yaml
    yml_dict = mesh_extension(mesh_obj, yaml_path)

    # Save mesh-extended dict as yaml
    with open('./mesh_interest.yml', 'w') as outfile:
        yaml.dump(yml_dict, outfile, default_flow_style=True)
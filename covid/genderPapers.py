import nltk 
from nltk.corpus import wordnet

def getSynonymsAntonymns(word):
    synonyms = [] 
    antonyms = [] 

    for syn in wordnet.synsets("female"): 
        for l in syn.lemmas(): 
            synonyms.append(l.name()) 
            if l.antonyms(): 
                antonyms.append(l.antonyms()[0].name())

    return set(synonyms),set(antonyms)

print(getSynonymsAntonymns("male")) 


############################### corona virus string match
# keywords from here https://coviz.apps.allenai.org/bc5cdr/?d=bc5cdr&l=40&ftm=11444


pattern_COVID = 'respiratory tract infection |virus infection |respiratory syncytial virus | \
                 lipopolysaccharide |death |acute respiratory distress syndrome |acute respiratory failure | \
                 H1N1 viral infection |rubella virus infection |influenza virus infection |human immunodeficiency virus | \
                 irritation of the respiratory tract |Zika Virus Infection |Ebola and Zika virus infection | \
                 porcine reproductive and respiratory syndrome |TAP |influenza virus A |Thrombocytopenia Syndrome Virus Infection | \
                 SARS-CoV-2 |respiratory syncitial virus |skin or mucous membrane lesions |upper respiratory infection | \
                 H5N1 viral infection |herpes simplex virus type 1 |human immunodeficiency virus type 1 |gastrointestinal viral infection | \
                 reproductive and respiratory syndrome virus infection |porcine reproductive and respiratory syndrome virus | \
                 hepatitis A virus |acquired immunodeficiency syndrome |parainfluenza virus 3 | \
                 nosocomial viral respiratory infections |coronavirus OC43 infection |IFN |H3N2 virus infection | \
                 dsRNA |dsDNA |long QT syndrome |liver cell necrosis |latent TB infection |Pulmonary Coronavirus Infection | \
                 Dengue virus Type |neurotropic coronavirus virus |Leukocyte adhesion deficiency II syndrome | \
                 Human T-cell leukemia virus type 1 |Human T-cell leukemia virus type | \
                 infection of the central nervous system |infection of the pulmonary parenchyma'

pattern_COVID = pattern_COVID.replace(' |','|(?i)')
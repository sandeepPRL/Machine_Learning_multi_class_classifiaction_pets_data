from __future__ import division
import json

import nltk
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import NMF
# from collections import Counter
import re
from collections import Counter
# import wordsegment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# import operator
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from tfidf import Tfidf
from cosine_similarity import Cosine
from non_negative import NMF



def safe_div(x, y, eps=1e-16):
    x = x.copy()
    y = y.copy()
    mask = np.logical_or(np.abs(x) < eps, np.abs(y) < eps)
    y[mask] = 1
    x[mask] = 1
    return x / y

def unsafe_div(x, y, eps=1e-16):
    with np.errstate(divide='ignore'):
        return x / y


class BadDevSetError(ValueError):
    pass


from sklearn.metrics import precision_recall_fscore_support

def prec_recall_fscore(y_pred, y_target, beta=1):
    # print('prec_recall_fscore called....')
    y_pred = y_pred.astype(np.bool)
    y_target = y_target.astype(np.bool)

    L = y_pred.shape[1]
    tp = np.zeros([L], dtype=np.int32)
    fp = np.zeros([L], dtype=np.int32)
    fn = np.zeros([L], dtype=np.int32)
    tn = np.zeros([L], dtype=np.int32)
    for i in range(y_pred.shape[0]):
        tp += np.logical_and(y_pred[i], y_target[i])
        fp += np.logical_and(y_pred[i], np.logical_not(y_target[i]))
        fn += np.logical_and(np.logical_not(y_pred[i]), y_target[i])
        tn += np.logical_and(np.logical_not(y_pred[i]), np.logical_not(y_target[i]))
    if np.count_nonzero(np.equal(tp + fn, 0)) > 0:
        raise BadDevSetError("Bad dev set")
    recall = unsafe_div(tp, tp + fn)
    precision = unsafe_div(tp, tp + fp)
    precision[(tp + fp) == 0] = 1.0

    #     no_positives = np.logical_and(tp == 0, fp == 0)
    #     no_positive_reports = np.logical_and(tp == 0, fn == 0)

    fscore = (1 + beta ** 2) * unsafe_div(precision * recall, beta ** 2 * precision + recall)
    fscore[np.equal(beta ** 2 * precision + recall, 0)] = 0.0

    #     # If there are no positives XOR no reports of positives, that's bad
    #     fscore[np.logical_xor(no_positives, no_positive_reports)] = 0.0
    #     # If there are no positives and no reports of positives, that's good
    #     fscore[np.logical_and(no_positives, no_positive_reports)] = 1.0
    assert np.count_nonzero(np.logical_not(np.isfinite(fscore))) == 0, "%s %s %s" % (precision, recall, fscore)
    return precision, recall, fscore


HOLD_OUT = 20

class ModelPlayground():

    def __init__(self, ignored={"diet"}):

        self.training_set2 = list(self.load_set5())
        self.training_set1 = self.load_set()
        # print("size of training set 2................")
        # print(self.training_set2)


        self.negations_dic = {"isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
                         "haven't": "have not", "hasn't": "has not", "hadn't": "had not", "won't": "will not",
                         "wouldn't": "would not", "don't": "do not", "doesn't": "does not", "didn't": "did not",
                         "can't": "can not", "couldn't": "could not", "shouldn't": "should not",
                         "mightn't": "might not",
                         "mustn't": "must not",
                         'abnormallymy': 'abnormally my', 'abouthe': 'about he', 'abouti': 'about i',
                         'acidmy': 'acid my',
                         'actionhe': 'action he', 'activei': 'active i', 'againdye': 'again dye', 'agobig': 'ago big',
                         'almostjackson': 'almost jackson', 'alsoii': 'also i', 'alsomy': 'also my',
                         'anymoremy': 'anymoremy',
                         'aroundmy': 'around my', 'asleep': 'a sleep', 'aspirini': 'aspirin i', 'awhilewe': 'awhile we',
                         'babymy': 'baby my', 'bei': 'be i', 'bettermy': 'better my', 'biggermy': 'bigger my',
                         'diarrhoeacat': 'dairrhea cat',
                         'diarrhoeami': 'dairrhea mi', 'diarrhoeawhat': 'dairrhea what',
                        'underweightshe': 'underweight she', 'healsshe': 'heals she'}
        self.WORDS = Counter(self.words(open('data.jsons').read()))
        self.COUNTS = Counter(self.WORDS)
        # print(COUNTS)
        N = sum(self.WORDS.values())

        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.Diarrhea = ["Vomiting", "Blood", "Depression", "Fever", "Weakness", "Lethargy", "weight", "loss",
                         "loose", "watery", "stool", "appetite", "Dehydration", "Depression",
                         "Increased", " bowel", "sounds", "flatulence", "Mucus", "Constipated",
                         "Urgency","defecating"]
        self.Vomiting = ["Lethargy", "loss", "appetite", "Frequency", "Diarrhea", "Dehydration",
                         "Blood", "vomit", "Weight", "Increase", "decrease", "thirst", "urination",
                         "Drinking", "change", "toilet", "patterns", "Weakness","feeling", "anxious",
                         "restlessness", "drooling", "swallowing"]
        self.Ear = ["Scratching", "Brown", "yellow", "bloody", "discharge", "Odor", "ear", "Swelling",
                    "Hair", "loss", "around", "shaking", "head", "tilt", "Unusual", "eye", "movements",
                    "Walking", "circles", "Hearing", "vomiting", "nausea", "unequally", "sized", "pupils",
                    "redness", "discharge", " grey", "bulging", "eardrum", "facial", "nerve", "damage",
                    "scaly", "skin", "whining", "pawing", "itching", "Aggressiveness", "seeds", "around",
                    "Inflammation", "Pain", "reluctance", "chewing", "Scabs", "crusting",
                    "Leaning", "side", "affected", "balance"]
        self.Itching = ["Detoxification", "Scratching", "Licking", "Biting", "Chewing", "Selftrauma",
                        "Inflammation", "skin", "Hair", "loss", "alopecia", "Bleeding",
                        "brownish", "red", "saliva", "stain", "appear", "fur", "licking", "Chewing", "Redness",
                        "Scaling", "skin", "Odor", "Oozing", "postules"]
        self.Dentistry = ["Decay", "plaque", "gum", "inflammation", "chewing", "hard", "objects", "Jaw", "Misalignment",
                          "tooth","tartar", "Bad", "Breath", "Red", "swollen", "bleeding","Nasal", "Discharge", "Sneezing",
                          "Teeth", "wear", "drooling", "Retained", "Baby"]
        self.Ticks = ["Arthritis", "Swelling", "joints", "lameness", "itchiness", "red", "skin", "inflammation", "fever",
                      "loss", "appetite", "Abdominal", "pain", "lethargy", "depression", "gastrointestinal", "illnesses",
                      "impair", "immune", "function", "anemia", "rash", "fatigue", "Nose", "bleed", "Dehydration", "Cough",
                      "Seizure"]
        self.Fleas = ["Droppings", "flea", "dirt", "coat", "Flea", "eggs", "environment",  "Allergic", "dermatitis",
                      "scratching", "licking", "biting", "Hair", "loss", "Scabs", "hot", "spots", "Pale", "gums",
                      "Chewing", "Rubbing", "Skin", "abrasions", "sores", "red", "weeping ", "bloody", "Pus", "oozing",
                      "pyoderma", "alopecia", "Pruritus", "Restlessness", "anemia", "Tapeworm", "segments", "around",
                      "dog", "anus","stool", "larvae" ]
        self.Depression = ["Tail", "chasing", "wagging", "Licking", "chewing", "Grooming", "Excessive", "Sleeping",
                           "Loss", "Appetite", "Repetitive", "Behaviors", "Acts", "Strange", "Isolation", "Aggression",
                           "Thundercap", "stopped", "working"]
        self.Worms = ["abdominal", "pain", "diarrhea", "nausea", "gas", "bloating", "fatigue", "weight", "loss", "tenderness",
                      "Visible", "worms", "eggs", "faeces", "fur", "area", "around", "dog's", "rear",
                      "Scratching", "rubbing", "Vomiting", "Weakness", "increased", "appetite",
                      "constant", "hunger", "Dull", "coat", "Swollen", "belly", "stool", "Anemia", "Potbelly", "Cough",
                      "Constipation"]
        self.Cough = ["strong", "cough", "honking", "sound", "runny", "nose", "sneezing", "reverse", "lethargy",
                      "loss of appetite", "low", "fever", "eye", "discharge", "respiratory", "infection", "kennel",
                      "pneumonia", "heart", "disease", "collapsing", "trachea", "Nasal", "discharge", "watery"]
        self.Urinary = ["Submission", "anxiety", "Excitement", "Marking", "territory", "Inability",
                        "passing", "small", "amount", "urine", "Blood", "cloudy", "Fever", "Loss", "bladder", "control",
                        "dribbling", "Straining", "Crying", "outin", "pain", "Soiling", "inappropriate", "places", "licking",
                        "opening", "Strong", "odor", "Lethargy", "Vomiting", "Changes", "appetite", "Weight", "loss", "Severe", "backpain",
                        "Increased", "water", "consumption", "Frequent", "Breaking", "housetraining"]
        self.Atesomething = ["Vomiting", "Diarrhea", "Loose", "Stool", "Chronic", "bad", "breath", "Straining", "bowel", "movement",
                             "Unable", "bowels", "Black" , "Dark" , "tarry", "stools", "Burping", "drooling", " contractions",
                              "Gastrointestinal", "lockage", "Choking", "Sneezing", "coughing", "Lethargy", "Collapse"]
        self.Eyes = ["Impaired", "vision", "Squinting", "Pawing", "rubbing", "Conjunctivitis",
                     "eye", "redness", "Swelling", "Pus", "discharge", "mucus" ,
                     "rot", "changes", "black", "color", "Dry", "drainage", "Excessive", "blinking",
                     "Cloudiness", "Swollen", "conjunctival", "blood", "vessels", "Chemosis", "Prominent", "nictitans",
                     "Corneal", "Puffy", "lids", "Stringy", "Watery", "Eyelids", "stick", "together", "scratching", "mois-tlooking", "sneezing",
                     "nasal", "High", "pressure", "Dilated", "pupil", "Vision", "loss",
                     "inflammation", "Sensitivity", "light", "abrasion", "ulcers",
                     "Cries", "yelps", "pain", "Depression", "Lethargy"]
        self.Vaccination = ["Vaccinations", "vaccined", "vaccines", "eye", "discharge", "nose", "conjunctivitis",
                            "vomiting", "diarrhea ", "loss", "appetite",
                            "watery", "feces", "blood", "mucous", "often", "foul", "odor", "spasms ", "seizures", "paralysis",
                            "eruptions", "around", "mouth", "swelling", "feet", "face", "pneumonia ", "skin", "eruptions",
                            "emaciation", "pancreatitis", "inflammatory", "bowel", "disease", "gastrointestinal", "lymphoma",
                            "epilepsy", "brain", "tumors", "itching ", "eruptions", "restlessness", "viciousness",
                            "avoidance", "company", "unusual", "affection", "desire", "travel",
                            "inability", "restrained", "self", "biting ", "strange", "crying", "howling", "gagging ",
                            "staring ", "swallowing", "wood", "stones", "inedibles",
                            "destruction", "blankets", "clothing", "increased", "sexual", "desire", "disturbed", "heart", "function",
                            "excitement", "jerky", "breathing", "Fever", "Tenderness", "Hives", "Swollen", "nymph", "nodes"]
        self.Injuries = ["Antisocial", "behaviour", "aggressive", "Loss", "appetite", "Dehydration", "Urination",
                         "Excessive", "sleeping", "yelping", "growling ", "snarling", "howling", "licking", "grooming",
                         "Panting", "Shallow", "breathing", "Restlessness", "Swelling", "paws", "legs", "face",
                         "Shaking ", "trembling", "limping ", "hobbling", "lethargy",
                         "Bloodshot", "dilated", "constricted", "pupils", "Demeanor"]
        self.Lumps = ["Swelling", "Raised", "fluid", "filled", "bumps", "cat", "skin", "Oozing", "ruptured", "bumps",
                      "scratching", "itching", "inflammation", "Bleeding",
                      "Development", "cysts", "head", "neck", "body", "upper", "legs",
                      "Release", "grayish", "white", "brown", "discharge", "cheesy", "consistency", "Lethargy",
                      "Hair", "loss", "site", "Black", "putrid", "smelling", "canine", "fever", "chewing",
                      "hair", "degranulation", "stomach", "ulceration", "redness", "Vomiting", "appetite",
                      "diarrhea"]

        labels = Counter(label for utt in self.training_set2 for label in utt["labels"])
        labels, _ = zip(*sorted(labels.items(), key=lambda x: (-x[1], x[0])))
        labels = sorted(labels)
        # print("labels is ")
        print(len(labels))

        rlabels = {label: i for i, label in enumerate(labels)}
        self.labels = labels
        self.rlabels = rlabels
        print('....................')
        print(self.rlabels)
        # # labels = Counter(label for utt in self.training_set2 for label in utt["labels"])
        # labels = Counter(label for utt in self.training_set2 for label in utt["labels"])
        #
        # # labels = Counter(self.get_labels5(self.training_set2))
        # labels, _ = zip(*sorted(labels.items(), key=lambda x: (-x[1], x[0])))
        # labels = sorted(labels)
        # print("labels is")
        # print(len(labels))
        # rlabels = {label: i for i, label in enumerate(labels)}
        # self.labels = labels
        # self.rlabels = rlabels
        # print("rlabels count.....................")
        # print(len(self.rlabels))

    def Alllist(self):
        return (set().union(self.Diarrhea, self.Ear, self.Lumps, self.Eyes, self.Injuries, self.Vaccination,
                            self.Atesomething, self.Urinary, self.Cough, self.Itching, self.Dentistry, self.Depression,
                            self.Fleas, self.Ticks, self.Worms, self.Vomiting))

    def listtostring(self, list):
        lkm = ' '.join(list)
        # print(lkm)
        asd = (lkm.lower())
        wer = asd.split()
        # print(wer)
        usa = []
        for w in wer:
           usa.append(w)
        # # print(usa)
        # stp=model.stop_words(usa)
        # print(len(stp))
        # cdm=model.unique_list(stp)
        # # print('unique cdm',cdm)
        # print(len(cdm))
        # tf=model.tfidf(cdm)
        # print('unique tfidf',tf)
        # ft=model.tfidf(usa)
        return usa
        # print('tf for notunique',ft)

    def load_set(self):
        data_dict = {'unknown': []}
        with open('data.jsons') as f:
            d = f.readline()
            while d:
                data = json.loads(d)
                labels = data['labels']
                for i in labels:
                    if i not in data_dict.keys():
                        data_dict[i] = []
                    data_dict[i].append(str(data['concern']))
                # for labels,concern in data_dict.items():
                #     print(concern)
                d = f.readline()
        return data_dict




    def load_set4(self):
        # data_dict = {'unknown': []}
        with open('data.jsons','r') as f:
            d = f.readline()
            while d:
                data = json.loads(d)
                labels = data['labels']
                # for i in labels:
                #     # if i not in data_dict.keys():
                #         data_dict[i] = []
                #     data_dict[i].append(labels)
                # # for labels,concern in data_dict.items():
                # #     print(concern)
                d = f.readline()
        return labels

    def get_labels5(self,dataset):
        return [utt["labels"] for utt in dataset]

    def load_set5(self):
        with open('data.jsons' , 'r') as f:
            for line in f:
                yield json.loads(line)

    def get_labels(self, dataset=None, multilabel=True):
        # print("get labels called.....................")
        y = np.zeros([985, len(self.labels)], dtype=np.int32)

        for i, utt in enumerate(self.training_set2):
            # print("in for loop............")
            for label in utt["labels"]:


                if label in self.rlabels:
                    # print("inside if condition...........")
                    # print(self.rlabels[label])
                    # print(i)
                    y[i,self.rlabels[label]] = 1
        return y


    def load_set1(self):
        data_dict1 = []
        with open('data.jsons') as f:
            for line in f:
                data = json.loads(line)
                labels = data['labels']
                data_dict1.append(data['concern'])
        return data_dict1

    def get_label_set(self, lable):
        return (self.training_set1[lable])

    def negation_hand(self, dataset):

        neg_pattern = re.compile(r'(' + '|'.join(self.negations_dic.keys()) + r')')

        return neg_pattern.sub(lambda x: self.negations_dic[x.group()], dataset)

    def tokenization(self, data):
        words_split = self.negation_hand(data)
        word_tokens = word_tokenize(words_split)
        return word_tokens

    def stop_words(self, token):
        stop_wrd = set(stopwords.words('english'))
        # filter_sentence = [w for w in token if not w in stop_wrd]
        filter_sentence = []
        for w in token:
            if w not in stop_wrd:
                filter_sentence.append(w)

        return filter_sentence

    def regular_exp(self, stopword):
        cleaned = [word for word in stopword if word.isalpha()]
        return cleaned

    def port_stem(self, regular_expression):
        port_stem = PorterStemmer()
        stem = []
        for w in regular_expression:
            stem.append(port_stem.stem(w))
        return stem


    def unique_list(self, stem):
        ulist = []

        [ulist.append(x) for x in stem if x not in ulist]
        return ulist

    def words(self,text): return re.findall(r'\w+', text.lower())

    def P(self,word):
        "Probability of `word`."
        N = sum(self.WORDS.values())
        return self.WORDS[word] / N

    def correction(self,word):
         "Most probable spelling correction for word."
         return max(self.candidates(word), key=self.P)

    def candidates(self,word):
         "Generate possible spelling corrections for word."
         return (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word])

    def known(self,words):
         "The subset of `words` that appear in the dictionary of WORDS."
         return set(w for w in words if w in self.WORDS)

    def edits1(self,word):
        "All edits that are one edit away from `word`."
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self,word):
         "All edits that are two edits away from `word`."
         return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    def correct(self,word):
        "Find the best spelling correction for this word."
    # Prefer edit distance 0, then 1, then 2; otherwise default to word itself.
        candidates = (self.known(self.edits0(word)) or
                  self.known(self.edits1(word)) or
                  self.known(self.edits2(word)) or
                  [word])
        return max(candidates, key=self.COUNTS.get)
    def known(self,words):
          "Return the subset of words that are actually in the dictionary."
          return {w for w in words if w in self.COUNTS}

    def edits0(self,word):
         "Return all strings that are zero edits away from word (i.e., just word itself)."
         return {word}

    def edits2(self,word):
          "Return all strings that are two edits away from this word."
          return {e2 for e1 in self.edits1(word) for e2 in self.edits1(e1)}
    def edits1(self,word):
          "Return all strings that are one edit away from this word."
          pairs      = self.splits(word)
          deletes    = [a+b[1:]           for (a, b) in pairs if b]
          transposes = [a+b[1]+b[0]+b[2:] for (a, b) in pairs if len(b) > 1]
          replaces   = [a+c+b[1:]         for (a, b) in pairs for c in self.alphabet if b]
          inserts    = [a+c+b             for (a, b) in pairs for c in self.alphabet]
          return set(deletes + transposes + replaces + inserts)

    def splits(self,word):
          "Return a list of all possible (first, rest) pairs that comprise word."
          return [(word[:i], word[i:])
                 for i in range(len(word)+1)]


    def correct_text(self,text):
          "Correct all the words within a text, returning the corrected text."
          return re.sub('[a-zA-Z]+', self.correct_match, text)

    def correct_match(self,match):
          "Spell-correct word in match, and preserve proper upper/lower/title case."
          word = match.group()
          return self.case_of(word)(self.correct(word.lower()))

    def case_of(self,text):
           "Return the case-function appropriate for text: upper, lower, title, or just str."
           return (str.upper if text.isupper() else
            str.lower if text.islower() else
            str.title if text.istitle() else
            str)
    def pdist(self,counter):
           "Make a probability distribution, given evidence from a Counter."
           N = sum(counter.values())
           return lambda x: counter[x]/N

# P = pdist(COUNTS)
    def Pwords(self,words):
           "Probability of words, assuming each word is independent of others."
           return self.product(self.P(w) for w in words)

    def product(self,nums):
          "Multiply the numbers together.  (Like `sum`, but with multiplication.)"
          result = 1
          for x in nums:
                result *= x
          return result
    def memo(self,f):
               "Memoize function f, whose args must all be hashable."
               cache = {}
               def fmemo(*args):
                    if args not in cache:
                         cache[args] = f(*args)
                    return cache[args]
               fmemo.cache = cache
               return fmemo
    def splits2(self,text, start=0, L=20):
                "Return a list of all (first, rest) pairs; start <= len(first) <= L."
                return [(text[:i], text[i:])
                           for i in range(start, min(len(text), L)+1)]

    def segment(self,text):
           "Return a list of words that is the most probable segmentation of text."
           if not text:
                return []
           else:
                candidates = ([first] + self.segment(rest)
                             for (first, rest) in self.splits2(text, 1))
                return max(candidates, key=self.Pwords)
    def get_segment_list(self,segment):
        seg = []
        for w in segment:
            wordseg = self.segment(w)
            for k in wordseg:
                seg.append(k)
        return seg



        return tfidf.transform(testset)
    def count_vec(self,uniq,concerns):

        vec = CountVectorizer(stop_words='english',max_features=200, vocabulary=uniq)
        vec.fit(concerns)

        cv_matrix = vec.fit_transform(concerns).toarray()
        return cv_matrix
    def accuracy(ypred, y):
            print('accuracy called...')
            ypred = ypred.astype(np.int32)
            if len(ypred.shape) < 2:
                # Single class classification
                return np.equal(y, ypred.squeeze(1)).astype(np.int32).sum() / y.shape[0]
            correct = 0
            y = y.astype(np.int32)
            for i in range(y.shape[0]):
                if np.count_nonzero(y[i]) == 0:
                    # Target is N/A
                    if np.count_nonzero(ypred[i]) == 0:
                        # No classes were predicted, 1 point
                        correct += 1
                elif np.count_nonzero(ypred[i]) > 0:
                    # Target is not N/A
                    # Choose one of the selected classes uniformly at random
                    yp = (ypred[i] > 0).astype(np.float32)
                    idx = np.random.choice(ypred[i].shape[0], size=[], p=yp / np.sum(yp))
                    # If the values are the same, correct answer
                    if y[i, idx] == ypred[i, idx]:
                        correct += 1
            return correct / y.shape[0]

    @staticmethod
    def average_f1(ypred, y):
            # print('average f1 called....')
            prec, recall, f1 = prec_recall_fscore(ypred, y)
            return f1.mean()

    @staticmethod
    def hold_out(X, y, held_out=HOLD_OUT):
            print('hold out called....')
            if isinstance(X, list):
                N = len(X)
                print("if hold",N)
            else:
                N = X.shape[1]
                print("else hold",N)
            held_out_idc = np.array(sorted(np.random.choice(N, size=[held_out], replace=False).tolist()))
            print('held_out idc................................')
            print(held_out_idc)
            remaining = np.array(sorted(set(range(N)) - set(held_out_idc.tolist())))
            print("remaning...........................",remaining)

            if isinstance(X, list):
                tidc = set(remaining.tolist())
                vidc = set(held_out_idc.tolist())
                X_train = [x for i, x in enumerate(X) if i in tidc]
                X_dev = [x for i, x in enumerate(X) if i in vidc]
                print("ifX_dev",X_dev)
            else:
                X_train = X[remaining]
                X_dev = X[held_out_idc]
                print("else X_dev", X_dev)
            y_train = y[remaining]
            y_dev = y[held_out_idc]

            return (remaining, held_out_idc), ((X_train, y_train), (X_dev, y_dev))

    def hold_out_dev(self, cls, X, y, metric, held_out=HOLD_OUT, number_samples=50, **kwargs):
           # print('hold out dev called...')
            train_accs = []
            dev_accs = []
            best = None
            while len(train_accs) < number_samples:
                try:
                    _, ((X_train, y_train), (X_dev, y_dev)) = self.hold_out(X, y, held_out=held_out)

                    clf = cls(**kwargs)
                    clf.fit(X_train, y_train)

                    train_accs.append(metric(clf.predict(X_train), y_train))
                    dev_acc = metric(clf.predict(X_dev), y_dev)
                    dev_accs.append(dev_acc)
                    if best is None or best[0] < dev_acc:
                        best = (dev_acc, clf)
                except BadDevSetError:
                    pass
            return np.array(train_accs), np.array(dev_accs), best

    def evaluate_clf(self, clf, X, y, metric, held_out=HOLD_OUT, number_samples=50):
            train_accs = []
            dev_accs = []
            while len(train_accs) < number_samples:
                try:
                    _, ((X_train, y_train), (X_dev, y_dev)) = self.hold_out(X, y, held_out=held_out)

                    train_accs.append(metric(clf.predict(X_train), y_train))
                    dev_accs.append(metric(clf.predict(X_dev), y_dev))
                except BadDevSetError:
                    pass
            return np.array(train_accs), np.array(dev_accs)

    def fit_transform_X(self, get_features, dataset=None):
            if dataset is None:
                dataset = self.training_set
            Xd = [get_features(utt) for utt in dataset]
            dvec = DictVectorizer(dtype=np.int32)
            X = dvec.fit_transform(Xd)
            return X, (get_features, dvec)

    def get_concerns(self, dataset=None):
            if dataset is None:
                dataset = self.training_set
            return [utt["concern"] for utt in dataset]

    def transform_X(self, state, dataset=None):
            get_features, dvec = state
            if dataset is None:
                dataset = self.training_set
            Xd = [self.get_features(utt) for utt in dataset]
            return dvec.transform(Xd), state

    def run_exp(self, X, y, metric, cls=DecisionTreeClassifier, **kwargs):
            train, dev, best_clf = self.hold_out_dev(cls, X, y, metric, **kwargs)
            return train, dev, best_clf

    @classmethod
    def plot_metric_histogram(cls, scores, ax=None):
            if ax is None:
                ax = plt.gca()
            ax.set_title("Mean = %.3f" % (np.mean(scores)))
            ax.hist(scores, range=(0, 1), bins=50, color="b")
            ax.axvline(np.mean(scores), color="r")



    def preprocess_words(self, label):
            label_data = self.get_label_set(label)
            print('////////////////////////')
            print(label_data)
            Neg = ' '.join(label_data)

            neg_hand = (self.negation_hand(Neg))

            Tok = self.tokenization(neg_hand)
            # print(Tok)
            s_word = (self.stop_words(Tok))
            # print(s_word)
            regular_ex = (self.regular_exp(s_word))

            uniq_list = (self.unique_list(regular_ex))
            uniq_list = self.get_segment_list(uniq_list)
            # print(uniq_list)
            all = self.Alllist()
            # print("all",all)
            # Tokens = self.tokenization(all)
            concerns=self.load_set1()
            # print("concerns:",concerns)
            # print(len(concerns))
            lst = self.listtostring(all)
            s_word = (self.stop_words(lst))
            uniq = self.unique_list(s_word)
            vector=self.countvector(uniq,concerns)

            return vector


    def preprocessing(self, is_tfidf=False, is_cosinesimilarty=False, is_Nmf=False):
        all = self.Alllist()
        #print("all uniq list")
        #print(all)
        lst = self.listtostring(all)

        uniq = self.unique_list(lst)
        concerns=[]

        concerns = self.load_set1()
        concerns = [self.tokenization(concerns[i]) for i in range(len(concerns))]
        concerns = [self.stop_words(concerns[i]) for i in range(len(concerns))]
        concerns = [self.regular_exp(concerns[i]) for i in range(len(concerns))]
        concerns = [self.port_stem(concerns[i]) for i in range(len(concerns))]
        concerns= [" ".join(concerns[i]) for i in range(len(concerns))]
        op1 = None
        if is_tfidf:
            
            op1 = Tfidf(concerns).tfidf()
        
        
        if is_cosinesimilarty:
            op1 = Cosine(op1).cosinesimilarty()
        
        
        if is_Nmf:
           op1=NMF(op1).non_negative_matrices()

        return op1



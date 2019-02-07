import json
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

data_dict = {'unknown':[]}
with open('data.jsons') as f:
    d = f.readline()
    while d :
        data = json.loads(d)
        labels = data['labels']
        for i in labels:
            if i not in data_dict.keys():
                data_dict[i] =[]
            data_dict[i].append(str(data['concern']))
        d = f.readline()
print(data_dict)
dia = (data_dict["diarrhea"])
diar = ''.join(dia)
print((diar))
negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                 "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                 "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                 "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                 "mustn't":"must not",
                 'abnormallymy': 'abnormally my', 'abouthe': 'about he','abouti':'about i', 'acidmy':'acid my',
                 'actionhe':'action he', 'activei':'active i', 'againdye': 'again dye', 'agobig':'ago big','almostjackson': 'almost jackson', 'alsoii':'also i', 'alsomy':'also my', 'anymoremy':'anymoremy',
                 'aroundmy': 'around my','asleep': 'a sleep','aspirini': 'aspirin i', 'awhilewe':'awhile we',
                 'babymy':'baby my','bei': 'be i','bettermy': 'better my', 'biggermy':'bigger my', 'diarrhoeacat': 'dairrhea cat',
                 'diarrhoeami': 'dairrhea mi','diarrhoeawhat': 'dairrhea what','underweightshe':'underweight she','healsshe': 'heals she'}
neg_pattern = re.compile(r'(' + '|'.join(negations_dic.keys()) + r')')
neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], diar)
print(neg_handled)
words_split = neg_handled.lower()
word_tokens = word_tokenize(words_split)
stop_wrd = set(stopwords.words('english'))
filter_sentence = [w for w in word_tokens if not w in stop_wrd]
filter_sentence = []
for w in word_tokens:
      if w not in stop_wrd:
          filter_sentence.append(w)
print(filter_sentence)
cleaned = [word for word in filter_sentence if word.isalpha()]
print(cleaned)
def port_stem(dataset):
#cleaned.sort()
 port_stem = PorterStemmer()
 stem = []
 for w in dataset:
     stem.append(port_stem.stem(w))
 return stem
stemming = port_stem(cleaned)
stemming.sort()
print(stemming)

from collections import Counter

def words(text): return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('data.jsons').read()))

def P(word, N=sum(WORDS.values())):

    return WORDS[word] / N

def correction(word):

    return max(candidates(word), key=P)

def candidates(word):

    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words):

    return set(w for w in words if w in WORDS)

def edits1(word):

    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):

    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

spell=[]
for w in cleaned:
    spell.append(correction(w))



po_stem = PorterStemmer()
stem=[]
for w in cleaned:
    # print(ps.stem(w))
    stem.append(po_stem.stem(w))
    # l=ps.stem(w)
print('stem = ',stem)
spell2=[]
for w in stem:
    spell2.append(correction(w))
print(spell2)
print(len(spell2))
def unique_list(l):
    ulist = []

    [ulist.append(x) for x in l if x not in ulist]
    return ulist
dup=unique_list(spell2)
print(dup))
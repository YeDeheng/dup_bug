from __future__ import print_function
import numpy as np
import os
import sys
import nltk
import re
import string
import stopwords
import unicodedata

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn import metrics
from scipy.sparse import coo_matrix, hstack
from sklearn.preprocessing import LabelEncoder
from text_unidecode import unidecode
from nltk.stem.porter import PorterStemmer

def get_severity_level(sev):   
    if sev == "critical":
        result = 1
    elif sev == "major":
        result = 2
    elif sev == "normal":
        result = 3
    elif sev == "minor":
        result = 4
    elif sev == "trivial":
        result = 5
    elif sev == "enhancement":
        result = 6
    else:
        result = 7        
    return result
 
def get_features_label(path):
    pri_label = []
    sev = []
    prod = ""
    comp = ""
    desc = ""
    desc_l = []
    doc_dict = {}
    doc_l = list()
    for f in os.listdir(path):          
        folder = f
        for d in os.listdir(os.path.join(path,folder)):
            doc_path = os.path.join(os.path.join(path,folder),d)   
            with open(doc_path) as doc:
                pri_label.append(folder)
                doc_dict['label'] = pri_label
                pri_label = []
                for i in doc:
                    if i.startswith("Summary:"):
                        desc+=i.strip("Summary:").strip()+" "
                    if (not i.startswith("ID")) and (not i.startswith("Summary")) and (not i.startswith("Product")) and (not i.startswith("Component")) and (not i.startswith("Version")) and (not i.startswith("Priority")) and (not i.startswith("Severity")):
                        desc+="".join(i.strip("Description:").strip())+" "
                    if i.startswith("Severity:"):
                        sev.append(i.strip("Severity:").strip())
                        doc_dict['sev'] = sev
                        sev = []
                    if i.startswith("Product:"):
                        prod = i.strip("Product:").strip()
                        doc_dict['prod'] = prod
                        prod = ""
                    if i.startswith("Component:"):
                        comp = i.strip("Component:").strip()
                        doc_dict['comp'] = comp
                        comp = ""
                    desc_l.append(desc)
                    doc_dict['desc'] = desc_l
                    desc_l = []
            desc = ""
            doc_l.append(doc_dict.copy())            
    return doc_l

def get_combine_matrix(feature1, feature2, feature3, feature4,
                       feature5, feature6, feature7, feature8,
                       feature9, feature10, feature11, feature12,
                       feature13, feature14, feature15, feature16):
    f1 = coo_matrix(feature1)
    f2 = coo_matrix(feature2)
    f3 = coo_matrix(feature3)
    f4 = coo_matrix(feature4)
    f5 = coo_matrix(feature5)
    f6 = coo_matrix(feature6)
    f7 = coo_matrix(feature7)
    f8 = coo_matrix(feature8)
    f9 = coo_matrix(feature9)
    f10 = coo_matrix(feature10)
    f11 = coo_matrix(feature11)
    f12 = coo_matrix(feature12)
    f13 = coo_matrix(feature13)
    f14 = coo_matrix(feature14)
    f15 = coo_matrix(feature15)
    f16 = coo_matrix(feature16)    
    matrix1 = hstack([f1,f2])
    matrix2 = hstack([matrix1,f3])
    matrix3 = hstack([matrix2,f4])
    matrix4 = hstack([matrix3,f5])
    matrix5 = hstack([matrix4,f6])
    matrix6 = hstack([matrix5,f7])
    matrix7 = hstack([matrix6,f8])
    matrix8 = hstack([matrix7,f9])
    matrix9 = hstack([matrix8,f10])
    matrix10 = hstack([matrix9,f11])
    matrix11 = hstack([matrix10,f12])
    matrix12 = hstack([matrix11,f13])
    matrix13 = hstack([matrix12,f14])
    matrix14 = hstack([matrix13,f15])
    matrix15 = hstack([matrix14,f16])
    return matrix15

def get_desc(path):
    desc = []
    for i in get_features_label(path):
        desc_train = i['desc'][0]
        desc.append(desc_train)
    return desc

def get_severity(path):
    sev = []
    for i in get_features_label(path):
        sev_train = get_severity_level(i['sev'][0])
        sev.append([sev_train])
    return sev

def get_priority(path):
    pri = []
    le = LabelEncoder()
    for i in get_features_label(path):
        pri_train = i['label']
        pri.append(pri_train)
    le.fit(pri)
    le_t = le.transform(pri)
    pri = []
    for l in le_t:
        pri.append(l[0])
    return pri

def get_product(path):
    prod = []
    le = LabelEncoder()
    for i in get_features_label(path):
        prod_train = i['prod']
        prod.append(prod_train)
    le.fit(prod)
    le_t = le.transform(prod)
    prod = []
    for l in le_t:
        prod.append([l])
    return prod

def get_component(path):
    comp = []
    le = LabelEncoder()
    for i in get_features_label(path):
        comp_train = i['comp']
        comp.append(comp_train)
    le.fit(comp)
    le_t = le.transform(comp)
    comp = []
    for l in le_t:
        comp.append([l])
    return comp

# Proportion of same product for P1 to P5
def get_pro_pri_P1(path):
    P1 = []
    for i in range(0,len(get_product(path))):
        if (i < 100):
            P1.append(get_product(path)[i][0])
        if (i == 100):
            break
    return P1

def get_pro_pri_P2(path):
    P2 = []
    for i in range(0,len(get_product(path))):
        if (i >= 100 and i < 200):
            P2.append(get_product(path)[i][0])
        if (i == 200):
            break
    return P2

def get_pro_pri_P3(path):
    P3 = []
    for i in range(0,len(get_product(path))):
        if (i >= 200 and i < 300):
            P3.append(get_product(path)[i][0])
        if (i == 300):
            break
    return P3

def get_pro_pri_P4(path):
    P4 = []
    for i in range(0,len(get_product(path))):
        if (i >= 300 and i < 400):
            P4.append(get_product(path)[i][0])
        if (i == 400):
            break
    return P4

def get_pro_pri_P5(path):
    P5 = []
    for i in range(0,len(get_product(path))):
        if (i >= 400):
            P5.append(get_product(path)[i][0])
    return P5

def get_pro_prop(path, P):
    seen = set()
    seen_a = []
    P1_pro = set()
    pro_prop = []
    for x in get_product(path):
        if x[0] not in seen:
            seen_a.append(x[0])
    for y in get_product(path):
        if y[0] in P:
            prop_P = P.count(y[0])/float(len(get_product(path)))
            pro_prop.append([prop_P])
        else:
            if y[0] in seen_a:
                prop_P = seen_a.count(y[0])/float(len(get_product(path)))  
                pro_prop.append([prop_P])        
    return pro_prop      

# Proportion of same component for P1 to P5
def get_comp_pri_P1(path):
    P1 = []
    for i in range(0,len(get_component(path))):
        if (i < 100):
            P1.append(get_component(path)[i][0])
        if (i == 100):
            break
    return P1

def get_comp_pri_P2(path):
    P2 = []
    for i in range(0,len(get_component(path))):
        if (i >= 100 and i < 200):
            P2.append(get_component(path)[i][0])
        if (i == 200):
            break
    return P2

def get_comp_pri_P3(path):
    P3 = []
    for i in range(0,len(get_component(path))):
        if (i >= 200 and i < 300):
            P3.append(get_component(path)[i][0])
        if (i == 300):
            break
    return P3

def get_comp_pri_P4(path):
    P4 = []
    for i in range(0,len(get_component(path))):
        if (i >= 300 and i < 400):
            P4.append(get_component(path)[i][0])
        if (i == 400):
            break
    return P4

def get_comp_pri_P5(path):
    P5 = []
    for i in range(0,len(get_component(path))):
        if (i >= 400):
            P5.append(get_component(path)[i][0])
    return P5

def get_comp_prop(path, P):
    seen = set()
    seen_a = []
    comp_prop = []
    for x in get_component(path):
        if x[0] not in seen:
            seen_a.append(x[0])
    for y in get_component(path):
        if y[0] in P:
            prop_P = P.count(y[0])/float(len(get_component(path)))
            comp_prop.append([prop_P])
        else:
            if y[0] in seen_a:
                prop_P = seen_a.count(y[0])/float(len(get_component(path)))  
                comp_prop.append([prop_P])        
    return comp_prop 

# Mean value of same product 
def get_pro_m(path):
    seen = set()
    seen_a = []
    pro_m = []
    for x in get_product(path):
        if x[0] not in seen:
            seen_a.append(x[0])
    for y in get_product(path):
        if y in seen_a:
            mean = seen_a.count(y)/float(len(get_product(path)))
            pro_m.append([mean])    
    return pro_m

# Mean value of same component 
def get_comp_m(path):
    seen = set()
    seen_a = []
    comp_m = []
    for x in get_component(path):
        if x[0] not in seen:
            seen_a.append(x[0])
    for y in get_component(path):
        if y in seen_a:
            mean = seen_a.count(y)/float(len(get_component(path)))
            comp_m.append([mean])    
    return comp_m  
    
# Preprocessing Function
def get_tokens(text):
    token = []
    for w in text:
        lowers = w.lower()
        replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
        replaced = lowers.translate(replace_punctuation)
        tokens = nltk.word_tokenize(replaced)
        token.append(tokens)      
    return token

def get_removed_stopwords(tokens):
    filtered_l = []
    for i in tokens:
        filtered = [x for x in i if x not in stopwords.stop_words]
        filtered_l.append(filtered)
    return filtered_l

def stem_tokens(tokens):
    stemmed = []
    for i in tokens:
        stem = [PorterStemmer().stem(unidecode(w)) for w in i if re.match("[a-zA-Z]+",unidecode(w))]        
        stemmed.append(" ".join(stem))
    return stemmed

# Initialised Working Directory, change to the directory where you saved the sample datasets folder
train_path = os.path.join("C:",os.sep,"Users","103922","Desktop","Sample Datasets","Eclipse","train")
test_path = os.path.join("C:",os.sep,"Users","103922","Desktop","Sample Datasets","Eclipse","test")

# Load Dataset
print("Loading Dataset")
categories = ["P1", "P2", "P3", "P4", "P5"]
test = load_files(test_path, categories = categories)

# Preprocessing
print("Preprocessing")
tokens_train = get_tokens(get_desc(train_path))
no_stpwd_train = get_removed_stopwords(tokens_train)
stemmed_train = stem_tokens(no_stpwd_train)
P1_pro_tr = get_pro_pri_P1(train_path)
P2_pro_tr = get_pro_pri_P2(train_path)
P3_pro_tr = get_pro_pri_P3(train_path)
P4_pro_tr = get_pro_pri_P4(train_path)
P5_pro_tr = get_pro_pri_P5(train_path)
P1_comp_tr = get_comp_pri_P1(train_path)
P2_comp_tr = get_comp_pri_P2(train_path)
P3_comp_tr = get_comp_pri_P3(train_path)
P4_comp_tr = get_comp_pri_P4(train_path)
P5_comp_tr = get_comp_pri_P5(train_path)
train_label = get_priority(train_path)

tokens_test = get_tokens(get_desc(test_path))
no_stpwd_test = get_removed_stopwords(tokens_test)
stemmed_test = stem_tokens(no_stpwd_test)
P1_pro_tst = get_pro_pri_P1(test_path)
P2_pro_tst = get_pro_pri_P2(test_path)
P3_pro_tst = get_pro_pri_P3(test_path)
P4_pro_tst = get_pro_pri_P4(test_path)
P5_pro_tst = get_pro_pri_P5(test_path)
P1_comp_tst = get_comp_pri_P1(test_path)
P2_comp_tst = get_comp_pri_P2(test_path)
P3_comp_tst = get_comp_pri_P3(test_path)
P4_comp_tst = get_comp_pri_P4(test_path)
P5_comp_tst = get_comp_pri_P5(test_path)
test_label = get_priority(test_path)

# Creates Object
count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

print("Create Model")
clf = MultinomialNB()
sgd = SGDClassifier(penalty='elasticnet', alpha=1e-3, n_iter=500, random_state=0)
lsvm = svm.SVC(kernel='linear')

# Feature Extraction for Training Set
X_train_counts = count_vect.fit_transform(stemmed_train)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

print("Training ...")
train_f = get_combine_matrix(X_train_tfidf,get_severity(train_path),
                             get_component(train_path),get_product(train_path),
                             get_pro_prop(train_path, P1_pro_tr),get_pro_prop(train_path, P2_pro_tr),
                             get_pro_prop(train_path, P3_pro_tr),get_pro_prop(train_path, P4_pro_tr),
                             get_pro_prop(train_path, P5_pro_tr),get_comp_prop(train_path, P1_comp_tr),
                             get_comp_prop(train_path, P2_comp_tr),get_comp_prop(train_path, P3_comp_tr),
                             get_comp_prop(train_path, P4_comp_tr),get_comp_prop(train_path, P5_comp_tr),
                             get_pro_m(train_path),get_comp_m(train_path))

# Fit to Model
print("Fitting ...")
clf = clf.fit(train_f, train_label)
sgd = sgd.fit(train_f, train_label)
lsvm = lsvm.fit(train_f, train_label)

# Feature Extraction for Testing
X_test_counts = count_vect.transform(stemmed_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

print("Testing ...")
test_f = get_combine_matrix(X_test_tfidf,get_severity(test_path),
                            get_component(test_path),get_product(test_path),
                            get_pro_prop(test_path, P1_pro_tst),get_pro_prop(test_path, P2_pro_tst),
                            get_pro_prop(test_path, P3_pro_tst),get_pro_prop(test_path, P4_pro_tst),
                            get_pro_prop(test_path, P5_pro_tst),get_comp_prop(test_path, P1_comp_tst),
                            get_comp_prop(test_path, P2_comp_tst),get_comp_prop(test_path, P3_comp_tst),
                            get_comp_prop(test_path, P4_comp_tst),get_comp_prop(test_path, P5_comp_tst),
                            get_pro_m(test_path),get_comp_m(test_path))

# Get predicted result
print("Result ...")
clf_predicted = clf.predict(test_f)
clf_accuracy_score = np.mean(clf_predicted == test_label)
print("clf accuracy score:", clf_accuracy_score)

sgd_predicted = sgd.predict(test_f)
sgd_accuracy_score = np.mean(sgd_predicted == test_label)
print("sgd accuracy score:", sgd_accuracy_score)

lsvm_predicted = lsvm.predict(test_f)
lsvm_accuracy_score = np.mean(lsvm_predicted == test_label)
print("lsvm accuracy score:", lsvm_accuracy_score,"\n")


# Evaluation of the performance on Test set 
print((metrics.classification_report(test_label, clf_predicted, target_names=test.target_names)))
print(metrics.confusion_matrix(test_label, clf_predicted))

print((metrics.classification_report(test_label, sgd_predicted, target_names=test.target_names)))
print(metrics.confusion_matrix(test_label, sgd_predicted))

print((metrics.classification_report(test_label, lsvm_predicted, target_names=test.target_names)))
print(metrics.confusion_matrix(test_label, lsvm_predicted))








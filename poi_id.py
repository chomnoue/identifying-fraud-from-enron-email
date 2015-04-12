#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
import numpy



### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

#preparing features for selection
features_list=data_dict[data_dict.keys()[0]].keys()
features_list.remove('poi')
features_list.remove('email_address')

features_list =['poi']+features_list

### Store to my_dataset for easy export below.
my_dataset = data_dict

       
### Extract features and labels from dataset for local testing
def extract_data():
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    return labels,features
labels,features=extract_data()
##applying Tree-based feature selection
from sklearn.ensemble import ExtraTreesClassifier#,RandomForestClassifier
clf = ExtraTreesClassifier()
clf.fit(features, labels)
feature_importances=[(features_list[ind+1],imp) for (ind,imp) in enumerate(clf.feature_importances_)]
##let us fix to 10 the number of features we will use
nfeatures=10
import heapq

top_features=heapq.nlargest(nfeatures,[imp[0] for imp in feature_importances],lambda x:x[1])

features_list = ['poi']+top_features

print "features to use : ",features_list

### Task 2: Remove outliers

#remove all zeroes first, to keep consistent the number of data poins used to build our model with 
#the data points in my_dataset
keys=sorted(my_dataset.keys())
print '*********************all zeroes**********************'
for key in keys:
    data_point=my_dataset[key]
    rem=None
    for feature in features_list:
        rem=True
        value=data_point[feature]
        if value != "NaN" and float(value)!=0:
            rem=False
            break
    if rem:
        print key,':'
        print my_dataset[key]
        del(my_dataset[key])
print '*********************end all zeroes**********************'

#remove non persons
del(my_dataset['THE TRAVEL AGENCY IN THE PARK'])
del(my_dataset['TOTAL'])

print "# data points after removing all 0s : ",len(my_dataset.keys())
##using pca to reduce dimensionality of features
from sklearn.decomposition import PCA



#print "# dat points for outliers removal ",len(labels)

def build_pca():
    labels, features = extract_data()
    pca=PCA(n_components=2)
    pca.fit(features)
    features_pca=pca.transform(features)
    return features_pca
    


def plot_features(features_pca,title):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.title(title)
    plt.xlabel("Second component")
    plt.ylabel("First component")
    for point in features_pca:
        plt.scatter(point[1],point[0])
    plt.show()

def build_pca_and_plot(title):
    features_pca=build_pca()
    plot_features(features_pca,title)
    return features_pca
#features_pca=build_pca_and_plot("Principal components before removing outliers")


#outliers_fraction=.95
#applying elliptic envelope to remove outliers

def outliers_from_ellipticEnvelope():
    from sklearn.covariance import EllipticEnvelope
    env=EllipticEnvelope()
    env.fit(features_pca)
    outlier_pred=env.decision_function(features_pca).ravel()
    return outlier_pred
def reshape(data_list):
    cols=1
    try:
        cols=len(data_list[0])
    except:
        pass
    return numpy.reshape(numpy.array(data_list), (len(data_list), cols))
    
def outliers_from_linear_regression():
    from sklearn.linear_model import LinearRegression
    import math
    reg = LinearRegression()
    first_feature,second_feature=zip(*features_pca)
    first_feature=reshape(first_feature)
    second_feature=reshape(second_feature)
    reg.fit(list(first_feature), list(second_feature))
    predictions=reg.predict(first_feature)
    outlier_pred=[math.fabs(predictions[i]-second_feature[i]) for i in range(len(predictions))]
    return outlier_pred

#from scipy import stats
    
#outlier_pred=outliers_from_linear_regression()#outliers_from_ellipticEnvelope()
#threshold = stats.scoreatpercentile(outlier_pred,
#                                            100 * outliers_fraction)
#outlier_pred = outlier_pred > threshold
# 
#keys=sorted(my_dataset.keys())
#outlier_keys=[keys[i] for i in range(len(outlier_pred)) if outlier_pred[i]]
#print "*********************outliers*********************"
#for outlier in outlier_keys:
#    #avoid deleting pois
#    if my_dataset[outlier]['poi']!=1:
#        print outlier,":"
#        print my_dataset[outlier]
#        del(my_dataset[outlier])
#print "*********************end outliers*********************"

print "# data points after removing outliers : ",len(my_dataset.keys())
#plot again after removing outliers
#build_pca_and_plot("Principal components after removing outliers")




### Task 3: Create new feature(s)

def create_feature(dataset,feature_name,function):
    for key in dataset.keys():
        data_point=dataset[key]
        value=function(data_point)
        data_point[feature_name]=value
        
def add(data_point,feature1,feature2):
    value1=data_point[feature1]
    value2=data_point[feature2]
    if value1=='NaN':
        value1=float(0)
    if value2=='NaN':
        value2=float(0)
    return value1+value2
        
from_messages='from_messages'
to_messages='to_messages'
from_this_person_to_poi='from_this_person_to_poi'
from_poi_to_this_person='from_poi_to_this_person'
exchanged_messages='exchaned_messages'
exchanged_messages_with_poi='exchaned_messages_with_poi'
create_feature(my_dataset,exchanged_messages,lambda data_point:add(data_point,from_messages,to_messages))
create_feature(my_dataset,exchanged_messages_with_poi,lambda data_point:add(data_point,from_this_person_to_poi,from_poi_to_this_person))

features_list=features_list+[exchanged_messages,exchanged_messages_with_poi]
for feature in [from_messages,to_messages,from_this_person_to_poi,from_poi_to_this_person]:
    features_list.remove(feature)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit

#clf = tree.DecisionTreeClassifier(min_samples_split=5) 
#clf=GaussianNB
clf = ExtraTreesClassifier()    # Provided to give you a starting point. Try a varity of classifiers.
#clf=RandomForestClassifier()


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#the following is time consuming, uncomment it if you want to tune parameters

#tuned_parameters={'n_estimators':[1,5,10,15,20],'max_features':["sqrt","log2",None],
#                  'min_samples_split':range(2,11),'criterion':['gini','entropy']}
#labels, features = extract_data()
#
#clf=GridSearchCV(clf, tuned_parameters, cv=StratifiedShuffleSplit(labels, 100, random_state = 42),
#                 scoring='recall')
#                 
#clf.fit(features,labels)
#
#clf=clf.best_estimator_

clf=ExtraTreesClassifier(criterion='gini', max_features=None,
           min_samples_split=2, n_estimators=1)

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)
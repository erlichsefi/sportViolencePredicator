import collections
import itertools
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import RandomOverSampler

from Mapper import Mapper,array_key,key_to_array
from imblearn.over_sampling import RandomOverSampler
from skmultilearn.problem_transform import LabelPowerset

from matplotlib import gridspec
import matplotlib.pyplot as plt



def remove_all_null(final_df):
    """removes all the row that contain NaN
    Args:
        final_df: the DataFrame to clean
    Returns:
        final_df: the DataFrame after the row removed
        droped_rows: a DataFrame of all row removed
"""
    final_df_c=final_df.copy()
    indexs_before_drop=final_df.index
    final_df=final_df.dropna(axis=0,how='any')
    removed=[item for item in indexs_before_drop if item not in final_df.index]
    print("Number of smaples that was removed becuse they contain 'NaN' ="+ str(len(removed)))
    assert(final_df.isna().sum().sum()==0)
    droped_rows=final_df_c.loc[removed,:]
    return final_df,droped_rows


'''

'''
def build_scale(X_test):
    scaler = MinMaxScaler()
    return scaler.fit(X_test)

def sacle(scaler,X):
    return scaler.transform(X)



def split_by_label(X,Y,the_test_size,if_shuffle=True):
    assert(the_test_size<1.0 and the_test_size>0.0)
    """split the data for test and train where #the_test_size from each label is in the test set,
    and (1-#the_test_size) , !this function don't fit to regression!
    Args:
        X: feature vector
        Y: labels for each feature vector
        the_test_size: the fraction of test
        if_shuffle: if to shuffle the data set before spliting (by defult True)
    Returns:
        X_train_folds,X_test_folds,Y_train_folds,Y_test_folds
        where each one is a list of sets
    """
    X_train_folds=[np.array([]),np.array([]),np.array([]),np.array([]),np.array([])]
    X_test_folds=[np.array([]),np.array([]),np.array([]),np.array([]),np.array([])]
    Y_train_folds=[np.array([]),np.array([]),np.array([]),np.array([]),np.array([])]
    Y_test_folds=[np.array([]),np.array([]),np.array([]),np.array([]),np.array([])]

    all_classes=np.unique(Y)

    # for each class
    for the_class in all_classes:
        # take all the smaples
        all_class_samples=X[np.where( Y == the_class )]
        # folds
        for i in range(5):
            # split that class to test and train
            X_train, X_test, y_train, y_test = train_test_split(all_class_samples,pd.DataFrame([the_class]*len(all_class_samples),columns=['label']),test_size=the_test_size,shuffle=if_shuffle)
            # append
            X_train_folds[i]= np.concatenate((X_train,X_train_folds[i]), axis=0) if X_train_folds[i].size else X_train
            X_test_folds[i]=np.concatenate((X_test,X_test_folds[i]), axis=0)   if X_test_folds[i].size else X_test
            Y_train_folds[i]=np.concatenate((y_train,Y_train_folds[i]), axis=0)  if Y_train_folds[i].size else y_train
            Y_test_folds[i]=np.concatenate((y_test,Y_test_folds[i]), axis=0) if Y_test_folds[i].size else y_test

    # checking the class size are the same
    for  X_train, X_test, y_train, y_test in zip(X_train_folds,X_test_folds,Y_train_folds,Y_test_folds):
        assert(class_size(np.concatenate((y_train,y_test), axis=0))==class_size(Y))
    return X_train_folds,X_test_folds,Y_train_folds,Y_test_folds



def split_5_fold(X,Y,the_test_size):
    """split the data for test and train where #the_test_size from all label is in the test set,
    and (1-#the_test_size) is in the train set ,
    Args:
        X: feature vector
        Y: labels for each feature vector
        the_test_size: the fraction of test
    Returns:
        X_train_folds,X_test_folds,Y_train_folds,Y_test_folds
        where each one is a list of sets
    """
    # the results
    X_train_folds=[np.array([]),np.array([]),np.array([]),np.array([]),np.array([])]
    X_test_folds=[np.array([]),np.array([]),np.array([]),np.array([]),np.array([])]
    Y_train_folds=[np.array([]),np.array([]),np.array([]),np.array([]),np.array([])]
    Y_test_folds=[np.array([]),np.array([]),np.array([]),np.array([]),np.array([])]


    kf = KFold(n_splits=5)
    i=0
    # for each split
    for train_indexs, test_indexs in kf.split(X):
            # take the corespending smaples
            X_train_folds[i]=X[train_indexs]
            X_test_folds[i]=X[test_indexs]
            Y_train_folds[i]=Y[train_indexs]
            Y_test_folds[i]=Y[test_indexs]
            assert(Y_test_folds[i].shape[0]==X_test_folds[i].shape[0])
            assert(Y_train_folds[i].shape[0]==X_train_folds[i].shape[0])
            i=i+1

    return X_train_folds,X_test_folds,Y_train_folds,Y_test_folds



def five_fold_avg_classifction(clf,X,Y,test_size,scoring_function=(lambda clf,x,y:accuracy_score(y, clf.predict(x)))):
    """ preform five fold test with the classificationself.
        - split the to train to test by label
        - for each fold from 5 folds
            - clone the original clf
            - scale train
            - fit
            - scale test
            - take measurements
        - averge the measurements
    Args:
        clf: a new classifer to train
        X: feature vector
        Y: labels for each feature vector
        test_size: the fraction of test
        scoring_function: the scoring function to use, by default accuracy
    Returns:
        1. avg socre according to default #scoring_function,by default accuracy
        2. avg auc socre
        3. avg confusion matrix
    """

    X_train_folds,X_test_folds,Y_train_folds,Y_test_folds=split_by_label(X, Y,the_test_size=test_size)

    socre_values=[]
    auc_values=[]
    con_matixs=[]
    f1_values=[]
    current_clf=None
    for  X_train, X_test, y_train, y_test in zip(X_train_folds,X_test_folds,Y_train_folds,Y_test_folds):
        # test the set sizes
        assert(X_train.shape[0]==y_train.shape[0])
        assert(X_test.shape[0]==y_test.shape[0])
        current_clf=deepcopy(clf)

        # scale and oversample
        scaler=build_scale(X_train)
        _X,_Y=overSampleAny(X_train, y_train.ravel())
        _X=sacle(scaler,_X)

        #fit
        current_clf=current_clf.fit(_X,_Y)

        # scale the test set
        _X_test=sacle(scaler,X_test)

        # scoring mathods
        socre_values.append(scoring_function(current_clf,_X_test,y_test))
        con_matixs.append(confusion_matrix(y_test, current_clf.predict(_X_test)))
        auc_values.append(auc(y_test, current_clf.predict(_X_test)))
    return np.average(socre_values),np.average(auc_values),np.average(con_matixs,axis=0)


def five_fold_avg_regresion(clf,X,Y,test_size,predict_function=(lambda clf,x,low,upp: clf.predict(x))):
    #print(str(Y.shape))
    """ preform five fold test to regresion problems.
        - split the to train to test
        - for each fold from 5 folds
            - clone the original clf
            - scale train
            - fit
            - scale test
            - take measurements
        - averge the measurements
    Args:
        clf: a new classifer to train
        X: feature vector
        Y: labels for each feature vector
        test_size: the fraction of test
        scoring_function: the predict function to use, by default clf.predict
    Returns:
        1. avg socre according to default #scoring_function,by default mean_squared_error
        2. the last y_pred
    """

    X_train_folds,X_test_folds,Y_train_folds,Y_test_folds=split_5_fold(X, Y,the_test_size=test_size)

    socre_values=[]
    conf_vlaues=[]
    current_clf=None
    last_pred=None
    for  X_train, X_test, y_train, y_test in zip(X_train_folds,X_test_folds,Y_train_folds,Y_test_folds):
        #print(str(y_train.shape))
        assert(X_train.shape[0]==y_train.shape[0])
        assert(X_test.shape[0]==y_test.shape[0])

        current_clf=deepcopy(clf)
        """we can't balance regression??"""
        #sample_weight_parm=create_smaple_weight(y_train)
        #current_clf.sample_weight = sample_weight_parm
        # build the scale and scale
        scaler=build_scale(X_train)
        _X=sacle(scaler,X_train)
        # fit
        current_clf=current_clf.fit(_X, y_train)

        _X_test=sacle(scaler,X_test)
        last_pred=predict_function(current_clf,_X_test,np.min(Y),np.max(Y))
        socre_values.append(mean_squared_error(y_test,last_pred))

    return np.average(socre_values),last_pred


def five_fold_avg_class_regresion(clf,X,Y,test_size,predict_function=(lambda clf,x,low,upp: clf.predict(x))):
    """ preform five fold test to regresion problems but output classification measurements
        - split the to train to test
        - for each fold from 5 folds
            - clone the original clf
            - fit
            - take measurements
        - averge the measurements
    Args:
        clf: a new classifer to train
        X: feature vector
        Y: labels for each feature vector
        test_size: the fraction of test
        predict_function: the predict function to use, by default clf.predict
    Returns:
        1. avg mean_squared_error
        2. avg confusion matrix
    """

    X_train_folds,X_test_folds,Y_train_folds,Y_test_folds=split_by_label(X, Y,the_test_size=test_size)

    socre_values=[]
    con_vlaues=[]
    current_clf=None

    for  X_train, X_test, y_train, y_test in zip(X_train_folds,X_test_folds,Y_train_folds,Y_test_folds):
        assert(len(np.unique(y_train))==len(np.unique(y_test)))
        assert(X_train.shape[0]==y_train.shape[0])
        assert(X_test.shape[0]==y_test.shape[0])
        current_clf=deepcopy(clf)
        ## we can balance here becuase we assume the inputs are classes but there is linear realtion between classes
        sample_weight_parm=create_smaple_weight(y_train.ravel())
        current_clf.sample_weight = sample_weight_parm
        # build the scale and overSampling
        scaler=build_scale(X_train)
        _X,_Y=overSampleAny(X_train, y_train.ravel())
        _X=sacle(scaler,_X)
        #fit
        current_clf=current_clf.fit(_X,_Y)
        #sacle the test
        _X_test=sacle(scaler,X_test)
        #assert(len(np.unique(y_test))==len(np.unique(predict_function(current_clf,_X_test))))
        matrix=confusion_matrix(y_test,predict_function(current_clf,_X_test,np.min(Y),np.max(Y)))
        assert (matrix.shape[0]==len(np.unique(Y)) and matrix.shape[1]==len(np.unique(Y)) )
        con_vlaues.append(matrix)
        socre_values.append(mean_squared_error(y_test,current_clf.predict(_X_test)))

    return np.average(socre_values),np.average(con_vlaues,axis=0)

def best_param(clf,tuned_parameters):
#    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                     'C': [1, 10, 100, 1000]},
#                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(deepcopy(clf), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
            print()

            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
            print()

def custem_score(clf,x,low,upp):
    """
    predict and round to the closest
    """
    y_pred=clf.predict(x)
    return np.minimum(np.maximum(low,np.around(y_pred)),upp)

def custem_score_ceil(clf,x,low,upp):
    """
    predict and round to ceil
    """
    y_pred=clf.predict(x)
    return np.minimum(np.maximum(low,np.ceil(y_pred)),upp)


def custem_score_floor(clf,x,low,upp):
    """
    predict and round to floor
    """
    y_pred=clf.predict(x)
    return np.minimum(np.maximum(low,np.floor(y_pred)),upp)

# def false_positive(clf,x,y):
#     cm1 = confusion_matrix(clf.predict(x), y)
#     FalsePositive = []
#     for i in range(len(np.unique(y))):
#          FalsePositive.append(sum(cm1[:,i]) - cm1[i,i])
#     return np.unique(y),FalsePositive


def regression_projected_classification_estimation(X,Y,TEST_VALUE):
        """
        run the estimation of regression where the regresion output a integer as a class.
        Args:
            X: feature vector
            Y: labels for each feature vector
            TEST_VALUE: the fraction of test
        Returns:
            Nothing

        """
        assert(Y.shape[0]==X.shape[0])

        if ((type(X) is not np.ndarray) or (type(Y) is not np.ndarray) ):
            print("we are hoping to get only numpy arrays please use: .values")
            return

        Y = np.array(Y).astype(int)

        print("---------- regression with classification output---------")

        closest_socre,closest_confusion_matrix=five_fold_avg_class_regresion(svm.SVR(kernel='rbf'),X,Y, test_size=TEST_VALUE,predict_function=custem_score)
        ceil_socre,ceil_confusion_matrix=five_fold_avg_class_regresion(svm.SVR(kernel='rbf'),X, Y, test_size=TEST_VALUE,predict_function=custem_score_ceil)
        floor_socre,floor_confusion_matrix=five_fold_avg_class_regresion(svm.SVR(kernel='rbf'),X, Y, test_size=TEST_VALUE,predict_function=custem_score_floor)




        true_class=class_size(Y)
        print("class size= "+str(true_class))

        plt.figure(1,figsize=(15, 15))
        gs = gridspec.GridSpec(2,4)

        plt.subplot(gs[0])
        print("svr with rbf - avg score with round the closest = "+str(closest_socre))
        plot_confusion_matrix(closest_confusion_matrix,true_class.keys(),title=" no round ")



        plt.subplot(gs[1])
        print("svr with rbf - avg score with ceil  = "+str(ceil_socre))
        plot_confusion_matrix(ceil_confusion_matrix,true_class.keys(),title=" round the ceil ")

        plt.subplot(gs[2])
        print("svr with rbf - avg score with floor = "+str(floor_socre))
        plot_confusion_matrix(floor_confusion_matrix,true_class.keys(),title=" round the floor ")

        plt.subplot(gs[3])
        print("svr with rbf - avg score with floor = "+str(floor_socre))
        plot_confusion_matrix(floor_confusion_matrix,true_class.keys(),title=" round the floor ")





def regression_estimation(X,Y,TEST_VALUE):
    clfs=[('svm',svm.SVR(kernel='rbf')),
      ('radom_forest',RandomForestRegressor(max_depth=10, random_state=0)),
     ('nn',MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15, 10), random_state=1))]
    input_regression_estimation(clfs,X,np.array(Y).astype(float),TEST_VALUE)

def multi_regression_estimation(X,Y,TEST_VALUE):
    #print(str(Y.shape))
    clfs=[('svm',MultiOutputRegressor(svm.SVR(kernel='rbf'))),
      ('radom_forest',RandomForestRegressor(max_depth=10, random_state=0)),
     ('nn',MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15, 10), random_state=1))]
    input_regression_estimation(clfs,X,Y,TEST_VALUE)

def input_regression_estimation(clfs,X,Y,TEST_VALUE):
    assert(Y.shape[0]==X.shape[0])
    #X,Y=overSampleAny(X,Y)
    if ((type(X) is not np.ndarray) or (type(Y) is not np.ndarray) ):
        print("we are hoping to get only numpy arrays please use: .values")
        return

    print("---------- regression---------")
    plt.figure(1,figsize=(15, 15))
    gs = gridspec.GridSpec(len(clfs),1)

    index=0
    for class_name,clf in  clfs:
        #print(str(Y.shape))
        socre,last_pred=five_fold_avg_regresion(clf,X, Y, test_size=TEST_VALUE)
        print(class_name+" - avg score with normal rmse = "+str(socre))


        plt.subplot(gs[index])
        plt.title(class_name)
        n, bins, patches = plt.hist(last_pred, 40, facecolor='blue', alpha=0.5)
        index=index+1
    print("------------------------------")

def plot_confusion_matrix(cm,classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# count the size of the class
def class_size(Y):
    unique, counts = np.unique(Y, return_counts=True)
    return dict(zip(unique, counts))


def absolute_value(val):
    a  = np.round(val/100.*sizes.sum(), 0)
    return a

def input_classification_estimation(clfs,X,Y,TEST_VALUE):
    assert(Y.shape[0]==X.shape[0])
    if (len(Y[0].shape)!=0):
        print("we are hoping to a vector of size n row with 1 value each row")
        print("if you are tring to do classification to more then on class the stupid, create one class")
        return
    if ((type(X) is not np.ndarray) or (type(Y) is not np.ndarray) ):
        print("we are hoping to get only numpy arrays please use: .values")
        return
    Y = np.array(Y).astype(int)
    print("----------classification------")




    true_class=class_size(Y)
    plt.figure(1,figsize=(5, 5))
    #plt.subplot(gs[0])
    plt.title("class distribution")
    plt.pie(true_class.values(), labels=true_class.keys(),
        autopct='%1.1f%%', shadow=True, startangle=140)

    index=2
    for class_name,clf in  clfs:
        socre,avg_acc,conf_matrix=five_fold_avg_classifction(clf,X, Y.ravel(), test_size=TEST_VALUE)
        print(class_name+" - mean accuracy is= "+str(socre))
        print(class_name+" - avg auc score is= "+str(avg_acc))
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        plt.figure(index,figsize=(10, 5))
        gs = gridspec.GridSpec(1,2)

        plt.subplot(gs[0])
        plt.title(class_name)

        plt.pie(conf_matrix.sum(axis=0), labels=np.unique(Y.ravel()),
            autopct='%1.1f%%', shadow=True, startangle=140)

        plt.subplot(gs[1])
        plt.title(class_name)
        plot_confusion_matrix(conf_matrix,np.unique(Y.ravel()),title=class_name)

        index=index+2
    print("------------------------------")


def classification_estimation(X,Y,TEST_VALUE):
    clfs=[('svm_ovo',svm.SVC(decision_function_shape='ovo',class_weight='balanced')),
      ('svm_ovr',svm.SVC(decision_function_shape='ovr',class_weight='balanced')),
      ('radom_forest',RandomForestClassifier(max_depth=10, random_state=0)),
     ('nn',MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10,10, 10,10), random_state=1))]
    input_classification_estimation(clfs,X,Y,TEST_VALUE)



from imblearn.over_sampling import SMOTE, ADASYN
def overSampling(X,y):
    """
     over sample the data, and balance the classes
    """
    ros = RandomOverSampler()
    #print(str(X.shape))
    return ros.fit_sample(X, y)

def dup_dataset(X,y):
    """
    duplicate the data set.
    """
    _X=np.append(X,X, axis=0)
    _Y=np.append(y,y, axis=0)
    return _X,_Y

def overSampleAny(X,y):
    """
    over smapling for any data type
    """
    _y=np.array([])
    f_y=None if len(y.shape)==2 else np.array([])
    mapper=Mapper()
    for row in y:
        _y=np.insert(_y,len(_y),mapper.add(array_key(row)))
    X_resampled, y_resampled = overSampling(X,_y)
    for val in y_resampled:
        label=key_to_array(mapper.revrese(val))
        #print(type(label))
        if (type(label) is list):
            f_y=np.array([label]) if f_y is None else np.concatenate((f_y, [label]), axis=0)
        else:
            f_y=np.insert(f_y,len(f_y),label)
    return X_resampled,f_y

def create_smaple_weight(Y):
    """
    create array of weights for each sample
    """
    maped=class_size(Y)
    total=len(Y)
    return [len(Y) / (len(np.unique(Y)) * maped[k]) for k in Y]

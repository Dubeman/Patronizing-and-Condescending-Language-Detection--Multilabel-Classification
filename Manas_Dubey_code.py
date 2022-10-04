import sys
import joblib
import argparse
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm 
from sklearn.neighbors import KNeighborsClassifier
from skmultilearn.adapt import MLkNN
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from skmultilearn.problem_transform import ClassifierChain
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest,chi2 
from tqdm import tqdm
from nltk.tokenize import word_tokenize, sent_tokenize, wordpunct_tokenize, TreebankWordTokenizer, TweetTokenizer, MWETokenizer
from nltk.corpus import stopwords
from nltk import ngrams
from sklearn.metrics import multilabel_confusion_matrix
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
import os
from nltk.tokenize import sent_tokenize, word_tokenize
import csv,warnings
warnings.filterwarnings(action = 'ignore')
import gensim  
import gensim.downloader as api
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument 
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import hamming_loss,accuracy_score




class DontPatronizeMe:

	def __init__(self, train_path, test_path):

		self.train_path = train_path
		self.test_path = test_path
		self.train_task1_df = None
		self.train_task2_df = None
		self.test_set_df = None

	def load_task1(self):
		"""
		Load task 1 training set and convert the tags into binary labels. 
		Paragraphs with original labels of 0 or 1 are considered to be negative examples of PCL and will have the label 0 = negative.
		Paragraphs with original labels of 2, 3 or 4 are considered to be positive examples of PCL and will have the label 1 = positive.
		It returns a pandas dataframe with paragraphs and labels.
		"""
		rows=[]
		with open(os.path.join(self.train_path)) as f:
			for line in f.readlines()[4:]:
				par_id=line.strip().split('\t')[0]
				art_id = line.strip().split('\t')[1]
				keyword=line.strip().split('\t')[2]
				country=line.strip().split('\t')[3]
				t=line.strip().split('\t')[4]#.lower()
				l=line.strip().split('\t')[-1]
				if l=='0' or l=='1':
					lbin=0
				else:
					lbin=1
				rows.append(
					{'par_id':par_id,
					'art_id':art_id,
					'keyword':keyword,
					'country':country,
					'text':t, 
					'label':lbin, 
					'orig_label':l
					}
					)
		df=pd.DataFrame(rows, columns=['par_id', 'art_id', 'keyword', 'country', 'text', 'label', 'orig_label']) 
		self.train_task1_df = df

	def load_task2(self, return_one_hot=True):
		# Reads the data for task 2 and present it as paragraphs with binarized labels (a list with seven positions, "activated or not (1 or 0)",
		# depending on wether the category is present in the paragraph).
		# It returns a pandas dataframe with paragraphs and list of binarized labels.
		tag2id = {
				'Unbalanced_power_relations':0,
				'Shallow_solution':1,
				'Presupposition':2,
				'Authority_voice':3,
				'Metaphors':4,
				'Compassion':5,
				'The_poorer_the_merrier':6
				}
		print('Map of label to numerical label:')
		print(tag2id)
		data = defaultdict(list)
		with open (os.path.join(self.train_path)) as f:
			for line in f.readlines()[4:]:
				par_id=line.strip().split('\t')[0]
				art_id = line.strip().split('\t')[1]
				text=line.split('\t')[2]#.lower()
				keyword=line.split('\t')[3]
				country=line.split('\t')[4]
				start=line.split('\t')[5]
				finish=line.split('\t')[6]
				text_span=line.split('\t')[7]
				label=line.strip().split('\t')[-2]
				num_annotators=line.strip().split('\t')[-1]
				labelid = tag2id[label]
				if not labelid in data[(par_id, art_id, text, keyword, country)]:
					data[(par_id,art_id, text, keyword, country)].append(labelid)

		par_ids=[]
		art_ids=[]
		pars=[]
		keywords=[]
		countries=[]
		labels=[]

		for par_id, art_id, par, kw, co in data.keys():
			par_ids.append(par_id)
			art_ids.append(art_id)
			pars.append(par)
			keywords.append(kw)
			countries.append(co)

		for label in data.values():
			labels.append(label)

		if return_one_hot:
			labels = MultiLabelBinarizer().fit_transform(labels)
		df = pd.DataFrame(list(zip(par_ids, 
									art_ids, 
									pars, 
									keywords,
									countries, 
									labels)), columns=['par_id',
														'art_id', 
														'text', 
														'keyword',
														'country', 
														'label',
														])
		self.train_task2_df = df


	def load_test(self):
		#self.test_df = [line.strip() for line in open(self.test_path)]
		rows=[]
		with open(self.test_path) as f:
			for line in f:
				t=line.strip().split('\t')
				rows.append(t)
		self.test_set_df = pd.DataFrame(rows, columns="par_id art_id keyword country text".split())




def arr_to_lab(x, str,n):
    r = []
    for i in x['label']:
        r.append(i[n])
    r = pd.DataFrame(r) 
    x[str] = r[0]








def preprocessing(args,input):
    #input_data_path = args.load
    preprocessing_df= input 
    
                                   #Stopwords to remove or not
    if args.stopwords:
            print('**removing stopwords**')
            STOPWORDS = set(stopwords.words('english'))
            count = 0
            for i in preprocessing_df['text']:
                    text =  ' '.join([word for word in i.split() if word not in STOPWORDS])
                    preprocessing_df['text'][count] = text
                    count += 1
        
               #pd.read_csv('/Users/manasdubey2022/Desktop/multilabel_cleaned.csv')
        
          #pd.read_csv('/Users/manasdubey2022/Desktop/df_multilabel.csv')
            
        
        
    #checking for lemmatization or stemming    
    if args.preprocess=='l':
        x = preprocessing_df
        print('**lemmatizing**')
        lemmatizer = WordNetLemmatizer()
        X = []
        for sentence in x['text']:
            word_list = word_tokenize(sentence)
            lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
            X.append(lemmatized_output)                                       # iterate through each sentence in the file
        df = pd.DataFrame(X)
        df.rename(columns = {0:'text'}, inplace = True)
        for i in ['unb','sha','pre','aut','met','comp','ptm']:
            df[i] = x[i]
        
        preprocessing_df = df
        
    elif args.preprocess=='s':
        print('**stemming**')
        x = preprocessing_df
        porter = PorterStemmer()
        data = []
        for sentence in x['text']:
            tokenized_words=word_tokenize(sentence)
            stemmed_output = ' '.join([porter.stem(w) for w in tokenized_words])
            data.append(stemmed_output)
            
        df = pd.DataFrame(data)
        df.rename(columns = {0:'text'}, inplace = True)
        for i in ['unb','sha','pre','aut','met','comp','ptm']:
            df[i] = x[i]
            
        preprocessing_df = df    
 
    else:
        preprocessing_df = input
    
    return preprocessing_df

def vectorize(args,input_df):
    
    x=input_df
    
    if args.input_type=='tfidf':
            #create vectorizer object
            k=x.iloc[:,int(x.shape[1])-7:]
            vectorizer=TfidfVectorizer(stop_words='english',ngram_range=(1,3),token_pattern=r'\b\w+\b')
                    
            #fit text data to created object
            tfidf = vectorizer.fit_transform(x['text'])
            terms=vectorizer.get_feature_names()
            tfidf = tfidf.toarray()
            vectorized_df = pd.DataFrame(tfidf)
            vectorized_df=pd.concat((vectorized_df,k),axis=1)
            
            
    elif args.input_type == 'gword2vec':
        print("**making embeddings**")
        wv = api.load('word2vec-google-news-300')  
        data = []
# iterate through each sentence in the file
        for i in range(len(x)):
            for sentence in sent_tokenize(x['text'][i]):
                words = []    
                for word in word_tokenize(sentence):  # tokenize the sentence into words
                    words.append(word.lower()) 
                    data.append(words)
             
        
        model1 = gensim.models.Word2Vec(data, min_count = 1, vector_size = args.features, window = 20, epochs=100, sg = 1)
        google_embed = []
        for i in range(len(data)):
            p = []
            for j in data[i]:
                p.append(model1.wv[j])
    
                google_embed.append(np.mean(p,axis=0))  
                
        df = pd.DataFrame(google_embed)
        df.rename(columns = {0:'text'}, inplace = True)
        for i in ['unb','sha','pre','aut','met','comp','ptm']:
            df[i] = x[i]
        vectorized_df = df
    
        
    return vectorized_df



def score_function(confusion_list): # Confusion matrix measures

  precision_score_1 = confusion_list[1][0]/(confusion_list[1][0]+confusion_list[0][1])

  recall_score_1 = confusion_list[1][0]/(confusion_list[1][1]+confusion_list[1][0])

  f1 = (2*precision_score_1*recall_score_1)/(precision_score_1 + recall_score_1) 

  
  
  return precision_score_1 , recall_score_1, f1
    
def classifier_binary(args):                                 
    
        #args.load gives you path to dataframe which has text and labels
        #input_data_path = args.load       
        input_type = args.input_type
        
        dataframe = pd.read_csv(input_data_path)            
  
        opt2 = args.name
        df_features = dataframe.drop(['label'], axis=1)
        
         
     # Training and Test Split           
        trn_data, tst_data, trn_cat, tst_cat = train_test_split(df_features, dataframe["label"], test_size=0.20, random_state=42,stratify=dataframe["label"])   

    # Naive Bayes Classifier    
        if opt2=='mn':      
            clf=MultinomialNB(alpha=0,fit_prior=True, class_prior=None)  
            clf_parameters = {
            'clf__alpha':(0,1),
            }  
    # SVM Classifier
        elif opt2=='ls': 
            clf = svm.LinearSVC(class_weight='balanced')  
            clf_parameters = {
            'clf__C':(0.1,1,2,10,50,100),
            }   
        elif opt2=='s':
            clf = svm.SVC(kernel='linear', class_weight='balanced')  
            clf_parameters = {
            'clf__C':(0.1,0.5,1,2,10,50,100),
            }   
    
    #k nearest neighbours
        elif opt2=='knn':
            print('\n\t ### Training k nearest neighbour Classifier ### \n')
            clf = KNeighborsClassifier()
            clf_parameters = { 
                'clf__n_neighbors' : [5,7,9,11,13,15],
               'clf__weights' : ['uniform','distance'],
               'clf__metric' : ['minkowski','euclidean','manhattan']}
    
    # Logistic Regression Classifier    
        elif opt2=='lr':    
            clf=LogisticRegression(class_weight='balanced') 
            clf_parameters = {
            'clf__solver':('newton-cg','lbfgs','liblinear'),
            }    
    # Decision Tree Classifier
        elif opt2=='dt':
            clf = DecisionTreeClassifier(random_state=40)
            clf_parameters = {
            'clf__criterion':('gini', 'entropy'), 
            'clf__max_features':( 'sqrt', 'log2'),
            'clf__ccp_alpha':(0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1),
            }  
    # Random Forest Classifier    
        elif opt2=='rf':
            clf = RandomForestClassifier(class_weight='balanced', max_depth=10)
            clf_parameters = {
                        'clf__criterion':('gini', 'entropy'), 
                        'clf__max_features':( 'sqrt', 'log2'),   
                        'clf__n_estimators':(30,50,100,200),
                        'clf__max_depth':(10,20),
                        }     
        else:
            print('!!!!!!!!! Wrong Input !!!!!!!!! \n')
            sys.exit(0)                                  
    # Feature Extraction      
        pipeline = Pipeline([('clf', clf)])   
            
        parameters={**clf_parameters} 
        grid = GridSearchCV(pipeline,parameters,scoring='f1_micro',cv=10)          
        grid.fit(trn_data,trn_cat)     
        clf= grid.best_estimator_
            
        predicted = clf.predict(tst_data)
        predicted =list(predicted)
            
            
        if args.save:
            filename = f'{opt2}{input_type}.sav'
            joblib.dump(clf, filename)
            
        print('********* Best Set of Parameters ********* \n\n')
        print(clf)


    # Evaluation
        print('\n Total documents in the training set: '+str(len(trn_data))+'\n')    
        print('\n Total documents in the test set: '+str(len(tst_data))+'\n')
        print ('\n Confusion Matrix \n')  
        print (confusion_matrix(tst_cat, predicted))  

        pr=precision_score(tst_cat, predicted, average='binary') 
        print ('\n Precision:'+str(pr)) 

        rl=recall_score(tst_cat, predicted, average='binary') 
        print ('\n Recall:'+str(rl))

        fm=f1_score(tst_cat, predicted, average='binary') 
        print ('\n Micro Averaged F1-Score:'+str(fm))

def classifier_multi_label(args,input_df) :
    print("***Classifying**")
    #args.load gives you path to dataframe which has text and labels
    #input_data_path = args.load       
    input_type = args.input_type    
    dataframe = input_df.dropna()        #pd.read_csv(input_data_path)  
        
    #else:
        #dataframe = vectorize(args)
               
    y = np.asarray(dataframe[dataframe.columns[int(dataframe.shape[1]-7):]])
    opt2 = args.name
    df_features = dataframe.iloc[:,:int(dataframe.shape[1]-7)] 
    trn_data, tst_data, trn_cat, tst_cat = train_test_split(df_features, y, test_size=0.15, random_state=42, shuffle = True)   
    
    if opt2 == 'BR':
        clf = BinaryRelevance() 


        clf_parameters =   {

      
      #  clf_parameters = {
       #                 'classifier' : [RandomForestClassifier()],
       #                'clf__max_features':( 'sqrt', 'log2'),   
       #                 'clf__n_estimators':(30,50,100,200),
       #                 'clf__max_depth':(10,20),

       #                 }    
        
         'classifier': [SVC()],
         'classifier__kernel': ['rbf', 'linear'],
         
         
        
        }

           
   #     clf_parameters = {
    #           'classifier' : [KNeighborsClassifier()],
    #          'clf__n_neighbors' : [5,7,9,11,13,15],
    #           'clf__weights' : ['uniform','distance'],
    #          'clf__metric' : ['minkowski','euclidean','manhattan']}
   
  
    if opt2 == 'mlknn': 
            clf =MLkNN()
            
            clf_parameters = {'k': range(1,5), 
                            's': [0.5, 0.7, 1.0]}


    if opt2 == 'classifier_chains':
            clf = ClassifierChain()
                
            clf_parameters = {
            'classifier': [SVC()],
            'classifier__kernel': ['linear','rbf'] 
            }
                                                
        # Feature Extraction       
                
    parameters={**clf_parameters} 
    grid = GridSearchCV(clf,parameters,scoring='f1_macro', n_jobs = -1)          
    grid.fit(trn_data,trn_cat)     
    clf= grid.best_estimator_
                
    predicted = clf.predict(tst_data)
    if args.save:
        filename = f'{opt2}{input_type}.sav'
        joblib.dump(clf, filename)               

    # Evaluation
    print('\n Total documents in the training set: '+str(len(trn_data))+'\n')    
    print('\n Total documents in the test set: '+str(len(tst_data))+'\n')
    print ('\n Confusion Matrix \n')  
    #print (confusion_matrix(tst_cat, predicted))
    cms = multilabel_confusion_matrix(tst_cat,predicted)
    X = []
    for i in range(len(cms)):
        x=score_function(cms[i])
        arr = np.asarray(x)
        X.append(arr)

    df_1 = pd.DataFrame(X)
    df_2 = pd.DataFrame(['unb','sha','pre', 'aut','met','comp','ptm'])
    df_2.rename(columns ={0 : 'pcl_category'}, inplace = True )
    confusion_matrix_df = pd.concat([df_1, df_2], axis=1)

    confusion_matrix_df.rename(columns = {0:'precison',1:'recall',2:'f1score'}, inplace = True)
    print(confusion_matrix_df) 
                   
    print('**************************')
                   
    
    
    pr=precision_score(tst_cat, predicted, average='macro') 
    print ('\n Precision:'+str(pr)) 

    rl=recall_score(tst_cat, predicted, average='macro') 
    print ('\n Recall:'+str(rl))

    fm=f1_score(tst_cat, predicted, average='macro') 
    print ('\n Macro Averaged F1-Score:'+str(fm))
    
    print('ACCURACY SCORE')       
    print(accuracy_score(tst_cat, predicted))
    
    print('HAMMING LOSS')
    print(hamming_loss(tst_cat, predicted))
            
    print('********* Best Set of Parameters ********* \n\n')
    print(clf)

        
if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--load',type=str, default=None, help="load the csv containing text corpora and their labels")
    #parser.add_argument('--load_model',type=str, default=None, help='load doc2vec model') 
    parser.add_argument('--name','-n', type=str, default=None, help="name of the classifier: (mn:multinomial, ls:linearSVC, s:kernelSVC, lr:logistic regression, dt:decision trees, rf:random forest, knn : k-nearest neighbour)")
    parser.add_argument('--input_type', type=str,default=None, help='Doc2vec input(name= doc2vec) or tfidf input(name=tfidf) or google(name= gword2vec) ')
    parser.add_argument('--save', action='store_true', help="name of the classifier")
    parser.add_argument('--preprocess', type=str, help="lemmatize(l) or stemming(s)")
    parser.add_argument('--stopwords', action='store_true', help="remove stopwords")
    parser.add_argument('--features',type=int ,default=None,help='length of feature vector')
    args = parser.parse_args()



    dpm = DontPatronizeMe(args.load, args.load)
    dpm.load_task2()

    dataframe= dpm.train_task2_df[['text','label']]
    count = 0
    for i in ['unb','sha','pre','aut','met','comp', 'ptm']:
            arr_to_lab(dataframe,i,count)
            count += 1
            
              
    df=preprocessing(args,dataframe)
    
    df_2=vectorize(args,df)
    classifier_multi_label(args,df_2)
        

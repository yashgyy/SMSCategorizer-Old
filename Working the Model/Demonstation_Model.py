
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


XY=pd.read_csv('File16.csv')


# In[4]:


XY


# In[5]:


XY.head()


# In[6]:


XY['labels'].value_counts()


# In[7]:


XY.describe()  #Exploatary Data Analysis


# In[8]:


XY['len']=XY['Body'].map(len)


# In[9]:


XY


# #### Following Function Removes Punctuation in String ####

# In[10]:


import string
def Remove_Punctuation(String):
    char1=''
    for X in String:
        if X in string.punctuation:
            char1+=" "
        else:
            char1+=X
    return char1

            
#Function to Remove Punctuation    
    


# In[11]:


XY['BODY1']=XY['Body'].apply(Remove_Punctuation)


# In[12]:


XY['BODY1']


# #### StopWords are Need to Be Removed for Proper Function ####

# ####  StopWords are common English Alphabets in String 

# In[13]:


from nltk.corpus import stopwords
def Remove_Punctuation1(String):
    Mess=stopwords.words('english')
    List1=String.split()
    List2=list()
    for X in List1:
        if X in Mess:
            pass
        else:
            List2.append(X)
    return ' '.join(List2)        
            
    


# In[14]:


XY['Body2']=XY['BODY1'].map(Remove_Punctuation1) 

# The Function Map will Pass every String of the Column and return the Function
    # to pass the newly Formed String


# In[15]:


XY


# In[16]:


from sklearn.feature_extraction.text import CountVectorizer


# #### Count Vectoriser Object Will Form the Dictionary of the Unique Object

# In[17]:


bow_Transformer=CountVectorizer().fit(XY['Body2'])


# In[18]:


print(len(bow_Transformer.vocabulary_))


# #### There are 2334 Unique Words In the Dataset

# In[19]:


#For example

bow_Transformer.get_feature_names()[1001]


# In[20]:


message_bow=bow_Transformer.transform(XY['Body2'])


# #### TFIDF
# 
# * Tf-Idf will Consist of Two terms in Particular and weigh down Term Inverse Document Frequency . Here the TF refers to 
# 
#     * TF, which measures how frequently a term occurs in a document. Since every document is different in length, it is possible that a term would appear much more times in long documents than shorter ones. 
#     
#     * IDF which measures how important a term is. While computing TF, all terms are considered equally important. However it is known that certain terms, such as "is", "of", and "that", may appear a lot of times but have little importance
#     
# ** IDF(t) = log_e(Total number of documents / Number of documents with term t in it) **

# In[21]:


from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer=TfidfTransformer().fit(message_bow)
tfidf4=tfidf_transformer.fit(message_bow)


# In[22]:


## Example of Tf Idf Transformer

message_FINAL=tfidf_transformer.transform(message_bow)


# In[23]:


message_FINAL.shape


# #### For Training the Model We will Use Random Forest Classification which has the added Method of Ensemble 

# In[24]:


from sklearn.ensemble import RandomForestClassifier


# In[25]:




# Here We are Traning ON TO THE WHOLE CATEGORY TO MEASURE THE ROBUTNESS
X=RandomForestClassifier()
X.fit(message_FINAL,XY['labels'])




# In[26]:


Predict=X.predict(message_FINAL)


# In[27]:


from sklearn.metrics import classification_report


# In[28]:


print(classification_report(XY['labels'],Predict))


# Classifying to the Report Well


# In[ ]:





# In[29]:


from sklearn.model_selection import train_test_split


# In[30]:


msg_train, msg_test, label_train, label_test=train_test_split(XY['BODY1'],XY['labels'],test_size=0.2)


# #### Same process Will be Followed For above Msg train

# In[31]:


msg_train


# In[32]:


msg_train1=msg_train.map(Remove_Punctuation)


# In[33]:


msg_train2=msg_train1.map(Remove_Punctuation1)


# In[34]:


msg_train2


# In[35]:


#Count Vectoriser Object


# In[36]:


XX=CountVectorizer()


# In[37]:


Final_Message=XX.fit_transform(msg_train2)


# In[38]:


XXX=TfidfTransformer()


# In[39]:


Final_Message=XXX.fit_transform(Final_Message)


# In[40]:


X.fit(Final_Message,label_train)


# In[ ]:





# In[ ]:





# In[43]:


def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# #### Here WE Need A Pipeline BECAUSE every message Need To go Through This Whole Process OF Vectorising, Transforming and Classification Thats WHy pipeline iS Essential

# In[44]:


from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', RandomForestClassifier()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


# In[45]:


pipeline.fit(msg_train,label_train)


# In[ ]:





# In[47]:


print(classification_report(predictions,label_test))


# In[49]:


from sklearn.externals import joblib


# ### The Following Files Will Save the Whole Model involving all the Language Processing like (Vectorising StopWords) and Final Classification Machine Learning Model all in One File

# In[55]:


joblib.dump(pipeline,"NLTK[RANDOMFOREST].pkl")


# In[51]:


classifier=joblib.load('Model1.pkl')


# In[54]:


classifier.predict(["Hello"])


# In[ ]:





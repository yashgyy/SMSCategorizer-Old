
### The  Sms Categoriser Consist of the Following Folders

* Deployed Model(Nltk+RandomForesr)

    * The Given Folder Consist of the Final Deployed Model ( Random Forest Classifier) in addition with with All the Corresponding NLTK Features . ** The Files are saved in Pickle Format and we can Directly Predict our Incoming Text Messages by making an Object of the Classifier and Passing the Iterator Containig the String **
    * The Python File Consist of the Script that basically serves as a Demonstration of predicting the Labels by Loading an Object of Classifier
    
* Flask(Web Framework)
    
    * With Combination with open source ngrok (that alllows us to make our localhost public) the Python script Consist of ** Flask web Framework ** the allows you to connect Through our Public based Http server and to fetch the contents where we will run our Python script containing the model 
    
* Working The Model

    * The Folder Consist of jupyter Notebook and Http files where our working of making a Random Forest Classifier Model is Shown. The Notebooks strikingly Highlights the Various Steps that goes through Cross Validation, Natural Language Processing so that the Features are in format so as to Pass through the Model, The Training of Machine Learning Algorithm and The Final Classification Report Highlighting Accuracy of Our Training Model
    

* Preparing Datset

    * The Folder shows the Working of pandas so as to Prepare the Dataset. The Dataset are taken from our ** Personal Text Messages ** , the spam Dataset and Singapore Personal Messages Dataset . The text Messages Are preprocessed using The Data Library . The messages from different Sources are then Combined so as to Make a Final csv file File16.csv
    
* ReadSms

    * The Android App allowed us to ** Fetch the Text Messages from Our Phone and convert it into a Text File ** which is ** Not a work of Open Source and especially made for us for this Process**

* SmsDroid 

    * The Open Source Project upon which we will Add the Seperate Activities for our Labelling and Shift The message Accordingly


#### https://www.kaggle.com/rtatman/the-national-university-of-singapore-sms-corpus (Personal Sms Dataset)

#### https://www.kaggle.com/uciml/sms-spam-collection-dataset

#### (Open Source) https://github.com/felixb/smsdroid


```python

```

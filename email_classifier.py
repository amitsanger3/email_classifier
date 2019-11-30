# Import some useful libraries

import numpy as np
import re
from nltk.stem import PorterStemmer
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


class SpamEmail(object):
    """
    Predict an email is spam or non-spam mail.
    """
    
    def __init__(self):
        
        self.X = pd.read_csv('X.csv') # X variable data from X.csv
        self.y = pd.read_csv('y.csv') # y variable data from y.csv
        self.vocabulary_dictionary = pd.read_csv('vocab.csv')['words'].to_list()
        self.X_train = None
        self.X_test = None
        self.y_train = None 
        self.y_test = None
        self.model = None
        self.email = None
        
    def get_model(self):
        return self.model
        
    def split_data(self, train_size, test_size):
        """
        Split our data after shuffling in particular training and testing size.
        :param train_size: float(0 to 1.0)
        :param test_size: float(0 to 1.0)
        :return: None
        """
        
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.y, train_size=train_size, test_size=test_size)
        self.X_train = x_train
        self.X_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        
        return None
    
    def train_model(self, c=0.1, kernel="linear"):
        """
        Train our classification model with SVM
        SVC: C-Support Vector Classification used for classification purpose.
            The implementation is based on libsvm.

        :param c: Penalty parameter C of the error term. (default=0.1)
        :param kernel: Specifies the kernel type to be used in the algorithm. (default=linear)
            It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable.
            If a callable is given it is used to pre-compute the kernel matrix from data matrices;
            that matrix should be an array of shape (n_samples, n_samples).

        :return: None
        """
        
        model = SVC(C=c, kernel=kernel)
        model.fit(self.X_train, self.y_train)
        self.model = model
        
        return None
    
    def prediction(self):
        """
        Now we taking the predictions and accuracy on training and testing
        data to know that our model is working fine or not.

        If our accuracies on both are good like nearly similar then our model is working
        fine otherwise, we need to change C value and train our model again and check our
        accuracies again.
        :return: training_accuracy(float), test_accuracy(float)
        """
        
        predictions_train = self.model.predict(self.X_train)
        predictions_test = self.model.predict(self.X_test)
        y_train1 = np.array(self.y_train[' 0'])
        y_test1 = np.array(self.y_test[' 0'])

        correct_predictions_train = y_train1[predictions_train == y_train1]
        correct_predictions_test = y_test1[predictions_test == y_test1]
                
        return (100*(len(correct_predictions_train)))/len(y_train1), (100*(len(correct_predictions_test)))/len(y_test1)
    
    def email_dictionary(self):
        email_df = pd.DataFrame(self.email.lower().split(), columns=['words'])

        return list(email_df['words'].unique())
    
    def emailVector(self):
        """
        emailVector: This function takes our email text and filter any unwanted letters or objects
        from it, stems all words from it to remove extra similar words and compare all words with
        vocabulary list to give an array vector of size of dictionay list.
        If any words of email match with spam vocabulary list it stored value as 1 in array vector,
        otherwise 0.
        :return: an array vector
        """
        # making every word in lower case
        emailWordsDictionary = self.email_dictionary()

        for word in emailWordsDictionary:
            if re.search('<[^<>]+>', word):
                # Strip all HTML
                # Looks for any expression that starts with < and ends with > and replace
                # it with a space
                emailWordsDictionary[int(emailWordsDictionary.index(word))]=' '
            elif re.search('[0-9]+', word):
                # Handle Numbers
                # Look for one or more characters between 0-9 and replace it with text 'number'
                emailWordsDictionary[int(emailWordsDictionary.index(word))]='number'
            elif re.search('(http|https)://[^\s]*', word):
                # Handle URLS
                # Look for strings starting with http:// or https:// and replace it with text 'httpaddr'
                emailWordsDictionary[int(emailWordsDictionary.index(word))]='httpaddr'
            elif re.search('[^\s]+@[^\s]+', word):
                # Handle Email Addresses
                # Look for strings with @ in the middle and reaplace it with text 'emailaddr'
                emailWordsDictionary[int(emailWordsDictionary.index(word))]='emailaddr'
            elif re.search('[$]+', word):
                # Handle $ sign
                # Look for $ sign and replace it with text 'dollar'
                emailWordsDictionary[int(emailWordsDictionary.index(word))]='dollar'

        special_characters = list(' @$/#.-:&*+=[]?!(){},''">_<;%')
        # To remove all special characters that mentioned in above special_characters list from email text.
        email_string = ' '.join(emailWordsDictionary)
        for char in special_characters:
            email_string.replace(char, '')

        self.email = email_string
        emailWordsDictionary = self.email_dictionary()

        # To remove any non alphanumeric characters
        for word in emailWordsDictionary:
            if re.search('[^a-zA-Z0-9]', word):
                emailWordsDictionary[int(emailWordsDictionary.index(word))]=''

        # Using PorterStemmer to stemming the words            
        ps = PorterStemmer()
        psWordDictionary = []
        for word in emailWordsDictionary:
            if len(word) > 0:
                try:
                    new_word = ps.stem(word)
                except:
                    continue
                if new_word not in psWordDictionary:
                    psWordDictionary.append(new_word)

        # Initializing dictionary_vector as zero vector
        dictionary_vector = np.zeros((1, len(self.vocabulary_dictionary)), dtype=np.int16)

        for word in psWordDictionary:
            if word in self.vocabulary_dictionary:
                # Replace 0 with 1 if word in both list match
                dictionary_vector[0][self.vocabulary_dictionary.index(word)] = 1

        return dictionary_vector

    def isEmailSpam(self, email):
        """
        isEmailSpam: gives the predictions that an email is spam or non-spam.

        :param email: the email text. (string)
        :return: result(a string )
        """
        self.email = email
        if len(self.email) > 0 and self.email != "False":
            email_vec = self.emailVector()
            predictions_email = self.model.predict(email_vec)

            if predictions_email[0] == 1:
                return "Your email is spam..."
            elif predictions_email == 0:
                return "Your email is not a spam..."
        else:
            return "Please enter your email..."

# if __name__ == '__main__':
#    isEmailSpam(email_text)


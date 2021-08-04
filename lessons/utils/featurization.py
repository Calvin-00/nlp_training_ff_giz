import numpy as np
import math


class Bow:
    def __init__(self):
        # the words will be store here
        self.all_word = set()
        #map words to index and reverse
        self.word_to_index = []
        self.index_to_word = []


    def fit(self,document):
        #check if its a list and its not empty
        if type(document) != list or len(document) < 1 or type(document[0]) != str:
            raise TypeError ('You must pass a lis of strings')

        list_sentences = document
        for sent in list_sentences:
            #split the sentence into words
            for word in sent.split():
                # add the words to the set
                self.all_word.add(word)

            for index,word in enumerate(self.all_word):
                self.word_to_index[word] = index
                self.index_to_word[index] = word

    def transform(self,data):
        if len(self.all_word) == 0:
            raise AttributeError('Tou must fit the data first')

        # if tou have one sentence
        if type(data) == str:
            transformed = self._transform_single(data.split())
        
        #if you have a list of sentences
        elif data == list or type(data[0]) == str:
            #create a empty matrix
            transformed = np.empty(len(data),len(self.all_word))

            for row,sentence in enumerate(data):
                #replace row with sentence BoW
                transformed[row] = self._transform_single(data.split())
        else:
            raise TypeError("You must pass either list or string")

        return transformed

    def fit_transform(self,data):
        self.fit(data)

        return self.transform(data)

    def _transform_single(self,list_words):

        #start with an array of zeros with length equal to total words
        transformed = np.zeros(len(self.all_word))
        for word in list_words:
            if word in self.all_word:
                #get the index of the word
                word_index = self.word_to_index[word]
                #increase the value by 1
                transformed[word_index]+=1

        return transformed



class TDIDF:
    def __init__(self,ignore_tokens = ['<SOS>',"<EOS>"],ignore_punctuations=False,lower_case=True):
        self.ignore_tokens = ignore_tokens

        if ignore_punctuations:
            self.ignore_tokens += [char for char in string.punctuation]

        self.lower_case = lower_case
        self.word_to_index = {}
        self.index_to_word = {}
        self.idf = {}
        self.num_documents = 0

    def term_frequency(self,sentence,ignore_tokens=['<SOS>',"<EOS>"],lower_case=False):
        word_dict ={}
        words = [token.lower() if lower_case else token for token in sentence.split() if token not in ignore_tokens]

        for word in words:
            word_dict[word] = word_dict.get(word,0)+1
        return word_dict

    def fit(self,data):

        self.num_documents = len(data)

        global_term_freq = {}
        list_sentences = data

        for sentence in list_sentences:
            words_in_sent = set()

            doc_freq = self.term_frequency(sentence,self.ignore_tokens,self.lower_case)

            for word in doc_freq:
                if word not in words_in_sent:
                    global_term_freq[word] = global_term_freq.get(word,0)+1
                    words_in_sent.add(word)

        #commpute idf = log(total documents/frequency)
        for word, frequency in global_term_freq.items():
            idf = math.log((1+self.num_documents)/(1+frequency))
            self.idf[word] = idf

        document_words = list(global_term_freq.keys())
        for word_position in range(len(document_words)):
            word = document_words[word_position]
            self.word_indexes[word] = word_position
            self.index_to_word[word_position] = word

    def transform(self, data):
        if isinstance(data,list):
            return self._transform_document(data)
        elif isinstance(data,str):
            return self._transform_sentence(data)

    def _transform_document(self,data):
        #used for batch processing
        to_transform = data
        sentence_arrays = []

        for sent in data:
            sentence_arrays.append(self._transform_sentence(sent))

        return np.matrix(sentence_arrays)

    def _tranform_sentence(self,data):
        tokens = [token.lower() if self.lower_case else token for token in data.split()]

        word_array = np.zeros(len(self.word_to_index))
        sent_td_idf = self._compute_sent_td_idf(data)

        for token in tokens:
            if token in self.word_to_index:
                token_index = self.word_to_index[token]
                # Add the tfidf value for each token in sentence to its position in vocabulary array.
                word_array[token_index] = sent_td_idf[token]
        return word_array

    def _compute_sentence_tf_idf(self, sentence):
        """
        Computes the tf_idf for a single sentence(document).
        """
        sentence_tf_idf = {}
        # Gets the document frequency by using the helper method
        document_frequency = self.term_frequency(sentence, self.ignore_tokens, self.lower_case)
        # Gets the total number of words in sentence
        total_words = sum(document_frequency.values())
        # Find individual term frequency value averaged by total number of words.
        averaged_frequency = {k:(float(v)/total_words) for k,v in document_frequency.items()}
        for term, tf in averaged_frequency.items():
            # Out of vocabulary words are simply zeroed. They are going to be removed later either way.
            # Computes the tfidf for each word by using word tf times the term idf
            sentence_tf_idf[term] = tf*self.idf.get(term, 0)
        return sentence_tf_idf



            

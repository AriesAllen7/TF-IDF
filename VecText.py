import numpy as np


class VecText:

    def __init__(self, corpus):
        self.corpus = corpus
        self.setVocabulary(self.extractVocabulary())

    def removeStopWords(self, stopwords):
        for i in range(len(self.corpus)):
            document = self.corpus[i]
            token_sequence = str.split(document)
            filtered_document = [w for w in token_sequence if not w in stopwords]
            self.corpus[i] = ' '.join(filtered_document)
            self.getVocabulary()

    def extractVocabulary(self):
        vocab = sorted(set(' '.join(self.corpus).split()))
        return vocab

    def setVocabulary(self, vocab):
        self.vocabulary = vocab

    def getVocabulary(self):
        return self.vocabulary

    def bagOfWords(self):
        BoWList = []
        for i in range(len(self.corpus)):
            BoW = {}
            document = self.corpus[i]
            token_sequence = str.split(document)
            for word in token_sequence:
                if word in BoW:
                    BoW[word] = BoW[word]+1
                else:
                    BoW[word] = 1
            BoWList.append(BoW)
        return BoWList

    def TF(self):
        TF = np.zeros((len(self.corpus), len(self.vocabulary)))
        BoWDoc = self.bagOfWords()
        for i in range(len(self.corpus)):
            BoW = BoWDoc[i]
            numWords = 0
            for j, word in enumerate(self.vocabulary):
                if word in BoW:
                    TF[i, j] = BoW[word]
                    numWords = numWords+BoW[word]
            TF[i, :] = TF[i, :]/numWords
        return TF

    def IDF(self):
        IDF = np.zeros((len(self.corpus), len(self.vocabulary)))
        BoWDoc = self.bagOfWords()
        for i in range(len(self.corpus)):
            BoW = BoWDoc[i]
            for j, word in enumerate(self.vocabulary):
                if word in BoW:
                    IDF[:, j] = IDF[1, j]+1
        return np.log10(len(self.corpus)/IDF)

    @staticmethod
    def TF_IDF(TF, IDF):
        return TF*IDF

    @staticmethod
    def cosSim(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))

    def query2vec(self, idf, query):
        vocab = self.getVocabulary()
        vec = np.zeros((1, len(vocab)))
        token_sequence = query.split(' ')
        BoW = {}
        for word in token_sequence:
            if word in BoW:
                BoW[word] = BoW[word] + 1
            else:
                BoW[word] = 1
        for i, word in enumerate(vocab):
            if word in BoW:
                vec[0, i] = BoW[word]/len(token_sequence)
            else:
                vec[0, i] = 0
        return vec*idf[1, :]

#!/usr/bin/env python

"""The simplest TF-IDF library imaginable.

Add your documents as two-element lists `[docname,
[list_of_words_in_the_document]]` with `addDocument(docname, list_of_words)`.
Get a list of all the `[docname, similarity_score]` pairs relative to a
document by calling `similarities([list_of_words])`.

See the README for a usage example.

"""
import numpy as np
import math
class TfIdf:
    def __init__(self):
        self.weighted = False
        self.documents = {}
        self.corpus_dict = {}
        self.sims = {}
    def add_document(self, doc_name, list_of_words):
        # building a dictionary
        doc_dict = {}
        for w in list_of_words:
            doc_dict[w] = doc_dict.get(w, 0.) + 1.0
            self.corpus_dict[w] = self.corpus_dict.get(w, 0.0) + 1.0
        # normalizing the dictionary
        length = float(len(list_of_words))
        
        for k in doc_dict:
            doc_dict[k] = 1 + doc_dict[k] / length

        # add the normalized document to the corpus  (TF)
        self.documents[doc_name]= doc_dict
    def similarities(self, list_of_words, queryName):
        """Returns a list of all the [docname, similarity_score] pairs relative to a
list of words.

        """

        # building the query dictionary
        
        query_dict = {}
        for w in list_of_words:
            query_dict[w] = query_dict.get(w, 0.0) + 1.0

        # normalizing the query
        length = float(len(list_of_words))

        for k in query_dict:
            query_dict[k] = 1+ query_dict[k] / length
        
        # computing the list of similarities
        scoreDic = {}
        for doc in self.documents:
            #每一篇文件要做的事情
            #1. 讀出名字跟dic
            #2. 比對是否存在該字,有的話計算tfidf 沒有則0 ,存在list中
            #3. 把list 轉換為array之後進行dot product
 #4. 最後把結果化為 sim:{q1:{D1:值,D2:值},q2:{...},q3{...}}
            docTFIDF = []
            qTFIDF=[]
            lengthDoc=0
            lengthQuery=0
            score = 0
            #1 得到該document的字典
            dicTemp = self.documents[doc]
            
            #2.
            for w in self.corpus_dict:
                if w in dicTemp:
                    docTFIDF.append(dicTemp[w]*math.log10((2265/self.corpus_dict[w])))     
                else :
                    docTFIDF.append(0)
                
                if w in query_dict:
                    qTFIDF.append(query_dict[w]*math.log10((2265/self.corpus_dict[w])))
                else:
                    qTFIDF.append(0)

            #3.
            arrayQuery = np.array(qTFIDF)
            arrayDoc = np.array(docTFIDF)
            
            lengthDoc = np.sqrt(arrayDoc.dot(arrayDoc))
            lengthQuery = np.sqrt(arrayQuery.dot(arrayQuery))

            score = arrayQuery.dot(arrayDoc)/(lengthDoc*lengthQuery)
            
            scoreDic[doc]=score
        
        #4.    
        self.sims[queryName] = sorted(scoreDic.items(), key=lambda d:d[1], reverse = True)
        #做排序    
            
            
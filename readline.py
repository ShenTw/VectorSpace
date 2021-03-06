import tfidf2
table = tfidf2.TfIdf()
with open('doc_list.txt','r' ) as L:
    for line0 in L:
        line0 = line0.strip('\n')
        path = "C:/Users/shen/.spyder-py3/Document/"
        file = path+line0
        with open(file,'r') as f:
            
            listOfWords=[]
            temp= []
            #每篇doc的初始化材料
            for line in f.readlines() :
                #    data=f.readline()
                line=line.strip('\n')
                line=line.strip('-1')
                temp = temp + line.split()
                #temp為該doc的字典(只有字)
                # print("doc : \n" ,line0, "temp : ", temp)
            for i in range(5,len(temp),+1):
                    #扣除前三行不需要的資訊(經切割後為五項)
                listOfWords.append(temp[i])
                 
                #listOfwords為該doc的字典 (字:次數)
                
                #print("Doc: ", line0, "words",listOfWords)
    
            table.add_document(line0, listOfWords)
            #存成一個documents 文件集 {doc:{字:字數}}
            #length = len(result)
            #總文件數量
  #          numpy.array([for word in result])
            

with open('query_list.txt','r') as Q:

    for line0 in Q:
        line0 = line0.strip('\n')
        path = "C:/Users/shen/.spyder-py3/Query/"
        file = path+line0
        with open(file,'r') as f:
            
            listOfWords = []
            temp = []
            for line in f.readlines():
                line = line.strip('\n')
                line = line.strip('-1')
                temp = temp + line.split()
                
            for i in range(0,len(temp),+1):
                listOfWords.append(temp[i])
            
            
        table.similarities(listOfWords,line0)
       
        #print("Query: ", line0,"is ",listOfWords) 

print("prepare to finalize!")     

count =0
for doc in table.sims['20002.query']:
    if count<13:
        print("result is : ", doc)
        count=count+1
    else:
        break

#print("result is : \n",table.sims['20002.query'])          
#print ("corpus is : ",table.corpus_dict)
            # total times
#print ("\ndocument is : " ,table.documents['VOM19980220.0700.0166']) 
        # documents : tf 

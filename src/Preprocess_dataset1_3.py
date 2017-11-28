import csv




def preprocess(list_of_files, list_of_non_numerical_columns, classify):
    top_map = dict()
    last_counts = dict()
    avg_val_per_column = []
    feature = []
    pred = []
    for j in list_of_non_numerical_columns:
        low_map = dict()
        top_map[j] = low_map
        last_counts[j] = 0
        
    for name in list_of_files:
        with open(name) as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader, None)
            if len(avg_val_per_column) == 0:
                head_arr = []
                intit_avg_array(avg_val_per_column, len(header))
                for i in range(0, len(header)):
                    head_arr.append(header[i])
                feature.append(head_arr)
                    
            for row in reader:
                f_train = []
                for i in range(0, len(row)):
                    if  i not in list_of_non_numerical_columns:
                        val = process(row[i],avg_val_per_column,i)
                    else:
                        val = fetch_num_value(row[i],top_map[i],last_counts,i,avg_val_per_column)
                    f_train.append(val)
                pred.append(f_train[-1])
                #del f_train[-1]
                if classify == 1:
                    f_train[-1] = int(f_train[-1])
                feature.append(f_train)
    replace_missing_values(feature,avg_val_per_column)
    if classify == 1:
        pred = [int(p) for p in pred]
        
    
    return (feature,pred)              
                
        
def compute_avgs(avg):
    fin_avg_array = []
    for val in avg:
        f_val = val[0]/val[1]
        fin_avg_array.append(round(f_val))
    return fin_avg_array
        

def replace_missing_values(feature,avgs):
    avgs = compute_avgs(avgs)
    #print avgs
    for f in feature:
        for i in range(0, len(f)):
            if f[i] == 'Missing':
                f[i] = avgs[i]
    
def intit_avg_array(arr,n):
    for j in range(0,n):
        arr.append([float(0),0])
        
def process(val,avg_val_per_column,i):
    if val in (None,""):
        return 'Missing'
    else:
        avg_val_per_column[i][0] = avg_val_per_column[i][0] + float(val)
        avg_val_per_column[i][1] = avg_val_per_column[i][1] + 1
        return float(val)

def fetch_num_value(val, l_map,last_counts,i,avg_val_per_column):
    if val in (None,""):
        return 'Missing'
    if val not in l_map:
        l_map[val] = float(last_counts[i])
        last_counts[i] = last_counts[i] + 1
    avg_val_per_column[i][0] = avg_val_per_column[i][0] + l_map[val]
    avg_val_per_column[i][1] = avg_val_per_column[i][1] + 1
    return l_map[val]


(f,p) =preprocess(['../data/raw/student-mat.csv','../data/raw/student-por.csv'],[0,1,3,4,5,8,9,10,11,15,16,17,18,19,20,21,22],0)
#(f,p) =preprocess(['student-mat.csv'],[0,1,3,4,5,8,9,10,11,15,16,17,18,19,20,21,22],0)
(f1,p1) =preprocess(['../data/raw/xAPI-Edu-Data.csv'],[0,1,2,3,4,5,6,7,8,13,14,15,16],1)

# ../data/dataset2/mergedata.csv    
myFile = open('../data/dataset1.csv', 'wb')
with myFile:  
   writer = csv.writer(myFile)
   writer.writerows(f)     
            
myFile = open('../data/dataset3.csv', 'wb')
with myFile:  
   writer = csv.writer(myFile)
   writer.writerows(f1)            
                    
                    
                    
                
                    
            
            


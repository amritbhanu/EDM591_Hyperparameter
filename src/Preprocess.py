import csv

def preprocess(name, list_of_non_numerical_columns):
    top_map = dict()
    last_counts = dict()
    feature = []
    pred = []
    for j in list_of_non_numerical_columns:
        low_map = dict()
        top_map[j] = low_map
        last_counts[j] = 0
        
    with open(name) as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)
        for row in reader:
            f_train = []
            for i in range(0, len(row)):
                if  i not in list_of_non_numerical_columns:
                    val = process(row[i])
                else:
                    val = fetch_num_value(row[i],top_map[i],last_counts,i )
                f_train.append(val)
            pred.append(f_train[-1])
            del f_train[-1]
            feature.append(f_train)
    return (feature,pred)

def process(val):
    if val in (None,""):
        return float(0)
    else:
        return float(val)

def fetch_num_value(val, l_map,last_counts,i):
    if val in (None,""):
        return float(0)
    if val not in l_map:
        l_map[val] = float(last_counts[i])
        last_counts[i] = last_counts[i] + 1
    return l_map[val]


(f,p) =preprocess('student-mat.csv',[0,1,3,4,5,8,9,10,11,15,16,17,18,19,20,21,22])
        
    
        
            
            
                    
                    
                    
                
                    
            
            


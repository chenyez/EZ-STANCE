import pandas as pd

file = '/home/czhao43/dataset_stance/tweet/data_subtaskA/raw_train_all_onecol.csv'
df = pd.read_csv(file)
tweets = [row['Text'] for idx, row in df.iterrows()]

length = [len(x.split(' ')) for x in tweets]
print("max:",max(length))
print("min:",min(length))
print("average:",sum(length)/len(length))


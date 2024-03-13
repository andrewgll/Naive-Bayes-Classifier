import pandas as pd

# This dataset is crafted from few datasets I found on Kaggle:
#
# https://www.kaggle.com/datasets/team-ai/spam-text-message-classification
# https://www.kaggle.com/datasets/venky73/spam-mails-dataset
# https://www.kaggle.com/datasets/shantanudhakadd/email-spam-detection-dataset-classification
#
# Just download it and rename to 1.csv, 2.csv, 3.csv and 4.csv If you want to run this

df = pd.read_csv("datasets/1.csv")  # Replace "your_dataset.csv" with the path to your dataset
df = df.dropna()
df = df.drop(axis=1, labels="Unnamed: 0")
df.rename(columns={'Body': 'Email'}, inplace=True)

df1 = pd.read_csv("datasets/2.csv")  # Replace "your_dataset.csv" with the path to your dataset
df1 = df1.drop(axis=1, labels=['Unnamed: 0.1', 'Unnamed: 0',])
df1.rename(columns={'Body': 'Email'}, inplace=True)


df2 = pd.read_csv("datasets/3.csv")  # Replace "your_dataset.csv" with the path to your dataset
df2.rename(columns={'Category': 'Label', 'Message': 'Email'}, inplace=True)
df2['Label'] = df2['Label'].apply(lambda x: 1 if x == 'spam' else 0)
df2 = df2[['Email', 'Label']]

df3 = pd.read_csv("datasets/4.csv")  # Replace "your_dataset.csv" with the path to your dataset
df3 = df3.drop(axis=1, labels=['label_num', 'Unnamed: 0',])
df3.rename(columns={'label': 'Label', 'text': 'Email'}, inplace=True)
df3['Label'] = df3['Label'].apply(lambda x: 1 if x == 'spam' else 0)
df3 = df3[['Email', 'Label']]

combined_df = pd.concat([df, df1, df2, df3])
combined_df.to_csv("dataset.csv")

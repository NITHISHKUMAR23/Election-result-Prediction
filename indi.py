import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px

BJP_NDA  = pd.read_csv('C:/Users/NIMMU/Desktop/project/Modi tweets_review.csv')
print(BJP_NDA)
INC_UPA = pd.read_csv('C:/Users/NIMMU/Desktop/project/Rahul tweets_review.csv')
print(INC_UPA)

BJP_NDA = pd.read_csv('C:/Users/NIMMU/Desktop/project/Modi tweets_review.csv')
INC_UPA = pd.read_csv('C:/Users/NIMMU/Desktop/project/Rahul tweets_review.csv')

print(BJP_NDA.head())
print(INC_UPA.head())

textblob1 = TextBlob(BJP_NDA["Tweet"][100])
print("MODI :",textblob1.sentiment)

len(BJP_NDA)
len(INC_UPA)

textblob2 = TextBlob(INC_UPA["Tweet"][10000])
print("Rahul :",textblob2.sentiment)

BJP_NDA["Tweet"][2787]

INC_UPA["Tweet"][2539]

def find_pol(review):
    if not isinstance(review, str):
        review = str(review)
    return TextBlob(review).sentiment.polarity

BJP_NDA["Sentiment Polarity"] = BJP_NDA["Tweet"].apply(find_pol)
print(BJP_NDA.tail())

INC_UPA["Sentiment Polarity"] = INC_UPA["Tweet"].apply(find_pol)
print(INC_UPA.tail())

BJP_NDA["Expression Label"] = np.where(BJP_NDA["Sentiment Polarity"]>0, "positive", "negative")
BJP_NDA["Expression Label"][BJP_NDA["Sentiment Polarity"]==0]="Neutral"
print(BJP_NDA.tail())

INC_UPA["Expression Label"] = np.where(INC_UPA["Sentiment Polarity"]>0, "positive", "negative")
INC_UPA["Expression Label"][BJP_NDA["Sentiment Polarity"]==0]="Neutral"
print(INC_UPA.tail())

reviews1 = BJP_NDA[BJP_NDA['Sentiment Polarity'] == 0.0000]
print(reviews1.shape)

cond1=BJP_NDA['Sentiment Polarity'].isin(reviews1['Sentiment Polarity'])
BJP_NDA.drop(BJP_NDA[cond1].index, inplace = True)
print(BJP_NDA.shape)

reviews2 = INC_UPA[INC_UPA['Sentiment Polarity'] == 0.0000]
print(reviews2.shape)

cond2=INC_UPA['Sentiment Polarity'].isin(reviews1['Sentiment Polarity'])
INC_UPA.drop(INC_UPA[cond2].index, inplace = True)
print(INC_UPA.shape)

# BJP & NDA
np.random.seed(10)
remove_n =324
drop_indices = np.random.choice(BJP_NDA.index, remove_n, replace=False)
df_subset_Modi = BJP_NDA.drop(drop_indices)
print(df_subset_Modi.shape)

# INC
np.random.seed(10)
remove_n =31
drop_indices = np.random.choice(INC_UPA.index, remove_n, replace=False)
df_subset_Rahul = INC_UPA.drop(drop_indices)
print(df_subset_Rahul.shape)

count_1 = df_subset_Modi.groupby('Expression Label').count()
print(count_1)

negative_per1 = (count_1['Sentiment Polarity'][0]/1000)*10
positive_per1 = (count_1['Sentiment Polarity'][1]/1000)*100

count_2 = df_subset_Rahul.groupby('Expression Label').count()
print(count_2)

negative_per2 = (count_2['Sentiment Polarity'][0]/1000)*100
positive_per2 = (count_2['Sentiment Polarity'][1]/1000)*100

Politicians = ['MODI','RAHUL']
lis_pos = [positive_per1, positive_per2]
lis_neg = [negative_per1, negative_per2]

fig = go.Figure(data=[
    go.Bar(name='Possibility to win', x=Politicians, y=lis_pos),
    go.Bar(name='Possibility to lose', x=Politicians, y=lis_neg)
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.show()

import plotly.graph_objects as go

# Assuming you have the positive and negative percentages for each politician
Politicians = ['MODI', 'RAHUL']
positive_per1 = 60  # Replace with actual value
positive_per2 = 40  # Replace with actual value
negative_per1 = 20  # Replace with actual value
negative_per2 = 30  # Replace with actual value

# Combine positive and negative percentages into a single list for pie chart
data = [positive_per1, negative_per1, positive_per2, negative_per2]

# Create labels for each slice (politician and sentiment)
labels = []
for politician, pos_per, neg_per in zip(Politicians, data[::2], data[1::2]):
    labels.append(f"{politician} (Positive)")
    labels.append(f"{politician} (Negative)")

# Create the pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=data)])

# Customize the pie chart (optional)
fig.update_traces(textposition='inside', textinfo='percent+label')  # Show labels and percentages inside slices

fig.show()


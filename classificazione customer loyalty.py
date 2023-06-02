#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 15:02:41 2023

@author: Chiara
"""

from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# download del primo dataset 
df = pd.read_csv("Downloads/ContactActive.csv", sep=",")

df.shape
df.head()

# elimino da CRM Type tutte le righe che non contengono Costumer
df = df[df['CRM Type'] != 'Lead']
df = df[df['CRM Type'] != 'Company']
df = df[df['CRM Type'] != 'Employee']
df = df[df['CRM Type'] != 'Ex- Employee']
df = df[df['CRM Type'] != 'Family']
df = df[df['CRM Type'] != 'Generic']
df = df[df['CRM Type'] != 'Press']
df = df[df['CRM Type'] != 'Stylist']

df.shape
df.head()
df.isnull().sum()

# eliminare tutte le features con tutti gli elementi nulli
df.dropna(axis='columns', how='all', inplace=True)

# elimino le colonne con valori mancanti superiori al 50 % e che non mi servono
df = df.drop(['Contact', '% Permanent Discount', 'Birthplace', 'Birthday', 'Day of Birth', 'Month of Birth', 'Year of Birth', 'Age Group', 'Age', 'City Counter', 'Postcode', 'State/Region', 'Country of residence Counter', 'Email Counter','Primary Telephone Counter', 'Mobile Counter', 'Wechat ID Counter', 'Best Region (Area)', 'Best Store', 'Spending', 'First Purchase Date', 'First Purchase Store (Created Store)', 'Last Purchase Date', 'Last Purchase Store'], axis=1)
df = df.drop(['Signature', 'Date Signature', 'Privacy Policy', 'Total Amount_CY (Base)', 'Data Privacy Policy', 'Update Consent for Marketing','Update Data CM', 'Update Consent for Analysis', 'Reachable', 'RFM Frequency', 'RFM Monetary', 'RFM Score', 'RFM Recency'], axis=1)
df = df.drop(['N Holiday Gift Received', 'N Travel set received', 'Loyalty Program (vuota)', 'Score (vuota)', 'Activation Date (vuota)', 'Expiration Date', 'Subscription', 'Date Loyalty','Website ID', 'Web Site Language', 'NS Subscription Form', 'Newsletter Subscription Date', 'Totale email cliccate', 'Totale email ricevute', 'Totale email spedite'], axis=1)
df = df.drop(['Ultima email aperta', 'Ultima email cliccata', 'Ultima email spedita', 'Last Interaction Date', 'N of transactions_CY', 'N of Transactions_LY', 'Total Transactions', 'Average Ticket (Base)','Average Ticket_CY (Base)', 'Average Ticket_LY (Base)', 'Average Transaction_CY', 'Average Transaction_LY', 'Average Transaction', 'Total Amount_CY', 'Total Amount_LY'], axis=1)
df = df.drop(['Total Amount', 'Total Amount_LY (Base)', 'Total Amount (Base)', 'Quantity_CY', 'Quantity_LY', 'Total Quantity', 'UPT_CY', 'UPT_LY', 'UPT', 'Return Quantity_CY','Return Quantity_LY', 'Total Return Quantity', 'Mtm Total Amount CY', 'Mtm Total Amount LY', 'Mtm Total amount (Last 7 years)', 'Mtm Total Amount CY (Base)'], axis=1)
df = df.drop(['Mtm Total Amount LY (Base)', 'Mtm Total amount (Last 7 years) (Base)', 'Mtm Total Quantity CY', 'Mtm Total Quantity LY', 'Mtm Total Quantity (Last 7 years)', 'Best Segment by Amount CY', 'Best Segment by Amount LY', 'Best Segment by Amount', 'Best Segment by Qty CY', 'Best Segment by Qty LY', 'Best Segment by Qty','# of Cards', 'Full Birthday Counter', 'Birthday DAY MONTH Counter', 'Reachable by @ OR mobile Counter', 'Reachable by @ or Tel or Mobile or WeChat Counter', 'Reachable by @ AND mobile Counter', 'Reachable by WeChat Counter', 'Best Sales Staff ID', 'Last Sales Staff ID', 'First Sales Staff ID'], axis=1)
df = df.drop(['Update  Data CA', 'Title'], axis=1)
df = df.drop(['Date CM','Date CA', 'Date SMS', 'Date CN', 'Date of Creation','Country of residence', 'City'], axis=1)

df.columns
df.shape

df.dtypes
df.isnull().sum()

# sostituisco i dati mancanti delle numeriche con la media
df['Data Collection Score'] = df['Data Collection Score'].fillna(df['Data Collection Score'].mean())

# sostituisco i dati mancanti delle categoriche con il valore più frequente
df['CRM Type'] = df['CRM Type'].fillna(df['CRM Type'].mode()[0])
df['Macro Area'] = df['Macro Area'].fillna(df['Macro Area'].mode()[0])
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Loyalty'] = df['Loyalty'].fillna(df['Loyalty'].mode()[0])
df['Consent for Marketing'] = df['Consent for Marketing'].fillna(df['Consent for Marketing'].mode()[0])
df['Consent for Analysis'] = df['Consent for Analysis'].fillna(df['Consent for Analysis'].mode()[0])
df['Consent for SMS'] = df['Consent for SMS'].fillna(df['Consent for SMS'].mode()[0])
df['Pref Email'] = df['Pref Email'].fillna(df['Pref Email'].mode()[0])
df['Pref Sms'] = df['Pref Sms'].fillna(df['Pref Sms'].mode()[0])
df['Pref Telephone'] = df['Pref Telephone'].fillna(df['Pref Telephone'].mode()[0])
df['Pref Mail'] = df['Pref Mail'].fillna(df['Pref Mail'].mode()[0])
df['Sales Habit'] = df['Sales Habit'].fillna(df['Sales Habit'].mode()[0])
df['Customer Habit'] = df['Customer Habit'].fillna(df['Customer Habit'].mode()[0])

df.isnull().sum()

#controllo duplicati
df.duplicated().sum()

# trasformo la variabile Loyalty in binaria
df['Loyalty'].value_counts()
df['Loyalty'] = df['Loyalty'].replace('12. Lost', 'Infedele')
df['Loyalty'] = df['Loyalty'].replace('11. Inactive', 'Infedele')
df['Loyalty'] = df['Loyalty'].replace('03. Prospect Store', 'Infedele')
df['Loyalty'] = df['Loyalty'].replace('10. Sleeper', 'Infedele')
df['Loyalty'] = df['Loyalty'].replace('04. New', 'Infedele')
df['Loyalty'] = df['Loyalty'].replace('07. Occasional Reactivated', 'Infedele')
df['Loyalty'] = df['Loyalty'].replace('06. Occasional Retained', 'Infedele')
df['Loyalty'] = df['Loyalty'].replace('08. Loyal Retained', 'Leale')
df['Loyalty'] = df['Loyalty'].replace('05. New Loyal', 'Leale')
df['Loyalty'] = df['Loyalty'].replace('09. Loyal Reactivated', 'Leale')

# unisco i valori
df['Customer Habit'].value_counts()
df['Customer Habit'] = df['Customer Habit'].replace('Traveller', 'Tourist')
df['Customer Habit'] = df['Customer Habit'].replace('International', 'Tourist')

df['Macro Area'].value_counts()
df['Macro Area'] = df['Macro Area'].replace('North America', 'USA')
df['Macro Area'] = df['Macro Area'].replace('Central America', 'USA')
df['Macro Area'] = df['Macro Area'].replace('South America', 'USA')
df['Macro Area'] = df['Macro Area'].replace('USA', 'America')

df['Macro Area'] = df['Macro Area'].replace('China', 'Asia')
df['Macro Area'] = df['Macro Area'].replace('Greather Russia', 'Asia')
df['Macro Area'] = df['Macro Area'].replace('Hong Kong', 'Asia')
df['Macro Area'] = df['Macro Area'].replace('India', 'Asia')
df['Macro Area'] = df['Macro Area'].replace('Macau', 'Asia')
df['Macro Area'] = df['Macro Area'].replace('Pakistan', 'Asia')

df['Macro Area'] = df['Macro Area'].replace('Italy', 'Europe')
df['Macro Area'] = df['Macro Area'].replace('UK', 'Europe')

df['Macro Area'] = df['Macro Area'].replace('nd', 'Africa')
df['Macro Area'] = df['Macro Area'].replace('Middle East', 'Africa')
df['Macro Area'] = df['Macro Area'].replace('Oceania', 'Africa')
df['Macro Area'] = df['Macro Area'].replace('Africa', 'Altro')

# encoder
Loyalty = LabelEncoder()
df['Loyalty'] = Loyalty.fit_transform(df['Loyalty'])
Loyalty.classes_
df['Loyalty'].value_counts(dropna=False)

Gender = LabelEncoder()
df['Gender'] = Gender.fit_transform(df['Gender'])
Gender.classes_
df['Gender'].value_counts(dropna=False)

df = df.rename(columns={'Macro Area': 'Macro_Area'})
Macro_Area = LabelEncoder()
df['Macro_Area'] =  Macro_Area.fit_transform(df['Macro_Area'])
df['Macro_Area'].value_counts(dropna=False)

df = df.rename(columns={'Consent for Marketing': 'Consent_for_Marketing'})
Consent_for_Marketing = LabelEncoder()
df['Consent_for_Marketing'] =  Consent_for_Marketing.fit_transform(df['Consent_for_Marketing'])
df['Consent_for_Marketing'].value_counts(dropna=False)

df = df.rename(columns={'Consent for Analysis': 'Consent_for_Analysis'})
Consent_for_Analysis = LabelEncoder()
df['Consent_for_Analysis'] =  Consent_for_Analysis.fit_transform(df['Consent_for_Analysis'])
df['Consent_for_Analysis'].value_counts(dropna=False)

df = df.rename(columns={'Consent for SMS': 'Consent_for_SMS'})
Consent_for_SMS = LabelEncoder()
df['Consent_for_SMS'] =  Consent_for_SMS.fit_transform(df['Consent_for_SMS'])
df['Consent_for_SMS'].value_counts(dropna=False)

df = df.rename(columns={'Consent for Newsletter': 'Consent_for_Newsletter'})
Consent_for_Newsletter = LabelEncoder()
df['Consent_for_Newsletter'] =  Consent_for_Newsletter.fit_transform(df['Consent_for_Newsletter'])
df['Consent_for_Newsletter'].value_counts(dropna=False)

df = df.rename(columns={'Pref Email': 'Pref_Email'})
Pref_Email = LabelEncoder()
df['Pref_Email'] =  Pref_Email.fit_transform(df['Pref_Email'])
df['Pref_Email'].value_counts(dropna=False)

df = df.rename(columns={'Pref Sms': 'Pref_Sms'})
Pref_Sms = LabelEncoder()
df['Pref_Sms'] =  Pref_Sms.fit_transform(df['Pref_Sms'])
df['Pref_Sms'].value_counts(dropna=False)

df = df.rename(columns={'Pref Telephone': 'Pref_Telephone'})
Pref_Telephone = LabelEncoder()
df['Pref_Telephone'] =  Pref_Telephone.fit_transform(df['Pref_Telephone'])
df['Pref_Telephone'].value_counts(dropna=False)

df = df.rename(columns={'Pref Mail': 'Pref_Mail'})
Pref_Mail = LabelEncoder()
df['Pref_Mail'] =  Pref_Mail.fit_transform(df['Pref_Mail'])
df['Pref_Mail'].value_counts(dropna=False)

df = df.rename(columns={'Sales Habit': 'Sales_Habit'})
Sales_Habit = LabelEncoder()
df['Sales_Habit'] =  Sales_Habit.fit_transform(df['Sales_Habit'])
df['Sales_Habit'].value_counts(dropna=False)

df = df.rename(columns={'Customer Habit': 'Customer_Habit'})
Customer_Habit = LabelEncoder()
df['Customer_Habit'] =  Customer_Habit.fit_transform(df['Customer_Habit'])
df['Customer_Habit'].value_counts(dropna=False)

df = df.rename(columns={'Unsubscribed Magnews': 'Unsubscribed_Magnews'})
Unsubscribed_Magnews = LabelEncoder()
df['Unsubscribed_Magnews'] =  Unsubscribed_Magnews.fit_transform(df['Unsubscribed_Magnews'])
df['Unsubscribed_Magnews'].value_counts(dropna=False)

df.dtypes

# eliminiamo tutti gli object e Customer perchè sono tutti customer
df = df.drop(['CRM Type'], axis=1)

# download del secondo dataset
df1 = pd.read_csv("Downloads/TransactionsDetails.csv", sep=",")

#controllo duplicati
df1.duplicated().sum()

# elimino tutti i services da PBI Item.cnl_ordertype
df1 = df1[df1['PBI Item.cnl_ordertype'] != 'Services']

# modifico il nome di customer in customer ID
df = df.rename(columns={'Customer ID': 'Customer'})

# elimino le colonne con valori mancanti superiori al 50 % e che non mi servono
df1 = df1.drop(['Receipt No.', 'Transaction Detail', 'Item', 'Size', 'Quantity On Sale', 'Price', 'Discount Amount', 'Total Amount', 'Exchange Rate', 'Loyalty Membership Month', 'Loyalty Membership No','Loyalty Total Amount', 'loyalty Transaction Score', 'Black edition Flag', 'Exclusive Flag', 'XXX 1934 Flag', 'Formal Flag', 'Accessories Flag', 'Sportwear Flag', 'PBI Item.cnl_itemId'], axis=1)
df1 = df1.drop(['PBI Item.cnl_sector', 'PBI Item.Segment', 'PBI Item.Item Category', 'PBI Item.Item Subcategory', 'PBI Item.cnl_article', 'PBI Item.Model', 'PBI Item.cnl_modeldescription', 'PBI Item.Color Code','PBI Item.Color', 'PBI Item.Drop', 'PBI Item.cnl_fitting', 'PBI Item.Variant', 'PBI Item.Composition', 'PBI Item.Height', 'PBI Item.cnl_lenght', 'PBI Item.Item Style', 'PBI Item.Size', 'PBI Item.cnl_uomidName'], axis=1)
df1 = df1.drop(['PBI Item.cnl_type', 'PBI Item.cnl_name', 'PBI Item.cnl_id', 'PBI Item.cnl_ItemType', 'Net Amount', 'Net Amount Base', 'Net Price', 'Net Price Base', 'Currency','Currency Year', 'Net Amount Euro', 'Personalizzato', 'YEAR CURRENCY', 'PBI LoyaltyCurrencySetting (2).Exchange Rate (vs €)', 'SALES PERSON CODE'], axis=1)
df1 = df1.drop(['Return Quantity'], axis=1)
df1 = df1.drop(['Date', 'Area'], axis=1)
df1 = df1.drop(['Store', 'Price (Base)', 'Discount Amount (Base)'], axis=1)

df1.dtypes
df1.isnull().sum()

# sostituisco i dati mancanti delle numeriche con la media
df1['Quantity'] = df1['Quantity'].fillna(df1['Quantity'].mean())

# sostituisco i dati mancanti delle categoriche con il valore più frequente
df1['PBI Item.cnl_ordertype'] = df1['PBI Item.cnl_ordertype'].fillna(df1['PBI Item.cnl_ordertype'].mode()[0])
df1['Total Amount (Base)'] = df1['Total Amount (Base)'].fillna(df1['Total Amount (Base)'].mode()[0])

# eliminare il segno dell'euro e trasformala in formato numerico
df1['Total Amount (Base)'] = df1['Total Amount (Base)'].str.replace('€', '').str.strip()
df1['Total Amount (Base)'] = df1['Total Amount (Base)'].str.replace(',', '').str.strip()
df1['Total Amount (Base)'] = df1['Total Amount (Base)'].str.replace(r'\s*-\s*', '-', regex=True)
df1['Total Amount (Base)'] = df1['Total Amount (Base)'].astype(float)

# raggruppo per 'customer' e calcolo la somma di 'quantity' per ogni consumatore
td_grouped_sum_quantity = df1.groupby('Customer')['Quantity'].sum().reset_index()

# raggruppo per 'customer' e calcolo la somma di 'total_amount_(base)' per ogni consumatore
td_grouped_sum_total_amount_base = df1.groupby('Customer')['Total Amount (Base)'].sum().reset_index()

# raggruppo per 'customer' e conto le transazioni uniche per ciascun cliente 
td_grouped_count = df1.groupby('Customer')['Transaction'].nunique().reset_index()

# Trovare il most frequent order type per ogni consumatore 
td_grouped_most_frequent_order_type = df1.groupby('Customer')['PBI Item.cnl_ordertype'].value_counts().groupby('Customer').idxmax().reset_index(name='most_frequent_order_type')
td_grouped_most_frequent_order_type['most_frequent_order_type'] = td_grouped_most_frequent_order_type['most_frequent_order_type'].apply(lambda x: x[1])

# Trovare l'ultimo anno in cui hanno acquistato i consumatori 
td_grouped_latest_year = df1.groupby('Customer')['Year'].max().reset_index(name='latest_year')

# unisco le colonne calcolate 
ca_updated = df.merge(td_grouped_sum_quantity, on='Customer', how='left')
ca_updated = ca_updated.merge(td_grouped_sum_total_amount_base, on='Customer',how='left')
ca_updated = ca_updated.merge(td_grouped_count, on='Customer', how='left')
ca_updated = ca_updated.merge(td_grouped_most_frequent_order_type, on='Customer',how='left')
ca_updated = ca_updated.merge(td_grouped_latest_year, on='Customer', how='left')

# rinomino 'Transaction' in 'number_of_purchases'
ca_updated.rename(columns={'Transaction': 'number_of_purchases'}, inplace=True)

ca_updated.dtypes
ca_updated.isnull().sum()

# sostituisco i dati mancanti delle numeriche con la media
ca_updated['Quantity'] = ca_updated['Quantity'].fillna(ca_updated['Quantity'].mean())
ca_updated['Total Amount (Base)'] = ca_updated['Total Amount (Base)'].fillna(ca_updated['Total Amount (Base)'].mean())
ca_updated['number_of_purchases'] = ca_updated['number_of_purchases'].fillna(ca_updated['number_of_purchases'].mean())
ca_updated['latest_year'] = ca_updated['latest_year'].fillna(ca_updated['latest_year'].mean())

# sostituisco i dati mancanti delle categoriche con il valore più frequente
ca_updated['most_frequent_order_type'] = ca_updated['most_frequent_order_type'].fillna(ca_updated['most_frequent_order_type'].mode()[0])

# encoder per most_frequent_order_type
most_frequent_order_type = LabelEncoder()
ca_updated['most_frequent_order_type'] =  most_frequent_order_type.fit_transform(ca_updated['most_frequent_order_type'])
ca_updated['most_frequent_order_type'].value_counts(dropna=False)

# eliminiamo Customer 
ca_updated = ca_updated.drop(['Customer'], axis=1)

# controlliamo se ci sono dati duplicati e eliminiamoli
ca_updated.duplicated().sum()
ca_updated = ca_updated.drop_duplicates()
ca_updated.duplicated().sum()

# cambiare i nomi 
ca_updated = ca_updated.rename(columns={'Data Collection Score': 'Data_Collection_Score'})
ca_updated = ca_updated.rename(columns={'Total Amount (Base)': 'Total_Amount_Base'})

# verifico gli outlier
import seaborn as sns
sns.boxplot(ca_updated['Data_Collection_Score'])
sns.boxplot(ca_updated['latest_year'])
sns.boxplot(ca_updated['most_frequent_order_type'])
sns.boxplot(ca_updated['number_of_purchases']) #forse eliminare 
sns.boxplot(ca_updated['Total_Amount_Base']) # forse eliminare
sns.boxplot(ca_updated['Quantity']) # forse eliminare 
sns.boxplot(ca_updated['Customer_Habit'])
sns.boxplot(ca_updated['Sales_Habit'])
sns.boxplot(ca_updated['Macro_Area'])

# verifico le distribuzioni 
plt.figure(figsize=(20,3))
plt.subplot(1,3,2)
sns.distplot(a=ca_updated.Gender,color='orange')
plt.title('Distribution Plot of Gender')

plt.figure(figsize=(20,3))
plt.subplot(1,3,2)
sns.distplot(a=ca_updated.Macro_Area,color='orange')
plt.title('Distribution Plot of Macro_Area')

plt.figure(figsize=(20,3))
plt.subplot(1,3,2)
sns.distplot(a=ca_updated.Consent_for_Marketing,color='orange')
plt.title('Distribution Plot of Consent_for_Marketing')

plt.figure(figsize=(20,3))
plt.subplot(1,3,2)
sns.distplot(a=ca_updated.Consent_for_Analysis,color='orange')
plt.title('Distribution Plot of Consent_for_Analysis')

plt.figure(figsize=(20,3))
plt.subplot(1,3,2)
sns.distplot(a=ca_updated.Consent_for_SMS,color='orange')
plt.title('Distribution Plot of Consent_for_SMS')

plt.figure(figsize=(20,3))
plt.subplot(1,3,2)
sns.distplot(a=ca_updated.Consent_for_Newsletter,color='orange')
plt.title('Distribution Plot of Consent_for_Newsletter')

plt.figure(figsize=(20,3))
plt.subplot(1,3,2)
sns.distplot(a=ca_updated.Pref_Email,color='orange')
plt.title('Distribution Plot of Pref_Email')

plt.figure(figsize=(20,3))
plt.subplot(1,3,2)
sns.distplot(a=ca_updated.Pref_Sms,color='orange')
plt.title('Distribution Plot of Pref_Sms')

plt.figure(figsize=(20,3))
plt.subplot(1,3,2)
sns.distplot(a=ca_updated.Pref_Telephone,color='orange')
plt.title('Distribution Plot of Pref_Telephone')

plt.figure(figsize=(20,3))
plt.subplot(1,3,2)
sns.distplot(a=ca_updated.Pref_Mail,color='orange')
plt.title('Distribution Plot of Pref_Mail')

plt.figure(figsize=(20,3))
plt.subplot(1,3,2)
sns.distplot(a=ca_updated.Unsubscribed_Magnews,color='orange')
plt.title('Distribution Plot of Unsubscribed_Magnews')

plt.figure(figsize=(20,3))
plt.subplot(1,3,2)
sns.distplot(a=ca_updated.most_frequent_order_type,color='orange')
plt.title('Distribution Plot of most_frequent_order_type')

plt.figure(figsize=(20,3))
plt.subplot(1,3,2)
sns.distplot(a=ca_updated.latest_year,color='orange')
plt.title('Distribution Plot of latest_year')

plt.figure(figsize=(20,3))
plt.subplot(1,3,2)
sns.distplot(a=ca_updated.number_of_purchases,color='orange')
plt.title('Distribution Plot of number_of_purchases')

plt.figure(figsize=(20,3))
plt.subplot(1,3,2)
sns.distplot(a=ca_updated.Total_Amount_Base,color='orange')
plt.title('Distribution Plot of Total_Amount_Base')

plt.figure(figsize=(20,3))
plt.subplot(1,3,2)
sns.distplot(a=ca_updated.Quantity,color='orange')
plt.title('Distribution Plot of Quantity')

plt.figure(figsize=(20,3))
plt.subplot(1,3,2)
sns.distplot(a=ca_updated.Data_Collection_Score,color='orange')
plt.title('Distribution Plot of Data_Collection_Score')

plt.figure(figsize=(20,3))
plt.subplot(1,3,2)
sns.distplot(a=ca_updated.Customer_Habit,color='orange')
plt.title('Distribution Plot of Customer_Habit')

plt.figure(figsize=(20,3))
plt.subplot(1,3,2)
sns.distplot(a=ca_updated.Sales_Habit,color='orange')
plt.title('Distribution Plot of Sales_Habit')

# descrizione statistica per ogni singola colonna
ca_updated['Gender'].describe()
ca_updated['Macro_Area'].describe()
ca_updated['Consent_for_Marketing'].describe()
ca_updated['Consent_for_Analysis'].describe()
ca_updated['Consent_for_SMS'].describe()
ca_updated['Consent_for_Newsletter'].describe()
ca_updated['Pref_Email'].describe()
ca_updated['Pref_Sms'].describe()
ca_updated['Pref_Telephone'].describe()
ca_updated['Pref_Mail'].describe()
ca_updated['Unsubscribed_Magnews'].describe()
ca_updated['most_frequent_order_type'].describe()
ca_updated['latest_year'].describe()
ca_updated['number_of_purchases'].describe()
ca_updated['Total_Amount_Base'].describe()
ca_updated['Quantity'].describe()
ca_updated['Data_Collection_Score'].describe()
ca_updated['Customer_Habit'].describe()
ca_updated['Sales_Habit'].describe()

# vediamo dal grafico se la y è bilanciata 
colors = ["#0101DF", "#DF0101"]
sns.countplot('Loyalty', data=ca_updated, palette=colors)
plt.title('Class Distributions \n (0: Infedele || 1: Leale)', fontsize=14)
ca_updated['Loyalty'].value_counts()

#matrice di correlazione scelgiere una delle due 
import matplotlib.pyplot as plt
import seaborn as sns
matrix = ca_updated.corr()
matrix["Loyalty"].sort_values(ascending=False)
corr_df = ca_updated.corr(method='pearson')
plt.figure(figsize=(15,7))
sns.heatmap(matrix, annot=True)
plt.show()

correlations = ca_updated.corr()['Loyalty'].sort_values()
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))

#DEFINIZIONE X E Y
y = ca_updated['Loyalty']
x = ca_updated.drop(columns=["Loyalty"], axis = 1) 
y.value_counts()

#STANDARDIZZAZIONE
from sklearn.preprocessing import StandardScaler
scale= StandardScaler()
scaled_x = scale.fit_transform(x) 
print(scaled_x)

#DIVIDO DATASET IN TEST E TRAIN
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, KFold
x_train, x_test, y_train, y_test = train_test_split(scaled_x,y,random_state=1,test_size=0.20, stratify = y) 

train_0, train_1 = len(y_train[y_train==0]), len(y_train[y_train==1])
test_0, test_1 = len(y_test[y_test==0]), len(y_test[y_test==1])
print('>Train: 0=%d, 1=%d, Test: 0=%d, 1=%d' % (train_0, train_1, test_0, test_1))
print(x_train, y_train)

#CLASSIFICATORI GENERICO
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report, f1_score
rf = RandomForestClassifier(n_estimators=20, random_state=1, class_weight= "balanced")
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=1)
p_grid = {"max_depth": [2, 3, 4, 5]}
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
clf = GridSearchCV(estimator=dt, param_grid=p_grid, cv=inner_cv, verbose=0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

#ORA PROVO BILANCIANDO IL DATASET - SMOTEENN
#SMOTEENN
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.combine import SMOTEENN
x, y = make_classification(n_classes=2, class_sep=2,weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
sme = SMOTEENN(random_state=42)
x_ress, y_ress= sme.fit_resample(x_train, y_train)
y_ress.value_counts()
sns.countplot(y_ress, palette=colors)
plt.title('Class Distributions \n (0: Infedele|| 1: Leale)', fontsize=14)
print(x_ress)

#RANDOM FOREST
rf.fit(x_ress, y_ress)
y_pred = rf.predict(x_test)
plot_confusion_matrix(rf, x_test, y_test, cmap="Purples")
print (classification_report(y_test, y_pred))

# Calcolo dell'importanza delle features
column_names = ['Gender', 'Macro_Area', 'Consent_for_Marketing',
                'Consent_for_Analysis', 'Consent_for_SMS', 'Consent_for_Newsletter',
                'Pref_Email', 'Pref_Sms', 'Pref_Telephone', 'Pref_Mail', 'Sales_Habit',
                'Customer_Habit', 'Data_Collection_Score', 'Unsubscribed_Magnews',
                'Quantity', 'Total_Amount_Base', 'number_of_purchases',
                'most_frequent_order_type', 'latest_year']
rf.feature_importances_
feature_importances = pd.Series(rf.feature_importances_, index=column_names).nlargest(19).plot(kind='barh')
plt.show()

# Curva AUC
from sklearn.metrics import roc_curve, auc

class_probabilities = rf.predict_proba(x_test)
preds = class_probabilities[:, 1]

fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

# Printing AUC
print(f"AUC for our classifier is: {roc_auc}")

# Plotting the ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#LOGISTIC REGRESSION
lr.fit(x_ress, y_ress)
y_pred = lr.predict(x_test)
predictions = lr.predict(x_test)
print(classification_report(y_test, predictions))
plot_confusion_matrix(lr, x_test, y_test, cmap="Reds")

# Calcolo dell'importanza delle features
feature_importance = lr.coef_[0]
normalized_importance = abs(feature_importance) / sum(abs(feature_importance))
feature_indices = range(len(normalized_importance))
plt.bar(feature_indices, normalized_importance)
plt.xticks(feature_indices, column_names, rotation='vertical')  
plt.show()

# Curva AUC
from sklearn.metrics import roc_curve, auc

class_probabilities = lr.predict_proba(x_test)
preds = class_probabilities[:, 1]

fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

# Printing AUC
print(f"AUC for our classifier is: {roc_auc}")

# Plotting the ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#DECISION TREE
clf.fit(x_ress, y_ress)
predictions = clf.predict(x_test)
print(classification_report(y_test, predictions))
plot_confusion_matrix(clf, x_test, y_test, cmap="Reds")

best_model = clf.best_estimator_

# Calcolo dell'importanza delle features
feature_importance = best_model.feature_importances_
feature_indices = range(len(feature_importance))
plt.barh(feature_indices, feature_importance)
plt.yticks(feature_indices, column_names)  
plt.show()

# Curva AUC
from sklearn.metrics import roc_curve, auc

class_probabilities = clf.predict_proba(x_test)
preds = class_probabilities[:, 1]

fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

# Printing AUC
print(f"AUC for our classifier is: {roc_auc}")

# Plotting the ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#NAIVE BAYES
gnb.fit(x_ress, y_ress)
y_pred = gnb.predict(x_test)
print(classification_report(y_test, predictions))
plot_confusion_matrix(gnb, x_test, y_test, cmap="Oranges")

# Calcolo dell'importanza delle features utilizzando la deviazione standard
class_priors = gnb.class_prior_
class_means = gnb.theta_ # Calcolo delle medie delle features per ogni classe
class_std = gnb.sigma_ # Calcolo delle deviazioni standard delle features per ogni classe
feature_importance = np.mean(class_std, axis=0)
feature_indices = range(len(feature_importance))
plt.barh(feature_indices, feature_importance)
plt.yticks(feature_indices, column_names)  
plt.show()

# Curva AUC
from sklearn.metrics import roc_curve, auc

class_probabilities = gnb.predict_proba(x_test)
preds = class_probabilities[:, 1]

fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

# Printing AUC
print(f"AUC for our classifier is: {roc_auc}")

# Plotting the ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#KNEIGHBORS
classifier.fit(x_ress, y_ress)
y_pred = classifier.predict(x_test)
print(classification_report(y_test,y_pred))
plot_confusion_matrix(classifier, x_test, y_test, cmap="Greens")

# Calcolo del punteggio F-test per ogni features
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(score_func=f_classif, k='all')
selector.fit(x_ress, y_ress)
feature_importance = selector.scores_
feature_indices = range(len(feature_importance))
plt.barh(feature_indices, feature_importance)
plt.yticks(feature_indices, column_names) 
plt.show()

# Curva AUC
from sklearn.metrics import roc_curve, auc

class_probabilities = classifier.predict_proba(x_test)
preds = class_probabilities[:, 1]

fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

# Printing AUC
print(f"AUC for our classifier is: {roc_auc}")

# Plotting the ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#XGBoost Classifier
!pip3 install xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from numpy import nan

model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(x_ress,y_ress)

XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1, eval_metric='mlogloss', gamma=0, gpu_id=-1, importance_type='gain', interaction_constraints='', learning_rate=0.300000012,max_delta_step=0, max_depth=6, min_child_weight=1, missing=nan, monotone_constraints='()', n_estimators=100, n_jobs=16,num_parallel_tree=1, objective='multi:softprob', random_state=0,reg_alpha=0, reg_lambda=1, scale_pos_weight=None, subsample=1,tree_method='exact', use_label_encoder=False,validate_parameters=1, verbosity=None)
y_pred = model.predict(x_test)
y_pred
plot_confusion_matrix(model, x_test, y_test, cmap="Reds")
print (classification_report(y_test, y_pred))

# Calcolo dell'importanza delle features           
model.feature_importances_
feature_importances = pd.Series(model.feature_importances_, index=column_names).nlargest(19).plot(kind='barh')
plt.show()

# Curva AUC
from sklearn.metrics import roc_curve, auc

class_probabilities = model.predict_proba(x_test)
preds = class_probabilities[:, 1]

fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

# Printing AUC
print(f"AUC for our classifier is: {roc_auc}")

# Plotting the ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# elimino "Consent_for_Analysis", "Consent_for_SMS", "Consent_for_Newsletter", "Total_Amount_Base", "Quantity"
ca_updated = ca_updated.drop(["Consent_for_Analysis", "Consent_for_SMS", "Consent_for_Newsletter", "Total_Amount_Base", "Quantity"], axis=1)

# analizzo i classificatori senza queste feature 

#DIVIDO DATASET IN TEST E TRAIN
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, KFold
x_train, x_test, y_train, y_test = train_test_split(scaled_x,y,random_state=1,test_size=0.20, stratify = y) 

train_0, train_1 = len(y_train[y_train==0]), len(y_train[y_train==1])
test_0, test_1 = len(y_test[y_test==0]), len(y_test[y_test==1])
print('>Train: 0=%d, 1=%d, Test: 0=%d, 1=%d' % (train_0, train_1, test_0, test_1))
print(x_train, y_train)

#CLASSIFICATORI GENERICO
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report, f1_score
rf = RandomForestClassifier(n_estimators=20, random_state=1, class_weight= "balanced")
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=1)
p_grid = {"max_depth": [2, 3, 4, 5]}
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
clf = GridSearchCV(estimator=dt, param_grid=p_grid, cv=inner_cv, verbose=0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

#ORA PROVO BILANCIANDO IL DATASET - SMOTEENN
#SMOTEENN
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.combine import SMOTEENN
x, y = make_classification(n_classes=2, class_sep=2,weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
sme = SMOTEENN(random_state=42)
x_ress, y_ress= sme.fit_resample(x_train, y_train)
y_ress.value_counts()
sns.countplot(y_ress, palette=colors)
plt.title('Class Distributions \n (0: Infedele|| 1: Leale)', fontsize=14)
print(x_ress)

#RANDOM FOREST
rf.fit(x_ress, y_ress)
y_pred = rf.predict(x_test)
plot_confusion_matrix(rf, x_test, y_test, cmap="Purples")
print (classification_report(y_test, y_pred))

rf.feature_importances_
pd.Series(rf.feature_importances_).nlargest(25).plot(kind='barh')
plt.show()
print(x_ress)

# Curva AUC
from sklearn.metrics import roc_curve, auc

class_probabilities = rf.predict_proba(x_test)
preds = class_probabilities[:, 1]

fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

# Printing AUC
print(f"AUC for our classifier is: {roc_auc}")

# Plotting the ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#LOGISTIC REGRESSION
lr.fit(x_ress, y_ress)
y_pred = lr.predict(x_test)
predictions = lr.predict(x_test)
print(classification_report(y_test, predictions))
plot_confusion_matrix(lr, x_test, y_test, cmap="Reds")


# Curva AUC
from sklearn.metrics import roc_curve, auc

class_probabilities = lr.predict_proba(x_test)
preds = class_probabilities[:, 1]

fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

# Printing AUC
print(f"AUC for our classifier is: {roc_auc}")

# Plotting the ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#DECISION TREE
clf.fit(x_ress, y_ress)
predictions = clf.predict(x_test)
print(classification_report(y_test, predictions))
plot_confusion_matrix(clf, x_test, y_test, cmap="Reds")

# Curva AUC
from sklearn.metrics import roc_curve, auc

class_probabilities = clf.predict_proba(x_test)
preds = class_probabilities[:, 1]

fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

# Printing AUC
print(f"AUC for our classifier is: {roc_auc}")

# Plotting the ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#NAIVE BAYES
gnb.fit(x_ress, y_ress)
y_pred = gnb.predict(x_test)
print(classification_report(y_test, predictions))
plot_confusion_matrix(gnb, x_test, y_test, cmap="Oranges")

# Curva AUC
from sklearn.metrics import roc_curve, auc

class_probabilities = gnb.predict_proba(x_test)
preds = class_probabilities[:, 1]

fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

# Printing AUC
print(f"AUC for our classifier is: {roc_auc}")

# Plotting the ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#KNEIGHBORS
classifier.fit(x_ress, y_ress)
y_pred = classifier.predict(x_test)
print(classification_report(y_test,y_pred))
plot_confusion_matrix(classifier, x_test, y_test, cmap="Greens")

# Curva AUC
from sklearn.metrics import roc_curve, auc

class_probabilities = classifier.predict_proba(x_test)
preds = class_probabilities[:, 1]

fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

# Printing AUC
print(f"AUC for our classifier is: {roc_auc}")

# Plotting the ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#XGBoost Classifier
!pip3 install xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from numpy import nan

model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(x_ress,y_ress)

XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1, eval_metric='mlogloss', gamma=0, gpu_id=-1, importance_type='gain', interaction_constraints='', learning_rate=0.300000012,max_delta_step=0, max_depth=6, min_child_weight=1, missing=nan, monotone_constraints='()', n_estimators=100, n_jobs=16,num_parallel_tree=1, objective='multi:softprob', random_state=0,reg_alpha=0, reg_lambda=1, scale_pos_weight=None, subsample=1,tree_method='exact', use_label_encoder=False,validate_parameters=1, verbosity=None)
y_pred = model.predict(x_test)
y_pred
plot_confusion_matrix(model, x_test, y_test, cmap="Reds")
print (classification_report(y_test, y_pred))
           
# Curva AUC
from sklearn.metrics import roc_curve, auc

class_probabilities = model.predict_proba(x_test)
preds = class_probabilities[:, 1]

fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

# Printing AUC
print(f"AUC for our classifier is: {roc_auc}")

# Plotting the ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


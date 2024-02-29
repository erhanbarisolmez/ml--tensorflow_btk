import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

dataFrame = pd.read_excel("merc.xlsx")
dataFrame.head()

dataFrame.describe() #all

dataFrame.isnull().sum() # column toplam eksik veri


plt.Figure(figsize=(7,5))
sbn.displot(dataFrame["price"]) # nasıl dağıldığı

sbn.countplot(dataFrame["year"])  # kaçtane olduğu

numerical_columns = dataFrame.select_dtypes(include=[np.number]) # sayısal sütunları seçme

correlation = numerical_columns.corr() # korelasyon hesaplama

price_correlation = correlation["price"].sort_values() #"price" sütunundaki diğer sütunlarla olan korelasyonu kontrol etme



plt.scatter(x=price_correlation["year"], y=price_correlation["price"], c="blue", alpha=0.5)

fig, ax = plt.subplots()
ax.plot(price_correlation["year"], price_correlation["price"])
plt.scatter(x=price_correlation["year"], y=price_correlation["price"], c="red", alpha=0.5)


dataFrame.sort_values("price", ascending=False).head(20) #en yüksek, true: en küçük

len(dataFrame) * 0.01
yüzdeDoksanDokuzDf = dataFrame.sort_values("price", ascending=False).iloc[131:]
plt.figure(figsize=(7,5))
sbn.displot(yüzdeDoksanDokuzDf["price"])
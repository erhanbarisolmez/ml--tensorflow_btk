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

price_correlation = correlation["price"].sort_values() #"price" sğtunundaki diğer sütunlarla olan korelasyonu kontrol etme
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns 
import matplotlib
import jovian

medical_df = pd.read_csv('./data/medical.csv')

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'
medical_df.describe()
#Yaş sayısal bir sütundur. Veri setindeki minimum yaş 18 ve maksimum yaş 64'tür.
# Böylece, 47 kutulu (her yıl için bir tane) histogram ve kutu grafiği kullanarak yaş 
#dağılımını görselleştirebiliriz. Grafiği etkileşimli hale getirmek için komployu kullanacağız,
# ancak Seaborn'u kullanarak da benzer grafikler oluşturabilirsiniz.
medical_df.age.describe()
fig = px.histogram(medical_df,
                   x="age",
                   marginal="box",
                   nbins=47,
                   title='Distribution of Age')
fig.update_layout(bargap=0.1)
fig.show()

#cinsiyet
medical_df.sex.describe()
fig = px.histogram(medical_df,
                   x="sex",
                   marginal='box', #kutu grafiği
                   nbins=60,
                   title='Distribution of Sex')
fig.update_layout(bargap=0.1) # grafil düzeni günceller, barların arasındaki boşluğu ayarlıyor
fig.show()

#vücut kitle indeksi
medical_df.bmi.describe()
fig = px.histogram(medical_df,
                   x='bmi',
                   marginal='box',
                   color_discrete_sequence=['red'],
                   title='Distribution of BMI (Body Mass Index)')
fig.update_layout(bargap=0.1)
fig.show()
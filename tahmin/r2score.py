from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np

# Veri setini yükleyelim veya oluşturalım
# Örnek olarak rastgele bir veri seti oluşturalım
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Veriyi eğitim ve test setlerine bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Doğrusal Regresyon (Simple Linear Regression - SLR)
slr_model = LinearRegression()
slr_model.fit(X_train, y_train)
slr_pred = slr_model.predict(X_test)

# Çoklu Doğrusal Regresyon (Multiple Linear Regression - MLR)
mlr_model = LinearRegression()
mlr_model.fit(X_train, y_train)
mlr_pred = mlr_model.predict(X_test)

# Polinomal Regresyon (Polynomial Regression - PR)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X_train)
pr_model = LinearRegression()
pr_model.fit(X_poly, y_train)
X_test_poly = poly_features.transform(X_test)
pr_pred = pr_model.predict(X_test_poly)

# Destek Vektör Makineleri (Support Vector Machines - SVM)
svm_model = SVR(kernel='linear')
svm_model.fit(X_train, y_train.flatten())
svm_pred = svm_model.predict(X_test)

# Karar Ağaçları (Decision Trees - DT)
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# Rassal Ormanlar (Random Forests - RF)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train.flatten())
rf_pred = rf_model.predict(X_test)

# R2 Hesaplamaları
slr_r2 = r2_score(y_test, slr_pred)
mlr_r2 = r2_score(y_test, mlr_pred)
pr_r2 = r2_score(y_test, pr_pred)
svm_r2 = r2_score(y_test, svm_pred)
dt_r2 = r2_score(y_test, dt_pred)
rf_r2 = r2_score(y_test, rf_pred)

print(f"SLR R2: {slr_r2}")
print(f"MLR R2: {mlr_r2}")
print(f"PR R2: {pr_r2}")
print(f"SVM R2: {svm_r2}")
print(f"DT R2: {dt_r2}")
print(f"RF R2: {rf_r2}")
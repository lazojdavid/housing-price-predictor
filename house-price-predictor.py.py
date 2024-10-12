#!/usr/bin/env python
# coding: utf-8

# In[1]:


import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np #desde la creación de un dataset
from zlib import crc32
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

housing = pd.read_csv("housing.csv")

# In[2]:


housing.head()


# In[3]:


housing.info()


# In[4]:


housing["ocean_proximity"].value_counts()


# In[5]:


housing.describe()


# In[6]:


housing.hist(bins=50, figsize=(20,15))
plt.show()

#SEPARACIÓN DE SETS PRIMERA VERSIÓN SIN CONSISTENCIA
# In[7]:


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[8]:


train_set, test_set = split_train_test(housing,0.2)
len(train_set)


# In[9]:


len(test_set)

#CREAR IDENTIFICADORES PARA MANTENER CONSISTENCIA
# In[10]:


def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32 # todo en 32 bits
def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio)) # devuelve TRUE o false
    return data.loc[~in_test_set], data.loc[in_test_set] # me retorna train set y  test set  , dependiendo si fue TRUE O FALSE


# In[11]:


housing_with_id = housing.reset_index() # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index") #ACA SE GUARDA DEPENDIENDO DE LOS RETURNS DE MIS DOS FUNCIONES PREVIAS
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")


# In[12]:


train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

#/////////////////////// ESTRATIFICACIÓN ///////////////////////////////
# In[13]:


housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf],labels=[1, 2, 3, 4, 5])

#CREA UNA NUEVA COLUMNA , INGRESOS POR CATEGORIA. PD.CUT CONVIERTE VALORES CONTINUOS EN CATEGORÍCOS, BINS SON LOS INTERAVLOS Y LABES LAS ETIQUETAS DE ELLOS


# In[14]:


housing["income_cat"].hist()
plt.show()


# In[15]:


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[16]:


strat_test_set["income_cat"].value_counts() / len(strat_test_set)
# MUESTRA EL PORCENTAJE DE CADA CATEGORÍA LA CATEGORIA 3 TIENE 35.05 % 


# In[17]:


# tengo que ver si estos resultados son representativos 


# In[18]:


# Proporciones en el conjunto original
original_proportions = housing["income_cat"].value_counts() / len(housing)
print(original_proportions)


# In[19]:


# Proporciones en el conjunto de prueba
test_proportions = strat_test_set["income_cat"].value_counts() / len(strat_test_set)
print(test_proportions)


# In[20]:


for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


# In[21]:


print(housing.columns)


# In[22]:


housing = strat_train_set.copy()

#Visualizing Geographical Data
# In[23]:


housing.plot(kind="scatter", x="longitude", y="latitude")
plt.show()


# In[24]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha =0.1)
plt.show()


# In[25]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
s=housing["population"]/100, label="population", figsize=(10,7),
c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()

#///////////////////////CORRELACIONES ///////////////////////////////

# In[26]:


corr_matrix = housing.select_dtypes(include=[np.number]).corr() #seleccionamos solo las variables numericas
corr_matrix["median_house_value"].sort_values(ascending=False)


# In[27]:


#En base a estas correlaciones es que seleccionamos nuestras variables para nuestros modelos

#EXPERIMENTAR CON COMBINACIONES DE ATRIBUTOS: PARA RESUMIR CARACTERISTICAS:
#POR EJEMPLO TENEMOS : HOUSEHOLDS : NUMERO TOTAL DE HOGARES
#POPULATION : NUMERO TOTAL DE PERSONAS
#TOTAL_RROMS : TOTAL DE HABITACIONES
#TOTAL_BEDROOMS: TOTAL DE HABITACIONES

#POPULATION_PER_HOUSE = POPLUATION / HOUSEHOLD
#ROOMS_PER_HOUSE = TOTAL_ROOMS / HOUSEHOLD
#BEDROOMS_PER_HOUSE = TOTAL_BEDROOMS / HOUSEHOLDS

#LUEGO VERIFICAMOS OTRA VEZ LA CORRELACIÓN

#HOUSING["POPLUTAION_PER_HOUSE"] = HOUSING["POPULATION"] / HOUSING["HOUSEHOLD"]

#Transformadores de forma manual y no automática. Es un ejemplo previo de que variable nueva se puede crear para ver su correlación
# In[28]:


housing["rooms_per_household"]=housing["total_rooms"]/housing["households"]
housing["bedrooms_per_household"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["populatio_per_household"]=housing["population"]/housing["households"]


# In[29]:


housing_num = housing.select_dtypes(include=[np.number])
corr_matrix = housing_num.corr()


# In[30]:


corr_matrix["median_house_value"].sort_values(ascending=False)

#INTERPRETACIÓN : PODEMOS INTERPRETAS QUE A MAYOR CANTIDAD DE PERSONAS POR ZONA, EN CASA, EL PRECIO DISMINUYE, LO QUE NOS DARIA A ENTENDER QUE SON ZONAS EXCLUSIVAS. TAMBIEN QUE SI HAY MAS CUARTOS EN UNA CASA, EL PRECIO DISMINUYE, LO QUE DA ENTENDER QUE OCUPARÍA MAS ESPACIO///////////////////////Prepare the Data for Machine Learning Algorithms ///////////////////////////////
# In[31]:


housing = strat_train_set.drop("median_house_value", axis=1) #RETIRAMOS LA VARIABLE PREDICTORA

#Predictores
#De nuestro datos de entrenamiento retiramos la variable objetivo, la que queremos predecir,
#si axis fuera 0 se eliminaria la fila, pero como es 1 se elimina la columna, drop() crea una copia mientras retira un valor, el original se queda intacto
# In[32]:


housing_labels = strat_train_set["median_house_value"].copy() #SE COPIA LA COLUMNA QUE SE QUIERE PREDICR A OTRA VARIABLE

#Variable a predecir
#se crea una nuva variable, y se guarda la copia de nuestra variable de predicción.///////////////DATA CLEANING ///////////////////////
# In[33]:


housing.info()

#Total_bedrooms tiene valores incompletos, lo que me da tres opcciones
#1. Eliminar toda la fila, lo cual puede afectar la data si son muchos valores faltantes
#2. Eliminar todo el atributo ( columna), si es que esta variable no tiene mucha correlación pero me quitaría un atributo importante
#3. Reemplazar los valores ( recomnedado ). 
#3.1 Con 0 suponiendo que no tengan habitaciones
#3.2 Con la media, pero si los valores tiene outliers(extremos numericos que no siguen la tendencia) el promedio reemplazado puede
#no representar la tendencia central
#3.3 La mediana, en caso tenga outliers

# In[34]:


from sklearn.impute import SimpleImputer


# In[35]:


imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)


# In[36]:


imputer.fit(housing_num) #AJUSTA A LOS DATOS DE ENTRENAMIENTO, SE GUARDA AUTOMATICAMENTE NE statistics_.


# In[37]:


imputer.statistics_ #medianas de todas las columnas


# In[38]:


housing_num.median().values


# In[39]:


X = imputer.transform(housing_num) #reemplaza todos los datos faltantes con las medianas, pero me X es un array de numpy

#El método transform completará todos los valores faltanates en el data frame housing_num, pero devolverá en forma de array
# In[40]:


housing_tr = pd.DataFrame(X, columns=housing_num.columns) #convierte todos los datos a un nuevo dataframe

#El array se convierte otra vez en una dataframe, pd.DataFrame, que recibe el array y las columnas originales.

#////////////Handling Text and Categorical Attributes////////////////
# In[41]:


housing_cat = housing[["ocean_proximity"]] #devuelve un dataframe

#La doble Llave [[]] regresa esta columna como un DataFrame lo cual es mas recomendado para aplicar transformaciones
# In[42]:


housing_cat.head(10)

#Convertiremos estas categorías a numeros
# In[43]:


from sklearn.preprocessing import OrdinalEncoder


# In[44]:


ordinal_encoder = OrdinalEncoder()

#OrdinaEncoder() transforma variables categóricas en númericas
# In[45]:


housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)


# In[46]:


housing_cat_encoded[:10]

#/////////////////////// ONE HOT ENCODER  MEJOR SOLUCIÓN //////////////////////////////
# In[47]:


from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot

### NOS DEVOLVERÁ UNA MATRIZ DE SCIPY


# In[48]:


housing_cat_1hot.toarray()

#CONVERTIMOS EL ARREGLO EN UNA MATRIZ DE NUMPY CON .TOARRAY()


# In[49]:


cat_encoder.categories_

#///////////////////Custom Transformers///////////////////////////////
# In[50]:


housing.head()


# In[51]:


# rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
# selecciono estas columnas porque en mi creación de nuevos atributos vi que estas cuatro me dan mejores resultados
#housing["population_per_house"] = housing["population"] / housing["households"] similar . 


# In[52]:


from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                        bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


# In[53]:


#Hiperámetro = comodín add_bedrooms_per_room, para la configuración del transformador 
#que hará las transformaciones de nuevos atributos rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
#population_per_household = X[:, population_ix] / X[:, households_ix]

#///////////////PIPELINE///////////////////////
#TRABAJAMOS PRIMERO CON LAS COLUMNAS NUMERICAS ARRIBA
# In[54]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([ # TRANSFORMADOR
    ('imputer', SimpleImputer(strategy="median")), #llenar valores faltantes con mediana
    ('attribs_adder', CombinedAttributesAdder()), # combinar atributos
    ('std_scaler', StandardScaler()), # estandarizar los valores
    ])
housing_num_tr = num_pipeline.fit_transform(housing_num)

#///////////////COLUMN TRANSFORM///////////////////////
# In[55]:


from sklearn.compose import ColumnTransformer
housing_num = housing.select_dtypes(include=[np.number])  # Selecciona solo columnas numéricas
num_attribs = list(housing_num) # COLUMAS NUMERICAS
cat_attribs = ["ocean_proximity"] #COLUMNAS CATEGORICAS
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs), #PARÁMETROS ( NOMBRE, TRANSOFORMADOR , LISTA DONDE SE APLICAN EL TRANSFORMADOR
    ("cat", OneHotEncoder(), cat_attribs),
    ])
housing_prepared = full_pipeline.fit_transform(housing)

#///// SELECCIONAR EL MODELO DE ENTRENAMIENTO ////////////////////////// selecionando modelo de regresión lineal ////////////////
# In[56]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels) #parámetros, data aplicado con los transformadores y los predictores,

#////////////EVALUCIÓN DE MODELO ///////////////
# In[57]:


some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared)) 


# In[58]:


# Predictions: [ 85657.90192014 305492.60737488 152056.46122456 186095.70946094,244550.67966089] son los valores de las columnas
#Y los comparo con mis labels originales :[72100.0, 279600.0, 82700.0, 112500.0, 238300.0]


# In[59]:


print("Labels:", list(some_labels))


# In[61]:


from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[64]:


from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)


# In[65]:


housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse

#////////////// VALIDACION - PARA EVITAR UNDERFITTIN Y OVERFITTING EN NUESTROS MODELOS /////////////////
# In[69]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


# In[70]:


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# In[71]:


display_scores(tree_rmse_scores) #arbol de decisión


# In[72]:


lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
scoring="neg_mean_squared_error", cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores) #linear regression


# In[74]:


from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
forest_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, forest_predictions)
forest_rmse = np.sqrt(forest_mse)
# Realiza la validación cruzada para obtener una mejor evaluación del modelo
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)  # Calcula el RMSE a partir de los scores negativos


# In[75]:


display_scores(forest_rmse_scores)

#///////////////////////FINE TUNE YOUR MODEL //////////////////////////Una vez hemos seleccionado el modelo de randomForestRegresor usamos la herramienta GridSearchCV la cual ajusta automaticament
#todos los parámetros con validacion cruzada
# In[76]:


from sklearn.model_selection import GridSearchCV
param_grid = [
{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)


# In[77]:


grid_search.best_params_


# In[78]:


grid_search.best_estimator_


# In[79]:


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

#///////////////// BUSQUEDA RANDOMIZED //////////////////
#EN VEZ DE EVALUAR TODAS LAS POSIBLES COMBINACIONES, LA BUSQUEDA RANDOM LO QUE HACE ES SELECCIONAR ALEATORIAMENTE
# In[80]:


feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances


# In[81]:


extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
importance_with_names = zip(feature_importances, attributes)
sorted_importance = sorted(importance_with_names, key=lambda x: x[0], reverse=True)
print("Importancias de las características en orden:")
for importance, name in sorted_importance:
    print(f"{name}: {importance:.6f}")

# In[82]:


final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse) # => evaluates to 47,730.2


# In[83]:


from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
loc=squared_errors.mean(),
scale=stats.sem(squared_errors)))


# In[84]:


# Supongamos que tienes un DataFrame `test_set` que contiene tus datos de prueba
X_test = test_set.drop("median_house_value", axis=1)  # Ajusta según tu columna objetivo
y_test = test_set["median_house_value"].copy()  # La columna objetivo

# Transformar los datos
X_test_prepared = full_pipeline.transform(X_test)


# In[85]:


# Usa el modelo final entrenado
final_model = grid_search.best_estimator_
joblib.dump(final_model, "my_final_model.pkl")
# Hacer predicciones
final_predictions = final_model.predict(X_test_prepared)


# In[86]:


from sklearn.metrics import mean_squared_error

# Calcular el MSE
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print(f"RMSE: {final_rmse}")


# In[87]:


import matplotlib.pyplot as plt

# Comparar las predicciones con los valores reales
plt.scatter(y_test, final_predictions)
plt.xlabel("Valores reales")
plt.ylabel("Predicciones")
plt.title("Comparación entre valores reales y predicciones")
plt.show()

# In[88]:


import pandas as pd

# Supongamos que deseas predecir el precio de una casa con estas características
# Características de la nueva casa, incluyendo ocean_proximity
new_house = {
    'longitude': -118.25,
    'latitude': 34.05,
    'housing_median_age': 20,
    'total_rooms': 800,
    'total_bedrooms': 3,
    'population': 300,
    'households': 150,
    'median_income': 5.0,
    'rooms_per_hhold': 5.33,
    'pop_per_hhold': 2.0,
    'bedrooms_per_room': 0.37,
    'ocean_proximity': '<1H OCEAN' 
}


# Crear un DataFrame
new_house_df = pd.DataFrame([new_house])


# In[89]:


# Transformar los datos
new_house_prepared = full_pipeline.transform(new_house_df)


# In[90]:


predicted_price = final_model.predict(new_house_prepared)


# In[91]:


print(f"El precio predicho de la casa es: ${predicted_price[0]:,.2f}")


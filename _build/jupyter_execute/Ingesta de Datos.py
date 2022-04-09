#!/usr/bin/env python
# coding: utf-8

# # Ingesta de Datos
# 
# ## Pipelines de Procesamiento de datos
# ## Datos numéricos
# 
# <p align="center">
# <img src="https://www.jeremyjordan.me/content/images/2018/01/Screen-Shot-2018-01-23-at-2.27.20-PM.png" style="width: 600px;"/>
# </p>
# 
# La **normalización** es una técnica que a menudo se aplica como parte de la preparación de datos para el aprendizaje automático. El objetivo de la **normalización** es cambiar los valores de las columnas numéricas en el conjunto de datos para usar una escala común, sin distorsionar las diferencias en los rangos de valores ni perder información. La normalización también es necesaria para que algunos algoritmos modelen los datos correctamente.
# 
# Por **ejemplo**, suponga que su conjunto de datos de entrada contiene una columna con valores que van de 0 a 1 y otra columna con valores que van de 10,000 a 100,000. La gran diferencia en la escala de los números podría causar problemas al intentar combinar los valores como características durante el modelado.
# 
# 
# La normalización evita estos problemas al crear nuevos valores que mantienen la distribución general y las proporciones en los datos de origen, mientras mantienen los valores dentro de una escala aplicada en todas las columnas numéricas utilizadas en el modelo.
# 
# **Tenemos varias opciones para transformar datos numéricos:** 
# - Cambiar todos los valores a una escala de 0 a 1 o transformar los valores representándolos como rangos de percentiles en lugar de valores absolutos.
# - Aplicar la normalización a una sola columna o a varias columnas en el mismo conjunto de datos.
# - Si necesita repetir el experimento o aplicar los mismos pasos de normalización a otros datos, puede guardar los pasos como una transformación de normalización y aplicarlos a otros conjuntos de datos que tengan el mismo esquema.
# 
# ## Normalizacion Lineal
# 
# - **Z-Score**: convierte todos los valores en una puntuación z. Los valores de la columna se transforman mediante la siguiente fórmula: 
# 
# <p align="center">
# <img src="https://user-images.githubusercontent.com/63415652/122654703-e36b2b00-d112-11eb-847e-5ca8ff3288c5.PNG" style="width: 300px;"/>
# </p>
# La media y la desviación estándar se calculan para cada columna por separado. Se utiliza la desviación estándar de la población.
# 
# - **MinMax**: el normalizador min-max cambia la escala linealmente cada característica al intervalo [0,1]. El cambio de escala al intervalo [0,1] se realiza cambiando los valores de cada característica para que el valor mínimo sea 0, y luego dividiendo por el nuevo valor máximo (que es la diferencia entre los valores máximo y mínimo originales). Los valores de la columna se transforman mediante la siguiente fórmula: 
# <p align="center">
# <img src="https://miro.medium.com/max/992/1*tbBScNpSloQ9uOd3QYNVtg.png" style="width: 300px;"/>
# </p>
# 
# La siguiente formula es para hacer un escalamiento de mix max pero en el intervalo de [-1, 1], donde a y b son el rango de valores que queremos dar es decir [a=-1 y b = 1]
# 
# <p align="center">
# <img src="https://static.platzi.com/media/user_upload/general%20formula-97b6d8eb-b206-4e45-9bb4-d44c948060b7.jpg" style="width: 300px;"/>
# </p>
# 
# Otros tipos de escalamiento
# - Cliping
# - Winsorizing
# - Log scaling
# 
# **Cuando usar la normalizacion lineal?** > En datos simétricos o en datos uniformemente distribuidos
# 
# ## Normalizacion no lineal
# 
# uando la distribución de datos no es simétrica sino sesgada se usa la transformación no lineal.
# Esto con el fin de tomar los datos con una distribución no simétrica y se transforman en una distribución que si es simétrica.
# Después de eso se aplican los escaladores lineales.
# 
# Algunos tipos de transformacion lineal:
# - Logística: los valores de la columna se transforman mediante la siguiente fórmula:
# <p align="center">
# <img src="https://user-images.githubusercontent.com/63415652/122654862-cedb6280-d113-11eb-9d0e-26a1991537d2.PNG" style="width: 300px;"/>
# </p>
# 
# - Log Normal: esta opción convierte todos los valores a una escala logarítmica normal. Los valores1 de la columna se transforman mediante la siguiente fórmula:
# <p align="center">
# <img src="https://user-images.githubusercontent.com/63415652/122654861-ce42cc00-d113-11eb-92cc-64fc408fa338.PNG" style="width: 300px;"/>
# </p>
# 
# Aquí μ y σ son los parámetros de la distribución, calculados empíricamente a partir de los datos como estimaciones de máxima verosimilitud, para cada columna por separado.
# - TanH: todos los valores se convierten a una tangente hiperbólica. Los valores de la columna se transforman mediante la siguiente fórmula:
# 
# <p align="center">
# <img src="https://user-images.githubusercontent.com/63415652/122654864-cf73f900-d113-11eb-9b7d-09ff12d92ca4.PNG"/>
# </p>
# 
# ## Escalamiento de datos numéricos
# ### Transformaciones Lineales
# La idea de hacer pre-procesamiento de datos, es para ayudar al algoritmo con su convergencia en los modelos de ML.
# Para eso utilizaremos la libreria **timeit**, verificando el tiempo de ejecucion de los modelos.
# 

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import timeit

from sklearn import datasets, linear_model
#aqui extraemos el indice de masa corporal y lo asignamos a X and Y. Por el momento solo usaremos X luego
#en el modelo de entrenamiento usaremos Y
X,y = datasets.load_diabetes(return_X_y=True) 
raw = X[:,None, 2]


# Hasta en la celda anterior lo que hemos hecho es importar las librerias que usaremos. Para el ejemplo usaremos un dataset de pacientes con diabetes donde `X` toma el valor de su primera columna (que es el indice de masa corporal) e igualmente `y`. 
# 
# Luego ``raw`` toma lo usaremos para hacer una escala en el intervalo $-1 > x >1$. Recordar que antes de hacer un escalamiento tenemos que verificar que nuestros datos no contengan una distribucion tan sesgada. Utilizaremos entonces la formula de escalamiento **MinMax** en el rango de [-1, 1]
# 
# 

# In[2]:


#reglas de escalamiento
max_raw = max(raw)
min_raw = min(raw)
# x_prima = -1 + (((raw-min_raw)*(2))/(max_raw - min_raw)) esta formula es la misma a la que le sigue
scaled = (2*raw - max_raw - min_raw)/(max_raw - min_raw)
#vamos a imprimir primero el histograma crudo sin realizar su escalamiento (axs[0]) y con su escalamiento (axs[1])
fig, axs = plt.subplots(2,1, sharex=True) #aqui le decimos que compartiremos el mismo eje x.
#tambien le decir que realizamos 2 figuras con una sola columna
axs[0].hist(raw) #histograma sin escalamiento
axs[1].hist(scaled) #histograma con escalamiento


# In[3]:


#modelo para entrenamiento, esto lo usaremos solo para medir el tiempo de ejecucion
#entre un modelo crudo `X` y otro con escalamiento `scaled`

def train_raw():
    linear_model.LinearRegression().fit(raw,y)

def train_scaled():
    linear_model.LinearRegression().fit(scaled, y)


# In[4]:


raw_time = timeit.timeit(train_raw, number=100)
scaled_time = timeit.timeit(train_scaled, number=100)
#como vemos el modelo no escalado y demora mas en completar el modelado que la distribucion normalizada.
#quizas no es muy grande la diferencia pero si podemos asumir que el tiempo de demora aumenta con datos mas grande.
print(raw_time, scaled_time)


# - max-min scaling: mejor para datos uniformemente dristribuidos
# - z-score scaling: mejor para datos distribuidos "normalmente" (forma de campa de gauss)
# De igual forma tenemos que comprobar esto para nuestros datos en especifico, es algo relativo.
# 
# ### Transformaciones No lineales.
# 
# En la introducción vimos que los precios de autos de segunda mano estan muy sesgados de forma negativa. Por esa razon utilizaremos ese ejemplo para realizar el escalamiento.

# In[5]:


df_price_cars = pd.read_csv('./sources/datasets/cars_.csv')
df_price_cars.price_usd.hist()
#Como vemos la distribucion se encuentra fuertemente sesgada entre 0> x > 10 000


# In[6]:


#Realizaremos una transformacion con tangente hiperbolica (tangh(x))

df_price_cars['price_usd'].apply(lambda x: np.tanh(x)).hist()


# En este caso la normalizacion tangh apilo todo en un solo valor (aprox 1). Sin embargo nosotros podemos calibrar esa distribucion dividiendolo por un numero. En este caso lo llamaremos `p` y le asignaremos ``10000``

# In[7]:


#Realizaremos una transformacion con tangente hiperbolica (tangh(x/p))
p = 10000
df_price_cars['price_usd'].apply(lambda x: np.tanh(x/p)).hist()


# Con esta calibracion re-distribuimos los datos de manera que queden mas uniforme.

# --------------------------------
# A modo de practica escalaremos el datasets `load_diabetes['bmi']` por medio de **z-score.**

# In[8]:


from sklearn.datasets import load_diabetes

diabetes=load_diabetes()

df_diabetes = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
df_diabetes


# In[9]:


u_diabetes  = df_diabetes['bmi'].mean()
std_diabetes = df_diabetes['bmi'].std()


# In[10]:


z_score = (df_diabetes['bmi'] - u_diabetes)/std_diabetes


# In[11]:


df_diabetes['bmi'].hist()


# In[12]:



fig, axs = plt.subplots(2,1, sharex=True)

axs[0].hist(df_diabetes['bmi'])
axs[1].hist(z_score)


# ## Mapeos numéricos
# 
# ![dummy](https://miro.medium.com/max/1400/1*80tflY8LxDFRmkD16u25RQ.png)
# 
# Luego de haber explicado el escalamiento o modelación de las variables numéricas iscretas y continuas. Nos preguntamos entonces... que se hace con aquellas características que son categóricas; características como idioma, sexo, grado academico, nacionalidad son algunos ejemplos que podemos encontrarnos en una problemática. Para procesar este tipo de dato realizamos un **Mapeo Numérico**, el cual consiste en un conjunto de combinaciones binarias que benefician su analisis en un modelo de machine learning. Los dos métodos que existen para interpretar variables categóricas son:
# 
# - Dummy: es la representación más compacta que se puede tener de los datos. Es mejor usarla cuando los inputs son variables linealmente independientes (no tienen un grado de correlación significativo). Es decir, las cuando se sabe que las categorías son independientes entre sí.
# 
# - One-hot: es más extenso. Permite incluir categorías que no estaban en el dataset inicialmente. De forma que si se filtra una categoría que no estaba incluida, igual se pueda representar numéricamente y no de error en el modelo (este modelo es más cool y es el que se usa).
# Hay errores en la notación de Pandas y los tratan como que ambos modelos son lo mismo, pero en la realidad el Dummy no se usa. Aún así, en Pandas el método es .get_dummies().
# 
# 
# **Vamos a la práctica**
# En nuestro caso utilizaremos la funcion get_dummies de pandas que en realidad es one-hot.

# In[13]:


import pandas as pd


# In[14]:


df_cars = pd.read_csv('./sources/datasets/cars_.csv')
df_cars.head(2)


# In[15]:


#Vamos a obtener el one-hot de engine type.
pd.get_dummies(df_cars['engine_type']).sample(4)


# In[16]:


#Vamos a utilizar scikitlearn para obtener one hot
import sklearn.preprocessing as preprocessing
encoder = preprocessing.OneHotEncoder(handle_unknown='ignore')
encoder.fit(df_cars[['engine_type']].values)


# In[17]:


#Aceite no es un tipo de motor valido por lo tanto scikit learn lo asigna como 0, igual sucede para otras
#categorias que no se encuentren dentro de la columna
encoder.transform([['gasoline'],['diesel'],['aceite']]).toarray()


# In[18]:


#Vamos a tratar una variable numerica discreta como categorica en este caso el año de fabricacion del auto
encoder.fit(df_cars[['year_produced']].values)


# In[19]:


#Vamos a introducir dos valores posibles y otro año erroneo 190
encoder.transform([[2009],[2016],[190]]).toarray()
#Si contamos todos los elementos dentro de las listas podemos ver los años posibles que podemos encontrar
#Y cuando vemos un 1 quiere decir que ha encontrado su categoria, sin embargo el año 190 (última lista)
#esta llena de 0 con lo cual no ha encontrado su categoría o año de fabricación.


# ## Correlaciones
# 
# Las correlacion es una medida que intenta determinar el nivel de relacion lineal que existe entre dos características.  
# 
# - **Varianza** Medición que indica el grado de variación conjunta de dos variables aleatorias respecto a sus medidas.
# 
# - **Coeficiente de correlacion** El coeficiente de correlación es la medida específica que cuantifica la intensidad de la relación lineal entre dos variables en un análisis de correlación.
# 
# Pero... que tienen en común estos conceptos. Ocurre que a través de la formula de varianza podemos calcular el alejamiento de cada dato con respecto a su media; por otra parte, la covarianza calcula el alejamiento de una variable con respecto a otra. Es decir, si la variable `x` aumenta como se comporta la variable `y`.
# 
# Otro aspecto a tener en cuenta es que la escala de cada variable normalmente es diferente. Un ejemplo hipotético es que la escala de `x` puede ser de 0 hasta 190 y la escala de `y` puede ser de 300 a 900, entonces es necesario estandarizar las escalas. Por tal razon el resultado de la covarianza se divide entre  "/" el producto de la desviacion estandar de `X` y `Y` 
# 
# Vamos a verlo a traves de formulas.
# 
# <p align = "center">
# <img src = "https://economipedia.com/wp-content/uploads/Varianza-formula.jpg" style="width:400px">
# </p>
# <p align = "center">
# formula de la varianza 
# </p>
# 
# <p align = "center">
# <img src = "https://economipedia.com/wp-content/uploads/2017/10/F%C3%B3rmula-de-la-Covarianza-tama%C3%B1o-extenso.jpg" style="width:400px">
# </p>
# <p align = "center">
# formula de la COvarianza
# </p>
# 
# <p align = "center">
# <img src = "https://i.pinimg.com/736x/64/0d/be/640dbe04c45777296ce29d72f3bc1ca9.jpg" style="width:400px">
# </p>
# <p align = "center">
# Coeficiente de correlación
# </p>
# 
# 
# El resultado de los coeficiente de correlacion entre dos variables pueden tener distintos patrones.
# <p align = "center">
# <img src = "https://i.imgur.com/0AKQKBi.png" style="width:700px">
# </p>
# <p align = "center">
# Gráficos de correlaciones
# </p>
# 
# Aun cuando veamos una correlación fuerte entre dos variables debemos tener en cuenta que 
# > Correlación no implica causalidad.
# 
# Es importante siempre revisar los datos arrojados y en ocasiones puede que la fuerte correlacion sea meramente coindicencia y no esten estrechamente correlacionadas.

# ## Matriz de varianza
# 
# <p align = "center">
# <img src = "https://miro.medium.com/max/1400/1*J6z7xcleH9wxHGGCLvDptg.jpeg" style="width:700px">
# </p>
# <p align = "center">
# Matriz de covarianza
# </p>
# 
# La matriz varianza–covarianza es una matriz cuadrada de dimensión nxm que recoge las varianzas en la diagonal principal y las covarianzas en los elementos de fuera de la diagonal principal.
# 
# En otras palabras, la matriz varianza-covarianza es una matriz que tiene el mismo número de filas y columnas y que tiene distribuidas las varianzas en la diagonal principal y las covarianzas en los elementos fuera de la diagonal principal.

# In[20]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as  sns
from sklearn.preprocessing import StandardScaler

iris = sns.load_dataset('iris')


# In[21]:


sns.pairplot(iris, hue='species')
#Aqui podemos ver a nivel de grafico el nivel de correlacion.


# In[22]:


#Utilizando scikit learn para obtener la matriz de covarianza
scaler = StandardScaler()
scaled = scaler.fit_transform(
    iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
)

#Scaled son los datos ya normalizados, y se usa la traspuesta para aplicar la correlacion
scaled.T


# In[23]:


#Obteniendo matriz de covarianza a traves de scikit learn
covariance_matrix = np.cov(scaled.T)
covariance_matrix


# In[24]:


#matriz de covarianza a traves de pandas
iris.corr()


# In[25]:


#matriz de covarianza a traves de seaborn
sns.heatmap(iris.corr(), annot=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





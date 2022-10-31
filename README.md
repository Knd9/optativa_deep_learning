# Primer Trabajo de Aprendizaje Profundo

> Aplicación de lo aprendido en la materia optativa **Aprendizaje Profundo** de la Diplomatura en Ciencias de Datos FaMAFyC 2022.

> **Integrantes**: Candela Spitale | Fernando Cardellino | Carina Giovine | Carlos Serra

> **Profesores**: Johanna Frau y Mauricio Mazuecos

> **Problema**: Predecir la categoría de un artículo de MELI a partir de los títulos.

## Análisis y Visualización

Hicimos un análisis y visualización de la base de datos de **Meli Challenge 2019**, en los conjuntos de datos de `entrenamiento`, `validación` y `test` del idioma **spanish**.

Para cada conjunto, vimos:

* Cantidad y nombre de columnas con sus tipos:
  - object: `language`, `label_quality`, `title`, `category`, `split`, `tokenized_title`, `data`
  - int64: `target`, `n_labels`, `size`

* Cantidad de registros totales (`size`): diferían entre conjuntos, lógico debido a la división usual de los tipos de conjuntos, más cantidad en el de entrenamiento.
* Cantidad de valores nulos: ninguna para cualquier conjunto
* Primeros registros del conjunto para visualizar algunos valores únicos
* Cantidad de categorías distintas (`n_labels`): igual cantidad para todos, 632
* Cantidad de títulos distintos: coincidían en la cantidad de registros

## Preprocesamiento y tokenización de los datos

Luego de encontrar que los conjuntos a tratar ya están preprocesados y tokenizados, con las columnas `data` referente a **title** y `target` referente a **category**, no fue necesario hacer un preprocesamiento y tokenización, con acceder a estas columnas bastaba. El procedimiento de este desarrollo se encuentra en el repositorio de la materia en el archivo [experiment/preprocess_meli_data](https://github.com/DiploDatos/AprendizajeProfundo/blob/master/experiment/preprocess_meli_data.ipynb).ipynb*, donde también se puede encontrar el origen de otras columnas como `tokenized_title`, `n_labels` y `size`.

Básicamente, primero se concatenan los 3 conjuntos para evitar que el proceso en los datos no asigne el mismo token a la misma palabra en los distintos conjuntos (train/validation vs test) y ser una posible causa de un bajo rendimiento en el conjunto de test. Para el **preprocesamiento** se utilizan los módulos `stopwords`, `word_tokenize` de la librería `nltk` y el módulo `preprocessing` de la librería `gensim` los conjuntos concatenados. En cuanto a la **tokenización** que sigue, se utiliza el modelo `Dictionary` de el módulo `corpora` de la librería `gensim` y varios métodos de este modelo para lograrlo.

Como resultado se guarda por un lado, el conjunto *spanish_token_to_index.json.gz* para usarlos en los embeddings y, por otro lado, los 3 conjuntos por separadp, para poder dividir en las etapas de entrenamiento, validación y prueba final, en los archivos *spanish.test.jsonl.gz*, *spanish.train.jsonl.gz*, *spanish.validation.jsonl.gz*

## Manejador del dataset

Creamos una clase para modelar un conjunto de datos (cualquiera de los 3 que se instancie), que hereda de la clase `IterableDataset` de PyTorch. Si bien, no permite hacer shuffling de datos de forma fácil como la clase `Dataset` de Pytorch, el conjunto de datos es bastante grande (y podría serlo aún más en otro año para levantarlo en memoria).

Instanciamos los 3 conjuntos de datos con este módulo.

Creamos los dataloaders para cada conjunto haciendo uso de la clase `Dataloader`, que nos ayuda a entrenar con *mini-batches* (elegimos `128` para agilizar el tiempo de entrenamiento) al modelo para aumentar la eficiencia evitando iterar de a un elemento.

Para su definición creamos el módulo `PadSequences` para usar de *collation function* en el parámetro `collate_fn` que, dado que trabajamos con secuencias de palabras (representadas por sus índices en un vocabulario) y que el dataloader espera que los datos del *batch* tengan la misma dimensión (para poder llevarlos todos a un tensor de dimensión fija), se necesita redefinir los valores máximo, mínimo y de relleno distintos de los que están por defecto, de forma que dada una lista de secuencias, devuelve un tensor con *padding* sobre dichas secuencias.

Como en este caso trabajamos con secuencias de palabras (representadas por sus índices en un vocabulario), cuando queremos buscar un *batch* de datos, el `DataLoader` de PyTorch espera que los datos del *batch* tengan la misma dimensión (para poder llevarlos todos a un tensor de dimensión fija). Esto lo podemos lograr mediante el parámetro de `collate_fn`. En particular, esta función se encarga de tomar varios elementos de un `Dataset` y combinarlos de manera que puedan ser devueltos como un tensor de PyTorch. Muchas veces la `collate_fn` que viene por defecto en `DataLoader` sirve (como se vio en el notebook 2), pero este no es el caso. Se define un módulo `PadSequences` que toma un valor mínimo, opcionalmente un valor máximo y un valor de relleno (*pad*) y dada una lista de secuencias, devuelve un tensor con *padding* sobre dichas secuencias.

## Clase para el modelo

Para la clasificación utilizamos un modelo de red perceptrón multicapa que cuenta con 4 capas ocultas. No profundizamos mucho para esta desición, entendimos que es arbitraria fuera de que el input y output obliguen a que al menos haya dos capas ocultas, y luego de explorar se podía decidir mejor.

En particular, tenemos la primera capa de `embeddings` que es rellenada con los valores de **word embeddings** (conversión del texto a una representación por vectores) continuos preentrenados en español de [SBW](https://crscardellino.ar/SBWCE/), de 300 dimensiones (descargado en la carpata `data`). Estos están en formato bz2, por lo cual con la librería `bz2` pudimos  descomprimir el archivo que los contiene. A su vez instanciamos el resto de las capas de la red con los tamaños pasados como argumento.

Además en la función de *forward*, aplicamos la matriz de embeddings ya creada al input y estandarizamos el ancho de la matriz (ya que el MLP lo necesita) con el promedio de cada vector de la matriz tensor. Posteriormente, aplicamos al resultado la función de activación `Relu` (mencionado en clase que es la que más se utiliza) a lo largo de las capas ocultas de la red, y luego aplicamos la capa del output.

### 1ra Parte: Red Perceptrón Multicapa

> Creamos las funciones:

* **train_model**: entrena el modelo; por épocas ejecuta el **Back Propagation** y la **Optimización** con el optimizador pasado, calculando la función de pérdida; y por último loguea la época, la iteración y la función de pérdida de entrenamiento (`train_loss`) cada `50` mini-batches.

* **train_and_eval**: hace algo similar que la función anterior solo que también para el conjunto de **validación** (también valida) y que loguea la función de pérdida de entrenamiento (`train_loss`) por época y además loguea la métrica de `balance_accuracy` (aprovechando que usamos el conjunto de validación, decidimos medirla en este conjunto para compararla luego con la del conjunto de **test**) y la función de pérdida para el conjunto de validación (`val_loss`).

* **test_model**: evaluamos y predecimos con el conjunto de test y reportamos la métrica de `balance_accuracy` para este conjunto.

y dos funciones más donde usamos MLFlow:

* **run_experiment**: ejecutamos un run del experimento; asignamos la función de pérdida `CrossEntropyLoss` al trabajar con un problema multiclase, llamamos a `train_and_eval` y a `train_model` con los dataloaders pasados y, si se desea además testear, llamamos a `test_model`. Registramos los **hiperparámetros**: la arquitectura del modelo, la función de pérdida, las épocas, la taza de aprendizaje y el optimizador.

* **run__mlflow_experiment**: ejecutamos un experimento; instanciamos el modelo pasando como parámetros el archivo de word embeddings, los datos tokenizados, el tamaño de vector (el tamaño de los embeddings, 300), el uso de barras de progreso activado, y los tamaños de las capas. Enviamos el modelo a GPU y loguemos los **hiperparámetros**. Corremos el run con los dataloaders de entrenamiento y validación y, si se desea testear, agregamos el dataloader de test. Por último logueamos las métricas devueltas del run en MLFlow y calculamos las predicciones de a batches guardándolas en un archivo nuevo comprimido como artefacto de MLFlow.

Por último, creamos dos experimentos:

* `experiment_w_3_epochs_l4`: para las etapas de entrenamiento y validación, el cual se comprime.

* `test_experiment_w_3_epochs`: para la etapa de testeo, el cual se comprime.

### Arquitectura e hiperparámetros:

En todos los runs del experimento hacemos una red perceptrón multicapa de 4 capas ocultas con los siguientes tamaños:

* 1024 para la primera capa, considerando que el tamaño de input es 300, vamos aumentando las dimensiones
* 2048 para la segunda capa oculta
* 4096 para la tercera capa oculta
* 2048 para la última capa, que discrimina entre 632 categorías de salida (casi el doble que el tamaño de input) reduciendo a la mitad en relación a la 3er capa oculta y duplicando en relación a la 1ra capa.

Además, utilizamos `3` **épocas**, de forma arbitraria considerando que es un mínimo para agilizar el modelo.

Variamos a modo de exploración algo aleatoria el **optimizador** y la **taza de aprendizaje**, si bien optamos por los dos optimizadores que se mencionaron más usuales en clase (`Adam` y `RMSprop`), y distintos valores de tazas de aprendizaje que reducimos a cantidad de dos (`0.0001` y `0.001`) debido a la gran demora en tiempo de ejecución y problemas de conexión con la máquina externa proporcionada que aumentaban el tiempo de dedicación al trabajo. Sin embargo, probando anteriormente con valores mayores de taza de aprendizaje nos dimos cuenta que a medida que aumentaba el valor se reducía el rendimiento del modelo, por lo cual, escogimos dos valores que ya devolvían diferencias grandes de la métrica considerada.

### Experimento de Entrenamiento y Validación

#### Vista general de runs

> **Duración** 4to Run | 3er Run | 2do Run | 1er Run

<img src='https://drive.google.com/uc?id=1clEZLoFcQ22-DmAMraWS_ekkEcn03h2Q' name='DuracionGeneral'>

> **Hiperparámetros** 4to Run | 3er Run | 2do Run | 1er Run

<img src='https://drive.google.com/uc?id=1p6dGLZIZsSHFPfzXYm_lT4NkAHIePk8X' name='ParamsGeneral'>

> **Métricas** 4to Run | 3er Run | 2do Run | 1er Run

<img src='https://drive.google.com/uc?id=1vtayyBbdGVGmQmy7VsCBKYZhwoJEw56-' name='MetricasGeneral'>

> Comparación de **loss** a través de las 3 épocas en entrenamiento y validación

<img src='https://drive.google.com/uc?id=1z84GJLy4O_fcD8exVrw3eTEs228znyiP' name='train_loss'>

<img src='https://drive.google.com/uc?id=1l0yIOfcAWIz1G4JTLmtNYiwVN8PHvyd4' name='val_loss'>

> Comparación **train_loss** vs **optimizador**

<img src='https://drive.google.com/uc?id=1Sz0ObmqQJ51eC75ztC-xZTWCWOCYwQT7' name='train_loss_vs_Optim'>

> Comparación **val_loss** vs **optimizador**

<img src='https://drive.google.com/uc?id=1XJjG0q_64P0CgVUxyfeMjjBytChHG6Ts' name='val_loss_vs_Optim'>

#### Vista particular de runs

* **1er Run**

> **Hiperparámetros**

<img src='https://drive.google.com/uc?id=1bJhIB1imNqzyiRfzmHpXWFr-uXVFcSCb' name='1erRunParametros'>

> **Métricas**

<img src='https://drive.google.com/uc?id=1vKkbjT44gdqaKRylkF1vRemPO-mhuxnP' name='1erRunMetricas'>

* **2do Run**

> **Hiperparámetros**

<img src='https://drive.google.com/uc?id=1AmwopSOJ59YGQevlGJkA1bM6MpLRUZpu' name='2doRunParametros'>

> **Métricas**

<img src='https://drive.google.com/uc?id=1_bzskr_nVujlbQpbBsenuyZjzN5rnWz6' name='2doRunMetricas'>

* **3er Run**

> **Hiperparámetros**

<img src='https://drive.google.com/uc?id=12_So6XTA8l5coeF-IyGnZDfm9V__KQc7' name='3erRunParametros'>

> **Métricas**

<img src='https://drive.google.com/uc?id=1rXuoB6RTCtGW0KU-rgr1_jy4x5VVYHIn' name='3erRunMetricas'>

* **4to Run**

> **Hiperparámetros**

<img src='https://drive.google.com/uc?id=1GtA1JRIBbtXKOuy7lECN7AEmElOxjaoq' name='4toRunParametros'>

> **Métricas**

<img src='https://drive.google.com/uc?id=1bwhGAyKWurdlY5gFT2VvQdH0Or7zMpIv' name='4toRunMetricas'>


### Experimento de Validación y Test con el mejor modelo entrenado

#### Optimizador **Adam** y Learning Rate **0.0001**

> **Hiperparámetros**

<img src='https://drive.google.com/uc?id=1if0f95ecZSwbC1pbVubp1JiCBhy9117F' name='TestParams'>

> **Métricas**

<img src='https://drive.google.com/uc?id=1hLwrnUcLn5mMSwK2XkgqT6S1zVHk_Xov' name='TestMetricas'>


## Conclusión general

* Se logró obtener un buen valor de la métrica `balanced_accuracy` con el conjunto de test: **0.81**.
* Como próximos pasos, luego de haber obtenido un resultado que consideramos satisfactorio, la idea sería ver si con una red más compleja los valores obtenidos para balanced_accuracy pueden incrementarse. [Ver 2da Parte](https://github.com/FCardellino/DeepLearning).

## Contenido:

* `TP_AprendizajeProfundo.ipynb`: Jupyter Notebook con el trabajo resuelto.
* `README.md`: Informe del trabajo presentado
* Directorio `data/`:
  - Directorio `experiments`: contiene los experimentos comprimidos `op_experiments_w_3epochs_4l.csv.gz` y `test_op_experiments_w_3epochs_4l.csv.gz`

## Notas:

* Durante la ejecución del trabajo se descargarán el siguiente directorio y archivo dentro del directorio `data/`:

  - Directorio `meli-challenge-2019`: datasets del Melli Challenge 2019. Conjuntos a utilizar referentes a entrenamiento, validación y test en español y su concatenación, respectivamente: `spanish.train.jsonl.gz`, `spanish.validation.jsonl.gz`, `spanish.test.jsonl.gz`, `spanish_token_to_index.json.gz`

  - `SBW-vectors-300-min5.txt.bz2`: archivo de Word Embeddings utilizado

  - Los experimentos comprimidos resultantes de una nueva ejecución se guardarán en el directorio `data/`. Los obtenidos para esta entrega se guardaron en el directorio `experiments` para que no colisionen los nombres durante otra ejecución y se pueda continuar con la misma.

* Se utilizó la máquina externa nabucodonosor proporcionada para ejecutar el trabajo con recursos más grandes. Se puede acceder a la misma y ejecutarlo entrando a la terminal y corriendo:

$ `ssh -L localhost:{PORT}:localhost:{PORT} {usernabu}@nabucodonosor.ccad.unc.edu.ar`

(Instalar todos los paquetes y crear entorno virtual dictados en [0_set_up](https://github.com/DiploDatos/AprendizajeProfundo/blob/master/0_set_up.ipynb)).

Luego:

$ `git clone https://github.com/Knd9/optativa_deep_learning.git`

$ `jupyter notebook --port {PORT} --no-browser`

Para ver MLFlow

Cerrar el jupyter y correr:

$ `mlflow ui --port {PORT}`

<h1 align="center">Re-MVAE Framework</h1>

<br>
<br>

<div align="justify">

En este directorio se encuentra todo el código fuente necesario para la repetición, planificación y ejecución de experimentos. Cada experimento se define formalmente dentro del directorio `experiments/`.

## 1. Crear un experimento

Para crear un experimento, será necesario crear un directorio dentro de `experiments/` que contendrá el código fuente del experimento. Supongamos, por ejemplo, que queremos crear un experimento llamado `example`.

```bash
mkdir experiments/example
```

Hecho esto, habrá que definir el experimento, creando para ello un fichero `args.json` en su directorio principal.

```bash
touch experiments/example/args.json
```

Dicho fichero contendrá un objeto JSON el cual contenga todos los argumentos necesarios para la creación de sus scripts y ejecución de los mismos. En la implementación base, los posibles argumentos son los siguientes:

```json
{
    "name": "[str: NOMBRE DEL DIRECTORIO DEL EXPERIMENTO (experiments/NOMBRE)]",
    "reader": "[str: NOMBRE DATASET (ver readers/)]",
    "text_architecture": "[str: NOMBRE DE LA ARQUITECTURA PARA TEXTO]",
    "image_architecture": "[str: NOMBRE DE LA ARQUITECTURA PARA IMAGEN]",
    "tokenizer_path": "[str: RUTA DONDE GUARDAR EL TOKENIZADOR DEL DATASET]",
    "dataset_length": "[int: NÚMERO DE MUESTRAS QUE QUEREMOS EN EL DATASET]",
    "image_size": "[int: RESOLUCIÓN DE LA IMAGE, ej. (128)x128]",
    "latent_dim": "[int: DIMENSIÓN DEL ESPACIO LATENTE]",
    "conv_dims": "[list[int,]: DIMENSIONES DE LA ARQUITECTURA DE IMAGEN, SI ES CONVOLUCIONAL]",
    "dims": "[list[int,]: DIMENSIONES DE LA ARQUITECTURA DE IMAGEN, SI ES LINEAL]",
    "embedding_dim": "[int: TAMAÑO DE EMBEDDINGS RNN]",
    "hidden_dim": "[int: TAMAÑO DEL ESTADO OCULTO RNN]",
    "context_length": "[int: TAMAÑO DE VENTANA DE CONTEXTO RNN]",
    "batch_size": "[int: TAMAÑO DEL BATCH]",
    "epochs": "[int: NÚMERO DE ÉPOCAS DE ENTRENAMIENTO]",
    "trainer": "[str: NOMBRE DEL TRAINER]",
    "training_method": "[int: TIPO DE ANNEALING A APLICAR]",
    "weights": "[dict(str, int): PESOS DE ENTRENAMIENTO]",
    "teacher_forcing_ratio": "[float: PROCENTAJE DE APLICACIÓN DE TEACHER FORCING]",
    "k": "[int: PESO ASOCIADO AL ANNEALING]",
    "x0": "[int: PESO ASOCIADO AL ANNEALING]",
    "results_dir": "[str: RUTA DE RESULTADOS]",
    "checkpoint_dir": "[str: RUTA DE CHECKPOINTS]",
    "checkpoint_steps": "[int: INTERVALO DE CHECKPOINTS]",
    "evaluators": "[list(str): EVALUADORES]"
}
```

Los parámetros `reader`, `text_architecture` e `image_architecture` deben coincidir con el módulo donde se encuentran definidas estas arquitecturas.

Para crear el experimento a partir del fichero `args.json`, se ha de usar el módulo `generator/`, según el siguiente comando:

```bash
python generator/generate.py experiments/example/args.json
```

Con esto, el experimento estará creado.

## 2. Ejecutar un experimento

Una vez creado el experimento, se puede acceder al directorio del mismo y ejecutarlo:

```bash
cd experiments/example
chmod +x execute.sh && execute.sh
```


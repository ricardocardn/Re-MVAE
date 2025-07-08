<h1 align="center">Re-MVAE Framework</h1>

<br>
<br>

<div align="justify">

En este directorio se encuentra todo el código fuente necesario para la repetición, planificación y ejecución de experimentos. Cada experimento se define formalmente dentro del directorio `experiments/`.

## 1. Crear una arquitectura

Para definir una nueva arquitectura variacional, se debe crear un nuevo directorio dentro del módulo `playground/architectures/`. La convención de nombres utilizada es `CamelCase`, ya que facilita su integración con el módulo `generator/`.

Una vez creado el directorio, hay que implementar la arquitectura para que pueda ser utilizada en los experimentos. Para ello, es fundamental definir una clase llamada `Builder`, la cual debe devolver un objeto que herede de la interfaz `TextVAE` o `ImageVAE` y que implemente la lógica específica de la arquitectura. Por convención, esta clase debe estar en un archivo llamado `builder.py`.

Además, para que el sistema pueda importar correctamente la clase, se debe añadir un archivo `__init__.py` dentro del directorio, con el siguiente contenido:

```python
from .builder import Builder
```

### Plantilla de instanciación (build.template)

Para permitir que el módulo generator/ sepa cómo instanciar la arquitectura, es necesario crear un archivo build.template en el mismo directorio. Este archivo contiene una plantilla de código que especifica cómo debe construirse la arquitectura utilizando la clase `Builder`. Por ejemplo:

```python
image_model = ImageBuilder().build(
    args["image_size"], args["input_channels"], args["latent_dim"], args["conv_dims"]
)
```

La plantilla asume que existen ciertos argumentos en el diccionario `args`, tales como `image_size`, `input_channels`, etc. Estos parámetros deben estar definidos en el archivo `args.json`, descrito en la sección 3. Crear un experimento, y pueden modificarse para incluir nuevos argumentos según se requiera.

### Importación de librerías externas (libs.template)

Si la arquitectura necesita librerías adicionales que no forman parte del ecosistema PyTorch, se deben importar explícitamente mediante el archivo `libs.template`. Este archivo contiene las instrucciones necesarias para realizar dichas importaciones en el contexto del experimento.

Por ejemplo, en la arquitectura `xLSTMSeq2seqBidirectionalAutoregressive`, el contenido de libs.template puede ser:

```python
from omegaconf import OmegaConf
from dacite import from_dict, Config as DaciteConfig
from xlstm import xLSTMBlockStackConfig
```

## 2. Importar un dataset

Para importar un conjunto de datos, simplemente se debe definir una clase encargada de leer y adaptar dichos datos a un formato interpretable por los modelos. Estas clases se denominan readers.

Para crear un nuevo Reader, es necesario generar un subdirectorio en el módulo `playground/readers/`, nombrado también en `CamelCase`. Dentro de este subdirectorio, los datos deben almacenarse en una carpeta `data/`, y la clase Reader debe implementarse en un archivo llamado `reader.py`. Esta clase será la responsable de cargar y transformar los datos, y el constructor SIEMPRE recibirá los mismos parámetros: `train`: boolean, `transform`: Callable y `len`: int.

La clase `Reader` debe extender la clase Dataset de PyTorch. Esto permite integrarla fácilmente con los DataLoader, lo que facilita el manejo eficiente de lotes durante el entrenamiento y garantiza una compatibilidad directa con el pipeline de entrenamiento del modelo.

A continuación, se muestra como ejemplo el Reader correspondiente al clásico dataset MNIST:

```python
class Reader(Dataset):
    def __init__(self, 
                train: Optional[bool] = True, 
                transform: Optional[Callable] = None, 
                len: Optional[int] = 2000
            ):

        self.dataset = datasets.MNIST(
            root='readers/MNISTImageDataset/data',
            train=train,
            download=True,
            transform=transform
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tensor:
        image, _ = self.dataset[idx]
        return image
```

## 3. Crear un evaluador

Para definir un nuevo evaluador, es necesario crear un subdirectorio dentro del módulo `playground/evaluators/`, utilizando la convención de nombres `CamelCase`, al igual que en el caso de las arquitecturas y los readers. Dentro de este subdirectorio se implementará la clase del evaluador, la cual debe ajustarse a la interfaz `Evaluator`, definida en el módulo `core/`. Esto garantiza una integración uniforme con el sistema de ejecución de experimentos y permite su uso dentro del entorno de evaluación.

La clase del evaluador debe implementar todos los métodos definidos en la interfaz `Evaluator`, incluyendo la lógica específica de evaluación, el manejo de resultados y cualquier métrica que se desee calcular. Esta estructura asegura que cualquier evaluador, independientemente del tipo de datos o del modelo utilizado, sea compatible con el resto del framework.

Además del código fuente del evaluador, se deben definir dos archivos de plantilla: <nombre del evaluador>.template y <nombre del evaluador>.call.template. El primero describe cómo se instancia el evaluador dentro del flujo de un experimento, y el segundo especifica cómo debe llamarse o ejecutarse durante la evaluación. Estas plantillas permiten al sistema generar dinámicamente el código necesario para incorporar el evaluador, manteniendo la flexibilidad y automatización del pipeline experimental.

Por ejemplo, en el caso de una evaluación basada en FID para imágenes, el archivo `ImageFIDEvaluator.template` podría contener:

```python
def evaluate_image_fid(image_model, loader, device, results_dir):
    print("\nEvaluating Image FID...")
    evaluator = ImageFIDEvaluator(
        model=image_model,
        dataset=loader,
        device=device,
        image_size=(56, 56)
    )
    fid_rec, fid_gen = evaluator.evaluate()
    print(f"FID Score for Reconstructed Images (Image Reconstruction): {fid_rec:.2f}")
    print(f"FID Score for Generated Images (Image Reconstruction): {fid_gen:.2f}")
    save_metrics({
        "FID_reconstructed": fid_rec,
        "FID_generated": fid_gen
    }, results_dir, "eval_image_fid.json")
```


Mientras que el archivo `ImageFIDEvaluator.call.template` incluiría simplemente la llamada correspondiente:

```python
evaluate_image_fid(image_model, loader, device, args["results_dir"])
```

Estas plantillas deben colocarse en el mismo subdirectorio del evaluador, y sus nombres deben coincidir con el identificador del evaluador, para que puedan ser referenciadas correctamente desde la configuración del experimento.

## 4. Crear un experimento

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

## 5. Ejecutar un experimento

Una vez creado el experimento, se puede acceder al directorio del mismo y ejecutarlo:

```bash
cd experiments/example
chmod +x execute.sh && execute.sh
```


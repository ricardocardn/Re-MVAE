<h1 align="center">Re-MVAE:<br>Recurrent Multimodal VAE</h1>

<br>
<br>

<p align="center"><img src="resources/remvae.png" width="80%"></p>

<br>

<div align="justify">

Este repositorio forma parte del Trabajo de Fin de Grado titulado: **Estudio acerca del entrenamiento con alineamiento de representaciones multimodales en el espacio latente**. El objetivo principal es lograr un alineamiento multimodal efectivo basado en Variational Autoencoders (VAE), con el fin de construir espacios latentes estructurados y manejables que permitan la generación cruzada entre modalidades (por ejemplo, generar texto a partir de imágenes y viceversa).

Este trabajo extiende la idea tradicional de los MVAE (Multimodal VAE) incorporando modelos recurrentes para la modalidad de texto, utilizando RNNs (LSTM, GRU, XLSTM), con el propósito de capturar la secuencialidad inherente a la información textual, mejorando así la calidad y coherencia de las representaciones latentes.

## 1. Objetivo principal

Podríamos decir que este trabajo se propone con la idea de abarcar los siguientes objetivos principales:

- Construir espacios latentes compartidos entre diferentes modalidades que estén alineados, de modo que representaciones de la misma información, pero en diferentes formatos (imagen, texto), sean cercanas en el espacio latente.
- Utilizar esta estructura latente para generación cruzada:
    - Reconstrucción multimodal
    - Traducción de modalidades (ej. imagen → texto)
    - Transferencia de estilos o contenido entre modalidades
- Explorar arquitecturas recurrentes para la parte textual, aprovechando la capacidad de las RNN para modelar dependencias temporales.

## 2. Estructura general del proyecto y módulos principales

El proyecto está diseñado para ser modular, flexible y escalable. A continuación se describen los módulos principales y su utilidad:
- **core/**
Contiene la implementación base del framework:
  - wrapper.py: Integra y coordina las diferentes modalidades dentro del VAE multimodal.
  - vae.py: Define el modelo base del Variational Autoencoder, incluyendo funciones de codificación y decodificación.
  - trainer.py: Módulo responsable del ciclo de entrenamiento, manejo de pérdidas, optimización y métricas.
    
- **architectures/**
Implementa las arquitecturas específicas para cada modalidad y variantes del modelo:
  - Encoders y decoders para imágenes (CNNs, autoencoders convolucionales).
  - Modelos recurrentes para texto (LSTM, GRU, XLSTM).
  - Wrappers y builders para combinar los módulos en modelos completos.
    
- **experiments/**
Carpeta que contiene scripts para ejecutar distintos experimentos: entrenamiento, evaluación y visualización para diferentes datasets y configuraciones de arquitectura.
Se pueden añadir nuevos experimentos simplemente creando nuevas carpetas y scripts que sigan la estructura establecida (train.py, eval.py, visualize.py).
Esto permite probar nuevas combinaciones de modelos y datasets sin alterar la estructura base.
En este directorio se encuentran, además, una serie de notebooks con los principales experimentos realizados, para contrastar los resultados sin necesidad de ejecutar nuevamente los mismos.

- **trainers/**
Implementa diferentes estrategias y variantes de entrenamiento, incluyendo técnicas avanzadas como annealing o entrenamiento adaptativo para mejorar la convergencia y el alineamiento de representaciones.

- **evaluators/**
Provee métricas y herramientas para evaluar modelos: calidad de reconstrucción, métricas específicas para imágenes (FID), métricas para texto (perplejidad), y evaluadores combinados para espacios multimodales.

- **readers/**
Módulos dedicados a cargar y preprocesar datasets multimodales, permitiendo su integración en el pipeline de entrenamiento.
Cada dataset está encapsulado en su propio lector, lo que facilita añadir nuevos conjuntos de datos. Para crear un nuevo dataset:
	1.	Crear un nuevo módulo en readers/ con la lógica para cargar y transformar los datos en el formato esperado (por ejemplo, normalización, tokenización).
	2.	Asegurarse de que el lector devuelva objetos compatibles con el sistema de entrenamiento (por ejemplo, tensores o batches multimodales).
	3.	Registrar el nuevo dataset para su uso en experimentos.

- **utils/**
Contiene funciones auxiliares para procesamiento de texto (tokenización, embeddings preentrenados como GloVe), manejo de arquitectura XLSTM y otros componentes generales.

- **tests/**
Pruebas unitarias para verificar la correcta funcionalidad de los módulos, garantizando robustez y facilidad para futuras modificaciones.

## 3. Añadir nuevos datasets

Para integrar un nuevo dataset multimodal, deberá crearse un módulo lector en la carpeta readers/, el cual contenga los datos del dataset en un directorio `data` y una clase `Reader` para cargarlos en el experimento. Esto permite adaptar el framework a nuevas fuentes de datos, facilitando la exploración en diferentes dominios o tipos de información multimodal.

## 4. Definición experimentos

Para crear un nuevo experimento, han de seguirse los siguientes pasos:
	1.	Crear una nueva carpeta dentro de experiments/ con un nombre representativo del experimento, por ejemplo, xlstm_custom_dataset.
	2.	Añadir scripts básicos:
   - train.py: Define el pipeline de entrenamiento, carga de datos y modelo.
   - eval.py: Implementa la evaluación del modelo entrenado.
   - visualize.py: Scripts para visualizar resultados, reconstrucciones o generación cruzada.
	3.	Configurar el uso de datasets y arquitecturas que se deseen probar.
	4.	Ejecutar los scripts para entrenar, evaluar y visualizar.

Esta modularidad facilita experimentar con nuevas arquitecturas, datasets o configuraciones sin alterar el núcleo del framework.

</div>

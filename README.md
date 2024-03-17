
# Laboratorio 7 - Inteligencia Artificial
## CC3045 - Semestre I - 2024

### Descripción
Este laboratorio se enfoca en la aplicación del aprendizaje por diferencia temporal (Temporal Difference Learning) en el juego Connect Four. Se requiere modificar el código del laboratorio anterior para implementar un agente que utilice TD learning y comparar su rendimiento con agentes basados en Minimax y Minimax con poda alpha-beta.

### Tareas
- **Task 1 - Teoría:**
  - Responder de forma clara y concisa las 5 preguntas teóricas relacionadas con TD learning, juegos simultáneos y equilibrio de Nash.
  - Las respuestas pueden ser subidas en un PDF o dentro del mismo Jupyter Notebook.

- **Task 2 - Connect Four:**
  - Hacer una copia del laboratorio anterior y modificarlo para implementar un agente que use TD learning.
  - Definir la representación del estado, el espacio de acción, implementar el algoritmo de aprendizaje TD, la función de actualización de valor, definir recompensas, implementar una estrategia de exploración y realizar un ciclo de entrenamiento.
  - Opcionalmente, se puede utilizar un enfoque de Machine Learning para aproximar la función de valor.
  - Hacer que el agente entrenado con TD learning juegue contra el agente que usa Minimax y luego contra el agente de Minimax con poda alpha-beta. Realizar al menos 50 juegos de cada caso (150 juegos en total).
  - Graficar la cantidad de victorias de cada uno de los agentes y colocarlas en un documento PDF.
  - Grabar un video mostrando solamente 3 juegos (uno de cada caso) y mencionar qué hace el agente entrenado con TD learning a nivel general y explicar por qué ganó más veces el agente que ganó.

### Entregables
1. Link al repositorio de los integrantes del grupo.
   - Subir también el código a Canvas por temas de Acreditación.
2. Link al video solicitado en las instrucciones.

### Evaluación
- [1.25 pts.] Task 1 (0.25 cada pregunta)
- [2.25 pts.] Task 2 - Agente
- [0.75 pts.] Task 2 - Gráficas
- [0.75 pts.] Task 2 - Video
Total: 5 pts.

## Requisitos Previos (Usamos CONDA !! :D)

- Anaconda o Miniconda instalado en su sistema. Si no tiene Anaconda o Miniconda, puede descargarlo desde [aquí](https://www.anaconda.com/products/individual) o [aquí](https://docs.conda.io/en/latest/miniconda.html), respectivamente.

## Configuración del Entorno

Siga estos pasos para configurar el entorno Conda necesario para ejecutar el notebook:

1. **Clonar el Repositorio**

Primero, clone el repositorio a su máquina local usando Git:

```bash
git clone https://github.com/AndresQuinto5/IA_LAB5.git
cd repositorio
```

2. **Crear el Entorno Conda**

Cree un entorno Conda utilizando el archivo `environment.yml` incluido en el repositorio. Esto instalará todas las dependencias necesarias.

```bash
conda env create -f environment.yml
```

3. **Activar el Entorno**

Una vez creado el entorno, actívelo con:

```bash
conda activate nombre_del_entorno
```

Reemplace `nombre_del_entorno` con el nombre del entorno especificado en el archivo `environment.yml`.

## Ejecutar el Jupyter Notebook

Con el entorno activado, inicie Jupyter Notebook o JupyterLab o abrir en VisualStudio Code:

```bash
jupyter notebook
# o
jupyter lab
```

Navegue hasta el notebook que desea ejecutar y ábralo. Ahora debería ser capaz de ejecutar todas las celdas sin problemas.

### VIDEO:


### Autores
- [Andres Quinto - 18288](https://github.com/AndresQuinto5)
- [Marlon Hernández - 15177](https://github.com/ivanhez)

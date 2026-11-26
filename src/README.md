# Person Re-Identification (Video-based) - Feature Extractor

Este repositorio contiene el módulo de extracción de características visuales (apariencia) para la re-identificación de personas (Re-ID) basado en video. Implementa arquitecturas de vanguardia basadas en **Vision Transformers (ViT)** y **CNNs**, diseñadas para generar representaciones vectoriales (embeddings) altamente discriminativas a partir de secuencias de imágenes.

---

## 1. Fundamentos Científicos y Justificación

La re-identificación de personas en video presenta retos significativos como oclusiones parciales, cambios de iluminación, variaciones de pose y ruido en la detección. Este módulo aborda estos problemas mediante las siguientes estrategias:

### 1.1. Comparativa: Vision Transformer (ViT) vs. CNN (ResNet)
En este proyecto, se incluyen ambos enfoques para permitir un análisis comparativo robusto:

*   **Vision Transformer (ViT):** 
    *   **Campo Receptivo Global:** A diferencia de las CNNs, que procesan información localmente mediante convoluciones, el mecanismo de *self-attention* de ViT permite que cada parche de la imagen interactúe con todos los demás desde las primeras capas. Esto es crucial para capturar dependencias de largo alcance (ej. relación entre la textura del calzado y la prenda superior).
    *   **Robustez a Oclusiones:** Al no depender exclusivamente de estructuras espaciales rígidas, ViT puede "enfocarse" en partes visibles del cuerpo incluso cuando otras están ocluidas, manteniendo una representación global coherente.
*   **CNN (ResNet):**
    *   **Sesgo Inductivo Local:** Las CNNs son excepcionales extrayendo patrones locales de textura y bordes. Son más eficientes en términos de cantidad de datos requeridos para converger, lo que sirve como una línea base (*baseline*) sólida y eficiente.

### 1.2. Agregación Temporal y Espacio de Características
*   **Sequence Averaging:** Se utiliza el promedio de los embeddings de cada frame para representar la secuencia. Científicamente, esto actúa como un filtro de paso bajo que reduce el impacto de frames con ruido o poses atípicas, estabilizando la identidad del individuo a lo largo del tiempo.
*   **Normalización L2:** Al proyectar los embeddings en una hiperesfera unitaria ($||e||_2 = 1$), la similitud de coseno se vuelve equivalente a la distancia euclidiana, lo que mejora la estabilidad del ranking en la fase de inferencia.

---

## 2. Descripción Detallada de la Arquitectura

### 2.1. Componentes de la Red (ViTEmbedder)
La implementación sigue las mejores prácticas de Re-ID (ej. *Strong Baseline for Person Re-ID*):

1.  **Backbone:** Vision Transformer (ej. `vit_base_patch16_224`) pre-entrenado en ImageNet. Extrae un vector de características globales a partir del token [CLS].
2.  **BNNeck (Batch Normalization Neck):** Se introduce una capa de Batch Normalization después de la extracción de características y antes del clasificador. 
    *   *Justificación:* Esto desacopla el espacio de características para la **Identity Loss** (que se calcula después de BN) y la **Triplet Loss** (que se calcula antes de BN), facilitando la convergencia simultánea de ambas pérdidas.
3.  **Global Pooling:** En el caso de CNNs, se utiliza `AdaptiveAvgPool2d` para reducir las dimensiones espaciales a un vector global.

---

## 3. Estrategia de Entrenamiento y Pérdidas

### 3.1. Funciones de Pérdida (Loss Functions)
Se emplea un esquema de pérdida combinada ($L_{total} = L_{ID} + L_{Triplet}$):

*   **Cross-Entropy con Label Smoothing ($L_{ID}$):** 
    *   *Definición:* $y_{ls} = (1 - \epsilon)y + \epsilon / N$.
    *   *Propósito:* Evita que el modelo se vuelva excesivamente confiado en las etiquetas de entrenamiento (identidades específicas), promoviendo una mejor generalización hacia identidades no vistas en el conjunto de test.
*   **Batch-Hard Triplet Loss ($L_{Triplet}$):**
    *   *Definición:* Se seleccionan dinámicamente el positivo más lejano (*hardest positive*) y el negativo más cercano (*hardest negative*) dentro de cada batch.
    *   *Propósito:* Fuerza al modelo a compactar los clusters de la misma identidad y a separar las identidades similares en el espacio de embeddings.

### 3.2. Muestreo PxK (Balanced Sampling)
Por cada iteración, el DataLoader selecciona $P$ identidades y $K$ secuencias por identidad. Esto es fundamental para que la *Triplet Loss* tenga suficientes pares comparativos válidos dentro de un mismo batch.

---

## 4. Estructura del Repositorio

```text
src/
├── configs/                # Hiperparámetros y configuración experimental
├── model/                  
│   ├── vit_embedder.py     # Implementación de la arquitectura Transformer
│   ├── cnn_embedder.py     # Implementación de la arquitectura ResNet
│   ├── losses.py           # Definiciones científicas de CE-LS y Triplet Loss
│   └── temporal_pooling.py # Lógica de agregación por secuencia
├── preprocess/             
│   ├── dataset.py          # Manejo de secuencias y estructuras de datos
│   └── my_dataset.py       # Adaptador para datasets personalizados
├── train_vit.py            # Pipeline de entrenamiento para ViT
├── train_cnn.py            # Pipeline de entrenamiento para CNN
├── evaluate_categorized.py # Evaluación analítica por atributos
├── visual_search.py        # Herramienta de validación cualitativa
└── visualize_embeddings.py # Análisis de distribución (t-SNE)
```

---

## 5. Glosario de Parámetros de Configuración

| Parámetro | Definición Científica | Impacto |
| :--- | :--- | :--- |
| `image_size` | Resolución de entrada [H, W]. | Afecta la capacidad de capturar detalles finos (ej. logotipos). |
| `sequence_len` | Número de frames por tracklet. | Mayor longitud = representación más estable pero mayor costo computacional. |
| `P` | Número de identidades en el batch. | Determina la diversidad de "clases" para la Triplet Loss. |
| `K` | Muestras por identidad en el batch. | Crucial para encontrar el *hardest positive* en la Triplet Loss. |
| `label_smoothing` | Factor de suavizado $\epsilon$. | Controla la regularización del clasificador. |
| `margin` | Margen $\alpha$ en Triplet Loss. | Define la distancia mínima de separación entre clusters de identidades. |

---

## 6. Ejecución y Evaluación

### Entrenamiento
```bash
python train_vit.py --config configs/vit_config.yaml
```

### Evaluación Categorizada
Este script es fundamental para la tesis, ya que desglosa el mAP y Rank-1 bajo condiciones de:
- **Iluminación:** Normal vs. Baja iluminación.
- **Oclusión:** Escenas despejadas vs. Escenas con obstáculos.

```bash
python evaluate_categorized.py --config configs/vit_config.yaml --checkpoint path/to/model.pth
```

### Análisis Cualitativo
Utiliza `visualize_embeddings.py` para generar mapas t-SNE que demuestren visualmente cómo el modelo agrupa las identidades en el espacio latente.

---

## 7. Validación Cruzada Open-Set 5-Fold (Pares Ambiguos en Prueba)

Para evaluar a las 72 identidades multicámara fuera de entrenamiento (*out-of-fold*, garantizando pares de alta similitud interclase compitiendo directamente en `test`):

```bash
# 1. Generar los 5 pliegues (semilla maestra 42; ajustar --images-root en Linux)
python src/preprocess/prepare_sgreid_5fold.py --images-root /ruta/al/dataset/result1/dataset/output

# 2. Ejecutar los 5 pliegues para CNN, ViT-Base y ViT-Small (reanudable)
python run_oe2_5fold.py --data-root /ruta/al/dataset/result1/dataset/output

# (Opcional) Incluir control ImageNet-1k para ViT-Small
python run_oe2_5fold.py --data-root /ruta/al/dataset/result1/dataset/output --include-in1k-control
```

La agregación estadística con media ± desviación estándar y pruebas pareadas sobre las 72 identidades independientes se genera en `outputs/oe2_sgreid_5fold/summary_5fold.json` mediante `src/aggregate_5fold.py`.

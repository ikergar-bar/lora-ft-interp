# LoRA Sentiment Interpretability — Project Spec

> **Una frase:** afinar un modelo pequeño con LoRA para clasificar sentimiento y diseccionar *dónde* y *cómo* vive esa nueva capacidad, usando el delta de LoRA (ΔW = BA) como objeto central de análisis.

Este documento es la fuente de verdad del proyecto. Está escrito para ser usado por agentes de código (p. ej. Claude Code): cada hito tiene criterios de aceptación verificables y hay una sección explícita de *fuera de alcance* para evitar scope creep. Un agente debe leer este archivo entero antes de actuar y no introducir dependencias ni cambios de diseño que lo contradigan sin dejarlo anotado.

---

## 1. Objetivo y pregunta de investigación

**Objetivo:** demostrar, end-to-end, el ciclo de adaptación de un LLM pequeño (fine-tuning con LoRA) y, sobre ese mismo artefacto, hacer un análisis de interpretabilidad que explique qué cambió internamente al adaptarlo.

**Pregunta de investigación (research question):**
> ¿El fine-tuning con LoRA para clasificación de sentimiento **crea** una dirección lineal de sentimiento, la **afila**, o la **reubica** respecto al modelo base? ¿En qué capas/módulos se concentra ese cambio y coincide la localización del delta con la localización causal del cómputo de sentimiento?

**Anclaje en la literatura:** punto de comparación principal Tigges et al. (2023), *Linear Representations of Sentiment in Large Language Models* (existencia de una dirección lineal de sentimiento y comportamiento de "resumen" en ciertos tokens). El proyecto contrasta los hallazgos del base con los del modelo afinado.

---

## 2. Restricciones de hardware y entorno (PRIMERA CLASE)

- **GPU:** 1× NVIDIA con **4 GB de VRAM**. Es una restricción dura. Todo el diseño está dimensionado para caber en 4 GB.
- **Implicaciones que YA están resueltas por el diseño:**
  - Modelo pequeño (GPT-2 small, 124M) → base en fp16 ~250 MB.
  - LoRA: el base va **congelado**, solo se entrena el adapter (pocos MB). El optimizador (AdamW) solo guarda estados de los parámetros del adapter.
  - SST-2 son frases cortas → `max_seq_len` bajo (64–128).
- **NO usar QLoRA / cuantización 4-bit del base.** No hace falta para caber en 4 GB y **contaminaría el análisis de interpretabilidad** (queremos un base numéricamente limpio para comparar contra el afinado). Esta es una decisión de diseño, no una preferencia.
- **Precisión:** preferir `bf16` si la GPU lo soporta (Ampere+). Una tarjeta de 4 GB suele ser Turing/Pascal → en ese caso usar `fp16`. El agente debe detectar la capacidad y elegir, dejándolo logueado.
- **Válvulas de memoria si algo no cupiera:** reducir batch size y compensar con `gradient_accumulation_steps`; activar `gradient_checkpointing`; bajar `max_seq_len`. En el peor caso, ejecutar el análisis (inferencia/caching) en CPU: GPT-2 small en CPU es tolerable para los tamaños de este proyecto.
- **Caching de activaciones (interpretabilidad):** en TransformerLens, `run_with_cache` cachea todo por defecto. En 4 GB hay que **limitar con `names_filter`** para cachear solo lo necesario, usar batches pequeños (8–32) y mover el cache a CPU cuando se acumule.

---

## 3. Stack tecnológico

| Componente | Elección | Motivo |
|---|---|---|
| Modelo | **GPT-2 small (124M)** | El modelo más estudiado en interpretabilidad; muchos puntos de referencia. Alternativa válida: **Pythia-160M** (checkpoints limpios). |
| Dataset | **SST-2** (GLUE) | Frases cortas, etiqueta binaria, ideal para modelo pequeño y para atribución a tokens de etiqueta. |
| Fine-tuning | **PEFT (LoRA)** sobre HuggingFace `transformers` | Estándar, adapter pequeño, base congelado. |
| Interpretabilidad | **TransformerLens** | Soporta GPT-2/Pythia de fábrica; hooks, `run_with_cache`, activation patching. |
| Probing / reducción dim. | scikit-learn + **umap-learn** | Probing lineal por capa + visualización del espacio de representaciones. |
| Tracking (opcional) | Weights & Biases o CSV local | Loguear configs y métricas para reproducibilidad. |

**Aviso de compatibilidad (guardrail importante):** TransformerLens es sensible a la versión de `transformers`. Al hacer setup, el agente debe fijar un conjunto de versiones compatible entre `transformers`, `peft` y `transformer-lens`, verificar que un `HookedTransformer.from_pretrained("gpt2")` carga sin error, y **pinear las versiones** en `requirements.txt`/`pyproject`. No actualizar dependencias a ciegas a mitad de proyecto.

> Nota: las versiones exactas deben verificarse en el momento del setup (este spec no las fija para no quedar obsoleto). Pinear lo que funcione.

---

## 4. Decisiones de diseño fijas

1. **Framing generativo, no cabeza de clasificación.** Plantear como `"Review: {texto} Sentiment:"` y predecir el token `positive`/`negative`. Esto habilita atribución logit a los tokens de etiqueta, patching y análisis de direcciones sobre el residual stream (la chicha interpretativa). Una cabeza de clasificación lo dificultaría.
2. **El delta de LoRA es el objeto de análisis central.** ΔW = BA es de rango bajo y manipulable: SVD, normas por capa, proyección de activaciones sobre sus direcciones.
3. **Base limpio.** Sin cuantización del base (ver §2).
4. **Reproducibilidad.** Semillas fijas, configs versionadas, resultados regenerables desde scripts/notebooks.

---

## 5. Estructura del repositorio

```
lora-sentiment-interp/
├── README.md                 # Resumen, hallazgo principal, cómo reproducir
├── AGENTS.md                 # Reglas de trabajo para agentes (puede referenciar este spec)
├── pyproject.toml / requirements.txt   # Dependencias PINEADAS
├── configs/
│   └── train.yaml            # Hiperparámetros (modelo, LoRA rank/alpha, batch, lr, seq_len)
├── data/                     # SST-2 cacheado (no versionar el dataset crudo)
├── src/
│   ├── data.py               # Carga/format de SST-2 al framing generativo
│   ├── train_lora.py         # Bucle de fine-tuning con PEFT
│   ├── eval.py               # Eval generativo: accuracy base vs afinado
│   ├── merge.py              # merge_and_unload + porteo a TransformerLens + check de paridad
│   ├── delta_analysis.py     # Normas por capa + SVD del ΔW
│   ├── directions.py         # Dirección lineal de sentimiento (probing) base vs afinado
│   ├── representations.py    # Probing por capa + UMAP de hidden states
│   └── patching.py           # (stretch) activation patching
├── notebooks/                # Exploración y figuras finales
├── artifacts/                # Adapters LoRA, modelos fusionados, figuras
└── results/                  # Métricas, tablas, plots exportados
```

---

## 6. Setup del entorno

```bash
# Entorno
python -m venv .venv && source .venv/bin/activate   # o conda/uv
pip install --upgrade pip

# Dependencias núcleo (PINEAR versiones tras verificar compatibilidad)
pip install torch --index-url <wheel CUDA acorde a la tarjeta>
pip install transformers peft datasets accelerate
pip install transformer-lens
pip install scikit-learn umap-learn matplotlib

# Verificación mínima de entorno (el agente DEBE pasar esto antes de seguir):
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "from transformer_lens import HookedTransformer; HookedTransformer.from_pretrained('gpt2'); print('TL OK')"
```

El agente debe ajustar la wheel de torch a la versión de CUDA del sistema y confirmar que la GPU se detecta antes de cualquier entrenamiento.

---

## 7. Plan de trabajo por hitos (8 semanas × 10 h)

Cada hito tiene un **criterio de aceptación** (DoD parcial) que el agente debe poder verificar. No avanzar al siguiente sin cumplirlo.

### M0 — Setup + baseline (semana 1)
- Cargar GPT-2 small y SST-2, implementar el framing generativo, montar el eval, dejar TransformerLens cargando el modelo.
- **Aceptación:** `eval.py` reporta accuracy del **base** sobre el split de validación (se espera floja); `HookedTransformer.from_pretrained("gpt2")` funciona; entorno verificado en GPU.

### M1 — Fine-tuning LoRA (semana 2)
- Bucle de entrenamiento con PEFT; entrenar; guardar adapter en `artifacts/`.
- **Aceptación:** accuracy del afinado **claramente > base**; entrenamiento cabe en 4 GB (loguear pico de VRAM); config reproducible en `configs/train.yaml`.

### M2 — Merge + porteo + delta (semana 3)
- `merge_and_unload()` del adapter sobre el base, portar el modelo fusionado a TransformerLens, **verificar paridad numérica** (logits del modelo HF fusionado ≈ logits en TransformerLens, dentro de tolerancia). Calcular normas del ΔW por capa y SVD.
- **Aceptación:** check de paridad pasa (diferencia máxima de logits por debajo de un umbral documentado); existe una tabla/plot de norma del delta por capa y un análisis de rango efectivo (SVD).

### M3 — Direcciones (semana 4)
- Estimar la dirección lineal de sentimiento (probing sobre el residual stream) en base y afinado; proyectar activaciones sobre ella.
- **Aceptación:** existe una dirección de sentimiento por modelo con métrica de separabilidad; comparación cuantitativa base vs afinado (¿crea/afila/reubica?).

### M4 — Representaciones (semana 5)
- Probing lineal por capa (accuracy de sondeo por capa) + UMAP de hidden states base vs afinado, coloreado por etiqueta.
- **Aceptación:** curva de probing por capa para ambos modelos; figuras UMAP comparativas exportadas a `results/`.

### M5 — Causalidad / stretch (semana 6, OPCIONAL por diseño)
- Activation patching: parchear activaciones del afinado en el base para localizar causalmente el cómputo de sentimiento; cruzar con la localización del delta (M2).
- **Aceptación:** mapa de efecto del patching por capa/posición; comparación con dónde el delta era mayor. (Si no se llega, el proyecto se sostiene sin esto.)

### M6 — Consolidar (semana 7)
- Sanity checks, una ablación (p. ej. variar el rank de LoRA y ver efecto en delta/direcciones), figuras finales limpias.
- **Aceptación:** al menos una ablación documentada; todas las figuras regenerables desde scripts/notebooks.

### M7 — Write-up + repo (semana 8, NO es relleno)
- README con el hallazgo principal y cómo reproducir; notebook limpio; post/informe corto contando la historia (pregunta → método → resultado).
- **Aceptación:** un tercero puede clonar y reproducir las figuras principales siguiendo el README; existe un post/informe breve con la conclusión.

---

## 8. Guardrails técnicos críticos

- **Paridad tras el merge (M2):** es el fallo silencioso más típico. Verificar SIEMPRE que el modelo fusionado en TransformerLens produce los mismos logits que el HF fusionado antes de sacar conclusiones interpretativas. Sin paridad, todo el análisis es ruido.
- **Memoria en 4 GB:** loguear el pico de VRAM en entrenamiento y en caching. Usar `names_filter` en `run_with_cache`. Batches de análisis pequeños. Mover cache a CPU cuando proceda.
- **Compatibilidad de versiones:** `transformers` ↔ `transformer-lens` es frágil. Pinear y no tocar.
- **Sin cuantización del base.** (Ver §2 y §4.)
- **Determinismo:** semilla global fija; loguear todas las configs.

---

## 9. Convenciones para agentes (AGENTS.md)

- Leer este spec entero antes de actuar. No contradecir las decisiones de diseño sin anotarlo explícitamente con justificación.
- Trabajar en **pasos pequeños y verificables**: implementar → ejecutar el criterio de aceptación del hito → continuar.
- **Entrenamiento en scripts** (`src/`), **análisis exploratorio en notebooks**, figuras finales regenerables desde código.
- No añadir dependencias nuevas sin pinearlas y justificar por qué.
- Loguear siempre: config usada, semilla, pico de VRAM, métricas.
- Si un criterio de aceptación no se cumple, **parar y reportar**, no seguir hacia adelante acumulando deuda.
- Preferir resultados pequeños y correctos a pipelines grandes sin verificar.

---

## 10. Definición de "hecho" (entregables)

1. Repo reproducible con adapter LoRA entrenado y modelo fusionado verificado.
2. Eval base vs afinado.
3. Análisis del delta: normas por capa + SVD.
4. Dirección de sentimiento base vs afinado (¿crea/afila/reubica?).
5. Probing por capa + UMAP comparativo.
6. (Stretch) Activation patching cruzado con la localización del delta.
7. README + post/informe corto con el hallazgo.

---

## 11. Fuera de alcance (NO hacer)

- Modelos grandes o cuantización 4-bit del base.
- Cabeza de clasificación (rompe el framing generativo).
- Multilingüe / euskera / español (otro proyecto; el modelo pequeño no lo soporta y desviaría el foco).
- Dataset distinto a SST-2 sin justificación fuerte.
- Sparse autoencoders u otras técnicas pesadas de interpretabilidad mecanicista no contempladas (posible trabajo futuro, no aquí).
- Servir el modelo / app / despliegue. Este proyecto es entrenamiento + análisis, no producto.

---

## 12. Riesgos y válvulas de escape

| Riesgo | Mitigación |
|---|---|
| El entrenamiento se atasca (M1) | Reducir dataset, entrenar menos módulos, bajar seq_len. |
| El merge no da paridad (M2) | Revisar dtype y nombres de capas; no avanzar hasta resolverlo. |
| Falta de tiempo | M5 es opcional por diseño; el proyecto se sostiene con M0–M4 + M7. |
| Choque de versiones | Pinear desde el día 1; no actualizar a mitad. |
| 4 GB se quedan cortos en algún punto | Bajar batch + grad. accumulation; análisis en CPU. |

---

## 13. Referencias

- Tigges et al. (2023), *Linear Representations of Sentiment in Large Language Models*.
- Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models*.
- TransformerLens (documentación de la librería).
- SST-2 / GLUE (Socher et al., 2013; Wang et al., 2018).

> Las versiones de librerías y la disponibilidad de herramientas deben verificarse en el momento del setup. Conviene revisar si hay alternativas más recientes para la parte de interpretabilidad antes de empezar.

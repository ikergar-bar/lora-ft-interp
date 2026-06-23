---
name: check-milestone
description: Verifica el criterio de aceptación (DoD parcial) de un hito del proyecto antes de avanzar al siguiente. Úsalo al cerrar cualquier hito M0–M7. Toma el hito (p. ej. "M2"), localiza su criterio en el spec, ejecuta el script correspondiente de src/ y reporta PASS/FAIL. Si falla, para y reporta — no avanzar acumulando deuda.
---

# check-milestone — Gate de aceptación por hito

Fuente de verdad: `lora-sentiment-interpretability-spec.md` §7 (criterios) y §9 (protocolo). Regla del proyecto: **no avanzar al siguiente hito sin cumplir el criterio de aceptación del actual.**

## Uso

Argumento: el hito a verificar, p. ej. `M1`.

## Procedimiento

1. **Localizar el criterio de aceptación** del hito en §7 del spec. Resumen:

   | Hito | Script(s) | Criterio de aceptación |
   |------|-----------|------------------------|
   | M0 | `eval.py`, setup | Accuracy del **base** en validación reportada; `HookedTransformer.from_pretrained("gpt2")` OK; entorno verificado en GPU. → ver skill `verify-env`. |
   | M1 | `train_lora.py`, `eval.py` | Accuracy del afinado **claramente > base**; entrenamiento cabe en 4 GB (loguear pico VRAM); config en `configs/train.yaml`. |
   | M2 | `merge.py`, `delta_analysis.py` | **Paridad** pasa (→ skill `check-parity`); tabla/plot de norma del delta por capa + análisis de rango (SVD). |
   | M3 | `directions.py` | Una dirección de sentimiento por modelo con métrica de separabilidad; comparación cuantitativa base vs afinado. |
   | M4 | `representations.py` | Curva de probing por capa para ambos modelos; figuras UMAP comparativas en `results/`. |
   | M5 (opcional) | `patching.py` | Mapa de efecto del patching por capa/posición; cruce con dónde el delta era mayor. |
   | M6 | (varios) | Al menos una ablación documentada; todas las figuras regenerables desde scripts/notebooks. |
   | M7 | `README.md` | Un tercero puede clonar y reproducir las figuras principales; informe breve con la conclusión. |

2. **Ejecutar el/los script(s)** del hito y capturar la salida (métricas, pico de VRAM, figuras).

3. **Comparar contra el criterio** y reportar PASS/FAIL con la evidencia concreta (números, rutas de figuras en `results/`).

## Criterio PASS

El criterio de aceptación del hito en §7 se cumple, con evidencia registrada (métrica/figura/log), no solo afirmado.

## Al cerrar un hito (PASS) — actualizar STATUS.md

Una vez el criterio pasa, actualizar `STATUS.md`:
1. Marcar el hito como ✅ PASS en la tabla y añadir el resultado clave (accuracy, umbral de paridad, etc.).
2. Añadir una sección `## Mx — Resultados y hallazgos` con los números y rutas de artefactos relevantes.
3. Registrar cualquier quirk o decisión de implementación que no esté en el spec (compatibilidades, cambios de API, etc.).
4. Actualizar la sección **Próximos pasos** para apuntar al siguiente hito.
5. Actualizar el campo **Milestone actual** al siguiente hito pendiente.

## Si FALLA

**Parar y reportar.** No avanzar al siguiente hito. Consultar §12 del spec (riesgos y válvulas de escape) para el hito en cuestión antes de reintentar.

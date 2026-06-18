---
name: check-parity
description: Verifica la paridad numérica tras fusionar el adapter LoRA (hito M2). Úsalo justo después de merge_and_unload y portar el modelo a TransformerLens, y SIEMPRE antes de sacar cualquier conclusión de interpretabilidad. Compara los logits del modelo HF fusionado con los del modelo porteado a TransformerLens dentro de una tolerancia documentada. Es el guardrail más crítico del proyecto.
---

# check-parity — Verificación de paridad tras el merge

Fuente de verdad: `lora-sentiment-interpretability-spec.md` §8 (guardrails) y M2. El spec lo llama **el fallo silencioso más típico**: sin paridad, todo el análisis posterior es ruido.

## Procedimiento

1. **Cargar ambos modelos** (vía `src/merge.py`):
   - Modelo HF fusionado: base + adapter con `merge_and_unload()`.
   - Modelo porteado a TransformerLens (`HookedTransformer`) desde el modelo fusionado.
   - Asegurar **mismo dtype** en ambos para la comparación (un mismatch fp16/fp32 es causa típica de falsa discrepancia).

2. **Comparar logits sobre un batch fijo** (semilla fija, prompts del framing generativo `"Review: {text} Sentiment:"`):
   - Forward de los mismos inputs por ambos modelos.
   - Calcular el diff máximo absoluto de logits: `max_abs_diff = (logits_hf - logits_tl).abs().max()`.

3. **Comparar contra tolerancia documentada.**
   - Documentar el umbral usado (p. ej. `atol` razonable para el dtype elegido) en `results/` junto al `max_abs_diff` obtenido.
   - Reportar PASS/FAIL.

## Criterio PASS

`max_abs_diff` por debajo del umbral documentado. Registrar el número y el umbral en `results/` (tabla o log), no solo "pasó".

## Si FALLA

**Parar. No avanzar a M3+.** Causas habituales y orden de revisión:
1. Mismatch de **dtype** entre HF y TransformerLens.
2. Mapeo de **nombres de capas** incorrecto al portear (LayerNorm folding, pesos no transferidos).
3. Procesamiento de tokenización/posición distinto entre ambos.

Resolver la paridad antes de cualquier análisis de delta, direcciones o patching.

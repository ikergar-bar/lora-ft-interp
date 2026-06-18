---
name: verify-env
description: Gate de setup obligatorio antes de cualquier entrenamiento o análisis (hito M0). Úsalo al preparar el entorno, instalar dependencias, o cuando se vaya a empezar a entrenar/cachear por primera vez. Detecta la GPU, elige bf16 vs fp16, verifica que torch y TransformerLens cargan, y pinea versiones. NO entrenar sin pasar esto.
---

# verify-env — Gate de verificación del entorno

Fuente de verdad: `lora-sentiment-interpretability-spec.md` §2 y §6. Este gate es **obligatorio** antes de entrenar o cachear: el spec dice que el agente DEBE pasarlo antes de seguir.

## Procedimiento

1. **Detectar GPU y capability.**
   ```bash
   python -c "import torch; print('cuda:', torch.cuda.is_available()); print('name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('capability:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)"
   ```
   - Si `cuda` es `False`: el análisis puede correr en CPU (GPT-2 small es tolerable), pero **avisa** y confirma con el usuario antes de entrenar en CPU.

2. **Elegir precisión y loguearla.**
   - Capability ≥ (8, 0) → Ampere+ → usar **`bf16`**.
   - Capability < (8, 0) → Turing/Pascal → usar **`fp16`**.
   - Deja la elección logueada y refleja `precision` / `fp16` en `configs/train.yaml`.

3. **Verificar carga de TransformerLens** (compatibilidad `transformers` ↔ `transformer-lens`):
   ```bash
   python -c "from transformer_lens import HookedTransformer; HookedTransformer.from_pretrained('gpt2'); print('TL OK')"
   ```

4. **Pinear versiones.** Una vez ambas verificaciones pasen, capturar versiones exactas y pinearlas en `requirements.txt`:
   ```bash
   python -c "import transformers, peft, transformer_lens, torch; print('transformers', transformers.__version__); print('peft', peft.__version__); print('transformer_lens', transformer_lens.__version__); print('torch', torch.__version__)"
   ```
   No actualizar dependencias a ciegas después (la pareja `transformers`↔`transformer-lens` es frágil).

## Criterio PASS

- `torch.cuda.is_available()` reportado y precisión elegida + logueada.
- `HookedTransformer.from_pretrained("gpt2")` carga sin error.
- Versiones de `transformers`, `peft`, `transformer-lens`, `torch` pineadas en `requirements.txt`.

## Si FALLA

Parar y reportar. No empezar M1 (entrenamiento) hasta que el gate pase. Errores de carga de TransformerLens suelen ser choque de versión con `transformers`: bajar/subir `transformers` a una versión compatible y volver a pinear.

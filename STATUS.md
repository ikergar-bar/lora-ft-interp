# Project Status

> Leer este archivo al inicio de cualquier sesión antes de actuar.
> Actualizar al completar cada hito o al descubrir algo relevante para el siguiente agente.

**Última actualización:** 2026-06-18
**Milestone actual:** M1 — Fine-tuning LoRA (pendiente)
**Milestone anterior completado:** M0 ✓

---

## Estado de hitos

| Hito | Estado | Nota |
|------|--------|------|
| M0 — Setup + baseline | ✅ PASS | accuracy base 61.7% en SST-2 validation |
| M1 — Fine-tuning LoRA | ⏳ pendiente | |
| M2 — Merge + porteo + delta | ⏳ pendiente | |
| M3 — Direcciones | ⏳ pendiente | |
| M4 — Representaciones | ⏳ pendiente | |
| M5 — Patching (stretch) | ⏳ pendiente | opcional por diseño |
| M6 — Consolidar | ⏳ pendiente | |
| M7 — Write-up + repo | ⏳ pendiente | |

---

## M0 — Resultados y hallazgos

- **Accuracy base GPT-2 small (zero-shot, framing generativo):** 61.7% sobre 872 ejemplos (`validation`).
- Resultado en `results/eval_base.json`.
- TransformerLens `HookedTransformer.from_pretrained("gpt2")` carga correctamente.
- Entorno: CPU-only por ahora (torch sin CUDA). El eval en CPU es tolerable para GPT-2 small.

---

## Entorno

- **Python:** 3.x via uv (venv en `.venv/`)
- **torch:** 2.7.1 — CPU-only. Para entrenar (M1), reinstalar con la wheel CUDA:
  `uv pip install torch --index-url https://download.pytorch.org/whl/cu<VER>`
- **Versiones pineadas en `requirements.txt`** (verificadas 2026-06-18).
- **GPU:** no verificada aún (sin CUDA wheel). Pendiente antes de M1.

---

## Decisiones y quirks descubiertos (no están en el spec)

1. **`datasets>=5.0` requiere namespace completo:** usar `"nyu-mll/glue"` en vez de `"glue"`. Afecta a `src/data.py` (ya corregido) y a `src/train_lora.py` cuando cargue el split de entrenamiento.
2. **`transformers 5.x` renombró `torch_dtype` → `dtype`** en `from_pretrained`. Ya corregido en `src/eval.py`.
3. **CPU fuerza `float32`:** `resolve_dtype` (en `src/utils.py`) degrada a fp32 en CPU automáticamente. Con CUDA volverá a fp16.
4. **Windows sin Developer Mode:** HuggingFace Hub no puede crear symlinks en la caché. Los warnings son cosméticos, no afectan al resultado.
5. **`src/utils.py` añadido** (no estaba en el spec): helpers compartidos (`set_seed`, `get_device`, `resolve_dtype`, `load_config`, `log_result`). Justificado para evitar duplicación entre módulos.
6. **PEFT auto-corrige `fan_in_fan_out` para `Conv1D`:** GPT-2 usa `Conv1D` (no `Linear`) en `c_attn`. PEFT detecta esto y setea `fan_in_fan_out=True` automáticamente con un warning. Comportamiento correcto, no requiere acción.
7. **`warmup_ratio` deprecado en transformers 5.x Trainer:** Muestra warning "will be removed in v5.2" pero sigue funcionando. Si deja de funcionar en futuras versiones, calcular `warmup_steps` explícitamente como `int(warmup_ratio * total_steps)` antes de construir `TrainingArguments`.

---

## Próximos pasos (M1)

`src/train_lora.py` ya implementado. Smoke-test en CPU con 64 ejemplos: PASS (adapter guardado en `artifacts/lora_adapter`, pipeline completo funciona).

**Pendiente para cerrar M1 (requiere GPU):**
1. Instalar torch con CUDA wheel: `uv pip install torch --index-url https://download.pytorch.org/whl/cu<VER>`
2. Pasar skill `verify-env` (GPU + fp16 detectado, versiones pineadas).
3. Borrar el adapter del smoke-test: `rm -rf artifacts/lora_adapter`
4. Run completo: `python -m src.train_lora` (67k ejemplos, 3 epochs, fp16). Confirmar pico VRAM < 4 GB.
5. Eval del afinado: `python -m src.eval --adapter artifacts/lora_adapter`. Confirmar accuracy **claramente > 0.617**.
6. Cerrar con skill `check-milestone M1` y actualizar este archivo (tabla + sección M1 + próximos pasos → M2).

# Poker IA - State Encoder

Este módulo construye una representación de estado para una red neuronal
en un entorno de Poker de 6 jugadores.

Características codificadas:
- Cartas propias: one-hot (52)
- Cartas comunitarias: one-hot (52)
 - Cartas comunitarias: separadas por street — `flop` (3 slots), `turn`, `river` (cada slot one-hot 52)
- Features derivados del board: rank histogram, suit counts, paired, monotone, straight potential, flush draw potential, etc.
 - Multiway information:
	 - `player_stacks`: vector con stacks relativos por jugador (normalizado)
	 - `aggressor_mask`: máscara por jugador que indica agresión (bet/raise)
	 - `last_raiser`: one-hot del último que subió
		- `in_hand_mask`: máscara por jugador que indica si está en mano — obligatorio; no se infiere automáticamente
 - `spr`: Stack-to-pot ratio (SPR) normalizado: `eff_stack / max(pot_size, 1)` (configurable `spr_ref`)
- Posición: one-hot (6)
- Stack efectivo relativo: escalar normalizado
- Tamaño del pozo: escalar normalizado
- Ronda: one-hot (preflop/flop/turn/river)
- Historial comprimido: últimas N acciones (one-hot + cantidad normalizada)
- Número de jugadores activos: escalar normalizado

Uso rápido:

1. Instalar dependencias mínimas:

```bash
python -m pip install -r requirements.txt
```

2. Ejecutar ejemplo desde el módulo:

```bash
python /storage/Proyectos/POKER_IA/poker_state.py
```

3. Correr tests:

```bash
pytest -q
```

Parámetros configurables (por `config` en `build_state`):
- `history_N` (int): número de acciones a mantener (por defecto 10)
- `stack_ref`, `pot_ref`, `amount_ref` (float): valores de normalización
- `n_players` (int): número de jugadores (por defecto 6)

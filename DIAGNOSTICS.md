# Diagnóstico: Divergencia del Solver en Casos 3D

Este documento registra el estado de la investigación sobre por qué los solvers fallan
consistentemente en el escenario 3D arterial/árbol microvascular, a pesar de funcionar
correctamente en casos 2D.

---

## Observaciones

### Qué funciona
- `stabilized_schur` (y variantes) converge en 2D: DFG benchmark, lid-driven, pipe_cylinder
- Los escenarios 2D son "razonables" en tamaño de DOFs y geometría

### Qué no funciona
- Todos los solvers divergen en el solver lineal para el escenario 3D arterial/árbol microvascular (`scenario_factory.py`)
- El fallo es consistente: no es intermitente ni dependiente de parámetros específicos
- Probado también con `unit_cube` (geometría simple 3D, sin estenosis) → también falla
  - **Esto descarta la geometría compleja como causa raíz**

---

## Descartados

| Hipótesis | Razón del descarte |
|-----------|-------------------|
| Elementos anisótropos / malla butterfly | Probado con `unit_cube` (elementos regulares) y tampoco converge |
| Problema con el espacio nulo global (presión) | Los configs usan BCs de Dirichlet en presión en la frontera; esto elimina el espacio nulo, no aplica |
| Escala del residuo con tamaño del sistema | Los solvers usan tolerancias relativas (rtol), no absolutas |
| Parámetros físicos incorrectos | El Re estimado es bajo (~0.03), el flujo es laminar, debería ser un caso fácil numéricamente |

---

## Sospechas Activas

### Sospecha 1 (Alta): Formulación inadecuada para 3D
La formulación SUPG/PSPG/LSIC estabilizada con P1-P1 funciona bien en 2D pero podría
tener problemas en 3D que no son evidentes. Posibles causas:
- Los parámetros de estabilización `tau_supg`, `tau_pspg`, `tau_lsic` fueron ajustados/validados
  solo en 2D; su comportamiento en 3D puede ser diferente
- `mesh.h()` sobreestima el diámetro del elemento en 3D al usar la arista más larga
  (para mallas tetraédricas, la arista diagonal de un tetraedro puede ser mucho mayor
  que la escala característica del problema). Debería usarse `CellDiameter` de UFL
  - Archivo: todos los solvers, e.g. `src/solvers/stabilized_schur.py:82-88`
- La formulación de Crank-Nicolson con `u_mid = 0.5*(u_sol + u_prev)` tanto en el
  término temporal como en el convectivo puede introducir inestabilidades adicionales
  en 3D

### Sospecha 3 (Media-Alta): Combinación presión-presión como BCs inadecuada
Los configs actuales usan `bc_type: inlet: pressure, outlet: pressure`, imponiendo
Dirichlet de presión en ambos extremos sin ninguna condición de velocidad en inlet/outlet
(solo no-slip en paredes).

Posibles problemas con esta configuración:
- Con flujo impulsado puramente por diferencia de presión y sin condición de velocidad
  prescrita, el sistema es matemáticamente bien puesto pero puede ser más difícil de
  resolver numéricamente: el solver de Newton arranca desde velocidad cero y debe
  "descubrir" el perfil de velocidad, lo que puede requerir muchas más iteraciones
- La combinación velocidad-presión (inlet: `velocity_parabolic`, outlet: `pressure`)
  es la más estándar en CFD incompresible y la que usan todos los benchmarks 2D que
  funcionan correctamente
- Es posible que el solver/precondicionador converja con velocidad impuesta en inlet
  aunque no lo haga con presión-presión, incluso para el mismo problema físico
- El escenario `unit_cube` tampoco tiene BCs bien definidas para presión-presión
  si no hay perfil de velocidad que ancle el problema

**A probar:** cambiar a `inlet: velocity_parabolic, outlet: pressure` en los configs
y verificar si el solver converge.

---

### Sospecha 2 (Alta): Precondicionadores no escalan a >100k DOFs
El sistema 3D tiene órdenes de magnitud más DOFs que el 2D:
- Malla 3D de estenosis: estimado 1-10M DOFs para velocidad+presión
- `unit_cube` con resolución moderada: >100k DOFs fácilmente

El precondicionador actual en `stabilized_schur` es débil para este tamaño:
- Velocidad: `GMRES + ASM (Additive Schwarz con ILU local)` — no provee corrección global
- Presión: `preonly + ASM` — para el bloque de Schur, ASM es completamente inadecuado
- `SELFP` como aproximación del complemento de Schur es pobre para flujo convectivo
- El número de iteraciones GMRES necesarias crece con el tamaño del problema;
  con un precondicionador débil, el FGMRES externo diverge

Comparación: `stabilized_lsc` usa el precondicionador LSC (Least-Squares Commutator)
para el bloque de presión, que es O(n) y escalable. `stabilized_pcd` usa PCD.
Estos deberían comportarse mejor en 3D grande, pero aún no se han probado sistemáticamente.

---

## Hallazgos Técnicos del Audit del Código

### Problema potencial: `updateSolution` con ghost DOFs en MPI
**Archivo:** `src/solvers/stabilized_schur.py:124-141`

```python
u_size_local = self.u_prev.x.petsc_vec.getLocalSize()
self.u_sol.x.petsc_vec.setValues(range(start_u, end_u), x.array_r[:u_size_local])
self.p_sol.x.petsc_vec.setValues(range(start_p, end_p), x.array_r[u_size_local:])
```

`x` es un vector bloque de `create_vector_block()`. En MPI, el layout de `x.array_r`
puede incluir DOFs fantasma entrelazados de forma diferente a la asumida. Si `u_size_local`
no coincide exactamente con el offset del bloque de presión en el vector bloque, se
escriben valores incorrectos. Esto podría corromper silenciosamente el estado en
corridas MPI con particionado no trivial (típico en 3D).

**Alternativa más robusta:** usar `x.getNestSubVecs()` o un mapeo explícito con el
index map de cada espacio.

---

### Problema potencial: `mesh.h()` vs `CellDiameter` en parámetros de estabilización
**Archivo:** `src/solvers/stabilized_schur.py:82-88` (y todos los solvers)

```python
h = Function(V_dg0)
h.x.array[:] = mesh.h(mesh.topology.dim, np.arange(...))
# h = CellDiameter(self.mesh)  <- comentado
```

`mesh.h()` retorna el diámetro del simplex (arista más larga). Para tetraedros 3D,
esto puede ser significativamente mayor que la escala característica del elemento,
produciendo `tau_supg` mayor al teórico → estabilización artificial excesiva.

`CellDiameter` de UFL usa el diámetro inscrito del elemento, que es más representativo
de la escala local de resolución.

---

### Problema conocido: PCD solver sin null space interno
**Archivo:** `src/solvers/stabilized_pcd.py:307`

Hay un comentario explícito `# ojo aca faltan cosas de null space` indicando que el
precondicionador PCD no tiene implementado el manejo del espacio nulo para sus subsolves
internos. Si se usa con BCs que no eliminan el espacio nulo, falla.

---

### Problema potencial: BDF2 — Jacobiano derivado una sola vez
**Archivo:** `src/solvers/stabilized_schur_bdf2.py`

Los coeficientes BDF (`bdf_a0`, `bdf_a1`, `bdf_a2`) son `Constant` y se actualizan en
`solveStep()`. El Jacobiano `J` se deriva de `F` una sola vez en `setup()`. Aunque UFL
usa evaluación lazy (los Constants se evalúan al momento del ensamblaje), existe
riesgo de que el compilador de formas cachee el Jacobiano con los valores del primer
ensamblaje (BDF1). Requiere verificación.

---

### Perfil parabólico de inlet asume eje X
**Archivo:** `src/experiments/scenario_factory.py:110-112`

```python
def inlet_profile_expression(x):
    r_sq = x[1]**2 + x[2]**2  # Asume inlet perpendicular al eje X
    return np.stack((val, np.zeros_like(val), np.zeros_like(val)))  # Vel solo en X
```

Si la arteria no está perfectamente alineada con el eje X, el perfil de inlet es
incorrecto. Para un árbol microvascular con múltiples ramas, cada inlet tiene orientación
diferente. Esto no causa divergencia del solver lineal directamente, pero produce
BCs físicamente incorrectas.

---

### `MAX_ITER = 20` en SNES para sistemas 3D grandes
**Archivo:** `src/solvers/stabilized_schur.py:41`

Con 20 iteraciones máximas en el solver de Newton, si el residual no cae suficientemente
rápido (lo cual ocurre cuando el precondicionador lineal es débil), SNES reporta
no convergencia aunque el problema sea resoluble con más iteraciones. Con un sistema
de 1M DOFs, cada iteración SNES puede requerir más iteraciones KSP para reducir
el residual, y si el precondicionador no es suficientemente bueno, el residual del
Newton no cae en 20 pasos.

---

## Próximos Pasos de Investigación

1. **Probar `stabilized_lsc` en `unit_cube`** — si converge, confirma que el
   problema es el precondicionador, no la formulación
2. **Probar `stabilized_schur` en `unit_cube` con opciones PETSc explícitas:**
   - Aumentar `MAX_ITER`
   - Usar `-ksp_monitor` y `-snes_monitor` para ver el comportamiento de convergencia
   - Forzar un precondicionador más fuerte via `-ksp_u_pc_type lu` para verificar si
     el problema es solo el precondicionador o también la formulación
3. **Verificar `updateSolution` en MPI:** Correr con 2 procesos en `unit_cube` y
   verificar que la solución coincide con corrida serial
4. **Reemplazar `mesh.h()` por `CellDiameter`** y observar si cambia el comportamiento
5. **Inspeccionar número de iteraciones KSP antes de divergencia** con verbose logging

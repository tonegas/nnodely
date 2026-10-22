# Analisi dell'ordinamento dei tensori in nnodely

Questo documento analizza il layout dei tensori di nnodely 2.0 (branch `release/2.0.0`):

- **Parte 1:** i problemi presenti oggi nel codice che dipendono dalle dimensioni.
- **Parti 2-3:** il confronto con un layout alternativo in stile Keras (P2), caso per caso.
- **Parti 4-6:** pro e contro dei due approcci, lato backend e lato utente.

I casi marcati ✅ *verificato* sono stati eseguiti sul codice attuale (Keras 3.12, backend TensorFlow). Il comportamento di P2 è un'analisi di progetto: P2 non è implementato, quindi i casi P2 sono dedotti dalle regole di broadcasting di Keras e numpy. Le uniche eccezioni sono i calcoli con `matmul`, verificati in numpy.

---

## 0. Le due convenzioni

### Attuale (L0)

```
(B, *D, T, S_inner, …, S_outer)
```

- **Dove è definito:** in `Shape.tuple = dim + (time,) + seq` ([src/nnodely/core/stream.py:78](../src/nnodely/core/stream.py)).
- **Ordine degli assi seq:** dal più interno al più esterno. Un `Loop` aggiunge il suo asse di rollout **in coda** ([loop.py:650-656](../src/nnodely/layers/loop.py)). Il `DataLoader` aggiunge ogni livello di finestra in coda con `sliding_window_view` ([dataloader.py:575](../src/nnodely/core/dataloader.py)).
- **Tra D e T:** l'asse dim viene **prima** dell'asse time, quindi una finestra di 5 istanti di un segnale a 3 componenti ha shape `(3, 5)`.
- **Broadcasting tra stream:** `BinaryOpImpl` ([layer.py:365-391](../src/nnodely/core/layer.py)) aggiunge `expand_dims(-1)` all'operando di rango minore. Gli assi vengono quindi allineati **da sinistra**, subito dopo il batch. Non è il broadcasting di numpy e Keras, che allinea da destra.

### Proposta P2 (stile Keras, feature in coda)

```
(B, S_outer, …, S_inner, T, D1, D2, …)
```

Regola unica, quella di numpy in ordine C: **ogni asse è più esterno di tutti quelli alla sua destra**.

- una sequenza contiene finestre temporali;
- ogni istante contiene un tensore di feature `(D1, D2, …)`.

Il blocco D mantiene il suo ordine interno: un valore `(3,3)` resta `(3,3)`. Si sposta soltanto in coda.

`Input(seq=(a, b))` si legge `(outer, inner)`, così l'ordine dichiarato coincide con quello del tensore.

---

## Parte 1 — Problemi presenti oggi legati alle dimensioni

Gravità: 🔴 risultato sbagliato senza errore · 🟠 errore o limitazione dove l'utente si aspetta che funzioni · 🟡 incoerenza, uso poco intuitivo, documentazione.

### 1.1 🟠 Le sequenze annidate richiedono `order="F"` ✅ *verificato*

Esempio dal notebook [layer_test.ipynb](test_nnodely2.0/layer_test.ipynb), cella 5: un integratore annidato su 12 campioni, 4 interni × 3 esterni, con in ingresso una rampa.

```python
x.reshape(1,1,1,4,3)             -> flatten(F): [0 3 9 18 19 23 30 40 42 47 55 66]   # SBAGLIATO, nessun errore
x.reshape(1,1,1,4,3, order="F")  -> flatten(F): [0 1 3 6 10 15 21 28 36 45 55 66]   # = cumsum
```

L'asse più esterno sta a destra. Nell'ordine C di numpy è l'asse più a destra a variare più velocemente, quindi i campioni consecutivi finiscono in chunk diversi. Senza `order="F"`, sia in ingresso (`reshape`) sia in uscita (`flatten`), il risultato è **sbagliato in silenzio**: la shape è giusta e quindi nessun controllo scatta.

Il `DataLoader` con `seq=(3,4)` produce `(N, D, T, S_in, S_out)`. Chi vuole dei chunk al posto delle finestre sovrapposte deve fare il reshape a mano, di nuovo con `order="F"`.

### 1.2 🟡 Le finestre multidimensionali vanno trasposte

Una serie con più componenti arriva di solito come tabella `(N, D)`, per esempio le colonne di un dataframe. Il layout di nnodely mette invece D prima di T:

- `Input("a", dim=2).sw(5)` ha shape `(B, 2, 5)`;
- una serie di matrici `(N, 3, 3)` usata come `Input("R", dim=(3,3), seq=-1)` va passata come `np.moveaxis(R, 0, -1).reshape(1, 3, 3, 1, N)`.

### 1.3 🔴 Broadcasting sbagliato con dim-rank diversi ✅ *verificato*

`BinaryOpImpl` conosce solo il rango del tensore, non i gruppi dim/time/seq. Quando due operandi hanno un numero diverso di assi dim, l'asse **time** di uno finisce allineato con un asse **dim** dell'altro.

| Espressione | Risultato | |
|---|---|---|
| `Input('a',dim=2).sw(3) + Input('b',dim=(2,3)).last()` | `Shape(D=(2,3), T=1)`; con `a=[[0,100,200],[0,0,0]]` e `b=arange(6)` dà `[[0,101,202],[3,4,5]]` | 🔴 la finestra temporale di `a` è stata sommata lungo il secondo asse dim di `b` |
| `Input(dim=1).sw(3) * Parameter(dim=(2,3))` | dichiarato `D=(2,3), T=1`; con `time=[10,20,30]` dà `[[0,20,60],[30,80,150]]` | 🔴 |
| `Input(dim=5).sw(5) * Constant(value=(5,5))` | `D=(5,5), T=1` | 🔴 |
| `Input('z',dim=2,seq=5).last() + Parameter(value=ones((2,5)))`, intendendo `(dim, seq)` ma senza `seq=1` | `Shape(D=(2,), T=5, S=(5,))` | 🔴 compare `T=5` dal nulla |
| `Input(dim=(3,)) + Input(dim=(3,2))` | funziona | 🟡 ma solo perché T (di taglia 1) si allinea con `dim[1]` |

Lo stesso schema "aggiungi gli assi mancanti in coda" è usato anche da `_broadcast_constant` ([validation.py:276-286](../src/nnodely/core/validation.py)) e da `_align_target` ([minimizer.py:207-209](../src/nnodely/core/minimizer.py)).

### 1.4 🟠 Errori dove numpy funzionerebbe ✅ *verificato*

| Espressione | Oggi | In numpy |
|---|---|---|
| `Input(dim=(2,3)) * Parameter(dim=3)` | errore | un peso per colonna |
| `Input(dim=(3,2)) * Parameter(dim=3)` | funziona, un peso per **riga** | errore |
| `Input(dim=3).sw(5) * Constant([5 valori])`, inteso come un peso per istante | errore: serve `time=0` | — |
| `Input(seq=(3,)) + Input(seq=(4,3))`, inteso come "un valore per passo del rollout" | errore poco chiaro: "automatic output shape inference failed" | — |

Il caso `seq=(4,) + seq=(4,3)` funziona e si allinea con l'asse **interno**: è il caso tipico di una finestra dentro un Loop.

### 1.5 🔴 `Atan2` salta l'allineamento ✅ *verificato*

`Atan2Impl` chiama direttamente `keras.ops.arctan2(y, x)` ([trigonometric.py:72-76](../src/nnodely/layers/trigonometric.py)). In questo modo usa il broadcasting di Keras, allineato da destra, e il batch finisce allineato con un asse dim.

```python
x = Input('x', seq=4)
Atan2()([x.last(), Constant('c1', value=[1.])])
# dichiarato Shape(D=(1,), T=1, S=(4,)), con batch 3 restituisce (3, 3, 1, 4)
```

Il controllo della shape fatto con batch=1 non se ne accorge.

### 1.6 🟠 Il valore iniziale del Loop è allineato da destra

`_align_to_shape` ([loop.py:151-159](../src/nnodely/layers/loop.py)) e `_state_shape` ([loop.py:672-689](../src/nnodely/layers/loop.py)) allineano l'init **da destra**, quindi con una regola diversa da `BinaryOpImpl`.

- Un init con `dim=3` e senza seq, collegato a un ingresso di callback con `dim=3, seq=2`, viene rifiutato:
  `Loop initial value … has shape (3, 1), which cannot feed callback input … of shape (3, 1, 2)` ✅ *verificato*.
- Con `dim=1` lo stesso caso funziona, ma solo perché gli assi allineati in modo sbagliato hanno tutti taglia 1.

### 1.7 🟡 Il significato di `axis=-1` cambia a seconda del contesto

| Dove | Cosa indica `-1` oggi |
|---|---|
| `SeqSelect`, `SeqRange`, `SeqConcatenate`, `SeqStack`, `SeqReverse` (default, [partitions.py:689-949](../src/nnodely/layers/partitions.py)) | l'ultimo asse seq, cioè il rollout del Loop **più esterno**. La documentazione ([partitions.py:6](../src/nnodely/layers/partitions.py), `NNODELY_AI_GUIDE.md:249`) dice erroneamente "innermost" |
| `seq_weights` del minimizer ([minimizer.py:259](../src/nnodely/core/minimizer.py), [utils.py:115-127](../src/nnodely/utils/utils.py)) | l'ultimo asse del tensore: il rollout se ci sono seq, **il time** se non ci sono |
| Codice utente in `KerasFunction` (`x.sum(axis=-1)`) ✅ *verificato* | l'asse **seq** se lo stream ha seq, altrimenti l'asse **time** |
| Loss Keras e `MaskedLoss` | riducono l'ultimo asse, cioè rollout o time. Le loss elemento per elemento non ne risentono, la crossentropy sì |

### 1.8 🟠 I Parameter in `KerasFunction` arrivano con il layout degli stream ✅ *verificato*

`ParameterImpl.call` ([parameter.py:67-72](../src/nnodely/layers/parameter.py)) aggiunge il batch e lo replica. `_value_layout` aggiunge anche l'asse time. Per esempio `W = Parameter("W", value=np.eye(3))` arriva alla funzione dell'utente come `(B, 3, 3, 1)`, mentre `x` arriva come `(B, 3, T)`.

```python
def call(self, x, W):
    return x @ W   # ValueError
```

Oggi almeno l'errore è esplicito. Per usare il parametro però bisogna conoscere il layout interno: `W[0, ..., 0]`, oppure un `einsum` con gli assi batch e time.

### 1.9 🟡 Incoerenze tra layer con più dimensioni ✅ *verificato*

| Layer | Comportamento |
|---|---|
| `Fir(F)` su `D=(2,3)` | `D=(F,2,3)`: **F va in testa** ([fir.py:76-88](../src/nnodely/layers/fir.py)) |
| `Linear(F, axis=k)` | F resta **al posto** dell'asse k |
| `Linear(F)` con `axis=None` | collassa **tutto** il D in `(F,)` |
| `BatchNorm` (default `axis=1`, [batchnorm.py:28](../src/nnodely/layers/batchnorm.py)) | con `D=(2,3)` normalizza solo D1: gamma ha shape `(2,)` |
| `PReLU` ([activations.py:104](../src/nnodely/layers/activations.py)), `Softmax` | accettano interi d'asse **contati come in Keras**, che dipendono dal layout |
| `Sum` ([arithmetic.py:196](../src/nnodely/layers/arithmetic.py)) | assi relativi ai dim, mappati con `1 + axis` |

`FirImpl` lavora già internamente nel layout `[B, *S, *D, 1, T]` e poi traspone per tornare a L0.

### 1.10 🟠 Altri difetti scoperti durante l'analisi

- **`LocalModel` rifiuta attivazioni con seq** ([localmodel.py:290-296](../src/nnodely/layers/localmodel.py)), anche se `Fuzzify` su un ingresso con seq le produce ✅ *verificato*.
- **`node.loop_outputs` su un Loop non spacchettato** ([loop.py:561](../src/nnodely/layers/loop.py)). Il nodo ha `return_all_outputs=False`, quindi `LoopOutputImpl` indicizza un tensore singolo: si ottiene un errore oppure, in silenzio, una fetta del batch ✅ *verificato*. Funziona solo lo spacchettamento con `__iter__`.
- **Keras completa da solo al massimo un asse finale.**
  - `(2,3)` viene accettato al posto di `(2,3,1)` per `Input(dim=3).last()`.
  - `(2,4)` per `x.sw(4)` (attesa `(B,1,4)`) dà errore.
  - `Input(seq=5)` accetta solo il rango completo.

  `Modely.__call__` non fa alcuna conversione ([modely.py:110-124](../src/nnodely/core/modely.py)).
- **`seq_length='full'`** è accettato solo se la seq dinamica è l'ultima ([dataloader.py:268](../src/nnodely/core/dataloader.py)).

### 1.11 🟡 Indici scritti a mano nel codice

Il layout non è definito in un solo punto. Circa 30 punti calcolano gli indici direttamente:

- **asse time = `1 + dim_rank`:**
  - [modely.py:262](../src/nnodely/core/modely.py), [loop.py:716](../src/nnodely/layers/loop.py), [roll.py:207](../src/nnodely/layers/roll.py);
  - `partitions.py:36,46,54,136,148,341,399,462`;
  - `parameter.py:128-138,185-190`.
- **asse seq:** `_tensor_axis = 2 + dim_rank + seq_axis` ([partitions.py:621](../src/nnodely/layers/partitions.py)).
- **rollout = `shape[-1]` / `seq[-1]`:**
  - `loop.py:608-628,650-669`;
  - `minimizer.py:128,209,245-259`;
  - `utils.py:115-127`, `validation.py:379`;
  - `integrate.py:396,444`, `ode.py:402,737`.
- **asse delle celle = 1:** `localmodel.py:71-82,124,161,199`.
- **slice sull'asse 1 e `expand_dims(-1)`:** `fuzzify.py:274-278`.
- **parsing della shape:** `_shape_from_tensor` ([layer.py:206-228](../src/nnodely/core/layer.py)) conta le seq da destra.

### 1.12 🟡 Documentazione

- Il notebook `layer_test.ipynb`, celle 3 e 5, descrive il layout come `(batch, T, D, …)`. Quello reale è `(B, D, T, …)`.
- La documentazione di `axis=-1` per le operazioni seq è sbagliata (vedi 1.7).

---

## Parte 2 — Come si comporterebbe P2

### 2.1 Regole di P2

1. **Layout:** `(B, S_outer…, S_inner, T, *D)`.
2. **Dichiarazione delle seq:** `Input(seq=(outer, inner))`. La seq dinamica (`-1`) è ammessa solo sul rollout, cioè su `seq[0]`.
3. **Valori dei Parameter:** le shape sono scritte come in numpy, con tempo e feature `(T, *D)`. Una finestra di pesi per 5 istanti di un segnale a 3 componenti ha shape `(5, 3)`.
4. **Broadcasting tra stream, allineato per gruppi.** Usa la `Shape` simbolica, non il rango del tensore:
   - gli assi seq mancanti si inseriscono in **posizione 1**, subito dopo il batch;
   - gli assi D mancanti si inseriscono **tra T e D**, in modo che il blocco D sia allineato da destra come in numpy;
   - T con taglia 1 fa broadcast normalmente.

   Il broadcasting di Keras da solo **non basta** nemmeno in P2: `(B,T,D)` contro `(B,S,T,D)` allineati da destra mettono B di fronte a S.
5. **Default degli assi seq:** `axis=0` indica il rollout più esterno, `axis=-1` il livello più interno.
6. **Parameter in `KerasFunction`:** arrivano come pesi grezzi, con la shape del valore e senza batch né T (vedi 2.3).

### 2.2 I casi della Parte 1 rivisti in P2

| # | Caso | Oggi (L0) | In P2 |
|---|---|---|---|
| 1.1 | Loop annidato da serie piatta | `reshape(1,1,1,inner,outer, order="F")` e `flatten(order="F")`; senza F il risultato è sbagliato in silenzio | `reshape(1, outer, inner, 1, 1)` e `flatten()`, in ordine C |
| 1.1 | `DataLoader`, `seq=(outer, inner)` | `(N, D, T, S_in, S_out)` | `(N, S_out, S_in, T, D)`; serve un `moveaxis(-1 → 1)` dopo ogni `sliding_window_view` |
| 1.2 | Finestra `a.sw(5)` con `dim=2` | `(B, 2, 5)`: righe del dataframe trasposte | `(B, 5, 2)`: le righe del dataframe così come sono |
| 1.2 | Serie di matrici `(N,3,3)` | `np.moveaxis(R,0,-1).reshape(1,3,3,1,N)` | `R.reshape(1, N, 1, 3, 3)` |
| 1.3 | `a(dim=2).sw(3) + b(dim=(2,3)).last()` | 🔴 sbagliato in silenzio | `(B,3,2)` contro `(B,1,2,3)`. Con il broadcasting di Keras: errore (`2` contro `3`). Con l'allineamento per gruppi: `a` diventa `(B,3,1,2)` e il blocco D fa broadcast da destra, `2` contro `3`, quindi errore esplicito. È il comportamento di numpy, che per un broadcast per riga vorrebbe `dim=(2,1)` |
| 1.3 | `Input(dim=1).sw(3) * Parameter(dim=(2,3))` | 🔴 sbagliato in silenzio | `(B,3,1)` contro `(1,2,3)`, allineati per gruppi: risultato `(B,3,2,3)`, cioè `T=3`, `D=(2,3)`. È ciò che ci si aspetta |
| 1.3 | `Input(dim=5).sw(5) * Constant((5,5))` | 🔴 `D=(5,5), T=1` | `D=(5,5)` fa broadcast su `T=5`: `(B,5,5,5)`. Se l'utente intendeva "un peso per istante e per componente" deve dichiarare `time=0`. L'ambiguità resta ma diventa visibile, perché la shape dell'output mostra `T=5` |
| 1.3 | `z(dim=2,seq=5).last() + Parameter(ones((2,5)))` senza `seq` | 🔴 compare `T=5` | il valore è `D=(2,5)`, contro `D=(2,)` di `z`: errore esplicito sul blocco D (`5` contro `2`). Per un peso per passo l'utente scrive `Parameter(value=ones((5,2)), seq=0)` |
| 1.4 | `Input(dim=(2,3)) * Parameter(dim=3)` | 🟠 errore | ✓ un peso per colonna, come numpy |
| 1.4 | `Input(dim=(3,2)) * Parameter(dim=3)` | funziona per riga | errore, come numpy: per riga serve `dim=(3,1)` |
| 1.4 | `Input(dim=3).sw(5) * Constant([5 valori])` per istante | 🟠 serve `time=0` | serve ancora `time=0`, perché un vettore 1-D di default è D. Con `Constant(value=ones((5,1)), time=0)` lo si scrive come in numpy |
| 1.4 | `seq=(4,) + seq=(R, 4)`, una finestra dentro il Loop | ✓ si allinea con l'asse interno | ✓ con la regola numpy (assi seq mancanti inseriti in posizione 1, livelli esistenti allineati da destra): `(1,4)` contro `(R,4)`. Stesso risultato di oggi |
| 1.4 | `seq=(3,) + seq=(R=3, 4)`, un valore per passo del rollout | 🟠 errore poco chiaro | 🟠 con la regola numpy: `(1,3)` contro `(3,4)` dà errore, ma con un messaggio chiaro sulle seq. L'utente deve dare all'operando un livello interno di taglia 1 (`seq=(3,1)`), come farebbe in numpy. L'alternativa (allineare sul rollout) renderebbe sbagliato il caso precedente: va scelta una sola semantica, e quella numpy è la più prevedibile |
| 1.5 | `Atan2` | 🔴 shape sbagliata | resta un 🔴 se `Atan2` non passa per `BinaryOpImpl`. La correzione è la stessa nei due layout |
| 1.6 | Init `dim=3` verso un ingresso `dim=3, seq=2` | 🟠 rifiutato | ✓ allineato da destra: `(1,3)` contro `(2,1,3)` funziona |
| 1.7 | `SeqSelect(-1)` | ultimo passo del rollout più esterno | ultimo elemento del livello **più interno**. Per il rollout serve `axis=0`, che diventa il nuovo default. Con una sola seq non cambia nulla |
| 1.7 | `seq_weights` | `shape[-1]`: rollout o time | `shape[-1]` sarebbe il blocco **D** (🔴 in silenzio se `len(w)==D`). Va reso esplicito: asse 1 se ci sono seq, altrimenti l'asse T |
| 1.7 | `x.sum(axis=-1)` in `KerasFunction` | seq o time | ✓ sempre l'ultima feature, come in Keras |
| 1.7 | Loss Keras (crossentropy) | riduce rollout o time | ✓ riduce le feature, che è la semantica Keras corretta |
| 1.8 | `x @ W` in `KerasFunction`, W come stream `(B,1,3,3)` | 🟠 `ValueError` | 🔴 `(B,B,T,3)` in silenzio (verificato in numpy, con B=4 si ottiene `(4,4,5,3)`) |
| 1.8 | `x @ W`, W come peso grezzo `(3,3)` | — | ✓ `(B,T,3)` |
| 1.9 | `Linear(F)` su `D=(2,3)` | `(F,)` | con il default `axis=-1` = `keras.layers.Dense`: `(2,F)`. Il comportamento "collassa tutto il D" va chiesto esplicitamente |
| 1.9 | `Fir(F)` su `D=(2,3)` | `(F,2,3)` | `(2,3,F)`, coerente con `Linear` e `Dense` |
| 1.9 | `BatchNorm` | `axis=1`, solo D1 | il default Keras `-1` normalizza solo l'ultimo asse D. Il default giusto è `axis=(-dim_rank…-1)`, tutto il blocco D, che Keras supporta |
| 1.9 | `PReLU`, `Softmax`, `Sum` con interi d'asse | contati come in Keras | contati sempre dentro il blocco D. Gli interi negativi coincidono con Keras |
| 1.10 | `LocalModel` / `Fuzzify` con seq | 🟠 | stesso problema; la correzione è indipendente dal layout |
| 1.10 | `loop_outputs` | 🔴 | indipendente dal layout |
| 1.10 | Keras completa un asse finale | `(2,3)` diventa `(2,3,1)`: aggiunge **T** | `(B,T)` diventa `(B,T,1)`: aggiunge **D=1**, cioè una colonna di un dataframe con un solo segnale. Più naturale |
| 1.11 | Indici scritti a mano | ~30 punti | ~30 punti, tutti da cambiare. Conviene comunque centralizzarli in `Shape` (`time_axis`, `rollout_axis`, `seq_axes`, `dim_axes`) |

### 2.3 I Parameter come stream e come pesi

Negli **operatori nnodely** (`x * W`, `x + W`) un Parameter resta uno stream e passa per l'allineamento per gruppi. Il comportamento è lo stesso nei due layout, ma in P2 coincide con numpy: un `Parameter(dim=3)` fa broadcast su `(…, 3)`.

In **`KerasFunction`**, in P2 diventa **necessario** passare i Parameter come pesi grezzi (`self.variable`, con la shape del valore), altrimenti `matmul` produce risultati sbagliati in silenzio (1.8). Le opzioni:

- **(a)** Passarli grezzi automaticamente. Più comodo, ma un Parameter avrebbe due semantiche.
- **(b)** Passarli in un argomento separato, per esempio `KerasFunction(Rot, params={"W": W})([x])`. È esplicito: chi legge vede subito quali argomenti sono stream e quali pesi.

In entrambi i casi `_shape_from_tensor` deve gestire un output privo di batch, calcolato solo dai pesi.

In L0 lo stesso cambiamento sarebbe solo un miglioramento d'uso, perché oggi l'errore è esplicito.

### 2.4 Il notebook `layer_test.ipynb` in P2

**Cella 3: una seq dinamica, D=1.** Le dichiarazioni dei layer non cambiano.

```python
inputs = {
    "ax_sq":    acc_x.reshape(1, -1, 1, 1),        # (B, S, T, D)
    "ay_sq":    acc_y.reshape(1, -1, 1, 1),
    "px0_init": np.array(500.0).reshape(1, 1, 1),  # (B, T, D)
    "py0_init": np.array(-500.0).reshape(1, 1, 1),
}
pos_x = np.asarray(res["int_loop1"]).flatten()
```

**Cella 5: Loop annidato.** `Input("ax_sq_sq", seq=(-1, -1))` resta uguale ma si legge `(outer, inner)`.

```python
inputs = {
    "ax_sq_sq":    acc_x.reshape(1, outer, inner, 1, 1),  # niente order="F"
    "ay_sq_sq":    acc_y.reshape(1, outer, inner, 1, 1),
    "px0_init_sq": np.array(500.0).reshape(1, 1, 1),
    "py0_init_sq": np.array(-500.0).reshape(1, 1, 1),
}
pos_x = np.asarray(res["int_loop1_sq"]).flatten()
```

**Stessa dinamica con un ingresso vettoriale.** `a = Input("a", dim=2, seq=-1)` al posto di `ax`/`ay`:

```python
acc = np.stack([acc_x, acc_y], axis=-1)            # (N, 2): le colonne di un dataframe
inputs = {
    "a":       acc.reshape(1, N, 1, 2),             # (B, S, T, D)
    "p0_init": np.array([500., -500.]).reshape(1, 1, 2),
}
# annidato: acc.reshape(1, outer, inner, 1, 2)
```

Gli stessi esempi oggi (L0):

```python
inputs = {
    "a":       acc.T.reshape(1, 2, 1, N),
    "a_sq_sq": acc.T.reshape(1, 2, 1, inner, outer, order="F"),
}
```

---

## Parte 3 — Backend: regole di broadcasting e implementazione

| Aspetto | L0 (attuale) | P2 |
|---|---|---|
| Regola di broadcasting di base | Allineamento **da sinistra**, con `expand_dims(-1)` in `BinaryOpImpl`. Indispensabile, perché da destra il batch finirebbe allineato con D | Allineamento **da destra** come numpy/Keras: corretto per D, per i Parameter `(D,)`, per gli scalari e per le statistiche di normalizzazione |
| Dove la regola di base sbaglia | dim-rank diversi (🔴 in silenzio, 1.3) | seq-rank diversi: `(B,T,D)` contro `(B,S,T,D)` mette B di fronte a S (🔴 in silenzio se B==1 o B==S); dim-rank diversi insieme a T: il D mancante va inserito **prima** di T |
| Regola necessaria | allineamento per gruppi (seq / T / D) usando la `Shape` | allineamento per gruppi (seq / T / D) usando la `Shape` |
| Coerenza delle regole interne | tre regole diverse: `BinaryOpImpl` da sinistra, init del Loop da destra, `Atan2` con il broadcasting Keras | una sola regola possibile, quella di numpy esteso ai gruppi |
| `Linear` | transpose + `tensordot` | `tensordot` sugli ultimi assi; con D a un asse è esattamente `Dense` |
| `Fir` | traspone in `[B,*S,*D,1,T]` e poi torna indietro | serve un transpose (T non è l'ultimo asse), oppure una `Conv1D` depthwise; l'output con F in coda è naturale |
| `IntegrateStep`, `Derivative` | `tensordot` sull'asse T più `moveaxis` | come oggi: T non è l'ultimo asse. Con D a un asse basta `matmul(op, x)` sull'asse T |
| `Roll`, shift della callback | asse `1 + dim_rank` | asse `1 + seq_rank`: dipende dalle seq, non dai dim |
| Rollout del Loop | ultimo asse; `_loop_axis` controlla un suffisso | asse 1; controlla un prefisso |
| `DataLoader` | `sliding_window_view` aggiunge in coda: naturale | serve un `moveaxis(-1 → 1)` per ogni livello |
| Normalizzazione | `reduce_axes = (0, 1+dim_rank…)`; statistiche `(D,1,…)` | `reduce_axes` = tutti tranne il blocco D; statistiche `(D…)` che fanno broadcast da destra |
| Loss Keras, `seq_weights`, mask | ultimo asse = rollout o time | servono assi espliciti; le loss riducono D, che è corretto |
| Layer Keras dentro `KerasFunction` (`Dense`, `LSTM`, `Conv1D`, `BatchNorm`) | incompatibili con il layout | compatibili: `Dense` subito; `LSTM`/`Conv1D` dopo un `Reshape` del blocco D e la fusione delle seq nel batch |
| Interoperabilità ONNX/Keras | export neutro rispetto al layout | uguale; gli utenti dei modelli esportati trovano il layout standard |
| Costo della migrazione | — | ~30 indici, tutti i layer con assi, `DataLoader`, `minimizer`, la serializzazione (config dei Parameter e del Loop) e ~100 asserzioni di shape nei test (`test_closed_loop`, `test_dataset`, `test_partitioning_windows`, `test_save_export`, `test_shape_convention`, `test_minimize`) |

---

## Parte 4 — Frontend: cosa vede l'utente

| Aspetto | L0 (attuale) | P2 |
|---|---|---|
| Da serie piatta a sequenze annidate | `order="F"` in ingresso e in uscita; dimenticarlo dà un risultato sbagliato in silenzio | reshape C |
| Da tabella `(N, D)` o dataframe | trasposizione | uso diretto |
| Serie di tensori `(N, D1, D2)` | `moveaxis` | reshape |
| Lettura della shape | "dall'interno all'esterno": D, T, S_in, S_out | "dall'esterno all'interno", come numpy, PyTorch e Keras |
| Dichiarazione `seq=(a, b)` | `(inner, outer)` | `(outer, inner)`, la stessa forma dei dati |
| `Parameter(value=…)` con finestra temporale | `(D, T)` | `(T, D)`, una riga per istante |
| Broadcasting di Parameter e Constant | regole proprie; casi simili a numpy danno errore (1.4) o sono sbagliati (1.3) | semantica numpy |
| `axis=-1` | significa cose diverse a seconda del contesto (1.7) | sempre l'ultima feature; il rollout è `axis=0` delle seq |
| `KerasFunction` | il codice deve conoscere il layout di nnodely | il codice Keras/numpy scritto normalmente funziona, se i Parameter arrivano grezzi |
| Uscite di `Linear` / `Fir` multidimensionali | F in posizioni diverse | F sempre in coda |
| Codice e modelli 2.0 esistenti | invariati | **rotti**: shape degli ingressi, valori dei Parameter, `seq=(…)`, `axis=-1` delle seq, modelli salvati |

---

## Parte 5 — Pro e contro

### L0 (layout attuale)

**Pro**
- Funziona, è testato ed è coerente con la logica di `sliding_window_view`: ogni livello di finestra si aggiunge in coda.
- Il caso più comune (finestra interna + rollout del Loop) si allinea già correttamente con il padding da sinistra.
- Nessuna migrazione: modelli salvati, test e documentazione restano validi.
- `IntegrateStep`, `Derivative` e `Roll` usano un asse time che non dipende dal numero di seq.

**Contro**
- `order="F"` (o trasposizioni) sono necessari per i dati reali, e dimenticarli dà risultati sbagliati in silenzio (1.1, 1.2).
- Broadcasting sbagliato in silenzio con dim-rank diversi (1.3), e semantica diversa da numpy (1.4).
- Tre regole di allineamento diverse dentro la libreria (1.3, 1.5, 1.6).
- `axis=-1` ambiguo (1.7); layer Keras e loss standard lavorano sull'asse sbagliato.
- Il codice dentro `KerasFunction` deve conoscere il layout interno (1.8).

### P2 (layout Keras)

**Pro**
- Un'unica regola mentale, quella di numpy in ordine C: i dati si costruiscono sempre con `reshape(B, S_out…, S_in, T, *D)`, qualunque sia il numero di D.
- Il broadcasting per i dim coincide con numpy: Parameter, scalari, statistiche e init del Loop funzionano senza regole speciali.
- `Linear` coincide con `Dense`; `BatchNorm`, le loss e i layer Keras dentro `KerasFunction` funzionano come si aspetta chi conosce Keras.
- Elimina i casi 1.1, 1.2, 1.6 e 1.7. I casi 1.3 e 1.4 diventano errori espliciti o comportamenti uguali a numpy.

**Contro**
- Migrazione su tutta la libreria, e rottura di tutto il codice e dei modelli 2.0 esistenti.
- Il broadcasting di Keras **non basta**: l'allineamento per gruppi resta necessario, e il caso "seq-rank diverso" diventa pericoloso (B allineato con S) se non viene gestito.
- La semantica delle seq va fissata: allineamento dall'interno (numpy) oppure sul rollout. Con una scelta ingenua il caso `seq=(4,) + seq=(R,4)`, oggi corretto, si rompe.
- `seq_weights`, le mask e `_align_target` hanno bisogno di assi espliciti: con `shape[-1]` peserebbero le feature, in silenzio.
- I Parameter in `KerasFunction` vanno passati come pesi grezzi, altrimenti `matmul` è sbagliato in silenzio. Questo introduce una doppia semantica, o un'API nuova.
- L'asse time dipende dal numero di seq; `DataLoader`, `Fir` e `IntegrateStep` hanno bisogno di `moveaxis`.

---

## Parte 6 — Quale è più intuitivo

**Per l'utente, P2.** Ogni operazione che un utente fa d'istinto con numpy, pandas o Keras dà il risultato atteso:

- reshape di una serie;
- colonne di un dataframe;
- pesi `(D,)`;
- `axis=-1`;
- `x @ W`;
- `Dense`, `BatchNorm`.

In L0 la stessa intuizione porta a errori silenziosi: 1.1, 1.3 e la cella 5 del notebook ne sono esempi concreti.

**Per lo sviluppatore backend, i due layout sono simili.** In nessuno dei due si può delegare il broadcasting a Keras. P2 ha però tre vantaggi:

- le eccezioni sono meno numerose e più prevedibili (solo seq e T, mentre il blocco D segue numpy);
- i layer Keras standard si possono riusare;
- c'è un solo modello di riferimento, numpy, per decidere come deve comportarsi un caso nuovo.

L0 ha il vantaggio di non richiedere lavoro.

### Raccomandazione

**Interventi da fare in ogni caso, indipendenti dal layout:**

1. Allineamento per gruppi (seq / T / D) basato sulla `Shape`, in `BinaryOpImpl`, `_broadcast_constant`, `_align_target`, nell'init di `Integrate`/`Derivative` e nell'init del Loop.
2. Far passare `Atan2` per lo stesso allineamento.
3. Correggere `loop_outputs` e il rifiuto delle seq in `LocalModel`.
4. Aggiungere a `Shape` gli helper `time_axis`, `rollout_axis`, `seq_axes`, `dim_axes`, e sostituire i ~30 indici scritti a mano.
5. Correggere la documentazione di `axis=-1` e i commenti del notebook.

**Cambiare layout:** se si decide di passare a P2, il momento giusto è **prima del rilascio 2.0**, dopo i punti 1–4, che da soli riducono il cambio a una modifica di `Shape.tuple` e degli helper più l'aggiornamento dei test. Se si resta su L0, conviene almeno fornire degli helper `to_sequence(x, inner, outer)` / `from_sequence(y)` e documentare `order="F"`.

# Chatbot RAG de TiendaPepe (TypeScript + pgvector)

Un RAG completo sobre los 103 documentos de `../docs`, sin frameworks:

- `indexar.ts`: la **indexación**, una vez. Lee los `.md` de `../docs` y sus subcarpetas, los trocea por secciones (`## `), calcula el embedding de cada fragmento y lo guarda en Postgres con **pgvector**.
- `rag.ts`: la **consulta**, en cada pregunta: recupera los fragmentos más parecidos con pgvector, se los pasa al modelo y responde citando la fuente.
- `chatbot.ts`: el chat en la terminal, línea a línea.
- `tui.tsx`: el mismo chat en una interfaz de terminal (hecha con [Ink](https://github.com/vadimdemedes/ink)), con un visor de los fragmentos recuperados al lado.
- `evaluar.ts` y `golden.json`: la **evaluación** con un golden set de 25 preguntas (ver más abajo).
- `metrics.ts` y `metrics.test.ts`: las métricas de la evaluación y sus tests.
- `bm25.ts`: búsqueda por palabras clave con BM25, en memoria, con la fórmula a la vista (y `bm25.test.ts`, sus tests).
- `similitud.ts`: la similitud coseno entre pares de frases, para ver qué es "cerca" y "lejos".
- `db.ts`: la conexión a Postgres.
- `log.ts`: ayudas para los logs: el principio de un vector, el de un texto, el tiempo y el coste en euros.
- `compose.yaml`: Postgres 17 con pgvector, en Docker.

El `package.json` lleva `"type": "module"` y los scripts se ejecutan con `tsx`. El `tsconfig.json` carga los tipos de Node (`@types/node`) para que el editor reconozca `process`.

## El corpus

| Carpeta | Documentos | Qué hay |
|---|---|---|
| `politicas/` | 14 | Devoluciones, envíos, garantía, pagos, Club Pepe, reacondicionados, privacidad… |
| `ayuda/` | 20 | Preguntas frecuentes: contraseña, cupones, pedido retrasado, Bizum, regalos… |
| `productos/` | 53 | Fichas de producto: características, precio, devolución y garantía |
| `guias/` | 10 | Guías de compra y de tallas |
| `empresa/` | 6 | Sobre nosotros, almacén, sostenibilidad… |

Son unos 16.700 tokens y 450 fragmentos. Todo es inventado, pero coherente: las fichas de producto aplican las mismas reglas que las políticas.

## Preparación

Necesitas Docker y Node 22.

```bash
npm install
npm run db        # arranca Postgres con pgvector (docker compose up -d --wait)
```

Crea un `.env` en esta carpeta (nunca en el código):

```
OPENAI_API_KEY=<tu clave del curso>
OPENAI_BASE_URL=https://ai.commitacademy.io/v1
# Opcionales (por defecto, los alias del servicio del curso):
# MODELO=fast                 # o smart, más capaz
# MODELO_EMBEDDING=embed
# DATABASE_URL=postgres://rag:rag@localhost:5433/rag
# USD_A_EUR=0.86              # tipo de cambio para mostrar el coste en euros
```

## Ejecutar

```bash
npm run indexar                       # una vez, o cuando cambien los documentos
npm run tui                           # interfaz con el visor de fragmentos
npm run chat                          # chat interactivo, línea a línea
npm run chat -- "¿Cuántos días tengo para devolver unos auriculares?"
npm run chat -- --sin-rag "¿Cuántos días tengo para devolver unos auriculares?"
npm run chat -- --ver-contexto "¿Puedo pagar a plazos?"
```

- `--sin-rag` hace la misma pregunta sin documentos: el modelo no conoce TiendaPepe y dirá que no lo sabe o se inventará un plazo.
- `--ver-contexto` enseña exactamente qué texto recibe el modelo.
- `-k 5` cambia el número de fragmentos recuperados.
- `--busqueda bm25` o `--busqueda hibrida` cambia cómo se buscan los fragmentos (ver más abajo).
- En la TUI: `/rag` activa o desactiva el RAG, `/busqueda hibrida` cambia el modo de búsqueda, `/k 5` cambia los fragmentos recuperados y `/salir` (o Ctrl+C) sale.

El servicio del curso expone los modelos con alias: `fast` y `smart` para chat y `embed` para embeddings (también `rerank`, que no se usa aquí). Con otro proveedor compatible con OpenAI, pon sus nombres de modelo en `MODELO` y `MODELO_EMBEDDING`, por ejemplo `text-embedding-3-small`.

Si cambias `MODELO_EMBEDDING`, vuelve a ejecutar `npm run indexar`: los vectores de modelos distintos no se pueden comparar. El índice HNSW de pgvector admite hasta 2.000 dimensiones; con un modelo de más dimensiones, `npm run indexar` fallará al crear el índice.

## Tres formas de buscar: vector, BM25 e híbrida

| Modo | Cómo busca | Bien en | Mal en |
|---|---|---|---|
| `vector` (por defecto) | Embedding de la pregunta + los más cercanos en pgvector | Sinónimos y paráfrasis ("clave" ↔ "contraseña") | Nombres propios, códigos y cifras exactas |
| `bm25` | Palabras en común, pesadas por lo raras que son (`bm25.ts`) | Palabras exactas: "Nova 8", "Oro", "Bizum" | Sinónimos: si no comparte palabras, no lo encuentra |
| `hibrida` | Las dos (20 candidatos cada una), fusionadas con RRF | Lo mejor de las dos | Es algo más compleja |

**BM25** puntúa cada fragmento por las palabras de la pregunta que contiene. Cada palabra pesa más si se repite en el fragmento (con rendimiento decreciente), si es rara en el corpus ("oro" sale en pocos fragmentos, "pedido" en muchos) y si el fragmento es corto. La fórmula está comentada en `bm25.ts`. Aquí se calcula en memoria para que se vea; en producción se usa un motor que ya lo traiga, como ParadeDB (BM25 dentro de Postgres) o Elasticsearch.

**RRF (Reciprocal Rank Fusion)** junta las dos listas usando posiciones, no puntuaciones, porque coseno y BM25 están en escalas distintas: cada fragmento suma `1 / (60 + posición)` en cada lista. Un fragmento que sale bien en las dos sube arriba.

Para compararlos con el golden set:

```bash
npm run evaluar -- --comparar
```

En la TUI, `/busqueda hibrida` y la pregunta del cliente Oro: el visor enseña en qué puesto salió cada fragmento en cada búsqueda (`vector #4 · bm25 #2`).

## Similitud entre frases

```bash
npm run similitud
npm run similitud -- "¿Hacéis envíos a Portugal?" "Enviamos a todos los países de la Unión Europea"
```

Calcula la similitud coseno con `embed` entre pares de frases parecidas, sin relación y "trampas": ¿cambia mucho el vector la negación ("quiero" frente a "no quiero devolver el pedido")? ¿Separa "banco" del parque y "banco" de cuentas? ¿Junta la misma frase en inglés y en español? Los modelos de embedding suelen fallar con la negación, pero compruébalo con el tuyo. Para cada par enseña también las palabras en común, que es lo único que vería BM25. No necesita la base de datos.

## Cómo de bien funciona: el golden set

`golden.json` tiene 25 preguntas cuya respuesta conocemos:

| Categoría | Preguntas | Qué se espera |
|---|---|---|
| `direct` | 19 | La respuesta está en un solo fragmento |
| `multi-doc` | 3 | Hacen falta dos fragmentos, por ejemplo "Soy cliente Oro, ¿cuántos días tengo para devolver la chaqueta Cumbre?" (Club Pepe + la ficha) |
| `no-answer` | 3 | No está en los documentos: el chatbot tiene que decir que no lo sabe |

Cada pregunta indica los fragmentos que la búsqueda debería traer (`expectedSources`) y los datos que debe contener la respuesta (`expectedFacts`; `"a|b"` acepta cualquiera de los dos).

```bash
npm run evaluar -- --solo-busqueda   # solo la búsqueda: rápido y casi gratis
npm run evaluar                      # búsqueda + respuestas (llama al modelo 25 veces)
npm run evaluar -- -k 5              # compara con más fragmentos
npm run evaluar -- --comparar        # vector, bm25 e híbrida, lado a lado
npm test                             # tests de las métricas, sin red
```

Mide las dos mitades del RAG por separado:

- **La búsqueda:** recall@k (¿cuánto de lo necesario se recuperó?), precision@k (¿cuánto de lo recuperado servía?), hit rate (preguntas con al menos un fragmento correcto) y MRR (1 si el fragmento correcto sale siempre el primero). Se calcula por categoría, porque las preguntas de varios documentos son las que más fallan.
- **La respuesta:** si contiene los datos esperados o, en las preguntas sin respuesta, si reconoce que no lo sabe.

Al final dice qué ha fallado (qué fragmento no llegó y qué respuesta dio) y sale con código 1 si no se llega a los mínimos de `evaluar.ts` (recall del 80 % y 80 % de respuestas correctas). Así se puede usar como test: cada vez que cambies el troceo, el modelo de embedding o k, vuelve a ejecutarlo y compara.

**Límites, a propósito:** comprobar la respuesta buscando palabras clave es rápido y gratuito, pero se equivoca a veces: una respuesta puede contener "15" y ser incorrecta. El siguiente paso es usar un LLM como juez, que lee la respuesta y la referencia y decide si son equivalentes. Y 25 preguntas son pocas: sirven para detectar regresiones, no para dar una cifra de calidad.

## Qué cuesta cada paso

Los dos scripts muestran el coste en euros de cada llamada:

- `indexar` muestra el coste de cada lote de embeddings y el total.
- `chat` muestra, para cada pregunta, el coste de la búsqueda (el embedding de la pregunta), el de la respuesta y la suma.

El coste no se calcula en el código. El servicio del curso (un proxy LiteLLM) informa del coste real de cada petición, en dólares, en la cabecera HTTP `x-litellm-response-cost`. Los scripts la leen con `.withResponse()` de la librería de OpenAI y la pasan a euros con `USD_A_EUR` (0,86 por defecto; ajústalo al cambio del día). Las funciones están en `log.ts`.

Con otro proveedor que no envíe esa cabecera, verás "coste no informado".

## Ver los datos en Postgres

```bash
docker compose exec db psql -U rag -d rag
```

```sql
SELECT source, left(content, 60) FROM chunks LIMIT 5;
SELECT count(*), vector_dims(embedding) FROM chunks GROUP BY 2;
```

Para parar la base de datos: `docker compose down`. Para borrarla también: `docker compose down -v`.

## Cómo se guarda y cómo se busca

- `indexar.ts`: `listDocuments()` → `chunkDocument()` → `embedAll()` → `saveChunks()`. Rehace la tabla `chunks` cada vez (`source`, `content`, `model` y `embedding vector(N)`), así que indexar dos veces no duplica nada. Al final crea un índice HNSW con distancia coseno.
- `rag.ts`: `ask()` llama a `retrieve()`, que según el modo busca en pgvector, con BM25 o con las dos, y después a `buildContext()` y al modelo. La búsqueda por vector usa el operador `<=>` (distancia coseno) de pgvector: `ORDER BY embedding <=> $1 LIMIT 3`. La similitud que se imprime es `1 - distancia`.

## Límites, a propósito

- Cada pregunta es independiente: no hay memoria de la conversación.
- No se reordenan los resultados con un modelo de *reranking* (el servicio del curso tiene uno, `rerank`): es el siguiente paso después de la búsqueda híbrida.
- El troceo por `## ` funciona porque los documentos están bien estructurados.

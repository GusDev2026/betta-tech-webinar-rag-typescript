/**
 * El RAG en sí, compartido por el chat de terminal (chatbot.ts) y la TUI (tui.tsx).
 *
 * ask() hace las tres fases de la consulta para una pregunta:
 *   1. Recuperar, de una de tres formas (SearchMode):
 *      - "vector": embedding de la pregunta + los más cercanos en pgvector.
 *      - "bm25": por palabras clave, con BM25 (bm25.ts).
 *      - "hybrid": las dos, fusionadas con Reciprocal Rank Fusion (RRF).
 *   2. Construir el contexto con los fragmentos y su fuente.
 *   3. Generar la respuesta con el modelo.
 */

import "dotenv/config";
import OpenAI from "openai";
import type { ChatCompletionMessageParam } from "openai/resources/chat/completions";
import { Bm25Index } from "./bm25.ts";
import { db, toVectorLiteral } from "./db.ts";
import { addCosts, costInEuros, formatEuros, previewVector } from "./log.ts";

const openai = new OpenAI(); // lee OPENAI_API_KEY y OPENAI_BASE_URL del entorno

// Alias del servicio del curso: "fast" (rápido y barato) o "smart" (más capaz).
export const CHAT_MODEL = process.env.MODELO ?? "fast";
// Tiene que ser el mismo modelo con el que se indexó.
export const EMBEDDING_MODEL = process.env.MODELO_EMBEDDING ?? "embed";

const SYSTEM_PROMPT = `Eres el asistente de atención al cliente de TiendaPepe, una tienda online.
Responde solo con la información del contexto que te doy, en español y en pocas frases.
Después de cada dato, cita su fuente entre corchetes, por ejemplo [politicas/devoluciones.md > Plazo general].
Si la respuesta no está en el contexto, di que no lo sabes y sugiere escribir a soporte.
No te inventes plazos, precios ni condiciones.`;

const SYSTEM_PROMPT_NO_RAG = "Eres el asistente de atención al cliente de TiendaPepe, una tienda online.";

export type SearchMode = "vector" | "bm25" | "hybrid";

export const SEARCH_MODES: Record<SearchMode, string> = {
  vector: "vector (embeddings)",
  bm25: "BM25 (palabras clave)",
  hybrid: "híbrida (vector + BM25)",
};

/** Para la línea de comandos: si el modo no es válido, avisa y termina sin traza de error. */
export function parseSearchModeOrExit(text: string): SearchMode {
  try {
    return parseSearchMode(text);
  } catch (error) {
    console.error(error instanceof Error ? error.message : error);
    process.exit(1);
  }
}

/** Acepta el modo en inglés o en español: "hybrid", "hibrida" o "híbrida". */
export function parseSearchMode(text: string): SearchMode {
  const mode = text.toLowerCase().normalize("NFD").replace(/\p{Diacritic}/gu, "");
  if (mode === "hibrida") return "hybrid";
  if (mode === "vector" || mode === "bm25" || mode === "hybrid") return mode;
  throw new Error(`Búsqueda desconocida "${text}": usa vector, bm25 o hibrida`);
}

export interface RetrievedChunk {
  source: string;
  content: string;
  /**
   * Depende del modo: similitud coseno (vector, de 0 a 1), puntuación BM25 (sin
   * techo) o puntuación RRF (híbrida, en torno a 0,01-0,03). Solo sirve para ordenar.
   */
  score: number;
  /** Posición del fragmento en cada búsqueda; null si esa búsqueda no lo trajo. */
  vectorRank: number | null;
  keywordRank: number | null;
}

export interface AskOptions {
  useRag: boolean;
  k: number;
  /** Cómo buscar. Por defecto, "vector". */
  mode?: SearchMode;
  /** Dónde escribir los logs de cada paso. Por defecto, en ningún sitio. */
  log?: (line: string) => void;
}

export interface AskResult {
  answer: string;
  chunks: RetrievedChunk[];
  context: string;
  vectorPreview: string;
  tokens: { input: number; output: number };
  /** En euros; null si el servicio no informa del coste. */
  cost: { retrieval: number | null; generation: number | null; total: number | null };
  /** En segundos. */
  seconds: { embedding: number; search: number; generation: number };
}

const secondsSince = (start: number) => (performance.now() - start) / 1000;

/** El embedding de un texto y lo que ha costado calcularlo. */
async function embed(text: string): Promise<{ vector: number[]; cost: number | null }> {
  // withResponse() da también las cabeceras HTTP, donde viene el coste.
  // encoding_format "float": sin esto, la librería pide base64 y el servicio del
  // curso no lo devuelve en el formato que ella espera.
  const { data: response, response: http } = await openai.embeddings
    .create({ model: EMBEDDING_MODEL, input: text, encoding_format: "float" })
    .withResponse();
  return { vector: response.data[0].embedding, cost: costInEuros(http.headers) };
}

interface StoredChunk {
  source: string;
  content: string;
}

/** Los `limit` fragmentos más cercanos a un vector, según pgvector. */
async function vectorSearch(vector: number[], limit: number) {
  // <=> es la distancia coseno de pgvector: 0 = mismo significado.
  // La similitud es 1 - distancia: 1 = mismo significado, 0 = nada que ver.
  const { rows } = await db.query<StoredChunk & { similarity: number }>(
    `SELECT source, content, 1 - (embedding <=> $1) AS similarity
       FROM chunks
      ORDER BY embedding <=> $1
      LIMIT $2`,
    [toVectorLiteral(vector), limit],
  );
  return rows;
}

// El índice BM25 vive en memoria: se construye la primera vez con los fragmentos de Postgres.
let keywordIndex: Bm25Index<StoredChunk> | null = null;

async function keywordSearch(question: string, limit: number) {
  if (!keywordIndex) {
    const { rows } = await db.query<StoredChunk>("SELECT source, content FROM chunks");
    keywordIndex = new Bm25Index(rows, (chunk) => chunk.content);
  }
  return keywordIndex.search(question, limit);
}

// Candidatos que se piden a cada búsqueda antes de fusionarlas.
const HYBRID_CANDIDATES = 20;
// La constante de RRF: suaviza la diferencia entre el primer puesto y los siguientes.
const RRF_K = 60;

/**
 * Reciprocal Rank Fusion: cada fragmento suma 1 / (60 + posición) en cada lista.
 * Usa posiciones, no puntuaciones, porque coseno y BM25 están en escalas distintas.
 */
function reciprocalRankFusion(
  vectorResults: StoredChunk[],
  keywordResults: StoredChunk[],
  k: number,
): RetrievedChunk[] {
  const fused = new Map<string, RetrievedChunk>();
  const add = (chunk: StoredChunk, rank: number, list: "vectorRank" | "keywordRank") => {
    const entry = fused.get(chunk.source) ?? { ...chunk, score: 0, vectorRank: null, keywordRank: null };
    entry[list] = rank;
    entry.score += 1 / (RRF_K + rank);
    fused.set(chunk.source, entry);
  };
  vectorResults.forEach((chunk, i) => add(chunk, i + 1, "vectorRank"));
  keywordResults.forEach((chunk, i) => add(chunk, i + 1, "keywordRank"));
  return [...fused.values()].sort((a, b) => b.score - a.score).slice(0, k);
}

export interface Retrieval {
  chunks: RetrievedChunk[];
  vectorPreview: string;
  /** En euros; null si el servicio no informa del coste. */
  cost: number | null;
  /** En segundos. */
  seconds: { embedding: number; search: number };
}

/** Recuperar: los k fragmentos más relevantes para la pregunta, según el modo. */
export async function retrieve(question: string, k: number, mode: SearchMode = "vector"): Promise<Retrieval> {
  // BM25 no necesita embedding: es gratis y no llama a ningún modelo.
  if (mode === "bm25") {
    const start = performance.now();
    const results = await keywordSearch(question, k);
    return {
      chunks: results.map(({ item, score }, i) => ({ ...item, score, vectorRank: null, keywordRank: i + 1 })),
      vectorPreview: "",
      cost: 0,
      seconds: { embedding: 0, search: secondsSince(start) },
    };
  }

  let start = performance.now();
  const { vector, cost } = await embed(question);
  const embeddingSeconds = secondsSince(start);

  start = performance.now();
  let chunks: RetrievedChunk[];
  if (mode === "vector") {
    const rows = await vectorSearch(vector, k);
    chunks = rows.map(({ similarity, ...chunk }, i) => ({
      ...chunk,
      score: similarity,
      vectorRank: i + 1,
      keywordRank: null,
    }));
  } else {
    const [vectorResults, keywordResults] = await Promise.all([
      vectorSearch(vector, HYBRID_CANDIDATES),
      keywordSearch(question, HYBRID_CANDIDATES),
    ]);
    chunks = reciprocalRankFusion(vectorResults, keywordResults.map((r) => r.item), k);
  }

  return {
    chunks,
    vectorPreview: previewVector(vector),
    cost,
    seconds: { embedding: embeddingSeconds, search: secondsSince(start) },
  };
}

/** "vector #1 · bm25 #3": en qué puesto salió el fragmento en cada búsqueda. */
export function describeRanks(chunk: RetrievedChunk): string {
  return [
    chunk.vectorRank !== null ? `vector #${chunk.vectorRank}` : null,
    chunk.keywordRank !== null ? `bm25 #${chunk.keywordRank}` : null,
  ]
    .filter(Boolean)
    .join(" · ");
}

/** La puntuación con los decimales que tienen sentido en cada modo. */
export const formatScore = (score: number, mode: SearchMode) =>
  score.toFixed(mode === "hybrid" ? 3 : 2);

/** Construir el contexto: cada fragmento con su fuente delante. */
export function buildContext(chunks: RetrievedChunk[]): string {
  return chunks.map((chunk) => `[${chunk.source}]\n${chunk.content}`).join("\n\n");
}

/** Los mensajes para el modelo: con RAG, el contexto va antes de la pregunta. */
function buildMessages(question: string, context: string | null): ChatCompletionMessageParam[] {
  if (context === null) {
    return [
      { role: "system", content: SYSTEM_PROMPT_NO_RAG },
      { role: "user", content: question },
    ];
  }
  return [
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user", content: `Contexto:\n${context}\n\nPregunta: ${question}` },
  ];
}

/** Responde una pregunta. Cada pregunta es independiente: no hay historial. */
export async function ask(question: string, options: AskOptions): Promise<AskResult> {
  const log = options.log ?? (() => {});
  const seconds = { embedding: 0, search: 0, generation: 0 };
  let chunks: RetrievedChunk[] = [];
  let vectorPreview = "";
  let retrievalCost: number | null = 0;

  // 1. Recuperar.
  if (options.useRag) {
    const mode = options.mode ?? "vector";
    const retrieval = await retrieve(question, options.k, mode);
    ({ chunks, vectorPreview } = retrieval);
    retrievalCost = retrieval.cost;
    Object.assign(seconds, retrieval.seconds);
    if (vectorPreview) log(`[embedding] pregunta -> ${vectorPreview}, ${formatEuros(retrievalCost)}`);
    log(`[búsqueda ${SEARCH_MODES[mode]}] ${chunks.length} fragmentos:`);
    for (const chunk of chunks) {
      log(`  ${formatScore(chunk.score, mode)}  ${chunk.source}  (${describeRanks(chunk)})`);
    }
  }

  // 2. Construir el contexto.
  const context = buildContext(chunks);

  // 3. Generar.
  const start = performance.now();
  const { data: response, response: http } = await openai.chat.completions
    .create({
      model: CHAT_MODEL,
      // Si el modelo "razona" antes de responder, su pensamiento también cuenta.
      max_completion_tokens: 1000,
      messages: buildMessages(question, options.useRag ? context : null),
    })
    .withResponse();
  seconds.generation = secondsSince(start);
  const generationCost = costInEuros(http.headers);
  const tokens = {
    input: response.usage?.prompt_tokens ?? 0,
    output: response.usage?.completion_tokens ?? 0,
  };
  const total = addCosts(retrievalCost, generationCost);
  log(
    `[${CHAT_MODEL}] ${tokens.input} tokens de entrada, ${tokens.output} de salida, ` +
      `${formatEuros(generationCost)}`,
  );
  log(`[coste] búsqueda ${formatEuros(retrievalCost)} + respuesta ${formatEuros(generationCost)} = ${formatEuros(total)}`);

  return {
    answer: response.choices[0].message.content ?? "(respuesta vacía)",
    chunks,
    context,
    vectorPreview,
    tokens,
    cost: { retrieval: retrievalCost, generation: generationCost, total },
    seconds,
  };
}

/** Conecta y comprueba que la base de datos está indexada con el mismo modelo. Devuelve cuántos fragmentos hay. */
export async function connect(): Promise<number> {
  try {
    await db.connect();
  } catch {
    throw new Error("No se puede conectar a Postgres: arráncalo con npm run db");
  }
  const { rows } = await db
    .query<{ model: string; total: number }>(
      "SELECT model, count(*)::int AS total FROM chunks GROUP BY model",
    )
    .catch(() => ({ rows: [] }));
  if (rows.length === 0) {
    throw new Error("No hay fragmentos: ejecuta primero npm run indexar");
  }
  if (rows[0].model !== EMBEDDING_MODEL) {
    throw new Error(`El índice se hizo con ${rows[0].model}: vuelve a ejecutar npm run indexar`);
  }
  return rows[0].total;
}

export const disconnect = () => db.end();

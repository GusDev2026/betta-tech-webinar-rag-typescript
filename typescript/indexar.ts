/**
 * Indexación de un RAG mínimo (TypeScript): se hace una vez.
 *
 * Lee los .md de ../docs (y sus subcarpetas), los trocea por secciones, calcula
 * un embedding de cada fragmento y lo guarda en Postgres con pgvector.
 *
 * Requisitos:
 *   npm install
 *   npm run db          (arranca Postgres con pgvector en Docker)
 *   Un archivo .env con tu clave (copia .env.example):
 *     OPENAI_API_KEY=sk-...
 *
 * Ejecutar:
 *   npm run indexar
 */

import "dotenv/config";
import { readdirSync, readFileSync } from "node:fs";
import OpenAI from "openai";
import { db, toVectorLiteral } from "./db.ts";
import { addCosts, costInEuros, elapsed, formatEuros, previewText, previewVector } from "./log.ts";

const openai = new OpenAI(); // lee OPENAI_API_KEY y OPENAI_BASE_URL del entorno

// Modelo de embedding: convierte texto en una lista de números (no genera texto).
const EMBEDDING_MODEL = process.env.MODELO_EMBEDDING ?? "text-embedding-3-small";

const DOCS_DIR = new URL("../docs/", import.meta.url);

// Textos por llamada de embeddings: los proveedores limitan el tamaño de cada petición.
const BATCH_SIZE = 100;

interface Chunk {
  source: string;
  content: string;
}

/** Lista los .md de la carpeta de documentos, con su subcarpeta: "politicas/envios.md". */
function listDocuments(): string[] {
  return readdirSync(DOCS_DIR, { recursive: true, encoding: "utf-8" })
    .filter((path) => path.endsWith(".md"))
    .sort();
}

/** Un fragmento por cada sección '## ' del documento. */
function chunkDocument(path: string): Chunk[] {
  const markdown = readFileSync(new URL(path, DOCS_DIR), "utf-8");
  const [header, ...sections] = markdown.split("\n## ");
  const title = header.trim().replace(/^# /, "");

  return sections.map((section) => {
    const lineBreak = section.indexOf("\n");
    const heading = section.slice(0, lineBreak).trim();
    return {
      // De dónde sale: es lo que el chatbot citará.
      source: `${path} > ${heading}`,
      // El título va dentro del texto para que el embedding sepa de qué
      // documento es ("Plazo general" solo no dice nada).
      content: `${title} - ${heading}\n${section.slice(lineBreak + 1).trim()}`,
    };
  });
}

/** Cuántos documentos y fragmentos hay en cada carpeta. */
function logChunkSummary(documents: string[], chunks: Chunk[]): void {
  console.log(`[trocear] ${documents.length} documentos -> ${chunks.length} fragmentos`);
  const folderOf = (path: string) => path.split("/")[0];
  for (const folder of [...new Set(documents.map(folderOf))]) {
    const docs = documents.filter((path) => folderOf(path) === folder).length;
    const pieces = chunks.filter((chunk) => folderOf(chunk.source) === folder).length;
    console.log(`  ${`${folder}/`.padEnd(12)} ${String(docs).padStart(3)} documentos, ${pieces} fragmentos`);
  }
}

/** Calcula los embeddings por lotes. Devuelve los vectores en el mismo orden. */
async function embedAll(
  chunks: Chunk[],
): Promise<{ vectors: number[][]; tokens: number; cost: number | null }> {
  const vectors: number[][] = [];
  const batches = Math.ceil(chunks.length / BATCH_SIZE);
  let tokens = 0;
  let cost: number | null = 0;
  for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
    const batch = chunks.slice(i, i + BATCH_SIZE);
    const start = performance.now();
    // withResponse() da también las cabeceras HTTP, donde viene el coste.
    const { data: response, response: http } = await openai.embeddings
      .create({
        model: EMBEDDING_MODEL,
        input: batch.map((chunk) => chunk.content),
        // Pedimos los números tal cual. Sin esto, la librería pide base64 y lo
        // descodifica ella, y algunos proxies no lo devuelven en el formato esperado.
        encoding_format: "float",
      })
      .withResponse();
    const batchCost = costInEuros(http.headers);
    cost = addCosts(cost, batchCost);
    vectors.push(...response.data.map((item) => item.embedding));
    tokens += response.usage.total_tokens;

    // Un ejemplo por lote: el primer fragmento y el principio de su vector.
    console.log(
      `\n[embeddings] lote ${i / BATCH_SIZE + 1}/${batches}: ${batch.length} textos, ` +
        `${response.usage.total_tokens} tokens, ${formatEuros(batchCost)}, ${elapsed(start)}`,
    );
    console.log(`  fuente: ${batch[0].source}`);
    console.log(`  texto:  ${previewText(batch[0].content)}`);
    console.log(`  vector: ${previewVector(response.data[0].embedding)}`);
  }
  return { vectors, tokens, cost };
}

/** Rehace la tabla entera: así indexar dos veces no duplica nada. */
async function saveChunks(chunks: Chunk[], vectors: number[][]): Promise<void> {
  const dimensions = vectors[0].length;
  console.log(`\n[pgvector] creando la tabla chunks con embedding vector(${dimensions})`);
  await db.query("CREATE EXTENSION IF NOT EXISTS vector");
  await db.query("DROP TABLE IF EXISTS chunks");
  await db.query(`
    CREATE TABLE chunks (
      id        serial PRIMARY KEY,
      source    text NOT NULL,
      content   text NOT NULL,
      model     text NOT NULL,
      embedding vector(${dimensions}) NOT NULL
    )`);
  let start = performance.now();
  for (const [i, chunk] of chunks.entries()) {
    await db.query(
      "INSERT INTO chunks (source, content, model, embedding) VALUES ($1, $2, $3, $4)",
      [chunk.source, chunk.content, EMBEDDING_MODEL, toVectorLiteral(vectors[i])],
    );
    if ((i + 1) % BATCH_SIZE === 0 || i + 1 === chunks.length) {
      console.log(`  insertadas ${i + 1}/${chunks.length} filas`);
    }
  }
  console.log(`  ${elapsed(start)}`);

  // Índice HNSW: búsqueda aproximada y rápida por distancia coseno.
  // Con 450 filas no se nota; con millones es lo que hace viable la búsqueda.
  start = performance.now();
  await db.query("CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops)");
  console.log(`[pgvector] índice HNSW (distancia coseno) creado en ${elapsed(start)}`);
}

// 1. Cargar y trocear.
const documents = listDocuments();
const chunks = documents.flatMap(chunkDocument);
logChunkSummary(documents, chunks);

// 2. Embeddings.
const { vectors, tokens, cost } = await embedAll(chunks);

// 3. Guardar en Postgres.
await db.connect();
await saveChunks(chunks, vectors);
await db.end();

console.log(
  `\n[listo] ${chunks.length} fragmentos de ${vectors[0].length} dimensiones en la tabla chunks, ` +
    `${tokens} tokens de embeddings, ${formatEuros(cost)}`,
);

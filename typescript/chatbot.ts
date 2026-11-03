/**
 * Chatbot de TiendaPepe con RAG, en la terminal: la consulta, en cada pregunta.
 * La lógica está en rag.ts; aquí solo se leen las preguntas y se imprime.
 * Para una interfaz con el visor de fragmentos al lado: npm run tui.
 *
 * Requisitos:
 *   npm install
 *   El .env de indexar.ts, la base de datos arrancada (npm run db) y haber
 *   ejecutado antes: npm run indexar
 *
 * Ejecutar:
 *   npm run chat                        # chat interactivo
 *   npm run chat -- "¿Cuántos días tengo para devolver unos auriculares?"
 *   npm run chat -- --sin-rag           # el mismo modelo, sin documentos
 *   npm run chat -- --ver-contexto      # enseña los fragmentos que recibe el modelo
 *   npm run chat -- --busqueda bm25     # vector (por defecto), bm25 o hibrida
 */

import { createInterface } from "node:readline";
import { parseArgs } from "node:util";
import { ask, connect, disconnect, parseSearchModeOrExit, SEARCH_MODES } from "./rag.ts";

const { values: options, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    "sin-rag": { type: "boolean", default: false },
    "ver-contexto": { type: "boolean", default: false },
    k: { type: "string", short: "k", default: "3" },
    busqueda: { type: "string", default: "vector" },
  },
});
const mode = parseSearchModeOrExit(options.busqueda);

/** Pregunta, imprime los pasos y devuelve la respuesta. */
async function answer(question: string): Promise<string> {
  const result = await ask(question, {
    useRag: !options["sin-rag"],
    k: Number(options.k),
    mode,
    log: console.log,
  });
  if (options["ver-contexto"] && !options["sin-rag"]) {
    console.log(`\n--- contexto que recibe el modelo ---\n${result.context}\n---`);
  }
  return result.answer;
}

/** Chat interactivo. Cada pregunta es independiente: no se guarda el historial. */
async function runChat(totalChunks: number): Promise<void> {
  const label = options["sin-rag"] ? "sin RAG" : `RAG, ${totalChunks} fragmentos, búsqueda ${SEARCH_MODES[mode]}`;
  console.log(`Chatbot de TiendaPepe (${label}). Escribe 'salir' para terminar.`);

  const terminal = createInterface({ input: process.stdin, output: process.stdout });
  terminal.setPrompt("\nTú: ");
  terminal.prompt();
  for await (const line of terminal) {
    const question = line.trim();
    if (question === "" || question === "salir") break;
    console.log(`\nTiendaPepe: ${await answer(question)}`);
    terminal.prompt();
  }
  terminal.close();
}

try {
  const totalChunks = await connect();
  if (positionals.length > 0) {
    console.log(`\nTiendaPepe: ${await answer(positionals.join(" "))}`);
  } else {
    await runChat(totalChunks);
  }
} catch (error) {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
} finally {
  await disconnect();
}

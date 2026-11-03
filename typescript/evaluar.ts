/**
 * Evaluación del RAG con un golden set: preguntas cuya respuesta conocemos.
 *
 * golden.json tiene, para cada pregunta:
 *   - expectedSources: los fragmentos que la búsqueda debería traer.
 *   - expectedFacts: datos que la respuesta debe contener ("a|b" = vale a o b).
 *   - category: "direct" (un fragmento), "multi-doc" (varios) o "no-answer"
 *     (no está en los documentos: el chatbot debe decir que no lo sabe).
 *
 * Mide las dos mitades del RAG por separado:
 *   1. La búsqueda: precision@k, recall@k, hit rate y MRR. Solo hace embeddings.
 *   2. La respuesta: si contiene los datos esperados o, si no hay respuesta, si
 *      dice que no lo sabe. Llama al modelo de chat en cada pregunta.
 *
 * Ejecutar:
 *   npm run evaluar                      # búsqueda + respuestas
 *   npm run evaluar -- --solo-busqueda   # solo la búsqueda: rápido y casi gratis
 *   npm run evaluar -- -k 5              # con 5 fragmentos por pregunta
 *   npm run evaluar -- --busqueda bm25   # vector (por defecto), bm25 o hibrida
 *   npm run evaluar -- --comparar        # los tres modos de búsqueda, lado a lado
 *
 * Sale con código 1 si no se alcanzan los mínimos: así se puede usar como test.
 */

import { readFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { db } from "./db.ts";
import { addCosts, formatEuros } from "./log.ts";
import { checkAnswer, scoreRetrieval, type GoldenCase } from "./metrics.ts";
import {
  ask,
  connect,
  disconnect,
  parseSearchModeOrExit,
  retrieve,
  SEARCH_MODES,
  type RetrievedChunk,
  type SearchMode,
} from "./rag.ts";

// Mínimos para dar la evaluación por buena. Súbelos a medida que mejore el RAG.
const MIN_RECALL = 0.8;
const MIN_ANSWER_ACCURACY = 0.8;

interface CaseResult {
  golden: GoldenCase;
  retrieved: string[];
  /** null en las preguntas sin respuesta: no hay fragmentos que encontrar. */
  precision: number | null;
  recall: number | null;
  reciprocalRank: number | null;
  /** null con --solo-busqueda. */
  answerOk: boolean | null;
  answer: string;
  cost: number | null;
}

const { values: options } = parseArgs({
  options: {
    "solo-busqueda": { type: "boolean", default: false },
    k: { type: "string", short: "k", default: "3" },
    busqueda: { type: "string", default: "vector" },
    comparar: { type: "boolean", default: false },
  },
});
const k = Number(options.k);
// Comparar solo mide la búsqueda: lo que cambia entre modos es qué fragmentos llegan.
const compare = options.comparar;
const retrievalOnly = options["solo-busqueda"] || compare;
const mode = parseSearchModeOrExit(options.busqueda);

async function evaluateCase(golden: GoldenCase, mode: SearchMode): Promise<CaseResult> {
  let chunks: RetrievedChunk[];
  let answer = "";
  let cost: number | null;

  if (retrievalOnly) {
    ({ chunks, cost } = await retrieve(golden.question, k, mode));
  } else {
    const result = await ask(golden.question, { useRag: true, k, mode });
    ({ chunks, answer } = result);
    cost = result.cost.total;
  }

  const retrieved = chunks.map((chunk) => chunk.source);
  return {
    golden,
    retrieved,
    ...scoreRetrieval(golden.expectedSources, retrieved),
    answerOk: retrievalOnly ? null : checkAnswer(golden, answer),
    answer,
    cost,
  };
}

/** Falla pronto si golden.json cita un fragmento que no está en la base de datos. */
async function checkGoldenSources(cases: GoldenCase[]): Promise<void> {
  const { rows } = await db.query<{ source: string }>("SELECT source FROM chunks");
  const indexed = new Set(rows.map((row) => row.source));
  const unknown = cases.flatMap((c) => c.expectedSources).filter((source) => !indexed.has(source));
  if (unknown.length > 0) {
    throw new Error(`golden.json cita fragmentos que no existen:\n  ${unknown.join("\n  ")}`);
  }
}

const average = (values: (number | null)[]) => {
  const known = values.filter((value): value is number => value !== null);
  return known.length === 0 ? null : known.reduce((sum, value) => sum + value, 0) / known.length;
};
const percent = (value: number | null) => (value === null ? "  -  " : `${(value * 100).toFixed(0).padStart(3)} %`);
const mark = (ok: boolean | null) => (ok === null ? "-" : ok ? "✓" : "✗");

function printRow(index: number, result: CaseResult): void {
  const columns = [
    String(index + 1).padStart(2),
    result.golden.category.padEnd(9),
    percent(result.recall),
    percent(result.precision),
    mark(result.answerOk).padStart(4),
    result.golden.question,
  ];
  console.log(columns.join("  "));
}

function printFailures(results: CaseResult[]): void {
  const failures = results.filter(
    (r) => (r.recall !== null && r.recall < 1) || r.answerOk === false,
  );
  if (failures.length === 0) return;

  console.log("\nQué ha fallado:");
  for (const result of failures) {
    console.log(`\n• ${result.golden.question}`);
    const missing = result.golden.expectedSources.filter((s) => !result.retrieved.includes(s));
    if (missing.length > 0) {
      console.log(`  no recuperado: ${missing.join(", ")}`);
      console.log(`  recuperado:    ${result.retrieved.join(", ")}`);
    }
    if (result.answerOk === false) {
      const expected =
        result.golden.category === "no-answer" ? "que diga que no lo sabe" : result.golden.expectedFacts.join(", ");
      console.log(`  esperado:  ${expected}`);
      console.log(`  respuesta: ${result.answer.replace(/\s+/g, " ").slice(0, 200)}`);
    }
  }
}

/** Las métricas agregadas de una pasada por el golden set. */
function summarize(results: CaseResult[]) {
  const answerable = results.filter((r) => r.golden.category !== "no-answer");
  const recallOf = (category: string) =>
    average(answerable.filter((r) => r.golden.category === category).map((r) => r.recall));
  return {
    recall: average(answerable.map((r) => r.recall)),
    precision: average(answerable.map((r) => r.precision)),
    mrr: average(answerable.map((r) => r.reciprocalRank)),
    hitRate: average(answerable.map((r) => ((r.recall ?? 0) > 0 ? 1 : 0))),
    recallDirect: recallOf("direct"),
    recallMultiDoc: recallOf("multi-doc"),
    answerAccuracy: average(results.map((r) => (r.answerOk === null ? null : r.answerOk ? 1 : 0))),
    cost: addCosts(...results.map((r) => r.cost)),
  };
}

function printSummary(results: CaseResult[]): boolean {
  const { recall, precision, mrr, hitRate, recallDirect, recallMultiDoc, answerAccuracy, cost } =
    summarize(results);

  console.log(`\nResumen (búsqueda ${SEARCH_MODES[mode]}, k = ${k}, ${results.length} preguntas)`);
  console.log(`  Búsqueda   recall@${k}    ${percent(recall)}   ¿cuánto de lo necesario se recuperó?`);
  console.log(`             precision@${k} ${percent(precision)}   ¿cuánto de lo recuperado servía?`);
  console.log(`             hit rate    ${percent(hitRate)}   preguntas con al menos un fragmento correcto`);
  console.log(`             MRR         ${mrr === null ? "  -  " : mrr.toFixed(2).padStart(5)}   1 = el correcto siempre sale primero`);
  console.log(`             recall direct    ${percent(recallDirect)}`);
  console.log(`             recall multi-doc ${percent(recallMultiDoc)}`);
  if (!retrievalOnly) {
    console.log(`  Respuesta  aciertos    ${percent(answerAccuracy)}`);
  }
  console.log(`  Coste      ${formatEuros(cost)}`);

  const passed =
    (recall ?? 0) >= MIN_RECALL && (retrievalOnly || (answerAccuracy ?? 0) >= MIN_ANSWER_ACCURACY);
  console.log(
    `\n${passed ? "✓ APROBADO" : "✗ SUSPENSO"}  (mínimos: recall ${MIN_RECALL * 100} %` +
      `${retrievalOnly ? "" : `, aciertos ${MIN_ANSWER_ACCURACY * 100} %`})`,
  );
  return passed;
}

/** Pasa el golden set con los tres modos de búsqueda y los compara. */
async function compareModes(cases: GoldenCase[]): Promise<void> {
  console.log(`Comparando los tres modos de búsqueda (k = ${k}, ${cases.length} preguntas)...\n`);
  const rows: [SearchMode, ReturnType<typeof summarize>][] = [];
  for (const candidate of Object.keys(SEARCH_MODES) as SearchMode[]) {
    const results: CaseResult[] = [];
    for (const golden of cases) results.push(await evaluateCase(golden, candidate));
    rows.push([candidate, summarize(results)]);
  }

  console.log(`búsqueda   recall@${k}  precision@${k}  hit rate   MRR  recall direct  recall multi-doc  coste`);
  for (const [candidate, m] of rows) {
    console.log(
      [
        candidate.padEnd(8),
        percent(m.recall).padStart(9),
        percent(m.precision).padStart(12),
        percent(m.hitRate).padStart(9),
        (m.mrr?.toFixed(2) ?? "-").padStart(5),
        percent(m.recallDirect).padStart(14),
        percent(m.recallMultiDoc).padStart(17),
        "  " + formatEuros(m.cost),
      ].join(" "),
    );
  }
}

try {
  const cases: GoldenCase[] = JSON.parse(readFileSync(new URL("./golden.json", import.meta.url), "utf-8"));
  await connect();
  await checkGoldenSources(cases);

  if (compare) {
    await compareModes(cases);
  } else {
    console.log(`Evaluando ${cases.length} preguntas${retrievalOnly ? " (solo búsqueda)" : ""}...\n`);
    console.log(" #  categoría  recall  prec.  resp.  pregunta");
    const results: CaseResult[] = [];
    for (const [index, golden] of cases.entries()) {
      const result = await evaluateCase(golden, mode);
      results.push(result);
      printRow(index, result);
    }

    printFailures(results);
    if (!printSummary(results)) process.exitCode = 1;
  }
} catch (error) {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
} finally {
  await disconnect();
}

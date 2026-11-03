/**
 * Similitud entre pares de frases, con el modelo de embedding de verdad.
 * Sirve para ver qué significa "cerca" y "lejos" en el espacio de embeddings.
 *
 * No necesita la base de datos: solo el .env con la clave.
 *
 * Ejecutar:
 *   npm run similitud                                  # los pares de ejemplo
 *   npm run similitud -- "frase uno" "frase dos"       # tu propio par
 */

import "dotenv/config";
import OpenAI from "openai";
import { tokenize } from "./bm25.ts";
import { costInEuros, formatEuros } from "./log.ts";
import { EMBEDDING_MODEL } from "./rag.ts";

const openai = new OpenAI();

type Pair = [string, string];

const EXAMPLES: { title: string; note: string; pairs: Pair[] }[] = [
  {
    title: "Parecidas: mismo significado, casi ninguna palabra en común",
    note: "Aquí ganan los embeddings: BM25 no las relacionaría.",
    pairs: [
      ["¿Cuántos días tengo para devolverlo?", "Plazo de devolución: 30 días naturales"],
      ["¿Me lo podéis traer mañana?", "Entrega garantizada al día siguiente laborable"],
      ["Se me ha olvidado la clave", "Cómo restablecer la contraseña de tu cuenta"],
      ["¿Puedo pagar con el móvil?", "Aceptamos tarjeta, PayPal y Bizum"],
    ],
  },
  {
    title: "Sin relación",
    note: "Deberían quedar claramente por debajo de las parecidas.",
    pairs: [
      ["¿Cuántos días tengo para devolverlo?", "La cafetera tiene un depósito de 1,8 litros"],
      ["Se me ha olvidado la clave", "Zapatillas de trail con suela de tacos de 5 mm"],
      ["¿Me lo podéis traer mañana?", "Filtro HEPA para habitaciones de 40 metros cuadrados"],
    ],
  },
  {
    title: "Trampas: dónde los embeddings se equivocan o sorprenden",
    note: "¿Cambia el vector con la negación? ¿Separa los dos sentidos de \"banco\"? ¿Y otro idioma?",
    pairs: [
      ["Quiero devolver el pedido", "No quiero devolver el pedido"],
      ["Me senté en un banco del parque", "Abrí una cuenta en el banco"],
      ["Return policy: 30 days", "Plazo de devolución: 30 días"],
    ],
  },
];

/** Similitud coseno: 1 = misma dirección (mismo significado), 0 = nada que ver. */
function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

const bar = (value: number) => {
  const filled = Math.round(Math.max(0, Math.min(1, value)) * 20);
  return "█".repeat(filled) + "░".repeat(20 - filled);
};

/** Las palabras que comparten las dos frases: lo único que vería una búsqueda por palabras clave. */
function sharedWords(a: string, b: string): string[] {
  const words = new Set(tokenize(b));
  return [...new Set(tokenize(a))].filter((word) => words.has(word));
}

/** Embeddings de todas las frases en una sola llamada. */
async function embedAll(texts: string[]): Promise<{ vectors: Map<string, number[]>; cost: number | null }> {
  const unique = [...new Set(texts)];
  const { data: response, response: http } = await openai.embeddings
    .create({ model: EMBEDDING_MODEL, input: unique, encoding_format: "float" })
    .withResponse();
  const vectors = new Map(unique.map((text, i) => [text, response.data[i].embedding]));
  return { vectors, cost: costInEuros(http.headers) };
}

function printPair([a, b]: Pair, vectors: Map<string, number[]>): void {
  const similarity = cosineSimilarity(vectors.get(a)!, vectors.get(b)!);
  const shared = sharedWords(a, b);
  console.log(`  ${similarity.toFixed(2)} ${bar(similarity)}  "${a}"`);
  console.log(`  ${" ".repeat(25)}  "${b}"`);
  console.log(
    `  ${" ".repeat(25)}  palabras en común: ${shared.length === 0 ? "ninguna" : shared.join(", ")}\n`,
  );
}

const custom = process.argv.slice(2);
if (custom.length !== 0 && custom.length !== 2) {
  console.error('Uso: npm run similitud -- "frase uno" "frase dos"');
  process.exit(1);
}

const groups = custom.length === 2 ? [{ title: "Tu par", note: "", pairs: [custom as Pair] }] : EXAMPLES;
const { vectors, cost } = await embedAll(groups.flatMap((group) => group.pairs.flat()));

console.log(`Similitud coseno con ${EMBEDDING_MODEL} (${vectors.values().next().value!.length} dimensiones)\n`);
for (const group of groups) {
  console.log(group.title);
  if (group.note) console.log(`  ${group.note}\n`);
  for (const pair of group.pairs) printPair(pair, vectors);
}
console.log("Los valores absolutos dependen del modelo: lo que importa es que las parecidas");
console.log("queden claramente por encima de las que no tienen relación.");
console.log(`\n[coste] ${formatEuros(cost)}`);

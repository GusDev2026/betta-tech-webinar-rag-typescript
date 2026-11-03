/**
 * Métricas de la evaluación (evaluar.ts). Son funciones puras, sin red ni base de
 * datos: se pueden probar por separado.
 */

export type Category = "direct" | "multi-doc" | "no-answer";

export interface GoldenCase {
  category: Category;
  question: string;
  expectedSources: string[];
  expectedFacts: string[];
}

// Frases con las que el chatbot reconoce que no lo sabe (sin tildes ni mayúsculas).
const REFUSAL =
  /no (lo )?se\b|no (tengo|dispongo|encuentro|aparece|consta|hay informacion)|no puedo (confirmar|responder|ayudar)|no (esta|figura) en|sin informacion/;

/** Minúsculas y sin tildes, para comparar textos sin que importe cómo se escriban. */
export const normalize = (text: string) =>
  text.toLowerCase().normalize("NFD").replace(/\p{Diacritic}/gu, "");

/** ¿Aparece la palabra o expresión en el texto, como palabra completa? */
export function containsTerm(text: string, term: string): boolean {
  const escaped = normalize(term).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`(^|[^\\p{L}\\p{N}])${escaped}($|[^\\p{L}\\p{N}])`, "u").test(normalize(text));
}

/** Comprueba la respuesta: los datos esperados o, si no hay respuesta, que lo reconozca. */
export function checkAnswer(golden: GoldenCase, answer: string): boolean {
  if (golden.category === "no-answer") return REFUSAL.test(normalize(answer));
  return golden.expectedFacts.every((fact) =>
    fact.split("|").some((alternative) => containsTerm(answer, alternative)),
  );
}

/** precision@k, recall@k y reciprocal rank de una búsqueda. */
export function scoreRetrieval(expected: string[], retrieved: string[]) {
  if (expected.length === 0) return { precision: null, recall: null, reciprocalRank: null };
  const hits = retrieved.filter((source) => expected.includes(source)).length;
  const firstHit = retrieved.findIndex((source) => expected.includes(source));
  return {
    precision: hits / retrieved.length,
    recall: hits / expected.length,
    reciprocalRank: firstHit === -1 ? 0 : 1 / (firstHit + 1),
  };
}

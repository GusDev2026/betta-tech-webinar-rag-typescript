/**
 * BM25: búsqueda por palabras clave, el algoritmo clásico de Elasticsearch o Lucene.
 * Lo implementamos aquí, en memoria, para que se vea la fórmula. En producción
 * usarías un motor que ya lo traiga (ParadeDB dentro de Postgres, Elasticsearch...).
 *
 * Para cada palabra de la pregunta que aparece en un fragmento, suma:
 *
 *   IDF(palabra) · tf · (K1 + 1) / (tf + K1 · (1 - B + B · longitud / longitudMedia))
 *
 *   - tf: cuántas veces aparece la palabra en el fragmento (con rendimiento decreciente).
 *   - IDF: lo rara que es la palabra en el corpus. "nova" pesa mucho; "pedido", poco.
 *   - longitud: un fragmento corto con la palabra puntúa más que uno largo.
 */

// Valores habituales de los dos parámetros de BM25.
const K1 = 1.2; // cuánto premia repetir una palabra antes de saturarse
const B = 0.75; // cuánto penaliza la longitud del fragmento (0 = nada, 1 = del todo)

// Palabras tan comunes en español que no ayudan a buscar.
const STOPWORDS = new Set(
  ("a al algo como con cual cuando de del desde donde el ella en entre es esta este esto " +
    "ha hay la las le lo los me mi mas muy no o para pero por que se si sin sobre su sus " +
    "te tu un una uno unos y ya yo puedo tengo teneis hace").split(" "),
);

/** Minúsculas, sin tildes, partido en palabras y sin las palabras vacías. */
export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .normalize("NFD")
    .replace(/\p{Diacritic}/gu, "")
    .split(/[^\p{L}\p{N}]+/u)
    .filter((word) => word.length > 1 && !STOPWORDS.has(word));
}

export interface Bm25Result<T> {
  item: T;
  score: number;
}

export class Bm25Index<T> {
  private readonly termCounts: Map<string, number>[];
  private readonly lengths: number[];
  private readonly averageLength: number;
  /** En cuántos fragmentos aparece cada palabra. */
  private readonly documentFrequency = new Map<string, number>();

  constructor(
    private readonly items: T[],
    getText: (item: T) => string,
  ) {
    this.termCounts = items.map((item) => {
      const counts = new Map<string, number>();
      for (const term of tokenize(getText(item))) counts.set(term, (counts.get(term) ?? 0) + 1);
      return counts;
    });
    this.lengths = this.termCounts.map((counts) => [...counts.values()].reduce((a, b) => a + b, 0));
    this.averageLength = this.lengths.reduce((a, b) => a + b, 0) / items.length;
    for (const counts of this.termCounts) {
      for (const term of counts.keys()) {
        this.documentFrequency.set(term, (this.documentFrequency.get(term) ?? 0) + 1);
      }
    }
  }

  /** Lo rara que es una palabra: alta si aparece en pocos fragmentos. */
  private idf(term: string): number {
    const n = this.documentFrequency.get(term) ?? 0;
    return Math.log(1 + (this.items.length - n + 0.5) / (n + 0.5));
  }

  /** Los `limit` fragmentos con más puntuación. Los que no comparten ninguna palabra no salen. */
  search(query: string, limit: number): Bm25Result<T>[] {
    const terms = [...new Set(tokenize(query))];
    return this.items
      .map((item, i) => {
        let score = 0;
        for (const term of terms) {
          const tf = this.termCounts[i].get(term) ?? 0;
          if (tf === 0) continue;
          const lengthNorm = 1 - B + (B * this.lengths[i]) / this.averageLength;
          score += (this.idf(term) * tf * (K1 + 1)) / (tf + K1 * lengthNorm);
        }
        return { item, score };
      })
      .filter((result) => result.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);
  }
}

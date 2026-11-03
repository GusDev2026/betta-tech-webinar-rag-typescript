/**
 * Ayudas para enseñar por consola qué está pasando, sin volcar todo.
 */

/** Los primeros números de un vector: "[0.0132, -0.0241, ..., 0.0045] (1536 números)". */
export function previewVector(vector: number[], count = 5): string {
  const head = vector.slice(0, count).map((n) => n.toFixed(4)).join(", ");
  const last = vector.at(-1)?.toFixed(4);
  return `[${head}, ..., ${last}] (${vector.length} números)`;
}

/** El principio de un texto, en una sola línea. */
export function previewText(text: string, length = 70): string {
  const line = text.replace(/\s+/g, " ").trim();
  return line.length > length ? `"${line.slice(0, length)}..."` : `"${line}"`;
}

/** Segundos desde un instante, con un decimal: "0.8 s". */
export const elapsed = (start: number): string => `${((performance.now() - start) / 1000).toFixed(1)} s`;

// Los proxies LiteLLM informan del coste de cada petición, en dólares, en la
// cabecera x-litellm-response-cost. Lo pasamos a euros con este tipo de cambio.
// La API de OpenAI no envía esa cabecera: entonces se muestra "coste no informado".
const USD_TO_EUR = Number(process.env.USD_A_EUR ?? "0.86");

/** Coste en euros de una petición, o null si el servicio no lo informa. */
export function costInEuros(headers: Headers): number | null {
  const usd = headers.get("x-litellm-response-cost");
  return usd === null ? null : Number(usd) * USD_TO_EUR;
}

/** Suma costes; si alguno es desconocido, el total también lo es. */
export const addCosts = (...costs: (number | null)[]): number | null =>
  costs.some((cost) => cost === null) ? null : costs.reduce<number>((sum, cost) => sum + cost!, 0);

/** "0,000123 €", con los decimales necesarios para que no salga 0,00 €. */
export function formatEuros(euros: number | null): string {
  if (euros === null) return "coste no informado";
  const decimals = euros === 0 || euros >= 0.01 ? 2 : Math.min(8, 2 - Math.floor(Math.log10(euros)));
  return `${euros.toFixed(decimals).replace(".", ",")} €`;
}

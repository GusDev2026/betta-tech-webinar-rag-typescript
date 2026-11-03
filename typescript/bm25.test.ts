/**
 * Tests de BM25, con un corpus de juguete. Ejecutar: npm test
 */

import assert from "node:assert/strict";
import { describe, test } from "node:test";
import { Bm25Index, tokenize } from "./bm25.ts";

const docs = [
  "Nivel Oro: los clientes Oro tienen 45 días para devolver un pedido.",
  "Plazo general: tienes 30 días para devolver un pedido.",
  "Envíos a Canarias: tardan de 5 a 8 días laborables.",
  "Nova 8: móvil Android con batería de 5.000 mAh.",
];
const index = new Bm25Index(docs, (doc) => doc);
const top = (query: string) => index.search(query, 1)[0]?.item;

describe("tokenize", () => {
  test("minúsculas, sin tildes y sin palabras vacías", () => {
    assert.deepEqual(tokenize("¿Cuántos días tengo para devolver el pedido?"), ["cuantos", "dias", "devolver", "pedido"]);
  });
});

describe("Bm25Index", () => {
  test("una palabra rara pesa más que una común", () => {
    // "devolver" y "pedido" salen en dos documentos; "oro", solo en uno.
    assert.equal(top("devolver pedido cliente oro"), docs[0]);
  });

  test("encuentra nombres propios y cifras exactas", () => {
    assert.equal(top("nova 8"), docs[3]);
    assert.equal(top("canarias"), docs[2]);
  });

  test("no entiende sinónimos: sin palabras en común, no hay resultado", () => {
    assert.deepEqual(index.search("plazo para reembolsar la compra", 4).map((r) => r.item), [docs[1]]); // solo por "plazo"
    assert.deepEqual(index.search("teléfono inteligente", 4), []);
  });
});

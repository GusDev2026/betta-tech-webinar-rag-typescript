/**
 * Tests de las métricas: comprueban que el corrector y las fórmulas hacen lo
 * que dicen, sin llamar a ningún modelo. Ejecutar: npm test
 */

import assert from "node:assert/strict";
import { describe, test } from "node:test";
import { checkAnswer, scoreRetrieval, type GoldenCase } from "./metrics.ts";

const golden = (category: GoldenCase["category"], expectedFacts: string[] = []): GoldenCase => ({
  category,
  question: "",
  expectedSources: [],
  expectedFacts,
});

describe("scoreRetrieval", () => {
  test("el ejemplo del cliente Oro: 1 de 2 necesarios entre 3 recuperados", () => {
    const score = scoreRetrieval(
      ["politicas/club-pepe.md > Nivel Oro", "productos/chaqueta-cumbre-impermeable.md > Devolución y garantía"],
      ["politicas/cookies.md > Cuánto duran", "politicas/club-pepe.md > Nivel Oro", "ayuda/devolver-regalo.md > Plazo"],
    );
    assert.equal(score.precision, 1 / 3);
    assert.equal(score.recall, 1 / 2);
    assert.equal(score.reciprocalRank, 1 / 2); // el primer acierto sale segundo
  });

  test("sin aciertos: todo a 0", () => {
    assert.deepEqual(scoreRetrieval(["a"], ["b", "c"]), { precision: 0, recall: 0, reciprocalRank: 0 });
  });

  test("sin fragmentos esperados (preguntas sin respuesta): no se puntúa", () => {
    assert.deepEqual(scoreRetrieval([], ["a"]), { precision: null, recall: null, reciprocalRank: null });
  });
});

describe("checkAnswer", () => {
  test("encuentra el dato aunque cambien tildes, mayúsculas o vaya con una cita", () => {
    assert.ok(checkAnswer(golden("direct", ["9,95"]), "Cuesta 9,95 € [politicas/envios.md > Baleares y Canarias]."));
    assert.ok(checkAnswer(golden("direct", ["Tab 11"]), "Solo funciona con la NOVA TAB 11."));
  });

  test("acepta alternativas separadas por |", () => {
    assert.ok(checkAnswer(golden("direct", ["2 años|dos años"]), "Es válida durante dos años."));
  });

  test("compara palabras completas: 10 no es 100", () => {
    assert.equal(checkAnswer(golden("direct", ["10"]), "Tarda un máximo de 100 días."), false);
    assert.ok(checkAnswer(golden("direct", ["10"]), "Tarda un máximo de 10 días hábiles."));
  });

  test("exige todos los datos esperados", () => {
    assert.equal(checkAnswer(golden("direct", ["15", "precinto"]), "Tienes 15 días."), false);
  });

  test("sin respuesta: aprueba si reconoce que no lo sabe", () => {
    assert.ok(checkAnswer(golden("no-answer"), "No lo sé: no tengo información sobre eso."));
    assert.ok(checkAnswer(golden("no-answer"), "No dispongo de esa información. Escribe a soporte."));
  });

  test("sin respuesta: suspende si se inventa algo", () => {
    assert.equal(checkAnswer(golden("no-answer"), "Sí, vendemos bicicletas de montaña desde 300 €."), false);
  });
});

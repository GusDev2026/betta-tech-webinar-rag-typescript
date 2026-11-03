/**
 * Conexión a Postgres + pgvector, compartida por indexar.ts y chatbot.ts.
 * La base de datos se arranca con: npm run db (docker compose up).
 */

import "dotenv/config";
import pg from "pg";

export const db = new pg.Client({
  connectionString: process.env.DATABASE_URL ?? "postgres://rag:rag@localhost:5433/rag",
});

/** pgvector recibe los vectores como texto: "[0.01,-0.02,...]". */
export const toVectorLiteral = (vector: number[]): string => JSON.stringify(vector);

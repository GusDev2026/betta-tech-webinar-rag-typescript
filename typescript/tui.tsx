/**
 * Chatbot de TiendaPepe con RAG, en una interfaz de terminal (TUI) hecha con Ink.
 * A la izquierda, el chat; a la derecha, los fragmentos que ha recuperado pgvector
 * para la última pregunta. La lógica del RAG está en rag.ts.
 *
 * Requisitos: los mismos que chatbot.ts (npm run db y npm run indexar).
 *
 * Ejecutar:
 *   npm run tui
 *
 * Comandos dentro de la TUI: /rag (activar o desactivar RAG), /k 5 (fragmentos
 * a recuperar), /busqueda bm25 (vector, bm25 o hibrida), /salir. También sale con Ctrl+C.
 */

import { Box, render, Text, useApp } from "ink";
import TextInput from "ink-text-input";
import { useState } from "react";
import { formatEuros } from "./log.ts";
import {
  ask,
  CHAT_MODEL,
  connect,
  describeRanks,
  disconnect,
  EMBEDDING_MODEL,
  formatScore,
  parseSearchMode,
  SEARCH_MODES,
  type AskResult,
  type SearchMode,
} from "./rag.ts";

interface Message {
  role: "user" | "assistant" | "system";
  text: string;
  meta?: string;
}

const EXAMPLES = [
  "¿Cuántos días tengo para devolver unos auriculares?",
  "Soy cliente Oro, ¿cuántos días tengo para devolver una chaqueta?",
  "¿La batería externa se puede llevar en el avión?",
  "¿Vendéis bicicletas?",
];

/** Una barra de 10 bloques: "███████░░░". `ratio` va de 0 a 1. */
const scoreBar = (ratio: number) => {
  const filled = Math.round(Math.max(0, Math.min(1, ratio)) * 10);
  return "█".repeat(filled) + "░".repeat(10 - filled);
};

const truncate = (text: string, length: number) => {
  const line = text.replace(/\s+/g, " ").trim();
  return line.length > length ? `${line.slice(0, length)}...` : line;
};

/** Pinta las citas [fuente] de la respuesta en otro color. */
function WithCitations({ text }: { text: string }) {
  const parts = text.split(/(\[[^\]]+\])/g);
  return (
    <Text>
      {parts.map((part, i) =>
        part.startsWith("[") ? (
          <Text key={i} color="cyan">
            {part}
          </Text>
        ) : (
          part
        ),
      )}
    </Text>
  );
}

function ChatPanel({ messages, thinking }: { messages: Message[]; thinking: boolean }) {
  return (
    <Box
      flexDirection="column"
      justifyContent="flex-end"
      overflow="hidden"
      width="50%"
      borderStyle="round"
      borderColor="gray"
      paddingX={1}
    >
      {messages.map((message, i) => (
        <Box key={i} flexDirection="column" marginBottom={1}>
          {message.role === "user" && <Text bold color="yellow">Tú: {message.text}</Text>}
          {message.role === "assistant" && (
            <>
              <Text bold color="green">TiendaPepe:</Text>
              <WithCitations text={message.text} />
              {message.meta && <Text dimColor>{message.meta}</Text>}
            </>
          )}
          {message.role === "system" && <Text color="magenta">{message.text}</Text>}
        </Box>
      ))}
      {thinking && <Text dimColor>Buscando y generando...</Text>}
    </Box>
  );
}

function ChunksPanel({ result, useRag, mode }: { result: AskResult | null; useRag: boolean; mode: SearchMode }) {
  // En vector, la barra es la similitud (0-1). En BM25 e híbrida no hay techo:
  // la barra es relativa al primer fragmento.
  const topScore = result?.chunks[0]?.score ?? 1;
  const ratio = (score: number) => (mode === "vector" ? score : score / topScore);
  return (
    <Box flexDirection="column" overflow="hidden" width="50%" borderStyle="round" borderColor="blue" paddingX={1}>
      <Text bold color="blue">
        Fragmentos recuperados
      </Text>

      {!result && (
        <Box flexDirection="column" marginTop={1}>
          <Text dimColor>Aquí verás qué fragmentos encuentra pgvector para cada pregunta.</Text>
          <Text dimColor>Prueba con:</Text>
          {EXAMPLES.map((example) => (
            <Text key={example} dimColor>
              {"  "}• {example}
            </Text>
          ))}
        </Box>
      )}

      {result && result.chunks.length === 0 && (
        <Box marginTop={1}>
          <Text color="magenta">Sin RAG: el modelo no ha recibido ningún fragmento.</Text>
        </Box>
      )}

      {result && result.chunks.length > 0 && (
        <Box flexDirection="column">
          <Text dimColor>
            embedding {result.seconds.embedding.toFixed(1)} s · búsqueda {result.seconds.search.toFixed(2)} s ·{" "}
            {CHAT_MODEL} {result.seconds.generation.toFixed(1)} s
          </Text>
          <Text dimColor>búsqueda {SEARCH_MODES[mode]}</Text>
          {result.vectorPreview && (
            <Text dimColor wrap="truncate-end">
              pregunta → {result.vectorPreview}
            </Text>
          )}
          {result.chunks.map((chunk, i) => (
            <Box key={chunk.source} flexDirection="column" marginTop={1}>
              <Text wrap="truncate-end">
                <Text bold>{i + 1}</Text> <Text color="green">{scoreBar(ratio(chunk.score))}</Text>{" "}
                <Text bold>{formatScore(chunk.score, mode)}</Text> <Text color="cyan">{chunk.source}</Text>
              </Text>
              {mode === "hybrid" && <Text color="yellow">  {describeRanks(chunk)}</Text>}
              <Text dimColor>{truncate(chunk.content.split("\n").slice(1).join(" "), 180)}</Text>
            </Box>
          ))}
        </Box>
      )}

      {!useRag && !result && (
        <Box marginTop={1}>
          <Text color="magenta">RAG desactivado: las preguntas irán sin documentos.</Text>
        </Box>
      )}
    </Box>
  );
}

function App({ totalChunks }: { totalChunks: number }) {
  const { exit } = useApp();
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [result, setResult] = useState<AskResult | null>(null);
  const [thinking, setThinking] = useState(false);
  const [useRag, setUseRag] = useState(true);
  const [k, setK] = useState(3);
  const [mode, setMode] = useState<SearchMode>("vector");

  const addMessage = (message: Message) => setMessages((previous) => [...previous, message]);

  async function handleSubmit(value: string) {
    const question = value.trim();
    setInput("");
    if (question === "" || thinking) return;

    // Comandos.
    if (question === "/salir") return exit();
    if (question === "/rag") {
      setUseRag(!useRag);
      return addMessage({ role: "system", text: `RAG ${useRag ? "desactivado" : "activado"}` });
    }
    if (question.startsWith("/busqueda")) {
      try {
        const next = parseSearchMode(question.split(" ")[1] ?? "");
        setMode(next);
        return addMessage({ role: "system", text: `Búsqueda ${SEARCH_MODES[next]}` });
      } catch (error) {
        return addMessage({ role: "system", text: error instanceof Error ? error.message : String(error) });
      }
    }
    if (question.startsWith("/k")) {
      const value = Number(question.split(" ")[1]);
      if (!Number.isInteger(value) || value < 1 || value > 10) {
        return addMessage({ role: "system", text: "Uso: /k <número entre 1 y 10>" });
      }
      setK(value);
      return addMessage({ role: "system", text: `Se recuperarán ${value} fragmentos` });
    }

    addMessage({ role: "user", text: question });
    setThinking(true);
    try {
      const answer = await ask(question, { useRag, k, mode });
      setResult(answer);
      addMessage({
        role: "assistant",
        text: answer.answer,
        meta:
          `${CHAT_MODEL} · ${answer.tokens.input} → ${answer.tokens.output} tokens · ` +
          `${formatEuros(answer.cost.total)}${useRag ? "" : " · sin RAG"}`,
      });
    } catch (error) {
      addMessage({ role: "system", text: `Error: ${error instanceof Error ? error.message : error}` });
    } finally {
      setThinking(false);
    }
  }

  return (
    <Box flexDirection="column" height={(process.stdout.rows ?? 30) - 1}>
      <Box justifyContent="space-between" paddingX={1}>
        <Text bold>
          TiendaPepe · RAG con pgvector <Text dimColor>({totalChunks} fragmentos, {EMBEDDING_MODEL} + {CHAT_MODEL})</Text>
        </Text>
        <Text>
          RAG <Text color={useRag ? "green" : "magenta"}>{useRag ? "activado" : "desactivado"}</Text> · {mode} · k={k}
        </Text>
      </Box>

      <Box flexGrow={1}>
        <ChatPanel messages={messages} thinking={thinking} />
        <ChunksPanel result={result} useRag={useRag} mode={mode} />
      </Box>

      <Box borderStyle="round" borderColor={thinking ? "gray" : "yellow"} paddingX={1}>
        <Text color="yellow">› </Text>
        <TextInput
          value={input}
          onChange={setInput}
          onSubmit={handleSubmit}
          placeholder="Escribe una pregunta sobre TiendaPepe"
        />
      </Box>
      <Text dimColor>
        {"  "}Enter: preguntar · /rag: activar o desactivar · /busqueda vector|bm25|hibrida · /k 5 · /salir
      </Text>
    </Box>
  );
}

try {
  const totalChunks = await connect();
  const app = render(<App totalChunks={totalChunks} />);
  await app.waitUntilExit();
} catch (error) {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
} finally {
  await disconnect();
}

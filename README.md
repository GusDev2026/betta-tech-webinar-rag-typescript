# RAG de TiendaPepe

Ejemplo de RAG sin frameworks, en TypeScript. Usa la API de OpenAI (o cualquier proveedor compatible).

- [`docs/`](docs): el corpus, 103 documentos Markdown de una tienda online inventada (políticas, ayuda, fichas de producto, guías y empresa).
- [`typescript/`](typescript): la indexación con embeddings en Postgres + pgvector, la búsqueda vectorial, BM25 e híbrida, un chatbot en terminal (CLI y TUI), la evaluación con un golden set y un script de similitud entre frases.

Para empezar, sigue el [README de `typescript/`](typescript/README.md).

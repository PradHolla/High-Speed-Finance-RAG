import { useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

function formatPages(chunk) {
  const pages = chunk.page_numbers || [];
  if (!pages.length) return "p.?";
  if (pages.length === 1) return `p.${pages[0]}`;
  return `p.${pages[0]}-${pages[pages.length - 1]}`;
}

function formatScore(score) {
  if (typeof score !== "number") return "-";
  return score.toFixed(4);
}

function dedupeChunks(chunks) {
  const seen = new Set();
  const output = [];
  for (const chunk of chunks || []) {
    const key = `${chunk.company}|${chunk.filing_year}|${chunk.filing_type}|${(chunk.text || "").slice(0, 240)}`;
    if (seen.has(key)) continue;
    seen.add(key);
    output.push(chunk);
  }
  return output;
}

function tokenizeBold(text) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, idx) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={idx}>{part.slice(2, -2)}</strong>;
    }
    return <span key={idx}>{part}</span>;
  });
}

function renderAssistantContent(content) {
  const lines = (content || "").split("\n").filter((line) => line.trim().length > 0);
  const listLike = lines.some((line) => line.trim().startsWith("*"));
  if (!listLike) {
    return <p className="message-content">{tokenizeBold(content || "")}</p>;
  }

  return (
    <div className="message-content">
      {lines.map((line, idx) => {
        const trimmed = line.trim();
        if (trimmed.startsWith("*")) {
          return (
            <div key={idx} className="answer-bullet">
              <span className="bullet-dot">•</span>
              <span>{tokenizeBold(trimmed.replace(/^\*\s*/, ""))}</span>
            </div>
          );
        }
        return (
          <p key={idx} className="answer-line">
            {tokenizeBold(trimmed)}
          </p>
        );
      })}
    </div>
  );
}

export default function App() {
  const [query, setQuery] = useState("");
  const [strategy, setStrategy] = useState("standard");
  const [company, setCompany] = useState("Microsoft");
  const [filingYear, setFilingYear] = useState("2025");
  const [filingType, setFilingType] = useState("10-K");
  const [topK, setTopK] = useState(5);
  const [messages, setMessages] = useState([]);
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState("");

  const [file, setFile] = useState(null);
  const [ingestLoading, setIngestLoading] = useState(false);
  const [ingestStatus, setIngestStatus] = useState("");
  const [openEvidence, setOpenEvidence] = useState({});
  const [expandedChunkText, setExpandedChunkText] = useState({});

  async function submitChat(event) {
    event.preventDefault();
    if (!query.trim()) return;

    const userMessage = { role: "user", content: query.trim() };
    setMessages((prev) => [...prev, userMessage]);
    setChatLoading(true);
    setChatError("");

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query.trim(),
          strategy,
          top_k: Number(topK),
          filters: {
            company,
            filing_year: filingYear,
            filing_type: filingType
          }
        })
      });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Chat request failed.");
      }

      const payload = await response.json();
      const assistantMessage = {
        role: "assistant",
        content: payload.answer,
        retrieved_chunks: payload.retrieved_chunks || []
      };
      setMessages((prev) => [...prev, assistantMessage]);
      setQuery("");
    } catch (err) {
      setChatError(err.message || "Unknown chat error.");
    } finally {
      setChatLoading(false);
    }
  }

  async function submitIngest(event) {
    event.preventDefault();
    if (!file) {
      setIngestStatus("Select a PDF file first.");
      return;
    }

    setIngestLoading(true);
    setIngestStatus("");
    try {
      const body = new FormData();
      body.append("file", file);
      body.append("company", company);
      body.append("filing_year", filingYear);
      body.append("filing_type", filingType);
      body.append("collection_name", "financial_reports");

      const response = await fetch(`${API_BASE}/ingest`, {
        method: "POST",
        body
      });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Ingestion failed.");
      }
      const payload = await response.json();
      setIngestStatus(`Ingestion complete: ${payload.source_file}`);
    } catch (err) {
      setIngestStatus(`Ingestion error: ${err.message || "Unknown error"}`);
    } finally {
      setIngestLoading(false);
    }
  }

  function toggleEvidence(index) {
    setOpenEvidence((prev) => ({ ...prev, [index]: !prev[index] }));
  }

  function toggleChunkText(key) {
    setExpandedChunkText((prev) => ({ ...prev, [key]: !prev[key] }));
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>Financial Filing RAG Chat</h1>
        <p>Chat with indexed SEC filing content and inspect retrieval evidence.</p>
      </header>

      <section className="control-grid">
        <form className="card" onSubmit={submitIngest}>
          <h2>Ingest One PDF</h2>
          <label>
            PDF file
            <input
              type="file"
              accept=".pdf"
              onChange={(event) => setFile(event.target.files?.[0] || null)}
            />
          </label>
          <div className="row">
            <label>
              Company
              <input value={company} onChange={(event) => setCompany(event.target.value)} />
            </label>
            <label>
              Filing Year
              <input value={filingYear} onChange={(event) => setFilingYear(event.target.value)} />
            </label>
            <label>
              Filing Type
              <select value={filingType} onChange={(event) => setFilingType(event.target.value)}>
                <option value="10-K">10-K</option>
                <option value="10-Q">10-Q</option>
                <option value="8-K">8-K</option>
                <option value="Other">Other (still PDF-based)</option>
              </select>
            </label>
          </div>
          <p className="helper-line">
            Supported filing types: SEC filing PDFs such as 10-K, 10-Q, and 8-K (plus other PDF filings with the
            same ingestion flow).
          </p>
          <button disabled={ingestLoading} type="submit">
            {ingestLoading ? "Ingesting..." : "Ingest Document"}
          </button>
          {ingestStatus ? <p className="status-line">{ingestStatus}</p> : null}
        </form>

        <form className="card" onSubmit={submitChat}>
          <h2>Ask a Question</h2>
          <div className="row">
            <label>
              Strategy
              <select value={strategy} onChange={(event) => setStrategy(event.target.value)}>
                <option value="standard">standard</option>
                <option value="comparison">comparison</option>
                <option value="extraction">extraction</option>
              </select>
            </label>
            <label>
              Top-K
              <input
                type="number"
                min="1"
                max="20"
                value={topK}
                onChange={(event) => setTopK(event.target.value)}
              />
            </label>
          </div>
          <p className="helper-line">
            Strategy controls answer style: <code>standard</code> for grounded QA, <code>comparison</code> for
            side-by-side analysis, and <code>extraction</code> for strict JSON output.
          </p>
          <label>
            Question
            <textarea
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Ask a financial question..."
              rows={4}
            />
          </label>
          <button disabled={chatLoading} type="submit">
            {chatLoading ? "Generating..." : "Send"}
          </button>
          {chatError ? <p className="error-line">{chatError}</p> : null}
        </form>
      </section>

      <section className="chat-card">
        <h2>Conversation</h2>
        <div className="chat-feed">
          {messages.length === 0 ? (
            <p className="empty-line">No messages yet. Ingest a file and ask a question.</p>
          ) : null}
          {messages.map((message, index) => (
            <article key={index} className={`message message-${message.role}`}>
              <div className="message-role">{message.role === "user" ? "You" : "Assistant"}</div>
              {message.role === "assistant"
                ? renderAssistantContent(message.content)
                : <p className="message-content">{message.content}</p>}
              {message.role === "assistant" && message.retrieved_chunks?.length ? (
                <div className="evidence-block">
                  <button
                    type="button"
                    className="link-button"
                    onClick={() => toggleEvidence(index)}
                  >
                    {openEvidence[index] ? "Hide evidence" : "Show evidence"}
                  </button>
                  {openEvidence[index] ? (
                    <div className="evidence-list">
                      {(() => {
                        const deduped = dedupeChunks(message.retrieved_chunks);
                        return (
                          <p className="evidence-summary">
                            Showing {deduped.length} unique chunks (from {message.retrieved_chunks.length} retrieved).
                          </p>
                        );
                      })()}
                      {dedupeChunks(message.retrieved_chunks).map((chunk, chunkIdx) => {
                        const previewKey = `${index}-${chunk.chunk_id || chunkIdx}`;
                        const text = chunk.text || "";
                        const isLong = text.length > 480;
                        const expanded = !!expandedChunkText[previewKey];
                        const visibleText = expanded || !isLong ? text : `${text.slice(0, 480)}...`;
                        return (
                        <div className="evidence-item" key={`${chunk.chunk_id || "chunk"}-${chunkIdx}`}>
                          <div className="evidence-meta">
                            <span className="meta-pill">rank {chunk.rank}</span>
                            <span className="meta-pill">score {formatScore(chunk.score)}</span>
                            <span className="meta-pill">
                              {chunk.company} {chunk.filing_year} {chunk.filing_type} {formatPages(chunk)}
                            </span>
                          </div>
                          <p>{visibleText}</p>
                          {isLong ? (
                            <button
                              type="button"
                              className="link-button"
                              onClick={() => toggleChunkText(previewKey)}
                            >
                              {expanded ? "Show less text" : "Show full text"}
                            </button>
                          ) : null}
                        </div>
                      )})}
                    </div>
                  ) : null}
                </div>
              ) : null}
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}

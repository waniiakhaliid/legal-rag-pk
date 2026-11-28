import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import "./LegalChat.css";

// ---- Change these ----
const APP_NAME = "Mizaan";
const TAGLINE = "Pakistani law, made clear";
const API_URL = "http://localhost:8000/chat"; // FastAPI /chat (SSE stream)
// ----------------------

const SUGGESTIONS = [
  "Is it a crime to stand in groups in public places?",
  "What is the punishment for rash driving?",
  "قانون میں چوری کی سزا کیا ہے؟",
];

function Scales() {
  return (
    <svg className="scales" viewBox="0 0 64 64" fill="none" aria-hidden="true">
      <path className="draw d1" d="M32 8v44" />
      <path className="draw d2" d="M18 52h28" />
      <path className="draw d3" d="M12 18h40" />
      <path className="draw d4" d="M12 18l-7 16a7 7 0 0 0 14 0l-7-16z" />
      <path className="draw d5" d="M52 18l-7 16a7 7 0 0 0 14 0l-7-16z" />
    </svg>
  );
}

function Splash({ exiting }) {
  return (
    <div className={`splash ${exiting ? "splash-exit" : ""}`}>
      <div className="splash-inner">
        <Scales />
        <svg className="write" viewBox="0 0 520 120" role="img" aria-label={APP_NAME}>
          <text x="50%" y="92" textAnchor="middle" className="write-text">
            {APP_NAME}
          </text>
        </svg>
        <p className="splash-urdu" lang="ur">میزان</p>
        <p className="splash-tag">{TAGLINE}</p>
      </div>
    </div>
  );
}

function TypingDots() {
  return (
    <div className="msg bot">
      <div className="bubble typing">
        <i /><i /><i />
      </div>
    </div>
  );
}

// public_sources(hits) shape isn't known here, so this reads common field names.
function sourceLabel(s) {
  if (typeof s === "string") return s;
  const doc = s.doc_title || s.document || s.doc || s.title || s.source || s.name;
  const sec = s.section || s.sec;
  return [doc, sec ? `Sec. ${sec}` : null].filter(Boolean).join(" · ") || "Source";
}

function Sources({ sources }) {
  if (!Array.isArray(sources) || sources.length === 0) return null;
  const labels = [...new Set(sources.map(sourceLabel))];
  return (
    <div className="sources">
      {labels.map((l) => (
        <span key={l} className="source-chip">{l}</span>
      ))}
    </div>
  );
}

// Parse one SSE block ("event: x\ndata: y") into { event, data }.
function parseSSE(block) {
  let event = "message";
  const dataLines = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
  }
  if (!dataLines.length) return null;
  try {
    return { event, data: JSON.parse(dataLines.join("\n")) };
  } catch {
    return null;
  }
}

export default function LegalChat() {
  const [phase, setPhase] = useState("splash"); // splash | exiting | chat
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const endRef = useRef(null);

  useEffect(() => {
    const t1 = setTimeout(() => setPhase("exiting"), 5600);
    const t2 = setTimeout(() => setPhase("chat"), 6300);
    return () => { clearTimeout(t1); clearTimeout(t2); };
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function send(text) {
    const message = (text ?? input).trim();
    if (!message || loading) return;

    // Backend keeps the last 6 turns; roles must be "user" | "assistant".
    const history = messages
      .filter((m) => m.content && !m.error)
      .slice(-6)
      .map((m) => ({ role: m.role, content: m.content.slice(0, 4000) }));

    setMessages((m) => [
      ...m,
      { role: "user", content: message },
      { role: "assistant", content: "", sources: [], streaming: true },
    ]);
    setInput("");
    setLoading(true);

    const updateLast = (fn) =>
      setMessages((m) => {
        const copy = [...m];
        copy[copy.length - 1] = fn(copy[copy.length - 1]);
        return copy;
      });

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, history }),
      });
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      const handle = (block) => {
        const evt = parseSSE(block);
        if (!evt) return;
        if (evt.event === "sources") updateLast((m) => ({ ...m, sources: evt.data }));
        else if (evt.event === "token") updateLast((m) => ({ ...m, content: m.content + evt.data }));
        else if (evt.event === "error") updateLast((m) => ({ ...m, content: evt.data, error: true }));
      };

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const blocks = buffer.split("\n\n");
        buffer = blocks.pop(); // keep any incomplete block
        blocks.forEach(handle);
      }
      if (buffer.trim()) handle(buffer);
    } catch (err) {
      updateLast((m) => ({
        ...m,
        content: m.content || `Could not reach the server (${err.message}). Is the backend running?`,
        error: !m.content,
      }));
    } finally {
      updateLast((m) => ({ ...m, streaming: false }));
      setLoading(false);
    }
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  return (
    <>
      {phase !== "chat" && <Splash exiting={phase === "exiting"} />}

      <div className={`app ${phase === "chat" ? "app-in" : ""}`}>
        <header className="header">
          <div className="brand">
            <span className="logo"><Scales /></span>
            <div>
              <div className="brand-name">{APP_NAME}</div>
              <div className="brand-sub">{TAGLINE}</div>
            </div>
          </div>
          <button className="ghost" onClick={() => setMessages([])} disabled={!messages.length || loading}>
            New chat
          </button>
        </header>

        <main className="messages">
          {messages.length === 0 && (
            <div className="empty">
              <h2>How can I help?</h2>
              <p>Ask in English or Urdu.</p>
              <div className="chips">
                {SUGGESTIONS.map((s) => (
                  <button key={s} className="chip" dir="auto" onClick={() => send(s)}>
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((m, i) => {
            if (m.role === "assistant" && m.streaming && !m.content) {
              return <TypingDots key={i} />;
            }
            return (
              <div key={i} className={`msg ${m.role === "user" ? "user" : "bot"}`}>
                <div className="col">
                  <div
                    className={`bubble ${m.role === "assistant" && !m.error ? "md" : ""} ${m.error ? "error" : ""}`}
                    dir="auto"
                  >
                    {m.role === "assistant" && !m.error ? (
                      <ReactMarkdown>{m.content}</ReactMarkdown>
                    ) : (
                      m.content
                    )}
                  </div>
                  {m.role === "assistant" && !m.error && <Sources sources={m.sources} />}
                </div>
              </div>
            );
          })}
          <div ref={endRef} />
        </main>

        <footer className="composer">
          <div className="composer-row">
            <textarea
              rows={1}
              dir="auto"
              maxLength={1000}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Type your legal question…"
            />
            <button className="send" onClick={() => send()} disabled={!input.trim() || loading} aria-label="Send">
              <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M22 2L11 13" /><path d="M22 2l-7 20-4-9-9-4 20-7z" />
              </svg>
            </button>
          </div>
          <p className="disclaimer">General legal information, not legal advice. Consult a qualified lawyer.</p>
        </footer>
      </div>
    </>
  );
}
"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { submitRun, listRuns, type Run } from "@/lib/api";

const STATUS: Record<string, { color: string; bg: string; label: string }> = {
  pending:      { color: "#a89278", bg: "rgba(168,146,120,0.1)", label: "Pending" },
  planning:     { color: "#2563eb", bg: "rgba(37,99,235,0.1)",   label: "Planning" },
  researching:  { color: "#d97706", bg: "rgba(217,119,6,0.1)",   label: "Researching" },
  synthesizing: { color: "#7c3aed", bg: "rgba(124,58,237,0.1)",  label: "Synthesizing" },
  completed:    { color: "#059669", bg: "rgba(5,150,105,0.1)",   label: "Completed" },
  failed:       { color: "#dc2626", bg: "rgba(220,38,38,0.1)",   label: "Failed" },
};

function StatusPill({ status }: { status: string }) {
  const s = STATUS[status] ?? STATUS.pending;
  const pulse = ["planning", "researching", "synthesizing"].includes(status);
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 6,
      background: s.bg, border: `1px solid ${s.color}30`,
      borderRadius: 100, padding: "3px 10px",
      fontSize: 12, fontWeight: 500, color: s.color,
    }}>
      <span style={{
        width: 6, height: 6, borderRadius: "50%", background: s.color, flexShrink: 0,
        animation: pulse ? "pulse-dot 1.5s ease-in-out infinite" : "none",
      }} />
      {s.label}
    </span>
  );
}

export default function Home() {
  const router = useRouter();
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [runs, setRuns] = useState<Run[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    listRuns().then(setRuns).catch(console.error);
  }, []);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!question.trim() || loading) return;
    setLoading(true);
    setError("");
    try {
      const run = await submitRun(question.trim());
      router.push(`/runs/${run.id}`);
    } catch (err) {
      setError(String(err));
      setLoading(false);
    }
  }

  return (
    <main style={{ position: "relative", zIndex: 1, maxWidth: 720, margin: "0 auto", padding: "80px 24px 120px" }}>
      <div className="animate-fade-in" style={{ marginBottom: 64 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 32 }}>
          <span className="mono" style={{ fontSize: 13, color: "var(--text-3)", letterSpacing: "0.05em" }}>
            // Agora
          </span>
        </div>
        <h1 className="serif animate-fade-up" style={{
          fontSize: 56, lineHeight: 1.08, color: "var(--text-1)",
          marginBottom: 16, fontWeight: 400,
        }}>
          Distributed<br />Multi-Agent Research
        </h1>
        <p className="animate-fade-up delay-1" style={{
          color: "var(--text-2)", fontSize: 16, maxWidth: 440, lineHeight: 1.7,
        }}>
          Ask a question. Let the system decompose, research, and synthesize.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="animate-fade-up delay-2" style={{ marginBottom: 64 }}>
        <div className="glass" style={{
          padding: "6px 6px 6px 20px",
          display: "flex", alignItems: "center", gap: 12,
          boxShadow: "var(--shadow-md)",
        }}>
          <input
            type="text"
            value={question}
            onChange={e => setQuestion(e.target.value)}
            placeholder="What would you like to research?"
            disabled={loading}
            style={{
              flex: 1, background: "transparent", border: "none", outline: "none",
              color: "var(--text-1)", fontSize: 15, fontFamily: "inherit",
              padding: "10px 0",
            }}
            onKeyDown={e => { if (e.key === "Enter") handleSubmit(e as never); }}
          />
          <button
            type="submit"
            disabled={loading || !question.trim()}
            style={{
              background: loading || !question.trim() ? "rgba(217,119,6,0.4)" : "var(--amber)",
              color: "#fff", border: "none", borderRadius: 10,
              padding: "12px 24px", fontSize: 14, fontWeight: 600,
              cursor: loading ? "not-allowed" : "pointer",
              fontFamily: "inherit", transition: "all 0.2s ease", whiteSpace: "nowrap",
            }}
          >
            {loading ? "…" : "Research →"}
          </button>
        </div>
        {error && <p style={{ color: "var(--red)", marginTop: 8, fontSize: 13 }}>{error}</p>}
      </form>

      {runs.length > 0 && (
        <div className="animate-fade-up delay-3">
          <div style={{
            display: "flex", alignItems: "center", gap: 8, marginBottom: 16,
            color: "var(--text-3)", fontSize: 12, fontWeight: 500,
            letterSpacing: "0.08em", textTransform: "uppercase",
          }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10"/><polyline points="12,6 12,12 16,14"/>
            </svg>
            Recent Runs
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {runs.map((run, i) => (
              <a key={run.id} href={`/runs/${run.id}`}
                className="glass transition-base animate-slide-in"
                style={{
                  display: "flex", alignItems: "center", justifyContent: "space-between",
                  padding: "16px 20px", textDecoration: "none", gap: 16,
                  borderRadius: 14, animationDelay: `${i * 0.05}s`,
                }}
                onMouseEnter={e => {
                  (e.currentTarget as HTMLElement).style.boxShadow = "var(--shadow-md)";
                  (e.currentTarget as HTMLElement).style.transform = "translateY(-1px)";
                }}
                onMouseLeave={e => {
                  (e.currentTarget as HTMLElement).style.boxShadow = "var(--shadow-sm)";
                  (e.currentTarget as HTMLElement).style.transform = "translateY(0)";
                }}
              >
                <div style={{ flex: 1, minWidth: 0 }}>
                  <p style={{
                    color: "var(--text-1)", fontSize: 14, fontWeight: 500,
                    overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                    marginBottom: 4,
                  }}>
                    {run.question}
                  </p>
                  <p style={{ fontSize: 12, color: "var(--text-3)" }}>{run.id.slice(0, 8)}</p>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 12, flexShrink: 0 }}>
                  <StatusPill status={run.status} />
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--text-3)" strokeWidth="2">
                    <polyline points="9,18 15,12 9,6"/>
                  </svg>
                </div>
              </a>
            ))}
          </div>
        </div>
      )}
    </main>
  );
}
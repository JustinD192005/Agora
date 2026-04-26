"use client";

import { useEffect, useState, use } from "react";
import { getRunDetails, type RunDetail, type Task } from "@/lib/api";

const TERMINAL = ["completed", "failed", "completed_partial"];

const STATUS: Record<string, { color: string; bg: string; label: string }> = {
  pending:           { color: "#a89278", bg: "rgba(168,146,120,0.1)", label: "Pending" },
  running:           { color: "#d97706", bg: "rgba(217,119,6,0.1)",   label: "Running" },
  completed:         { color: "#059669", bg: "rgba(5,150,105,0.1)",   label: "Completed" },
  completed_partial: { color: "#059669", bg: "rgba(5,150,105,0.1)",   label: "Partial" },
  failed:            { color: "#dc2626", bg: "rgba(220,38,38,0.1)",   label: "Failed" },
  planning:          { color: "#2563eb", bg: "rgba(37,99,235,0.1)",   label: "Planning" },
  researching:       { color: "#d97706", bg: "rgba(217,119,6,0.1)",   label: "Researching" },
  synthesizing:      { color: "#7c3aed", bg: "rgba(124,58,237,0.1)",  label: "Synthesizing" },
};

function Pill({ status }: { status: string }) {
  const s = STATUS[status] ?? STATUS.pending;
  const pulse = ["running", "planning", "researching", "synthesizing"].includes(status);
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 5,
      background: s.bg, border: `1px solid ${s.color}30`,
      borderRadius: 100, padding: "3px 10px",
      fontSize: 11, fontWeight: 500, color: s.color, whiteSpace: "nowrap",
    }}>
      <span style={{
        width: 5, height: 5, borderRadius: "50%", background: s.color, flexShrink: 0,
        animation: pulse ? "pulse-dot 1.5s ease-in-out infinite" : "none",
      }} />
      {s.label}
    </span>
  );
}

function ResearcherCard({ task, index }: { task: Task; index: number }) {
  const subQ = (task.input?.sub_question as string) ?? "...";
  const iterations = task.output?.iterations as number | undefined;
  const citations = (task.output?.citations as unknown[])?.length ?? 0;
  const terminated = task.output?.terminated_reason as string | undefined;
  const isActive = task.status === "running";

  return (
    <div className="glass-2 animate-fade-up" style={{
      padding: "16px",
      border: `1px solid ${isActive ? "rgba(217,119,6,0.35)" : "rgba(180,160,130,0.2)"}`,
      transition: "all 0.3s ease",
      boxShadow: isActive ? "0 0 0 3px rgba(217,119,6,0.08)" : "none",
      animationDelay: `${index * 0.05}s`,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 8, marginBottom: 10 }}>
        <span className="mono" style={{ fontSize: 10, color: "var(--text-3)", paddingTop: 2 }}>#{index + 1}</span>
        <Pill status={task.status} />
      </div>
      <p style={{ fontSize: 13, color: "var(--text-1)", lineHeight: 1.5, marginBottom: 10 }}>
        {subQ}
      </p>
      {(iterations !== undefined || citations > 0) && (
        <div className="mono" style={{ fontSize: 11, color: "var(--text-3)", display: "flex", gap: 12 }}>
          {iterations !== undefined && <span>{iterations} iterations</span>}
          {citations > 0 && <span style={{ color: "var(--green)" }}>{citations} citations</span>}
          {terminated && terminated !== "finish" && <span style={{ color: "var(--red)" }}>{terminated}</span>}
        </div>
      )}
    </div>
  );
}

function DownloadButton({ url }: { url: string }) {
  return (
    <a
      href={url}
      download
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        background: "var(--amber)",
        color: "#fff",
        borderRadius: 8,
        padding: "7px 14px",
        fontSize: 12,
        fontWeight: 600,
        textDecoration: "none",
        fontFamily: "inherit",
        transition: "opacity 0.15s",
      }}
    >
      <span style={{ fontSize: 14 }}>↓</span>
      <span>Download PDF</span>
    </a>
  );
}
export default function RunPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [run, setRun] = useState<RunDetail | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    async function poll() {
      try {
        const data = await getRunDetails(id);
        if (!active) return;
        setRun(data);
        if (!TERMINAL.includes(data.status)) setTimeout(poll, 2500);
      } catch (err) {
        if (active) setError(String(err));
      }
    }
    poll();
    return () => { active = false; };
  }, [id]);

  if (error) {
    return (
      <main style={{ position: "relative", zIndex: 1, maxWidth: 760, margin: "0 auto", padding: "60px 24px" }}>
        <p style={{ color: "var(--red)" }}>{error}</p>
      </main>
    );
  }

  if (!run) {
    return (
      <main style={{ position: "relative", zIndex: 1, maxWidth: 760, margin: "0 auto", padding: "60px 24px" }}>
        <div style={{ display: "flex", gap: 6, alignItems: "center", color: "var(--text-3)" }}>
          <div style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--amber)", animation: "pulse-dot 1s ease-in-out infinite" }} />
          <span className="mono" style={{ fontSize: 12 }}>Loading...</span>
        </div>
      </main>
    );
  }

  const planner = run.tasks.find(t => t.kind === "planner");
  const researchers = run.tasks.filter(t => t.kind === "researcher");
  const synthesizer = run.tasks.find(t => t.kind === "synthesizer");
  const subQCount = planner?.output ? ((planner.output.sub_questions as unknown[])?.length ?? 0) : 0;
  const successCount = researchers.filter(t => ["completed", "completed_partial"].includes(t.status)).length;
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  const pdfUrl = apiUrl + "/runs/" + run.id + "/report.pdf";

  return (
    <main style={{ position: "relative", zIndex: 1, maxWidth: 760, margin: "0 auto", padding: "60px 24px 100px" }}>

      <a href="/" className="mono animate-fade-in" style={{
        fontSize: 12, color: "var(--text-3)", textDecoration: "none",
        display: "inline-flex", alignItems: "center", gap: 6,
        marginBottom: 40, letterSpacing: "0.06em", transition: "color 0.15s",
      }}>
        ← Agora
      </a>

      <div className="animate-fade-up" style={{ marginBottom: 48 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16, flexWrap: "wrap" }}>
          <Pill status={run.status} />
          <span className="mono" style={{ fontSize: 11, color: "var(--text-3)" }}>{run.id.slice(0, 8)}</span>
        </div>
        <h1 className="serif" style={{ fontSize: 32, fontWeight: 400, lineHeight: 1.3, color: "var(--text-1)" }}>
          {run.question}
        </h1>
      </div>

      <div className="animate-fade-up delay-1">

        <div style={{ marginBottom: 32 }}>
          <div className="mono" style={{ fontSize: 10, color: "var(--text-3)", letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: 12 }}>
            01 / Planner
          </div>
          <div className="glass" style={{
            padding: "16px 20px", display: "flex",
            justifyContent: "space-between", alignItems: "center",
          }}>
            <span style={{ color: "var(--text-2)", fontSize: 14 }}>
              {planner ? "Decomposed into " + subQCount + " sub-questions" : "Generating research plan..."}
            </span>
            <Pill status={planner?.status ?? "pending"} />
          </div>
        </div>

        {researchers.length > 0 && (
          <div style={{ marginBottom: 32 }}>
            <div className="mono" style={{
              fontSize: 10, color: "var(--text-3)", letterSpacing: "0.12em",
              textTransform: "uppercase", marginBottom: 12,
              display: "flex", justifyContent: "space-between",
            }}>
              <span>02 / Researchers</span>
              <span style={{ color: successCount > 0 ? "var(--green)" : "var(--text-3)" }}>
                {successCount}/{researchers.length} complete
              </span>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 10 }}>
              {researchers.map((task, i) => (
                <ResearcherCard key={task.id} task={task} index={i} />
              ))}
            </div>
          </div>
        )}

        {synthesizer && (
          <div style={{ marginBottom: 32 }}>
            <div className="mono" style={{ fontSize: 10, color: "var(--text-3)", letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: 12 }}>
              03 / Synthesizer
            </div>
            <div className="glass" style={{
              padding: "16px 20px", display: "flex",
              justifyContent: "space-between", alignItems: "center",
              border: synthesizer.status === "running" ? "1px solid rgba(124,58,237,0.3)" : "1px solid var(--border)",
              boxShadow: synthesizer.status === "running" ? "0 0 0 3px rgba(124,58,237,0.06)" : "var(--shadow-sm)",
              transition: "all 0.4s ease",
            }}>
              <span style={{ color: "var(--text-2)", fontSize: 14 }}>
                {synthesizer.status === "completed"
                  ? "Synthesized " + ((synthesizer.output?.citations as unknown[])?.length ?? 0) + " citations"
                  : "Integrating findings..."}
              </span>
              <Pill status={synthesizer.status} />
            </div>
          </div>
        )}
      </div>

      {run.final_answer && (
        <div className="animate-fade-up delay-2">

          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
            <div className="mono" style={{ fontSize: 10, color: "var(--text-3)", letterSpacing: "0.12em", textTransform: "uppercase" }}>
              Synthesized Answer
            </div>
            <DownloadButton url={pdfUrl} />
          </div>

          <div className="glass" style={{
            padding: "28px", borderLeft: "3px solid var(--amber)",
            lineHeight: 1.85, color: "var(--text-1)", fontSize: 16,
            whiteSpace: "pre-wrap", boxShadow: "var(--shadow-md)",
            marginBottom: 32,
          }}>
            {run.final_answer}
          </div>

          {Array.isArray(synthesizer?.output?.citations) && (synthesizer.output.citations as unknown[]).length > 0 && (
            <div>
              <div className="mono" style={{ fontSize: 10, color: "var(--text-3)", letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: 12 }}>
                Citations
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {(synthesizer.output.citations as Array<{ url: string; quote: string }>).map((c, i) => (
                  <div key={i} className="glass-2 animate-slide-in" style={{ padding: "14px 16px", animationDelay: (i * 0.05) + "s" }}>
                    <div style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
                      <span className="mono" style={{ fontSize: 11, color: "var(--amber)", fontWeight: 500, flexShrink: 0, marginTop: 2 }}>
                        [{i + 1}]
                      </span>
                      <div>
                        <a href={c.url} target="_blank" rel="noopener noreferrer" className="mono" style={{
                          fontSize: 11, color: "var(--blue)", textDecoration: "none",
                          display: "block", marginBottom: 4,
                          overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 580,
                        }}>
                          {c.url}
                        </a>
                        <p style={{ fontSize: 13, color: "var(--text-2)", fontStyle: "italic", lineHeight: 1.5 }}>
                          &quot;{c.quote}&quot;
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </main>
  );
}
const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Run {
  id: string;
  question: string;
  status: string;
  final_answer: string | null;
}

export interface Task {
  id: string;
  kind: string;
  status: string;
  input: Record<string, unknown>;
  output: Record<string, unknown> | null;
  started_at: string | null;
  completed_at: string | null;
  error: string | null;
}

export interface RunDetail {
  id: string;
  question: string;
  status: string;
  final_answer: string | null;
  expected_researchers: number | null;
  created_at: string;
  completed_at: string | null;
  tasks: Task[];
}

export async function submitRun(question: string, bustCache = false): Promise<Run> {
  const res = await fetch(`${API}/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, bust_cache: bustCache }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function listRuns(): Promise<Run[]> {
  const res = await fetch(`${API}/runs`, { cache: "no-store" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getRunDetails(id: string): Promise<RunDetail> {
  const res = await fetch(`${API}/runs/${id}/details`, { cache: "no-store" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
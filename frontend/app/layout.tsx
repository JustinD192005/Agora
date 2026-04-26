import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Agora — Distributed Research",
  description: "Multi-agent research platform",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <div style={{ position: "relative", zIndex: 1, minHeight: "100vh" }}>
          {children}
        </div>
      </body>
    </html>
  );
}
import "./globals.css";
import type { ReactNode } from "react";

export const metadata = {
  title: "Sign-Sync",
  description: "Break communication barriers between sign and speech users."
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div id="cursor-glow" />
        {children}
      </body>
    </html>
  );
}

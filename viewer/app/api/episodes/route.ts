import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export async function GET() {
  const dir = path.join(process.cwd(), "public", "episodes");
  let episodes: string[] = [];
  try {
    episodes = fs
      .readdirSync(dir)
      .filter((f) => f.endsWith(".json"))
      .map((f) => f.replace(/\.json$/, ""))
      .sort();
  } catch {
    episodes = [];
  }
  return NextResponse.json({ episodes });
}

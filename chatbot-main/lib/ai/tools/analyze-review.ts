import { tool } from "ai";
import { z } from "zod";

export const analyzeReview = tool({
  description:
    "Analyze the sentiment of a product review using the RAG (Retrieval-Augmented Generation) pipeline. " +
    "Encodes the review into a 128-dim vector, retrieves the most similar reviews from a pgvector database, " +
    "and returns a grounded sentiment explanation. Call this whenever the user submits a product review.",
  parameters: z.object({
    review: z.string().describe("The product review text to analyze"),
    top_k: z
      .number()
      .int()
      .min(1)
      .max(10)
      .default(5)
      .describe("Number of similar reviews to retrieve"),
  }),
  execute: async ({ review, top_k }) => {
    const backendUrl = process.env.BACKEND_URL ?? "http://localhost:8000";
    const res = await fetch(`${backendUrl}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ review, top_k }),
    });
    if (!res.ok) {
      throw new Error(`RAG backend error ${res.status}: ${await res.text()}`);
    }
    return await res.json();
  },
});

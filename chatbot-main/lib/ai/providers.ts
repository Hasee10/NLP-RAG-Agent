import { createOpenAI } from "@ai-sdk/openai";
import { isTestEnvironment } from "../constants";

function buildOpenRouter() {
  return createOpenAI({
    baseURL: "https://openrouter.ai/api/v1",
    apiKey: process.env.OPENROUTER_API_KEY ?? "",
    headers: {
      "HTTP-Referer": "http://localhost:3000",
      "X-Title": "RAG Sentiment Agent",
    },
  });
}

export function getLanguageModel(modelId: string) {
  if (isTestEnvironment) {
    const { chatModel } = require("./models.mock");
    const { customProvider } = require("ai");
    return customProvider({ languageModels: { "chat-model": chatModel } }).languageModel("chat-model");
  }
  return buildOpenRouter()(modelId);
}

export function getTitleModel() {
  if (isTestEnvironment) {
    const { titleModel } = require("./models.mock");
    const { customProvider } = require("ai");
    return customProvider({ languageModels: { "title-model": titleModel } }).languageModel("title-model");
  }
  return buildOpenRouter()("meta-llama/llama-3.3-70b-instruct:free");
}

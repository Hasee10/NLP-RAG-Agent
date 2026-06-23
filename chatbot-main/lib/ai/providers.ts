import { createOpenAI } from "@ai-sdk/openai";
import { isTestEnvironment } from "../constants";

function buildGroq() {
  return createOpenAI({
    baseURL: "https://api.groq.com/openai/v1",
    apiKey: process.env.GROQ_API_KEY ?? "",
  });
}

export function getLanguageModel(modelId: string) {
  if (isTestEnvironment) {
    const { chatModel } = require("./models.mock");
    const { customProvider } = require("ai");
    return customProvider({ languageModels: { "chat-model": chatModel } }).languageModel("chat-model");
  }
  return buildGroq()(modelId);
}

export function getTitleModel() {
  if (isTestEnvironment) {
    const { titleModel } = require("./models.mock");
    const { customProvider } = require("ai");
    return customProvider({ languageModels: { "title-model": titleModel } }).languageModel("title-model");
  }
  return buildGroq()("llama-3.1-8b-instant");
}

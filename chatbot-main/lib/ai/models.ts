export const DEFAULT_CHAT_MODEL = "deepseek/deepseek-r1:free";

export const titleModel = {
  id: "deepseek/deepseek-r1:free",
  name: "DeepSeek R1 (Free)",
  provider: "deepseek",
  description: "Fast model for title generation",
};

export type ModelCapabilities = {
  tools: boolean;
  vision: boolean;
  reasoning: boolean;
};

export type ChatModel = {
  id: string;
  name: string;
  provider: string;
  description: string;
  gatewayOrder?: string[];
  reasoningEffort?: "none" | "minimal" | "low" | "medium" | "high";
};

export const chatModels: ChatModel[] = [
  {
    id: "deepseek/deepseek-r1:free",
    name: "DeepSeek R1 (Free)",
    provider: "deepseek",
    description: "Free reasoning model — default",
  },
  {
    id: "meta-llama/llama-3.3-70b-instruct:free",
    name: "Llama 3.3 70B (Free)",
    provider: "meta-llama",
    description: "Meta's open-source 70B model, free tier",
  },
  {
    id: "google/gemma-3-27b-it:free",
    name: "Gemma 3 27B (Free)",
    provider: "google",
    description: "Google Gemma free tier",
  },
  {
    id: "mistralai/mistral-7b-instruct:free",
    name: "Mistral 7B (Free)",
    provider: "mistralai",
    description: "Mistral lightweight free model",
  },
];

// Static capability map — all free models; tool support varies but we handle gracefully.
const STATIC_CAPABILITIES: Record<string, ModelCapabilities> = {
  "deepseek/deepseek-r1:free":                  { tools: false, vision: false, reasoning: true  },
  "meta-llama/llama-3.3-70b-instruct:free":     { tools: true,  vision: false, reasoning: false },
  "google/gemma-3-27b-it:free":                 { tools: false, vision: false, reasoning: false },
  "mistralai/mistral-7b-instruct:free":         { tools: false, vision: false, reasoning: false },
};

export async function getCapabilities(): Promise<Record<string, ModelCapabilities>> {
  return STATIC_CAPABILITIES;
}

export const isDemo = process.env.IS_DEMO === "1";

export type GatewayModelWithCapabilities = ChatModel & {
  capabilities: ModelCapabilities;
};

export function getActiveModels(): ChatModel[] {
  return chatModels;
}

export async function getAllGatewayModels(): Promise<GatewayModelWithCapabilities[]> {
  const caps = await getCapabilities();
  return chatModels.map((m) => ({
    ...m,
    capabilities: caps[m.id] ?? { tools: false, vision: false, reasoning: false },
  }));
}

export const allowedModelIds = new Set(chatModels.map((m) => m.id));

export const modelsByProvider = chatModels.reduce(
  (acc, model) => {
    if (!acc[model.provider]) {
      acc[model.provider] = [];
    }
    acc[model.provider].push(model);
    return acc;
  },
  {} as Record<string, ChatModel[]>
);

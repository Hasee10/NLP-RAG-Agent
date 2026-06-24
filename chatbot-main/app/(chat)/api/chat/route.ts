import { geolocation, ipAddress } from "@vercel/functions";
import {
  convertToModelMessages,
  createUIMessageStream,
  createUIMessageStreamResponse,
  generateId,
  stepCountIs,
  streamText,
} from "ai";
import { checkBotId } from "botid/server";
import { after } from "next/server";
import { createResumableStreamContext } from "resumable-stream";
import { auth, type UserType } from "@/app/(auth)/auth";
import { entitlementsByUserType } from "@/lib/ai/entitlements";
import {
  allowedModelIds,
  chatModels,
  DEFAULT_CHAT_MODEL,
  getCapabilities,
} from "@/lib/ai/models";
import { type RequestHints, systemPrompt } from "@/lib/ai/prompts";
import { getLanguageModel } from "@/lib/ai/providers";
import { createDocument } from "@/lib/ai/tools/create-document";
import { editDocument } from "@/lib/ai/tools/edit-document";
import { requestSuggestions } from "@/lib/ai/tools/request-suggestions";
import { updateDocument } from "@/lib/ai/tools/update-document";
import { isProductionEnvironment } from "@/lib/constants";
import {
  createStreamId,
  deleteChatById,
  getChatById,
  getMessageCountByUserId,
  getMessagesByChatId,
  saveChat,
  saveMessages,
  updateChatTitleById,
  updateMessage,
} from "@/lib/db/queries";
import type { DBMessage } from "@/lib/db/schema";
import { ChatbotError } from "@/lib/errors";
import { checkIpRateLimit } from "@/lib/ratelimit";
import type { ChatMessage } from "@/lib/types";
import { convertToUIMessages, generateUUID } from "@/lib/utils";
import { generateTitleFromUserMessage } from "../../actions";
import { type PostRequestBody, postRequestBodySchema } from "./schema";

export const maxDuration = 60;

// ── Medical RAG ──────────────────────────────────────────────────────────────
type MedicalRagData = {
  answer: string;
  citations: string[];
  disclaimer: string;
  chunks: { text: string; source?: string; type?: string; body_system?: string; similarity?: number }[];
  model: string;
  body_system: string | null;
  chunk_type: string | null;
  sources_used: number;
};

async function fetchMedicalRag(text: string): Promise<MedicalRagData | null> {
  try {
    // Medical RAG lives on the same HF Space as sentiment, just a different endpoint
    const url = process.env.BACKEND_URL ?? "http://localhost:8000";
    const res = await fetch(`${url}/medical-query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: text.slice(0, 2000), top_k: 5 }),
      signal: AbortSignal.timeout(20000),
    });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

function formatMedicalContext(data: MedicalRagData): string {
  const sources = (data.chunks ?? [])
    .map((c, i) => `  [${i + 1}] (${c.source ?? "unknown"} / ${c.type ?? ""}) "${c.text.slice(0, 200)}"`)
    .join("\n");
  return [
    `<medical_rag>`,
    `grounded_answer: ${data.answer}`,
    `body_system: ${data.body_system ?? "general"}`,
    `sources:\n${sources}`,
    `disclaimer: ${data.disclaimer}`,
    `</medical_rag>`,
  ].join("\n");
}

// ── Sentiment RAG ─────────────────────────────────────────────────────────────
type SentimentRagData = {
  predicted_sentiment: string;
  explanation: string;
  retrieved: { sentiment: string; similarity: number; text: string }[];
};

async function fetchSentimentRag(text: string): Promise<SentimentRagData | null> {
  try {
    const url = process.env.BACKEND_URL ?? "http://localhost:8000";
    const res = await fetch(`${url}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ review: text.slice(0, 5000), top_k: 5 }),
      signal: AbortSignal.timeout(15000),
    });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

function formatSentimentContext(data: SentimentRagData): string {
  const pct = (s: number) => `${(s * 100).toFixed(0)}%`;
  const neighbors = (data.retrieved ?? [])
    .map((r, i) => `  ${i + 1}. [${r.sentiment}] ${pct(r.similarity)} match — "${r.text.slice(0, 120)}"`)
    .join("\n");
  return [
    `<sentiment_rag>`,
    `sentiment: ${data.predicted_sentiment}`,
    `neighbors:\n${neighbors}`,
    `explanation: ${data.explanation}`,
    `</sentiment_rag>`,
  ].join("\n");
}

// ── Query routing ─────────────────────────────────────────────────────────────
const MEDICAL_KW = [
  "symptom", "disease", "diagnosis", "treatment", "medication", "drug", "dose",
  "pain", "fever", "cancer", "diabetes", "heart", "blood", "lung", "kidney",
  "brain", "nerve", "bone", "muscle", "skin", "allergy", "infection", "surgery",
  "therapy", "vaccine", "vitamin", "diet", "exercise", "health", "medical",
  "doctor", "hospital", "pharmacy", "prescription", "side effect", "cause",
  "prevent", "cure", "chronic", "acute", "hypertension", "stroke", "asthma",
];

function isMedicalQuery(text: string): boolean {
  const lower = text.toLowerCase();
  return MEDICAL_KW.some((kw) => lower.includes(kw));
}

function looksLikeReview(text: string): boolean {
  const words = text.trim().split(/\s+/).length;
  const isQuestion = text.trim().endsWith("?");
  const startsWithQuestion = /^(how|what|why|when|where|explain|describe|tell|is |are |do |does )/i.test(text.trim());
  return words >= 8 && !isQuestion && !startsWithQuestion && !isMedicalQuery(text);
}

function getStreamContext() {
  try {
    return createResumableStreamContext({ waitUntil: after });
  } catch (_) {
    return null;
  }
}

export { getStreamContext };

export async function POST(request: Request) {
  let requestBody: PostRequestBody;

  try {
    const json = await request.json();
    requestBody = postRequestBodySchema.parse(json);
  } catch (_) {
    return new ChatbotError("bad_request:api").toResponse();
  }

  try {
    const { id, message, messages, selectedChatModel, selectedVisibilityType } =
      requestBody;

    const [botCheck, session] = await Promise.all([
      checkBotId().catch(() => ({ isBot: false })),
      auth(),
    ]);

    if (botCheck?.isBot) {
      return new ChatbotError("bad_request:api").toResponse();
    }

    if (!session?.user) {
      return new ChatbotError("unauthorized:chat").toResponse();
    }

    const chatModel = allowedModelIds.has(selectedChatModel)
      ? selectedChatModel
      : DEFAULT_CHAT_MODEL;

    await checkIpRateLimit(ipAddress(request));

    const userType: UserType = session.user.type;

    const messageCount = await getMessageCountByUserId({
      id: session.user.id,
      differenceInHours: 1,
    });

    if (messageCount > entitlementsByUserType[userType].maxMessagesPerHour) {
      return new ChatbotError("rate_limit:chat").toResponse();
    }

    const isToolApprovalFlow = Boolean(messages);

    const chat = await getChatById({ id });
    let messagesFromDb: DBMessage[] = [];
    let titlePromise: Promise<string> | null = null;

    if (chat) {
      if (chat.userId !== session.user.id) {
        return new ChatbotError("forbidden:chat").toResponse();
      }
      messagesFromDb = await getMessagesByChatId({ id });
    } else if (message?.role === "user") {
      await saveChat({
        id,
        userId: session.user.id,
        title: "New chat",
        visibility: selectedVisibilityType,
      });
      titlePromise = generateTitleFromUserMessage({ message });
    }

    let uiMessages: ChatMessage[];

    if (isToolApprovalFlow && messages) {
      const dbMessages = convertToUIMessages(messagesFromDb);
      const approvalStates = new Map(
        messages.flatMap(
          (m) =>
            m.parts
              ?.filter(
                (p: Record<string, unknown>) =>
                  p.state === "approval-responded" ||
                  p.state === "output-denied"
              )
              .map((p: Record<string, unknown>) => [
                String(p.toolCallId ?? ""),
                p,
              ]) ?? []
        )
      );
      uiMessages = dbMessages.map((msg) => ({
        ...msg,
        parts: msg.parts.map((part) => {
          if (
            "toolCallId" in part &&
            approvalStates.has(String(part.toolCallId))
          ) {
            return { ...part, ...approvalStates.get(String(part.toolCallId)) };
          }
          return part;
        }),
      })) as ChatMessage[];
    } else {
      uiMessages = [
        ...convertToUIMessages(messagesFromDb),
        message as ChatMessage,
      ];
    }

    const { longitude, latitude, city, country } = geolocation(request);

    const requestHints: RequestHints = {
      longitude,
      latitude,
      city,
      country,
    };

    if (message?.role === "user") {
      await saveMessages({
        messages: [
          {
            chatId: id,
            id: message.id,
            role: "user",
            parts: message.parts,
            attachments: [],
            createdAt: new Date(),
          },
        ],
      });
    }

    const modelConfig = chatModels.find((m) => m.id === chatModel);
    const modelCapabilities = await getCapabilities();
    const capabilities = modelCapabilities[chatModel];
    const isReasoningModel = capabilities?.reasoning === true;
    const supportsTools = capabilities?.tools === true;

    const modelMessages = await convertToModelMessages(uiMessages);

    // Route query to medical RAG, sentiment RAG, or both in parallel
    const latestUserText = message?.parts
      ?.filter((p: Record<string, unknown>) => p.type === "text")
      .map((p: Record<string, unknown>) => String(p.text ?? ""))
      .join(" ") ?? "";

    const [medicalData, sentimentData] = await Promise.all([
      isMedicalQuery(latestUserText) ? fetchMedicalRag(latestUserText) : Promise.resolve(null),
      looksLikeReview(latestUserText) ? fetchSentimentRag(latestUserText) : Promise.resolve(null),
    ]);

    const contextParts: string[] = [];
    if (medicalData) contextParts.push(formatMedicalContext(medicalData));
    if (sentimentData) contextParts.push(formatSentimentContext(sentimentData));
    const ragContext = contextParts.length > 0 ? contextParts.join("\n\n") : null;

    const stream = createUIMessageStream({
      originalMessages: isToolApprovalFlow ? uiMessages : undefined,
      execute: async ({ writer: dataStream }) => {
        const result = streamText({
          model: getLanguageModel(chatModel),
          system: ragContext
            ? `${systemPrompt({ requestHints, supportsTools: false })}\n\n${ragContext}`
            : systemPrompt({ requestHints, supportsTools }),
          messages: modelMessages,
          stopWhen: stepCountIs(5),
          experimental_activeTools:
            (isReasoningModel && !supportsTools) || ragContext
              ? []
              : [
                  "createDocument",
                  "editDocument",
                  "updateDocument",
                  "requestSuggestions",
                ],
          providerOptions: {
            ...(modelConfig?.gatewayOrder && {
              gateway: { order: modelConfig.gatewayOrder },
            }),
            ...(modelConfig?.reasoningEffort && {
              openai: { reasoningEffort: modelConfig.reasoningEffort },
            }),
          },
          tools: {
            createDocument: createDocument({
              session,
              dataStream,
              modelId: chatModel,
            }),
            editDocument: editDocument({ dataStream, session }),
            updateDocument: updateDocument({
              session,
              dataStream,
              modelId: chatModel,
            }),
            requestSuggestions: requestSuggestions({
              session,
              dataStream,
              modelId: chatModel,
            }),
          },
          experimental_telemetry: {
            isEnabled: isProductionEnvironment,
            functionId: "stream-text",
          },
        });

        dataStream.merge(
          result.toUIMessageStream({ sendReasoning: isReasoningModel })
        );

        if (titlePromise) {
          try {
            const title = await titlePromise;
            dataStream.write({ type: "data-chat-title", data: title });
            updateChatTitleById({ chatId: id, title });
          } catch (_) {
            /* non-fatal */
          }
        }
      },
      generateId: generateUUID,
      onFinish: async ({ messages: finishedMessages }) => {
        if (isToolApprovalFlow) {
          for (const finishedMsg of finishedMessages) {
            const existingMsg = uiMessages.find((m) => m.id === finishedMsg.id);
            if (existingMsg) {
              await updateMessage({
                id: finishedMsg.id,
                parts: finishedMsg.parts,
              });
            } else {
              await saveMessages({
                messages: [
                  {
                    id: finishedMsg.id,
                    role: finishedMsg.role,
                    parts: finishedMsg.parts,
                    createdAt: new Date(),
                    attachments: [],
                    chatId: id,
                  },
                ],
              });
            }
          }
        } else if (finishedMessages.length > 0) {
          await saveMessages({
            messages: finishedMessages.map((currentMessage) => ({
              id: currentMessage.id,
              role: currentMessage.role,
              parts: currentMessage.parts,
              createdAt: new Date(),
              attachments: [],
              chatId: id,
            })),
          });
        }
      },
      onError: (error) => {
        if (
          error instanceof Error &&
          error.message?.includes(
            "AI Gateway requires a valid credit card on file to service requests"
          )
        ) {
          return "AI Gateway requires a valid credit card on file to service requests.";
        }
        // LLM unavailable — surface the RAG backend result directly
        if (medicalData) {
          return [
            `*(LLM temporarily unavailable — showing raw RAG result)*`,
            ``,
            medicalData.answer,
            ``,
            `*${medicalData.disclaimer}*`,
          ].join("\n");
        }
        if (sentimentData) {
          return `*(LLM temporarily unavailable)*\n\nSentiment: **${sentimentData.predicted_sentiment}**\n\n${sentimentData.explanation}`;
        }
        return "The AI model is temporarily unavailable. Please try again in a moment, or switch to a different model using the selector above.";
      },
    });

    return createUIMessageStreamResponse({
      stream,
      async consumeSseStream({ stream: sseStream }) {
        if (!process.env.REDIS_URL) {
          return;
        }
        try {
          const streamContext = getStreamContext();
          if (streamContext) {
            const streamId = generateId();
            await createStreamId({ streamId, chatId: id });
            await streamContext.createNewResumableStream(
              streamId,
              () => sseStream
            );
          }
        } catch (_) {
          /* non-critical */
        }
      },
    });
  } catch (error) {
    const vercelId = request.headers.get("x-vercel-id");

    if (error instanceof ChatbotError) {
      return error.toResponse();
    }

    if (
      error instanceof Error &&
      error.message?.includes(
        "AI Gateway requires a valid credit card on file to service requests"
      )
    ) {
      return new ChatbotError("bad_request:activate_gateway").toResponse();
    }

    console.error("Unhandled error in chat API:", error, { vercelId });
    return new ChatbotError("offline:chat").toResponse();
  }
}

export async function DELETE(request: Request) {
  const { searchParams } = new URL(request.url);
  const id = searchParams.get("id");

  if (!id) {
    return new ChatbotError("bad_request:api").toResponse();
  }

  const session = await auth();

  if (!session?.user) {
    return new ChatbotError("unauthorized:chat").toResponse();
  }

  const chat = await getChatById({ id });

  if (chat?.userId !== session.user.id) {
    return new ChatbotError("forbidden:chat").toResponse();
  }

  const deletedChat = await deleteChatById({ id });

  return Response.json(deletedChat, { status: 200 });
}

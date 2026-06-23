import { createClient } from "redis";

import { ChatbotError } from "@/lib/errors";

const MAX_MESSAGES = 10;
const MAX_GUEST_ACCOUNTS_PER_HOUR = 3;
const TTL_SECONDS = 60 * 60;

let client: ReturnType<typeof createClient> | null = null;

function getClient() {
  if (!client && process.env.REDIS_URL) {
    client = createClient({ url: process.env.REDIS_URL });
    client.on("error", () => undefined);
    client.connect().catch(() => {
      client = null;
    });
  }
  return client;
}

export async function checkIpRateLimit(ip: string | undefined) {
  if (!ip) {
    return;
  }

  const redis = getClient();
  if (!redis?.isReady) {
    // Redis unavailable — fall through (don't block legitimate traffic)
    return;
  }

  try {
    const key = `ip-rate-limit:${ip}`;
    const [count] = await redis
      .multi()
      .incr(key)
      .expire(key, TTL_SECONDS, "NX")
      .exec();

    if (typeof count === "number" && count > MAX_MESSAGES) {
      throw new ChatbotError("rate_limit:chat");
    }
  } catch (error) {
    if (error instanceof ChatbotError) {
      throw error;
    }
  }
}

export async function checkGuestCreationRateLimit(ip: string | undefined): Promise<boolean> {
  if (!ip) return true;

  const redis = getClient();
  if (!redis?.isReady) return true;

  try {
    const key = `guest-creation:${ip}`;
    const [count] = await redis
      .multi()
      .incr(key)
      .expire(key, TTL_SECONDS, "NX")
      .exec();

    return typeof count !== "number" || count <= MAX_GUEST_ACCOUNTS_PER_HOUR;
  } catch {
    return true;
  }
}

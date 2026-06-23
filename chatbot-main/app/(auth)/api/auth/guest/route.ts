import { ipAddress } from "@vercel/functions";
import { NextResponse } from "next/server";
import { getToken } from "next-auth/jwt";
import { signIn } from "@/app/(auth)/auth";
import { isDevelopmentEnvironment } from "@/lib/constants";
import { checkGuestCreationRateLimit } from "@/lib/ratelimit";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const rawRedirect = searchParams.get("redirectUrl") || "/";
  const redirectUrl =
    rawRedirect.startsWith("/") && !rawRedirect.startsWith("//")
      ? rawRedirect
      : "/";

  const token = await getToken({
    req: request,
    secret: process.env.AUTH_SECRET,
    secureCookie: !isDevelopmentEnvironment,
  });

  if (token) {
    const base = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
    return NextResponse.redirect(new URL(`${base}/`, request.url));
  }

  const allowed = await checkGuestCreationRateLimit(ipAddress(request));
  if (!allowed) {
    return NextResponse.json(
      { error: "Too many guest accounts created from this IP. Please try again later." },
      { status: 429 },
    );
  }

  return signIn("guest", { redirect: true, redirectTo: redirectUrl });
}

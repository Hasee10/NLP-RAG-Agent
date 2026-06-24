-- ============================================================
-- Supabase Security Fixes
-- Run this once in the Supabase SQL Editor
-- ============================================================

-- ============================================================
-- 1. Enable Row Level Security on ALL tables
--    (the app uses the postgres superuser via POSTGRES_URL,
--     which bypasses RLS — so this only blocks direct API/anon
--     access, the app is unaffected)
-- ============================================================

ALTER TABLE public."User"        ENABLE ROW LEVEL SECURITY;
ALTER TABLE public."Chat"        ENABLE ROW LEVEL SECURITY;
ALTER TABLE public."Message_v2"  ENABLE ROW LEVEL SECURITY;
ALTER TABLE public."Vote_v2"     ENABLE ROW LEVEL SECURITY;
ALTER TABLE public."Document"    ENABLE ROW LEVEL SECURITY;
ALTER TABLE public."Suggestion"  ENABLE ROW LEVEL SECURITY;
ALTER TABLE public."Stream"      ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.medical_chunks ENABLE ROW LEVEL SECURITY;

-- ============================================================
-- 2. Drop any existing policies to avoid conflicts
-- ============================================================

DO $$
DECLARE
  tbl TEXT;
  tables TEXT[] := ARRAY['"User"','"Chat"','"Message_v2"','"Vote_v2"',
                          '"Document"','"Suggestion"','"Stream"','medical_chunks'];
BEGIN
  FOREACH tbl IN ARRAY tables LOOP
    EXECUTE format('
      DO $inner$
      DECLARE r RECORD;
      BEGIN
        FOR r IN SELECT policyname FROM pg_policies
                 WHERE schemaname = ''public'' AND tablename = %L
        LOOP
          EXECUTE format(''DROP POLICY IF EXISTS %%I ON public.%s'', r.policyname);
        END LOOP;
      END;
      $inner$;
    ', trim(both '"' from tbl), tbl);
  END LOOP;
END;
$$;

-- ============================================================
-- 3. Deny-all policy for chatbot tables
--    Only the postgres superuser (POSTGRES_URL app connection)
--    and service_role key can access these — both bypass RLS.
--    This blocks anon/authenticated Supabase client access.
-- ============================================================

CREATE POLICY "deny_anon_api" ON public."User"
  AS RESTRICTIVE FOR ALL TO anon, authenticated USING (false);

CREATE POLICY "deny_anon_api" ON public."Chat"
  AS RESTRICTIVE FOR ALL TO anon, authenticated USING (false);

CREATE POLICY "deny_anon_api" ON public."Message_v2"
  AS RESTRICTIVE FOR ALL TO anon, authenticated USING (false);

CREATE POLICY "deny_anon_api" ON public."Vote_v2"
  AS RESTRICTIVE FOR ALL TO anon, authenticated USING (false);

CREATE POLICY "deny_anon_api" ON public."Document"
  AS RESTRICTIVE FOR ALL TO anon, authenticated USING (false);

CREATE POLICY "deny_anon_api" ON public."Suggestion"
  AS RESTRICTIVE FOR ALL TO anon, authenticated USING (false);

CREATE POLICY "deny_anon_api" ON public."Stream"
  AS RESTRICTIVE FOR ALL TO anon, authenticated USING (false);

-- medical_chunks: allow SELECT only (read-only public medical data)
-- The RAG backend uses service_role key (bypasses RLS) but this
-- also safely allows authenticated users to read if ever needed.
CREATE POLICY "allow_select" ON public.medical_chunks
  FOR SELECT TO anon, authenticated USING (true);

-- ============================================================
-- 4. Fix: Sensitive Columns Exposed — public."User" password hash
--    Revoke direct column access from API roles
-- ============================================================

REVOKE SELECT ON public."User" FROM anon, authenticated;
GRANT  SELECT (id, email, name, "emailVerified", image, "isAnonymous",
               "createdAt", "updatedAt")
  ON public."User" TO authenticated;

-- ============================================================
-- 5. Fix: Function Search Path Mutable
--    Add explicit search_path to prevent search path injection
-- ============================================================

ALTER FUNCTION public.match_medical_chunks(vector, int, text, text)
  SET search_path = public, pg_catalog;

-- match_reviews may or may not exist — guard with DO block
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM pg_proc p
    JOIN pg_namespace n ON n.oid = p.pronamespace
    WHERE n.nspname = 'public' AND p.proname = 'match_reviews'
  ) THEN
    EXECUTE 'ALTER FUNCTION public.match_reviews SET search_path = public, pg_catalog';
  END IF;
END;
$$;

-- ============================================================
-- 6. Note on "Extension in Public" (public.vector)
--    Moving the vector extension to a separate schema would
--    require recreating all vector columns. Leave it in public
--    for now — it is a low-severity advisory, not a data risk.
-- ============================================================

-- ============================================================
-- Verification: check RLS is enabled on all tables
-- ============================================================

SELECT tablename, rowsecurity
FROM pg_tables
WHERE schemaname = 'public'
  AND tablename IN ('User','Chat','Message_v2','Vote_v2',
                    'Document','Suggestion','Stream','medical_chunks')
ORDER BY tablename;

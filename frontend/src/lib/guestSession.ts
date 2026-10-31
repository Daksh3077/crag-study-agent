import type { SupabaseClient, Session } from "@supabase/supabase-js";
import { supabase } from "./supabaseClient";

export function createSessionLoader(client: SupabaseClient | null) {
  let pending: Promise<Session> | undefined;
  async function load(): Promise<Session> {
    if (!client) throw new Error("The private workspace connection is not configured.");
    const { data, error } = await client.auth.getSession();
    if (error) throw new Error("Your private session could not be restored. Please retry; do not clear browser storage.");
    if (data.session) return data.session;
    const created = await client.auth.signInAnonymously();
    if (created.error || !created.data.session) {
      if (created.error?.code === "anonymous_provider_disabled") {
        throw new Error("Private guest workspaces are not enabled yet. Enable anonymous sign-ins in Supabase Authentication, then retry.");
      }
      throw new Error("Could not open your private workspace. Please retry shortly. If this persists, check the authentication service settings.");
    }
    return created.data.session;
  }
  return function ensureSession(): Promise<Session> {
    if (!pending) pending = load().finally(() => { pending = undefined; });
    return pending;
  };
}

const loadSession = createSessionLoader(supabase);
export async function ensurePrivateSession(): Promise<Session> {
  // Coordinate first visits in multiple tabs as well as concurrent API calls.
  if (typeof navigator !== 'undefined' && navigator.locks) {
    return await navigator.locks.request('crag-private-session', () => loadSession());
  }
  return loadSession();
}

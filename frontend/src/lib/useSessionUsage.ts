import { useCallback, useEffect, useState } from "react";
import { getSessionUsage, type SessionUsage } from "../services/api";

export function useSessionUsage() {
  const [usage, setUsage] = useState<SessionUsage | null>(null);
  const [usageError, setUsageError] = useState("");
  const refreshUsage = useCallback(async () => {
    try { setUsage(await getSessionUsage()); setUsageError(""); }
    catch { setUsage(null); setUsageError("Could not load your session allowance. Retry to continue."); }
  }, []);
  useEffect(() => {
    void refreshUsage();
    const refresh = () => { void refreshUsage(); };
    window.addEventListener("focus", refresh);
    return () => window.removeEventListener("focus", refresh);
  }, [refreshUsage]);
  return { usage, usageError, refreshUsage };
}

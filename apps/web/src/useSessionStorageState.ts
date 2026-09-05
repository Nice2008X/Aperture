import { useEffect, useState } from "react";

/** Same as useState, but the value survives a same-tab reload via sessionStorage — unlike useLocalStorageState, it does NOT persist across tabs/windows or a fresh launch after the browser fully closes (crash-recovery included, for a plain relaunch rather than a browser-driven "restore tabs"). Use this for a marker that should only outlive "the user refreshed this same tab a moment ago," not "forever, in every tab, until something else clears it." */
export function useSessionStorageState<T>(key: string, initial: T) {
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = window.sessionStorage.getItem(key);
      return raw !== null ? (JSON.parse(raw) as T) : initial;
    } catch {
      return initial;
    }
  });

  useEffect(() => {
    try {
      window.sessionStorage.setItem(key, JSON.stringify(value));
    } catch {
      // storage unavailable (private browsing, quota, ...) — state just won't persist
    }
  }, [key, value]);

  return [value, setValue] as const;
}

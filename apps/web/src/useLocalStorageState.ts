import { useEffect, useState } from "react";

/** Same as useState, but the value survives a page reload via localStorage. */
export function useLocalStorageState<T>(key: string, initial: T) {
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = window.localStorage.getItem(key);
      return raw !== null ? (JSON.parse(raw) as T) : initial;
    } catch {
      return initial;
    }
  });

  useEffect(() => {
    try {
      window.localStorage.setItem(key, JSON.stringify(value));
    } catch {
      // storage unavailable (private browsing, quota, ...) — collapse state just won't persist
    }
  }, [key, value]);

  return [value, setValue] as const;
}

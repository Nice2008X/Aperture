import { createContext, useContext, useEffect, useMemo, type ReactNode } from "react";
import { useLocalStorageState } from "../useLocalStorageState.js";
import { translations, type Language, type TranslationKey } from "../i18n.js";

interface LanguageContextValue {
  language: Language;
  setLanguage: (l: Language) => void;
  t: (key: TranslationKey) => string;
}

const LanguageContext = createContext<LanguageContextValue | null>(null);

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [language, setLanguage] = useLocalStorageState<Language>("app:language", "en");

  useEffect(() => {
    document.documentElement.setAttribute("lang", language);
  }, [language]);

  const value = useMemo<LanguageContextValue>(() => {
    const dict = translations[language] ?? translations.en;
    return {
      language,
      setLanguage,
      t: (key: TranslationKey) => dict[key] ?? translations.en[key] ?? key,
    };
  }, [language, setLanguage]);

  return <LanguageContext.Provider value={value}>{children}</LanguageContext.Provider>;
}

export function useTranslation(): LanguageContextValue {
  const ctx = useContext(LanguageContext);
  if (!ctx) throw new Error("useTranslation must be used within a LanguageProvider");
  return ctx;
}

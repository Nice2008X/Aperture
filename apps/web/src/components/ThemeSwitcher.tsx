import { useEffect } from "react";
import { useLocalStorageState } from "../useLocalStorageState.js";
import { useTranslation } from "./LanguageContext.js";
import type { TranslationKey } from "../i18n.js";

export type Theme = "dark" | "light" | "pastel" | "sepia";

const THEMES: { id: Theme; labelKey: TranslationKey; swatchBg: string; swatchAccent: string }[] = [
  { id: "dark", labelKey: "theme.dark", swatchBg: "#12141c", swatchAccent: "#4f46e5" },
  { id: "light", labelKey: "theme.light", swatchBg: "#ffffff", swatchAccent: "#4338ca" },
  { id: "pastel", labelKey: "theme.pastel", swatchBg: "#faf0e1", swatchAccent: "#a04a2a" },
  { id: "sepia", labelKey: "theme.sepia", swatchBg: "#eee0c0", swatchAccent: "#7a4e1b" },
];

export function useTheme() {
  const [theme, setTheme] = useLocalStorageState<Theme>("app:theme", "dark");

  useEffect(() => {
    if (theme === "dark") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  return { theme, setTheme };
}

export function ThemeSwitcher({ theme, onChange }: { theme: Theme; onChange: (t: Theme) => void }) {
  const { t } = useTranslation();
  return (
    <div className="theme-switcher" role="radiogroup" aria-label={t("theme.colorTheme")}>
      {THEMES.map((entry) => (
        <button
          key={entry.id}
          className={"theme-swatch" + (theme === entry.id ? " selected" : "")}
          style={{ background: entry.swatchBg }}
          onClick={() => onChange(entry.id)}
          title={t(entry.labelKey)}
          aria-label={t(entry.labelKey)}
          aria-pressed={theme === entry.id}
        >
          <span className="theme-swatch-dot" style={{ background: entry.swatchAccent }} />
        </button>
      ))}
    </div>
  );
}

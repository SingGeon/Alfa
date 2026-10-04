import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import legacy from "./legacy.json";
import { extras } from "./extras";

// The EN/RO dictionary carried over from the original vanilla frontend
// (legacy.json, generated from the old vanilla frontend's i18n.js) plus keys added with
// the redesign (extras.ts). The old strings embedded emoji/arrow prefixes
// as poor-man's icons ("📈 ETH Dashboard", "☆ Pin", "Next →"); the new UI
// draws real icons, so those decorations are stripped once, here.
export type Lang = "en" | "ro";
type Dict = Record<string, string>;

const DECOR_PREFIX = /^[^\p{L}\p{N}("'$±+{-]+\s*/u;
const DECOR_SUFFIX = /\s*[→←]$/u;
function clean(dict: Dict): Dict {
  return Object.fromEntries(Object.entries(dict).map(([k, v]) => [k, v.replace(DECOR_PREFIX, "").replace(DECOR_SUFFIX, "")]));
}

const DICTS: Record<Lang, Dict> = {
  en: { ...clean((legacy as Record<Lang, Dict>).en), ...extras.en },
  ro: { ...clean((legacy as Record<Lang, Dict>).ro), ...extras.ro },
};

const STORAGE_KEY = "lang";

function readStoredLang(): Lang {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    if (v === "en" || v === "ro") return v;
  } catch {
    /* storage unavailable (private mode etc.) */
  }
  return "en";
}

export type TFn = (key: string, vars?: Record<string, string | number>) => string;

interface I18nValue {
  lang: Lang;
  setLang: (l: Lang) => void;
  t: TFn;
  locale: string;
}

const I18nContext = createContext<I18nValue | null>(null);

export function I18nProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(readStoredLang);

  const setLang = useCallback((l: Lang) => {
    setLangState(l);
    try {
      localStorage.setItem(STORAGE_KEY, l);
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    document.documentElement.lang = lang;
  }, [lang]);

  const value = useMemo<I18nValue>(() => {
    const dict = DICTS[lang];
    const t: TFn = (key, vars) => {
      let s = dict[key] ?? DICTS.en[key] ?? key;
      if (vars) for (const [k, v] of Object.entries(vars)) s = s.replaceAll(`{${k}}`, String(v));
      return s;
    };
    return { lang, setLang, t, locale: lang === "ro" ? "ro-RO" : "en-US" };
  }, [lang, setLang]);

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n(): I18nValue {
  const ctx = useContext(I18nContext);
  if (!ctx) throw new Error("useI18n outside I18nProvider");
  return ctx;
}

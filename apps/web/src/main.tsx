import React from "react";
import ReactDOM from "react-dom/client";
import { setApiBase } from "@aperture/api-client";
import { App } from "./App.js";
import { LanguageProvider } from "./components/LanguageContext.js";
import "./styles.css";

setApiBase(import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000");

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <LanguageProvider>
      <App />
    </LanguageProvider>
  </React.StrictMode>
);

import { StrictMode, lazy, Suspense } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, Link, Navigate, RouterProvider, useLocation, useSearchParams } from "react-router";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import "@fontsource-variable/geist";
import "@fontsource-variable/geist-mono";
import "./styles/index.css";
import { I18nProvider, useI18n } from "./i18n";
import { AppShell, Page } from "./components/Shell";
import { buttonClass } from "./components/ui";
import Landing from "./pages/Landing";

// Route-level code splitting: lightweight-charts only loads on chart pages.
const Dashboard = lazy(() => import("./pages/Dashboard"));
const Scout = lazy(() => import("./pages/Scout"));
const AssetDetail = lazy(() => import("./pages/AssetDetail"));
const Evaluation = lazy(() => import("./pages/eval/Evaluation"));
const History = lazy(() => import("./pages/eval/History"));
const Visual = lazy(() => import("./pages/eval/Visual"));
const Population = lazy(() => import("./pages/Population"));
const Fund = lazy(() => import("./pages/Fund"));

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 2_000, refetchOnWindowFocus: false, retry: 1 },
  },
});

function Lazy({ children }: { children: React.ReactNode }) {
  return (
    <Suspense
      fallback={
        <div className="flex h-[60vh] items-center justify-center">
          <div className="size-6 animate-spin rounded-full border-2 border-line-strong border-t-accent" />
        </div>
      }
    >
      {children}
    </Suspense>
  );
}

/** Old vanilla-frontend URLs keep working (bookmarks, shared links). */
function LegacyDetail() {
  const [p] = useSearchParams();
  const type = p.get("type");
  const id = p.get("id");
  return <Navigate to={type && id ? `/asset/${type}/${encodeURIComponent(id)}` : "/scout"} replace />;
}
function LegacyRedirect({ to }: { to: string }) {
  const { search, hash } = useLocation();
  return <Navigate to={`${to}${search}${hash}`} replace />;
}

function NotFound() {
  const { t } = useI18n();
  return (
    <Page>
      <div className="flex min-h-[50vh] flex-col items-center justify-center gap-6 text-center">
        <span className="font-display text-[120px] font-bold leading-none tracking-[-0.06em] text-bone/10">404</span>
        <p className="text-dim">{t("ui.notFound")}</p>
        <Link to="/" className={buttonClass("outline", "md")}>{t("ui.backHome")}</Link>
      </div>
    </Page>
  );
}

const router = createBrowserRouter([
  {
    element: <AppShell />,
    children: [
      { path: "/", element: <Landing /> },
      { path: "/dashboard", element: <Lazy><Dashboard /></Lazy> },
      { path: "/scout", element: <Lazy><Scout /></Lazy> },
      { path: "/asset/:type/:id", element: <Lazy><AssetDetail /></Lazy> },
      { path: "/evaluation", element: <Lazy><Evaluation /></Lazy> },
      { path: "/history", element: <Lazy><History /></Lazy> },
      { path: "/visual", element: <Lazy><Visual /></Lazy> },
      { path: "/population", element: <Lazy><Population /></Lazy> },
      { path: "/fund", element: <Lazy><Fund /></Lazy> },
      { path: "/index.html", element: <Navigate to="/" replace /> },
      { path: "/dashboard.html", element: <LegacyRedirect to="/dashboard" /> },
      { path: "/scout.html", element: <LegacyRedirect to="/scout" /> },
      { path: "/detail.html", element: <LegacyDetail /> },
      { path: "/evaluation.html", element: <LegacyRedirect to="/evaluation" /> },
      { path: "/history.html", element: <LegacyRedirect to="/history" /> },
      { path: "/visual.html", element: <LegacyRedirect to="/visual" /> },
      { path: "/population.html", element: <LegacyRedirect to="/population" /> },
      { path: "/fund.html", element: <LegacyRedirect to="/fund" /> },
      { path: "*", element: <NotFound /> },
    ],
  },
]);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <I18nProvider>
        <RouterProvider router={router} />
      </I18nProvider>
    </QueryClientProvider>
  </StrictMode>,
);

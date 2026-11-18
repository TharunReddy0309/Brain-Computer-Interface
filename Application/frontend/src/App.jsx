import EvaluatePage from "./pages/EvaluatePage.jsx";
import OverviewPage from "./pages/OverviewPage.jsx";
import ResearchPage from "./pages/ResearchPage.jsx";

export default function App() {
  const path = window.location.pathname.replace(/\/+$/, "") || "/";

  if (path === "/evaluate") return <EvaluatePage />;
  if (path === "/research") return <ResearchPage />;
  return <OverviewPage />;
}

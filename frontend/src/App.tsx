import { Navigate, Route, Routes } from "react-router-dom";
import AppHeader from "./components/AppHeader";
import ChatPage from "./pages/chat/ChatPage";
import DataPage from "./pages/data/DataPage";
import ReportsPage from "./pages/reports/ReportsPage";

export default function App() {
  return (
    <div className="flex h-screen flex-col">
      <AppHeader />
      <main className="min-h-0 flex-1">
        <Routes>
          <Route path="/" element={<Navigate to="/chat" replace />} />
          <Route path="/reports" element={<ReportsPage />} />
          <Route path="/data" element={<DataPage />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="*" element={<Navigate to="/chat" replace />} />
        </Routes>
      </main>
    </div>
  );
}

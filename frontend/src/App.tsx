import { lazy, Suspense } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import Landing from "./pages/Landing";
const Dashboard = lazy(() => import("./pages/Dashboard"));
const StudyWorkspace = lazy(() => import("./pages/StudyWorkspace"));
const Analytics = lazy(() => import("./pages/Analytics"));
export default function App() {
  return <Suspense fallback={<p role="status" className="p-8 text-slate-400">Loading your workspace…</p>}><Routes>
    <Route path="/" element={<Landing />} />
    <Route path="/login" element={<Navigate to="/dashboard" replace />} />
    <Route path="/signup" element={<Navigate to="/dashboard" replace />} />
    <Route path="/dashboard" element={<Dashboard />} />
    <Route path="/workspace" element={<StudyWorkspace />} />
    <Route path="/analytics" element={<Analytics />} />
    <Route path="*" element={<Landing />} />
  </Routes></Suspense>;
}

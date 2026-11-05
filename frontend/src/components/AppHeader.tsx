import { Link, NavLink } from "react-router-dom";
import { BookOpen } from "lucide-react";
export default function AppHeader() {
  return <header className="app-header">
    <Link to="/dashboard" className="wordmark"><span className="brand-mark"><BookOpen size={21} aria-hidden="true" /></span><span>CRAG<span className="wordmark-sub">Study studio</span></span></Link>
    <nav className="app-nav" aria-label="Main navigation"><NavLink to="/dashboard">Library</NavLink><NavLink to="/workspace">Study room</NavLink><NavLink to="/analytics">Insights</NavLink></nav>
  </header>;
}

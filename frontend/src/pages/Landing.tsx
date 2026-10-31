import { Link } from "react-router-dom";
import { ArrowUpRight, BookOpen, FileText, Quote, ScanLine, Layers, Check } from "lucide-react";
export default function Landing() {
  return <div className="landing">
    <nav className="landing-nav"><Link to="/" className="wordmark"><span className="brand-mark"><BookOpen size={22} /></span><span>CRAG<span className="wordmark-sub">STUDY STUDIO</span></span></Link><Link to="/dashboard" className="secondary">Enter your workspace <ArrowUpRight size={16} /></Link></nav>
    <main><section className="landing-hero">
      <div><p className="eyebrow">A little clarity. A lot more learning.</p><h1>Your notes.<br />A clearer <em>mind.</em></h1><p className="hero-description">Turn a pile of pages into a place to think. Explore your notes, untangle tricky topics, and trace answers back to the source.</p><div className="hero-actions"><Link to="/dashboard" className="primary">Open your study room <ArrowUpRight size={18} /></Link><a href="#how-it-works" className="text-slate-300 text-sm">Take a look around ↓</a></div><p className="hero-caption">Typed notes · Handwritten pages · Your own pace</p></div>
      <div className="note-scene" aria-label="Illustrative example of studying with source citations"><div className="paper-back"><span>01 / THE NOTEBOOK</span><h2>Make room<br />for understanding.</h2><div className="paper-rule" /><div className="paper-rule" /><div className="paper-rule short" /><span className="paper-scribble">connect the ideas ↗</span></div><div className="example-answer"><div className="example-label"><Quote size={18} /> A MOMENT OF CLARITY <span>EXAMPLE</span></div><h3>What makes a stack different?</h3><p>A stack follows <strong>last in, first out</strong>. Think of a pile of books: the last one placed on top is the first one you take off.</p><div className="example-source"><FileText size={15} /> Example notes · page 12</div></div><span className="scene-tag"><Check size={15} /> Keep the source in sight</span></div>
    </section><section id="how-it-works" className="features-section"><div className="section-intro"><p className="eyebrow">FROM UPLOAD TO UNDERSTANDING</p><h2>A calmer way to study.</h2><p>One space for your material, your questions, and the evidence behind each answer.</p></div><div className="feature-grid">{[
      {icon:ScanLine, number:'01', title:'Bring your pages', text:'Upload typed or handwritten material. See which pages are readable and which need a closer look.'},
      {icon:Layers, number:'02', title:'Find your rhythm', text:'Ask a question, make a revision sheet, or practise with a quiz. Choose the mode that fits your session.'},
      {icon:Quote, number:'03', title:'Follow the evidence', text:'Open cited source pages and review checking results. Missing details and uncertain handwriting stay visible.'},
    ].map(f=><article key={f.number} className="feature-card"><div><f.icon size={24} /><span>{f.number}</span></div><h3>{f.title}</h3><p>{f.text}</p></article>)}</div></section></main>
    <footer className="landing-footer"><span>CRAG / STUDY STUDIO</span><span>Stay curious. Check the source.</span></footer>
  </div>;
}

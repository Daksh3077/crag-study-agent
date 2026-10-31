import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";

function normalizeMath(content: string): string {
  // Models also emit LaTeX bracket delimiters. Preserve literal examples in code.
  return content.split(/(```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]*`)/g).map((part, index) =>
    index % 2 ? part : part
      .replace(/\\\[([\s\S]*?)\\\]/g, (_match, math: string) => "\n$$\n" + math.trim() + "\n$$\n")
      .replace(/\\\(([^\n]*?)\\\)/g, (_match, math: string) => "$" + math + "$"),
  ).join("");
}

export default function AnswerContent({ content }: { content: string }) {
  return <div className="answer-content"><ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]} components={{
    a: ({ children, ...props }) => <a {...props} target="_blank" rel="noopener noreferrer">{children}</a>,
  }}>{normalizeMath(content)}</ReactMarkdown></div>;
}

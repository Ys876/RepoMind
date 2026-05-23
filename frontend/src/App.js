import { useState } from "react";
import axios from "axios";

function App() {
  const [githubUrl, setGithubUrl] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [indexed, setIndexed] = useState(false);
  const [status, setStatus] = useState("");

  const handleIndex = async () => {
    setLoading(true);
    setStatus("Cloning and indexing repo...");
    try {
      const res = await axios.post("http://localhost:8000/index", {
        github_url: githubUrl,
      });
      setIndexed(true);
      setStatus(res.data.message);
    } catch (err) {
      setStatus("Error indexing repo. Check the URL and try again.");
    }
    setLoading(false);
  };

  const handleAsk = async () => {
    setLoading(true);
    setAnswer("");
    setStatus("Thinking...");
    try {
      const res = await axios.post("http://localhost:8000/ask", {
        question: question,
      });
      setAnswer(res.data.answer);
      setStatus("");
    } catch (err) {
      setStatus("Error getting answer.");
    }
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: "800px", margin: "40px auto", padding: "0 20px", fontFamily: "monospace" }}>
      <h1 style={{ fontSize: "24px", marginBottom: "8px" }}>RepoMind</h1>
      <p style={{ color: "#666", marginBottom: "32px" }}>Ask natural language questions about any GitHub repository</p>

      {/* index section */}
      <div style={{ marginBottom: "24px" }}>
        <p style={{ marginBottom: "8px", fontWeight: "bold" }}>1. Paste a GitHub URL</p>
        <div style={{ display: "flex", gap: "8px" }}>
          <input
            type="text"
            placeholder="https://github.com/user/repo"
            value={githubUrl}
            onChange={(e) => setGithubUrl(e.target.value)}
            style={{ flex: 1, padding: "8px 12px", border: "1px solid #ccc", borderRadius: "4px", fontFamily: "monospace" }}
          />
          <button
            onClick={handleIndex}
            disabled={loading || !githubUrl}
            style={{ padding: "8px 16px", background: "#000", color: "#fff", border: "none", borderRadius: "4px", cursor: "pointer" }}
          >
            {loading && !indexed ? "Indexing..." : "Index Repo"}
          </button>
        </div>
      </div>

      {/* ask section */}
      <div style={{ marginBottom: "24px" }}>
        <p style={{ marginBottom: "8px", fontWeight: "bold" }}>2. Ask a question</p>
        <div style={{ display: "flex", gap: "8px" }}>
          <input
            type="text"
            placeholder="How does routing work?"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            disabled={!indexed}
            style={{ flex: 1, padding: "8px 12px", border: "1px solid #ccc", borderRadius: "4px", fontFamily: "monospace", opacity: indexed ? 1 : 0.5 }}
          />
          <button
            onClick={handleAsk}
            disabled={loading || !indexed || !question}
            style={{ padding: "8px 16px", background: "#000", color: "#fff", border: "none", borderRadius: "4px", cursor: "pointer" }}
          >
            {loading && indexed ? "Thinking..." : "Ask"}
          </button>
        </div>
      </div>

      {/* status */}
      {status && (
        <p style={{ color: "#666", marginBottom: "16px", fontSize: "14px" }}>{status}</p>
      )}

      {/* answer */}
      {answer && (
        <div style={{ background: "#f5f5f5", padding: "16px", borderRadius: "4px", whiteSpace: "pre-wrap", lineHeight: "1.6" }}>
          <p style={{ fontWeight: "bold", marginBottom: "8px" }}>Answer</p>
          {answer}
        </div>
      )}
    </div>
  );
}

export default App;
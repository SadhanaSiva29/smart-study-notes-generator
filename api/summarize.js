const res = await fetch("/api/summarize", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text })
});
const data = await res.json();

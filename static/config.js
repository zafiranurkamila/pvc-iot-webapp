// Base URL of the inference API. Served by server.py itself (local PC or Hugging Face Space): same origin.
// Served by Vercel (static dashboard only): the Hugging Face Space that runs the model.
window.PVC_API_BASE = location.hostname.endsWith(".vercel.app")
  ? "https://zafiranurkamila-pvc-origin-monitor.hf.space"
  : "";

const form = document.querySelector("#chat-form");
const input = document.querySelector("#message");
const send = document.querySelector("#send");
const messages = document.querySelector("#messages");
const status = document.querySelector("#status");

function setStatus(label, state) {
  status.lastChild.textContent = ` ${label}`;
  status.className = `status ${state || ""}`;
}

function addMessage(role, text, meta = "") {
  const welcome = document.querySelector(".welcome");
  if (welcome) welcome.remove();
  const row = document.createElement("div");
  row.className = `message ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  row.appendChild(bubble);
  if (meta) {
    const info = document.createElement("div");
    info.className = "meta";
    info.textContent = meta;
    bubble.appendChild(info);
  }
  messages.appendChild(row);
  messages.scrollTop = messages.scrollHeight;
  return row;
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error();
    setStatus("Local · Ready", "ready");
  } catch {
    setStatus("Server unavailable", "error");
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message || send.disabled) return;
  addMessage("user", message);
  input.value = "";
  input.style.height = "auto";
  send.disabled = true;
  setStatus("Thinking", "");
  const typing = addMessage("assistant", "Working…");
  typing.querySelector(".bubble").classList.add("typing");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({message})
    });
    const data = await response.json();
    typing.remove();
    if (!response.ok) throw new Error(data.error || "Request failed");
    addMessage(
      "assistant",
      data.response,
      `${data.task_type} · ${data.difficulty} · ${data.tier}${data.escalated ? " · escalated" : ""}`
    );
    setStatus("Local · Ready", "ready");
  } catch (error) {
    typing.remove();
    addMessage("assistant", `Error: ${error.message}`);
    setStatus("Request failed", "error");
  } finally {
    send.disabled = false;
    input.focus();
  }
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
});

checkHealth();

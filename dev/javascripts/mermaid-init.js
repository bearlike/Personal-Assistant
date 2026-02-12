const renderMermaid = () => {
  const blocks = document.querySelectorAll("pre.mermaid code");
  blocks.forEach((code) => {
    const container = document.createElement("div");
    container.className = "mermaid";
    container.textContent = code.textContent || "";
    const pre = code.parentElement;
    if (pre) {
      pre.replaceWith(container);
    }
  });
  mermaid.initialize({ startOnLoad: true });
};

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", renderMermaid);
} else {
  renderMermaid();
}

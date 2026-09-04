// SPDX-FileCopyrightText: PyPSA Contributors
//
// SPDX-License-Identifier: MIT

// Strip doctest prompts (>>> / ...) and output lines from the copy button, so
// users copy runnable code while the source keeps >>> for the doctest suite.
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".highlight").forEach((block) => {
    const btn = block.querySelector(".md-clipboard");
    const code = block.querySelector("code");
    if (!btn || !code) return;

    const raw = code.innerText;
    if (!/^\s*>>> /m.test(raw)) return; // only REPL blocks

    // Capture phase + stopPropagation runs before ClipboardJS's delegated handler.
    btn.addEventListener(
      "click",
      (e) => {
        e.stopPropagation();
        e.preventDefault();
        const cleaned = raw
          .split("\n")
          .filter((l) => /^\s*(>>>|\.\.\.) /.test(l))
          .map((l) => l.replace(/^\s*(>>>|\.\.\.) ?/, ""))
          .join("\n");
        navigator.clipboard.writeText(cleaned + "\n");
      },
      true,
    );
  });
});

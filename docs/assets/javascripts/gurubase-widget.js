// SPDX-FileCopyrightText: PyPSA Contributors
//
// SPDX-License-Identifier: MIT

document.addEventListener("DOMContentLoaded", () => {
  // Load the GuruBase widget
  const guruScript = document.createElement("script");
  guruScript.src = "https://widget.gurubase.io/widget.latest.min.js";
  guruScript.defer = true;
  guruScript.id = "guru-widget-id";

  // Configure widget settings
  const widgetSettings = {
    "data-widget-id": "3a8zez8jVEVRucKLlyrKs0hKb2-xxpTg4sZkfwY17JM",
    "data-text": "Ask AI",
    "data-margins": JSON.stringify({ bottom: "30px", right: "30px" }),
    "data-light-mode": "auto",
    "data-name": "PyPSA",
    "data-icon-url": "https://raw.githubusercontent.com/PyPSA/PyPSA/master/docs/assets/logo/logo.svg",
    "data-bg-color": "#D10A49", 
    "data-overlap-content": "true",
    "data-tooltip":
      "Ask questions about PyPSA. Please note that questions and answers are visible anonymously to the PyPSA team and GuruBase.",
    "data-tooltip-side": "left",
 
  };   

  // Add widget settings as data attributes
  Object.entries(widgetSettings).forEach(([key, value]) => {
    guruScript.setAttribute(key, value);
  });

  document.body.appendChild(guruScript);
});

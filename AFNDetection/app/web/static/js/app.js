/* ─────────────────────────────────────────────────────────
   AFNDetection — App JS (Canvas BG + Interactions)
   ───────────────────────────────────────────────────────── */

/* ══ CANVAS PARTICLE NETWORK ══ */
(function initCanvas() {
  const canvas = document.getElementById("bgCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  let W, H, particles;
  const COUNT = 55;
  const COLORS = ["rgba(245,200,66,", "rgba(0,212,180,", "rgba(232,51,74,"];

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function makeParticle() {
    const col = COLORS[Math.floor(Math.random() * COLORS.length)];
    return {
      x: Math.random() * W,
      y: Math.random() * H,
      r: Math.random() * 1.8 + 0.4,
      vx: (Math.random() - 0.5) * 0.28,
      vy: (Math.random() - 0.5) * 0.28,
      col,
      alpha: Math.random() * 0.4 + 0.1,
    };
  }

  function init() {
    resize();
    particles = Array.from({ length: COUNT }, makeParticle);
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);

    /* Draw connecting lines */
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const a = particles[i], b = particles[j];
        const dx = a.x - b.x, dy = a.y - b.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 130) {
          const opacity = (1 - dist / 130) * 0.12;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.strokeStyle = `rgba(245,200,66,${opacity})`;
          ctx.lineWidth = 0.6;
          ctx.stroke();
        }
      }
    }

    /* Draw particles */
    particles.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = p.col + p.alpha + ")";
      ctx.fill();

      p.x += p.vx;
      p.y += p.vy;
      if (p.x < -10) p.x = W + 10;
      if (p.x > W + 10) p.x = -10;
      if (p.y < -10) p.y = H + 10;
      if (p.y > H + 10) p.y = -10;
    });

    requestAnimationFrame(draw);
  }

  init();
  draw();
  window.addEventListener("resize", resize);
})();

/* ══ REVEAL ON SCROLL ══ */
(function initReveal() {
  const els = document.querySelectorAll(".reveal");
  const io = new IntersectionObserver(
    entries => {
      entries.forEach((e, i) => {
        if (e.isIntersecting) {
          setTimeout(() => e.target.classList.add("visible"), i * 80);
          io.unobserve(e.target);
        }
      });
    },
    { threshold: 0.12 }
  );
  els.forEach(el => io.observe(el));
})();

/* ══ DOM REFS ══ */
const analyzeBtn       = document.getElementById("analyzeBtn");
const btnText          = document.getElementById("btnText");
const newsInput        = document.getElementById("newsInput");
const resultCard       = document.getElementById("resultCard");
const predictionLabel  = document.getElementById("predictionLabel");
const verdictIcon      = document.getElementById("verdictIcon");
const verdictSub       = document.getElementById("verdictSub");
const probabilitiesEl  = document.getElementById("probabilities");
const disclaimerText   = document.getElementById("disclaimerText");
const warningBox       = document.getElementById("warningBox");
const warningText      = document.getElementById("warningText");
const errorBox         = document.getElementById("errorBox");
const errorText        = document.getElementById("errorText");
const charCount        = document.getElementById("charCount");

/* ── CHAR COUNTER ── */
newsInput.addEventListener("input", () => {
  const n = newsInput.value.length;
  charCount.textContent = n.toLocaleString("ar-MA");
  charCount.parentElement.classList.toggle("has-text", n > 0);
});

/* ══ HELPERS ══ */
function hideMsg(box, txt) {
  box.classList.add("hidden");
  if (txt) txt.textContent = "";
}
function showMsg(box, txt, msg) {
  txt.textContent = msg;
  box.classList.remove("hidden");
}

function getClass(labelEn) {
  const k = (labelEn || "").toLowerCase().trim();
  if (k === "fake") return "fake";
  if (k === "real") return "real";
  return "other";
}

/* ══ RENDER PROBABILITIES ══ */
function renderProbabilities(items) {
  probabilitiesEl.innerHTML = "";
  items.forEach(item => {
    const cls = getClass(item.label_en);
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <div class="prob-top">
        <span>${item.label_ar}</span>
        <span class="prob-pct ${cls}">${item.percentage.toFixed(1)}٪</span>
      </div>
      <div class="prob-track">
        <div class="prob-fill ${cls}" data-width="${item.percentage}"></div>
      </div>`;
    probabilitiesEl.appendChild(row);
  });

  /* Animate bars */
  requestAnimationFrame(() => {
    probabilitiesEl.querySelectorAll(".prob-fill").forEach((fill, i) => {
      setTimeout(() => {
        fill.style.width = fill.dataset.width + "%";
      }, i * 100 + 80);
    });
  });
}

/* ══ VERDICT DISPLAY ══ */
function showVerdict(predictionAr, labelEn) {
  const cls = getClass(labelEn);
  predictionLabel.textContent = predictionAr;
  predictionLabel.className = `verdict-label is-${cls}`;

  const iconWrap = verdictIcon.parentElement;
  iconWrap.className = `verdict-icon-wrap is-${cls}`;

  if (cls === "fake") {
    verdictIcon.textContent = "✕";
    verdictSub.textContent  = "FAKE · مزيف";
  } else if (cls === "real") {
    verdictIcon.textContent = "✓";
    verdictSub.textContent  = "REAL · حقيقي";
  } else {
    verdictIcon.textContent = "?";
    verdictSub.textContent  = "UNKNOWN";
  }
}

/* ══ MAIN PREDICTION ══ */
async function runPrediction() {
  hideMsg(warningBox, warningText);
  hideMsg(errorBox, errorText);
  resultCard.classList.add("hidden");

  const text = newsInput.value.trim();
  if (!text) {
    showMsg(errorBox, errorText, "يرجى إدخال نص قبل التحليل.");
    return;
  }

  analyzeBtn.disabled = true;
  analyzeBtn.classList.add("btn-loading");
  btnText.textContent = "جاري التحليل...";

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const payload = await res.json();

    if (!res.ok) {
      showMsg(errorBox, errorText, payload.error || "حدث خطأ أثناء التحليل.");
      return;
    }
    if (payload.warning) showMsg(warningBox, warningText, payload.warning);

    showVerdict(payload.prediction_ar, payload.prediction_en || "");
    disclaimerText.textContent = payload.disclaimer || "";
    renderProbabilities(payload.probabilities || []);
    resultCard.classList.remove("hidden");

  } catch {
    showMsg(errorBox, errorText, "تعذر الاتصال بالخادم. تأكد أن التطبيق يعمل.");
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.classList.remove("btn-loading");
    btnText.textContent = "تحليل النص";
  }
}

analyzeBtn.addEventListener("click", runPrediction);

/* ── Allow Ctrl+Enter to submit ── */
newsInput.addEventListener("keydown", e => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") runPrediction();
});
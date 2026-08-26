// ---------------------------------------------------------------------------
// Theme switcher — Light / Dark / System, remembered across visits
// ---------------------------------------------------------------------------
const THEME_KEY = "creditxcore-theme";
const htmlEl = document.documentElement;
const themeButtons = document.querySelectorAll("#themeSwitch button");

function applyTheme(choice) {
  htmlEl.setAttribute("data-theme", choice);
  themeButtons.forEach(btn => {
    btn.classList.toggle("active", btn.dataset.themeChoice === choice);
  });
  localStorage.setItem(THEME_KEY, choice);
}

themeButtons.forEach(btn => {
  btn.addEventListener("click", () => applyTheme(btn.dataset.themeChoice));
});

// On load: use saved preference, or default to "system"
applyTheme(localStorage.getItem(THEME_KEY) || "system");

// ---------------------------------------------------------------------------
// Creditxore frontend logic — talks to the FastAPI backend's /predict endpoint.
// Change API_URL below if your backend runs on a different host/port.
// ---------------------------------------------------------------------------
const API_URL = "https://creditxcore.onrender.com/predict";

const form = document.getElementById('applicantForm');
const submitBtn = document.getElementById('submitBtn');
const verdictBody = document.getElementById('verdictBody');

function val(id) {
  return document.getElementById(id).value;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  submitBtn.disabled = true;
  submitBtn.textContent = "Assessing...";

  const income = parseFloat(val('income'));
  const loan = parseFloat(val('loan'));

  const payload = {
    "Income": income,
    "Age": parseFloat(val('age')),
    "Loan": loan,
    "Loan to Income": Math.round((loan / income) * 10000) / 10000,
    "Employment_Type": val('employment'),
    "Credit_History_Years": parseFloat(val('creditHistory')),
    "Num_Existing_Loans": parseInt(val('numLoans')),
    "Late_Payments_24M": parseInt(val('latePayments')),
    "Credit_Utilization_Pct": parseFloat(val('utilization')),
    "EMI_Burden_Ratio_Pct": parseFloat(val('emiBurden')),
    "Num_Dependents": parseInt(val('dependents')),
    "City_Tier": val('cityTier'),
    "Savings_Balance": parseFloat(val('savings'))
  };

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error("Server responded with " + res.status);
    const data = await res.json();
    renderVerdict(data);
  } catch (err) {
    verdictBody.innerHTML = `<div class="error-box">Could not reach backend at ${API_URL}.<br>Make sure "uvicorn main:app --reload --port 8000" is running.<br><br>${err.message}</div>`;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Assess Risk";
  }
});

function renderVerdict(data) {
  const isHighRisk = data.decision.toUpperCase().includes("HIGH");
  const stampClass = isHighRisk ? "review" : "approve";
  const stampLabel = isHighRisk ? "Review Required" : "Approved";
  const pct = (data.default_probability * 100).toFixed(2);

  const reasonsHtml = data.top_reasons.map(r => {
    const dirClass = r.direction.includes("increase") ? "up" : "down";
    const arrow = r.direction.includes("increase") ? "↑" : "↓";
    return `<div class="reason-row">
      <span class="reason-name">${r.feature}</span>
      <span class="reason-dir ${dirClass}">${arrow} ${r.direction}</span>
    </div>`;
  }).join("");

  verdictBody.innerHTML = `
    <div class="stamp-wrap">
      <div class="stamp ${stampClass}">${stampLabel}</div>
      <div class="prob-readout">
        <div class="num">${pct}%</div>
        <div class="label">Default Probability</div>
      </div>
    </div>
    <div class="reasons">
      <h3>Top Contributing Factors</h3>
      ${reasonsHtml}
    </div>
  `;
}

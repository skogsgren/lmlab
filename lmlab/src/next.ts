const nextTableBody = document.getElementById("nextTableBody");
const nextTextBox = document.getElementById("nextTextBox") as HTMLInputElement;
const nextSlider = document.getElementById("nextSlider") as HTMLInputElement;
const nextSliderText = document.getElementById("nextSliderText");
const nextStatusText = document.getElementById("nextStatusText");
const nextTokenCount = document.getElementById("nextTokenCount");
const nextRow = document.getElementById("nextRow");

type TokenPrediction = {
  token: string;
  id: number;
  prob: number;
  temp_prob: number;
};

type TokenPredictionResponse = {
  input: {
    text: string;
    temp: number;
    truncated: boolean;
    tokens: string[];
  };
  topk: TokenPrediction[];
};

function renderTokens(tokens: string[]) {
    if (!nextRow) return;
    nextRow.innerHTML = "";
    tokens.forEach((token, i) => {
        let el = document.createElement("div");
        el.className = "tokBox";
        el.innerHTML = `<span class="tokText">${token}</span>`;
        nextRow.appendChild(el);
    });
}

function renderTopK(topk: TokenPrediction[], truncated: boolean) {
    if (!nextTableBody || !nextSlider || !nextStatusText) return;
    nextTableBody.innerHTML = "";
    setTruncated(truncated);
    if (truncated) return;

    for (const item of topk) {
        const tr = document.createElement("tr");
        tr.className = "nextTokenRow";
        if (!truncated) {
            tr.addEventListener("click", () => { updateText(item.token); });
        } else {
            tr.style.opacity = "0.5";
            tr.style.cursor = "default";
        }
        const td1 = document.createElement("td");
        const tokLabel = item.token === "<0x0A>" ? "<|newline|>" : item.token;
        td1.innerHTML = `<span class="probTok">${tokLabel.replace(/ /g, "·")}</span>`;
        const td2 = document.createElement("td");
        const pct = (item.prob * 100).toFixed(1);
        td2.innerHTML = `<div class="probCell"><div class="probBar"><div class="probBarFill" style="width:${pct}%"></div></div><span class="probPct">${pct}%</span></div>`;
        tr.append(td1, td2);
        nextTableBody.appendChild(tr);
    }
}

function scrub_whitespace(x: string) {
    if (x === "<0x0A>") return "\n";
    const whitespace_markers = ["▁", "Ġ"]
    for (const w of whitespace_markers) x = x.replace(w, " ");
    return x;
}

function updateText(new_token: string) {
    if (!nextTextBox) return;
    let text = nextTextBox.value + scrub_whitespace(new_token);
    nextTextBox.value = text;
    nextRequest();
}

async function nextRequest() {
    if (!nextStatusText || !nextTokenCount) return;
    nextStatusText.textContent = "Laddar...";
    let inp = nextTextBox.value;
    const temp = parseFloat(nextSlider.value);
    let data: TokenPredictionResponse;
    try {
        let resp = await fetch("/next", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ inp, temp }),
        });
        if (!resp.ok) {
            let msg = await resp;
            throw new Error(`HTTP ${resp.status}: ${msg || "request failed"}`);
        }
        data = await resp.json();
    } catch (err: unknown) {
        if (err instanceof Error) {
            nextStatusText.textContent = `Fel: ${err.message}`;
        } else {
            nextStatusText.textContent = "Fel: Okänt fel";
        }
        return;
    }
    renderTokens(data.input.tokens);
    renderTopK(data.topk, data.input.truncated);

    nextTokenCount.textContent = data.input.tokens.length.toString();
    nextStatusText.textContent = "Redo.";

    return data;
}

function sample(data: TokenPrediction[]): string {
    const total = data.reduce((s, d) => s + d.temp_prob, 0);
    const r = Math.random() * total;
    let cum = 0;
    for (const item of data) {
        cum += item.temp_prob;
        if (r < cum) return item.token;
    }
    return data[data.length - 1]!.token;
}

async function nextSample() {
    let data = await nextRequest();
    if (!data) return;
    let sampled_token = scrub_whitespace(sample(data.topk))
    updateText(sampled_token);
}

function updateSliderValue() {
    if (!nextSlider || !nextSliderText) return;
    nextSliderText.textContent = parseFloat(nextSlider.value).toFixed(1);
}

function setTruncated(truncated: boolean) {
    const els = [nextTextBox, nextSlider] as (HTMLElement | null)[];
    const btns = ["nextProbButton", "nextSampleButton"].map(id => document.getElementById(id));

    if (truncated) {
        els.forEach(el => el?.setAttribute("disabled", ""));
        btns.forEach(btn => btn?.setAttribute("disabled", ""));
        if (nextStatusText) nextStatusText.textContent = "Tokengräns nådd";
        if (nextTableBody) nextTableBody.closest("table")!.style.display = "none";

        const existing = document.getElementById("limitBanner");
        if (!existing) {
            const banner = document.createElement("div");
            banner.id = "limitBanner";
            banner.innerHTML = `
                <p>Tokengränsen har nåtts. Rensa texten för att börja om.</p>
                <button id="resetButton">Rensa och börja om</button>
            `;
            nextTableBody!.closest(".nextTableWrapper")!.after(banner);
            document.getElementById("resetButton")!.addEventListener("click", resetState);
        }
    } else {
        els.forEach(el => el?.removeAttribute("disabled"));
        btns.forEach(btn => btn?.removeAttribute("disabled"));
        if (nextStatusText) nextStatusText.textContent = "Redo.";
        if (nextTableBody) nextTableBody.closest("table")!.style.display = "";
        document.getElementById("limitBanner")?.remove();
    }
}

function resetState() {
    if (nextTextBox) nextTextBox.value = "";
    if (nextRow) nextRow.innerHTML = "";
    if (nextTokenCount) nextTokenCount.textContent = "0";
    setTruncated(false);
    if (nextTableBody) nextTableBody.innerHTML = "";
}

window.addEventListener("load", function () {
    if (nextSlider && nextSliderText) {
        updateSliderValue();
        nextSlider.addEventListener("input", updateSliderValue);
    }

    let nextProbButton = document.getElementById("nextProbButton");
    if (nextProbButton) nextProbButton.addEventListener("click", nextRequest);

    let nextSampleButton = document.getElementById("nextSampleButton");
    if (nextSampleButton && nextSlider) nextSampleButton.addEventListener("click", nextSample);
});

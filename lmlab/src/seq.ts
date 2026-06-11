const seqActiveNote = document.getElementById("seqActiveNote");
const seqStatusText = document.getElementById("seqStatusText");
const seqTableBody = document.getElementById("seqTableBody");
const seqTableWrapper = document.getElementById("seqTableWrapper");
const seqTextBox = document.getElementById("seqTextBox") as HTMLInputElement;
const seqRow = document.getElementById("seqRow");
const seqSummaryNote = document.getElementById("seqSummaryNote");

let seqData: SeqResponse;
let activeTokenIndex: number = 0;
let seqState: boolean = false;

type Token = string;

interface TopKToken {
  token: string;
  prob: number;
}

type TopK = TopKToken[][];

interface ActualToken {
  token: string;
  prob: number;
  logprob: number;
  pos: number;
}

interface Summary {
  log_prob: number;
  log10_prob: number;
  avg_log_prob: number;
  num_predicted: number;
}

interface SeqResponse {
  tokens: Token[];
  topk: TopK;
  actual: ActualToken[];
  summary: Summary;
}

function renderTokenLabel(token: string): string {
    return (token === "<0x0A>" ? "<|newline|>" : token);
}


function renderTokens(tokens: Token[]) {
    if (!seqRow) return;
    seqRow.innerHTML = "";
    tokens.forEach((token, i) => {
        const el = document.createElement("div");
        el.className = "seqBox";
        if (i === activeTokenIndex) {
            el.classList.add("seqActiveToken");
        }
        el.innerHTML = `<span class="tokText">${token}</span>`;
        el.addEventListener("click", () => {
            setActiveToken(i);
            activeTokenIndex = i;
            renderTokens(tokens);
        });
        seqRow.appendChild(el);
    });
}

function renderSummary(summary: Summary) {
    if (!seqSummaryNote) return;
    seqSummaryNote.innerHTML = `
        <h3>Sammanfattning över ${summary.num_predicted} tokens</h3>
        <ul>
        <li><i>log10 P(text)</i> = ${summary.log10_prob.toFixed(3)}</li>
        <li><i>Sannolikhet</i> ≈ ${Math.exp(summary.log10_prob).toFixed(40)}</li>
        </ul>
    `;
}


function setActiveToken(i: number) {
    if (!seqState || !seqTableBody) return;
    seqTableBody.innerHTML = "";
    const distribution = seqData.topk[i];
    if (!distribution) return;
    distribution.forEach(({ token, prob }) => {
        const tr = document.createElement("tr");
        tr.className = "nextTokenRow";
        const pct = (prob * 100).toFixed(1);
        const td1 = document.createElement("td");
        td1.innerHTML = `<span class="probTok">${renderTokenLabel(token)}</span>`;
        const td2 = document.createElement("td");
        td2.innerHTML = `<div class="probCell"><div class="probBar"><div class="probBarFill" style="width:${pct}%"></div></div><span class="probPct">${pct}%</span></div>`;
        tr.append(td1, td2);
        seqTableBody.appendChild(tr);
    });

    if (!seqActiveNote) return;
    const a = seqData.actual[i];
    seqActiveNote.textContent = a
        ? `Valt token: "${a.token}" | P = ${a.prob.toFixed(6)} | log P = ${a.logprob.toFixed(3)}`
        : "";
}


async function seqRequest() {
    seqState = false;
    if (!seqStatusText || !seqTextBox || !seqTableWrapper) {
        throw new Error("Skadad HTML.");
    }

    seqTableWrapper.style.display = "block";
    seqStatusText.textContent = "Laddar..."

    let data: SeqResponse;
    let inp = seqTextBox.value;
    try {
        let resp = await fetch("/seq", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ inp }),
        });
        if (!resp.ok) {
            let msg = await resp;
            throw new Error(`HTTP ${resp.status}: ${msg || "request failed"}`);
        }
        data = await resp.json();
    } catch (err: unknown) {
        if (err instanceof Error) {
            seqStatusText.textContent = `Fel: ${err.message}`;
        } else {
            seqStatusText.textContent = "Fel: Okänt fel";
        }
        return;
    }

    seqData = data;
    seqState = true;
    renderTokens(seqData.tokens);
    renderSummary(seqData.summary);
    setActiveToken(0);
    seqStatusText.textContent = "Redo."
}

function seqCount() {
    const seqTextCount = document.getElementById("seqTextCount");
    if (!seqTextCount) return;
    seqTextCount.textContent = seqTextBox.value.length.toString();
}

window.addEventListener("load", function () {
    let seqButton = document.getElementById("seqButton");
    if (seqButton) seqButton.addEventListener("click", seqRequest);
    if (seqTextBox) seqTextBox.addEventListener("input", seqCount);
    seqCount();
});

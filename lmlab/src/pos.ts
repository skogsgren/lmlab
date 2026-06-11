const posSentence = document.getElementById("posSentence");
const posTextBox = document.getElementById("posTextBox") as HTMLInputElement;
const posStatusText = document.getElementById("posStatusText");

type PosResponse = {
  inp: string;
  tags: string[];
  words: [string, string][];
};

let posData: PosResponse;
let posState: boolean = false;

async function posRequest() {
    posState = false;
    if (!posStatusText || !posSentence || !posTextBox) {
        throw new Error("Skadad HTML.");
    }
    posStatusText.textContent = "Laddar...";
    posSentence.innerHTML = "";

    let inp = posTextBox.value;
    try {
        let resp = await fetch("/pos", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ inp }),
        });
        if (!resp.ok) {
            let msg = await resp;
            throw new Error(`HTTP ${resp.status}: ${msg || "request failed"}`);
        }
        posData = await resp.json();
        posState = true;
    } catch (err: unknown) {
        if (err instanceof Error) {
            posStatusText.textContent = `Fel: ${err.message}`;
        } else {
            posStatusText.textContent = "Fel: Okänt fel";
        }
        return;
    }
    if (!posState) return;
    for (const [i, [word, pos]] of posData.words.entries()) {
        const opts = posData.tags.map(t => `<option value="${t}">${t}</option>`).join("");

        const token = document.createElement("div");
        token.className = "posToken";

        token.innerHTML = `
            <div class="posWord">${word}</div>

            <select data-idx="${i}">
                <option value="" disabled selected>---</option>
                ${opts}
            </select>

            <div class="posModelTag"></div>
        `;

        posSentence?.appendChild(token);
    }

    posStatusText.textContent = "Redo.";
}

function posCollect() {
    if (!posStatusText || !posSentence || !posData) return;
    if (!posState) return;

    const selects =
        posSentence.querySelectorAll<HTMLSelectElement>("select");

    const incomplete =
        [...selects].some(s => !s.value);

    if (incomplete) {
        posStatusText.textContent =
            "Fyll i alla taggar först.";
        return;
    }

    let differences = 0;

    selects.forEach(select => {
        const idx = parseInt(select.dataset.idx!);
        const token = select.closest(".posToken");
        const modelTag = token?.querySelector<HTMLDivElement>(".posModelTag");
        const entry = posData.words[idx];
        if (!entry) return;
        const [, model] = entry;
        const manual = select.value;
        if (!modelTag || !token)
            return;
        modelTag.textContent =
            `Modell: ${model}`;
        modelTag.classList.add("visible");
        token.classList.remove("posDifferent");
        if (manual !== model) {
            token.classList.add("posDifferent");
            differences++;
        }
    });

    posStatusText.textContent =
        `${differences} skillnader mellan din annotering och modellen.`;
}

function posCount() {
    const posTextCount = document.getElementById("posTextCount");
    if (!posTextCount) return;
    posTextCount.textContent = posTextBox.value.length.toString();
}

window.addEventListener("load", function () {
    let posButton = document.getElementById("posButton");
    if (posButton) posButton.addEventListener("click", posRequest);
    let posCollectButton = document.getElementById("posCollectButton");
    if (posCollectButton) posCollectButton.addEventListener("click", posCollect);
    if (posTextBox) posTextBox.addEventListener("input", posCount);

    document.querySelectorAll<HTMLLIElement>("#posExampleList li").forEach(li => {
        li.addEventListener("click", () => {
            posTextBox.value = li.textContent ?? "";
            posCount();
            posRequest();
        });
    });

    posCount();
});

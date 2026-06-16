let tokTable = document.getElementById("tokTable");
let tokStatusText = document.getElementById("tokStatusText");
let tokTextBox = document.getElementById("tokTextBox") as HTMLInputElement;

type TokResponse = {
  [modelName: string]: [string, number][];
};

async function tokRequest() {
    let tokState = null;
    if (!tokStatusText || !tokTable || !tokTextBox) {
        throw new Error("Skadad HTML.");
    }
    const tbody = tokTable.querySelector("tbody");
    if (!tbody) {
        throw new Error("Skadad HTML.");
    }
    tbody.replaceChildren();

    tokStatusText.textContent = "Laddar...";
    let inp = tokTextBox.value;
    let data: TokResponse;
    try {
        let resp = await fetch("tok", {
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
            tokStatusText.textContent = `Fel: ${err.message}`;
        } else {
            tokStatusText.textContent = "Fel: Okänt fel";
        }
        return;
    }

    console.log(data);

    for (const [model, tokens] of Object.entries(data)) {
        const row = document.createElement("tr");

        const modelCell = document.createElement("td");
        modelCell.textContent = model;
        modelCell.className = "tokSpecText";

        const tokensCell = document.createElement("td");

        for (const [token, tokenId] of tokens) {
            const tokenBox = document.createElement("div");
            tokenBox.className = "tokBox";

            const tokenText = document.createElement("div");
            tokenText.className = "tokText";
            tokenText.textContent = renderTokenLabel(token);

            const tokenIdText = document.createElement("div");
            tokenIdText.className = "tokID";
            tokenIdText.textContent = tokenId.toString();

            tokenBox.append(tokenText, tokenIdText);
            tokensCell.appendChild(tokenBox);
        }

        row.append(modelCell, tokensCell);
        tbody.appendChild(row);
    }
    tokStatusText.textContent = "Redo.";
}

function renderTokenLabel(token: string): string {
    if (token === "") return "�";
    if (token === "<0x0A>") return "<|newline|>";

    return token
        .split("")
        .map(ch => {
            const code = ch.codePointAt(0)!;
            if (code === 0xFFFD) return "�";
            if (/\s/.test(ch)) return "_";
            return ch;
        })
        .join("");
}



function tokCount() {
    const tokTextCount = document.getElementById("tokTextCount");
    if (!tokTextCount) return;
    tokTextCount.textContent = tokTextBox.value.length.toString();
}

window.addEventListener("load", function () {
    let tokButton = document.getElementById("tokButton");
    if (tokButton) tokButton.addEventListener("click", tokRequest);
    if (tokTextBox) tokTextBox.addEventListener("input", tokCount);
    tokCount();
});

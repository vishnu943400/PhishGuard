// Grab all the elements we need from the page
const emailText = document.getElementById("email-text");
const classifyBtn = document.getElementById("classify-btn");
const clearBtn = document.getElementById("clear-btn");
const compareToggle = document.getElementById("compare-toggle");
const errorMsg = document.getElementById("error-msg");
const loading = document.getElementById("loading");

const resultCard = document.getElementById("result-card");
const verdict = document.getElementById("verdict");
const confidenceValue = document.getElementById("confidence-value");
const confidenceBar = document.getElementById("confidence-bar");

const compareCard = document.getElementById("compare-card");
const compareBody = document.getElementById("compare-body");


// runs when the Classify button is clicked
classifyBtn.addEventListener("click", function () {
    const text = emailText.value.trim();

    // hide old results and errors first
    errorMsg.textContent = "";
    resultCard.hidden = true;
    compareCard.hidden = true;

    if (text === "") {
        errorMsg.textContent = "Please paste some email text first.";
        return;
    }

    loading.hidden = false;
    classifyBtn.disabled = true;

    // ask the Flask backend for the main prediction
    fetch("/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text })
    })
        .then(function (response) {
            return response.json();
        })
        .then(function (data) {
            showResult(data);

            // if the "compare models" checkbox is ticked, also fetch that
            if (compareToggle.checked) {
                return fetch("/classify_compare", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ text: text })
                })
                    .then(function (response) {
                        return response.json();
                    })
                    .then(function (data) {
                        showCompareTable(data.results);
                    });
            }
        })
        .catch(function (error) {
            errorMsg.textContent = "Something went wrong. Is the Flask server running?";
            console.log(error);
        })
        .finally(function () {
            loading.hidden = true;
            classifyBtn.disabled = false;
        });
});


// runs when the Clear button is clicked
clearBtn.addEventListener("click", function () {
    emailText.value = "";
    errorMsg.textContent = "";
    resultCard.hidden = true;
    compareCard.hidden = true;
});


function showResult(data) {
    const isSpam = data.prediction_code === 1;

    verdict.textContent = data.prediction;
    verdict.className = "alert " + (isSpam ? "spam" : "ham");

    confidenceValue.textContent = data.confidence;
    confidenceBar.style.width = data.confidence + "%";
    confidenceBar.className = "progress-bar " + (isSpam ? "spam" : "ham");

    resultCard.hidden = false;
}


function showCompareTable(results) {
    compareBody.innerHTML = "";

    for (const modelName in results) {
        const row = results[modelName];
        const tr = document.createElement("tr");
        tr.innerHTML =
            "<td>" + modelName + "</td>" +
            "<td>" + row.prediction + "</td>" +
            "<td>" + row.confidence + "%</td>";
        compareBody.appendChild(tr);
    }

    compareCard.hidden = false;
}

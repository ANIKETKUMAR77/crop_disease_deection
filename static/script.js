async function predictDisease() {
    let fileInput = document.getElementById("imageUpload");
    let lang = document.getElementById("language").value;

    let preview = document.getElementById("preview");
    let loading = document.getElementById("loading");
    let resultBox = document.getElementById("resultBox");

    let diseaseText = document.getElementById("disease");
    let descText = document.getElementById("description");
    let solText = document.getElementById("solution");

    if (fileInput.files.length === 0) {
        alert("Please select a file.");
        return;
    }

    // Show preview
    let file = fileInput.files[0];
    preview.src = URL.createObjectURL(file);
    preview.style.display = "block";

    let formData = new FormData();
    formData.append("image", file);
    formData.append("lang", lang);

    loading.style.display = "block";
    resultBox.style.display = "none";

    try {
        let response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        let result = await response.json();

        loading.style.display = "none";
        resultBox.style.display = "block";

        diseaseText.innerText = "🌿 " + result.disease;
        descText.innerHTML = "<b>Description:</b> " + result.description;
        solText.innerHTML = "<b>Solution:</b> " + result.solution;

    } catch (error) {
        loading.style.display = "none";
        alert("Error in prediction");
    }
}
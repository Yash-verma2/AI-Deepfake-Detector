const form = document.getElementById("uploadForm");
const loader = document.getElementById("loader");
const resultBox = document.getElementById("resultBox");
const predictionText = document.getElementById("predictionText");
const confidenceText = document.getElementById("confidenceText");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  loader.style.display = "block";
  resultBox.style.display = "none";

  const formData = new FormData(form);
  try {
    const res = await fetch("http://192.168.165.245:7860/predict", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    loader.style.display = "none";
    resultBox.style.display = "block";

    if (data && data.prediction && data.confidence !== undefined) {
      predictionText.textContent = `Prediction: ${data.prediction}`;
      confidenceText.textContent = `Confidence: ${data.confidence}%`;

      const ctx = document.getElementById("confidenceChart").getContext("2d");
      new Chart(ctx, {
        type: "doughnut",
        data: {
          labels: ["Confidence", "Uncertainty"],
          datasets: [{
            data: [data.confidence, 100 - data.confidence],
            backgroundColor: ["#10B981", "#4B5563"],
            borderWidth: 2,
          }]
        },
        options: {
          cutout: "70%",
          plugins: {
            legend: { display: false },
          }
        }
      });
    }
  } catch (err) {
    loader.style.display = "none";
    alert("Error uploading file. Please try again.");
    console.error(err);
  }
});

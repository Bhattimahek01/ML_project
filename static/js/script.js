document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const resultOverlay = document.getElementById('result-overlay');
    const closeBtn = document.getElementById('close-result');

    // UI Elements for result
    const resultIcon = document.getElementById('result-icon');
    const resultTitle = document.getElementById('result-title');
    const resultScore = document.getElementById('result-score');
    const resultBody = document.querySelector('.result-card');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const submitBtn = form.querySelector('.submit-btn');
        const originalBtnText = submitBtn.innerHTML;
        submitBtn.innerHTML = 'Analyzing...';
        submitBtn.disabled = true;

        // Gather Data
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            if (!response.ok) throw new Error('Prediction failed');

            const result = await response.json();
            showResult(result);

        } catch (error) {
            alert('Error: ' + error.message);
        } finally {
            submitBtn.innerHTML = originalBtnText;
            submitBtn.disabled = false;
        }
    });

    closeBtn.addEventListener('click', () => {
        resultOverlay.classList.remove('active');
    });

    // Close on click outside
    resultOverlay.addEventListener('click', (e) => {
        if (e.target === resultOverlay) {
            resultOverlay.classList.remove('active');
        }
    });

    function showResult(data) {
        const percentage = (data.probability * 100).toFixed(1);

        if (data.prediction === 1) {
            // High Risk
            resultIcon.innerHTML = '⚠️';
            resultTitle.textContent = 'High Risk Detected';
            resultTitle.style.color = '#ff4b4b';
            resultScore.innerHTML = `Probability of Cardiovescular Disease: <strong>${percentage}%</strong>`;
        } else {
            // Low Risk
            resultIcon.innerHTML = '✅';
            resultTitle.textContent = 'Low Risk Detected';
            resultTitle.style.color = '#00b09b';
            resultScore.innerHTML = `Probability of Cardiovescular Disease: <strong>${percentage}%</strong>`;
        }

        resultOverlay.classList.add('active');
    }
});

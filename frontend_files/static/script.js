document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('upload-form');
    const submitBtn = document.getElementById('submit-btn');
    const loadingDiv = document.getElementById('loading-div');
    const resultDiv = document.getElementById('result-div');
    const resultText = document.getElementById('result-text');

    form.addEventListener('submit', (event) => {
        event.preventDefault();

        // Disable the submit button and show loading screen
        submitBtn.disabled = true;
        loadingDiv.style.display = 'block';

        // Create a FormData object to send the file via AJAX
        const formData = new FormData(form);
        const xhr = new




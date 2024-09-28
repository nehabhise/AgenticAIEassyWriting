document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('inputForm');
    const submitButton = document.getElementById('submitButton');
    const loaderContainer = document.getElementById('loaderContainer');
    const resultContainer = document.getElementById('resultContainer');
    const resultText = document.getElementById('resultText');

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        
        //const method = document.querySelector('input[name="method"]:checked').value;
        const method = document.getElementById("method").value;
        const userInput = document.getElementById('user_input').value;

    
        // Fetch token from the server-side script
        const response = await fetch('/get-token');
        const data = await response.json();
        const token = data.token;        
        const contentType = 'application/json';
        
        // Show loader and disable the submit button
        loaderContainer.style.display = 'block';
        submitButton.style.display = 'none';
        resultContainer.style.display = 'none';

        const requestOptions = {
            method: 'POST',
            headers: {
                'Content-Type': contentType,
                'Authorization': `Bearer ${token}`, // Include token in Authorization header
            },
            body: JSON.stringify({ method, user_input: userInput }),
        };
        
        try {
            const response = await fetch('/submit', requestOptions);
            const result = await response.json();
            console.log("Result value:",result.result)
            // Update the result container based on the response
            const resultTextContent = result.result ? ` ${result.result}` : `Error: ${result.error}`;

            resultText.innerText = resultTextContent;
            resultContainer.style.display = 'block';
            
            // Scroll to the result container
            resultContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
        } catch (error) {
            console.error('Error:', error);
        } finally {
            // Hide loader and show submit button
            loaderContainer.style.display = 'none';
            submitButton.style.display = 'block';
        }
    });
});

// Long-term forecast data handler

window.longTermDataHandler = {
    fetchLongTermForecastData: function() {
        console.log("Fetching long-term forecast data...");
        
        // Update the URL to match the blueprint route
        fetch('/long-term/api/forecast')
            .then(response => {
                console.log("Response status:", response.status);
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log("Long-term forecast data received:", data);
                // Call the global function to update the UI
                window.longTermForecast.updateLongTermUI(data);
            })
            .catch(error => {
                console.error("Error fetching long-term forecast data:", error);
                // Show error message
                document.querySelector('.loading').style.display = 'none';
                document.getElementById('forecast-error').style.display = 'block';
            });
    }
};
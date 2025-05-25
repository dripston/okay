// Forecast data fetching and error handling

// Function to fetch forecast data with retry
async function fetchForecastDataWithRetry(maxRetries = 5, delay = 3000) {
    let retries = 0;
    
    while (retries < maxRetries) {
        try {
            const response = await fetch('/api/forecast/short-term');
            console.log('Response status:', response.status);
            
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('Received data:', data);
            
            // Check if we have an error message from the API
            if (data.error) {
                throw new Error(data.message || 'Error fetching forecast data');
            }
            
            // Check if we have actual forecast data
            if (!data.forecast || data.forecast.length === 0) {
                throw new Error('No forecast data available');
            }
            
            // Inside the fetchForecastDataWithRetry function, update the data normalization part:
            
            // Normalize data format if needed
            data.forecast = data.forecast.map(day => ({
                date: day.date,
                tavg: day.tavg || day.temp_avg,
                tmin: day.tmin || day.temp_min,
                tmax: day.tmax || day.temp_max,
                prcp: day.prcp || day.precipitation || 0,
                humidity: day.humidity,
                wspd: day.wspd || day.wind_speed,
                pres: day.pres || day.pressure,
                // Ensure we're using the exact condition from CSV
                condition: day.condition || 'Unknown'
            }));
            
            // Store the data
            window.forecastData = data;
            
            // Update the UI using the global function
            if (window.weatherApp && typeof window.weatherApp.updateForecastUI === 'function') {
                window.weatherApp.updateForecastUI(data);
            } else {
                console.error('updateForecastUI function not found in global scope');
                throw new Error('Required function not available');
            }
            
            return true;
        } catch (error) {
            console.error(`Attempt ${retries + 1} failed:`, error);
            retries++;
            
            if (retries >= maxRetries) {
                // Show error after all retries fail
                showError(error);
                return false;
            }
            
            // Wait before retrying
            await new Promise(resolve => setTimeout(resolve, delay));
            console.log(`Retrying... (${retries}/${maxRetries})`);
        }
    }
}

// Function to show error
function showError(error) {
    // Hide loading indicator
    const loadingElement = document.querySelector('.loading');
    if (loadingElement) {
        loadingElement.style.display = 'none';
    }
    
    // Show error message
    const forecastError = document.getElementById('forecast-error');
    if (forecastError) {
        forecastError.style.display = 'block';
    }
    
    // Clear any existing forecast cards
    const forecastCards = document.getElementById('forecast-cards');
    if (forecastCards) {
        forecastCards.innerHTML = `
            <div class="error-card">
                <i class="fas fa-exclamation-circle"></i>
                <p>Unable to retrieve forecast data. Our prediction model is currently unavailable.</p>
                <p class="error-details">${error.message}</p>
                <button id="retry-forecast-btn" class="btn btn-primary mt-3">
                    <i class="fas fa-sync-alt"></i> Retry
                </button>
            </div>
        `;
        
        // Add event listener to retry button
        const retryBtn = document.getElementById('retry-forecast-btn');
        if (retryBtn) {
            retryBtn.addEventListener('click', () => {
                // Show loading again
                forecastCards.innerHTML = `
                    <div class="loading">
                        <div class="loading-spinner"></div>
                        <p>Loading forecast data...</p>
                    </div>
                `;
                
                // Try fetching again
                fetchForecastDataWithRetry();
            });
        }
    }
    
    // Clear charts if the function exists in global scope
    if (window.weatherCharts && typeof window.weatherCharts.clearCharts === 'function') {
        window.weatherCharts.clearCharts();
    }
}

// Export functions to global namespace
window.forecastData = {
    fetchForecastDataWithRetry,
    showError
};
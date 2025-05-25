// Long-term forecast JavaScript - Main file

document.addEventListener('DOMContentLoaded', function() {
    // Initialize variables
    let longTermData = null;
    
    // Create a global namespace for our app
    window.longTermForecast = window.longTermForecast || {};
    
    // Show loading indicator immediately
    const loadingElement = document.querySelector('.loading');
    if (loadingElement) {
        loadingElement.style.display = 'flex';
    }
    
    // Start fetching data
    window.longTermDataHandler.fetchLongTermForecastData();
    
    // Add event listener for window resize to redraw charts
    window.addEventListener('resize', () => {
        if (longTermData) {
            window.longTermCharts.updateLongTermCharts(longTermData.forecast);
        }
    });
    
    // Function to update the long-term forecast UI - expose to global namespace
    window.longTermForecast.updateLongTermUI = function(data) {
        // Store data locally
        longTermData = data;
        
        // Hide loading indicator
        const loadingElement = document.querySelector('.loading');
        if (loadingElement) {
            loadingElement.style.display = 'none';
        }
        
        // Update charts
        if (window.longTermCharts && typeof window.longTermCharts.updateLongTermCharts === 'function') {
            window.longTermCharts.updateLongTermCharts(data.forecast);
        }
        
        // Create monthly forecast cards
        if (window.longTermCharts && typeof window.longTermCharts.createMonthlyForecastCards === 'function') {
            window.longTermCharts.createMonthlyForecastCards(data.forecast);
        }
        
        // Create seasonal highlights
        if (window.longTermCharts && typeof window.longTermCharts.createSeasonalHighlights === 'function') {
            window.longTermCharts.createSeasonalHighlights(data.forecast);
        }
        
        // Check for weather alerts
        if (window.longTermCharts && typeof window.longTermCharts.checkLongTermWeatherAlerts === 'function') {
            window.longTermCharts.checkLongTermWeatherAlerts(data.forecast);
        }
        
        // Update last update time
        const lastUpdateElement = document.getElementById('last-update-time');
        if (lastUpdateElement) {
            const now = new Date();
            lastUpdateElement.textContent = `Last updated: ${now.toLocaleString()}`;
        }
        
        // Add refresh button functionality
        const refreshBtn = document.getElementById('refresh-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                window.longTermDataHandler.fetchLongTermForecastData();
            });
        }
    };
});
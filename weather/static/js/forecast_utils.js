/**
 * Forecast Utilities
 * Helper functions for the weather forecast application
 */

// Global variables
let updateInterval = null;

/**
 * Function to start countdown for next update
 * @param {Date} nextUpdateTime - The time when the next update will occur
 */
function startUpdateCountdown(nextUpdateTime) {
    const countdownElement = document.getElementById('next-update-countdown');
    if (!countdownElement) return;
    
    // Clear any existing interval
    if (updateInterval) {
        clearInterval(updateInterval);
    }
    
    // Update the countdown every second
    updateInterval = setInterval(() => {
        const now = new Date();
        const diff = nextUpdateTime - now;
        
        if (diff <= 0) {
            // Time's up, clear interval and refresh data
            clearInterval(updateInterval);
            countdownElement.textContent = 'Updating...';
            if (typeof fetchForecastDataWithRetry === 'function') {
                fetchForecastDataWithRetry();
            }
            return;
        }
        
        // Calculate minutes and seconds
        const minutes = Math.floor(diff / 60000);
        const seconds = Math.floor((diff % 60000) / 1000);
        
        // Display the countdown
        countdownElement.textContent = `${minutes}m ${seconds}s`;
    }, 1000);
}

/**
 * Function to update last update time
 * @param {string|Date} timestamp - The timestamp of the last update
 */
function updateLastUpdateTime(timestamp) {
    const lastUpdateElement = document.getElementById('last-update-time');
    if (lastUpdateElement) {
        const date = new Date(timestamp);
        lastUpdateElement.textContent = date.toLocaleString('en-US', {
            weekday: 'short',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    }
}

/**
 * Function to get weather recommendation based on conditions
 * @param {Object} day - The forecast data for a day
 * @param {string} condition - The weather condition
 * @returns {string} HTML for the recommendation
 */
function getWeatherRecommendation(day, condition) {
    if (day.prcp > 10) {
        return '<div class="recommendation"><i class="fas fa-umbrella"></i> Carry an umbrella, heavy rain expected</div>';
    } else if (day.prcp > 5) {
        return '<div class="recommendation"><i class="fas fa-cloud-rain"></i> Rain likely, be prepared</div>';
    } else if (day.tavg > 30) {
        return '<div class="recommendation"><i class="fas fa-tint"></i> Stay hydrated, hot weather expected</div>';
    } else if (day.wspd > 15) {
        return '<div class="recommendation"><i class="fas fa-wind"></i> Windy conditions, secure loose items</div>';
    } else if (condition.includes('Sunny') && day.tavg > 25) {
        return '<div class="recommendation"><i class="fas fa-sun-haze"></i> Use sunscreen if outdoors</div>';
    } else if (day.tavg < 15) {
        return '<div class="recommendation"><i class="fas fa-mitten"></i> Dress warmly, chilly weather expected</div>';
    } else if (condition.includes('Partly Cloudy') && day.prcp < 2) {
        return '<div class="recommendation"><i class="fas fa-cloud-sun"></i> Great day for outdoor activities</div>';
    } else {
        return '<div class="recommendation"><i class="fas fa-check-circle"></i> Comfortable weather conditions</div>';
    }
}

// Export functions for use in other files
window.forecastUtils = {
    startUpdateCountdown,
    updateLastUpdateTime,
    getWeatherRecommendation
};
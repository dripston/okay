// Weather alerts handling

// Function to check for weather alerts
function checkWeatherAlerts(forecast) {
    const alertsContainer = document.getElementById('weather-alerts-container');
    const noAlertsMessage = document.getElementById('no-alerts-message');
    
    if (!alertsContainer) return;
    
    // Clear existing alerts
    alertsContainer.innerHTML = '';
    let hasAlerts = false;
    
    // Check for high temperatures
    const highTempDays = forecast.filter(day => day.tmax > 35);
    if (highTempDays.length > 0) {
        hasAlerts = true;
        const alert = document.createElement('div');
        alert.className = 'weather-alert high-temp';
        alert.innerHTML = `
            <div class="alert-icon"><i class="fas fa-temperature-high"></i></div>
            <div class="alert-content">
                <h4>Extreme Heat Warning</h4>
                <p>Temperatures exceeding 35°C expected on ${highTempDays.length} day(s). 
                Stay hydrated and avoid prolonged sun exposure.</p>
                <p class="alert-days">
                    ${highTempDays.map(day => {
                        const date = new Date(day.date);
                        return `<span>${date.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })}: ${day.tmax.toFixed(1)}°C</span>`;
                    }).join(', ')}
                </p>
            </div>
        `;
        alertsContainer.appendChild(alert);
    }
    
    // Check for heavy rain alerts
    const heavyRainDays = forecast.filter(day => day.prcp > 10);
    if (heavyRainDays.length > 0) {
        hasAlerts = true;
        const alert = document.createElement('div');
        alert.className = 'weather-alert heavy-rain';
        alert.innerHTML = `
            <div class="alert-icon"><i class="fas fa-cloud-showers-heavy"></i></div>
            <div class="alert-content">
                <h4>Heavy Rainfall Alert</h4>
                <p>Heavy rainfall (>10mm) expected on ${heavyRainDays.length} day(s). 
                Be prepared for potential flooding and traffic disruptions.</p>
                <p class="alert-days">
                    ${heavyRainDays.map(day => {
                        const date = new Date(day.date);
                        return `<span>${date.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })}: ${day.prcp.toFixed(1)}mm</span>`;
                    }).join(', ')}
                </p>
            </div>
        `;
        alertsContainer.appendChild(alert);
    }
    
    // Check for strong wind alerts
    const strongWindDays = forecast.filter(day => day.wspd > 20);
    if (strongWindDays.length > 0) {
        hasAlerts = true;
        const alert = document.createElement('div');
        alert.className = 'weather-alert strong-wind';
        alert.innerHTML = `
            <div class="alert-icon"><i class="fas fa-wind"></i></div>
            <div class="alert-content">
                <h4>Strong Wind Advisory</h4>
                <p>Strong gusty winds (>20 m/s) expected on ${strongWindDays.length} day(s). 
                Secure loose objects outdoors and be cautious while driving.</p>
                <p class="alert-days">
                    ${strongWindDays.map(day => {
                        const date = new Date(day.date);
                        return `<span>${date.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })}: ${day.wspd.toFixed(1)} m/s</span>`;
                    }).join(', ')}
                </p>
            </div>
        `;
        alertsContainer.appendChild(alert);
    }
    
    // Show or hide the "no alerts" message
    if (noAlertsMessage) {
        noAlertsMessage.style.display = hasAlerts ? 'none' : 'block';
    }
    
    // If no alerts were added and no message exists, add one
    if (!hasAlerts && !noAlertsMessage) {
        const noAlerts = document.createElement('p');
        noAlerts.className = 'no-alerts';
        noAlerts.id = 'no-alerts-message';
        noAlerts.textContent = 'No active weather alerts for this location at this time.';
        alertsContainer.appendChild(noAlerts);
    }
}

// Export functions
window.weatherAlerts = {
    checkWeatherAlerts
};
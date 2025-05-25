// Short-term forecast JavaScript - Main file

document.addEventListener('DOMContentLoaded', function() {
    // Initialize variables
    let forecastData = null;
    let updateInterval = null;
    let statusCheckInterval = null;
    
    // Create a global namespace for our app
    window.weatherApp = window.weatherApp || {};
    
    // Show loading indicator immediately
    const loadingElement = document.querySelector('.loading');
    if (loadingElement) {
        loadingElement.style.display = 'flex';
    }
    
    // Start fetching data with retry
    window.forecastData.fetchForecastDataWithRetry();
    
    // Add event listener for window resize to redraw charts
    window.addEventListener('resize', () => {
        if (forecastData) {
            window.weatherCharts.updateCharts(forecastData.forecast);
        }
    });
    
    // Function to update the forecast UI - expose to global namespace
    window.weatherApp.updateForecastUI = function(data) {
        // Store data locally
        forecastData = data;
        
        // Hide loading indicator and progress bar if they exist
        const loadingElement = document.querySelector('.loading');
        if (loadingElement) {
            loadingElement.style.display = 'none';
        }
        
        if (window.uiHelpers && typeof window.uiHelpers.hideProgressBar === 'function') {
            window.uiHelpers.hideProgressBar();
        }
        
        // Ensure UI elements exist
        if (window.uiHelpers && typeof window.uiHelpers.ensureUIElementsExist === 'function') {
            window.uiHelpers.ensureUIElementsExist();
        }
        
        // Update current weather
        updateCurrentWeather(data.forecast[0]);
        
        // Update forecast cards
        updateForecastCards(data.forecast);
        
        // Update charts
        if (window.weatherCharts && typeof window.weatherCharts.updateCharts === 'function') {
            window.weatherCharts.updateCharts(data.forecast);
        }
        
        // Check for weather alerts
        if (window.weatherAlerts && typeof window.weatherAlerts.checkWeatherAlerts === 'function') {
            window.weatherAlerts.checkWeatherAlerts(data.forecast);
        }
        
        // Add refresh button
        if (window.uiHelpers && typeof window.uiHelpers.addRefreshButton === 'function') {
            window.uiHelpers.addRefreshButton(window.forecastData.fetchForecastDataWithRetry);
        }
        
        // Update last update time
        if (window.uiHelpers && typeof window.uiHelpers.updateLastUpdateTime === 'function') {
            window.uiHelpers.updateLastUpdateTime(new Date().toISOString());
        }
        
        // Start countdown for next update (10 minutes)
        const nextUpdate = new Date();
        nextUpdate.setMinutes(nextUpdate.getMinutes() + 10);
        
        if (window.uiHelpers && typeof window.uiHelpers.startUpdateCountdown === 'function') {
            window.uiHelpers.startUpdateCountdown(nextUpdate, window.forecastData.fetchForecastDataWithRetry);
        }
    };
    
    // Function to update current weather display
    function updateCurrentWeather(todayData) {
        const currentDate = new Date(todayData.date);
        const formattedDate = currentDate.toLocaleDateString('en-US', { 
            weekday: 'long', 
            month: 'long', 
            day: 'numeric' 
        });
        
        // Update date
        const currentDateElement = document.getElementById('current-date');
        if (currentDateElement) {
            currentDateElement.textContent = formattedDate;
        }
        
        // Update temperature
        const tempElement = document.querySelector('.temperature');
        if (tempElement) {
            tempElement.textContent = `${todayData.tavg.toFixed(1)}°C`;
        }
        
        // Update weather icon based on condition
        const weatherIcon = document.querySelector('.weather-icon i');
        if (weatherIcon) {
            const iconClass = weatherConditions.getWeatherIcon(todayData.condition, todayData.prcp);
            weatherIcon.className = `fas ${iconClass}`;
        }
        
        // Update weather details
        const details = document.querySelectorAll('.weather-details .detail span');
        if (details && details.length >= 3) {
            details[0].textContent = `Humidity: ${todayData.humidity.toFixed(0)}%`;
            details[1].textContent = `Wind: ${todayData.wspd.toFixed(1)} m/s`;
            details[2].textContent = `Pressure: ${todayData.pres.toFixed(1)} hPa`;
        }
        
        // Update wind indicator
        updateWindIndicator(todayData.wspd);
        
        // Update "how it feels" text
        updateWeatherFeelText(todayData);
    }
    
    // Function to update wind indicator
    function updateWindIndicator(windSpeed) {
        const windIndicator = document.getElementById('wind-indicator');
        if (!windIndicator) return;
        
        // Update wind arrow rotation based on wind speed
        const windArrow = windIndicator.querySelector('.wind-arrow');
        if (windArrow) {
            // Rotate arrow based on wind speed (just for visual effect)
            const rotation = Math.min(windSpeed * 10, 180);
            windArrow.style.transform = `rotate(${rotation}deg)`;
            
            // Change color based on wind speed
            if (windSpeed > 20) {
                windArrow.style.borderColor = '#ff4d4d'; // Strong wind (red)
            } else if (windSpeed > 10) {
                windArrow.style.borderColor = '#ffa64d'; // Moderate wind (orange)
            } else {
                windArrow.style.borderColor = '#4d94ff'; // Light wind (blue)
            }
        }
    }
    
    // Function to update "how it feels" text
    function updateWeatherFeelText(weatherData) {
        const feelTextElement = document.getElementById('weather-feel-text');
        if (!feelTextElement) return;
        
        let feelText = 'How it feels: ';
        
        // Calculate apparent temperature (simple version)
        const temp = weatherData.tavg;
        const humidity = weatherData.humidity;
        const windSpeed = weatherData.wspd;
        
        // Determine how it feels based on multiple factors
        if (temp > 30 && humidity > 70) {
            feelText += 'Hot and humid, quite uncomfortable';
        } else if (temp > 30) {
            feelText += 'Hot but dry';
        } else if (temp > 25 && humidity > 70) {
            feelText += 'Warm and humid';
        } else if (temp > 25) {
            feelText += 'Pleasantly warm';
        } else if (temp > 20) {
            feelText += 'Comfortable';
        } else if (temp > 15) {
            feelText += 'Slightly cool';
        } else {
            feelText += 'Cool';
        }
        
        // Add wind effect
        if (windSpeed > 20) {
            feelText += ', very windy';
        } else if (windSpeed > 10) {
            feelText += ', breezy';
        }
        
        // Add rain effect
        if (weatherData.prcp > 10) {
            feelText += ', heavy rain expected';
        } else if (weatherData.prcp > 5) {
            feelText += ', rainy';
        } else if (weatherData.prcp > 0) {
            feelText += ', light rain possible';
        }
        
        feelTextElement.textContent = feelText;
    }
    
    // Function to update forecast cards
    function updateForecastCards(forecast) {
        const forecastCards = document.getElementById('forecast-cards');
        forecastCards.innerHTML = '';
        
        // In the updateForecastCards function:
        
        // Create a card for each day in the forecast
        forecast.forEach((day, index) => {
            const date = new Date(day.date);
            const dayOfWeek = date.toLocaleDateString('en-US', { weekday: 'short' });
            const dayOfMonth = date.toLocaleDateString('en-US', { day: 'numeric' });
            const month = date.toLocaleDateString('en-US', { month: 'short' });
            
            // Use the exact condition from CSV data
            const weatherCondition = day.condition || 'Unknown';
            
            // Get the appropriate weather icon using our utility function
            const weatherIcon = window.weatherConditions.getWeatherIcon(weatherCondition, day.prcp);
            
            // Get condition color
            const conditionColor = window.weatherConditions.getWeatherConditionColor(weatherCondition, day.prcp);
            
            // Create the forecast card
            const card = document.createElement('div');
            card.className = 'forecast-card';
            card.innerHTML = `
                <div class="forecast-date">
                    <span class="day">${dayOfWeek}, ${dayOfMonth} ${month}</span>
                </div>
                <div class="forecast-icon">
                    <i class="fas ${weatherIcon}" style="color: ${conditionColor};"></i>
                </div>
                <div class="forecast-condition">
                    ${weatherCondition}
                </div>
                <div class="forecast-temp">
                    ${day.tavg.toFixed(1)}°C
                </div>
                <div class="forecast-minmax">
                    <span class="min"><i class="fas fa-arrow-down"></i> ${day.tmin.toFixed(1)}°</span>
                    <span class="max"><i class="fas fa-arrow-up"></i> ${day.tmax.toFixed(1)}°</span>
                </div>
                <div class="forecast-details">
                    <div class="detail">
                        <i class="fas fa-tint"></i> ${day.prcp.toFixed(1)} mm
                    </div>
                    <div class="detail">
                        <i class="fas fa-wind"></i> ${day.wspd.toFixed(1)} m/s
                    </div>
                    <div class="detail">
                        <i class="fas fa-water"></i> ${day.humidity}%
                    </div>
                </div>
                ${weatherConditions.getWeatherRecommendation(day)}
            `;
            
            // Add the card to the container
            forecastCards.appendChild(card);
        });
    }
});
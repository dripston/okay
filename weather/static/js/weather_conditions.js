// Weather conditions handling

// Function to get the appropriate weather icon based on condition
function getWeatherIcon(condition, precipitation) {
    // Use the exact condition from CSV data
    switch(condition) {
        case 'Sunny':
            return 'fa-sun';
        case 'Partly Cloudy':
            return 'fa-cloud-sun';
        case 'Cloudy':
            return 'fa-cloud';
        case 'Light Rain':
            return 'fa-cloud-sun-rain';
        case 'Rain':
            return 'fa-cloud-rain';
        case 'Heavy Rain':
            return 'fa-cloud-showers-heavy';
        case 'Thunderstorm':
            return 'fa-bolt';
        case 'Snow':
            return 'fa-snowflake';
        case 'Fog':
        case 'Mist':
            return 'fa-smog';
        default:
            // Fallback based on precipitation if condition is missing or unknown
            if (precipitation > 10) {
                return 'fa-cloud-showers-heavy';
            } else if (precipitation > 5) {
                return 'fa-cloud-rain';
            } else if (precipitation > 0) {
                return 'fa-cloud-sun-rain';
            } else {
                return 'fa-cloud';
            }
    }
}

// Function to get weather condition color
function getWeatherConditionColor(condition, precipitation) {
    switch(condition) {
        case 'Sunny':
            return '#FF9500';
        case 'Partly Cloudy':
            return '#34C759';
        case 'Cloudy':
            return '#8E8E93';
        case 'Light Rain':
            return '#30B0C7';
        case 'Rain':
            return '#147EFB';
        case 'Heavy Rain':
            return '#5856D6';
        case 'Thunderstorm':
            return '#AF52DE';
        default:
            // Fallback based on precipitation
            if (precipitation > 10) {
                return '#5856D6'; // Heavy rain color
            } else if (precipitation > 5) {
                return '#147EFB'; // Rain color
            } else if (precipitation > 0) {
                return '#30B0C7'; // Light rain color
            } else {
                return '#8E8E93'; // Default gray for cloudy
            }
    }
}

// Function to get weather recommendation based on condition
function getWeatherRecommendation(day) {
    const condition = day.condition;
    const precipitation = day.prcp || day.precipitation || 0;
    
    // Use exact condition from CSV for recommendations
    switch(condition) {
        case 'Light Rain':
            return '<div class="recommendation"><i class="fas fa-cloud-sun-rain"></i> Light rain expected, consider an umbrella</div>';
        case 'Rain':
            return '<div class="recommendation"><i class="fas fa-cloud-rain"></i> Rain likely, be prepared</div>';
        case 'Heavy Rain':
            return '<div class="recommendation"><i class="fas fa-umbrella"></i> Carry an umbrella, heavy rain expected</div>';
        case 'Cloudy':
            return '<div class="recommendation"><i class="fas fa-cloud"></i> Cloudy conditions expected</div>';
        case 'Partly Cloudy':
            return '<div class="recommendation"><i class="fas fa-cloud-sun"></i> Partly cloudy, good for outdoor activities</div>';
        case 'Sunny':
            if (day.tavg > 30 || day.temp_avg > 30) {
                return '<div class="recommendation"><i class="fas fa-sun"></i> Sunny and hot, stay hydrated</div>';
            } else {
                return '<div class="recommendation"><i class="fas fa-sun"></i> Sunny conditions, great day ahead</div>';
            }
        case 'Thunderstorm':
            return '<div class="recommendation"><i class="fas fa-bolt"></i> Thunderstorms expected, stay indoors</div>';
        case 'Snow':
            return '<div class="recommendation"><i class="fas fa-snowflake"></i> Snowy conditions, dress warmly</div>';
        case 'Fog':
        case 'Mist':
            return '<div class="recommendation"><i class="fas fa-smog"></i> Foggy conditions, drive carefully</div>';
        default:
            // Generic recommendation based on precipitation
            if (precipitation > 0) {
                return '<div class="recommendation"><i class="fas fa-cloud-rain"></i> Precipitation expected</div>';
            } else {
                return '<div class="recommendation"><i class="fas fa-check-circle"></i> Comfortable weather conditions</div>';
            }
    }
}

// Export functions
window.weatherConditions = {
    getWeatherIcon,
    getWeatherConditionColor,
    getWeatherRecommendation
};
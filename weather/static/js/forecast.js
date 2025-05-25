document.addEventListener('DOMContentLoaded', function() {
    // Tab switching functionality
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Remove active class from all buttons and contents
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Add active class to clicked button and corresponding content
            button.classList.add('active');
            const tabId = button.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
        });
    });
    
    // For now, we'll use placeholder data since the API isn't implemented yet
    displayPlaceholderData();
});

// Display placeholder data until API is implemented
function displayPlaceholderData() {
    // Display placeholder current weather
    const weatherCard = document.getElementById('current-weather-card');
    weatherCard.innerHTML = `
        <div class="weather-icon">
            <i class="fas fa-sun"></i>
        </div>
        <div class="weather-info">
            <h3>Today, ${new Date().toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' })}</h3>
            <p class="temperature">28.5 °C</p>
            <div class="weather-details">
                <div class="detail">
                    <i class="fas fa-tint"></i>
                    <span>Humidity: 65%</span>
                </div>
                <div class="detail">
                    <i class="fas fa-wind"></i>
                    <span>Wind: 3.2 m/s</span>
                </div>
                <div class="detail">
                    <i class="fas fa-compress-alt"></i>
                    <span>Pressure: 1012 hPa</span>
                </div>
                <div class="detail">
                    <i class="fas fa-cloud-rain"></i>
                    <span>Precipitation: 0.0 mm</span>
                </div>
            </div>
        </div>
    `;

    // Display placeholder short-term forecast
    const shortTermContainer = document.getElementById('short-term-cards');
    shortTermContainer.innerHTML = '';
    
    // Generate 7 days of placeholder forecast
    const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
    const today = new Date();
    
    for (let i = 1; i <= 7; i++) {
        const forecastDate = new Date();
        forecastDate.setDate(today.getDate() + i);
        const dayName = days[forecastDate.getDay()];
        const dateStr = forecastDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        
        // Randomize some weather data for demonstration
        const temp = (25 + Math.random() * 8).toFixed(1);
        const precip = (Math.random() * 5).toFixed(1);
        const humidity = (60 + Math.random() * 20).toFixed(0);
        const wind = (2 + Math.random() * 4).toFixed(1);
        
        // Choose icon based on precipitation
        let weatherIcon = 'fa-sun';
        if (precip > 3) {
            weatherIcon = 'fa-cloud-showers-heavy';
        } else if (precip > 0.5) {
            weatherIcon = 'fa-cloud-rain';
        } else if (humidity > 75) {
            weatherIcon = 'fa-cloud';
        }
        
        const card = document.createElement('div');
        card.className = 'forecast-card';
        card.innerHTML = `
            <div class="forecast-date">${dayName}, ${dateStr}</div>
            <div class="forecast-icon">
                <i class="fas ${weatherIcon}"></i>
            </div>
            <div class="forecast-temp">${temp} °C</div>
            <div class="forecast-details">
                <div>Precipitation: ${precip} mm</div>
                <div>Humidity: ${humidity}%</div>
                <div>Wind: ${wind} m/s</div>
            </div>
        `;
        
        shortTermContainer.appendChild(card);
    }
    
    // Display placeholder long-term forecast
    const longTermContainer = document.getElementById('long-term-forecast');
    longTermContainer.innerHTML = '';
    
    const months = ['January', 'February', 'March', 'April', 'May', 'June'];
    const currentMonth = today.getMonth();
    
    for (let i = 0; i < 6; i++) {
        const monthIndex = (currentMonth + i) % 12;
        const monthName = months[i]; // Using predefined array for demo
        
        // Randomize monthly data
        const avgTemp = (24 + Math.random() * 6).toFixed(1);
        const totalPrecip = (50 + Math.random() * 150).toFixed(0);
        const avgHumidity = (65 + Math.random() * 15).toFixed(0);
        
        const card = document.createElement('div');
        card.className = 'monthly-card';
        card.innerHTML = `
            <div class="monthly-header">
                <span class="monthly-name">${monthName}</span>
                <span class="monthly-avg">${avgTemp} °C</span>
            </div>
            <div class="monthly-details">
                <div class="monthly-detail">
                    <i class="fas fa-cloud-rain"></i>
                    <span>Total Precipitation: ${totalPrecip} mm</span>
                </div>
                <div class="monthly-detail">
                    <i class="fas fa-tint"></i>
                    <span>Avg. Humidity: ${avgHumidity}%</span>
                </div>
                <div class="monthly-detail">
                    <i class="fas fa-temperature-high"></i>
                    <span>Avg. High: ${(parseFloat(avgTemp) + 5).toFixed(1)} °C</span>
                </div>
                <div class="monthly-detail">
                    <i class="fas fa-temperature-low"></i>
                    <span>Avg. Low: ${(parseFloat(avgTemp) - 5).toFixed(1)} °C</span>
                </div>
            </div>
        `;
        
        longTermContainer.appendChild(card);
    }
    
    // Create placeholder charts
    createPlaceholderCharts();
}

// Create placeholder charts until API is implemented
function createPlaceholderCharts() {
    // Temperature chart for short-term forecast
    const tempCtx = document.getElementById('temperature-chart').getContext('2d');
    const days = [];
    const temps = [];
    const tempMin = [];
    const tempMax = [];
    
    // Generate 7 days of data
    const today = new Date();
    for (let i = 1; i <= 7; i++) {
        const date = new Date();
        date.setDate(today.getDate() + i);
        days.push(date.toLocaleDateString('en-US', { weekday: 'short' }));
        
        const baseTemp = 25 + Math.random() * 5;
        temps.push(baseTemp.toFixed(1));
        tempMin.push((baseTemp - 3 - Math.random() * 2).toFixed(1));
        tempMax.push((baseTemp + 3 + Math.random() * 2).toFixed(1));
    }
    
    new Chart(tempCtx, {
        type: 'line',
        data: {
            labels: days,
            datasets: [
                {
                    label: 'Average Temperature (°C)',
                    data: temps,
                    borderColor: '#00B5D8',
                    backgroundColor: 'rgba(0, 181, 216, 0.1)',
                    fill: true,
                    tension: 0.3
                },
                {
                    label: 'Min Temperature (°C)',
                    data: tempMin,
                    borderColor: '#2563EB',
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.3
                },
                {
                    label: 'Max Temperature (°C)',
                    data: tempMax,
                    borderColor: '#DC2626',
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.3
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Temperature Forecast (°C)'
                }
            }
        }
    });
    
    // Precipitation chart for short-term forecast
    const precipCtx = document.getElementById('precipitation-chart').getContext('2d');
    const precip = [];
    
    for (let i = 0; i < 7; i++) {
        precip.push((Math.random() * 10).toFixed(1));
    }
    
    new Chart(precipCtx, {
        type: 'bar',
        data: {
            labels: days,
            datasets: [{
                label: 'Precipitation (mm)',
                data: precip,
                backgroundColor: 'rgba(37, 99, 235, 0.6)',
                borderColor: 'rgba(37, 99, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Precipitation Forecast (mm)'
                }
            }
        }
    });
    
    // Monthly temperature chart for long-term forecast
    const monthlyTempCtx = document.getElementById('monthly-temperature-chart').getContext('2d');
    const months = ['January', 'February', 'March', 'April', 'May', 'June'];
    const monthlyTemps = [];
    const monthlyTempMin = [];
    const monthlyTempMax = [];
    
    for (let i = 0; i < 6; i++) {
        const baseTemp = 22 + i + Math.random() * 3;
        monthlyTemps.push(baseTemp.toFixed(1));
        monthlyTempMin.push((baseTemp - 5 - Math.random() * 2).toFixed(1));
        monthlyTempMax.push((baseTemp + 5 + Math.random() * 2).toFixed(1));
    }
    
    new Chart(monthlyTempCtx, {
        type: 'line',
        data: {
            labels: months,
            datasets: [
                {
                    label: 'Average Temperature (°C)',
                    data: monthlyTemps,
                    borderColor: '#00B5D8',
                    backgroundColor: 'rgba(0, 181, 216, 0.1)',
                    fill: true,
                    tension: 0.3
                },
                {
                    label: 'Min Temperature (°C)',
                    data: monthlyTempMin,
                    borderColor: '#2563EB',
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.3
                },
                {
                    label: 'Max Temperature (°C)',
                    data: monthlyTempMax,
                    borderColor: '#DC2626',
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.3
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Monthly Temperature Forecast (°C)'
                }
            }
        }
    });
    
    // Monthly precipitation chart for long-term forecast
    const monthlyPrecipCtx = document.getElementById('monthly-precipitation-chart').getContext('2d');
    const monthlyPrecip = [];
    
    for (let i = 0; i < 6; i++) {
        monthlyPrecip.push((50 + Math.random() * 150).toFixed(0));
    }
    
    new Chart(monthlyPrecipCtx, {
        type: 'bar',
        data: {
            labels: months,
            datasets: [{
                label: 'Total Precipitation (mm)',
                data: monthlyPrecip,
                backgroundColor: 'rgba(37, 99, 235, 0.6)',
                borderColor: 'rgba(37, 99, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Monthly Precipitation Forecast (mm)'
                }
            }
        }
    });
}
// Long-term forecast charts

window.longTermCharts = {
    // Function to update all charts
    updateLongTermCharts: function(forecastData) {
        console.log("Updating long-term charts with data:", forecastData);
        
        // Extract data for charts
        const dates = forecastData.map(item => item.date);
        const temps = forecastData.map(item => item.tavg);
        const minTemps = forecastData.map(item => item.tmin);
        const maxTemps = forecastData.map(item => item.tmax);
        const precipitation = forecastData.map(item => item.prcp);
        
        // Update temperature chart
        this.createTemperatureChart(dates, minTemps, temps, maxTemps);
        
        // Update precipitation chart
        this.createPrecipitationChart(dates, precipitation);
        
        // Update monthly averages chart
        this.createMonthlyAveragesChart(forecastData);
    },
    
    // Create temperature chart
    createTemperatureChart: function(dates, minTemps, avgTemps, maxTemps) {
        const ctx = document.getElementById('temperature-chart').getContext('2d');
        
        // Create or update chart
        if (window.temperatureChart) {
            window.temperatureChart.destroy();
        }
        
        window.temperatureChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: this.formatDatesForChart(dates),
                datasets: [
                    {
                        label: 'Min Temp (°C)',
                        data: minTemps,
                        borderColor: '#30B0C7',
                        backgroundColor: 'rgba(48, 176, 199, 0.1)',
                        tension: 0.4,
                        fill: false
                    },
                    {
                        label: 'Avg Temp (°C)',
                        data: avgTemps,
                        borderColor: '#FF9500',
                        backgroundColor: 'rgba(255, 149, 0, 0.1)',
                        tension: 0.4,
                        fill: false
                    },
                    {
                        label: 'Max Temp (°C)',
                        data: maxTemps,
                        borderColor: '#FF3B30',
                        backgroundColor: 'rgba(255, 59, 48, 0.1)',
                        tension: 0.4,
                        fill: false
                    }
                ]
            },
            // Update the options in createTemperatureChart function
            options: {
                responsive: true,
                maintainAspectRatio: false, // Change to false
                plugins: {
                    title: {
                        display: true,
                        text: 'Temperature Forecast (6 Months)'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            maxRotation: 45,
                            minRotation: 45
                        }
                    }
                }
            }
        });
    },
    
    // Create precipitation chart
    createPrecipitationChart: function(dates, precipitation) {
        const ctx = document.getElementById('precipitation-chart').getContext('2d');
        
        // Create or update chart
        if (window.precipitationChart) {
            window.precipitationChart.destroy();
        }
        
        window.precipitationChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: this.formatDatesForChart(dates),
                datasets: [{
                    label: 'Precipitation (mm)',
                    data: precipitation,
                    backgroundColor: 'rgba(20, 126, 251, 0.5)',
                    borderColor: '#147EFB',
                    borderWidth: 1
                }]
            },
            // Similarly update the options in createPrecipitationChart function
            options: {
                responsive: true,
                maintainAspectRatio: false, // Change to false
                plugins: {
                    title: {
                        display: true,
                        text: 'Precipitation Forecast (6 Months)'
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            maxRotation: 45,
                            minRotation: 45
                        }
                    }
                }
            }
        });
    },
    
    // Helper function to format dates
    formatDatesForChart: function(dates) {
        return dates.filter((date, index) => index % 15 === 0).map(date => {
            const d = new Date(date);
            return `${d.getDate()}/${d.getMonth()+1}`;
        });
    },
    
    // Create monthly averages chart
    createMonthlyAveragesChart: function(forecastData) {
        // Implementation for monthly averages chart
    },
    
    // Create monthly forecast cards
    createMonthlyForecastCards: function(forecastData) {
        const container = document.getElementById('monthly-forecast-cards');
        container.innerHTML = '';
        
        // Group data by month
        const monthlyData = this.groupByMonth(forecastData);
        
        // Create a card for each month
        for (const [month, data] of Object.entries(monthlyData)) {
            const card = document.createElement('div');
            card.className = 'monthly-card';
            
            const avgTemp = this.calculateAverage(data.map(item => item.tavg));
            const totalPrecip = this.calculateSum(data.map(item => item.prcp));
            
            card.innerHTML = `
                <h4>${month}</h4>
                <div class="monthly-card-content">
                    <div class="monthly-temp">
                        <i class="fas fa-temperature-high"></i>
                        <span>${avgTemp.toFixed(1)}°C</span>
                    </div>
                    <div class="monthly-precip">
                        <i class="fas fa-cloud-rain"></i>
                        <span>${totalPrecip.toFixed(1)} mm</span>
                    </div>
                </div>
            `;
            
            container.appendChild(card);
        }
    },
    
    // Helper function to group data by month
    groupByMonth: function(forecastData) {
        const months = {};
        const monthNames = ['January', 'February', 'March', 'April', 'May', 'June', 
                           'July', 'August', 'September', 'October', 'November', 'December'];
        
        forecastData.forEach(item => {
            const date = new Date(item.date);
            const monthName = monthNames[date.getMonth()];
            
            if (!months[monthName]) {
                months[monthName] = [];
            }
            
            months[monthName].push(item);
        });
        
        return months;
    },
    
    // Helper function to calculate average
    calculateAverage: function(values) {
        return values.reduce((sum, value) => sum + value, 0) / values.length;
    },
    
    // Helper function to calculate sum
    calculateSum: function(values) {
        return values.reduce((sum, value) => sum + value, 0);
    },
    
    // Create seasonal highlights
    createSeasonalHighlights: function(forecastData) {
        // Implementation for seasonal highlights
    },
    
    // Check for weather alerts
    checkLongTermWeatherAlerts: function(forecastData) {
        // Implementation for weather alerts
    }
};
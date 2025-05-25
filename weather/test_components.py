import asyncio
from iot_stations import IoTStationManager
from satellite_feed import INSAT3DFeed
from crowdsource_data import CrowdsourceManager
from data_normalizer import WeatherDataNormalizer
import json

async def test_all_components():
    # Initialize all components
    iot_manager = IoTStationManager()
    satellite_feed = INSAT3DFeed()
    crowdsource_manager = CrowdsourceManager()
    normalizer = WeatherDataNormalizer()
    
    # Fetch data from all sources
    print("\n=== Fetching Data from All Sources ===")
    iot_data = await iot_manager.fetch_all_stations()
    satellite_data = await satellite_feed.get_current_data()
    mobile_reports = await crowdsource_manager.simulate_mobile_reports(num_reports=50)
    crowdsource_data = await crowdsource_manager.process_reports(mobile_reports)
    
    # Normalize data from each source
    print("\n=== Normalizing Data ===")
    normalized_iot = normalizer.normalize_iot_data(iot_data)
    normalized_satellite = normalizer.normalize_satellite_data(satellite_data)
    normalized_crowdsource = normalizer.normalize_crowdsource_data(crowdsource_data)
    
    # Combine all normalized data
    all_data = normalized_iot + [normalized_satellite, normalized_crowdsource]
    
    # Resolve conflicts and get final data
    print("\n=== Final Aggregated Weather Data ===")
    final_data = normalizer.resolve_conflicts(all_data)
    print(json.dumps(final_data, indent=2))

if __name__ == "__main__":
    asyncio.run(test_all_components())
from smolagents import tool
import requests, os, dotenv

@tool
def get_weather(city: str) -> str:
    """This tool returns the current weather and temperature
    in Celsius degrees in a given city. If the 'city' arguments
    is an empty string, then it returns the current weather and
    temperature in the user's current location.
    Args:
        city: A string containing the name of the city."""
    #API key should be placed in a .env file in the root
    #directory of the project.
    if not isinstance(city, str):
        return "Error: argument must be a string."
    dotenv.load_dotenv()
    #Currently using API from https://www.weatherapi.com/
    #which provides a free plan.
    api_key = os.getenv("WEATHER_API_KEY")
    #If no city is given, then get weather in current location
    #based on IP address
    location = city if city else "auto:ip"
    url = f"https://api.weatherapi.com/v1/current.json?key={api_key}&q={location}"
    res = requests.get(url)
    if res.status_code == 200:
        res = res.json()
        location = res['location']['name']
        region = res['location']['region']
        country = res['location']['country']
        temperature = res['current']['temp_c']
        description = res['current']['condition']['text']
        return f"The weather conditions in {location}, {region}/{country} is: {description}. And the current temperature is {temperature}°C."
    else:
        return "Error: Could not fetch the results." + res.text


if __name__ == "__main__":
    print(get_weather(""))

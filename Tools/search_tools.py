# Import the json module for JSON data handling
import json
# Import the os module to access environment variables
import os
# Import the requests module to make HTTP requests
import requests
# Import the tool decorator from the langchain library to create tools
from langchain.tools import tool

# Define a class called SearchTools


class SearchTools():

    # Define a class method with the @tool decorator, named "Search the internet"
    @tool("Search the internet")
    def search_internet(query):
        """Useful to search the internet about a given topic and return relevant results"""
        # Print a message indicating that the search is being performed
        print("Searching the internet...")
        # Define the maximum number of results to be returned
        top_result_to_return = 5
        # URL of the Serper search service
        url = "https://google.serper.dev/search"
        # Create a JSON payload with the query and the number of results
        payload = json.dumps(
            {"q": query, "num": top_result_to_return, "tbm": "nws"})
        # Define the request headers, including the Serper API key
        headers = {
            'X-API-KEY': os.environ['SERPER_API_KEY'],
            'content-type': 'application/json'
        }
        # Make a POST request to the URL with the headers and payload
        response = requests.request("POST", url, headers=headers, data=payload)
        # Check if the 'organic' key is present in the JSON response
        if 'organic' not in response.json():
            # Return an error message if the 'organic' key is not present
            return "Sorry, I couldn't find anything about that; there could be an error with your Serper API key."
        else:
            # Extract the results from the 'organic' key in the JSON response
            results = response.json()['organic']
            # Initialize a list to store the result strings
            string = []
            # Print the results (limited by top_result_to_return)
            print("Results:", results[:top_result_to_return])
            # Iterate over the results, up to the defined limit
            for result in results[:top_result_to_return]:
                try:
                    # Try to extract the date of the result
                    date = result.get('date', 'Date not available')
                    # Add the result details to the list as a formatted string
                    string.append('\n'.join([
                        f"Title: {result['title']}",
                        f"Link: {result['link']}",
                        f"Date: {date}",  # Include the date in the result
                        f"Snippet: {result['snippet']}",
                        "\n-----------------"
                    ]))
                except KeyError:
                    # Continue to the next result if there is a KeyError
                    continue

            # Return the list of result strings as a single string
            return '\n'.join(string)

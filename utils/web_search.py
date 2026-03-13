from langchain_community.tools import DuckDuckGoSearchRun

def search_web(query):

    search = DuckDuckGoSearchRun()

    results = search.run(query)

    return results
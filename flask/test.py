from google import genai
client = genai.Client(api_key="AIzaSyCbZPjW9FHt9_mJ6I3K_iRlhAQnH6opaNo")
resp = client.models.generate_content(model="gemini-2.5-flash", contents="Hello")
print(resp.text)

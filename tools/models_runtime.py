import time
from concurrent.futures import ThreadPoolExecutor
from google import genai
from google.genai import types
import rag

c = genai.Client(api_key=rag.GEMINI_API_KEY,
                 http_options=types.HttpOptions(timeout=30_000))

names = [m.name.replace("models/", "") for m in c.models.list()
         if "generateContent" in (m.supported_actions or [])]
print(len(names), "models found")

def test(name):
    t = time.perf_counter()
    try:
        r = c.models.generate_content(model=name, contents="Reply with the word OK.")
        return f"{time.perf_counter()-t:6.2f}s  OK      {name}"
    except Exception as e:
        return f"{time.perf_counter()-t:6.2f}s  FAIL    {name}  {getattr(e,'code','')}"

with ThreadPoolExecutor(8) as ex:
    for line in sorted(ex.map(test, names)):
        print(line)
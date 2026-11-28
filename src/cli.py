"""
Terminal tester. Run from the project root:

    uv run python src/cli.py            # full chat
    uv run python src/cli.py --debug    # also show rewritten query, chunks and timings
"""
import sys
import time

from rag import prepare_query, retrieve, stream_answer

DEBUG = "--debug" in sys.argv


def main():
    history: list[dict] = []
    print("LegalRAG CLI. Ask in English or Urdu. Type 'quit' to exit, 'reset' to clear history.\n")

    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not message:
            continue
        if message.lower() in {"quit", "exit"}:
            break
        if message.lower() == "reset":
            history.clear()
            print("(history cleared)\n")
            continue

        t0 = time.perf_counter()
        prep = prepare_query(message, history)
        t1 = time.perf_counter()
        hits = retrieve(prep["search_query"], message, prep["doc_type"])
        t2 = time.perf_counter()

        if DEBUG:
            print(f"\n[language={prep['language']}  doc_type={prep['doc_type']}]")
            print(f"[search_query] {prep['search_query']}")
            for i, h in enumerate(hits, 1):
                print(f"  [{i}] {h['score']:.3f}  {h['doc_label']}  "
                      f"sec={h['section_number'] or '-'}  ({h['content_type']})  "
                      f"{h['text'][:90].replace(chr(10), ' ')}...")
            print(f"[timing] prepare_query={t1 - t0:.2f}s  retrieve={t2 - t1:.2f}s")
            print("\n\n")
            

        print("Bot: ", end="", flush=True)
        answer = ""
        t3 = time.perf_counter()
        first = True
        try:
            for token in stream_answer(message, hits, prep["language"], history):
                if first:
                    if DEBUG:
                        print(f"[first token after {time.perf_counter() - t3:.2f}s] ",
                              end="", flush=True)
                    first = False
                print(token, end="", flush=True)
                answer += token
        except Exception as e:
            print(f"\n[error] {type(e).__name__}: {str(e)[:300]}\n")
            continue

        if DEBUG:
            print(f"\n[timing] total generation={time.perf_counter() - t3:.2f}s", end="")
        print("\n")

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
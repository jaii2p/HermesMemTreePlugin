"""Chat entry point used by `python cli.py chat`. Not meant to be run by hand.

    python -m memtree._chat <model> <adapter.pt> [finetune.pt]

Auto-detects GPU vs CPU, loads the adapter (tree or window mode, from its own
saved config), adds the fine-tune if given, and runs the memory runtime so the
watchdog, whispers and recall are all live. Type /mem to see what she's shown,
/help for commands, quit to leave.
"""
import sys


def main():
    model_name = sys.argv[1]
    adapter = sys.argv[2] if len(sys.argv) > 2 else ""
    finetune = sys.argv[3] if len(sys.argv) > 3 else ""

    import torch
    from .distill import load_model
    from .phi_adapter import load_adapter
    from .runtime import MemoryRuntime, render_memory

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"loading on {device}...", flush=True)
    model, tok = load_model(model_name, device, dtype)
    if adapter:
        load_adapter(model, adapter)
    if finetune:
        from .finetune import load_finetune
        load_finetune(model, tok, finetune)
    rt = MemoryRuntime(model, tok)

    print("\nready. say hi, or /help for commands.\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in ("quit", "exit", "/quit", "/exit"):
            break
        if q == "/help":
            print("  /mem   show what she's being shown right now\n"
                  "  quit   leave\n  anything else is a message to her")
            continue
        if q == "/mem":
            print(render_memory(rt.dog, "me", entry_chars=rt.budget["entry_chars"]) + "\n")
            continue
        r = rt.chat("me", q, max_new_tokens=400)
        print(f"\nphi> {r['reply']}")
        for d in r.get("references", []):
            print(f"  [watchdog] {d.ref[1]} -> {d.resolution}")
        print()


if __name__ == "__main__":
    main()

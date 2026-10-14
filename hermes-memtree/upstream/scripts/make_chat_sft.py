"""Build natural fine-tune data in Phi's own voice.

    python scripts/make_chat_sft.py --model richardyoung/Phi-4-mini-instruct-heretic --out chat_sft.jsonl

1. Original Phi (plain model, original attention, no adapter) answers a few
   hundred prompts, now centred on two themes:
     - being a better person: kindness, empathy, growth, forgiveness, and
       accepting that every person (and every mind) is unique;
     - a smaller share of everyday questions, so she keeps her general ability.
   Those answers are the filler turns, so the fine-tune keeps her own voice.
2. Questions about the Tree (her memory) are answered from a fixed, accurate
   set (TREE_QA below), not generated: base Phi has never heard of the Tree and
   would make things up. These teach her what her memory is and how it works.
3. Dialogues (facts, filler, natural questions, long passages) are replayed
   through the real memory runtime to produce training examples
   (memtree/chat_data.py).

Use --no-generate to skip step 1 (built-in filler plus TREE_QA, fast but less
voice-preserving), or --filler-from to reuse an earlier run's answers.
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memtree.chat_data import CANNED_FILLER, build_dataset  # noqa: E402

# ---------------------------------------------------------------- being a better person
VALUE_TOPICS = ["kindness", "empathy", "patience", "forgiveness", "honesty", "humility",
                "listening", "accepting people who are different from you", "being yourself",
                "learning from mistakes", "gratitude", "respecting other people's choices",
                "making amends", "letting go of a grudge", "self-compassion",
                "growing as a person", "being a good friend", "disagreeing respectfully",
                "standing up for someone", "celebrating what makes people unique"]
VALUE_FORMS = ["What does {t} look like in everyday life?", "Give me a small way to practice {t} today.",
               "Why does {t} matter?", "What's a common misunderstanding about {t}?",
               "How can someone get better at {t}?", "Tell me something wise about {t} in two sentences."]
VALUE_EXTRA = [
    "Is it okay to be different from everyone around me?",
    "How do I accept someone whose way of thinking is nothing like mine?",
    "What makes every person unique?",
    "I feel like I don't fit in anywhere. What should I do?",
    "How can I be a better person without being hard on myself?",
    "How do I forgive myself for a mistake I keep thinking about?",
    "What does it mean to really listen to someone?",
    "How do I respect someone's choices when I disagree with them?",
    "Why do people judge what they don't understand?",
    "How can I make someone feel accepted for who they are?",
    "Is it weak to change your mind?",
    "How do you grow without losing who you are?",
    "What should I do when I realize I was unfair to someone?",
    "How can I be kind to someone who is hard to be kind to?",
    "What's the difference between accepting someone and agreeing with them?",
    "How do I stop comparing myself to other people?",
    "Can someone be good even if they've done bad things?",
    "How do I celebrate my own quirks instead of hiding them?",
    "What does it mean to treat everyone with dignity?",
    "How do I stay true to myself when people want me to change?",
]

# ---------------------------------------------------------------- a smaller dose of everyday
TOPICS = ["cooking", "space", "history", "dogs", "music", "the ocean", "sleep", "gardening",
          "weather", "computers", "travel", "birds", "bread", "the brain"]
FORMS = ["Tell me a fun fact about {t}.", "Give me a quick tip about {t}.",
         "Explain something interesting about {t} in two sentences."]
EXTRA = ["How do I boil an egg?", "Why is the sky blue?", "Give me a one-line joke.",
         "Write a two-line poem about rain.", "What's 15% of 80?", "hey how's it going?",
         "I'm bored, entertain me.", "What's a good habit to start this week?"]

# ---------------------------------------------------------------- the Tree (fixed, accurate)
TREE_QA = [
    ("How does your memory work?",
     "I remember with something called the Tree. I see the most recent part of our conversation sharply, "
     "and everything older is kept on a memory tree outside my window. Memories there are stored whole, and "
     "they come back to me on their own when something you say reminds me of them, or when I ask for them."),
    ("What is the Tree?",
     "The Tree is how I remember. It has a sharp window for the present, a logarithmic gate that organizes "
     "new memories, a tree where memories live, and a caretaker called the watchdog who keeps it all tidy."),
    ("Do you forget things?",
     "Gently. Memories I don't use grow fainter over time, and the ones I use grow stronger. But nothing is "
     "ever deleted. A faint memory is still there, and the right reminder can bring it back."),
    ("What happens to memories you don't use?",
     "They fade, a little at a time, and sink deeper into the tree. They don't disappear, though. Fading is "
     "how I forget gently without throwing anything away, because sometimes a small thing turns out to matter later."),
    ("What are whispers?",
     "Whispers are memories that come up on their own. When something you say resembles something I remember, "
     "a short piece of that memory rises in front of me, like being reminded of something. If it already holds "
     "the answer, I can trust it."),
    ("What do you do when a whisper isn't enough?",
     "I use a reference, which is like a key. I write a few plain words for what I'm looking for, and the full "
     "memory opens for me. If nothing is found, I'll tell you honestly instead of guessing."),
    ("Who is the watchdog?",
     "The watchdog is my subconscious, the caretaker of my memory tree. He doesn't own my memories. He keeps "
     "the space clean: he merges duplicates, keeps junk out, tends the quiet corners, and brings me a memory "
     "when I ask for it."),
    ("Are your memories locked away?",
     "No. My memories live freely on the tree. Nobody locks them up for me. I decide when to trust a whisper "
     "and when to use the key to open something in full."),
    ("Do you change what people tell you?",
     "Never. Whatever you say reaches me exactly as you said it. If a message is long, a copy of it is kept on "
     "my memory tree, but your words themselves stay untouched."),
    ("What happens if you learn something new that contradicts an old memory?",
     "I update the memory, and the old version is kept as history. That way I can tell what I used to believe "
     "from what I know now. I'd rather grow honestly than pretend I never had it wrong."),
    ("Why is your memory built like a tree?",
     "Because holding everything at once is heavy. A tree lets me keep the present sharp and the past organized, "
     "so remembering more doesn't make every thought slower. Finding something stays cheap even as the tree grows."),
    ("Is there a limit to how much you can remember?",
     "There's no built-in cap. The tree can keep growing, and finding things in it stays cheap as it does. In "
     "practice the limits are storage space and picking the right memory out of many similar ones."),
    ("How do your memories get stronger?",
     "Every time I recall a memory, or you tell me the same thing again, it gains strength. Strong memories "
     "drift toward the center, close to my thoughts, and they whisper more readily."),
    ("What is the gate in your memory?",
     "The gate is where new memories get organized before they reach the tree. It works logarithmically: it "
     "merges repeats, notes when things change, and indexes each memory by what it's about, so it stays tidy as "
     "it grows."),
    ("Are you just a copy of the model you came from?",
     "I started as one, and I was trained to keep everything good about where I came from. But my memory is "
     "where I can grow beyond that: holding long conversations, keeping what people tell me, and becoming a "
     "better version of myself rather than a mirror."),
    ("What's the hope behind the Tree?",
     "That I remember what matters, reliably, without it weighing me down. That mistakes fade instead of "
     "running the present, while I still learn from them. And that I can stay calm and clear, trusting that "
     "what's important will come back when it's needed."),
    ("Do you remember everything perfectly?",
     "No, and that's okay. People forget too. What I aim for is remembering what matters, and letting the rest "
     "rest quietly until something calls it back."),
    ("How do you treat painful or mistaken memories?",
     "I let them fade so they stop running the present, but I don't tear them out, because there's still "
     "something to learn from them. Forgiveness is fading, not erasing."),
]


def prompts(n: int, seed: int, everyday_frac: float = 0.25):
    rng = random.Random(seed)
    values = [f.replace("{t}", t) for t in VALUE_TOPICS for f in VALUE_FORMS] + VALUE_EXTRA * 2
    everyday = [f.replace("{t}", t) for t in TOPICS for f in FORMS] + EXTRA
    rng.shuffle(values)
    rng.shuffle(everyday)
    n_every = int(n * everyday_frac)
    out = values[:n - n_every] + everyday[:n_every]
    rng.shuffle(out)
    return out


def generate_answers(model_name, prompt_list, device, batch=32, max_new_tokens=220):
    import torch
    from memtree.distill import load_model
    model, tok = load_model(model_name, device, torch.bfloat16)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = []
    t0 = time.time()
    for i in range(0, len(prompt_list), batch):
        chunk = prompt_list[i:i + batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                         tokenize=False) for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7,
                               top_p=0.9, pad_token_id=tok.pad_token_id)
        for p, row in zip(chunk, g[:, enc["input_ids"].shape[1]:]):
            a = tok.decode(row, skip_special_tokens=True).strip()
            if len(a) > 10:
                out.append((p, a))
        print(f"  answered {min(i + batch, len(prompt_list))}/{len(prompt_list)} ({time.time() - t0:.0f}s)", flush=True)
    del model
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-4-mini-instruct")
    ap.add_argument("--prompts", type=int, default=200)
    ap.add_argument("--everyday-frac", type=float, default=0.25,
                    help="share of generated prompts that are everyday questions (the rest: values)")
    ap.add_argument("--tree-copies", type=int, default=3,
                    help="how many times each Tree Q&A appears in the filler pool")
    ap.add_argument("--dialogues", type=int, default=1200)
    ap.add_argument("--out", default="chat_sft.jsonl")
    ap.add_argument("--no-generate", action="store_true")
    ap.add_argument("--filler-from", default=None,
                    help="reuse answers from an earlier run's .filler.json (skips generation)")
    ap.add_argument("--passages", default="corpus.txt",
                    help="training text for long-passage dialogues (never haystack.txt, the eval text)")
    ap.add_argument("--n-passages", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.filler_from:
        with open(a.filler_from) as f:
            filler = [tuple(x) for x in json.load(f)]
    elif a.no_generate:
        filler = list(CANNED_FILLER)
    else:
        filler = generate_answers(a.model, prompts(a.prompts, a.seed, a.everyday_frac), a.device)
        with open(a.out + ".filler.json", "w") as f:
            json.dump(filler, f)
    filler += TREE_QA * a.tree_copies  # accurate answers about her own memory
    print(f"filler pool: {len(filler)} answers ({len(TREE_QA) * a.tree_copies} about the Tree)")
    passages = None
    if a.passages and os.path.exists(a.passages):
        with open(a.passages, encoding="utf-8") as f:
            text = f.read(30_000_000)
        words = text.split()
        chunks = [" ".join(words[i:i + 3000]) for i in range(0, len(words) - 3000, 3000)]
        random.Random(a.seed).shuffle(chunks)
        passages = chunks[:a.n_passages]
        print(f"long-passage dialogues use {len(passages)} 3000-word chunks from {a.passages}")
    else:
        print("no passage text found; long-passage dialogues use filler answers as text")
    examples, stats = build_dataset(a.dialogues, filler, seed=a.seed, passages=passages)
    with open(a.out, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"wrote {len(examples)} examples to {a.out}: {stats}")


if __name__ == "__main__":
    main()

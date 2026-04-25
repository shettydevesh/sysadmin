"""Debug script for GRPO training - run in Colab to test before HF Spaces."""

import torch
import re

# Check GPU
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"  # Start small

print(f"\nLoading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Fix tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

print(f"Vocab size: {len(tokenizer)}")
print(f"Pad token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
print(f"EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
print(f"Model loaded on {model.device}")

# Test simple generation first
print("\n--- Test 1: Simple generation ---")
test_input = "Hello, how are you?"
inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
print(f"Input shape: {inputs.input_ids.shape}")
print(f"Max token ID in input: {inputs.input_ids.max().item()}")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
print(f"Output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
print("✅ Simple generation works!")

# Test chat template
print("\n--- Test 2: Chat template ---")
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
]

try:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"Chat template prompt (first 200 chars): {prompt[:200]}...")

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    print(f"Input shape: {inputs.input_ids.shape}")
    print(f"Max token ID: {inputs.input_ids.max().item()}")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Response: {response}")
    print("✅ Chat template works!")
except Exception as e:
    print(f"❌ Chat template failed: {e}")

# Test GRPO-style generation
print("\n--- Test 3: GRPO episode simulation ---")

SYSTEM_PROMPT = """You are an SRE agent. Diagnose Linux issues.
Use <think> for reasoning, <bash> for commands.
Example: <think>Check status</think><bash>systemctl status nginx</bash>"""

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "nginx won't start"},
]

try:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Response:\n{response}")

    # Parse
    bash_match = re.search(r"<bash>(.*?)</bash>", response, re.DOTALL)
    if bash_match:
        print(f"\n✅ Extracted command: {bash_match.group(1).strip()}")
    else:
        print("\n⚠️ No <bash> tag found in response")

except Exception as e:
    print(f"❌ GRPO generation failed: {e}")
    import traceback
    traceback.print_exc()

# Test training step
print("\n--- Test 4: Training backward pass ---")
try:
    model.train()

    inputs = tokenizer("Test input for training", return_tensors="pt", truncation=True).to(model.device)
    outputs = model(**inputs, labels=inputs.input_ids)

    loss = outputs.loss
    print(f"Loss: {loss.item()}")

    loss.backward()
    print("✅ Backward pass works!")

    # Clear gradients
    model.zero_grad()

except Exception as e:
    print(f"❌ Training failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("All tests complete! If all passed, GRPO should work.")

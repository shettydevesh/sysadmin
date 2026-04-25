"""Agent wrapper for Qwen model inference."""

import re
from typing import Optional
from dataclasses import dataclass


@dataclass
class AgentConfig:
    """Configuration for the agent."""
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    max_new_tokens: int = 512
    temperature: float = 0.7
    device: str = "auto"
    load_in_4bit: bool = True


SYSTEM_PROMPT = """You are an expert SRE agent diagnosing and fixing Linux server issues.

When given a problem:
1. First, use diagnostic commands to understand the issue
2. Identify the root cause before attempting fixes
3. Apply targeted fixes, not blind restarts
4. Verify the fix worked

Output format:
<think>
Your reasoning about what to check or do next
</think>
<bash>single command here</bash>

Only output ONE command per response. Wait for output before the next command."""


class SysadminAgent:
    """Agent that uses Qwen to diagnose and fix Linux issues."""

    def __init__(self, config: Optional[AgentConfig] = None, model=None, tokenizer=None):
        """Initialize the agent.

        Args:
            config: Agent configuration
            model: Pre-loaded model (optional, for trained models)
            tokenizer: Pre-loaded tokenizer (optional)
        """
        self.config = config or AgentConfig()
        self.model = model
        self.tokenizer = tokenizer
        self.conversation = []

    def load_model(self):
        """Load the model and tokenizer."""
        if self.model is not None:
            return  # Already loaded

        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        print(f"Loading model: {self.config.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )

        if self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=quantization_config,
                device_map=self.config.device,
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map=self.config.device,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )

        print("Model loaded successfully")

    def reset(self):
        """Reset the conversation history."""
        self.conversation = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def get_action(self, observation: str) -> tuple[str, str]:
        """Get the next action given an observation.

        Args:
            observation: The output from the environment

        Returns:
            Tuple of (command, thinking)
        """
        if self.model is None:
            self.load_model()

        # Add observation to conversation
        self.conversation.append({"role": "user", "content": observation})

        # Generate response
        text = self.tokenizer.apply_chat_template(
            self.conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        # Add response to conversation
        self.conversation.append({"role": "assistant", "content": response})

        # Parse thinking and command
        thinking = self._extract_thinking(response)
        command = self._extract_command(response)

        return command, thinking

    def _extract_thinking(self, response: str) -> str:
        """Extract thinking from response."""
        match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _extract_command(self, response: str) -> str:
        """Extract command from response."""
        match = re.search(r"<bash>(.*?)</bash>", response, re.DOTALL)
        return match.group(1).strip() if match else "echo 'No command found'"


class RandomAgent:
    """Baseline agent that issues random diagnostic commands."""

    COMMANDS = [
        "systemctl status nginx",
        "journalctl -xe --no-pager | tail -50",
        "df -h",
        "ls -la /etc/nginx/",
        "cat /etc/nginx/nginx.conf",
        "ps aux | head -20",
        "ss -tlnp",
        "free -m",
        "top -bn1 | head -15",
        "systemctl restart nginx",
    ]

    def __init__(self):
        self.command_index = 0

    def reset(self):
        """Reset the agent."""
        self.command_index = 0

    def get_action(self, observation: str) -> tuple[str, str]:
        """Get the next action (cycles through commands)."""
        import random
        command = random.choice(self.COMMANDS)
        return command, "Random baseline"


def load_trained_agent(checkpoint_path: str, config: Optional[AgentConfig] = None) -> SysadminAgent:
    """Load a trained agent from a checkpoint.

    Args:
        checkpoint_path: Path to the LoRA checkpoint or merged model
        config: Optional agent configuration

    Returns:
        SysadminAgent with the trained model
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch

    config = config or AgentConfig()

    print(f"Loading trained model from: {checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )

    # Load base model
    if config.load_in_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=quantization_config,
            device_map=config.device,
            trust_remote_code=True,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            device_map=config.device,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, checkpoint_path)

    return SysadminAgent(config=config, model=model, tokenizer=tokenizer)

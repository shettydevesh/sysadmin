---
title: Sysadmin Game - GRPO Training
emoji: 🛠️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
hardware: t4-small
---

# Sysadmin Game: Train LLMs on Real Linux Troubleshooting

Train Qwen2.5-Coder to diagnose and fix broken Linux systems using GRPO reinforcement learning.

## What This Does

1. Breaks a Linux container (disk full, nginx config error, port conflict, etc.)
2. Model attempts to diagnose and fix using shell commands
3. Gets reward based on actual fix success
4. Learns from trial and error

## Usage

Click the "Training" tab to start GRPO training with live environment feedback.

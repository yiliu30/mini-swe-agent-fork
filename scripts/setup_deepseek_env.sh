#!/usr/bin/env bash
# Source before running mini-extra swebench with DeepSeek
# NOTE: Uses standard DeepSeek API (OpenAI-compatible), NOT the
# Anthropic-compatible endpoint which doesn't support custom tool definitions.
# Copy your key from ~/.claude/settings.json (ANTHROPIC_AUTH_TOKEN) or
# set DEEPSEEK_API_KEY=sk-... before sourcing this script.
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-your-deepseek-api-key}"
echo "DeepSeek environment configured for mini-swe-agent"

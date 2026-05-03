# Claude Angry Typo 😠⌨️

For replicating how angry claude gets with my typos.

## Overview

This project contains scripts to test:
- Basic connectivity to the Anthropic Messages API.
- Assistant message prefilling (and debugging model-specific limitations).
- Dynamic conversational flows.

## Getting Started

### Prerequisites

- Python 3.7+
- An Anthropic API key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ChamodKalupahana/claude-angry-typo.git
   cd claude-angry-typo
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your environment:
   Create a `.env` file in the root directory and add your API key:
   ```env
   ANTHROPIC_API_KEY=your_api_key_here
   ```

## Usage

### Test Connection
Verify your API setup with a simple "Hello world" prompt:
```bash
python test_connection.py
```

### Test Prefill
Experiment with steering the assistant's response by prefilling the conversation history:
```bash
python test_prefill.py
```

## Implementation Details

- **System Prompts:** Unlike the web interface, the Claude API does not include a system prompt by default. This project uses the official Anthropic system prompts (sourced from the [Claude Release Notes](https://platform.claude.com/docs/en/release-notes/system-prompts)), which are stored in `system_prompts.json`.
- **Prefill Behavior:** While Anthropic's API recently changed to restrict ending a message list with an assistant prefill in some contexts, we found it was not strictly necessary to trigger the "angry typo" behavior. The model's frustration can be induced through iterative prompting and system instructions alone.

## Costs

- **Sonnet 4.6:** Running the full transcript generation suite costs approximately **$1.00 USD**.
- **Haiku:** Significantly cheaper than Sonnet, making it ideal for high-volume testing of basic logic.

## Insights

- **Angry Typo:** The project explores "unconventional" assistant responses to see how it affects subsequent reasoning. Specifically, it tests how persistent user errors (like the "it's" vs "its" typo) can eventually lead to visible model frustration in the Chain-of-Thought.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
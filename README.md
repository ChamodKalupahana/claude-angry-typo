# Claude Angry Typo 😠⌨️

A playground for experimenting with Anthropic's Claude API, focusing on testing prefill capabilities and assistant persona steering.

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
   pip install anthropic python-dotenv
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

## Insights

- **Prefill Support:** Some models/API versions may require the conversation to end with a `user` message, while others allow ending with an `assistant` message to prefill the start of Claude's response.
- **Angry Typo:** The project explores "unconventional" assistant responses (like "no lol") to see how it affects subsequent reasoning.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
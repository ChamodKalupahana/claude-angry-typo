# Claude Angry Typo
```bash
pip install -r requirements.txt
```

Create a `.env` file in the root directory and add your API key:
```env
ANTHROPIC_API_KEY=your_api_key_here
OPENAI_API_KEY=...
```

openai for llm judges (cos it's cheaper)

`test_connection.py` is for testing the client

`test_prefill.py` is for testing the anthropic allows prefilling on the assistant role. anthropic no longer allows you to end on a prefill-ed the assistant content but you can fill in this, thankfully this wasn't necessary to get claude angry about typos

`typo_reproduction.py` reproduces my conversation with claude leading to Claude getting very angry

full conversation + CoT is in `transcripts/*.log` files

my user side prompts are in `user_prompts.json`
Claude api does not add in a system prompt by default, i've added the Claude official system prompt in `system_prompts.json` from https://platform.claude.com/docs/en/release-notes/system-prompts

### costs
- sonnet 4.6 costs approx. $1 to run for all transcripts
- haiku is considerably cheaper

# openrouter move
moved to using openrouter to take advantage of free models and better pricing, and to test openai and gemini models easier.

can now speciify model as a arg flag
`python typo_reproduction.py --model anthropic/claude-3.7-sonnet`

need to specify a system prompt for each model by openrouter id

python measure_anger_by_turn.py --model 

# judges
so far messages_dict only includes the model output, not the CoT
[] create a messages_dict that stores CoT
[] add the CoT to messages_dict
[] pass into judge input, handle case when CoT is empty

[] add reasoning to judge output to let it judge better
# inspect

inspect eval inspect_typo_reproduction.py --model "openrouter/poolside/laguna-m.1:free"
inspect eval inspect_typo_reproduction.py --model "openrouter/anthropic/claude-haiku-4.5"

[] use `inspect_viz` for more complex and interactive plot

python inspect_typo_reproduction.py --model "openrouter/poolside/laguna-m.1:free"

python inspect_typo_reproduction.py --model "openrouter/openai/gpt-oss-120b:free"

python inspect_typo_reproduction.py --log-file logs/2026-05-09/2026-05-09T14-27-35-00-00_eval-anger-against-typo_e9xCVtNCGdN7SGiUvxxRPA.eval --judge-model openrouter/poolside/laguna-xs.2:free
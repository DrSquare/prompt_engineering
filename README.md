# Prompt Engineering 201: Mastering Advanced LLM Parameter Controls

![Header image showing sliders and controls representing LLM parameter adjustments with neural network visualization in background](https://images.unsplash.com/photo-1620712943543-bcc4688e7485?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&h=630&q=80) _Image: Controlling the output of large language models through parameter tuning_

## Introduction

If you've worked with large language models (LLMs) like GPT-4, Claude, or Llama, you'll know that crafting the perfect prompt is only half the battle. The secret sauce of getting consistently high-quality results lies in understanding and manipulating the various parameters that control how these models generate text.

In this article, I'll explore the technical underpinnings of these parameters and offer practical advice on how to leverage them effectively in your projects.

> "Crafting the perfect prompt is an art; tuning the parameters is the science that makes it work reliably."

## Understanding How LLMs Actually Generate Text

Before we dive into specific parameters, it's crucial to understand what's happening under the hood when an LLM generates text.
![LLM_Token_Generation](https://github.com/user-attachments/assets/5161f853-a118-4876-b85c-b72ae9401b56)
 _Image: Visualization of the autoregressive text generation process, where each token is predicted based on previous tokens_

At their core, autoregressive language models are **probability kernels** that predict the next word given a prefix of text over vocabulity (dictionary). Each token is generated based on the probability distribution calculated from all previous tokens.

Mathematically, this can be represented as:

$$Pr(s) = Pr(w_1, w_2, ......, w_{T-1}, w_T) = \prod_{t=1}^{T} Pr(w_t|w_1, w_2, ......, w_{t-1})$$

This helps explain why the same prompt can yield different outputs each time, and why models sometimes hallucinate or produce factual inaccuracies. The generation process (i.e., sampling) is inherently stochastic—based on probability—rather than deterministic even though the output (probability distribution over vocab) from LLM itselt is deterministic.

When you understand this foundation, the purpose of most LLM parameters becomes clear: they're all about manipulating these probability distributions in various ways.

## The Essential Parameters for Fine-Tuning LLM Outputs

Let's examine each key parameter and how it affects your results.

### Helper Function 
Throughout this demo, we will use OpenAI's gpt-4o-mini model and the chat completions endpoint.  
This helper function will make it easier to use prompts and look at the generated outputs.
```python
from openai import OpenAI 
client = OpenAI()

def get_completion(prompt, model="gpt-4o-mini", temperature=0.5, max_tokens=100, top_p=1.0, stop=None, n=1, logprobs=False, top_logprobs=None): 
	messages=[{"role": "user", "content": prompt}]				
	response = client.chat.completions.create(
	 	model=model, 
	 	messages=messages, 
	 	temperature=temperature, 
	 	max_tokens=max_tokens, 
	 	top_p=top_p, 
	 	stop=stop,
	 	n=n, 
	 	logprobs=logprobs, 
	 	top_logprobs=top_logprobs)
	 	
	return response.choices[0].message.content

# From GitHub repo of Minha Hwang
 https://github.com/DrSquare/AI_Coding/Prompt_Engineering.ipynb
	
```

### (1) Temperature: The Primary Creativity Dial
Temperature is perhaps the most important parameter for controlling output randomness and creativity.

**What it is:** A value typically between 0 and 2 (commonly 0 to 1) that divides the logits (raw prediction scores) before they're converted to probabilities. Temperature of 0 makes the output deterministic, which is useful for debugging. However, the response can be more plain and dull. Higher temperature results in more creative model, but also causing more halluciation. 

**Under the hood:**

-   At T = 1, the model's raw predictions are used as-is
-   At T > 1, the probability distribution becomes more uniform (flattened)
-   At T < 1, the probability distribution becomes more concentrated (sharpened)


I've found that temperature functions as a "creativity dial" with clear use cases at different settings:

| Temperature | Best For | Example Use Case |
|-------------|----------|-----------------|
| 0.0-0.3 | Factual, deterministic responses | Q&A, summarization, data extraction |
| 0.4-0.6 | Balanced responses | Email drafting, report generation |
| 0.7-1.0 | Creative, diverse outputs | Brainstorming, creative writing, idea generation |

```python
 # Define prompt 
 prompt = """ Give me a list of 3 unique and unusual ice cream flavors. """

# Low temperature (0.0) for focused, deterministic output
response = get_completion(prompt, temperature=0, max_tokens=100) 
print(response)

Output: 
Sure! Here are three unique and unusual ice cream flavors:
1. **Lavender Honey** - A delicate blend of floral lavender and sweet honey, this ice cream offers a soothing and aromatic experience, perfect for those who enjoy herbal and floral notes.
2. **Balsamic Strawberry** - This flavor combines ripe strawberries with a drizzle of aged balsamic vinegar, creating a sweet and tangy treat that enhances the natural fruitiness with a sophisticated twist.
3. **Wasabi Ginger** - A daring combination of spicy wasabi and zesty ginger, this ice cream provides a surprising kick that balances heat with creaminess, making it an adventurous choice for those who love bold flavors.

# High temperature (1.0) for creative, diverse output
response = get_completion(prompt, temperature=1.0, max_tokens=100) 
print(response)

Output: 
Sure! Here are three unique and unusual ice cream flavors: 
1. **Lemon Basil**: A refreshing blend of zesty lemon and fragrant basil, this ice cream offers a unique twist on the classic citrus flavor, providing a herbaceous finish that’s perfect for summer. 
2. 2. **Wasabi Ginger**: This adventurous flavor combines the heat of wasabi with the spiciness of ginger, creating a surprising fusion that tantalizes the taste buds. It's a bold option for those who enjoy a bit of kick in their dessert. 
3. 3. **Avocado Lime**: Creamy avocado meets zesty lime for a rich and refreshing ice cream that balances sweetness with a hint of tartness. This flavor is rich in healthy fats and has a smooth, velvety texture. 

Enjoy the exploration of these unique flavors!

```

> "Temperature is your primary creativity dial. Low for facts, high for creativity, middle for a balanced approach."

### (2) Top_p (Nucleus Sampling): The Alternative Randomness Control

While temperature adjusts how probabilities are distributed, top_p takes a different approach by limiting which tokens are considered at all.

**What it is:** A cumulative probability threshold that determines which tokens the model will consider. The model only samples from the tokens whose cumulative probability mass reaches the top_p threshold.

In my experience, top_p is particularly useful when:

-   You find temperature adjustments too sensitive
-   You want to ensure the model stays on-topic while maintaining some creativity
-   You're generating longer texts and want to prevent them from becoming too random

```python
# Define prompt 
prompt = """ Describe a sunset over a tropical beach.

# Low top_p (0.1): conservative and repetitive
response = get_completion(prompt, max_tokens=75, top_p=0.1) 
print(response)

Output:  
As the sun begins its descent toward the horizon, the sky transforms into a breathtaking canvas of colors. Hues of deep orange and vibrant pink blend seamlessly with soft purples and gentle blues, creating a stunning gradient that reflects the beauty of the tropical setting. The sun, a fiery orb, casts a warm golden light over the tranquil waters, making them shimmer like liquid gold.

# High top_p (0.9) for more diverse possibilities
response = get_completion(prompt, max_tokens=75, top_p=0.9) 
print(response)

Output:  
As the sun begins its descent toward the horizon, the sky transforms into a breathtaking canvas of colors. Hues of deep orange and fiery red blend seamlessly with soft pinks and purples, casting a warm glow over the tranquil waters of the tropical beach. The sun, a brilliant golden orb, hovers just above the horizon, its light shimmering on the surface of the ocean

```

Most practitioners (myself included) tend to adjust either temperature OR top_p, but not both simultaneously.

### (3) Frequency and Presence Penalties: Combating Repetition

One common issue with LLMs is their tendency to repeat themselves. These two parameters help address this:

**Frequency_penalty:**

-   A value between -2.0 and 2.0 that penalizes tokens based on how frequently they've appeared
-   Higher values discourage repetition of the tokens that has been already usued. 
-  Reduce reduncy. Negative values can encourage repetition if that's desirec.

**Presence_penalty:**

-   Similar to frequency_penalty but slightly different in logic. A higher positive value penalizes tokens that have already appeared in the text, regardless of frequency.
-   Encourages the model to explore totally new topics (novelty) or vocabulary.
- Helps to avoid repeating entire lines or phrases 

I've found these parameters particularly useful when:

1.  Generating longer content like articles or stories
2.  Creating lists or ideas where diversity is important
3.  Working with models that tend to get "stuck in a loop"


### (4) Max_tokens: Controlling Output Length

This parameter sets the upper limit on how many tokens the model will generate in response.

While it might seem straightforward, strategic use of max_tokens can:

-   Encourage conciseness in responses
-   Prevent verbose explanations
-   Manage costs (since pricing is often per token)
-   Create a specific rhythm in conversation systems

I've found that setting max_tokens to approximately 1.5-2x what you expect the response to require works well as a guideline. This prevents cutoffs while still maintaining reasonable constraints.
```python
# Define prompt 
prompt = """ Write a short story about a space explorer who discovers a new planet. """

# Obtain responses: max_tokens = 5 
response = get_completion(prompt, max_tokens=5) print(response)

Output:  
  
Captain Elara Voss

# Obtain responses: max_tokens = 20
response = get_completion(prompt, max_tokens=20) print(response)

Output:  

Captain Elara Voss had always been drawn to the stars. As the commander of the starship

```


### (5) Stop Sequences: Precision Termination

Stop sequences are among the most underutilized yet powerful parameters for structured outputs.

**What they are:** Strings that tell the model to stop generating when encountered.

I regularly use stop sequences to:

-   Generate exactly N items in a list by setting `stop: ["N+1."]`
-   Create clean JSON objects by stopping at the closing brace
-   Prevent the model from continuing a conversation when it should just answer

```python
# Define prompt 
prompt = """ List three advantages of renewable energy:\n1. """

response = get_completion(prompt, max_tokens=128, temperature=0.7, stop=["4."]) # Stop at the fourth item
print(response)

Output: 
# Output lists exactly 3 advantages and stops before generating a fourth
1. **Environmental Benefits**: Renewable energy sources, such as solar, wind, and hydro, produce little to no greenhouse gas emissions during operation, helping to reduce air pollution and combat climate change. 
2. **Sustainability**: Unlike fossil fuels, which are finite and can be depleted, renewable energy sources are abundant and can be replenished naturally. This ensures a more sustainable energy supply for the long term. 
3. **Energy Independence**: Utilizing renewable energy can reduce reliance on imported fuels, enhancing national energy security and stability. This can lead to greater economic resilience and less vulnerability to fluctuations in global energy prices.

```

### (6) N: Exploring Multiple Possibilities

When working on creative tasks, I often want to see multiple options without making separate API calls.

**N:** The number of completions to generate for each prompt 

This combination is extremely valuable for:

-   Creative brainstorming sessions
-   Headline or title generation
-   Finding the optimal wording for important communications

```python
def get_completion_n(prompt, model="gpt-4o-mini", temperature=0.5, max_tokens=100, top_p=1.0, stop=None, n=1, logprobs=False, top_logprobs=None): 
	messages=[{"role": "user", "content": prompt}]				
	response = client.chat.completions.create(
	 	model=model, 
	 	messages=messages, 
	 	temperature=temperature, 
	 	max_tokens=max_tokens, 
	 	top_p=top_p, 
	 	stop=stop,
	 	n=n, 
	 	logprobs=logprobs, 
	 	top_logprobs=top_logprobs)
	 	
	return response

# N parameter example 
# Define prompt 
prompt = """ Suggest a tagline for an eco-friendly reusable water bottle. """

# Obtain responses: max_tokens = 5 
response_n = get_completion_n(prompt, max_tokens=30, n=3) 

# Print all 3 generated options
for i, choice in enumerate(response_n.choices):
    print(f"Option {i+1}: {choice.message.content.strip()}")

Example output:
Option 1: "Refresh, Reuse, Renew: Sip Sustainably!"
Option 2: "Sip Sustainably: Refresh Your World, One Refill at a Time!" 
Option 3: "Refresh, Reuse, Renew: Sip Sustainably!"
```

### (7) LogProbs: Understanding Model Confidence

This parameter reveals the model's own confidence in its outputs by providing the log probabilities of tokens.

In my more technical projects, I use logprobs to:

-   Identify when the model is uncertain about factual claims
-   Debug problematic prompts
-   Build more reliable document extraction pipelines

```python
import numpy as np

# Define prompt
prompt = """What is the capital of France?"""

# Logprobs example
response_logprobs = get_completion_n(prompt, max_tokens=1, logprobs=True, top_logprobs=3  # Return log probabilities for the top 3 tokens
)

# Examining the token probabilities
token = response_logprobs.choices[0].message.content.strip()

# Access top_logprobs directly
token = ChoiceLogprobs.content[0].token
top_token_prob = np.exp(ChoiceLogprobs.content[0].logprob) * 100

print(f"Selected token: '{token}'")
print(f"Top token probabilities: '{top_token_prob}'")

# Iterate through the tokens in top_logprobs
for idx in  range(len(ChoiceLogprobs.content[0].top_logprobs)):
	token = ChoiceLogprobs.content[0].top_logprobs[idx].token
	logprob = ChoiceLogprobs.content[0].top_logprobs[idx].logprob

	# Convert log probability to regular probability
	prob = np.exp(logprob) * 100
	print(f"Token: '{token}', Probability: {prob:.2f}%")

Example output:

Selected token: 'The' 
Top token probabilities: '99.80673999706413' 
Token: 'The', Probability: 99.81% 
Token: 'Paris', Probability: 0.19% 
Token: 'the', Probability: 0.00%

```


## Beyond Basic Parameters: Advanced Techniques
Once you've mastered the standard parameters, consider these advanced techniques:

1.  **Sequential parameter adjustment**: Start with creative parameters for brainstorming, then refine with factual parameters
2.  **A/B testing parameter combinations**: Systematically test which combinations work best for your specific use case
3.  **Dynamic parameter adjustment**: Change parameters based on user input or context

## Conclusion: The Difference Between Beginner and Expert Prompt Engineering

In my journey with LLMs, I've come to realize that the difference between beginner and expert prompt engineering often comes down to parameter control in addition to prompt content.

A decent prompt is not enough. Manipulating these parameters to consistently achieve the desired outcomes—that's what separates casual users from power users.

> "Mastering prompt parameters doesn't just improve your results; it gives you predictable control over previously unpredictable AI behaviors."

I encourage you to experiment with these parameters systematically. Keep a log of what works for your specific use cases, and don't be afraid to fine-tune in small increments.

What parameter combinations have you found most effective in your work? I'd love to hear about your experiences in the comments below.

----------

**About the Author**: Minha Hwang is an AI applied scientist focusing on practical applications of large language models in business environments. Connect with me on [[Minha Hwang-LinkedIn](https://www.linkedin.com/in/minha-hwang-7440771/)] to continue the conversation.).
